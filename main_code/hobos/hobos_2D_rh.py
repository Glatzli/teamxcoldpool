"""Script to plot HOBOS Relative Humidity data, and differences between AROME/UKMO/WRF/ICON model and HOBOS

Note: it's not possible to read in METEOGRAM data for all Models, because the stations locations of the HOBOS are not
available, so we need the 3D data first level fixed time """
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from main_code.AROME.read_in_arome import read_3D_variables_AROME
from main_code.ICON_model.read_icon_model_3D import read_icon_fixed_point_and_time
from main_code.UKMO_model.read_ukmo import get_ukmo_fixed_point_lowest_level
from main_code.WRF_Helen.read_wrf_helen import read_wrf_fixed_point_and_time
from main_code.confg import dem_file_hobos_extent, dir_PLOTS, hobos_file
from main_code.hobos.hobos_2D_temp import plot_wind_observation, plot_wind_arome, plot_wind_ukmo, plot_wind_icon, \
    plot_wind_wrf_acinn


def calc_rh_arome(df_model):
    """function to calculate the relative humidity for AROME Model"""
    df_model["T_K"] = mpcalc.temperature_from_potential_temperature(pressure=df_model["p"],
                                                                    potential_temperature=df_model["th"])
    df_model["T"] = df_model["T_K"].metpy.convert_units("degC")

    df_model["Td"] = mpcalc.dewpoint_from_specific_humidity(pressure=df_model["p"],
                                                            temperature=df_model["T"],
                                                            specific_humidity=df_model["q"])

    rh = mpcalc.relative_humidity_from_dewpoint(df_model["T"], df_model["Td"])

    return rh.metpy.convert_units("percent")


def extract_rh_for_slice(model_name, extent, my_time):
    """extract the rh for a slice of latitudes and longitudes"""
    # TODO only implemented for AROME until now, would need some more time to do it also for other models
    min_lon, max_lon, min_lat, max_lat = extent

    if model_name == "AROME":
        df_arome = read_3D_variables_AROME(variables=["p", "th", "q"], method="sel",
                                           lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat), slice_lat_lon=True,
                                           level=90.0, time=pd.to_datetime(my_time))
        return calc_rh_arome(df_arome)


def extract_rh_points(model_name, lon, lat, my_time):
    """extract the rh for a single point"""
    time = pd.to_datetime(my_time)
    if model_name == "AROME":
        df_arome = read_3D_variables_AROME(variables=["th", "p", "q"], method="sel",
                                           lon=lon, lat=lat, slice_lat_lon=False,
                                           level=90.0, time=time)

        df_arome["relative_humidity"] = calc_rh_arome(df_arome)
        return df_arome["relative_humidity"].metpy.unit_array.magnitude

    elif model_name == "UKMO":

        df_ukmo = get_ukmo_fixed_point_lowest_level(lat=lat, lon=lon)
        return df_ukmo.loc[str(my_time)]["relative_humidity"]

    elif model_name == "WRF_ACINN":
        ds_wrf_3d_lowest_level = read_wrf_fixed_point_and_time(longitude=lon,
                                                               latitude=lat, day=time.day, hour=time.hour,
                                                               minute=time.minute)

        ds_wrf_3d_lowest_level = ds_wrf_3d_lowest_level.isel(bottom_top=0)

        return ds_wrf_3d_lowest_level["relative_humidity"].values

    elif model_name == "ICON":  # needs brutal long
        if time.day == 16:
            df_icon_nearest = read_icon_fixed_point_and_time(day=time.day, hour=time.hour, my_lon=lon,
                                                             my_lat=lat)

            return df_icon_nearest.isel(height=-1)["relative_humidity"].values
        else:
            return 0
    else:
        raise AssertionError(f"No model named like {model_name}")


def calc_filtered_dict(model_name, my_time):
    """
    Calculate the station data dictionary with rh differences between HOBOS and a specified model.
    Exclude NaN values in HOBOS rh data.

    Args:
    model_name (str): Name of the model used to compare with HOBOS.
    my_time (datetime): The specific time for which the data is calculated.

    Returns:
    dict: A dictionary containing rh data and differences for stations.
    """
    station_data_dict = {}

    # Get unique station keys (assuming 'STATION_KEY' is a dimension in your dataset)
    station_keys = ds['STATION_KEY'].values

    for station_key in station_keys:
        # Select data for the current station key
        station_data = ds.sel(STATION_KEY=station_key)

        # Extract latitude, longitude, and Hobos rh data
        lat = station_data['lat'].values.item()
        lon = station_data['lon'].values.item()
        hobos_rh = station_data['rh'].sel(TIME=my_time, method="nearest").values.item()

        # Read model data and calculate rh
        model_rh_values = extract_rh_points(model_name, lon, lat, my_time)

        # Check if Hobos rh is not NaN to include in the result
        if not np.isnan(hobos_rh):
            station_data_dict[station_key] = {
                'latitude': lat,
                'longitude': lon,
                'hobos_rh': hobos_rh,
                'model_rh_values': model_rh_values,
                'rh_difference': hobos_rh - model_rh_values
            }

    return station_data_dict


def find_absolute_min_max_rh_diff(initial_time, model_name, variable="rh_difference", intervals=12,
                                  interval_duration=2):
    """
    Find the absolute minimum and maximum of rh difference over specified time intervals.

    Args:
    initial_time (datetime): The start time for calculating rh differences.
    variable (str): The key in the dictionary to calculate min and max for (e.g., 'rh_difference').
    model_name (str): Name of the model to use.
    intervals (int, optional): Number of time intervals to compute. Defaults to 12.
    interval_duration (int, optional): Hours between each time interval. Defaults to 2 hours.

    Returns:
    tuple: A tuple containing the overall minimum and maximum rh differences.
    """
    time_intervals = [initial_time + pd.Timedelta(hours=interval_duration * i) for i in range(intervals)]
    abs_min, abs_max = float('inf'), float('-inf')

    for my_time in time_intervals:
        data = list(calc_filtered_dict(model_name, my_time).values())

        if data:
            current_min = min(data, key=lambda x: x[variable])[variable]
            current_max = max(data, key=lambda x: x[variable])[variable]
            abs_min = min(abs_min, current_min)
            abs_max = max(abs_max, current_max)

    return abs_min, abs_max


def plot_2D_rh_diff_hobos_model(model_name, add_wind=False):
    """
    Plot differences of rh between HOBOS - model at specified intervals.

    Args:
        model_name (str): Name of the model used
        add_wind (binary): If true, also winds are plotted on top

    """
    time_intervals = [pd.to_datetime("2017-10-15T14:00:00") + pd.Timedelta(hours=2 * i) for i in range(12)]

    abs_min_temp, abs_max_temp = find_absolute_min_max_rh_diff(initial_time=pd.to_datetime("2017-10-15T14:00:00"),
                                                               variable="rh_difference",
                                                               model_name=model_name)

    biggest_diff_t = max(abs(abs_min_temp), abs(abs_max_temp))

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
    with rasterio.open(dem_file_hobos_extent) as clipped:
        dem_data = clipped.read(1)

    for i, my_time in enumerate(time_intervals):
        # Filter data and calculate temperature differences for the current time interval (similar to previous steps)

        data_dict = calc_filtered_dict(my_time=my_time, model_name=model_name)
        # Extract latitude, longitude, and temperature differences from filtered_data_dict
        all_lat = [value['latitude'] for value in data_dict.values()]
        all_lon = [value['longitude'] for value in data_dict.values()]
        all_rh_diff = [value['rh_difference'] for value in data_dict.values()]

        # Get the current subplot
        ax = axs[i]
        min_lon = np.min(all_lon) - 0.01
        max_lon = np.max(all_lon) + 0.01
        min_lat = np.min(all_lat) - 0.01
        max_lat = np.max(all_lat) + 0.01

        extent = [min_lon, max_lon, min_lat, max_lat]

        contour_h = ax.contour(dem_data, colors="gray", extent=extent, origin="upper", alpha=0.5)
        ax.clabel(contour_h, inline=True, fontsize=8, colors='black', fmt='%1.0f')

        # Plot rh differences on the current subplot using scatter plot
        sc = ax.scatter(all_lon, all_lat, c=all_rh_diff, s=100, alpha=0.7, cmap="BrBG",
                        transform=ccrs.PlateCarree(), vmin=-50, vmax=50)  # set it to 50

        if add_wind:

            plot_wind_observation(ax, my_time, color="red")  # plot always the observation data

            if model_name == "AROME":
                plot_wind_arome(ax, my_time, min_lon, max_lon, min_lat, max_lat)

            if model_name == "UKMO":
                plot_wind_ukmo(ax, my_time, min_lon, max_lon, min_lat, max_lat)

            elif model_name == "ICON":
                time = pd.to_datetime(my_time)
                if time.day > 15:  # TODO only day 16
                    plot_wind_icon(ax, time)

            elif model_name == "WRF_ACINN":
                plot_wind_wrf_acinn(ax, my_time, min_lon, max_lon, min_lat, max_lat)

        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False  # Disable labels on top
        gl.right_labels = False  # Disable labels on right

        # Add colorbar to the current subplot
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label(u'ΔRH (%)')
        ax.set_extent(extent, crs=ccrs.PlateCarree())  # if I do not set the extent, then all observations are visible

        # Set title for the current subplot
        ax.set_title(f'{my_time}')

    # Adjust layout and spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust top margin to leave space for suptitle

    plt.suptitle(f"Relative Humidity Difference: Hobos - {model_name}", fontsize=16)
    # Show the plot

    if add_wind:
        plt.savefig(f"{dir_PLOTS}/hobos/humidity_and_wind/2D_RHDiff_HOBOS_{model_name}_wind.png")
    else:
        plt.savefig(f"{dir_PLOTS}/hobos/humidity/2D_RhDiff_HOBOS_{model_name}.png")


def plot_3D_AROME_rh_as2D(model_name):
    """Plot the Temperature of the AROME Model in the background, in the front plot the Hobos observations
    :param model_name: The name of the model used


    :param add_wind:


    """
    padding = 0.02
    min_lon = 11.3398110103 - padding
    max_lon = 11.4639758751 + padding
    min_lat = 47.2403724414 - padding
    max_lat = 47.321
    extent = [min_lon, max_lon, min_lat, max_lat]

    time_intervals = [pd.to_datetime("2017-10-15T14:00:00") + pd.Timedelta(hours=2 * i) for i in
                      range(12)]  # Generate 12 time intervals

    # Create a 4x3 grid of subplots
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Flatten the axs array to iterate over all subplots
    axs = axs.flatten()

    with rasterio.open(dem_file_hobos_extent) as clipped:
        dem_data = clipped.read(1)

    # loop over 2h time steps
    for i, time in enumerate(time_intervals):
        my_cmap = "BrBG"
        df_model = extract_rh_for_slice(model_name=model_name, extent=extent, my_time=time)

        ax = axs[i]

        contour_h = ax.contour(dem_data, colors="gray", extent=extent, origin="upper", alpha=0.5)
        ax.clabel(contour_h, inline=True, fontsize=8, colors='black', fmt='%1.0f')

        pcm = ax.pcolormesh(df_model.longitude, df_model.latitude, df_model,
                            shading='auto', cmap=my_cmap, vmin=0, vmax=100)

        # Add colorbar
        cbar = fig.colorbar(pcm, ax=ax, label='RH (%)')
        cbar.set_label('Relative Humidity (%)')

        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        station_keys = ds['STATION_KEY'].values

        all_lat = [ds.sel(STATION_KEY=station_key)['lat'].values for station_key in station_keys]
        all_lon = [ds.sel(STATION_KEY=station_key)['lon'].values for station_key in station_keys]
        all_rh = [ds.sel(STATION_KEY=station_key)['rh'].sel(TIME=time, method="nearest").values for
                  station_key in
                  station_keys]

        # Plot all temperature data on the map
        ax.scatter(all_lon, all_lat, c=all_rh, s=100, cmap=my_cmap, alpha=1, transform=ccrs.PlateCarree(),
                   edgecolors='black', linewidth=1.5, vmin=0, vmax=100)

        # Add map features
        gl = ax.gridlines(draw_labels=True, auto_inline=False)
        gl.top_labels = False  # Disable labels on top
        gl.right_labels = False  # Disable labels on right

        # Set title
        ax.set_title(f'{time}')
        ax.set_extent([min_lon, max_lon - 0.01, min_lat, max_lat])

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Hobos RH',
                                  markerfacecolor='white', markersize=10, markeredgecolor="black")]
    ax.legend(handles=legend_elements, loc='lower right', fontsize='medium')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust top margin to leave space for suptitle
    plt.suptitle(f"Relative Humidity: HOBOS in the front, and {model_name} in the back")

    plt.savefig(f"{dir_PLOTS}/hobos/humidity/2D_RH_HOBOS_AROME.png")


if __name__ == '__main__':
    # Open the hobos observations
    ds = xr.open_dataset(hobos_file)

    # start only one model_name per time
    # plot_2D_rh_diff_hobos_model(model_name="AROME", add_wind=True)
    # plot_2D_rh_diff_hobos_model(model_name="UKMO", add_wind=True)
    # plot_2D_rh_diff_hobos_model(model_name="ICON", add_wind=True)
    # plot_2D_rh_diff_hobos_model(model_name="WRF_ACINN", add_wind=True)

    plot_3D_AROME_rh_as2D(model_name="AROME")
    plt.show()
