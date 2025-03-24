"""Script to plot HOBOS temperature data, and differences between AROME/UKMO/WRF/ICON model and HOBOS

Added also wind data on top of the models

Note: it's impossible to read in METEOGRAM data, because the stations locations of the HOBOS are not available,
so we need the 3D data first level fixed time
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from metpy.units import units

from main_code.AROME.read_in_arome import read_3D_variables_AROME
from main_code.ICON_model.read_icon_model_3D import read_icon_fixed_point_and_time
from main_code.UKMO_model.read_ukmo import get_ukmo_fixed_point_lowest_level, get_rotated_index_of_lat_lon
from main_code.WRF_Helen.read_wrf_helen import read_wrf_fixed_point_and_time, read_wrf_fixed_time
from main_code.confg import dem_file_hobos_extent, dir_PLOTS, ukmo_folder, hobos_file, icon_folder_3D, \
    MOMMA_stations_PM, momma_our_period_file, cities


def extract_temperature_for_slice(model_name, extent, my_time):
    """extract the temperature for a slice of latitudes and longitudes"""
    # TODO only implemented for AROME until now, would need some more time to do it also for other models
    min_lon, max_lon, min_lat, max_lat = extent

    if model_name == "AROME":
        df_arome = read_3D_variables_AROME(variables=["p", "th"], method="sel",
                                           lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat), slice_lat_lon=True,
                                           level=90.0, time=pd.to_datetime(my_time))

        arome_temp = mpcalc.temperature_from_potential_temperature(df_arome["p"],
                                                                   df_arome["th"])

        return arome_temp.metpy.convert_units("degC")


def extract_temperature_points(model_name, lon, lat, my_time):
    """extract the temperature for a single point"""
    time = pd.to_datetime(my_time)
    if model_name == "AROME":
        df_arome = read_3D_variables_AROME(variables=["p", "th"], method="sel",
                                           lon=lon, lat=lat, slice_lat_lon=False,
                                           level=90.0, time=time)

        arome_temp = mpcalc.temperature_from_potential_temperature(df_arome["p"],
                                                                   df_arome["th"])
        arome_temp = arome_temp.metpy.convert_units("degC")

        return arome_temp.metpy.unit_array.magnitude
    elif model_name == "UKMO":

        df_ukmo = get_ukmo_fixed_point_lowest_level(lat=lat, lon=lon)
        return df_ukmo.loc[str(my_time)]["temperature"]

    elif model_name == "WRF_ACINN":
        ds_wrf_3d_lowest_level = read_wrf_fixed_point_and_time(longitude=lon,
                                                               latitude=lat, day=time.day, hour=time.hour,
                                                               minute=time.minute)
        ds_wrf_3d_lowest_level = ds_wrf_3d_lowest_level.isel(bottom_top=0)
        return ds_wrf_3d_lowest_level["temperature"].values

    elif model_name == "ICON":  # needs brutal long
        if time.day == 16:
            df_icon_nearest = read_icon_fixed_point_and_time(day=time.day, hour=time.hour, my_lon=lon,
                                                             my_lat=lat)

            return df_icon_nearest.isel(height=-1)["temperature"].values
        else:
            return 0
    else:
        raise AssertionError(f"No model named like {model_name}")


def calc_filtered_dict(model_name, my_time):
    """
    Calculate the station data dictionary with temperature differences between HOBOS and a specified model.
    Exclude NaN values in HOBOS temperature data.

    Args:
    model_name (str): Name of the model used to compare with HOBOS.
    my_time (datetime): The specific time for which the data is calculated.

    Returns:
    dict: A dictionary containing temperature data and differences for stations.
    """
    station_data_dict = {}

    # Get unique station keys (assuming 'STATION_KEY' is a dimension in your dataset)
    station_keys = ds['STATION_KEY'].values

    for station_key in station_keys:
        # Select data for the current station key
        station_data = ds.sel(STATION_KEY=station_key)

        # Extract latitude, longitude, and Hobos temperature data
        lat = station_data['lat'].values.item()
        lon = station_data['lon'].values.item()
        hobos_temp = station_data['ta'].sel(TIME=my_time, method="nearest").values.item()

        # Read model data and calculate temperature
        model_temp = extract_temperature_points(model_name, lon, lat, my_time)

        # Check if Hobos temperature is not NaN to include in the result
        if not np.isnan(hobos_temp):
            station_data_dict[station_key] = {
                'latitude': lat,
                'longitude': lon,
                'hobos_temperature': hobos_temp,
                'model_temperature': model_temp,
                'temperature_difference': hobos_temp - model_temp
            }

    return station_data_dict


def find_absolute_min_max_temp_diff(initial_time, model_name, variable="temperature_difference", intervals=12,
                                    interval_duration=2):
    """
    Find the absolute minimum and maximum of temperature difference over specified time intervals.

    Args:
    initial_time (datetime): The start time for calculating temperature differences.
    variable (str): The key in the dictionary to calculate min and max for (e.g., 'temperature_difference').
    model_name (str): Name of the model to use.
    intervals (int, optional): Number of time intervals to compute. Defaults to 12.
    interval_duration (int, optional): Hours between each time interval. Defaults to 2 hours.

    Returns:
    tuple: A tuple containing the overall minimum and maximum temperature differences.
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


def plot_wind_arome(ax, my_time, min_lon, max_lon, min_lat, max_lat):
    """plot AROME wind on lowest level"""
    df_arome = read_3D_variables_AROME(variables=["u", "v", "z"], method="sel", level=90,
                                       lat=slice(min_lat, max_lat),
                                       lon=slice(min_lon, max_lon),
                                       slice_lat_lon=True)

    u_sub = df_arome['u'].sel(time=pd.to_datetime(my_time))
    v_sub = df_arome['v'].sel(time=pd.to_datetime(my_time))

    # Plot wind vectors
    ax.quiver(df_arome.longitude, df_arome.latitude, u_sub, v_sub,
              color='black', scale=50)


def plot_wind_ukmo(ax, my_time, min_lon, max_lon, min_lat, max_lat):
    """plot ukmo wind on lowest level"""
    min_x, min_y = get_rotated_index_of_lat_lon(longitude=min_lon, latitude=min_lat)  # unten links
    max_x, max_y = get_rotated_index_of_lat_lon(longitude=max_lon, latitude=max_lat)  # oben rechts

    # Initialize an empty dictionary to store the processed data arrays
    data_arrays = {}

    # Iterate over the variable names, loading and processing each dataset
    for var in ["u", "v"]:
        # Construct file path and open dataset
        file_path = f"{ukmo_folder}/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_{var}.nc"
        data = xr.open_dataset(file_path)

        # Select data for the specific time and extract a subset based on latitude and longitude
        data_at_time = data.sel(time=str(my_time))
        subset = data_at_time.isel(grid_latitude=slice(min_y, max_y), grid_longitude=slice(min_x, max_x),
                                   model_level_number=0, bnds=1)

        # Store the relevant variable from the dataset in the dictionary
        if var == "v":
            data_arrays["transformed_y_wind"] = subset["transformed_y_wind"]
        elif var == "u":
            data_arrays["transformed_x_wind"] = subset["transformed_x_wind"]

        # Print height values as a debug statement
        print(f"Height values for {var}:", subset["level_height"].values)  # Debug statement

    # Create a new dataset from the dictionary of data arrays
    df_wrf_reprojected = xr.Dataset(data_arrays)

    lon0, lat0 = -168.6, 42.7

    # Instantiate the Cartopy rotated pole projection
    source_proj = ccrs.RotatedPole(pole_longitude=lon0, pole_latitude=lat0)
    target_proj = ccrs.PlateCarree()

    # Extract rotated latitude and longitude values and create 2D grids
    lonr, latr = df_wrf_reprojected["grid_longitude"].values, df_wrf_reprojected["grid_latitude"].values
    xx, yy = np.meshgrid(lonr, latr)

    x_points, y_points = target_proj.transform_points(source_proj, xx, yy)[..., :2].T

    # Plotting
    ax.coastlines()

    u_wind = df_wrf_reprojected['transformed_x_wind']
    v_wind = df_wrf_reprojected['transformed_y_wind']

    # Plot wind vectors
    ax.quiver(x_points, y_points, u_wind, v_wind, transform=target_proj, scale=50)


def plot_wind_icon(ax, my_time):
    """plot ICON wind on lowest level"""
    time = pd.to_datetime(my_time)
    icon_file = f'ICON_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{time.day:02d}T{time.hour:02d}0000Z.nc'

    # Load dataset
    ds_icon = xr.open_dataset(f"{icon_folder_3D}/{icon_file}")

    # Get the variables
    clon = np.rad2deg(ds_icon['clon'].values)
    clat = np.rad2deg(ds_icon['clat'].values)

    u_10m = ds_icon['u_10m'].squeeze().values
    v_10m = ds_icon['v_10m'].squeeze().values

    ax.quiver(clon, clat, u_10m, v_10m, color='black', scale=50)


def plot_wind_wrf_acinn(ax, my_time, min_lon, max_lon, min_lat, max_lat):
    """plot WRF-ACINN wind on lowest level"""
    ds_wrf = read_wrf_fixed_time(my_time=str(my_time), min_lon=10, max_lon=13, min_lat=46,
                                 max_lat=50,
                                 lowest_level=True)  # read lowest level of WRF

    # read AROME model, to reproject the ds_wrf to the ds_arome
    df_arome = read_3D_variables_AROME(variables=["u", "v", "z"], method="sel", level=90,
                                       lat=slice(min_lat, max_lat),
                                       lon=slice(min_lon, max_lon),
                                       slice_lat_lon=True)

    df_arome = df_arome.isel(time=0)
    df_arome = df_arome.drop_vars("nz").drop_vars("time")

    df_reprojected = df_arome.salem.transform(ds_wrf,
                                              interp='spline')  # reproject the WRF grid to the AROME (lat, lon) grid

    u_sub = df_reprojected['u']
    v_sub = df_reprojected['v']

    # Plot wind vectors
    ax.quiver(df_reprojected.longitude, df_reprojected.latitude, u_sub, v_sub,
              color='black', scale=50)


def plot_wind_observation(ax, my_time, color):
    """Read in wind observed data and plot TAWES and MOMMA data in green
    :param ax: axes of the figure
    :param my_time: the selected time
    :param color: select a color for the observations (in temp I have selected green, in rh red.)
    """
    # TAWES DD and FF
    for tawes_station, tawes_values in cities.items():
        if tawes_station != "Muenchen":
            df_tawes = pd.read_csv(tawes_values["csv"])

            df_tawes['time'] = pd.to_datetime(df_tawes['time'])

            # Filter the DataFrame where 'time' equals 'my_time'
            filtered_df = df_tawes[df_tawes['time'] == pd.to_datetime(my_time)]
            if len(filtered_df["FF"]) != 0:
                u, v = mpcalc.wind_components(filtered_df["FF"].values.item() * units("m/s"),
                                              filtered_df["DD"].values.item() * units("degree"))

                ax.quiver(tawes_values["lon"], tawes_values["lat"], u.magnitude, v.magnitude, scale=50,
                          color=color)

    # MOMMA station, wdir and wspeed
    for momma_station, momma_info in MOMMA_stations_PM.items():
        df_observation_Momma = xr.open_dataset(momma_our_period_file).sel(STATION_KEY=momma_info["key"])
        df_momma_time = df_observation_Momma.sel(time=pd.to_datetime(my_time))
        u_momma, v_momma = mpcalc.wind_components(df_momma_time["wspeed"].values * units("m/s"),
                                                  df_momma_time["wdir"].values * units("degree"))

        ax.quiver(df_momma_time["lon"].values, df_momma_time["lat"].values, u_momma.magnitude,
                  v_momma.magnitude, scale=50,
                  color=color)


def plot_2D_temp_diff_hobos_model(model_name, add_wind=False):
    """
    Plot differences of temperature between HOBOS - model at specified intervals.

    Args:
    initial_time (datetime): The start time for plotting temperature differences.
    model_name (str): Name of the model used in temperature calculations.
    add_wind (binary): True or False (default) if you want to add the wind quiver
    """
    if model_name not in ["ICON", "UKMO", "AROME", "WRF_ACINN"]:
        raise AssertionError(
            f"Only MODELS: AROME; ICON; WRF_ACINN; UKMO; are implemented, your {model_name} is not valid!")

    time_intervals = [pd.to_datetime("2017-10-15T14:00:00") + pd.Timedelta(hours=2 * i) for i in range(12)]

    abs_min_temp, abs_max_temp = find_absolute_min_max_temp_diff(initial_time=pd.to_datetime("2017-10-15T14:00:00"),
                                                                 variable="temperature_difference",
                                                                 model_name=model_name)

    biggest_diff_t = max(abs(abs_min_temp), abs(abs_max_temp))

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
    # open the DEM once
    with rasterio.open(dem_file_hobos_extent) as clipped:
        dem_data = clipped.read(1)

    # time_intervals
    for i, my_time in enumerate(time_intervals):
        # Filter data and calculate temperature differences for the current time interval (similar to previous steps)

        data_dict = calc_filtered_dict(my_time=my_time, model_name=model_name)
        # Extract latitude, longitude, and temperature differences from filtered_data_dict
        all_lat = [value['latitude'] for value in data_dict.values()]
        all_lon = [value['longitude'] for value in data_dict.values()]
        all_temp_diff = [value['temperature_difference'] for value in data_dict.values()]

        # Get the current subplot
        ax = axs[i]
        min_lon = np.min(all_lon) - 0.01
        max_lon = np.max(all_lon) + 0.01
        min_lat = np.min(all_lat) - 0.01
        max_lat = np.max(all_lat) + 0.01

        extent = [min_lon, max_lon, min_lat, max_lat]

        contour_h = ax.contour(dem_data, colors="gray", extent=extent, origin="upper",
                               alpha=0.5)  # need to specify upper origin
        ax.clabel(contour_h, inline=True, fontsize=8, colors='black', fmt='%1.0f')

        # Plot temperature differences on the current subplot using scatter plot
        sc = ax.scatter(all_lon, all_lat, c=all_temp_diff, s=100, alpha=0.7, cmap="coolwarm",
                        transform=ccrs.PlateCarree(), vmin=-10,
                        vmax=10)  # TODO change here vmin, vmax (set it to -10 and +10 to compare between models), for one model you can use biggest_diff_t

        if add_wind:

            plot_wind_observation(ax, my_time, color="green")  # plot always the observation data

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
        cbar.set_label(u'ΔT (°C)')
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        # Set title for the current subplot
        ax.set_title(f'{my_time}')

    # Adjust layout and spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust top margin to leave space for suptitle

    plt.suptitle(f"Temperature Difference: Hobos - {model_name}", fontsize=16)
    # Show the plot
    if add_wind:
        plt.savefig(f"{dir_PLOTS}/hobos/temperature_and_wind/2D_TemperatureDiff_HOBOS_{model_name}_points_wind.png")
    else:
        plt.savefig(f"{dir_PLOTS}/hobos/temperature/2D_TemperatureDiff_HOBOS_{model_name}_points.png")


def plot_3D_AROME_temp_as2D(model_name):
    """Plot the Temperature of the AROME Model in the background, in the front plot the Hobos observations"""
    padding = 0.02
    min_lon = 11.3398110103 - padding
    max_lon = 11.4639758751 + padding
    min_lat = 47.2403724414 - padding
    max_lat = 47.321
    extent = [min_lon, max_lon, min_lat, max_lat]

    time_intervals = [pd.to_datetime("2017-10-15T14:00:00") + pd.Timedelta(hours=2 * i) for i in
                      range(12)]  # Generate 12 time intervals

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Flatten the axs array to iterate over all subplots
    axs = axs.flatten()
    # fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Loop over each time interval and plot the temperature differences
    with rasterio.open(dem_file_hobos_extent) as clipped:
        dem_data = clipped.read(1)

    for i, time in enumerate(time_intervals):
        my_cmap = "turbo"
        temp_model = extract_temperature_for_slice(model_name=model_name, extent=extent, my_time=time)

        ax = axs[i]

        contour_h = ax.contour(dem_data, colors="gray", extent=extent, origin="upper", alpha=0.5)
        ax.clabel(contour_h, inline=True, fontsize=8, colors='black', fmt='%1.0f')

        pcm = ax.pcolormesh(temp_model.longitude, temp_model.latitude, temp_model,
                            shading='auto', cmap=my_cmap, vmin=5, vmax=25)

        # Add colorbar
        cbar = fig.colorbar(pcm, ax=ax, label='Temperature (degC)')
        cbar.set_label('Temperature (degC)')

        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        station_keys = ds['STATION_KEY'].values

        all_lat = [ds.sel(STATION_KEY=station_key)['lat'].values for station_key in station_keys]
        all_lon = [ds.sel(STATION_KEY=station_key)['lon'].values for station_key in station_keys]
        all_ta = [ds.sel(STATION_KEY=station_key)['ta'].sel(TIME=time, method="nearest").values for station_key in
                  station_keys]

        # Plot all temperature data on the map
        ax.scatter(all_lon, all_lat, c=all_ta, s=100, cmap=my_cmap, alpha=1, transform=ccrs.PlateCarree(),
                   vmin=5, vmax=25, edgecolors='black', linewidth=1.5)

        # Add map features
        gl = ax.gridlines(draw_labels=True, auto_inline=False)
        gl.top_labels = False  # Disable labels on top
        gl.right_labels = False  # Disable labels on right

        # Set title
        ax.set_title(f'{time}')
        ax.set_extent([min_lon, max_lon - 0.01, min_lat, max_lat])

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Hobos Temperature',
                                  markerfacecolor='white', markersize=10, markeredgecolor="black")]
    ax.legend(handles=legend_elements, loc='lower right', fontsize='medium')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust top margin to leave space for suptitle
    plt.suptitle(f"Temperatures: HOBOS in the front, and {model_name} in the back")

    plt.savefig(f"{dir_PLOTS}/hobos/temperature/2D_Temperature_HOBOS_{model_name}.png")


if __name__ == '__main__':
    # Open the hobos observations
    ds = xr.open_dataset(hobos_file)

    # start only one model_name per time
    # plot_2D_temp_diff_hobos_model(model_name="ICON")
    # plot_2D_temp_diff_hobos_model(model_name="AROME", add_wind=True)
    plot_2D_temp_diff_hobos_model(model_name="UKMO", add_wind=True)
    # plot_2D_temp_diff_hobos_model(model_name="WRF_ACINN", add_wind=True)

    # hobos temp: min 5.36, max 24.49
    # plot_3D_AROME_temp_as2D(model_name="AROME")

    plt.show()
