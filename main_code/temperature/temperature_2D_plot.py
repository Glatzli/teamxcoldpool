"""Make some plots of the AROME model 2D temperature"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from main_code.AROME.model_topography import add_lat_lon_plot
from main_code.AROME.read_in_arome import read_2D_variables_AROME, read_3D_variables_AROME
from main_code.confg import cities, dir_PLOTS


def plot_combined_png(ds, df_z_level1, time):
    """Png-Plot combined of temperature (observations and models) of different stations"""
    # Load the model dataset

    plt.figure(figsize=(12, 8))

    # Iterate over each city to plot model and observation data
    for city, info in cities.items():
        # Plot model data
        temp_series_model = ds.sel(longitude=info['lon'], latitude=info['lat'], method='nearest')['tsk']
        height_of_model = \
            df_z_level1.sel(longitude=info['lon'], latitude=info['lat'], method='nearest').interp(time=time)['z']
        temp_c_3d_timeSeries = df_z_level1.sel(longitude=info['lon'], latitude=info['lat'], method='nearest')['T']

        height_model_city = f"{height_of_model.values:.0f}"

        # PLot 2D
        plt.plot(temp_series_model.time, temp_series_model, label=f"{city} AROME_2D {height_model_city}m",
                 color=info["color"])

        # Plot first level of 3D
        plt.plot(temp_c_3d_timeSeries.time, temp_c_3d_timeSeries, linestyle='-.',
                 label=f"{city} AROME_3D {height_model_city}m",
                 color=info["color"])

        # Plot observation data
        if city == "Muenchen":
            df = pd.read_csv(info['csv'], index_col=False)
        else:
            df = pd.read_csv(info['csv'])
        df['time'] = pd.to_datetime(df['time'])
        start_time = "2017-10-15 14:00:00"
        end_time = "2017-10-16 12:00:00"
        df_final = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        plt.plot(df_final.time, df_final.TLMAX, '--', label=f"{city} Observation {info['hoehe']}m", color=info["color"])

    # Customize the plot
    plt.title('Temperature Comparison: AROME Model Output vs Observations')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dir_PLOTS}/temperature/all_cities_temp_time_series_combined.png')


def plot_combined_plotly(ds, df_z_level1):
    """Interactive Plot: use plotly to make the plot combined of temperature (observations and models)"""
    # Create a Plotly figure
    fig = go.Figure()

    # Iterate over each city to plot model and observation data
    for city, info in cities.items():
        # Extract model data for the city
        temp_series_model = ds.sel(longitude=info['lon'], latitude=info['lat'], method='nearest')[
            'tsk'].to_dataframe().reset_index()

        height_of_model = \
            df_z_level1.sel(longitude=info['lon'], latitude=info['lat'], method='nearest').interp(
                time=fixed_point_time)['z'].item()

        temp_3d_df = df_z_level1.sel(longitude=info['lon'], latitude=info['lat'], method='nearest')[
            'T'].to_dataframe().reset_index()

        height_model_city = f"{height_of_model:.0f}m"

        # Add model data trace
        fig.add_trace(go.Scatter(x=temp_series_model['time'], y=temp_series_model['tsk'],
                                 legendgroup=f"{city}",
                                 legendgrouptitle_text=f"{city}",
                                 mode='lines', name=f"{city} AROME_2D {height_model_city}",
                                 line=dict(color=info["color"])))

        fig.add_trace(go.Scatter(x=temp_3d_df['time'], y=temp_3d_df['T'],
                                 legendgroup=f"{city}",
                                 legendgrouptitle_text=f"{city}",
                                 mode='lines+markers', name=f"{city} AROME_3D {height_model_city}",
                                 line=dict(color=info["color"])))

        # Load and process observation data
        if city == "Muenchen":
            df = pd.read_csv(info['csv'], index_col=False)
        else:
            df = pd.read_csv(info['csv'])

        print(df)

        df['time'] = pd.to_datetime(df['time'])
        start_time = "2017-10-15 14:00:00"
        end_time = "2017-10-16 12:00:00"
        df_final = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

        # Add observation data trace
        fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['TLMAX'],
                                 legendgroup=f"{city}",
                                 legendgrouptitle_text=f"{city}",
                                 mode='lines', name=f"{city} Observation {info['hoehe']}m", connectgaps=True,
                                 line=dict(color=info["color"], width=8)))

    # Customize the layout
    fig.update_layout(title='Temperature Comparison: AROME Model Output vs Observations',
                      xaxis_title='Time',
                      yaxis_title='Temperature (°C)',
                      legend_title='Stations',
                      template='plotly_white')

    fig.write_html(f"{dir_PLOTS}/temperature/all_cities_temp_time_combined.html")

    # Show the figure
    fig.show()


def spot_stable_stratification(df_first_model_level, df_surface):
    """Calculate the difference between surface temp (0m) and temperature at first model level 5m
    In order to spot cold pools
    """
    for x in ["latitude", "longitude"]:
        # Convert to float64 and round to 3 decimal places for temp_data_not_tuned
        df_first_model_level[x] = df_first_model_level[x].astype(float).round(3)

        # Convert to float64 and round to 3 decimal places for temp_data_tuned
        df_surface[x] = df_surface[x].astype(float).round(3)

    return df_surface - df_first_model_level  # negative if the 0m temperature is smaller


def plot_temperature_height(ds_2D, ds_3D, surface=True, showDiff=False):
    """
    Plots the 2D temperature on the surface and height contours.

    Parameters:
    - ds: xarray.Dataset containing the 2D temperature data.
    - df_3D: xarray.Dataset containing the 3D data for height contours.
    - surface: bool, if True, uses 'tsk' from the 2D as the temperature, surface temp 0m.
             If False, uses 'T' from 'df_3D' as the temperature, from first model level.
    """
    # Determine the global min and max temperature values across both datasets for a consistent color scale
    min_temp = min(ds_2D['tsk'].sel(time=fixed_point_time).min(), ds_3D['T'].sel(time=fixed_point_time).min())
    max_temp = max(ds_2D['tsk'].sel(time=fixed_point_time).max(), ds_3D['T'].sel(time=fixed_point_time).max())

    # Settings for height contours
    basis = 10
    dist = 500 * abstand
    min_height = np.floor(ds_3D["z"].min().item() / basis) * basis
    max_height = np.floor(ds_3D["z"].max().item() / basis) * basis
    height_range = np.arange(min_height.magnitude, max_height.magnitude + 1, dist)

    # Terrain colormap and normalization
    terrain_cmap = plt.get_cmap('gray_r')
    norm = plt.Normalize(vmin=min_height.magnitude, vmax=max_height.magnitude)

    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    # Choose the temperature data based on whether it's from the surface or 10m
    if surface:
        temp_data = ds_2D['tsk'].sel(time=fixed_point_time)
        title = f'Surface Temperature at {fixed_point_time}'
        filename = f"{dir_PLOTS}/temperature/temp2D_and_height_at_{fixed_point_time}.png"
    else:
        temp_data = ds_3D['T'].sel(time=fixed_point_time)
        title = f'Surface Temperature first level 3D AROME at {fixed_point_time}'
        filename = f"{dir_PLOTS}/temperature/temp3D_and_height_at_{fixed_point_time}.png"

    if showDiff:
        temp_data = spot_stable_stratification(df_surface=ds_2D['tsk'].sel(time=fixed_point_time),
                                               df_first_model_level=ds_3D['T'].sel(time=fixed_point_time))
        title = f"Difference of surface T (0m) minus T 1st model level (5m) AROME Modell at {fixed_point_time}"
        filename = f"{dir_PLOTS}/temperature/difference_0m_5m_{fixed_point_time}_{abstand}Degree.png"
        # have to recalculte new colorbar based on the differences
        min_temp = temp_data.min()
        max_temp = temp_data.max()

    # Plot temperature
    temp_plot = ax.pcolormesh(temp_data['longitude'], temp_data['latitude'], temp_data, shading='auto', cmap='coolwarm',
                              vmin=min_temp.metpy.unit_array.item().magnitude,
                              vmax=max_temp.metpy.unit_array.item().magnitude)
    fig.colorbar(temp_plot, ax=ax, label='Temperature (°C)')

    # Overlay height contours
    cs = ax.contour(ds_3D['longitude'], ds_3D['latitude'], ds_3D["z"].sel(time=fixed_point_time),
                    height_range, transform=ccrs.PlateCarree(),
                    colors=[terrain_cmap(norm(level)) for level in height_range])
    ax.clabel(cs, cs.levels, inline=True, fontsize=10)

    # Plot cities and labels
    for city, coord in cities.items():
        marker_color = "black"  # Color for city markers
        ax.plot(coord['lon'], coord['lat'], 'x', color=marker_color)
        if city == "Innsbruck Airport":
            label_offset = -0.05
        elif city == "Rinn":
            label_offset = -0.08
        else:
            label_offset = 0
        ax.text(coord['lon'], coord['lat'] + label_offset, f"{city} {coord['hoehe']}m")

    # Set plot titles and labels
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    add_lat_lon_plot(ax, ds_2D, rotate=90)  # Assuming this function adjusts the tick labels

    plt.savefig(filename)


if __name__ == '__main__':
    # prepare  temperature
    fixed_point_time = "2017-10-16T05:00:00"

    longitudes = [city_info['lon'] for city_info in cities.values()]
    abstand = 0.2
    max_longitude = max(longitudes) + abstand
    min_longitude = min(longitudes) - abstand

    latitudes = [city_info['lat'] for city_info in cities.values()]
    max_latitude = max(latitudes) + abstand
    min_latitude = min(latitudes) - abstand

    latitude = slice(min_latitude, max_latitude)
    longitude = slice(min_longitude, max_longitude)

    # read the 2D vars
    df_2D = read_2D_variables_AROME(variableList=["tsk"], lat=latitude, lon=longitude, slice_lat_lon=True)
    df_2D["tsk"] = df_2D['tsk'].metpy.convert_units('degC')

    # read the 3D vars at level 1
    df_3D = read_3D_variables_AROME(variables=["z", "th", "p"], method="sel", level=90, lat=latitude, lon=longitude,
                                    slice_lat_lon=True)

    df_3D["temperature"] = mpcalc.temperature_from_potential_temperature(df_3D["p"], df_3D["th"])
    df_3D['T'] = df_3D['temperature'].metpy.convert_units('degC')

    # plot it
    # plot_combined_png(df_2D, df_3D, fixed_point_time)
    plot_temperature_height(df_2D, df_3D, surface=False)
    # plot_combined_plotly(df_2D, df_3D)
    plot_temperature_height(df_2D, df_3D, surface=True)
    plt.show()
