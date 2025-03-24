"""Script to plot the temperature timeseries (meteogram at 2m and lowest level of 3D models) of the different models
with observations """

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import pandas as pd
import plotly.graph_objects as go
from metpy.units import units

from main_code.AROME.read_in_arome import read_2D_variables_AROME, read_3D_variables_AROME, read_timeSeries_AROME
from main_code.ICON_model.read_icon_model_meteogram import icon_short_names_dict, read_icon_at_lowest_level
from main_code.UKMO_model.read_ukmo import get_ukmo_fixed_point_lowest_level, get_ukmo_height_of_city_name
from main_code.WRF_Helen.read_wrf_meteogram import read_lowest_level_meteogram_helen
from main_code.confg import cities, station_files_zamg, dir_PLOTS


def plot_combined_png(df_surface, df_5m, df_2m, time, city_name, city_infos):
    """Png-Plot combined of temperature (observations and AROME model) of different stations"""

    height_model_city = f"{df_5m.interp(time=time)['z'].values:.0f}"  # as label

    # PLot 2D (0m)
    plt.plot(df_surface['tsk'].time, df_surface['tsk'], linestyle='--',
             label=f"{city_name} AROME_2D {height_model_city}m",
             color=city_infos["color"])

    # Plot first level of 3D (5m)
    plt.plot(df_5m['T'].time, df_5m['T'], linestyle='-.',
             label=f"{city_name} AROME_3D {height_model_city}m",
             color=city_infos["color"])

    # plot 2m
    plt.plot(df_2m["ts_t"].time, df_2m["ts_t"], label=f"2m {city_name} AROME", color=city_infos["color"])

    # Plot observation data
    if city_name == "Muenchen":
        df = pd.read_csv(city_infos['csv'], index_col=False)
    else:
        df = pd.read_csv(city_infos['csv'])
    df['time'] = pd.to_datetime(df['time'])
    start_time = "2017-10-15 14:00:00"
    end_time = "2017-10-16 12:00:00"
    df_final = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    plt.plot(df_final.time, df_final.TLMAX, label=f"{city_name} Observation {city_infos['hoehe']}m",
             color=city_infos["color"])

    # Customize the plot
    plt.title('Temperature Comparison: AROME, ICON, UKMO Model Output vs Observations')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)


def plot_combined_plotly(df_surface, df_5m, df_2m, time, city_name, city_infos):
    """Interactive Plot: use plotly to make the plot combined of temperature (observations and models)
    time is only used to determine the height
    """

    height_model_city_3D = f"{df_5m.interp(time=time)['z'].values:.0f}m"  # as label
    height_model_surface = f"{df_surface.interp(time=time)['hgt'].values:.0f}m"
    timeseries_arome_height = f"{df_surface.interp(time=time)['hgt'].values + 2:.0f}m"  # on 2m

    # UKMO
    data_ukmo = get_ukmo_fixed_point_lowest_level(city_name)
    ukmo_station_height = f"{get_ukmo_height_of_city_name(city_name):.0f}m"
    fig.add_trace(
        go.Scatter(x=data_ukmo.index, y=data_ukmo["temperature"], legendgroup=f"{city_name}",
                   legendgrouptitle_text=f"{city_name}",
                   mode='lines', name=f"UKMO_3D {ukmo_station_height}",
                   line=dict(color=city_infos["color"], width=2, dash="dashdot")))

    # WRF ACINN, needs some time
    data_wrf_helen = read_lowest_level_meteogram_helen(city_name, add_temperature=True)

    wrf_station_height = f"{data_wrf_helen['height'].values:.0f}m"
    fig.add_trace(
        go.Scatter(x=data_wrf_helen.time, y=data_wrf_helen["temperature"], legendgroup=f"{city_name}",
                   legendgrouptitle_text=f"{city_name}",
                   mode='lines', name=f"WRF-ACINN {wrf_station_height}",
                   line=dict(color=city_infos["color"], width=6, dash="dashdot")))

    fig.add_trace(go.Scatter(x=df_5m['time'], y=df_5m['T'],
                             legendgroup=f"{city_name}",
                             legendgrouptitle_text=f"{city_name}",
                             mode='lines+markers', name=f"AROME_3D {height_model_city_3D}",
                             line=dict(color=city_infos["color"])))

    fig.add_trace(go.Scatter(x=df_2m['time'], y=df_2m['ts_t'],
                             legendgroup=f"{city_name}",
                             legendgrouptitle_text=f"{city_name}",
                             mode='lines', name=f"AROME_Timeseries on 2m {timeseries_arome_height}",
                             line=dict(color=city_infos["color"], width=4, dash='dot')))

    # Add model data trace
    fig.add_trace(go.Scatter(x=df_surface['time'], y=df_surface['tsk'],
                             legendgroup=f"{city_name}",
                             legendgrouptitle_text=f"{city_name}",
                             mode='lines', name=f"AROME_2D getuned on 0.5m {height_model_surface}",
                             line=dict(color=city_infos["color"], width=4, dash="dash")))

    if city_name in icon_short_names_dict:
        df_icon_t2m, icon_height = read_icon_at_lowest_level(city_name)
        icon_height = f"{icon_height:.0f}m"
        fig.add_trace(go.Scatter(x=df_icon_t2m["time"], y=df_icon_t2m["t2m_C"], legendgroup=f"{city_name}",
                                 legendgrouptitle_text=f"{city_name}",
                                 mode='lines', name=f"ICON_Timeseries on 2m {icon_height}",
                                 line=dict(color=city_infos["color"], width=4, dash="solid")))

    # Load and process observation data
    if city_name == "Muenchen":
        df = pd.read_csv(city_infos['csv'], index_col=False)
    else:
        df = pd.read_csv(city_infos['csv'])

    df['time'] = pd.to_datetime(df['time'])

    df_final = df[(df['time'] >= "2017-10-15 14:00:00") & (df['time'] <= "2017-10-16 12:00:00")]

    # Add observation data trace
    fig.add_trace(go.Scatter(x=df_final['time'], y=df_final['TLMAX'],
                             legendgroup=f"{city_name}",
                             legendgrouptitle_text=f"{city_name}",
                             mode='lines', name=f"Observation {city_infos['hoehe']}m", connectgaps=True,
                             line=dict(color="black", width=8)))

    # Customize the layout
    fig.update_layout(title='Temperature Comparison: AROME Model Output vs Observations',
                      xaxis_title='Time',
                      yaxis_title='Temperature (°C)',
                      legend_title='Stations',
                      template='plotly_white')


def main_routine(city_name, city_info):
    """main_lidar_routine routine to plot temperature timeseries"""
    df_2D = read_2D_variables_AROME(variableList=["tsk", "hgt"], lat=city_info["lat"], lon=city_info["lon"],
                                    slice_lat_lon=False)
    df_2D["tsk"] = df_2D['tsk'].metpy.convert_units('degC')

    # read the 3D vars at level 1
    df_3D = read_3D_variables_AROME(variables=["z", "th", "p"], method="sel", level=90, lat=city_info["lat"],
                                    lon=city_info["lon"],
                                    slice_lat_lon=False)

    df_3D["temperature"] = mpcalc.temperature_from_potential_temperature(df_3D["p"], df_3D["th"])
    df_3D['T'] = df_3D['temperature'].metpy.convert_units('degC')

    # read timeseries
    matching_stations = [city_name_short for city_name_short, v in station_files_zamg.items() if
                         v["name"] == city_name]

    if len(matching_stations) != 0:
        df_2m = read_timeSeries_AROME(location=matching_stations[0])
        df_2m = df_2m.sel(time=slice('2017-10-15T14:00:00', '2017-10-16T12:00:00'))
        df_2m["ts_t"] = df_2m["ts_t"] * units("K")
        df_2m["ts_t"] = df_2m["ts_t"].metpy.convert_units('degC')

        plot_combined_plotly(df_surface=df_2D, df_2m=df_2m, df_5m=df_3D, time=fixed_point_time, city_name=city_name,
                             city_infos=city_info)


if __name__ == '__main__':
    # prepare  temperature
    fixed_point_time = "2017-10-16T05:00:00"

    city_names = list(cities.keys())
    fig = go.Figure()
    for city_name, city_info in cities.items():
        main_routine(city_name, city_info)  # call main_lidar_routine routine

    fig.write_html(f"{dir_PLOTS}/temperature/timeseries_temp.html")

    fig.show()
