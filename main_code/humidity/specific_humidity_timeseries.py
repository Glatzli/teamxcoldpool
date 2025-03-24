"""Make some plots of the specific humidity over time"""

import warnings

import metpy.calc as mpcalc
import pandas as pd
import plotly.graph_objects as go
from metpy.units import units

from main_code.AROME.read_in_arome import read_timeSeries_AROME, read_3D_variables_AROME
from main_code.ICON_model.read_icon_model_meteogram import icon_short_names_dict, read_icon_at_lowest_level
from main_code.UKMO_model.read_ukmo import get_ukmo_fixed_point_lowest_level, get_ukmo_height_of_city_name
from main_code.WRF_Helen.read_wrf_meteogram import read_lowest_level_meteogram_helen
from main_code.confg import cities, station_files_zamg, dir_PLOTS

warnings.filterwarnings('ignore')


def plot_combined_plotly(df_arome_2m, time, df_arome_3D_5m, name_of_city, city_infos):
    """Interactive Plot: use plotly to make the plot combined of specific humidity
    """

    height_model_city_3D = f"{df_arome_3D_5m.interp(time=time)['z'].values:.0f}m"  # as label
    timeseries_arome_height = f"{df_arome_3D_5m.interp(time=time)['z'].values - 3:.0f}m"  # on 2m

    data_ukmo = get_ukmo_fixed_point_lowest_level(name_of_city)
    ukmo_station_height = f"{get_ukmo_height_of_city_name(name_of_city):.0f}m"

    # Meteogram of WRF ACINN
    df_wrf_acinn = read_lowest_level_meteogram_helen(name_of_city)

    fig.add_trace(
        go.Scatter(x=df_wrf_acinn.time, y=df_wrf_acinn["specific_humidity"], legendgroup=f"{name_of_city}",
                   legendgrouptitle_text=f"{name_of_city}",
                   mode='lines', name=f"WRF_ACINN",
                   line=dict(color=city_infos["color"], width=6, dash="dashdot")))

    fig.add_trace(
        go.Scatter(x=data_ukmo.index, y=data_ukmo["specific_humidity"], legendgroup=f"{name_of_city}",
                   legendgrouptitle_text=f"{name_of_city}",
                   mode='lines', name=f"UKMO_3D {ukmo_station_height}",
                   line=dict(color=city_infos["color"], width=2, dash="dashdot")))

    fig.add_trace(go.Scatter(x=df_arome_3D_5m['time'], y=df_arome_3D_5m['q'] * 1000,
                             legendgroup=f"{name_of_city}",
                             legendgrouptitle_text=f"{name_of_city}",
                             mode='lines+markers', name=f"AROME_3D {height_model_city_3D}",
                             line=dict(color=city_infos["color"])))

    # Load and process Tawes OBS data
    df = pd.read_csv(city_infos['csv'])
    df['time'] = pd.to_datetime(df['time'])
    start_time = "2017-10-15 14:00:00"
    end_time = "2017-10-16 12:00:00"
    df_final = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    # add specific humidity g/kg
    df_final["sp"] = 1000 * mpcalc.specific_humidity_from_dewpoint(pressure=df_final["P"].values * units("hPa"),
                                                                   dewpoint=df_final["TP"].values * units("degC"))

    if name_of_city in icon_short_names_dict:
        df_icon, icon_height = read_icon_at_lowest_level(name_of_city)
        icon_height = f"{icon_height:.0f}m"  # on 2m
        fig.add_trace(go.Scatter(x=df_icon["time"], y=df_icon["sh"], legendgroup=f"{name_of_city}",
                                 legendgrouptitle_text=f"{name_of_city}",
                                 mode='lines', name=f"ICON_Timeseries on 2m {icon_height}",
                                 line=dict(color=city_infos["color"], width=4)))

    # Add observation data trace
    fig.add_trace(go.Scatter(x=df_arome_2m['time'], y=df_arome_2m['ts_q'],
                             legendgroup=f"{name_of_city}",
                             legendgrouptitle_text=f"{name_of_city}",
                             mode='lines', name=f"AROME_Timeseries on 2m {timeseries_arome_height}",
                             line=dict(color=city_infos["color"], width=4, dash='dot')))

    fig.add_trace(go.Scatter(x=df_final["time"], y=df_final['sp'],
                             legendgroup=f"{name_of_city}",
                             legendgrouptitle_text=f"{name_of_city}",
                             mode='lines', name=f"Observation {city_infos['hoehe']}m", connectgaps=True,
                             line=dict(color="black", width=8)))

    # Customize the layout
    fig.update_layout(title='Specific Humidity Comparison: AROME Model Output vs Observations',
                      xaxis_title='Time',
                      yaxis_title='Specific Humidity (g/kg)',
                      legend_title='Stations',
                      template='plotly_white')

    fig.write_html(f"{dir_PLOTS}/humidity/zeitreihen_specific_humidity.html")


if __name__ == '__main__':
    fixed_point_time = "2017-10-16T05:00:00"

    fig = go.Figure()

    for city_name, city_info in cities.items():
        df_3D = read_3D_variables_AROME(variables=["q", "z"], method="sel", slice_lat_lon=False,
                                        lon=city_info["lon"],
                                        lat=city_info["lat"],
                                        level=90)

        # read timeseries
        matching_stations = [city_name_short for city_name_short, v in station_files_zamg.items() if
                             v["name"] == city_name]

        if (matching_stations.__len__() == 0) or (city_name == "Muenchen"):
            continue
        else:
            df_2m = read_timeSeries_AROME(location=matching_stations[0])
            df_2m = df_2m.sel(time=slice('2017-10-15T14:00:00', '2017-10-16T12:00:00'))
            df_2m["ts_q"] = df_2m["ts_q"] * 1000 * units("g/kg")

            plot_combined_plotly(df_arome_2m=df_2m, time=fixed_point_time, df_arome_3D_5m=df_3D, name_of_city=city_name,
                                 city_infos=city_info)

    fig.show()
