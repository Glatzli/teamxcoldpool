"""Wind at 2m observation and Model (Timeseries)"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MultipleLocator
from metpy.units import units

from main_code.AROME.read_in_arome import read_timeSeries_AROME
from main_code.ICON_model.read_icon_model_meteogram import read_icon_at_lowest_level, icon_short_names_dict
from main_code.UKMO_model.read_ukmo import get_ukmo_fixed_point_lowest_level
from main_code.WRF_Helen.read_wrf_meteogram import read_wrf_without_resampling
from main_code.confg import MOMMA_stations_PM, MOMMA_stations, momma_our_period_file, station_files_zamg, stations_ibox, \
    dir_PLOTS


def plot_winds(dates, ws, wd, subplot_index, name_station=None, model_name=None, plot_range=None):
    """Function to plot wind speed and direction on subplots with formatted date ticks.

        :param dates: Dates for the x-axis.
        :param ws: Wind speed data.
        :param wd: Wind direction data.
        :param subplot_index: Index for the subplot position.
        :param name_station: [Optional] Name of the station for the plot title.
        :param model_name: [Optional] The name of the model
        :param plot_range: [Optional] Y-axis range for wind speed plot.

    """
    ax1 = plt.subplot(3, 2, subplot_index)  # Use subplot_index to determine subplot position

    # Plot wind speed
    ax1.plot(dates, ws, label='Wind Speed')
    ymin, ymax, ystep = plot_range if plot_range else (0, 20, 2)
    ax1.set_ylim(ymin, ymax)
    ax1.yaxis.set_major_locator(MultipleLocator(ystep))
    ax1.fill_between(dates, ws, 0, alpha=0.1)
    ax1.set_ylabel('Wind Speed (m/s)')
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left')

    # Title setup based on station name availability
    if name_station:
        ax1.set_title(f'Observation for {name_station}')
    elif model_name:
        ax1.set_title(f'{model_name} MODEL')
    else:
        ax1.set_title('Unknown MODEL')

    # Twin axis for wind direction (on the right)
    ax2 = ax1.twinx()
    ax2.plot(dates, wd, '.k', label='Wind Direction', markersize=2)
    ax2.set_ylabel('Direction (degrees)')
    ax2.set_ylim(0, 360)
    ax2.set_yticks(np.arange(45, 405, 90))
    ax2.set_yticklabels(['NE', 'SE', 'SW', 'NW'])
    ax2.legend(loc='upper right')

    # Set up date formatting and tick locating
    locator = mdates.HourLocator(interval=2)  # Locate every 2 hours
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    # Rotate x-tick labels for better visibility
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to ensure everything fits without overlap


def plot_momma_obs(PM_number):
    """Function to read in Momma Data and call function plot_winds_four_panel for observation"""

    name_station = MOMMA_stations_PM[PM_number]["name"]
    lon = MOMMA_stations_PM[PM_number]["longitude"]
    lat = MOMMA_stations_PM[PM_number]["latitude"]
    height = MOMMA_stations_PM[PM_number]["height"]

    station_key_str = [k for k, v in MOMMA_stations.items() if v == name_station]
    station_key = int(station_key_str[0])
    df_observation_Momma = xr.open_dataset(momma_our_period_file).sel(STATION_KEY=station_key)

    plot_winds(dates=df_observation_Momma.time.values, ws=df_observation_Momma["wspeed"].values,
               wd=df_observation_Momma["wdir"].values, subplot_index=1,
               name_station=f"{name_station} ({lon},{lat}) {height}m")


def plot_zamg_station(df, station_name, lon, lat, hoehe):
    """Plot given the df the wind and direction of the given zamg csv"""
    df['time'] = pd.to_datetime(df['time'])
    start_time = "2017-10-15 14:00:00"
    end_time = "2017-10-16 12:00:00"
    df_final = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

    plot_winds(dates=df_final.time.values, ws=df_final["FF"], wd=df_final["DD"], subplot_index=1,
               name_station=f"{station_name} ({lon},{lat}) {hoehe}m")


def plot_ibox_station(df, station_name, lon, lat, height):
    """Plot the wind for the ibox station (15min averages)"""
    u_ibox = df["mean_u1"].values * units("m/s")  # is the streamwise
    v_ibox = df["mean_v1"].values * units("m/s")  # spanwise (should always be 0) before the coordinate rotation

    if v_ibox.sum() != 0:
        print("Is the spanwise component should be 0 m/s")

    # wdir = mpcalc.wind_direction(u, v, convention='from')
    wdir_ibox = df["wdir1"] * units("degree")
    wspeed_ibox = mpcalc.wind_speed(u_ibox, v_ibox)
    df['time'] = pd.to_datetime(df['Datetime'])
    plot_winds(dates=df.time.values, ws=wspeed_ibox, wd=wdir_ibox, subplot_index=1,
               name_station=f"{station_name} ({lon},{lat}) {height}m")


if __name__ == '__main__':
    """Main Routine to plot all Winddirections and velocties from the timeseries"""
    names_MOMMA = []
    for i in np.arange(2, 11):
        names_MOMMA.append(f"PM{i:02d}")

    timeseries_names_no_MOMMA = ["IAO", "JEN", "KUF", "LOWI", "NF10", "NF27", "SF1", "SF8", "VF0"]
    all_timeseries_names = timeseries_names_no_MOMMA + names_MOMMA

    print(len(all_timeseries_names))

    for name in all_timeseries_names:
        # read AROME timeseries, get wdir and wspeed
        df_timeseries = read_timeSeries_AROME(name).sel(time=slice('2017-10-15T14:00:00', '2017-10-16T12:00:00'))

        u = df_timeseries.ts_u * units("m/s")
        v = df_timeseries.ts_v * units("m/s")

        wdir = mpcalc.wind_direction(u, v, convention='from')
        wspeed = mpcalc.wind_speed(u, v)

        plt.figure(figsize=(16, 10))  # Adjust figure size based on the number of subplots

        # MOMMA

        # Check if the station name starts with "PM" and plot accordingly
        if name.startswith("PM"):
            plot_momma_obs(name)
        # If the station name is in the station_files dictionary, read the CSV and plot
        elif name in station_files_zamg:
            df = pd.read_csv(station_files_zamg[name]["filepath"])
            plot_zamg_station(df, station_name=station_files_zamg[name]["name"], lon=station_files_zamg[name]["lon"],
                              lat=station_files_zamg[name]["lat"], hoehe=station_files_zamg[name]["hoehe"])
        elif name in stations_ibox:
            df = pd.read_csv(stations_ibox[name]["filepath"])
            plot_ibox_station(df, station_name=stations_ibox[name]["name"], lon=stations_ibox[name]["longitude"],
                              lat=stations_ibox[name]["latitude"], height=stations_ibox[name]["height"])

        # read ICON timeseries, get wdir and wspeed
        if name in icon_short_names_dict.keys():
            df_icon_wind, icon_height = read_icon_at_lowest_level(name)

            u_icon = df_icon_wind.u.values * units("m/s")
            v_icon = df_icon_wind.v.values * units("m/s")

            wdir_icon = mpcalc.wind_direction(u_icon, v_icon, convention='from')
            wspeed_icon = mpcalc.wind_speed(u_icon, v_icon)

            plot_winds(dates=df_icon_wind.time.values, ws=wspeed_icon, wd=wdir_icon, subplot_index=3, model_name="ICON")

        # get ukmo data
        data_ukmo = get_ukmo_fixed_point_lowest_level(name)
        plot_winds(dates=data_ukmo.index, ws=data_ukmo["windspeed"].values, wd=data_ukmo["wind_dir"].values,
                   subplot_index=4, model_name="UKMO")

        data_wrf = read_wrf_without_resampling(name)
        u_wrf = data_wrf.ts_u * units("m/s")
        v_wrf = data_wrf.ts_v * units("m/s")

        wdir_wrf = mpcalc.wind_direction(u_wrf, v_wrf, convention='from')
        wspeed_wrf = mpcalc.wind_speed(u_wrf, v_wrf)
        plot_winds(dates=data_wrf.time.values, ws=wspeed_wrf, wd=wdir_wrf, subplot_index=5, model_name="WRF-ACINN")

        # Now plot the AROME Timeseries station data
        plot_winds(dates=df_timeseries.time.values, ws=wspeed, wd=wdir, subplot_index=2, model_name="AROME")

        plt.tight_layout()
        plt.savefig(f"{dir_PLOTS}/wind/wind_{name}")
    plt.show()
