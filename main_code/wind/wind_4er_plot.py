"""Plot a figure with 4 stations, inside show the wind direction and windspeed of the different models and
observation"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MultipleLocator
from metpy.units import units

from main_code.AROME.read_in_arome import read_timeSeries_AROME
from main_code.ICON_model.read_icon_model_meteogram import read_icon_at_lowest_level
from main_code.UKMO_model.read_ukmo import get_ukmo_fixed_point_lowest_level
from main_code.WRF_Helen.read_wrf_meteogram import read_lowest_level_meteogram_helen
from main_code.confg import MOMMA_stations_PM, MOMMA_stations, momma_our_period_file, station_files_zamg, stations_ibox, \
    colordict, dir_PLOTS


def plot_winds_four_panel(dates, ds_obs, ds_arome, ds_icon, ds_wrf, ds_ukmo, plot_index, station_name):
    """Function to plot wind speed and direction for observations, AROME model, and ICON model on subplots."""
    # Assuming a figure with 18 subplots (6 rows x 3 columns)
    ax1 = plt.subplot(2, 2, plot_index)

    # Plot observation wind speed and direction
    ax1.plot(dates, ds_obs["WindSpeed"].values, label='Obs Wind Speed', color=colordict["RADIOSONDE"])
    ax1.fill_between(dates, ds_obs["WindSpeed"].values, 0, alpha=0.1)
    # Plot AROME model wind speed and direction
    ax1.plot(dates, ds_arome["WindSpeed"].values, label='AROME Wind Speed', color=colordict["AROME"])
    ax1.fill_between(dates, ds_arome["WindSpeed"].values, 0, alpha=0.1)

    # plot WRF model wind speed and dir
    ax1.plot(dates, ds_wrf["WindSpeed"].values, label='WRF-ACINN Wind Speed', color=colordict["WRF_ACINN"])
    ax1.fill_between(dates, ds_wrf["WindSpeed"].values, 0, alpha=0.1)

    # Plot ICON model wind speed and direction
    ax1.plot(dates, ds_icon["WindSpeed"].values, label='ICON Wind Speed', color=colordict["ICON"])
    ax1.fill_between(dates, ds_icon["WindSpeed"].values, 0, alpha=0.1)

    ax1.plot(dates, ds_ukmo["windspeed"].values, label='UKMO Wind Speed', color=colordict["UKMO"])
    ax1.fill_between(dates, ds_ukmo["windspeed"].values, 0, alpha=0.1)

    ymin, ymax, ystep = (0, 20, 2)
    ax1.set_ylim(ymin, ymax)
    ax1.yaxis.set_major_locator(MultipleLocator(ystep))

    ax1.set_ylabel('Wind Speed (m/s)')
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left')
    ax1.set_title(f'{station_name}')

    ax2 = ax1.twinx()
    ax2.plot(dates, ds_obs["WindDirection"].values, '.', label='Obs Wind Direction', color=colordict["RADIOSONDE"],
             markersize=7)
    ax2.plot(dates, ds_arome["WindDirection"].values, '.', label="AROME Wind Direction", color=colordict["AROME"])
    ax2.plot(dates, ds_wrf["WindDirection"].values, '.', label="WRF-ACINN Wind Direction", color=colordict["WRF_ACINN"])

    ax2.plot(dates, ds_icon["WindDirection"], '.', label='ICON Wind Direction', color=colordict["ICON"])
    ax2.plot(dates, ds_ukmo["wind_dir"], '.', label='UKMO Wind Direction', color=colordict["UKMO"])
    ax2.set_ylabel('Direction (degrees)')
    ax2.set_ylim(0, 360)
    ax2.set_yticks(np.arange(45, 405, 90))
    ax2.set_yticklabels(['NE', 'SE', 'SW', 'NW'])
    ax2.legend(loc='upper right')

    # Set date formatting
    locator = mdates.HourLocator(interval=2)  # Locate every 2 hours
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Ensure everything fits without overlap

    # Additional setup for wind speed and direction ranges and labels can be added here
    ax1.set_ylabel('Wind Speed (m/s)')
    ax1.set_xlabel('Time')


def fetch_arome_data(name):
    """read in arome data"""
    df_timeseries = read_timeSeries_AROME(name).sel(time=slice('2017-10-15T14:00:00', '2017-10-16T12:00:00'))

    u = df_timeseries.ts_u * units("m/s")
    v = df_timeseries.ts_v * units("m/s")

    wdir = mpcalc.wind_direction(u, v, convention='from')
    wspeed = mpcalc.wind_speed(u, v)

    return wdir, wspeed, df_timeseries.time.values


def read_in_MOMMA(name):
    """Function to read in MOMMA station by name"""
    momma_name_station = MOMMA_stations_PM[name]["name"]

    station_key_str = [k for k, v in MOMMA_stations.items() if v == momma_name_station]
    station_key = int(station_key_str[0])
    df_observation_Momma = xr.open_dataset(momma_our_period_file).sel(STATION_KEY=station_key)

    ws = df_observation_Momma["wspeed"].values
    wd = df_observation_Momma["wdir"].values
    return wd, ws, df_observation_Momma.time.values, momma_name_station


def fetch_observation_data(name):
    """read obs data"""
    if name.startswith("PM"):
        return read_in_MOMMA(name)

    elif name in station_files_zamg:
        df = pd.read_csv(station_files_zamg[name]["filepath"])
        df['time'] = pd.to_datetime(df['time'])
        start_time = "2017-10-15 14:00:00"
        end_time = "2017-10-16 12:00:00"
        df_final = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        ws = df_final["FF"]
        wd = df_final["DD"]

        return wd, ws, df_final.time.values, station_files_zamg[name]["name"]

    elif name in stations_ibox:

        df = pd.read_csv(stations_ibox[name]["filepath"])
        # its counted from the bottom, so at the station Hochhaeuser we take the second level (not the first) because
        # the first level is too low (1.5m)
        if name == "NF27":
            u_ibox = df["mean_u2"].values * units("m/s")  # is the streamwise
            v_ibox = df["mean_v2"].values * units("m/s")  # spanwise (should always be 0) before the coordinate rotation
            wdir_ibox = df["wdir2"].values * units("degree")

        else:
            u_ibox = df["mean_u1"].values * units("m/s")  # is the streamwise
            v_ibox = df["mean_v1"].values * units("m/s")  # spanwise (should always be 0) before the coordinate rotation
            wdir_ibox = df["wdir1"].values * units("degree")

    if v_ibox.sum() != 0:
        print("Is the spanwise component should be 0 m/s")

    wspeed_ibox = mpcalc.wind_speed(u_ibox, v_ibox)
    df['time'] = pd.to_datetime(df['Datetime'])

    return wdir_ibox, wspeed_ibox, df.time.values, stations_ibox[name]["name"]


def fetch_icon_data(name):
    """read icon data"""
    df_icon_wind, icon_height = read_icon_at_lowest_level(name)

    u_icon = df_icon_wind.u.values * units("m/s")
    v_icon = df_icon_wind.v.values * units("m/s")

    wdir_icon = mpcalc.wind_direction(u_icon, v_icon, convention='from')
    wspeed_icon = mpcalc.wind_speed(u_icon, v_icon)

    return wdir_icon, wspeed_icon, df_icon_wind.time.values


def fetch_wrf_helen_data(name):
    """read WRF data"""
    df_wrf = read_lowest_level_meteogram_helen(name)

    u_wrf = df_wrf.ts_u.values * units("m/s")
    v_wrf = df_wrf.ts_v.values * units("m/s")

    wdir_icon = mpcalc.wind_direction(u_wrf, v_wrf, convention='from')
    wspeed_icon = mpcalc.wind_speed(u_wrf, v_wrf)

    return wdir_icon, wspeed_icon, df_wrf.time.values


def fetch_ukmo_data(name):
    data_ukmo = get_ukmo_fixed_point_lowest_level(name)
    return data_ukmo


def convert_to_dataframe(dates, ws, wd):
    df = pd.DataFrame({
        'WindSpeed': ws,
        'WindDirection': wd
    }, index=pd.to_datetime(dates))
    return df


if __name__ == '__main__':

    names_MOMMA = []
    for i in np.arange(2, 11):
        names_MOMMA.append(f"PM{i:02d}")

    timeseries_names_no_MOMMA = ["IAO", "JEN", "KUF", "LOWI", "NF10", "NF27", "SF1", "SF8", "VF0"]
    all_timeseries_names = timeseries_names_no_MOMMA + names_MOMMA

    # Example call for each station
    subplot_index = 1  # Start at 1, increment for each new subplot
    stat_names = []
    plt.figure(figsize=(16, 12))

    for station in all_timeseries_names:
        stat_names.append(station)
        if subplot_index % 5 == 0:
            plt.tight_layout()
            plt.savefig(f"{dir_PLOTS}/wind/wind_4er_plot_{stat_names}")

            stat_names = []
            subplot_index = 1
            plt.figure(figsize=(16, 12))  # Create a large figure to hold all subplots
        # Fetch data for this station
        # Assuming you have functions that return wind speed and direction for each model and observations
        wd_obs, ws_obs, dates_obs, name_station = fetch_observation_data(station)
        wd_arome, ws_arome, dates_arome = fetch_arome_data(station)
        wd_icon, ws_icon, dates_icon = fetch_icon_data(station)
        wd_wrf, ws_wrf, dates_wrf = fetch_wrf_helen_data(station)

        # Assuming your data retrieval functions have already fetched the data
        if station in names_MOMMA or station in stations_ibox:
            df_obs = convert_to_dataframe(dates_obs, ws_obs, wd_obs)
        else:
            df_obs = convert_to_dataframe(dates_obs, ws_obs.values, wd_obs.values)
        df_arome = convert_to_dataframe(dates_arome, ws_arome, wd_arome)
        df_icon = convert_to_dataframe(dates_icon, ws_icon, wd_icon)
        df_wrf_resampled = convert_to_dataframe(dates_wrf, ws_wrf, wd_wrf)  # is already resampled to 30min

        df_ukmo = fetch_ukmo_data(station)

        # Define the frequency for resampling
        frequency = '30min'  # 10 minutes

        # Resample dataframes to the specified frequency
        df_obs_resampled = df_obs.resample(frequency).interpolate()
        df_arome_resampled = df_arome.resample(frequency).interpolate()
        df_icon_resampled = df_icon.resample(frequency).interpolate()
        df_ukmo_resampled = df_ukmo.resample(frequency).interpolate()

        plot_winds_four_panel(dates=df_arome_resampled.index, ds_obs=df_obs_resampled, ds_arome=df_arome_resampled,
                              ds_icon=df_icon_resampled, ds_ukmo=df_ukmo_resampled, ds_wrf=df_wrf_resampled,
                              plot_index=subplot_index, station_name=name_station)
        subplot_index += 1

    plt.show()
