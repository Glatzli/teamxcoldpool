"""Read in ICON Meteogram
It includes certain stations (listed in the dict below)
"""
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from metpy.calc import specific_humidity_from_mixing_ratio
from metpy.units import units

# dict that holds as keys station_names, as values the icon model names
from main_code.confg import icon_folder_meteogram

icon_short_names_dict = {
    'IAO': 'Innsbruck Atmospheric Observatory',
    'LOWI': 'AWS Innsbruck airport',
    'KUF': 'AWS Kufstein',
    'JEN': 'AWS Jenbach',
    "PM02": 'PIANO M02',
    "PM03": 'PIANO M03',
    "PM04": 'PIANO M04',
    "PM05": 'PIANO M05',
    "PM06": 'PIANO M06 (rooftop site, 50 m AGL)',
    "PM07": 'PIANO M07',
    "PM08": 'PIANO M08',
    "PM09": 'PIANO M09',
    "PM10": 'PIANO M10',
    "VF0": 'i-Box VF0',
    "SF8": 'i-Box SF8',
    "SF1": 'i-Box SF1',
    "NF10": 'i-Box NF10',
    "NF27": 'i-Box NF27',
    'Innsbruck Uni': 'Innsbruck Atmospheric Observatory',
    'Innsbruck Airport': 'AWS Innsbruck airport',
    'Kufstein': 'AWS Kufstein',
    'Jenbach': 'AWS Jenbach'
}


def plot_variable_at_lowest_level(df_icon, variable, city_name):
    plt.figure(figsize=(15, 7))
    plt.plot(df_icon.iloc[:, 0], df_icon.iloc[:, 1], label='Degree C')
    plt.title(f'{variable} at {city_name}')
    plt.xlabel('Date and Time')

    # Set up the locator for every 3 hours and the formatter for the dates on x-axis
    locator = mdates.HourLocator(interval=2)  # Locate every 3 hours
    formatter = mdates.ConciseDateFormatter(locator)  # Format the date concisely

    plt.gca().xaxis.set_major_locator(locator)  # Apply the locator to the x-axis
    plt.gca().xaxis.set_major_formatter(formatter)  # Apply the formatter to the x-axis

    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.legend()
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()


def read_icon_at_lowest_level(station_name):
    """read Icon Meteogram for specific station
        ::param station_name: selected station_name
        ::returns pd.DataFrame: time, t2m_C, u, v, w, tke, sh (specific humidity)
    """
    data = xr.open_dataset(f"{icon_folder_meteogram}/METEOGRAM_TeamEx_IFS_ICON_ctl_v1.nc")

    # Decode station names and find index of the specified station
    decoded_station_names = np.char.decode(data.station_name.values)

    station_index = np.where(decoded_station_names == icon_short_names_dict[f"{station_name}"])[0][0]

    t2m_data = data.sel(nstations=station_index, nvars=0,
                        max_nlevs=89)

    # Convert temperature from Kelvin to Celsius and filter based on date
    t2m_celsius = t2m_data['values'] - 273.15

    # Select water vapor mixing ratio data for the specified station and level
    u_data = data.sel(nstations=station_index, nvars=1,
                      max_nlevs=89)  # cannot take level 90 because then sometimes underground

    v_data = data.sel(nstations=station_index, nvars=2,
                      max_nlevs=89)  # cannot take level 90 because then sometimes underground

    w_data = data.sel(nstations=station_index, nvars=3,
                      max_nlevs=89)  # cannot take level 90 because then sometimes underground

    tke = data.sel(nstations=station_index, nvars=4,
                   max_nlevs=89)  # cannot take level 90 because then sometimes underground

    sh_data = data.sel(nstations=station_index, nvars=5,
                       max_nlevs=89)  # cannot take level 90 because then sometimes underground
    sh_hum = specific_humidity_from_mixing_ratio(
        sh_data['values'] * units('kg/kg')) * 1000  # convert it from water mixing ratio to specific humidity

    icon_height = np.round(u_data["heights"].values, 2)
    print(f'The height of {icon_short_names_dict[f"{station_name}"]} is: {icon_height}')

    # Convert date strings to datetime objects and filter dates
    dates = pd.to_datetime(np.char.decode(data['date'].values))
    mask = dates >= pd.Timestamp('2017-10-15 14:00:00', tz='UTC')

    u_ms = u_data['values'] * units('m/s')  # convert it from water mixing ratio to specific humidity
    v_ms = v_data['values'] * units('m/s')
    w_ms = w_data["values"] * units("m/s")
    tke = tke["values"] * units("m**2/s**2")

    df_icon = pd.DataFrame({
        "time": dates[mask],
        "t2m_C": t2m_celsius[mask],
        "u": u_ms[mask],
        "v": v_ms[mask],
        "w": w_ms[mask],
        "tke": tke[mask],
        "sh": sh_hum[mask]
    })
    return df_icon, icon_height


def read_icon_meteogram_fixed_time(station_name, time):
    """
    Read vertical profile of all icon variables at specific station and time

    :param time: the specific time
    :param station_name: selected station name
    :return: A pd.DataFrame with time and temperature in Celsius
    """

    # Load the dataset
    data = xr.open_dataset(f"{icon_folder_meteogram}/METEOGRAM_TeamEx_IFS_ICON_ctl_v1.nc")

    # Decode station names and find index of the specified station
    decoded_station_names = np.char.decode(data.station_name.values)
    station_index = np.where(decoded_station_names == icon_short_names_dict[f"{station_name}"])[0][0]

    # Select temperature data for the specified station and level
    all_variables_data = data.sel(nstations=station_index)
    # print(all_variables_data["station_lon"].values)
    # print(all_variables_data["station_lat"].values)
    dates = pd.to_datetime(np.char.decode(data['date'].values)).round('s')
    time_diff = np.abs(dates - pd.Timestamp(time, tz='UTC'))
    min_diff_index = np.argmin(time_diff)

    selected_data = all_variables_data.isel(time=min_diff_index).sel(
        max_nlevs=slice(0, 90))  # keep only the specific time, and kick last nlevel (has no sense last level

    heights = np.round(selected_data.sel(nvars=5)["heights"].values, 2)

    temp = selected_data.sel(nvars=0)["values"] - 273.15
    u_wind = selected_data.sel(nvars=1)["values"]
    v_wind = selected_data.sel(nvars=2)["values"]
    w_wind = selected_data.sel(nvars=3)["values"]
    tke = selected_data.sel(nvars=4)["values"]
    sh_hum = specific_humidity_from_mixing_ratio(
        selected_data.sel(nvars=5)['values'] * units(
            'kg/kg')) * 1000  # convert it from water mixing ratio to specific humidity g/kg

    df_icon = pd.DataFrame({
        "height": heights,
        "temp_C": temp,
        "u_wind": u_wind,
        "v_wind": v_wind,
        "w_wind": w_wind,
        "tke": tke,
        "sh": sh_hum
    })

    return df_icon


if __name__ == '__main__':
    df_rh, icon_height = read_icon_at_lowest_level("LOWI")

    plot_variable_at_lowest_level(df_rh, "specific humidity", "Innsbruck Uni")  # create an example plot
    df, h = read_icon_at_lowest_level("IAO")
    print(read_icon_at_lowest_level("Innsbruck Uni"))
    df2 = read_icon_meteogram_fixed_time("IAO", '2017-10-15 16:00:00')
    print(df2)
