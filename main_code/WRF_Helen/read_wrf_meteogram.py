"""Script to read in WRF HELEN METEOGRAM, pro station is one file there"""

import metpy.calc as mpcalc
import pandas as pd
import xarray as xr
from metpy.units import units

# dict with the lat lon information adjusted to the names of the WRF files
from main_code.WRF_Helen.read_wrf_helen import read_wrf_fixed_point
from main_code.confg import wrf_folder

# list of all available stations in the WRF-Meteogram
stations_wrf = {
    "PM10": (11.2143, 47.2744),
    "PM09": (11.2607, 47.2615),
    "PM02": (11.3118, 47.2615),
    "LOWI": (11.3567, 47.26),
    "Innsbruck Airport": (11.3567, 47.26),
    "IAO": (11.3857, 47.2640),
    "Innsbruck Uni": (11.3857, 47.2640),
    "PM05": (11.3903, 47.2784),
    "PM06": (11.3960, 47.2620),
    "PM03": (11.3986, 47.2473),
    "PM04": (11.4105, 47.21033),  # added a 3 to rounding errors
    "PM07": (11.4123, 47.2787),
    "PM08": (11.5700, 47.2931),
    "SF1": (11.61623, 47.3165),  # added a 3 due to rounding errors
    "VF0": (11.6222, 47.3053),
    "SF8": (11.6525, 47.3255),
    "NF10": (11.6730, 47.2998),
    "NF27": (11.6312, 47.2876),
    "Jenbach": (11.7581, 47.3889),
    "JEN": (11.7581, 47.3889),
    "Kufstein": (12.1628, 47.5753),
    "KUF": (12.1628, 47.5753)
}


def format_coords(lon, lat):
    """
    Converts decimal degree coordinates into a formatted string with degrees and exact decimal minutes.

    Parameters:
    - lon (float): Longitude in decimal degrees.
    - lat (float): Latitude in decimal degrees.

    Returns:
    - str: A string formatted as "DDEMMM_DDNMMM", e.g., "11E2143_47N2744".
    """
    # Extract integer degrees from longitude and latitude
    lon_deg = int(lon)
    lat_deg = int(lat)

    # Calculate minutes directly from the decimal part, extending to four digits
    lon_full = abs(lon) * 10000
    lat_full = abs(lat) * 10000

    lon_min = int(lon_full % 10000)  # Extracting the last four digits
    lat_min = int(lat_full % 10000)  # Extracting the last four digits

    # Determine direction based on positive or negative values
    lon_dir = 'E' if lon >= 0 else 'W'
    lat_dir = 'N' if lat >= 0 else 'S'

    # Format the string
    formatted_lon = f"{abs(lon_deg)}{lon_dir}{lon_min:04d}"  # Ensure minute part is zero-padded to 4 digits
    formatted_lat = f"{abs(lat_deg)}{lat_dir}{lat_min:04d}"  # Similarly for latitude

    if formatted_lon[-1] == '0':  # need to check twice (at LOWI)
        formatted_lon = formatted_lon[:-1]
        if formatted_lon[-1] == '0':
            formatted_lon = formatted_lon[:-1]

    if formatted_lat[-1] == '0':
        formatted_lat = formatted_lat[:-1]
        if formatted_lat[-1] == '0':
            formatted_lat = formatted_lat[:-1]

    return f"{formatted_lon}_{formatted_lat}"


def read_meteogram_wrf_helen(longitude: float, latitude: float):
    """Read in WRF Meteogram from Helen at a fixed location (lat, lon)

    :param latitude:
    :param longitude:

    """
    location_format = format_coords(lon=longitude, lat=latitude)
    print(location_format)  # Output should be "11E2143_47N2744"

    filepath = f"{wrf_folder}/WRF_ACINN_timeseries/WRF_ACINN_20171015T1200Z_CAP02_timeseries_30min_{location_format}_HCW.nc"
    return xr.open_dataset(filepath)


def resample_30min(ds):
    """Function to resample seconds to 30min values"""
    base_time = pd.Timestamp('2017-10-15 12:00:00')  # Adjust base time as needed
    ds['time'] = pd.to_datetime(ds['ts_time'], unit='s', origin=base_time)

    # Resample dataset to half-hourly intervals
    # This assumes your data is continuous and starts from a time that aligns with 12 UTC
    ds_resampled = ds.resample(time='30Min').mean()
    ds_resampled = ds_resampled.drop_vars("ts_time")
    return ds_resampled


def read_lowest_level_meteogram_helen(city_name, add_temperature=False):
    """get the lowest level of the Meteogram by WRF Helen, no height and pressure available!
    :param city_name: the name of the city
    :param add_temperature: Binary if temperature from 3D sould be added to meteogram (takes a lot of time to calculate)
    :return: meteogram dataframe

    """

    ds_wrf = read_meteogram_wrf_helen(longitude=stations_wrf[city_name][0],
                                      latitude=stations_wrf[city_name][1])
    ds_wrf = resample_30min(ds_wrf)
    ds_wrf = ds_wrf.isel(time=slice(4, None))

    # calc variables
    ds_wrf["specific_humidity"] = mpcalc.specific_humidity_from_mixing_ratio(
        ds_wrf["ts_q_mixingratio"] * units('kg/kg')) * 1000

    if add_temperature:
        # temperature is missing in Meteogram from WRF, have to use the one from 3D WRF
        # needs a lot of time!
        df_wrf_lowest_level = read_wrf_fixed_point(longitude=stations_wrf[city_name][0],
                                                   latitude=stations_wrf[city_name][1],
                                                   variable_list=["th", "p", "time", "z"], lowest_level=True)

        ds_wrf["temperature"] = df_wrf_lowest_level[
            "temperature"]
        ds_wrf["height"] = df_wrf_lowest_level["height"]
    ds_wrf = ds_wrf.metpy.dequantify()
    return ds_wrf.isel(verticallevel=0)


def read_wrf_without_resampling(city_name):
    """get the lowest level of the Meteogram by WRF Helen, without resempling, to keep a lot of points
    Needed for wind plot
    """

    ds_wrf = read_meteogram_wrf_helen(longitude=stations_wrf[city_name][0],
                                      latitude=stations_wrf[city_name][1])

    ds_wrf['time'] = pd.to_datetime(ds_wrf['ts_time'], unit='s', origin=pd.Timestamp('2017-10-15 12:00:00'))

    ds_wrf = ds_wrf.isel(time=slice(4, None)).isel(verticallevel=0)

    return ds_wrf


def create_lowest_level_wrf_zamg():
    """deprecated

    If it takes too long to read the lowest level inside the plots, than you can create separate files of the lowest
    level of WRF due to RAM overflow (SIGKILL) """
    stations = ["LOWI", "KUF", "JEN", "IAO"]  # stations zamg
    for station in stations:
        ds = read_lowest_level_meteogram_helen(station, add_temperature=True)
        print(ds)
        ds.to_netcdf(
            f"/home/wieser/Dokumente/Teamx/teamxcoldpool/Data/wrf_acinn_temperature_files/wrf_temp_{station}.nc")
