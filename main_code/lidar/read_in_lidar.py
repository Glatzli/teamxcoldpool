"""Skript to read in LIDAR"""

import warnings
from os.path import join, basename

import glob2 as glob2
import pandas as pd
import xarray as xr

from main_code.confg import vars_lidar

warnings.filterwarnings("ignore")


def process_lidar(path_lidar, start_date, end_date, resample=True, name='SL88', resample_time=10):
    """
    Process lidar variable. Stored in a xarray dataset, not interpolated over
    mean heights

    Parameters
    ----------
    path_lidar : str
        Path where to find lidar data.
    end_date : str
        End date data.
    resample: bool, optional
        True: the dataset is resampled with resample_time. Default: True
    resample_time: int, optional
        time of resampling (mins). Default: 10

    Returns
    -------
    ds : xarray dataset
        dataset with u,v,w,ff,dd as a function of time and height.
        :param start_date:

    """

    # get path of SL88 lidar data
    path_lidar = glob2.glob(join(path_lidar, f"{name}*.nc"))

    # define start and end date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # iterate over available files
    i = 0
    for file, path in zip([basename(f) for f in path_lidar], path_lidar):
        day = pd.to_datetime(file.split(sep='_')[1])  # get date from file name
        if day.date() < start_date.date() or day.date() > end_date.date():
            continue
        with xr.open_dataset(path, decode_times=False) as temp:
            datetimes = pd.to_datetime(temp.time.values, unit='s', utc=False)

            ds_temp = xr.Dataset({'u': (['height', 'time'], temp.ucomp.values),
                                  'v': (['height', 'time'], temp.vcomp.values),
                                  'w': (['height', 'time'], temp.wcomp.values),
                                  'ff': (['height', 'time'], temp.ff.values),
                                  'dd': (['height', 'time'], temp.dd.values)},
                                 coords={'height': temp.height.values + temp.alt,
                                         'time': datetimes.to_pydatetime()}
                                 )

        if i == 0:
            i = 1
            ds = ds_temp
            continue
        ds = ds.merge(ds_temp)  # combine datasets

    # resample data to xx minute mean
    if resample is True:
        ds = ds.resample(time=f'{resample_time}min', label='right').mean()

    # adapt dataset
    ds = ds.assign_attrs(dict(Name=f"Lidar {name}",
                              longitude=temp.lon,
                              latitude=temp.lat,
                              altitude=temp.alt,
                              station=f"Lidar {name}",
                              resample_time_min=resample_time if resample is True else ''))

    ds = ds.sel(time=slice(start_date, end_date))

    # invert height coordinate to match model data
    ds = ds.isel(height=slice(None, None, -1))

    return ds


def read_lidar_obs(path=None, name=None, start_date='2017-10-15 14:00',
                   end_date='2017-10-16 12:00',
                   resample=True, resample_time=10):
    """read lidar observation and return quantified dataset"""
    ds = None  # Initialize ds to None

    if not path:
        # path lidar (either SL88 or SLXR142)
        path = '/Data/Observations/LIDAR/SL88_vad_l2'
    if not name:
        name = "SL88"

    ds = process_lidar(path, start_date, end_date, resample=resample, name=name,
                       resample_time=resample_time)

    ds = ds.sel(height=ds.height <= 1600)
    for var, units in vars_lidar.items():
        if var in ds:
            ds[var].attrs['units'] = units

    return ds.metpy.quantify()
