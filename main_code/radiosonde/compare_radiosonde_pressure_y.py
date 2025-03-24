"""Compare different models (ICON, AROME, UKMO, WRF_ACINN) with radiosonde, in one plot with pressure as y-variable"""
import warnings

import metpy
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import wrf
import xarray as xr
from matplotlib import pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
from netCDF4 import Dataset

from main_code.AROME.read_in_arome import read_3D_variables_AROME
from main_code.ICON_model.read_icon_model_3D import read_icon_fixed_point_and_time
from main_code.UKMO_model.read_ukmo import read_ukmo_fixed_point_and_time
from main_code.WRF_Helen.read_wrf_helen import read_wrf_fixed_point_and_time
from main_code.confg import radiosonde_csv, station_files_zamg, colordict

warnings.filterwarnings("ignore")

# Specify the launch date and time
launch_date = pd.to_datetime('2017-10-16 02:15:05',
                             format='%Y-%m-%d %H:%M:%S')

time_for_model = pd.to_datetime('2017-10-16 03:00:00',
                                format='%Y-%m-%d %H:%M:%S')  # TODO can change here the time


def read_in_radiosonde_obs():
    """read csv and plot the radiosonde observation of the airport of ibk"""
    df = pd.read_csv(radiosonde_csv, comment="#")

    # Drop rows with NaN values in any column
    df_cleaned = df.dropna()

    # Convert 'time' from seconds since launch to actual datetime
    df_cleaned['time'] = pd.to_timedelta(df_cleaned['time'], unit='s') + launch_date

    # convert also the units of some other variables
    df_cleaned.loc[:, 'pressure'] = df.loc[df_cleaned.index, 'pressure'] / 100
    df_cleaned.loc[:, 'dewpoint'] = df.loc[df_cleaned.index, 'dewpoint'] - 273.15
    df_cleaned.loc[:, 'temperature'] = df.loc[df_cleaned.index, 'temperature'] - 273.15

    df_unique = df_cleaned.drop_duplicates(['time', 'geopotential height'])

    # Then set the index and convert to xarray
    df_unique.set_index(['time'], inplace=True)

    u = df_unique["windspeed"].values * units.meter / units.second

    p = df_unique['pressure'].values * units.hPa
    T = df_unique['temperature'].values * units.degC
    Td = df_unique['dewpoint'].values * units.degC
    wind_speed = u.to(units.knots)
    wind_dir = df_unique['wind direction'].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)
    z = df_unique["geopotential height"].values * units.meter

    ds = xr.Dataset()

    # Add variables to the dataset
    ds['u_wind'] = xr.DataArray(u.magnitude, dims=('height',),
                                coords={'height': df_unique["geopotential height"].values},
                                attrs={'units': str(u.units)})
    ds['v_wind'] = xr.DataArray(v.magnitude, dims=('height',),
                                coords={'height': df_unique["geopotential height"].values},
                                attrs={'units': str(v.units)})
    ds['pressure'] = xr.DataArray(p.magnitude, dims=('height',),
                                  coords={'height': df_unique["geopotential height"].values},
                                  attrs={'units': str(p.units)})
    ds['temperature'] = xr.DataArray(T.magnitude, dims=('height',),
                                     coords={'height': df_unique["geopotential height"].values},
                                     attrs={'units': str(T.units)})
    ds['dewpoint'] = xr.DataArray(Td.magnitude, dims=('height',),
                                  coords={'height': df_unique["geopotential height"].values},
                                  attrs={'units': str(Td.units)})

    return ds.metpy.quantify()


def read_in_arome(time, method, lon, lat):
    """plot the MODEL output of AROME as it would be a Radiosonde"""
    my_variable_list = ["p", "q", "th", "u", "v", "z"]

    if (method == "sel") | (method == "interp"):
        print(f"Your selected method is {method}")
    else:
        raise AttributeError(
            "You have to define a method (sel or interp) how the point near the LOWI should be selected")

    df_final = read_3D_variables_AROME(variables=my_variable_list, method=method, lon=lon, lat=lat, time=time)

    # print(df_final["p"].metpy.unit_array.magnitude) Extract values

    df_final["windspeed"] = metpy.calc.wind_speed(df_final["u"], df_final["v"])
    df_final["wind direction"] = metpy.calc.wind_direction(df_final["u"], df_final["v"], convention='from')
    df_final["temperature"] = metpy.calc.temperature_from_potential_temperature(df_final["p"], df_final["th"])
    df_final["dewpoint"] = metpy.calc.dewpoint_from_specific_humidity(pressure=df_final["p"],
                                                                      temperature=df_final["temperature"],
                                                                      specific_humidity=df_final["q"])

    p = df_final["p"].metpy.unit_array.to(units.hPa)  # Metadata is removed
    T = df_final["temperature"].metpy.unit_array.to(units.degC)

    Td = df_final["dewpoint"].metpy.unit_array
    wind_speed = df_final["windspeed"].metpy.unit_array.to(units.knots)
    wind_dir = df_final['wind direction']
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    ds = xr.Dataset()

    # Add variables to the dataset
    ds['u_wind'] = xr.DataArray(u.magnitude, dims=('height',),
                                coords={'height': df_final["z"].values},
                                attrs={'units': str(u.units)})
    ds['v_wind'] = xr.DataArray(v.magnitude, dims=('height',),
                                coords={'height': df_final["z"].values},
                                attrs={'units': str(v.units)})
    ds['pressure'] = xr.DataArray(p.magnitude, dims=('height',),
                                  coords={'height': df_final["z"].values},
                                  attrs={'units': str(p.units)})
    ds['temperature'] = xr.DataArray(T.magnitude, dims=('height',),
                                     coords={'height': df_final["z"].values},
                                     attrs={'units': str(T.units)})
    ds['dewpoint'] = xr.DataArray(Td.magnitude, dims=('height',),
                                  coords={'height': df_final["z"].values},
                                  attrs={'units': str(Td.units)})

    return ds.metpy.quantify()


def read_in_wrf_eth():
    filepath = f"/media/wieser/PortableSSD/Dokumente/TEAMx/output/wrfout_cap11_d02_2017-10-16_00-00-00"
    wrfin = Dataset(filepath)

    x_y = wrf.ll_to_xy(wrfin=wrfin, timeidx=6, latitude=station_files_zamg["LOWI"]["lat"],
                       longitude=station_files_zamg["LOWI"]["lon"])
    #   - return_val[0,...] will contain the X (west_east) values.
    #         - return_val[1,...] will contain the Y (south_north) values.

    p1 = wrf.getvar(wrfin, "pressure", timeidx=6)
    T1 = wrf.getvar(wrfin, "tc", timeidx=6)
    Td1 = wrf.getvar(wrfin, "td", timeidx=6)
    u1 = wrf.getvar(wrfin, "ua", timeidx=6)
    v1 = wrf.getvar(wrfin, "va", timeidx=6)
    z1 = wrf.getvar(wrfin, "z", timeidx=6)  # komisch startet bei 1000m!

    p = p1[:, x_y[1], x_y[0]] * units.hPa
    T = T1[:, x_y[1], x_y[0]] * units.degC
    Td = Td1[:, x_y[1], x_y[0]] * units.degC
    u = v1[:, x_y[1], x_y[0]] * units('m/s')
    v = u1[:, x_y[1], x_y[0]] * units('m/s')
    z = z1[:, x_y[1], x_y[0]] * units('meter')

    print(z)

    ds = xr.Dataset()

    # Add variables to the dataset
    ds['u_wind'] = xr.DataArray(u.values, dims=('height',),
                                coords={'height': z.values},
                                attrs={'units': 'm/s'})
    ds['v_wind'] = xr.DataArray(v.values, dims=('height',),
                                coords={'height': z.values},
                                attrs={'units': 'm/s'})
    ds['pressure'] = xr.DataArray(p.values, dims=('height',),
                                  coords={'height': z.values},
                                  attrs={'units': 'hPa'})
    ds['temperature'] = xr.DataArray(T.values, dims=('height',),
                                     coords={'height': z.values},
                                     attrs={'units': 'C'})
    ds['dewpoint'] = xr.DataArray(Td.values, dims=('height',),
                                  coords={'height': z.values},
                                  attrs={'units': 'C'})

    return ds.metpy.quantify()


def read_in_ukmo():
    """read ukmo data"""
    data_ukmo = read_ukmo_fixed_point_and_time(city_name="LOWI", time=time_for_model)
    return data_ukmo


def read_radiosonde_wrf_acinn():
    """read in radiosonde data from WRF ACINN"""
    df_acinn = read_wrf_fixed_point_and_time(day=16, hour=3, minute=0, latitude=station_files_zamg["LOWI"]["lat"],
                                             longitude=station_files_zamg["LOWI"]["lon"])
    return df_acinn


def plot_radiosonde_t_ff_dd(df_obs, df_arome, df_ukmo, df_wrf_acinn, df_icon, zoom_in=False):
    """Plot the radiosonde profiles of the observed and the models (temp, ff, dd)

    zoom_in: is only for the wind plot
    """
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.55, 0.85))

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(df_obs["pressure"].values, df_obs["temperature"].values, color=colordict["RADIOSONDE"],
              label="Observation", linewidth=2)
    skew.plot(df_obs["pressure"].values, df_obs["dewpoint"].values, color=colordict["RADIOSONDE"], linewidth=2)

    skew.plot(df_arome["pressure"].values, df_arome["temperature"].values, color=colordict["AROME"], label="AROME",
              linewidth=2)
    skew.plot(df_arome["pressure"].values, df_arome["dewpoint"].values, color=colordict["AROME"], linewidth=2)

    skew.plot(df_ukmo["pressure"].values, df_ukmo["temperature"].values, 'orange', color=colordict["UKMO"],
              label="UKMO", linewidth=2)
    skew.plot(df_ukmo["pressure"].values, df_ukmo["dewpoint"].values, 'orange', color=colordict["UKMO"], linewidth=2)

    skew.plot(df_wrf_acinn["pressure"].values, df_wrf_acinn["temperature"].values, color=colordict["WRF_ACINN"],
              label="WRF-ACINN",
              linewidth=2)
    skew.plot(df_wrf_acinn["pressure"].values, df_wrf_acinn["dewpoint"].values, color=colordict["WRF_ACINN"],
              linewidth=2)

    skew.plot(df_icon["pressure"].values, df_icon["temperature"].values, color=colordict["ICON"], label="ICON",
              linewidth=2)
    skew.plot(df_icon["pressure"].values, df_icon["dewpoint"].values, color=colordict["ICON"], linewidth=2)

    # Change to adjust data limits and give it a semblance of what we want
    skew.ax.set_adjustable('datalim')

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    plt.title(
        f"Model time {pd.to_datetime(time_for_model).strftime('%Y-%m-%d %H:%M')}, Launch Time: {pd.to_datetime(launch_date).strftime('%Y-%m-%d %H:%M')}")

    skew.ax.set_xlabel('Temperature (°C)')
    skew.ax.set_ylabel('Pressure (hPa)')
    skew.ax.set_xlim(-25, 30)
    skew.ax.set_ylim(1000, 100)

    plt.legend()

    plt.savefig("/home/wieser/Dokumente/Teamx/teamxcoldpool/Plots/radiosonde_LOWI/model_vs_obs_radiosonde_ibk.png")

    fig, axarr = plt.subplots(nrows=1, ncols=2)
    axlist = axarr.flatten()

    ff_obs = mpcalc.wind_speed(df_obs["u_wind"], df_obs["v_wind"])
    dd_obs = mpcalc.wind_direction(df_obs["u_wind"], df_obs["v_wind"])

    ff_arome = mpcalc.wind_speed(df_arome["u_wind"], df_arome["v_wind"])
    dd_arome = mpcalc.wind_direction(df_arome["u_wind"], df_arome["v_wind"])

    ff_icon = mpcalc.wind_speed(df_icon["u"], df_icon["v"])
    dd_icon = mpcalc.wind_direction(df_icon["u"], df_icon["v"])

    ff_wrf_helen = mpcalc.wind_speed(df_wrf_acinn["u"], df_wrf_acinn["v"])
    dd_wrf_helen = mpcalc.wind_direction(df_wrf_acinn["u"], df_wrf_acinn["v"])

    ff_ukmo = df_ukmo["windspeed"]
    dd_ukmo = df_ukmo["wind_dir"]

    axlist[0].plot(ff_arome, df_arome['pressure'], label="AROME", color=colordict["AROME"])
    axlist[0].plot(ff_ukmo, df_ukmo["pressure_at_wind_levels"], label="UKMO", color=colordict["UKMO"])
    axlist[0].plot(ff_icon, df_icon["pressure"], label="ICON", color=colordict["ICON"])
    axlist[0].plot(ff_wrf_helen, df_wrf_acinn["pressure"], label="WRF-ACINN", color=colordict["WRF_ACINN"])
    axlist[0].plot(ff_obs, df_obs['pressure'], label="Observation", color=colordict["RADIOSONDE"], linewidth=2)

    axlist[0].invert_yaxis()
    # ax.set_yscale('log')
    axlist[0].set_ylabel("Pressure (hpa)")
    axlist[0].set_xlabel("Wind speed (m/s)")
    axlist[0].set_title("Wind Speed")
    axlist[0].legend()
    axlist[0].grid(True)

    axlist[1].plot(dd_arome, df_arome['pressure'], label="AROME", color=colordict["AROME"])
    axlist[1].plot(dd_ukmo, df_ukmo["pressure_at_wind_levels"], label="UKMO", color=colordict["UKMO"])
    axlist[1].plot(dd_wrf_helen, df_wrf_acinn["pressure"], label="WRF-ACINN", color=colordict["WRF_ACINN"])
    axlist[1].plot(dd_icon, df_icon["pressure"], label="ICON", color=colordict["ICON"])
    axlist[1].plot(dd_obs, df_obs['pressure'], label="Observation", color=colordict["RADIOSONDE"], linewidth=2)
    axlist[1].invert_yaxis()
    # ax.set_yscale('log')
    axlist[1].set_ylabel("Pressure (hpa)")
    axlist[1].set_xlabel("Wind Direction (degree)")
    axlist[1].set_title("Wind Direction")
    axlist[1].grid(True)
    xticks = np.arange(0, 361, 90)
    xtick_labels = ['N', 'E', 'S', 'W', 'N']
    axlist[1].set_xticks(xticks)
    axlist[1].set_xticklabels(xtick_labels)

    if zoom_in:
        axlist[0].set_ylim(top=800)
        axlist[1].set_ylim(top=800)

        plt.savefig("/home/wieser/Dokumente/Teamx/teamxcoldpool/Plots/radiosonde_LOWI/comparison_wind_zoom_in_p.png")
    else:
        plt.savefig("/home/wieser/Dokumente/Teamx/teamxcoldpool/Plots/radiosonde_LOWI/comparison_wind_p.png")


if __name__ == '__main__':
    """plot the radiosonde of innsbruck and the AROME Model
    Can change time of the model on the upper part of the script: time_mean_of_launch"""
    df_obs = read_in_radiosonde_obs()

    df_arome = read_in_arome(time=time_for_model, method="sel", lat=station_files_zamg["LOWI"]["lat"],
                             lon=station_files_zamg["LOWI"]["lon"])

    # df_wrf_eth = read_in_wrf_eth() # not used WRF ETH

    df_ukmo = read_in_ukmo()

    df_icon = read_icon_fixed_point_and_time(day=16, hour=3, my_lon=station_files_zamg["LOWI"]["lon"],
                                             my_lat=station_files_zamg["LOWI"]["lat"])

    df_wrf_acinn = read_radiosonde_wrf_acinn()  # read in wrf acinn

    plot_radiosonde_t_ff_dd(df_obs=df_obs, df_arome=df_arome, df_ukmo=df_ukmo,
                            df_wrf_acinn=df_wrf_acinn, df_icon=df_icon)
    # alternative read in with zoom of the wind plot
    plot_radiosonde_t_ff_dd(df_obs=df_obs, df_arome=df_arome, df_ukmo=df_ukmo,
                            df_wrf_acinn=df_wrf_acinn, df_icon=df_icon, zoom_in=True)

    plt.show()
