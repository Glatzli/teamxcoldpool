"""Compare different models (ICON and AROME) with radiosonde, in one plot with height as y-variable"""
import warnings

import metpy
import metpy.calc as mpcalc
import pandas as pd
import wrf
import xarray as xr
from matplotlib import pyplot as plt
from metpy.units import units
from netCDF4 import Dataset

from main_code.AROME.read_in_arome import read_3D_variables_AROME
from main_code.ICON_model.read_icon_model_meteogram import read_icon_meteogram_fixed_time
from main_code.confg import radiosonde_csv, station_files_zamg, dir_PLOTS

warnings.filterwarnings("ignore")

# Specify the launch date and time
launch_date = pd.to_datetime('2017-10-16 02:15:05',
                             format='%Y-%m-%d %H:%M:%S')

time_for_model = pd.to_datetime('2017-10-16 03:00:00',
                                format='%Y-%m-%d %H:%M:%S')


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


def plot_radiosonde_t_ff_dd(df_obs, df_arome, df_wrf_eth, df_icon, max_height=None):
    """Plot the radiosonde profiles of the observed and the models (temp, ff, dd)"""
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plotting temperature vs height for various datasets
    plt.plot(df_obs["temperature"].values, df_obs["height"].values, 'k', label="Obs: Radiosonde", linewidth=2)
    plt.plot(df_arome["temperature"].values, df_arome["height"].values, 'r', label="AROME")
    # plt.plot(df_wrf_eth["temperature"].values, df_wrf_eth["height"].values, 'g', label="WRF ETH", linewidth=2)
    plt.plot(df_icon["temp_C"].values, df_icon["height"].values, color='orange', label="ICON", linewidth=2)

    # Setting the axes' limits (adjust these values based on the actual range of your data)
    plt.ylim(-1, 4000)  # Modify this to fit your data's range if necessary
    plt.xlim(-5, 20)
    # Adding labels, title, legend, and grid
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Height (m)')
    plt.title('Temperature Profile by Height')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dir_PLOTS}/radiosonde_LOWI/comparison_temp_icon_wrf_arome.png")
    # Show the plot
    # plt.show()

    fig, axarr = plt.subplots(nrows=1, ncols=2)
    axlist = axarr.flatten()

    ff_obs = mpcalc.wind_speed(df_obs["u_wind"], df_obs["v_wind"])
    dd_obs = mpcalc.wind_direction(df_obs["u_wind"], df_obs["v_wind"])

    ff_arome = mpcalc.wind_speed(df_arome["u_wind"], df_arome["v_wind"])
    dd_arome = mpcalc.wind_direction(df_arome["u_wind"], df_arome["v_wind"])

    # ff_wrf_eth = mpcalc.wind_speed(df_wrf_eth["u_wind"], df_wrf_eth["v_wind"])
    # dd_wrf_eth = mpcalc.wind_direction(df_wrf_eth["u_wind"], df_wrf_eth["v_wind"])

    icon_u_wind = df_icon["u_wind"].values * units("m/s")
    icon_v_wind = df_icon["v_wind"].values * units("m/s")
    ff_icon = mpcalc.wind_speed(icon_u_wind, icon_v_wind)
    dd_icon = mpcalc.wind_direction(icon_u_wind, icon_v_wind)

    axlist[0].plot(ff_obs, df_obs['height'], 'k', label="Observation")
    axlist[0].plot(ff_arome, df_arome['height'], 'r', label="AROME")
    # axlist[0].plot(ff_wrf_eth, df_wrf_eth["height"], 'g', label="WRF_ETH")
    axlist[0].plot(ff_icon, df_icon["height"], 'orange', label="ICON")
    # axlist[0].invert_yaxis()
    # ax.set_yscale('log')
    axlist[0].set_ylim(min(df_obs['height'].values) - 1, max_height)
    axlist[0].set_ylabel("Height (m)")
    axlist[0].set_xlabel("Wind speed (m/s)")
    axlist[0].set_title("Wind Speed")
    axlist[0].legend()

    axlist[1].plot(dd_obs, df_obs['height'], 'k', label="Observation")
    axlist[1].plot(dd_arome, df_arome['height'], 'r', label="AROME")
    # axlist[1].plot(dd_wrf_eth, df_wrf_eth["height"], 'g', label="WRF_ETH")
    axlist[1].plot(dd_icon, df_icon["height"], 'orange', label="ICON")
    # axlist[1].invert_yaxis()
    # ax.set_yscale('log')
    axlist[1].set_ylim(min(df_obs['height'].values) - 1, max_height)
    axlist[1].set_ylabel("height (m)")
    axlist[1].set_xlabel("Wind Direction (degree)")
    axlist[1].set_title("Wind Direction")

    plt.suptitle(f"ICON, AROME and Radiosonde from LOWI at {time_for_model}")

    plt.savefig(f"{dir_PLOTS}/radiosonde_LOWI/comparison_icon_arome_wind.png")


if __name__ == '__main__':
    """plot the radiosonde of innsbruck and the AROME Model
    Can change time of the model on the upper part of the script: time_mean_of_launch"""
    df_obs = read_in_radiosonde_obs()

    df_arome = read_in_arome(time=time_for_model, method="sel", lat=station_files_zamg["LOWI"]["lat"],
                             lon=station_files_zamg["LOWI"]["lon"])
    # height, u_wind, v_wind,

    df_wrf_eth = read_in_wrf_eth()  # read it in, even if it is not used finally

    df_icon = read_icon_meteogram_fixed_time("LOWI", time_for_model)
    print(df_icon)

    # plot_radiosonde_t_ff_dd(df_obs, df_arome, df_wrf_eth, df_icon)
    plot_radiosonde_t_ff_dd(df_obs, df_arome, df_wrf_eth, df_icon, max_height=4000)

    plt.show()
