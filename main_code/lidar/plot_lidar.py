""" Script to plot the LIDAR observations and the models to get a time height plot of the wind, add also the
potential temperature for models

Hint: only plot one model per time (faster)
"""
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from metpy.units import units

from main_code.AROME.read_in_arome import read_3D_variables_AROME
from main_code.ICON_model.read_icon_model_3D import find_min_index, read_icon_fixed_point
from main_code.UKMO_model.read_ukmo import get_ukmo_height_of_specific_lat_lon, \
    get_rotated_index_of_lat_lon
from main_code.WRF_Helen.read_wrf_helen import read_wrf_fixed_point
from main_code.confg import lidar_obs_folder, icon_folder_3D, ukmo_folder
from main_code.lidar.customized_barbs import barbs
from main_code.lidar.read_in_lidar import read_lidar_obs


def add_contour_pot_temp(ds_interpolated, ax):
    """adds potential temperature contours"""
    ds_interpolated["th"] = ds_interpolated["th"] * units.kelvin
    pot_min = ds_interpolated["th"].min()
    pot_max = ds_interpolated["th"].max()
    levels = np.arange(pot_min, pot_max, 1 * units.kelvin)
    levels = np.round(levels, 0)

    contour = ds_interpolated["th"].plot.contour(x='time', y='height', colors='darkgreen',
                                                 levels=levels, extend="max", ax=ax)
    ax.clabel(contour, levels[1::2], inline=True, fontsize=10)  # # label every second level
    return contour


def plot_time_height(df, lidar_station, quiver=True, model_name=None):
    """plot the horizontal windspeed"""
    if model_name is None:
        title = f"OBS LIDAR {lidar_station} [{round(df.attrs['longitude'], 3)}E, {round(df.attrs['latitude'], 3)}N, {df.attrs['altitude']}m]: " \
                f"Interpolated wind speed over time "
        save_fig_path = f"../../Plots/lidar/lidar_obs_interp_quiver_{lidar_station}.png"
        if not quiver:
            save_fig_path = f"../../Plots/lidar/lidar_obs_interp_{lidar_station}.png"

    else:
        title = f"MODEL {model_name}: wind (time-height plot) at station {lidar_station} - {round(np.round(df.z.mean(dim='time').values.min()), 2)}m"
        save_fig_path = f"../../Plots/lidar/{model_name}_{lidar_station}.png"
        if not quiver:
            save_fig_path = f"../../Plots/lidar/{model_name}_{lidar_station}_th.png"

    colorbar_kwargs = {'label': 'horizontal wind speed (m s$^{-1}$)',
                       'pad': 0.01,  # fraction between colorbar and plot (default: 0.05)
                       }

    start_x = np.datetime64('2017-10-15 14:00')
    end_x = np.datetime64('2017-10-16 12:00')
    ylim = [0, 1600]
    ax_kwargs = {'xlim': (start_x, end_x),
                 'ylim': ylim}

    # set target grid points for quiver/barb plot
    target_time = pd.date_range(ax_kwargs['xlim'][0], ax_kwargs['xlim'][1], freq='10min')
    target_heights = np.arange(ax_kwargs['ylim'][0], ax_kwargs['ylim'][1] + 100, 10)

    fig, ax = plt.subplots(figsize=(20, 8))

    if model_name == "ICON":
        # need to chunk it to ensure time is in one chunk (data size too large)
        df_model_rechunked = df.chunk({'time': -1})

        # Now perform the interpolation
        ds_interpolated = df_model_rechunked.interpolate_na(dim='time')
    else:
        ds_interpolated = df.interpolate_na(dim='time')

    # add wind speed
    levels = np.arange(0, 17, 1)
    ds_interpolated["ff"].plot.contourf(x='time', y='height', cbar_kwargs=colorbar_kwargs, cmap="magma_r",
                                        levels=levels, extend="max", ax=ax)

    # add pot temp
    if model_name is not None:
        t_x = add_contour_pot_temp(ds_interpolated, ax)
        h1, _ = t_x.legend_elements()
        ax.legend([h1[0]], ['Potential Temperature from Model in K'], loc='upper right')

    plt.title(title)
    plt.ylim(600, 1500)

    pu = df.u.sel(time=target_time, height=target_heights, method='nearest')
    pv = df.v.sel(time=target_time, height=target_heights, method='nearest')

    if quiver:
        qv = ax.quiver(pu.time, pu.height, pu.T, pv.T, pivot='middle', linewidth=0.5,
                       headlength=2, headaxislength=2, headwidth=2, width=0.0025)
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')

    else:
        # convert m/s to knots and plot barb plot
        pu_kn = pu.metpy.convert_units("knots")
        pv_kn = pv.metpy.convert_units("knots")
        # adapted barbs method in customized_barbs.py
        qv = barbs(ax, pu_kn.time, pv_kn.height, pu_kn.T, pv_kn.T, pivot='middle', rounding=True,
                   fill_empty=True, sizes=dict(emptybarb=0.025), length=4, lw=.5)
        # plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')


# Read and prepare models especially for the LIDAR plots
def read_arome_lidar(lat, lon, method):
    """read in AROME model for Lidar"""
    my_variable_list = ["th", "u", "v", "w", "z"]

    df_model = read_3D_variables_AROME(variables=my_variable_list, method=method, lon=lon,
                                       lat=lat)  # if time = None it selects the whole time_range

    # specific read in
    df_model["ff"] = mpcalc.wind_speed(df_model["u"], df_model["v"])
    df_model["dd"] = mpcalc.wind_direction(df_model["u"], df_model["v"], convention="from")

    df_model["nz"] = df_model["z"].mean(dim='time').values
    return df_model.rename({'nz': 'height'})


def read_icon_lidar(lon, lat, day=16):
    """prepare ICON MODEL for lidar plot"""

    # open a an arbitrary ICON file
    ds_icon = xr.open_dataset(
        f"{icon_folder_3D}/ICON_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{str(day)}T{12:02d}0000Z.nc")

    # determine the nearest grid cell index
    nearest_grid_cell = find_min_index(ds_icon, my_lon=lon, my_lat=lat)

    # read in the nearest grid cell, and combine over time
    ds = read_icon_fixed_point(day=day, nearest_grid_cell=nearest_grid_cell)
    height_exclude_first = ds["z_ifc"].values[:, 1:]
    # Create the corrected DataArray
    new_height = xr.DataArray(height_exclude_first, coords={'time': ds['time'], 'height': ds['height']},
                              dims=['time', 'height'])

    # Add the corrected variable to the dataset
    ds['z'] = new_height * units("m")
    ds = ds.drop_vars("z_ifc")
    df_model = ds.drop_vars("height_3")

    df_model['u'].metpy.convert_units('m/s')
    df_model['v'].metpy.convert_units('m/s')
    df_model['z'].metpy.convert_units('m')

    df_model['u'] = df_model['u'].metpy.quantify()
    df_model['v'] = df_model['v'].metpy.quantify()

    ff_list = []
    dd_list = []

    # Loop over each time index (has something to do with the read in function, is a dask array)
    for t in range(len(df_model.time)):
        # Select the data at the current time index
        u_current = df_model['u'].isel(time=t)
        v_current = df_model['v'].isel(time=t)

        # Calculate wind speed and direction
        ff_current = mpcalc.wind_speed(u_current, v_current)
        dd_current = mpcalc.wind_direction(u_current, v_current, convention='from')

        # Append to list
        ff_list.append(ff_current)
        dd_list.append(dd_current)

    # Concatenate lists into new DataArrays and assign to df_model
    df_model['ff'] = xr.concat(ff_list, dim='time')
    df_model['dd'] = xr.concat(dd_list, dim='time')

    df_model['pressure'] = df_model['pres'] / 100.0
    p = df_model['pressure'].values * units.hPa

    # Calculate temperature from potential temperature
    temp_C = df_model["temp"] - 273.15
    df_model["temperature"] = temp_C
    temp_C = temp_C * units("degC")

    df_model["th"] = mpcalc.potential_temperature(p, temp_C)  # calc potential temperature

    df_model["nz"] = df_model["z"].mean(dim='time').values  # hier habe ich nz

    df_model['z'] = df_model['z'].metpy.dequantify()
    df_model['ff'] = df_model['ff'].metpy.dequantify()
    df_model['dd'] = df_model['dd'].metpy.dequantify()
    if 'height' in df_model.variables:  # height are just the levels, want to have the values of nz
        df_model = df_model.drop_vars('height')

    return df_model.rename({'nz': 'height'})


def read_ukmo_lidar(my_lat, my_lon):
    """Read in UKMO model data with specified latitude and longitude for lidar data extraction. """
    xi, yi = get_rotated_index_of_lat_lon(latitude=my_lat, longitude=my_lon)

    # Variables to be read from the files
    variables = ["w", "z", "th", "q", "p"]  # "u", "v" were read in below separately
    datasets = []

    for var in variables:
        # Open the dataset for each variable
        ds = xr.open_dataset(f"{ukmo_folder}/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_{var}.nc")

        # Select the data within the time range and the specific grid points
        dat = ds.sel(time=slice("2017-10-15T14:00:00", "2017-10-16T12:00:00.000000000"))
        data_final = dat.isel(grid_latitude=yi, grid_longitude=xi)

        # Collect all data into a list for merging
        datasets.append(data_final)

    # Merge all datasets into a single xarray.Dataset
    combined_ds = xr.merge(datasets)
    combined_ds["model_level_number"] = combined_ds['level_height'].values
    combined_ds = combined_ds.drop_vars('level_height')
    combined_ds = combined_ds.rename({'model_level_number': 'level_height'})

    u_data = xr.open_dataset(f"{ukmo_folder}/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_u.nc").sel(
        time=slice("2017-10-15T14:00:00", "2017-10-16T12:00:00.000000000")).isel(grid_latitude=yi, grid_longitude=xi)

    u_data["model_level_number"] = u_data['level_height'].values
    u_data = u_data.drop_vars('level_height')
    u_data = u_data.rename({'model_level_number': 'level_height'})

    v_data = xr.open_dataset(f"{ukmo_folder}/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_v.nc").sel(
        time=slice("2017-10-15T14:00:00", "2017-10-16T12:00:00.000000000")).isel(grid_latitude=yi, grid_longitude=xi)

    v_data["model_level_number"] = v_data['level_height'].values
    v_data = v_data.drop_vars('level_height')
    v_data = v_data.rename({'model_level_number': 'level_height'})

    target_levels = combined_ds['level_height'].values

    u_interpolated = u_data.interp(level_height=target_levels, method='linear')
    v_interpolated = v_data.interp(level_height=target_levels, method='linear')

    combined_ds = xr.merge([combined_ds, u_interpolated, v_interpolated], compat='override')

    station_altitude = get_ukmo_height_of_specific_lat_lon(my_lon=my_lon, my_lat=my_lat)
    combined_ds["level_height"] = np.add(combined_ds["level_height"], station_altitude)
    combined_ds = combined_ds.rename({'level_height': 'height'})

    combined_ds = combined_ds.drop_vars(
        ["rotated_latitude_longitude", "latitude_longitude", "level_height_bnds", "sigma_bnds"])

    combined_ds = combined_ds.rename({
        'upward_air_velocity': 'w',
        'geopotential_height': 'z',
        'air_potential_temperature': 'th',
        'specific_humidity': 'q',
        'air_pressure': 'p',
        'transformed_x_wind': 'u',
        'transformed_y_wind': 'v'
    })

    combined_ds['w'] = combined_ds['w'].metpy.convert_units('m/s')
    combined_ds['z'] = combined_ds['z'].metpy.convert_units('meters')
    combined_ds['th'] = combined_ds['th'].metpy.convert_units('K')
    combined_ds['q'] = combined_ds['q'].metpy.convert_units('kg/kg')
    combined_ds['p'] = combined_ds['p'].metpy.convert_units('Pa')
    combined_ds['u'] = combined_ds['u'].metpy.convert_units('m/s')
    combined_ds['v'] = combined_ds['v'].metpy.convert_units('m/s')

    wind_speed = mpcalc.wind_speed(combined_ds['u'], combined_ds['v']).metpy.quantify()
    wind_direction = mpcalc.wind_direction(combined_ds['u'], combined_ds['v']).metpy.quantify()

    # Add wind speed and direction to the dataset
    combined_ds['ff'] = wind_speed.metpy.convert_units('m/s')
    combined_ds['dd'] = wind_direction.metpy.convert_units('degrees')

    # Close all opened datasets to free resources
    for ds in datasets:
        ds.close()

    return combined_ds


def main_lidar_routine(model_name):
    """Main routine to create LIDAR plots. Creates one plot per station per model or observation

    :param model_name: The name of the model
    """
    all_lidar_names = ["SL74", "SL75", "SL88", "SLXR142"]
    all_lidar_names = ["SL74"]  # sometimes for ICON and WRF it is better to specify only one station_name (due to
    # computer capacity limitations)

    # make a for loop over all lidar stations
    for lidar_name in all_lidar_names:
        assert lidar_name in all_lidar_names, "Selected lidar name does not exist!"

        # read always the observation LIDAR (extract lat, lon)
        df_obs = read_lidar_obs(path=f'{lidar_obs_folder}/{lidar_name}_vad_l2', name=lidar_name)
        plot_time_height(df_obs, lidar_name, quiver=False)  # plot observation

        lidar_lon = df_obs.attrs["longitude"]
        lidar_lat = df_obs.attrs["latitude"]
        print(f"Selected lidar station {lidar_name} is located at: {lidar_lon, lidar_lat}")

        if model_name == "AROME":
            df_model_arome = read_arome_lidar(lat=lidar_lat, lon=lidar_lon, method="interp")
            plot_time_height(df_model_arome, lidar_name, quiver=False, model_name=f"AROME")

        elif model_name == "UKMO":

            df_ukmo = read_ukmo_lidar(my_lat=lidar_lat,
                                      my_lon=lidar_lon)  # read in UKMO MODEL
            plot_time_height(df_ukmo, lidar_name, quiver=False, model_name=f"UKMO")

        elif model_name == "WRF_ACINN":
            df_wrf = read_wrf_fixed_point(start_day=15, end_day=16, latitude=lidar_lat,
                                          longitude=lidar_lon)
            plot_time_height(df_wrf, lidar_name, quiver=False, model_name="WRF")
        elif model_name == "ICON":

            df_icon = read_icon_lidar(lon=lidar_lon, lat=lidar_lat, day=16)  # TODO until now only day 16 available
            plot_time_height(df_icon, lidar_name, quiver=False, model_name="ICON")

        else:
            print(f"No model found with the name: {model_name}! Only the observation is plotted!")


if __name__ == '__main__':
    main_lidar_routine(model_name="UKMO")

    plt.show()
