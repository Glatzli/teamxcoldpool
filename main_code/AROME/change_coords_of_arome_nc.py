"""This script was used to convert the `variables` lat lon time (nz if 3D) to `coordinates`
Initially the coords were the indexes record, X,Y (level) first, so we reduced the filesize by a lot"""
import os.path

import xarray as xr

# TODO files are not any more available, I deleted them after I converted the variables to coordinates
# change directories
dir_2D = "/home/wieser/SeaDrive/Meine Bibliotheken/TEAMX/AROME_TEAMx_CAP_2D_fields"
dir_3D = "/home/wieser/SeaDrive/Meine Bibliotheken/TEAMX/AROME_TEAMx_CAP_3D_fields_new/AROME_TEAMx_CAP_3D_fields"

variables_2D = [
    {'name': 'hfs', 'description': 'Sensible heat flux at the surface (Wm-2)'},
    {'name': 'hgt', 'description': 'Surface geopotential height (m)'},
    {'name': 'lfs', 'description': 'Latent heat flux at the surface (Wm-2)'},
    {'name': 'lwd', 'description': 'Longwave incoming radiation at the surface (Wm-2)'},
    {'name': 'lwnet', 'description': 'Longwave net radiation at the surface (Wm-2)'},
    {'name': 'lwu', 'description': 'Longwave outgoing radiation at the surface (Wm-2) derived from lwnet - lwd'},
    {'name': 'pre', 'description': 'Surface precipitation (kgm-2) same as mm'},
    {'name': 'ps', 'description': 'Surface pressure (Pa)'},
    {'name': 'swd', 'description': 'Shortwave incoming radiation at the surface (Wm-2)'},
    {'name': 'swnet', 'description': 'Shortwave net radiation at the surface (Wm-2)'},
    {'name': 'swu', 'description': 'Shortwave reflected radiation at the surface (Wm-2) derived from swnet - swd'},
    {'name': 'tsk', 'description': 'Surface temperature (K)'}
]

variables_3D = [
    {'name': 'ciwc', 'description': 'Specific cloud ice water content (kg/kg)'},
    {'name': 'clwc', 'description': 'Specific cloud liquid water content (kg/kg)'},
    {'name': 'p', 'description': 'Pressure (Pa)'},
    {'name': 'q', 'description': 'Specific humidity (kg/kg)'},
    {'name': 'th', 'description': 'Potential temperature (K)'},
    {'name': 'tke', 'description': 'Turbulent kinetic energy (m²/s²)'},
    {'name': 'u', 'description': 'Zonal wind component (m/s)'},
    {'name': 'v', 'description': 'Meridional wind component (m/s)'},
    {'name': 'w', 'description': 'Vertical wind velocity (m/s)'},
    {'name': 'z', 'description': 'Geopotential height (m)'}
]


def save_vars_2D(var_name, var_filename):
    """Save 2D variables"""
    filepath = os.path.join(dir_2D, var_filename)
    ds = xr.open_dataset(filepath)

    latitudes = ds['latitude'].isel(record=0, X=0).values
    longitudes = ds['longitude'].isel(record=0, Y=0).values

    # Step 2: Define New Dataset with 1D Coordinates
    new_ds = xr.Dataset(
        {
            var_name: (('time', 'latitude', 'longitude'), ds[var_name].data)
        },
        coords={
            'time': ds['time'].values,
            'latitude': ('latitude', latitudes),
            'longitude': ('longitude', longitudes)
        }
    )

    # Copy attributes if necessary
    new_ds.attrs = ds.attrs

    # save xarray dataset
    output_dir = "../Data/AROME/AROME_TEAMx_CAP_2D_fields/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # This creates the directory if it doesn't exist

    output_path = os.path.join(output_dir, var_filename)

    new_ds.to_netcdf(output_path)


def save_vars_3D(var_name: str, var_filename: str):
    """Create new dimensions and coordinates of netcdf file and save it to the local folder Data

    :param var_name: str of the variable
    :param var_filename: the filename of the variable to save it
    :return:
    """
    # save xarray dataset
    filepath_original = os.path.join(dir_3D, var_filename)
    data = xr.open_dataset(filepath_original)

    # print(ds["level"])  # 0 bis 89
    nz = data["nz"].isel(record=0).values  # 1 bis 90 ist ein float
    latitudes = data['latitude'].isel(record=0, x=0).values
    longitudes = data['longitude'].isel(record=0, y=0).values

    # Step 2: Define New Dataset with 1D Coordinates
    new_3d_ds = xr.Dataset(
        {
            var_name: (('time', 'nz', 'latitude', 'longitude'), data[var_name].data)
        },
        coords={
            'time': data['time'].values,
            'latitude': ('latitude', latitudes),
            'longitude': ('longitude', longitudes),
            'nz': ('nz', nz)
        }
    )

    # Copy attributes if necessary
    new_3d_ds.attrs = data.attrs
    output_directory = "../Data/AROME/AROME_TEAMx_CAP_3D_fields/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # This creates the directory if it doesn't exist

    output_path = os.path.join(output_directory, var_filename)

    new_3d_ds.to_netcdf(output_path)


def var_2d():
    """from (record, X, Y) to (time, latitude, longitude)"""
    # If you need just the names in a list:
    variable_names_2D = [var['name'] for var in variables_2D]

    for var in variable_names_2D:
        print(f"{var} change of coords")
        save_vars_2D(var_name=var,
                     var_filename=f"AROME_Geosphere_20171015T1200Z_CAP02_2D_30min_1km_best_{var}.nc")

    print("2D variables completed")


def var_3d(var):
    """ from (record, level, x, y) to (time, nz, latitude, longitude)

    Calls the function save_vars_3D which creates a new xarray Dataset with new coordinates
    :param var: variable
    """

    save_vars_3D(var_name=var,
                 var_filename=f"AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_{var}.nc")

    print("3D variables completed")


def change_coords_dims():
    """read_lidar_obs routine for changing coords and dims of xarray Dataset """
    # for 2D variables
    var_2d()

    # for 3D variables
    variable_names_3D = [var['name'] for var in variables_3D]

    for var in variable_names_3D:
        if var != "ciwc":
            print(f"{var} change of coords")
            var_3d(var)
