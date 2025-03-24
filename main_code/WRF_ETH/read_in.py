# https://wrf-python.readthedocs.io/en/latest/basic_usage.html
# Import the tools we are going to need today:
# https://fabienmaussion.info/2018/01/06/wrf-projection/
# Projection problems with WRF output
import cartopy
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature
import numpy as np
import salem
import wrf
import xarray as xr  # netCDF library
from cartopy import crs
from matplotlib import pyplot as plt
from metpy.units import units
from netCDF4 import Dataset
from salem import mercator_grid, Map
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, CoordPair, ll_to_xy)

from main_code.confg import station_files_zamg


def plot_arome_in_crs(extent):
    """The AROME has a lat lon grid (PlateCarree) plot it as LambertConformal()"""
    ds = xr.open_dataset(
        '/Data/AROME/AROME_TEAMx_CAP_2D_fields/AROME_Geosphere_20171015T1200Z_CAP02_2D_30min_1km_best_hgt.nc')

    ds = ds.isel(time=0)
    # print(ds["hgt"].values)

    z = ds.hgt
    # extent = [-4.25, 7.5, 42.25, 51]
    # central_lon = np.mean(extent[:2])
    # central_lat = np.mean(extent[2:])
    # print(central_lat)

    ax = plt.axes(projection=ccrs.PlateCarree())

    resol = '50m'  # use data at this scale
    bodr = cartopy.feature.NaturalEarthFeature(category='cultural',
                                               name='admin_0_boundary_lines_land', scale=resol, facecolor='none',
                                               alpha=0.7)
    land = cartopy.feature.NaturalEarthFeature('physical', 'land',
                                               scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
    ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean',
                                                scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes',
                                                scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
    rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines',
                                                 scale=resol, edgecolor='b', facecolor='none')

    # ax.add_feature(land, facecolor='beige')
    ax.add_feature(ocean, linewidth=0.2)
    ax.add_feature(lakes)
    ax.add_feature(rivers, linewidth=0.5)
    ax.add_feature(bodr, linestyle='--', edgecolor='k')
    z.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=0, cmap='terrain')

    xl = ax.gridlines(draw_labels=True)
    xl.top_labels = False
    xl.right_labels = False
    # We set the extent of the map

    ax.set_extent(extent, crs=ccrs.PlateCarree())


def plot_cartopy_nc(filepath):
    """Use cartopy to plot it"""
    ds = xr.open_dataset(filepath)
    ds = ds.isel(Time=0)

    ds['XLAT'] = ds['XLAT'].isel(west_east=0).values
    ds['XLONG'] = ds['XLONG'].isel(south_north=0).values

    z = ds.HGT

    extent = [ds.XLONG.min(), ds.XLONG.max(), ds.XLAT.min(), ds.XLAT.max()]
    print(ds.XLONG.min())
    print(ds.XLONG.max())

    # plot_arome_in_crs(extent)
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6370000, semiminor_axis=6370000)

    lcc = ccrs.LambertConformal(globe=globe, central_latitude=46.847816, central_longitude=10.808777,
                                standard_parallels=(46.4, 47.2))
    ax = plt.axes(projection=lcc)
    z.plot(ax=ax, transform=lcc, cmap='terrain')
    ax.coastlines()

    ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
    # ax.set_extent([xx.min(), xx.max(), yy.min(), yy.max()], crs=lcc)

    plt.show()


def plot_wrf_nc(filepath):
    """use wrf to plot wrf_output, does work, but does not plot labels"""

    # Open the NetCDF file
    ncfile = Dataset(filepath)

    z = getvar(ncfile, "HGT")  # Terrain height m

    """This works to detect the x and y index with a latitude and longitude"""
    x_y = ll_to_xy(wrfin=ncfile, latitude=station_files_zamg["IAO"]["lat"], longitude=station_files_zamg["IAO"]["lon"])

    p1 = getvar(ncfile, "pressure", timeidx=-1)

    p = p1[:, x_y[0], x_y[1]] * units.hPa
    p = p1[:, x_y[0], x_y[1]] * units.hPa
    print(p)
    exit()
    print(station_files_zamg["IAO"]["lat"])

    # z = getvar(ncfile, "z")  # model height m

    lats, lons = latlon_coords(z)

    cart_proj = get_cartopy(z)

    print(cart_proj)

    # Create a figure
    fig = plt.figure(figsize=(12, 6))
    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=cart_proj)

    # Download and add the states and coastlines

    ax.coastlines('50m', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
    # Make the contour outlines and filled contours for the smoothed sea level
    # pressure.
    # plt.contour(to_np(lons), to_np(lats), to_np(z), 10, colors="black",
    #            transform=crs.PlateCarree())
    plt.contourf(to_np(lons), to_np(lats), to_np(z),
                 transform=crs.PlateCarree(),
                 cmap='terrain')

    # Add a color bar
    plt.colorbar(ax=ax, shrink=.98, label='Height')

    # Set the map bounds
    ax.set_xlim(cartopy_xlim(z))
    ax.set_ylim(cartopy_ylim(z))
    ax.scatter(x, y, marker='o', color='red', label='Station IAO')

    # Add the gridlines
    ax.gridlines(color="black", linestyle="dotted")
    xl = ax.gridlines(draw_labels=True)
    xl.top_labels = False
    xl.right_labels = False
    plt.title("Sea Level Pressure (hPa)")
    plt.show()
    # plt.savefig("HGT_wrf_output.png")


if __name__ == '__main__':
    filepath = f"/media/wieser/PortableSSD/Dokumente/TEAMx/output/wrfout_cap11_d02_2017-10-16_00-00-00"

    # plot_cartopy_nc()
    ds = salem.open_wrf_dataset(filepath)

    ds_fixed = ds.isel(time=0)
    ds_fixed = ds_fixed.isel(bottom_top=0)

    model_height = ds_fixed.Z.salem.subset(corners=(
        (station_files_zamg["IAO"]["lon"] - 0.1, station_files_zamg["IAO"]["lat"] - 0.1),
        (station_files_zamg["IAO"]["lon"] + 0.1, station_files_zamg["IAO"]["lat"] + 0.1)), crs=salem.wgs84)

    smap = model_height.salem.get_map(data=model_height, cmap='RdYlBu_r')

    print(model_height)

    wrfin = Dataset(filepath)
    x_y = wrf.ll_to_xy(wrfin=wrfin,
                       latitude=station_files_zamg["IAO"]["lat"],
                       longitude=station_files_zamg["IAO"]["lon"])

    z1 = wrf.getvar(wrfin, "z", timeidx=6)
    z = z1[:, x_y[0], x_y[1]] * units('meter')
    print(z.isel(bottom_top=0))


    smap.set_points(station_files_zamg["IAO"]["lon"], station_files_zamg["IAO"]["lat"])

    smap.set_text(station_files_zamg["IAO"]["lon"] + 0.001, station_files_zamg["IAO"]["lat"] + 0.001, 'IAO',
                  fontsize=12)

    smap.set_points(station_files_zamg["LOWI"]["lon"], station_files_zamg["LOWI"]["lat"])
    smap.set_text(station_files_zamg["LOWI"]["lon"] - 0.02, station_files_zamg["LOWI"]["lat"] + 0.001, 'LOWI',
                  fontsize=12)

    smap.visualize()
    # t2_sub.salem.quick_map()
    plt.show()

    grid = mercator_grid(center_ll=(station_files_zamg["IAO"]["lon"], station_files_zamg["IAO"]["lat"]),
                         extent=(0.5e5, 0.5e5))

    smap = Map(grid, nx=500)

    smap.visualize(addcbar=False)
    smap.set_data(ds_fixed.Z)
    smap.set_points(station_files_zamg["IAO"]["lon"], station_files_zamg["IAO"]["lat"])

    smap.visualize()
    # ds_fixed.Z.salem.quick_map(cmap='terrain')
    # plt.savefig("HGT_salem_output.png")
    plt.show()


    # plot_wrf_nc(filepath)

    # try to select a horizontal plot
    ncfile = Dataset(filepath)
    z = getvar(ncfile, "HGT", timeidx=wrf.ALL_TIMES)  # Terrain height m
    print(z)
    exit()

    z = getvar(ncfile, "z")
    wspd = getvar(ncfile, "uvmet_wspd_wdir", units="kt")[0, :]

    wspd = wspd.isel(bottom_top=slice(0, 30))
    z = z.isel(bottom_top=slice(0, 30))

    # Define the cross section start and end points
    start_point = CoordPair(lat=station_files_zamg["IAO"]["lat"], lon=station_files_zamg["IAO"]["lon"])
    end_point = CoordPair(lat=station_files_zamg["JEN"]["lat"], lon=station_files_zamg["KUF"]["lon"])
    wspd_cross = vertcross(wspd, z, wrfin=ncfile, start_point=start_point,
                           end_point=end_point, latlon=True, meta=True)

    # Create the figure
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes()

    # Make the contour plot
    wspd_contours = ax.contourf(to_np(wspd_cross), cmap="jet")

    # Add the color bar
    plt.colorbar(wspd_contours, ax=ax)

    # Set the x-ticks to use latitude and longitude labels.
    coord_pairs = to_np(wspd_cross.coords["xy_loc"])
    x_ticks = np.arange(coord_pairs.shape[0])
    x_labels = [pair.latlon_str(fmt="{:.2f}, {:.2f}")
                for pair in to_np(coord_pairs)]
    ax.set_xticks(x_ticks[::20])
    ax.set_xticklabels(x_labels[::20], rotation=45, fontsize=8)

    # Set the y-ticks to be height.
    vert_vals = to_np(wspd_cross.coords["vertical"])
    v_ticks = np.arange(vert_vals.shape[0])
    ax.set_yticks(v_ticks[::20])
    ax.set_yticklabels(vert_vals[::20], fontsize=8)

    # Set the x-axis and  y-axis labels
    ax.set_xlabel("Latitude, Longitude", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)

    plt.title("Vertical Cross Section of Wind Speed (kt)")

    plt.show()
