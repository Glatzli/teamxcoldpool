"""Script to plot the height"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy
import wrf
from cartopy import crs
from matplotlib import pyplot
from netCDF4 import Dataset
from scipy.interpolate import griddata
from wrf import getvar, to_np, get_cartopy, latlon_coords, GeoBounds, CoordPair, cartopy_xlim, cartopy_ylim

from main_code.confg import station_files_zamg

filepath = f"/media/wieser/PortableSSD/Dokumente/TEAMx/output/wrfout_cap11_d02_2017-10-16_00-00-00"


def value_600():
    # Load your dataset
    ds = Dataset(filepath)

    # Get the variable "z" at time index 6
    z1 = wrf.getvar(ds, "z", timeidx=6)
    z1 = z1.isel(bottom_top=0)
    wrf_proj = wrf.get_cartopy(z1)
    lats, lons = wrf.latlon_coords(z1)

    # Create a figure and GeoAxes with the Lambert Conformal projection
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=wrf_proj)

    # Plot the variable z1
    mesh = ax.pcolormesh(lons, lats, z1, transform=ccrs.PlateCarree(), cmap='viridis')
    ax.plot(station_files_zamg["IAO"]["lon"], station_files_zamg["IAO"]["lat"], 'bo', markersize=5,
            transform=ccrs.PlateCarree())
    # Add coastlines
    ax.coastlines()

    # Add gridlines
    ax.gridlines(draw_labels=True)
    # Add longitude and latitude labels manually

    # Add labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title('Plot of z1 on Lambert Conformal projection')
    plt.colorbar(mesh, ax=ax, label='Height (m)')
    plt.show()

    point_lon = station_files_zamg["IAO"]["lon"]
    point_lat = station_files_zamg["IAO"]["lat"]

    point_height = griddata((lons.values.flatten(), lats.values.flatten()), z1.values.flatten(), (point_lon, point_lat),
                            method='linear')
    print(point_height)


if __name__ == '__main__':
    """TODO transform point (Innsbruck to Lcc)"""
    from shapely.geometry import Point
    import pyproj

    # Define the point in WGS84 (longitude, latitude)
    lon = station_files_zamg["IAO"]["lon"]
    lat = station_files_zamg["IAO"]["lat"]
    point = Point(lon, lat)

    ds = Dataset(filepath)

    wrf_proj = pyproj.CRS(proj='lcc',  # projection type: Lambert Conformal Conic
                           lat_1=ds.TRUELAT1, lat_2=ds.TRUELAT2,  # Cone intersects with the sphere
                           lat_0=ds.MOAD_CEN_LAT, lon_0=ds.STAND_LON,  # Center point
                           a=6370000, b=6370000)  # This is it! The Earth is a perfect sphere

    # Define the WGS84 and Lambert Conformal Conic (LCC) projections
    wgs84 = pyproj.CRS("EPSG:4326")  # WGS84 CRS

    # Transform the point to Lambert Conformal Conic projection
    transformer = pyproj.Transformer.from_crs(wgs84, wrf_proj, always_xy=True)
    point_lcc = transformer.transform(lon, lat)

    # Print the transformed point
    print("Transformed point (LCC):", point_lcc)

    wrf_file = Dataset(filepath)

    # Get the terrain height
    terrain = getvar(wrf_file, "ter", timeidx=0)

    # Get the cartopy object and the lat,lon coords
    cart_proj = get_cartopy(terrain)
    lats, lons = latlon_coords(terrain)
    print(lats)

    # Create a figure and get the GetAxes object
    fig = pyplot.figure(figsize=(10, 7.5))
    geo_axes = pyplot.axes(projection=cart_proj)

    # Download and add the states and coastlines
    # See the cartopy documentation for more on this.

    # geo_axes.coastlines('50m', linewidth=0.8)

    # Set the contour levels
    levels = numpy.arange(250., 5000., 250.)

    # Make the contour lines and fill them.
    pyplot.contour(to_np(lons), to_np(lats),
                   to_np(terrain), levels=levels,
                   colors="black",
                   transform=crs.PlateCarree())
    pyplot.contourf(to_np(lons), to_np(lats),
                    to_np(terrain), levels=levels,
                    transform=crs.PlateCarree(),
                    cmap="terrain")

    # Add a color bar. The shrink often needs to be set
    # by trial and error.
    pyplot.colorbar(ax=geo_axes, shrink=.99)
    # Set up the x, y extents

    # Determine the center of the domain in grid coordinates
    slp_shape = terrain.shape

    start_y = 0
    center_y = int(slp_shape[-2] / 2.) - 1
    center_x = int(slp_shape[-1] / 2.) - 1
    end_x = int(slp_shape[-1]) - 1

    print(center_x)

    # Get the lats and lons for the start, center, and end points
    # (Normally you would just set these yourself)
    center_latlon = wrf.xy_to_ll(wrf_file,
                                 [center_x, end_x],
                                 [start_y, center_y])

    start_lat = center_latlon[0, 0]
    print(center_latlon)

    end_lat = center_latlon[0, 1]
    start_lon = center_latlon[1, 0]
    end_lon = center_latlon[1, 1]

    # Set the extents
    geo_bounds = GeoBounds(CoordPair(lat=47.2, lon=11.3),
                           CoordPair(lat=47.3, lon=11.4))
    print(geo_bounds)

    geo_axes.set_xlim(cartopy_xlim(terrain, geobounds=geo_bounds))  # projects coordinates
    geo_axes.set_ylim(cartopy_ylim(terrain, geobounds=geo_bounds))
    geo_axes.plot(station_files_zamg["IAO"]["lon"], station_files_zamg["IAO"]["lat"], 'bo', markersize=5,
                  transform=ccrs.PlateCarree())
    geo_axes.plot(point_lcc, 'ro', markersize=5)

    geo_axes.gridlines(draw_labels=True)

    pyplot.show()
