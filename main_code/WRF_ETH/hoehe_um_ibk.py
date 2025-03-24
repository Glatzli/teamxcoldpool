"""Suche mir Höhe rund um Indexe (von Innsbruck) von WRF MODEL"""

import matplotlib.pyplot as plt
import numpy as np
import wrf
from netCDF4 import Dataset
from wrf import getvar, get_cartopy, latlon_coords

from main_code.confg import station_files_zamg

filepath = f"/media/wieser/PortableSSD/Dokumente/TEAMx/output/wrfout_cap11_d02_2017-10-16_00-00-00"

if __name__ == '__main__':
    # read in file
    wrfin = Dataset(filepath)
    # Get the z
    z1 = getvar(wrfin, "z", timeidx=6)
    ter = getvar(wrfin, "ter", timeidx=6)

    cart_proj = get_cartopy(z1)  # Get the cartopy object and the lat,lon coords

    lats, lons = latlon_coords(z1)
    print(lats.sel(south_north=345, west_east=360))  # könnte so Innsbruck suchen

    point_lon = station_files_zamg["IAO"]["lon"]
    point_lat = station_files_zamg["IAO"]["lat"]

    x_y = wrf.ll_to_xy(wrfin=wrfin, timeidx=6, longitude=station_files_zamg["IAO"]["lon"],
                       latitude=station_files_zamg["IAO"]["lat"])  # liefert mir exakte indexe

    print(x_y[0])  # index west_east_index
    print(x_y[1])  # index south_north

    z1 = z1.isel(bottom_top=0)
    # around innsbruck
    south_north_index = np.arange(345, 348, 1)
    west_east_index = np.arange(360, 367, 1)

    data_dict = {}
    for s in south_north_index:
        for e in west_east_index:
            lat = z1.isel(south_north=s, west_east=e)["XLAT"]
            lon = z1.isel(south_north=s, west_east=e)["XLONG"]
            h = z1.isel(south_north=s, west_east=e).values  # model height
            t = ter.isel(south_north=s, west_east=e).values  # terrain height
            data_dict[(s, e)] = {'lat': lat, 'lon': lon, 'h': h, 't': t}

    # Extract coordinates and heights from the data_dict
    coordinates = list(data_dict.keys())
    data_values = list(data_dict.values())

    # Extract s and e coordinates
    s_coordinates, e_coordinates = zip(*coordinates)

    # Create scatter plot
    plt.figure(figsize=(16, 8))
    plt.scatter(e_coordinates, s_coordinates, c=[data['h'] for data in data_values], cmap='viridis', marker='o')

    plt.colorbar(label='Terrain Height')
    plt.ylabel('South North Index (s)')
    plt.xlabel('West East Index (e)')
    plt.title(f'Rund um Innsbruck Uni ({point_lon}, {point_lat})')

    # Add text labels for each point
    for (s, e), data in zip(coordinates, data_values):
        plt.text(e, s, s=f'Lat: {data["lat"]:.2f}\nLon: {data["lon"]:.2f}\nModel Height: {data["h"]:.2f}m\nTerrain Height: {data["t"]:.2f}m', fontsize=8)

    plt.grid(True)
    # plt.savefig("/home/wieser/Dokumente/Teamx/teamxcoldpool/Plots/wrf_eth_model/Arome_first_plots/hoehe_model.png")
    plt.show()
