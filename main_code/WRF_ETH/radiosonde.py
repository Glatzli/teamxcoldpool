"""Script to plot WRF model to compare with radiosonde"""
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import wrf
from metpy.plots import SkewT
from metpy.units import units
from netCDF4 import Dataset

from main_code.confg import station_files_zamg

if __name__ == '__main__':
    filepath = f"/media/wieser/PortableSSD/Dokumente/TEAMx/output/wrfout_cap11_d02_2017-10-16_00-00-00"
    wrfin = Dataset(filepath)

    x_y = wrf.ll_to_xy(wrfin=wrfin, longitude=station_files_zamg["LOWI"]["lon"],
                       latitude=station_files_zamg["LOWI"]["lat"])
    #  - return_val[0,...] will contain the X (west_east) values.
    #         - return_val[1,...] will contain the Y (south_north) values.

    p1 = wrf.getvar(wrfin, "pressure", timeidx=6)
    T1 = wrf.getvar(wrfin, "tc", timeidx=6)
    Td1 = wrf.getvar(wrfin, "td", timeidx=6)
    u1 = wrf.getvar(wrfin, "ua", timeidx=6)
    v1 = wrf.getvar(wrfin, "va", timeidx=6)#
    z1 = wrf.getvar(wrfin, "z", timeidx=6)

    # GELÖST: hier liegt der Fehler begraben schuae mir jetzt hoehe an (Meine Daten haben zuerst bottom_top, dann south_north steht in 346 drin
    print(p1)
    print(x_y[0])
    print(x_y[1])

    p = p1[:, x_y[1], x_y[0]] * units.hPa
    T = T1[:, x_y[1], x_y[0]] * units.degC
    Td = Td1[:, x_y[1], x_y[0]] * units.degC
    u = v1[:, x_y[1], x_y[0]] * units('m/s')
    v = u1[:, x_y[1], x_y[0]] * units('m/s')
    z = z1[:, x_y[1], x_y[0]] * units('m/s')
    print(z)

    # Example of defining your own vertical barb spacing
    skew = SkewT()

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')

    # Set spacing interval--Every 50 mb from 1000 to 100 mb
    my_interval = np.arange(100, 1000, 50) * units('mbar')

    # Get indexes of values closest to defined interval
    ix = mpcalc.resample_nn_1d(p, my_interval)

    # Plot only values nearest to defined interval values
    skew.plot_barbs(p[ix], u[ix], v[ix])

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-60, 40)
    skew.ax.set_xlabel('Temperature ($^\circ$C)')
    skew.ax.set_ylabel('Pressure (hPa)')

    plt.show()
    # plt.savefig('SkewT_Advanced.png', bbox_inches='tight')
