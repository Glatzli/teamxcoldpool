"""Script to cut all ibox observations to our period"""
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import pandas as pd
from metpy.units import units

from main_code.confg import dir_PLOTS, ibox_folder
from main_code.wind.wind_timeseries import plot_winds


def cut_ibox(name):
    """Cut Ibox observations to our period, attention year 2017 has 2 parts (they are not equally cut!)"""
    if name == "vf0":
        ibox_infile = f"{ibox_folder}/{name}_15min/proc_{name}_blockavg_15min_2017part2.csv"
    elif name == "nf27":
        ibox_infile = f"{ibox_folder}/{name}_15min/proc_{name}_blockavg_15min_2017.csv"
    else:
        ibox_infile = f"{ibox_folder}/{name}_15min/proc_{name}_blockavg_15min_2017part1.csv"
    final_csv = pd.read_csv(ibox_infile)
    df = final_csv.drop(0)
    df = df.rename(columns={'Date/Time': 'Datetime'})
    print(df)

    # Convert 'Datetime' column to datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d/%m/%Y %H:%M:%S")

    # Set 'Datetime' column as the index
    df.set_index('Datetime', inplace=True)
    start_date = '2017-10-15 14:00:00'
    end_date = '2017-10-16 12:00:00'
    df = df[start_date:end_date]

    df.to_csv(f"/home/wieser/Dokumente/Teamx/teamxcoldpool/Data/Observations/Ibox/{name}.csv")


def plot_1min_data():
    """can use also the 1min wind data from `https://acinn-data.uibk.ac.at/station/101/WINDSONIC4/`"""
    df2 = pd.read_csv(
        "/Data/Observations/Ibox/acinn_data_i-Box Kolsass_SONIC2_8164e376/data.csv",
        sep=";", comment="#")

    wspeed2 = df2["wind_speed_1"].values * units("m/s")
    wdir2 = df2["avg_wdir1"].values * units("m/s")

    plot_winds(dates=df2["rawdate"], ws=wspeed2, wd=wdir2, subplot_index=2, name_station=f"{name}")

    plt.savefig(f"{dir_PLOTS}/wind/Difference_1min_15min_{name}.png")


if __name__ == '__main__':
    name = "nf27"
    # cut_ibox(name=name) # I have already cutted the ibox and saved the new files onto the same names

    df = pd.read_csv(f"{ibox_folder}/{name}.csv")

    u = df["mean_u1"].values * units("m/s")  # is the streamwise
    v = df["mean_v1"].values * units("m/s")  # spanwise (should always be 0) before the coordinate rotation

    if v.sum() != 0:
        print("Is the spanwise component should be 0 m/s")

    # wdir = mpcalc.wind_direction(u, v, convention='from')
    wdir = df["wdir1"] * units("degree")
    wspeed = mpcalc.wind_speed(u, v)

    plt.figure(figsize=(16, 10))

    plot_winds(dates=df["Datetime"], ws=wspeed, wd=wdir, subplot_index=1, name_station=f"{name}")

    plt.show()
