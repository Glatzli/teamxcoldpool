"""Plot specific humidity with height compare it to the radiosonde of IBK
can not calculate the relative humidity of Hatpro, because hatrpo does not measure pressure
"""
from datetime import datetime

from matplotlib import pyplot as plt
from metpy.calc import specific_humidity_from_dewpoint

from main_code.ICON_model.read_icon_model_meteogram import read_icon_meteogram_fixed_time
from main_code.UKMO_model.read_ukmo import read_ukmo_fixed_point_and_time
from main_code.WRF_Helen.read_wrf_helen import read_wrf_fixed_point_and_time
from main_code.confg import station_files_zamg, colordict, dir_PLOTS
from main_code.lidar.compare_radiosonde_hatpro_hobos import read_arome, change_height, model_time
from main_code.radiosonde.compare_radiosonde_pressure_y import read_in_radiosonde_obs


def plot_sh_with_height(model_time):
    """create the plot of the specific humidity with height"""
    plt.figure()
    plt.plot(df_arome_fixed_time.q * 1000, df_arome_fixed_time["height"], label="MODEL: Arome",
             color=colordict["AROME"])
    plt.plot(df_icon_sh.sh, df_icon_sh.height, label="MODEL: Icon", color=colordict["ICON"])
    plt.plot(df_ukmo["specific_humidity"].values, df_ukmo["geopotential_height"].values, label="MODEL: UKMO",
             color=colordict["UKMO"])
    plt.plot(df_wrf_helen["specific_humidity"].values, df_wrf_helen["z"].values, label="MODEL: WRF ACINN",
             color=colordict["WRF_ACINN"])
    plt.plot(df_radiosonde["sh"].values, df_radiosonde["height"].values, label="OBS: Radiosonde",
             color=colordict["RADIOSONDE"])

    plt.ylim([0, 10000])

    if '.' in model_time:
        base_time, fractional = model_time.split('.')
        # Cut off the fractional part after six digits
        fractional = fractional[:6]
        model_time = f"{base_time}.{fractional}"

    # Now parse the datetime string, which has been adjusted to include only six digits in fractional seconds
    parsed_time = datetime.strptime(model_time, "%Y-%m-%dT%H:%M:%S.%f")

    # Format the datetime to include only day and hour
    formatted_time = parsed_time.strftime("%Y.%m.%d at %H:%M")

    plt.title(f"{my_location}: Specific humidity at {formatted_time}")
    plt.xlabel("Specific humidity [g/kg]")
    plt.ylabel("height [m]")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{dir_PLOTS}/humidity/comparison_specific_hum_{model_time}.png",
        dpi=300,
        bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    my_location = "LOWI"
    # ICON Model
    df_icon_sh = read_icon_meteogram_fixed_time(my_location, time=model_time)

    df_ukmo = read_ukmo_fixed_point_and_time(city_name=my_location, time=model_time)

    dt = datetime.fromisoformat(model_time.replace('T', ' '))  # extract day, hour, min from model_time
    df_wrf_helen = read_wrf_fixed_point_and_time(day=dt.day, hour=dt.hour,
                                                 latitude=station_files_zamg[my_location]["lat"],
                                                 longitude=station_files_zamg[my_location]["lon"], minute=dt.minute)

    # AROME Model
    df_arome = read_arome(location=my_location)
    df_arome_fixed_time = change_height(df_arome.sel(time=model_time))

    # read radiosonde
    df_radiosonde = read_in_radiosonde_obs()
    df_radiosonde["sh"] = specific_humidity_from_dewpoint(df_radiosonde["pressure"], df_radiosonde["dewpoint"]) * 1000

    plot_sh_with_height(model_time)
