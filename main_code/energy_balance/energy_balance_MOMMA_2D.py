"""Surface energy balance: MOMMA (mobile stations) as observations, compare with 2D model output from AROME
Residual sign definition: positive if flux is directed towards the surface
"""

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from metpy.units import units

from main_code.AROME.read_in_arome import read_2D_variables_AROME
from main_code.confg import momma_our_period_file, MOMMA_stations, dir_PLOTS


def calculate_shf(ds):
    """Calculate Sensible heat flux"""
    # Constants with units
    cp = 1005 * units('J/kg/K')  # Specific heat of air at constant pressure
    R = 287 * units('J/kg/K')  # Specific gas constant for dry air

    # Convert temperatures from Celsius to Kelvin and assign units
    Ta = (ds['ta'] * units('degC')).metpy.convert_units("K")  # Air temperature
    Ts = (ds['ts1'] * units('degC')).metpy.convert_units("K")  # Surface temperature
    U = ds['wspeed'] * units('m/s')  # Wind speed
    P = (ds['pa'] * units('hPa')).metpy.convert_units('Pa')  # Convert pressure to Pascals

    # Assuming a typical value for CH over land, this could vary
    CH = 0.01  # Dimensionless

    # Calculate air density
    rho = (P / (R * Ta)).metpy.convert_units('kg/m^3')

    # Calculate sensible heat flux and assign it back to the dataset with units
    ds["H"] = - (rho * cp * CH * U * (Ta - Ts)).metpy.convert_units(
        'W/m^2')  # define it negative because it is going away

    return ds


def calculate_lhf(ds):
    """Calculate Latent heat flux"""
    # Constants with units
    Lv = 2.5e6 * units('J/kg')  # Latent heat of vaporization
    R = 287 * units('J/kg/K')  # Specific gas constant for dry air

    # Convert temperatures from Celsius to Kelvin and assign units
    Ta = (ds['ta'] * units('degC')).metpy.convert_units("K")  # Air temperature
    U = ds['wspeed'] * units('m/s')  # Wind speed
    P = (ds['pa'] * units('hPa')).metpy.convert_units('Pa')  # Convert pressure to Pascals
    RH = ds['rh'] * units('percent')  # Relative humidity

    # Calculate specific humidity of the air
    mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(relative_humidity=RH, temperature=Ta, pressure=P)

    q_a = mpcalc.specific_humidity_from_mixing_ratio(mixing_ratio)

    # Calculate saturation specific humidity at the surface temperature
    mixing_ratio_sat = mpcalc.saturation_mixing_ratio(total_press=P, temperature=Ta)

    q_s = mpcalc.specific_humidity_from_mixing_ratio(mixing_ratio_sat)

    # Assuming a typical value for CE over land, similar to CH (small hill, covered by grass...)
    CE = 0.01  # Dimensionless

    # Calculate air density
    rho = (P / (R * Ta)).metpy.convert_units('kg/m^3')

    # Calculate latent heat flux and assign it back to the dataset with units
    ds["LHF"] = - (rho * Lv * CE * U * (q_s - q_a)).metpy.convert_units(
        'W/m^2')  # Negative, as it's energy leaving the surface

    return ds


def plot_energy_balance(df_obs, name):
    """plot the surface energy balance that was observed at the MOMMAA station and the 2D AROME MODEL
    The AROME model is read in inside this function
    """

    surface_energy_balance_obs = -(
            df_obs["nr"] * units("W/m^2") + df_obs['H'] + df_obs['LHF'])  # define it also negative

    fig = go.Figure()

    # Plotting the observed components
    fig.add_trace(
        go.Scatter(x=df_obs['time'], y=df_obs['nr'], mode='lines', name='NET R Observation', marker_color='orange'))
    fig.add_trace(go.Scatter(x=df_obs['time'], y=df_obs['H'], mode='lines', name='HFS (sensible heat flux) Observation',
                             marker_color='saddlebrown'))
    fig.add_trace(go.Scatter(x=df_obs['time'], y=df_obs['LHF'], mode='lines', name='LHF (latent heat flux) Observation',
                             marker_color='royalblue'))
    fig.add_trace(go.Scatter(x=df_obs['time'], y=surface_energy_balance_obs, mode='lines',
                             name='Residual Observation',
                             marker_color='purple', line=dict(width=3)))
    # Add AROME Modell

    variable_list = ["hfs", "lfs", "lwd", "lwnet", "lwu", "swd", "swnet", "swu"]
    df_arome_2D = read_2D_variables_AROME(variableList=variable_list, lon=df_obs.lon, lat=df_obs.lat,
                                          slice_lat_lon=False)

    # Add traces for each component of radiation
    fig.add_trace(
        go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['swd'], mode='lines+markers', name='+SW (insolation) AROME',
                   marker_color='gold'))
    fig.add_trace(
        go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['swu'], mode='lines+markers', name='-SW (reflection) AROME',
                   marker_color='red'))
    fig.add_trace(
        go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['lwd'], mode='lines+markers', name='+LW (infrared) AROME',
                   marker_color='cyan'))
    fig.add_trace(
        go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['lwu'], mode='lines+markers', name='-LW (infrared) AROME',
                   marker_color='green'))

    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['hfs'], mode='lines+markers',
                             name='HFS (sensible heat flux) AROME',
                             marker_color='saddlebrown', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['lfs'], mode='lines+markers',
                             name='LFS (latent heat flux) AROME',
                             marker_color='royalblue', line=dict(dash='dash')))

    net_radiation = df_arome_2D['swd'] + df_arome_2D['lwd'] + df_arome_2D['swu'] + df_arome_2D['lwu']
    surface_energy_balance = -(net_radiation + df_arome_2D['hfs'] + df_arome_2D[
        'lfs'])  # define it also negative to show the equality for radiation (input)

    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=net_radiation, mode='lines', name='NET R AROME',
                             line=dict(color='orange', dash='dash')))
    fig.add_trace(
        go.Scatter(x=df_arome_2D['time'], y=surface_energy_balance, mode='lines', name='Residual',
                   line=dict(color='purple', width=3, dash='dash')))

    # Update layout
    fig.update_layout(title=f'Simplified Surface Energy Balance for {name}', xaxis_title='Time (UTC)',
                      yaxis_title='Energy rate (W/m²)', legend_title='Legend')

    fig.write_html(f"{dir_PLOTS}/energy_balance/energy_balance_2D_0m_vs_3D_5m_AROME/energy_balance_station_{name}.html")

    fig.show()


if __name__ == '__main__':
    # Attention station Innsbruck_Ölberg and Innsbruck_Hotel Hilton have no net radiation measurements
    """Task to calculate and plot the surface energy balance"""

    # read_EC_stations()  # read the big MOMMA file (with large time period) and cut it to our 24 hours
    for station_key in np.arange(0, 9):
        name = MOMMA_stations[str(station_key)]

        df_station = xr.open_dataset(momma_our_period_file).sel(STATION_KEY=station_key)

        df_station_with_lhf = calculate_lhf(df_station)
        df_station_with_lhf_shf = calculate_shf(df_station_with_lhf)

        plot_energy_balance(df_station_with_lhf_shf, name)

    plt.show()
