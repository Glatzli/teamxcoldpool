# Observations

## General Information

Observations data can be downloaded under: `https://fileshare.uibk.ac.at/d/89d6f7237ef746d0b73b/` or for
LIDAR: `https://zenodo.org/records/4674773#.YgYT7i-cbOQ`

Afterwards I subsetted large observation data to our period.

The locations of the stations are plotted in `AROME/model_topography.py`

## IBOX Observations

Code: `/home/wieser/Dokumente/Teamx/teamxcoldpool/main_code/energy_balance/energy_balance_EC.py`

- to subset data to our period use: `prepare_EC_files()` which calls `read_EC_stations()`. The in_file
  is `PIANO_EC_FluxData_QC_30min_v1-00.nc`, the out_file is `EC_30min_final`. Same for the 1min file.
- to read in new datasets use `prepare_EC_datasets()`

The radiation data can be downloaded from `https://acinn-data.uibk.ac.at/pages/station-list.html` or for the turbulent
fluxes `https://zenodo.org/records/7845952`

Stations:

- VF0 = Kolsass (Strahlung im Datensatz RAWDATA)
- SF8 = Terfens (keine Strahlung)
- SF1 = Eggen (keine Strahlung fuer diesen Zeitraum)
- NF10 = Weerberg (Strahlung im Datensatz RADIAT1)
- NF27 = Hochhaeuser (Strahlung im Datensatz RADIAT1)

VF0 is located at the almost flat valley floor. The site is surrounded by grassland and agricultural fields. (47.305°N,
11.622°E, 545 m MSL)

SF8 is located at the foot of the north sidewall next to a steep embankment between an agricultural field and a concrete
parking lot. (47.326°N, 11.652°E, 575 m MSL)

SF1 is located on an almost flat plateau running along the northern valley sidewall. The site is mainly surrounded by
grassland and agricultural fields. (47.317°N, 11.616°E, 829 m MSL)

NF10 is located on an approximately 10 deg slope on the south sidewall, covered by grassland. (47.300°N, 11.673°E, 930 m
MSL)

NF27 is located on a steep, grass-covered slope on the south sidewall, with a slope angle of about 25 deg. (47.288°N,
11.631°E, 1009 m MSL)

30 minutes averaged fluxes: `PIANO_EC_FluxData_QC_30min_v1-00.nc`

- `site_id` - Station ID (Units: none)
- `zsl` - Altitude (Units: m)
- `inst` - Instruments (Units: none)
- `zm` - Measurement height (Units: m)
- `zd` - Zero plane displacement height (Units: m)
- `z0` - Roughness length (Units: m)
- `sonicangle` - Sonic angle (Units: degree)
- `comments` - Comments (Units: none)
- `ustar` - Friction velocity (Units: m s^-1)
- `h` - Sensible heat flux (Units: W m^-2)
- `le` - Latent heat flux (Units: W m^-2)
- `fco2` - Carbon dioxide flux (Units: umol m^-2 s^-1)
- `zeta` - Stability parameter (Units: none)
- `tke` - Turbulent kinetic energy (Units: m^2 s^-2)
- `wspeed` - Wind speed (Units: m s^-1)
- `wdir` - Wind direction (Units: degree)
- `wunrot` - Unrotated vertical wind speed (Units: m s^-1)
- `sigu` - Sigma u (Units: m s^-1)
- `sigv` - Sigma v (Units: m s^-1)
- `sigw` - Sigma w (Units: m s^-1)
- `sigt` - Sigma T (Units: K)
- `epu` - Epsilon u (Units: m^2 s^-3)
- `epv` - Epsilon v (Units: m^2 s^-3)
- `epw` - Epsilon w (Units: m^2 s^-3)

With 1 minutes values: `PIANO_EC_MetData_QC_1min_v1-00.nc`

- `site_id` - Station ID (Units: none)
- `zsl` - Altitude (Units: m)
- `zm` - Measurement height (Units: m)
- `zd` - Zero plane displacement height (Units: m)
- `z0` - Roughness length (Units: m)
- `comments` - Comments (Units: none)
- `ta` - Air temperature (Units: Degrees Celsius)
- `rh` - Relative humidity (Units: %)
- `pa` - Barometric pressure (Units: hPa)
- `prec` - Precipitation amount (Units: mm)
- `ts1` - Soil temperature 1 (Units: Degrees Celsius)
- `ts2` - Soil temperature 2 (Units: Degrees Celsius)
- `ts3` - Soil temperature 3 (Units: Degrees Celsius)
- `vwc` - Volumetric water content (Units: %)
- `shf1` - Soil heat flux 1 (Units: W m^-2)
- `shf2` - Soil heat flux 2 (Units: W m^-2)
- `swin` - Incoming shortwave radiation (Units: W m^-2)
- `swout` - Outgoing shortwave radiation (Units: W m^-2)
- `lwin` - Incoming longwave radiation (Units: W m^-2)
- `lwout` - Outgoing longwave radiation (Units: W m^-2)
- `wspeed` - Wind speed (Units: m s^-1)
- `wdir` - Wind direction (Units: degree)
- `gust` - Gust speed (Units: m s^-1)

### Hatpro:

Code: `/teamxcoldpool/main_code/lidar/read_in_hatpro.py`

The `microwave radiometer` operates at two different frequency bands to retrieve vertical profiles of temperature and
humidity. It is set up to alternate between vertical measurements and scans along the axis of the Inn Valley to add more
information in the boundary layer.

General:

- Data to be used with caution only once trained with night radiosonde ascent, they are better at night
- `rawdate`: v01, v02,... v39
- `vertical levels`, see VerticalLevels_HATPRO.pdf: v01 (0m) to v39(10000m)

`data_HATPRO_humidity.csv`:

- `humidity`: unit g/m^3

`data_HATPRO_temp.csv`:

- `T`: unit K

### Hobos (hobos_final.nc):

Code: `teamxcoldpool/main_code/hobos/hobos_read_in.py`:

- Subset the original `201710_hobo.nc` to our time period and saved the file as `hobos_final.nc` (function
  code: `subset_hobos()`)
- Read in: `read_in_hobos()`

General Info:

- 50 Dataloggers around Innsbruck
- Station_KEY [ 0,1,2... 50]

Variables:

- `ta` (temperature, units C)
- `rh` (relative Humidity, units %)

### MOMMMA:

Code: until now only used for wind_plots (wind/wind_4er_plot.py) `read_in_MOMMA()` and in `energy_balance_MOMMA_2D.py`(
read xarray)

General Information:
Nine portable automatic weather station around Innsbruck:

- M02: Völs
- M03: Innsbruck/Bergisel
- M04: Patsch/Pfaffenbichl
- M05: Innsbruck/Ölberg
- M06: Innsbruck/Hotel Hilton
- M07: Innsbruck/Saggen/Kettenbrücke
- M08: Volders
- M09: Unterperfuss
- M10: Inzing/Zirl/Modellflugplatz

With the following variables:

- `time`: time
- `momaa_id`: MOMAA_ID
- `zsl`: altitude
- `lat`: latitude
- `lon`: longitude
- `comments`: comments
- `ta`: air_temperature
- `ta_raw`: air_temperature_raw
- `rh`: relative_humidity
- `rh_raw`: relative_humidity_raw
- `wdir`: wind_from_direction
- `wdir_stddev`: wind_from_direction_standard_deviation
- `nr`: net_radiation
- `pa`: air_pressure
- `rrmax`: rain_rate_max
- `prec`: precipitation_amount
- `ts1`: soil_temperature_level1
- `ts2`: soil_temperature_level2
- `vent_flag`: ventilation_flag
- `batt_volt`: battery_voltage
- `wspeed`: mean horizontal wind speed
- `wspeed_max`: max horizontal wind speed

### ZAMG Tawes:

Code :  `df = pd.read_csv(city_infos['csv'])`
Download csv data from ZAMG Datahub (https://data.hub.zamg.ac.at)

Stations used (are defined in `confg.py` `station_files_zamg`):

- Kufstein (KUF)
- Innsbruck Uni (IAO)
- Innsbruck Airport (LOWI)
- Jenbach (JEN)

### LIDAR

Code: `teamxcoldpool/main_code/lidar/read_in_lidar.py`

Measurements with four scanning Doppler wind lidars (SL74, SL75, SL88 and SLXR142) were collected during the PIANO field
campaign in the Inn Valley at Innsbruck, Austria. Three of the lidars (SL74, SL75 and SLXR142) were installed on tall
buildings and arranged on a triangle to perform coplanar scans for dual- and triple-Doppler lidar analysis. The fourth
lidar (SL88) was installed along the northern side of this triangle on a lower building to measure vertical wind
profiles. The exact locations are given in the list below. A map showing the lidar locations can be found in Haid et
al. (2020).

use only level 2 (are derived from level 1 data)

- vertical wind profiles (subfolders SL74_vad_l2, SL75_vad_l2, SL88_vad_l2, SLXR142_vad_l2)
- two-dimensional wind fields derived on a horizontal plane (subfolder ppi3_l2) and on two vertical planes (subfolders
  rhiew_l2 and rhisn_l2).
