# TEAMxColdPool Project

### Observations

- Time series from 17. Oct 2015 14:00 (ignore spin up time) to 16. Oct 2017 12:00
- Observational Data Description in `./Data/Data_structure.md` (HOBOS, HATPRO, LIDAR, TAWES, IBOX, MOMMA, EC stations)

Variable name declaration:

- temperature (degC), only in Hobos it is called "ta"
- relative_humidity (%), only in Hobos it is called "rh"

### Models

Under: `https://webdata.wolke.img.univie.ac.at/TEAMx/` the models of the other groups can be found. Successfully
downloaded:

- [x] ICON (first day is missing for 3D ICON)
- [x] UKMO
- [x] AROME
- [x] WRF_ACINN
- [ ] ICON-2TE_BLM-GUF
- [ ] WRF with YSU (Yonsei University) PBL scheme

- Model Description in `./Data/Model_description.md` (AROME, UKMO, ICON, WRF_ACINN)
- WRF-ETH deprecated has the wrong time specification

## Process to reconstruct code:

- download all the necessary observations and models
- install all necessary packages, see the `requirements.txt`
- adjust names in `confg.py` (files and folders where observations/models/outputs are)
- change coordinates (set `lat, lon, time` instead of `X,Y,record`) of AROME MODEL `change_coords_of_arome_nc.py` (
  reduces filesize by a lot) (Don't forget to change `dir_2D` and `dir_3D` on top of the file)
- preparation_cutting (cut MOMMA, Ibox, TAWES and other large observations files to our time period),
  the `MOMMA_our_period.nc` file is uploaded.
- also run `model_topography.py` in AROME that creates a new DEM used in HOBOS (or use the dem that is already
  uploaded, `dem_cut_hobos.tif`)

## Code information (main_code)

In general the Models (UKMO, AROME, ICON, WRF_Helen) have their own directory, where mostly read in functions are
defined. Since all models have a different structure (single files per time step vs. one file how keeps all the
information), also the python scripts to read in are very different. In general there should be
a `read_model_fixed_point()` and a `read_model_fixed_point_and_time`, sometimes (e.g. WRF) for slice plotting you can
also find a `read_model_fixed_time()`, mostly extracting variables on the lowest level. You can find an example of a
slice lat, lon in `hobos` when plotting the quiver wind plots.

Different is AROME, here you only have three read-in functions, one per dataset (timeseries, 2D or 3D).

The Plots can be found in single
directories: `energy_balance, hobos, humidity, lidar (inside is also hatpro), radiosonde, temperature, wind`. Especially
ICON and WRF-ACINN sometimes need a lot of time to be read in, due to their data format. So it is common practice
plotting one figure after another (e.g. LIDAR).

## Next steps

- [ ] implement the new ICON model
- [ ] implement the first day of the ICON model October 15 (up to now only 16 Oct.)
- [ ] implement WRF with YSU (Yonsei University) PBL scheme

## List of all Plots:

- radiosonde
- distribution of temperature in the InnValley (2D tuned on the ground) and first model level from 3D (not tuned)
    - interactive (with plotly)
    - and static
    - and GIF of temperature
- surface energy balance
- contour plot of heights
- HATPRO and LIDAR: profiles, Time height plots
- Timeseries (of wind, temperature, specific humidity) at stations, with lowest level (so tuning is visible)
- u, v wind time series plots, with velocity above (wind_4er_plot)
- HOBOS temperature 2D differences, and time height plots

# First Model Results (just a small summary, can be deleted)

### AROME Meteogram (2m Timeseries), 3D (at 5m above ground), 2D (tuned at 0.5m)

- Temperature:
    - Generally to warm
- Specific Humidity:
    - Similar to observations
- Wind:
    - winddirection quite accurate
    - wind speed a lit bit too small
- LIDAR:
    - best

### ICON Meteogram (2m Timeseries, interpolated), 3D (5m above ground, selected the nearest grid point)

- Temperature:
    - Generally to warm, same as AROME 3D (Model level 5m)
- Specific Humidity:
    - Always to wet, probably due to clouds
- Wind:
    - winddirection seems to be off sometimes
    - wind speed similar to AROME model
- Lidar
    - October 16 appears to be poorly represented

### UKMO (Model Level 5m above ground)

- Temperature:
    - Generally to warm, but comparable to AROME 3D
- Specific Humidity:
    - Mostly to dry, but comparable to AROME Model - also specific humidity with height comparable to AROME
- Wind:
    - winddirection seems quite good, although it's the wind from the first model level (5m)
    - wind speed similar to AROME model
- LIDAR:
    - Wind at LIDAR too slow, wind direction changes only in the night (too late)

### WRF-ACINN (Meteogram, 3D)

- Temperature:
    - as bad as other models
- Specific Humidity:
    - best model, especially dewpoint is very accurate
- Wind:
    - often overestimates wind speed
- LIDAR:
    - overestimates wind maximum at 18 UTC, while none at 08 UTC of 16. Oct