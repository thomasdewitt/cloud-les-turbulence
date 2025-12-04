# cloud-les-turbulence

## Purpose

To analyze the turbulent scaling properties in multiple LES datasets of convective atmospheres.

## Layout

### load_data.py

Contains functions for loading LES data from multiple models and simulations. Each function loads a single (configurable) variable from potentially multiple .nc files and returns a stacked array where dimensions are (nx, ny, nz, nt) and a tuple of coordinates (x, y, z).

Available simulations:

#### SAM_COMBLE

- **Base Path**: `/Volumes/BLUE/COMBLE/SHEBA_Init_dSST_6p5_grid_100m_Tlength_15hr/OUT_3D/`
- **Files**:
  - `RCELAND_640x640x110_comble_16_0000003600.nc`
  - `RCELAND_640x640x110_comble_16_0000005400.nc`
  - `RCELAND_640x640x110_comble_16_0000007200.nc`
- **Available Variables**: QV, QN (total condensate), QI, TABS, W, U
- **Dimensions**: (nx=640, ny=640, nz=110)
- **Function**: `load_SAM_COMBLE(variable, single_timestep=False)`

#### SAM_DYCOMS

- **Base Path**: `/Volumes/BLUE/DYCOMS/`
- **File Pattern**: `DYCOMS_RF01_640x640x640_dt0.25sec_320_{TIMESTEP}_{VARGROUP}.nc`
- **Timesteps**: 0000043200, 0000050400, 0000057600
- **Variable Groups**:
  - `T_QV`: Contains TABS, QV
  - `U_V`: Contains U, V
  - `W_QN`: Contains W, QN
- **Available Variables**: QV, QN (total condensate), TABS, W, U
- **Dimensions**: (nx=640, ny=640, nz=531)
- **Function**: `load_SAM_DYCOMS(variable, single_timestep=False)`

#### SAM_TWPICE

- **Base Path**: `/Volumes/BLUE/TWPICE/`
- **File Structure**: Separate directory for each variable (`OUT_3D.{VAR}/`)
- **File Pattern**: `TWPICE_LPT_3D_{VARIABLE}_{TIMESTEP}.nc`
- **Timesteps**: 0000000150, 0000001800, 0000003450
- **Available Variables**: QV, QC, QI, TABS, W, U
- **Dimensions**: (nx=2048, ny=2048, nz=255)
- **Note**: Data array has dimensions (time, y, x, z) - function handles transposition
- **Function**: `load_SAM_TWPICE(variable, single_timestep=False)`

#### SAM_RCEMIP

- **Base Path**: `/Volumes/BLUE/RCEMIP/SAM_CRM/RCE_small_les300/3D/`
- **Files**:
  - `RCEMIP_SST300_480x480x146-200m-2s_480_0001512000.nc`
  - `RCEMIP_SST300_480x480x146-200m-2s_480_0001728000.nc`
  - `RCEMIP_SST300_480x480x146-200m-2s_480_0001944000.nc`
  - `RCEMIP_SST300_480x480x146-200m-2s_480_0002160000.nc`
- **Available Variables**: QV, QN (total condensate), TABS, W, U
- **Dimensions**: (nx=480, ny=480, nz=146)
- **Function**: `load_SAM_RCEMIP(variable, single_timestep=False)`

#### CM1_RCEMIP

- **Base Path**: `/Volumes/BLUE/RCEMIP/CM1/RCE_small_les300/3D/`
- **Files**:
  - `CM1_RCE_small_les300_3D_allvars_hour0840.nc`
  - `CM1_RCE_small_les300_3D_allvars_hour0960.nc`
  - `CM1_RCE_small_les300_3D_allvars_hour1080.nc`
  - `CM1_RCE_small_les300_3D_allvars_hour1200.nc`
- **Available Variables**: clw (cloud liquid water), cli (cloud ice), ta (temperature), wa (vertical velocity), ua (eastward wind), hus (specific humidity)
- **Dimensions**: (ni=540, nj=540, nk=146)
- **Note**: Variables use different naming convention (CF compliant). clw, cli, and hus are in g/g, converted to g/kg by load function (×1000).
- **Function**: `load_CM1_RCEMIP(variable, single_timestep=False)`

#### SAM_RCEMIP_large

- **Base Path**: `/Volumes/BLUE/RCEMIP/SAM_CRM/RCE_large300/3D/`
- **Files**:
  - `SAM_CRM_RCE_large300_3D_0000660600.nc`
  - `SAM_CRM_RCE_large300_3D_0000680400.nc`
  - `SAM_CRM_RCE_large300_3D_0000700200.nc`
  - `SAM_CRM_RCE_large300_3D_0000720000.nc`
- **Available Variables**: QV, ta, ua, va, wa, clw, cli, plw, pli, hus, hur
  - QV: water vapor mixing ratio
  - ta: air temperature
  - ua, va, wa: eastward, northward, vertical wind components
  - clw, cli: cloud liquid water, cloud ice
  - plw, pli: precipitating liquid water, precipitating ice
  - hus: specific humidity
  - hur: relative humidity
- **Dimensions**: (nx=1536, ny=1536, nz=146) [3km domain]
- **Function**: `load_SAM_RCEMIP_large(variable, single_timestep=False)`

#### CM1_RCEMIP_large

- **Base Path**: `/Volumes/BLUE/RCEMIP/CM1/RCE_large300/3D/`
- **Files**:
  - `CM1_RCE_large300_3D_allvars_hour1860.nc`
  - `CM1_RCE_large300_3D_allvars_hour2040.nc`
  - `CM1_RCE_large300_3D_allvars_hour2220.nc`
  - `CM1_RCE_large300_3D_allvars_hour2400.nc`
- **Available Variables**: clw, cli, plw, pli, ta, ua, va, wa, pa, hus, hur, tntr, tntrs, tntrl
  - clw, cli: cloud liquid water, cloud ice
  - plw, pli: precipitating liquid water, precipitating ice
  - ta: air temperature
  - ua, va, wa: eastward, northward, vertical wind components
  - pa: pressure
  - hus: specific humidity
  - hur: relative humidity
  - tntr, tntrs, tntrl: total, shortwave, longwave radiative heating rates
- **Dimensions**: (ni=1620, nj=1620, nk=146) [3km domain]
- **Function**: `load_CM1_RCEMIP_large(variable, single_timestep=False)`

#### HRRR

- **Base Path**: `/Volumes/BLUE/HRRR/`
- **File Pattern**: `YYYYMMDD/tHHz/*_wrfprsf_f{00|48}_valid_*.grib2`
- **Available Variables**: t, u, v, gh, q, dpt, clwmr, rwmr, snmr, w, r
  - t: temperature
  - u, v, w: eastward, northward, vertical wind components
  - gh: geopotential height
  - q: specific humidity (water vapor)
  - dpt: dew point
  - clwmr: cloud water mixing ratio
  - rwmr: rain water mixing ratio
  - snmr: snow water mixing ratio
  - r: relative humidity
- **Dimensions**: Variable (nx, ny, nz_interpolated, nt) - interpolated to uniform height grid
- **Notes**:
  - Loads isobaric level data and interpolates to uniform height grid
  - Supports forecast hour selection (f00=analysis, f48=48-hour forecast)
  - Supports seasonal filtering ('summer'=Jun-Aug, 'fall'=Sep-Nov)
  - Water variables (q, clwmr, snmr, rwmr) converted from kg/kg to g/kg (×1000)
- **Function**: `load_HRRR(variable, single_timestep=False, load_forecast=False, season=None)`

### compute_scaling_functions.py

Computes scaling functions (structure functions and power spectral densities) for LES datasets.

#### Process

For each dataset and variable (U, W, QV, QT where QT = QV+QC+QI or QV+QN):
- Loads data within specified altitude range for each dataset
- Calculates order-2 structure functions in vertical and horizontal directions
- Calculates power spectral densities in vertical and horizontal directions
- Saves all results to `scaling_functions/scaling_functions_all_datasets.nc`

#### Altitude Ranges

- SAM_COMBLE: 2000-4000m
- SAM_DYCOMS: 650-850m
- SAM_TWPICE: 5000-10000m
- SAM_RCEMIP: 5000-10000m
- SAM_RCEMIP_large: 5000-10000m
- CM1_RCEMIP: 5000-10000m
- CM1_RCEMIP_large: 5000-10000m
- HRRR: 2000-4000m

#### Output

- NetCDF file: `scaling_functions/scaling_functions_all_datasets.nc`
- Contains groups for each dataset with structure functions and spectra for each variable

### plot_scaling_functions.py

Creates visualization plots from the scaling functions netCDF file.

#### Output

For each dataset and variable, creates a 2-panel figure:
- **Left panel**: Vertical and horizontal structure functions on log-log plot with fitted Hurst exponents
- **Right panel**: Vertical and horizontal power spectral densities on log-log plot with reference slopes

Includes reference lines with theoretical slopes:
- General variables: horizontal spectral slope -5/3, vertical spectral slope -11/5
- W variable: horizontal spectral slope -7/9, vertical spectral slope -3/5

Figures saved to `scaling_functions/{DATASET}_{VARIABLE}_scaling.png`

### plot_profiles.py

Generates vertical profile visualizations and statistics for LES datasets.

#### Process

For each dataset and variable:
- Plots vertical profiles showing mean and ±1σ spatial variability
- Calculates profile statistics:
  - `mean_profile`: mean over all x, y, t dimensions
  - `normalized_log_std`: standard deviation of ln(data/mean_profile) over all x, y, t

#### Output

- **PNG files**: `profiles/{DATASET}_t{TIMESTEP}_{VARIABLE}.png`
- **NetCDF files**: `profiles/{DATASET}_profiles.nc` containing profile statistics

## Usage

### Basic Data Loading

```python
from load_data import load_SAM_DYCOMS, load_HRRR

# Load all timesteps
data, (x, y, z) = load_SAM_DYCOMS('QV', single_timestep=False)
# Shape: (nx, ny, nz, nt)

# Load only first timestep
data, (x, y, z) = load_SAM_DYCOMS('QV', single_timestep=True)
# Shape: (nx, ny, nz, 1)

# Load HRRR with options
data, (x, y, z) = load_HRRR('q', single_timestep=True,
                             load_forecast=True, season='summer')
```

### Generate Scaling Function Analysis

```bash
# Compute structure functions and spectra
python compute_scaling_functions.py

# Plot the results
python plot_scaling_functions.py
```

### Generate Vertical Profiles

```bash
python plot_profiles.py
```

This will generate vertical profile plots and netCDF statistics for all configured datasets.

## Notes

- All functions normalize data to shape (nx, ny, nz, nt) regardless of original netCDF ordering
- Coordinate arrays (x, y, z) are returned as float32 for consistency
- Data arrays are converted to float32 for memory efficiency
- Grid spacing (dx) is calculated from coordinate arrays as `dx = x[1] - x[0]`
- Some datasets have combined condensate variables (QN) rather than separate liquid (QC) and ice (QI)
- **Unit normalization**: Water/ice variables are normalized to g/kg across all datasets:
  - SAM datasets: QV, QC, QI, QN in g/kg (no conversion)
  - CM1 datasets: clw, cli, hus in g/g → converted to g/kg (×1000) by load functions
  - HRRR: q, clwmr, rwmr, snmr in kg/kg → converted to g/kg (×1000) by load function
- **QT variable**: Total moisture calculated as sum of water vapor + cloud water + cloud ice
  - SAM_COMBLE, SAM_TWPICE: QT = QV + QC + QI (or QV + QN + QI)
  - SAM_DYCOMS, SAM_RCEMIP: QT = QV + QN
  - CM1: QT = hus + clw + cli
  - HRRR: QT = q + clwmr (optionally + snmr)

## Requirements

- numpy
- netCDF4
- xarray (for HRRR GRIB2 files)
- cfgrib (for HRRR GRIB2 files)
- scipy (for HRRR interpolation)
- matplotlib (for plotting scripts)
- scaleinvariance (for compute_scaling_functions.py)
