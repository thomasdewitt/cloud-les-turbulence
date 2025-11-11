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
- **Available Variables**: QV, QC, QI, TABS, W, U
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
  - `RCEMIP_SST300_480x480x146-200m-2s_480_0000086400.nc`
  - `RCEMIP_SST300_480x480x146-200m-2s_480_0000777600.nc`
  - `RCEMIP_SST300_480x480x146-200m-2s_480_0001468800.nc`
  - `RCEMIP_SST300_480x480x146-200m-2s_480_0002160000.nc`
- **Available Variables**: QV, QN (total condensate), TABS, W, U
- **Dimensions**: (nx=480, ny=480, nz=146)
- **Function**: `load_SAM_RCEMIP(variable, single_timestep=False)`

#### CM1_RCEMIP

- **Base Path**: `/Volumes/BLUE/RCEMIP/CM1/RCE_small_les300/3D/`
- **Files**:
  - `CM1_RCE_small_les300_3D_allvars_hour0000.nc`
  - `CM1_RCE_small_les300_3D_allvars_hour0408.nc`
  - `CM1_RCE_small_les300_3D_allvars_hour0816.nc`
  - `CM1_RCE_small_les300_3D_allvars_hour1200.nc`
- **Available Variables**: clw (cloud liquid water), cli (cloud ice), ta (temperature), wa (vertical velocity), ua (eastward wind)
- **Dimensions**: (ni=540, nj=540, nk=146)
- **Note**: Variables use different naming convention (CF compliant). clw and cli are in g/g, converted to g/kg by plotting script.
- **Function**: `load_CM1_RCEMIP(variable, single_timestep=False)`

### plot_optical_depth.py

Simple script that loads each timestep of each simulation, calculates optical depth from cloud water and ice content, and plots/saves visualizations to `optical_depth_images/` directory.

#### Calculation Details

- **Water Path**: Calculated using air density at each level, water content, and grid spacing
- **Optical Depth**: Related to water path via `LWP = 0.6292 * tau * re`, where re is effective radius
  - Liquid water: re = 10 μm
  - Cloud ice: re = 30 μm (not currently differentiated in optical depth calc)
- **Opacity**: Converted from optical depth via `opacity = 1 - exp(-tau)`
- **Visualization**: Blue-to-white colormap where blue = clear sky, white = opaque cloud

#### Output

Images are saved as `{DATASET}_{IDENTIFIER}.png` in `optical_depth_images/` directory with dimensions matching input grid.

## Usage

### Basic Loading

```python
from load_data import load_SAM_COMBLE

# Load all timesteps of QC (cloud water)
qc_data, (x, y, z) = load_SAM_COMBLE('QC', single_timestep=False)
# Shape: (640, 640, 110, 3)

# Load only first timestep
qc_single, (x, y, z) = load_SAM_COMBLE('QC', single_timestep=True)
# Shape: (640, 640, 110, 1)
```

### Generate Optical Depth Visualizations

```bash
python plot_optical_depth.py
```

This will process all five datasets and save visualizations to `optical_depth_images/`.

## Notes

- All functions normalize data to shape (nx, ny, nz, nt) regardless of original netCDF ordering
- Coordinate arrays (x, y, z) are returned as float32 for consistency
- Data arrays are converted to float32 for memory efficiency
- Grid spacing (dx) is calculated from coordinate arrays as `dx = x[1] - x[0]`
- Some datasets have combined condensate variables (QN) rather than separate liquid (QC) and ice (QI)

## Requirements

- numpy
- netCDF4
- matplotlib
- pathlib (standard library)
