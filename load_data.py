"""
Load functions for various LES datasets and HRRR atmospheric data.

Each function loads a single variable from potentially multiple files and returns:
  - data: stacked array with dimensions (nx, ny, nz, nt)
  - coords: tuple of (x, y, z) coordinate arrays
"""

import numpy as np
import netCDF4 as nc
import xarray as xr
from pathlib import Path


def load_SAM_COMBLE(variable, single_timestep=False):
    """
    Load a variable from SAM COMBLE simulation.

    Parameters
    ----------
    variable : str
        Variable name: 'QV', 'QN', 'QI', 'TABS', 'W', 'U'
        Note: QN is total non-precipitating condensate (liquid + ice)
    single_timestep : bool
        If True, return only first timestep with shape (nx, ny, nz, 1)
        If False, stack all timesteps to shape (nx, ny, nz, nt)

    Returns
    -------
    data : np.ndarray
        Data array with dimensions (nx, ny, nz, nt)
    coords : tuple
        (x, y, z) coordinate arrays
    """
    base_path = "/Volumes/BLUE/COMBLE/SHEBA_Init_dSST_6p5_grid_100m_Tlength_15hr/OUT_3D"
    files = [
        # "RCELAND_640x640x110_comble_16_0000003600.nc",
        # "RCELAND_640x640x110_comble_16_0000005400.nc",
        "RCELAND_640x640x110_comble_16_0000007200.nc",
    ]

    data_list = []
    x = None
    y = None
    z = None

    for filename in files[:1] if single_timestep else files:
        filepath = Path(base_path) / filename
        with nc.Dataset(filepath, 'r') as ds:
            # Load coordinates (same for all files)
            if x is None:
                x = ds.variables['x'][:].astype(np.float32)
                y = ds.variables['y'][:].astype(np.float32)
                z = ds.variables['z'][:].astype(np.float32)

            # Load data: shape is (time, z, y, x), need to transpose to (x, y, z, time)
            var_data = ds.variables[variable][:].astype(np.float32)  # (1, z, y, x)
            var_data = np.transpose(var_data, (3, 2, 1, 0))  # (x, y, z, 1)
            data_list.append(var_data)

    # Stack all timesteps
    data = np.concatenate(data_list, axis=3)  # (x, y, z, nt)

    return data, (x, y, z)


def load_SAM_DYCOMS(variable, single_timestep=False):
    """
    Load a variable from SAM DYCOMS simulation.

    Parameters
    ----------
    variable : str
        Variable name: 'QV', 'QN', 'TABS', 'W', 'U'
    single_timestep : bool
        If True, return only first timestep with shape (nx, ny, nz, 1)
        If False, stack all timesteps to shape (nx, ny, nz, nt)

    Returns
    -------
    data : np.ndarray
        Data array with dimensions (nx, ny, nz, nt)
    coords : tuple
        (x, y, z) coordinate arrays
    """
    base_path = "/Volumes/BLUE/DYCOMS"

    # Map variables to their files
    var_to_files = {
        'QV': ['T_QV'],
        'TABS': ['T_QV'],
        'U': ['U_V'],
        'V': ['U_V'],
        'W': ['W_QN'],
        'QN': ['W_QN'],
    }

    if variable not in var_to_files:
        raise ValueError(f"Variable {variable} not available in SAM_DYCOMS")

    file_patterns = var_to_files[variable]
    timesteps = [
        "0000043200",
        # "0000050400",
        # "0000057600",
    ]

    data_list = []
    x = None
    y = None
    z = None

    for timestep in (timesteps[:1] if single_timestep else timesteps):
        for pattern in file_patterns:
            filepath = Path(base_path) / f"DYCOMS_RF01_640x640x640_dt0.25sec_320_{timestep}_{pattern}.nc"
            with nc.Dataset(filepath, 'r') as ds:
                if variable not in ds.variables:
                    continue

                # Load coordinates
                if x is None:
                    x = ds.variables['x'][:].astype(np.float32)
                    y = ds.variables['y'][:].astype(np.float32)
                    z = ds.variables['z'][:].astype(np.float32)

                # Load data: shape is (time, z, y, x), need to transpose to (x, y, z, time)
                var_data = ds.variables[variable][:].astype(np.float32)  # (1, z, y, x)
                var_data = np.transpose(var_data, (3, 2, 1, 0))  # (x, y, z, 1)
                data_list.append(var_data)
                break  # Only load from one file per timestep

    data = np.concatenate(data_list, axis=3)

    return data, (x, y, z)


def load_SAM_TWPICE(variable, single_timestep=False):
    """
    Load a variable from SAM TWPICE simulation.

    Parameters
    ----------
    variable : str
        Variable name: 'QV', 'QC', 'QI', 'TABS', 'W', 'U'
    single_timestep : bool
        If True, return only first timestep with shape (nx, ny, nz, 1)
        If False, stack all timesteps to shape (nx, ny, nz, nt)

    Returns
    -------
    data : np.ndarray
        Data array with dimensions (nx, ny, nz, nt)
    coords : tuple
        (x, y, z) coordinate arrays
    """
    base_path = "/Volumes/BLUE/TWPICE"

    timesteps = [
        "0000000150",
        # "0000001800",
        # "0000003450",
    ]

    data_list = []
    x = None
    y = None
    z = None

    for timestep in (timesteps[:1] if single_timestep else timesteps):
        filepath = Path(base_path) / f"OUT_3D.{variable}" / f"TWPICE_LPT_3D_{variable}_{timestep}.nc"
        with nc.Dataset(filepath, 'r') as ds:
            # Load coordinates
            if x is None:
                x = ds.variables['x'][:].astype(np.float32)
                y = ds.variables['y'][:].astype(np.float32)
                z = ds.variables['z'][:].astype(np.float32)

            # Load data: shape is (time, y, x, z), need to transpose to (x, y, z, time)
            var_data = ds.variables[variable][:].astype(np.float32)  # (1, y, x, z)
            var_data = np.transpose(var_data, (2, 1, 3, 0))  # (x, y, z, 1)
            data_list.append(var_data)

    data = np.concatenate(data_list, axis=3)

    return data, (x, y, z)


def load_SAM_RCEMIP(variable, single_timestep=False):
    """
    Load a variable from SAM RCEMIP simulation.

    Parameters
    ----------
    variable : str
        Variable name: 'QV', 'QN', 'TABS', 'W', 'U'
        Note: QN represents total condensate (QC + QI)
    single_timestep : bool
        If True, return only first timestep with shape (nx, ny, nz, 1)
        If False, stack all timesteps to shape (nx, ny, nz, nt)

    Returns
    -------
    data : np.ndarray
        Data array with dimensions (nx, ny, nz, nt)
    coords : tuple
        (x, y, z) coordinate arrays
    """
    base_path = "/Volumes/BLUE/RCEMIP/SAM_CRM/RCE_small_les300/3D"
    files = [
        "RCEMIP_SST300_480x480x146-200m-2s_480_0001512000.nc",
        "RCEMIP_SST300_480x480x146-200m-2s_480_0001728000.nc",
        "RCEMIP_SST300_480x480x146-200m-2s_480_0001944000.nc",
        "RCEMIP_SST300_480x480x146-200m-2s_480_0002160000.nc",
    ]

    data_list = []
    x = None
    y = None
    z = None

    for filename in files[:1] if single_timestep else files:
        filepath = Path(base_path) / filename
        with nc.Dataset(filepath, 'r') as ds:
            # Load coordinates
            if x is None:
                x = ds.variables['x'][:].astype(np.float32)
                y = ds.variables['y'][:].astype(np.float32)
                z = ds.variables['z'][:].astype(np.float32)

            # Load data: shape is (time, z, y, x), need to transpose to (x, y, z, time)
            var_data = ds.variables[variable][:].astype(np.float32)  # (1, z, y, x)
            var_data = np.transpose(var_data, (3, 2, 1, 0))  # (x, y, z, 1)
            data_list.append(var_data)

    data = np.concatenate(data_list, axis=3)

    return data, (x, y, z)


def load_CM1_RCEMIP(variable, single_timestep=False):
    """
    Load a variable from CM1 RCEMIP simulation (small, les300).

    Parameters
    ----------
    variable : str
        Variable name: 'clw', 'cli', 'ta', 'wa', 'ua', 'hus'
        - clw: cloud liquid water
        - cli: cloud ice
        - ta: air temperature
        - wa: vertical velocity
        - ua: eastward wind
        - hus: specific humidity (water vapor)
    single_timestep : bool
        If True, return only first timestep with shape (nx, ny, nz, 1)
        If False, stack all timesteps to shape (nx, ny, nz, nt)

    Returns
    -------
    data : np.ndarray
        Data array with dimensions (nx, ny, nz, nt)
    coords : tuple
        (x, y, z) coordinate arrays
    """
    base_path = "/Volumes/BLUE/RCEMIP/CM1/RCE_small_les300/3D"
    files = [
        "CM1_RCE_small_les300_3D_allvars_hour0840.nc",
        "CM1_RCE_small_les300_3D_allvars_hour0960.nc",
        "CM1_RCE_small_les300_3D_allvars_hour1080.nc",
        "CM1_RCE_small_les300_3D_allvars_hour1200.nc",
    ]

    data_list = []
    x = None
    y = None
    z = None

    for filename in files[:1] if single_timestep else files:
        filepath = Path(base_path) / filename
        with nc.Dataset(filepath, 'r') as ds:
            # Load coordinates
            if x is None:
                x = ds.variables['x'][:].astype(np.float32)
                y = ds.variables['y'][:].astype(np.float32)
                z = ds.variables['z'][:].astype(np.float32)

            # Load data: shape is (time, nk, nj, ni), need to transpose to (ni, nj, nk, time)
            var_data = ds.variables[variable][:].astype(np.float32)  # (1, nk, nj, ni)
            var_data = np.transpose(var_data, (3, 2, 1, 0))  # (ni, nj, nk, 1)
            data_list.append(var_data)

    data = np.concatenate(data_list, axis=3)

    return data, (x, y, z)


def load_SAM_RCEMIP_large(variable, single_timestep=False):
    """
    Load a variable from SAM RCEMIP large (3km) simulation.

    Parameters
    ----------
    variable : str
        Variable name: 'QV', 'ta', 'ua', 'va', 'wa', 'clw', 'cli', 'plw', 'pli', 'hus', 'hur'
        - QV: water vapor mixing ratio
        - ta: air temperature
        - ua: eastward wind (X component)
        - va: northward wind (Y component)
        - wa: vertical velocity (Z component)
        - clw: cloud liquid water
        - cli: cloud ice
        - plw: precipitating liquid water
        - pli: precipitating ice
        - hus: specific humidity
        - hur: relative humidity
    single_timestep : bool
        If True, return only first timestep with shape (nx, ny, nz, 1)
        If False, stack all timesteps to shape (nx, ny, nz, nt)

    Returns
    -------
    data : np.ndarray
        Data array with dimensions (nx, ny, nz, nt)
    coords : tuple
        (x, y, z) coordinate arrays
    """
    base_path = "/Volumes/BLUE/RCEMIP/SAM_CRM/RCE_large300/3D"
    files = [
        "SAM_CRM_RCE_large300_3D_0000660600.nc",
        "SAM_CRM_RCE_large300_3D_0000680400.nc",
        "SAM_CRM_RCE_large300_3D_0000700200.nc",
        "SAM_CRM_RCE_large300_3D_0000720000.nc",
    ]

    data_list = []
    x = None
    y = None
    z = None

    for filename in files[:1] if single_timestep else files:
        filepath = Path(base_path) / filename
        with nc.Dataset(filepath, 'r') as ds:
            # Load coordinates (same for all files)
            if x is None:
                x = ds.variables['x'][:].astype(np.float32)
                y = ds.variables['y'][:].astype(np.float32)
                z = ds.variables['z'][:].astype(np.float32)

            # Load data: shape is (time, z, y, x), need to transpose to (x, y, z, time)
            var_data = ds.variables[variable][:].astype(np.float32)  # (1, z, y, x) or (z, y, x)
            if var_data.ndim == 4:
                # Time-dependent variable: (time, z, y, x)
                var_data = np.transpose(var_data, (3, 2, 1, 0))  # (x, y, z, time)
            elif var_data.ndim == 3:
                # Time-independent variable: (z, y, x)
                # Expand to (x, y, z, 1) for consistency
                var_data = np.transpose(var_data, (2, 1, 0))  # (x, y, z)
                var_data = var_data[..., np.newaxis]  # (x, y, z, 1)
            data_list.append(var_data)

    data = np.concatenate(data_list, axis=3)

    return data, (x, y, z)


def load_CM1_RCEMIP_large(variable, single_timestep=False):
    """
    Load a variable from CM1 RCEMIP large (3km) simulation.

    Parameters
    ----------
    variable : str
        Variable name: 'clw', 'cli', 'plw', 'pli', 'ta', 'ua', 'va', 'wa', 'pa', 'hus', 'hur', 'tntr', 'tntrs', 'tntrl'
        - clw: cloud liquid water
        - cli: cloud ice
        - plw: precipitating liquid water
        - pli: precipitating ice
        - ta: air temperature
        - ua: eastward wind
        - va: northward wind
        - wa: vertical velocity
        - pa: pressure
        - hus: specific humidity
        - hur: relative humidity
        - tntr: total radiative heating rate
        - tntrs: shortwave radiative heating rate
        - tntrl: longwave radiative heating rate
    single_timestep : bool
        If True, return only first timestep with shape (nx, ny, nz, 1)
        If False, stack all timesteps to shape (nx, ny, nz, nt)

    Returns
    -------
    data : np.ndarray
        Data array with dimensions (nx, ny, nz, nt)
    coords : tuple
        (x, y, z) coordinate arrays
    """
    base_path = "/Volumes/BLUE/RCEMIP/CM1/RCE_large300/3D"
    files = [
        "CM1_RCE_large300_3D_allvars_hour1860.nc",
        "CM1_RCE_large300_3D_allvars_hour2040.nc",
        "CM1_RCE_large300_3D_allvars_hour2220.nc",
        "CM1_RCE_large300_3D_allvars_hour2400.nc",
    ]

    data_list = []
    x = None
    y = None
    z = None

    for filename in files[:1] if single_timestep else files:
        filepath = Path(base_path) / filename
        with nc.Dataset(filepath, 'r') as ds:
            # Load coordinates
            if x is None:
                x = ds.variables['x'][:].astype(np.float32)
                y = ds.variables['y'][:].astype(np.float32)
                z = ds.variables['z'][:].astype(np.float32)

            # Load data: shape is (time, nk, nj, ni), need to transpose to (ni, nj, nk, time)
            var_data = ds.variables[variable][:].astype(np.float32)  # (1, nk, nj, ni)
            var_data = np.transpose(var_data, (3, 2, 1, 0))  # (ni, nj, nk, 1)
            data_list.append(var_data)

    data = np.concatenate(data_list, axis=3)

    return data, (x, y, z)


def load_HRRR(variable, single_timestep=False, load_forecast=False, season=None):
    """
    Load a variable from HRRR (High-Resolution Rapid Refresh) forecast data.

    Loads full 3D isobaric data, calculates heights from geopotential height,
    and interpolates to a uniform height grid.

    Parameters
    ----------
    variable : str
        Variable name available on isobaric levels:
        't' (temperature), 'u' (eastward wind), 'v' (northward wind),
        'gh' (geopotential height), 'q' (specific humidity),
        'dpt' (dew point), 'clwmr' (cloud water), 'rwmr' (rain water),
        'snmr' (snow water), 'w' (vertical velocity), 'r' (relative humidity)

    single_timestep : bool
        If True, return only first timestep with shape (nx, ny, nz, 1)
        If False, stack all available timesteps to shape (nx, ny, nz, nt)
        Default: False

    load_forecast : bool
        If True, load 48-hour forecasts (f48 files)
        If False, load init/analysis data (f00 files)
        Default: False

    season : str, optional
        Filter data by season: 'summer' (Jun-Aug) or 'fall' (Sep-Nov)
        If None, use all available data
        Default: None

    Returns
    -------
    data : np.ndarray
        Data array with dimensions (nx, ny, nz, nt) interpolated to uniform heights
    coords : tuple
        (x, y, z_uniform) coordinate arrays where z_uniform is uniform height grid
    """
    # Determine which forecast hour to load
    forecast_hour = 'f48' if load_forecast else 'f00'

    # Season-to-months mapping
    season_months = {
        'summer': [6, 7, 8],      # June, July, August
        'fall': [9, 10, 11],      # September, October, November
    }

    # Find all HRRR files with the requested forecast hour
    base_path = Path("/Volumes/BLUE/HRRR")

    if not base_path.exists():
        raise FileNotFoundError(f"HRRR data directory not found: {base_path}")

    # Search all date directories for matching files
    matching_files = []
    search_pattern = f"*_wrfprsf_{forecast_hour}_valid_*.grib2"

    for date_dir in sorted(base_path.glob("????????")):  # Match YYYYMMDD format
        if date_dir.is_dir():
            # Extract month from directory name (YYYYMMDD)
            try:
                date_str = date_dir.name
                month = int(date_str[4:6])

                # Filter by season if specified
                if season is not None:
                    if season not in season_months:
                        raise ValueError(f"Unknown season: {season}. Use 'summer' or 'fall'")
                    if month not in season_months[season]:
                        continue
            except (ValueError, IndexError):
                continue

            for time_dir in sorted(date_dir.glob("t*z")):
                matching_files.extend(sorted(time_dir.glob(search_pattern)))

    if not matching_files:
        raise FileNotFoundError(f"No HRRR files found with forecast hour {forecast_hour}")

    # Load only first timestep if requested, otherwise load all
    grib_files = matching_files[:1] if single_timestep else matching_files

    # Load and stack data from all timesteps
    var_data_list = []
    gh_data_list = []

    for grib_file in grib_files:
        ds = xr.open_dataset(str(grib_file), engine='cfgrib',
                            filter_by_keys={'typeOfLevel': 'isobaricInhPa'},
                            decode_timedelta=False)

        # Check variable exists
        if variable not in ds.data_vars:
            available_vars = [v for v in ds.data_vars if not v.startswith('_')]
            raise ValueError(f"Variable '{variable}' not found. Available: {available_vars}")

        # Load data and geopotential height
        # Shape: (isobaricInhPa, y, x)
        var_data_list.append(ds[variable].values.astype(np.float32))
        gh_data_list.append(ds['gh'].values.astype(np.float32))

        # Get coordinates from first file
        if len(var_data_list) == 1:
            pressure_levels = ds.isobaricInhPa.values.astype(np.float32)
            lon = ds.longitude.values.astype(np.float32)
            lat = ds.latitude.values.astype(np.float32)

        ds.close()

    # Stack data along time dimension
    # Shape: (nt, isobaricInhPa, y, x)
    var_data = np.stack(var_data_list, axis=0)
    gh_data = np.stack(gh_data_list, axis=0)

    # gh is already geopotential height in meters
    height_levels = gh_data  # Shape: (nt, nz, ny, nx)

    # Transpose to (nt, x, y, z)
    var_data = np.transpose(var_data, (0, 3, 2, 1))  # (nt, x, y, nz)
    height_levels = np.transpose(height_levels, (0, 3, 2, 1))  # (nt, x, y, nz)

    nt, nx, ny, nz_orig = var_data.shape

    # Create uniform height grid
    # Use the full range of actual heights in the data
    h_min_data = np.nanmin(height_levels)
    h_max_data = np.nanmax(height_levels)

    # Automatically determine dz from median spacing between pressure levels
    # Sample from middle of first timestep
    h_sample = height_levels[0, nx//2, ny//2, :]  # Sample one column
    h_sample_sorted = np.sort(h_sample[~np.isnan(h_sample)])
    if len(h_sample_sorted) > 1:
        dh = np.diff(h_sample_sorted)
        dz = float(np.median(dh))
        # Round to reasonable value
        dz = int(dz / 100) * 100  # Round to nearest 100m
        if dz < 100:
            dz = 100.0
    else:
        dz = 500.0  # Fallback default

    z_uniform = np.arange(h_min_data, h_max_data + dz, dz, dtype=np.float32)
    nz_new = len(z_uniform)

    # Interpolate variable data to uniform height grid
    # Sort heights along z-axis and handle NaNs
    from scipy.interpolate import interp1d


    # For each (t, x, y) position, we need to:
    # 1. Sort the height/value pair by height
    # 2. Remove NaNs
    # 3. Interpolate to uniform grid

    # Vectorized approach: reshape to process all columns at once
    var_interp = np.full((nt, nx, ny, nz_new), np.nan, dtype=np.float32)

    # Reshape to (n_columns, nz_orig) where n_columns = nt*nx*ny
    n_cols = nt * nx * ny
    nz_orig = height_levels.shape[-1]
    h_flat = height_levels.reshape(n_cols, nz_orig)
    v_flat = var_data.reshape(n_cols, nz_orig)

    # Use mean heights across all columns as reference for interpolation
    h_mean = np.mean(h_flat, axis=0)  # Shape: (nz_orig,)

    # Print height range information
    heights_km = h_mean / 1000.0
    print(f"Height range (km): {np.nanmin(heights_km):.2f} to {np.nanmax(heights_km):.2f}")

    print(f"Interpolating {nt}×{nx}×{ny} columns to uniform height grid (dz={dz:.0f}m)...",
          end='', flush=True)

    # Vectorized interpolation: use axis=0 for values along columns
    f = interp1d(h_mean, v_flat, axis=1, kind='linear',
                 bounds_error=False, fill_value=np.nan)
    var_flat = f(z_uniform)

    # Reshape back to (nt, nx, ny, nz_new)
    var_interp = var_flat.reshape(nt, nx, ny, nz_new)

    print(" Done!", flush=True)

    # Transpose from (nt, nx, ny, nz_new) to (nx, ny, nz_new, nt)
    var_interp = np.transpose(var_interp, (1, 2, 3, 0))

    # Convert water variables from g/kg to kg/kg
    if variable in ['q', 'clwmr', 'snmr', 'rwmr']:
        var_interp = var_interp * 1000.0

    # Create coordinate arrays
    # For x and y, use simple grid indices multiplied by grid spacing
    dx = np.mean(np.diff(lon[0, :]))
    dy = np.mean(np.diff(lat[:, 0]))
    x_coords = np.arange(nx, dtype=np.float32) * dx
    y_coords = np.arange(ny, dtype=np.float32) * dy

    return var_interp, (x_coords, y_coords, z_uniform)
