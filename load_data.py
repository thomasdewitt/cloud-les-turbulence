"""
Load functions for various LES datasets.

Each function loads a single variable from potentially multiple .nc files and returns:
  - data: stacked array with dimensions (nx, ny, nz, nt)
  - coords: tuple of (x, y, z) coordinate arrays
"""

import numpy as np
import netCDF4 as nc
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
    Load a variable from CM1 RCEMIP simulation.

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
