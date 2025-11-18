"""
Compute scaling functions (structure functions and spectra) for LES datasets.

For each dataset and variable (U, QV, QT where QT = QV+QC+QI):
  - Calculate vertically and horizontally separated structure functions
  - Calculate vertically and horizontally separated power spectral densities
  - Save all results to a master netCDF file with metadata
"""

import numpy as np
import netCDF4 as nc
import scaleinvariance
from pathlib import Path
from datetime import datetime
import load_data


# Altitude ranges for each dataset (meters)
ALTITUDE_RANGES = {
    'SAM_COMBLE': (200, 1200),
    'SAM_DYCOMS': (10, 900),
    'SAM_TWPICE': (5000, 10000),
    'SAM_RCEMIP': (5000, 10000),
    'SAM_RCEMIP_large': (5000, 10000),
    'CM1_RCEMIP': (5000, 10000),
    'CM1_RCEMIP_large': (5000, 10000),
    'HRRR': (2000, 4000), 
}

# Variables to compute for each dataset
DATASET_VARIABLES = {
    # 'SAM_COMBLE': {
    #     'U': 'U',
    #     'W': 'W',
    #     'QV': 'QV',
    #     'QT': ['QV', 'QN', 'QI']
    # },
    # 'SAM_DYCOMS': {
    #     'U': 'U',
    #     'W': 'W',
    #     'QV': 'QV',
    #     'QT': ['QV', 'QN']
    # },
    # 'SAM_TWPICE': {
    #     'U': 'U',
    #     'W': 'W',
    #     'QV': 'QV',
    #     'QT': ['QV', 'QC', 'QI']
    # },
    # 'SAM_RCEMIP': {
    #     'U': 'U',
    #     'W': 'W',
    #     'QV': 'QV',
    #     'QT': ['QV', 'QN']
    # },
    # 'CM1_RCEMIP': {
    #     'U': 'ua',
    #     'W': 'wa',
    #     'QV': 'hus',  # hus = specific humidity (water vapor)
    #     'QT': ['hus', 'clw', 'cli']
    # },
    # 'SAM_RCEMIP_large': {
    #     'U': 'ua',
    #     'W': 'wa',
    #     'QV': 'QV',
    #     'QT': ['QV', 'clw', 'cli']
    # },
    # 'CM1_RCEMIP_large': {
    #     'U': 'ua',
    #     'W': 'wa',
    #     'QV': 'hus',  # hus = specific humidity (water vapor)
    #     'QT': ['hus', 'clw', 'cli']
    # },
    'HRRR': {
        'U': 'u',  # u = eastward wind component
        'W': 'w',  # w = vertical velocity
        'QV': 'q',  # q = specific humidity (water vapor)
        'QT': ['q', 'clwmr']  # q + cloud water mixing ratio
    },
}

STRUCTURE_FUNCTION_ORDER = 2


def load_and_slice_data(dataset_name, variable, z_range):
    """
    Load data and slice to altitude range.

    Parameters
    ----------
    dataset_name : str
        Name of dataset (SAM_COMBLE, SAM_DYCOMS, etc.)
    variable : str
        Variable name
    z_range : tuple
        (z_min, z_max) in meters

    Returns
    -------
    data_sliced : np.ndarray
        Data array sliced to altitude range, shape (nx, ny, nz_subset, nt)
    z_subset : np.ndarray
        Height values within the range
    """
    z_min, z_max = z_range

    # Load data from appropriate loader
    if dataset_name == 'SAM_COMBLE':
        data, (x, y, z) = load_data.load_SAM_COMBLE(variable)
    elif dataset_name == 'SAM_DYCOMS':
        data, (x, y, z) = load_data.load_SAM_DYCOMS(variable)
    elif dataset_name == 'SAM_TWPICE':
        data, (x, y, z) = load_data.load_SAM_TWPICE(variable)
    elif dataset_name == 'SAM_RCEMIP':
        data, (x, y, z) = load_data.load_SAM_RCEMIP(variable)
    elif dataset_name == 'SAM_RCEMIP_large':
        data, (x, y, z) = load_data.load_SAM_RCEMIP_large(variable)
    elif dataset_name == 'CM1_RCEMIP':
        data, (x, y, z) = load_data.load_CM1_RCEMIP(variable)
    elif dataset_name == 'CM1_RCEMIP_large':
        data, (x, y, z) = load_data.load_CM1_RCEMIP_large(variable)
    elif dataset_name == 'HRRR':
        data, (x, y, z) = load_data.load_HRRR(variable,
                                              season='summer',
                                             single_timestep=True,
                                             load_forecast=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Find indices within altitude range
    z_indices = np.where((z >= z_min) & (z <= z_max))[0]

    # Slice data: (nx, ny, nz_subset, nt)
    data_sliced = data[:, :, z_indices, :]
    z_subset = z[z_indices]

    return data_sliced, z_subset


def compute_structure_functions(data, order=STRUCTURE_FUNCTION_ORDER):
    """
    Compute vertical and horizontal structure functions.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (nx, ny, nz, nt)
    order : int
        Order of structure function

    Returns
    -------
    results : dict
        Dictionary with keys: lags_v, sf_v, lags_h, sf_h (all 1D arrays or None)
    """
    # Vertical structure function (along z-axis, axis=2)
    lags_v, sf_v = scaleinvariance.structure_function_analysis(
        data, axis=2, order=order, lags='powers of 1.1'
    )

    # Horizontal structure function (along x-axis, axis=0)
    lags_h, sf_h = scaleinvariance.structure_function_analysis(
        data, axis=0, order=order, lags='powers of 1.1'
    )

    return {
        'lags_v': lags_v, 'sf_v': sf_v,
        'lags_h': lags_h, 'sf_h': sf_h
    }


def compute_spectra(data):
    """
    Compute vertical and horizontal power spectral densities.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (nx, ny, nz, nt)

    Returns
    -------
    results : dict
        Dictionary with keys: freq_v, psd_v, freq_h, psd_h (all 1D arrays or None)
    """
    # Vertical spectrum (along z-axis, axis=2)
    freq_v, psd_v = scaleinvariance.spectral_analysis(
        data, axis=2, nbins=100
    )

    # Horizontal spectrum (along x-axis, axis=0)
    freq_h, psd_h = scaleinvariance.spectral_analysis(
        data, axis=0, nbins=100
    )

    return {
        'freq_v': freq_v, 'psd_v': psd_v,
        'freq_h': freq_h, 'psd_h': psd_h
    }


def save_to_netcdf(ds, dataset_name, variable_name, sf_results, spec_results):
    """
    Save structure function and spectral results to netCDF group.

    Parameters
    ----------
    ds : netCDF4.Dataset
        NetCDF dataset
    dataset_name : str
        Name of dataset
    variable_name : str
        Name of variable (U, QV, QT)
    sf_results : dict
        Structure function results with keys lags_v, sf_v, lags_h, sf_h
    spec_results : dict
        Spectral results with keys freq_v, psd_v, freq_h, psd_h
    """
    grp_name = dataset_name
    if grp_name not in ds.groups:
        grp = ds.createGroup(grp_name)
    else:
        grp = ds.groups[grp_name]

    # Save vertical structure function
    if sf_results['sf_v'] is not None:
        dim_name = f'lag_v_{variable_name}'
        if dim_name not in grp.dimensions:
            grp.createDimension(dim_name, len(sf_results['lags_v']))

        v = grp.createVariable(f'{variable_name}_lags_vertical', 'f4', (dim_name,))
        v[:] = sf_results['lags_v']
        v.units = 'grid points'
        v.description = 'Lags for vertical structure function'

        v = grp.createVariable(f'{variable_name}_sf_vertical', 'f4', (dim_name,))
        v[:] = sf_results['sf_v']
        v.units = 'data units'
        v.description = f'Order-{STRUCTURE_FUNCTION_ORDER} vertical structure function'

    # Save horizontal structure function
    if sf_results['sf_h'] is not None:
        dim_name = f'lag_h_{variable_name}'
        if dim_name not in grp.dimensions:
            grp.createDimension(dim_name, len(sf_results['lags_h']))

        v = grp.createVariable(f'{variable_name}_lags_horizontal', 'f4', (dim_name,))
        v[:] = sf_results['lags_h']
        v.units = 'grid points'
        v.description = 'Lags for horizontal structure function'

        v = grp.createVariable(f'{variable_name}_sf_horizontal', 'f4', (dim_name,))
        v[:] = sf_results['sf_h']
        v.units = 'data units'
        v.description = f'Order-{STRUCTURE_FUNCTION_ORDER} horizontal structure function'

    # Save vertical spectrum
    if spec_results['psd_v'] is not None:
        dim_name = f'freq_v_{variable_name}'
        if dim_name not in grp.dimensions:
            grp.createDimension(dim_name, len(spec_results['freq_v']))

        v = grp.createVariable(f'{variable_name}_frequencies_vertical', 'f4', (dim_name,))
        v[:] = spec_results['freq_v']
        v.units = '1/grid points'
        v.description = 'Frequencies for vertical spectrum'

        v = grp.createVariable(f'{variable_name}_psd_vertical', 'f4', (dim_name,))
        v[:] = spec_results['psd_v']
        v.units = 'power'
        v.description = 'Vertical power spectral density'

    # Save horizontal spectrum
    if spec_results['psd_h'] is not None:
        dim_name = f'freq_h_{variable_name}'
        if dim_name not in grp.dimensions:
            grp.createDimension(dim_name, len(spec_results['freq_h']))

        v = grp.createVariable(f'{variable_name}_frequencies_horizontal', 'f4', (dim_name,))
        v[:] = spec_results['freq_h']
        v.units = '1/grid points'
        v.description = 'Frequencies for horizontal spectrum'

        v = grp.createVariable(f'{variable_name}_psd_horizontal', 'f4', (dim_name,))
        v[:] = spec_results['psd_h']
        v.units = 'power'
        v.description = 'Horizontal power spectral density'


def main():
    """Compute scaling functions for all datasets and save to master netCDF."""
    output_dir = Path('scaling_functions')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'scaling_functions_all_datasets.nc'

    print("=" * 70)
    print("Computing Scaling Functions for LES Datasets")
    print("=" * 70)

    # Create master netCDF
    print(f"\nCreating master netCDF: {output_file}")
    ds = nc.Dataset(str(output_file), 'w', format='NETCDF4')

    # Global metadata
    ds.title = 'Scaling Functions for LES Cloud Datasets'
    ds.author = 'Thomas D. DeWitt'
    ds.created = datetime.now().isoformat()
    ds.description = 'Structure functions and power spectral densities computed from LES datasets'
    ds.structure_function_order = STRUCTURE_FUNCTION_ORDER
    ds.reference = 'Lovejoy & Schertzer scaling analysis framework'

    # Process each dataset
    for dataset_name in DATASET_VARIABLES.keys():
        print(f"\n{dataset_name}")
        print("-" * 70)

        z_range = ALTITUDE_RANGES[dataset_name]
        var_config = DATASET_VARIABLES[dataset_name]

        print(f"Altitude range: {z_range[0]:,}m to {z_range[1]:,}m")

        # Process U variable
        if var_config.get('U'):
            var_name_loaded = var_config['U']
            print(f"  Loading U ({var_name_loaded})...", end=' ')
            data_u, z_u = load_and_slice_data(dataset_name, var_name_loaded, z_range)
            print(f"shape {data_u.shape}")

            print(f"    Computing structure functions...", end=' ')
            sf_u = compute_structure_functions(data_u)
            print("done")

            print(f"    Computing spectra...", end=' ')
            spec_u = compute_spectra(data_u)
            print("done")

            save_to_netcdf(ds, dataset_name, 'U', sf_u, spec_u)
            print(f"  ✓ Saved U to netCDF")
            del data_u, sf_u, spec_u

        # Process W variable
        if var_config.get('W'):
            var_name_loaded = var_config['W']
            print(f"  Loading W ({var_name_loaded})...", end=' ')
            data_w, z_w = load_and_slice_data(dataset_name, var_name_loaded, z_range)
            print(f"shape {data_w.shape}")

            print(f"    Computing structure functions...", end=' ')
            sf_w = compute_structure_functions(data_w)
            print("done")

            print(f"    Computing spectra...", end=' ')
            spec_w = compute_spectra(data_w)
            print("done")

            save_to_netcdf(ds, dataset_name, 'W', sf_w, spec_w)
            print(f"  ✓ Saved W to netCDF")
            del data_w, sf_w, spec_w

        # Process QV variable
        if var_config.get('QV'):
            var_name_loaded = var_config['QV']
            print(f"  Loading QV ({var_name_loaded})...", end=' ')
            data_qv, z_qv = load_and_slice_data(dataset_name, var_name_loaded, z_range)
            print(f"shape {data_qv.shape}")

            print(f"    Computing structure functions...", end=' ')
            sf_qv = compute_structure_functions(data_qv)
            print("done")

            print(f"    Computing spectra...", end=' ')
            spec_qv = compute_spectra(data_qv)
            print("done")

            save_to_netcdf(ds, dataset_name, 'QV', sf_qv, spec_qv)
            print(f"  ✓ Saved QV to netCDF")
            del data_qv, sf_qv, spec_qv

        # Process QT (sum of moisture variables)
        print(f"  Loading QT (sum of {', '.join(var_config['QT'])})...", end=' ')
        data_qt = None
        for var in var_config['QT']:
            data, z_qt = load_and_slice_data(dataset_name, var, z_range)
            if data_qt is None:
                data_qt = data.copy()
            else:
                data_qt += data

        print(f"shape {data_qt.shape}")

        print(f"    Computing structure functions...", end=' ')
        sf_qt = compute_structure_functions(data_qt)
        print("done")

        print(f"    Computing spectra...", end=' ')
        spec_qt = compute_spectra(data_qt)
        print("done")

        save_to_netcdf(ds, dataset_name, 'QT', sf_qt, spec_qt)
        print(f"  ✓ Saved QT to netCDF")
        del data_qt, sf_qt, spec_qt

    ds.close()

    print("\n" + "=" * 70)
    print(f"✓ Master netCDF saved: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
