"""
Plot vertical profiles for LES cloud datasets.

Generates vertical profiles showing mean and ±1std for all variables.
Saves both PNG visualizations and netCDF files with profile statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import netCDF4 as nc
import load_data


def plot_profile(z, variable_name, data_slice, output_file, figsize=(8, 6)):
    """
    Plot vertical profile with mean and ±1std shading (single timestep).

    Parameters
    ----------
    z : np.ndarray
        Height levels (m)
    variable_name : str
        Name of variable for plot title
    data_slice : np.ndarray
        Data array with shape (nx, ny, nz) for a single timestep
    output_file : str or Path
        Output file path
    figsize : tuple
        Figure size
    """
    # Compute mean and std across spatial (x, y) dimensions only
    mean_profile = np.mean(data_slice, axis=(0, 1))  # (nz,)
    std_profile = np.std(data_slice, axis=(0, 1))    # (nz,)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot mean line
    ax.plot(mean_profile, z, 'k-', linewidth=2, label='Mean')

    # Plot ±1std shading
    ax.fill_betweenx(z, mean_profile - std_profile, mean_profile + std_profile,
                      alpha=0.3, color='gray', label='±1σ')

    ax.set_xlabel(variable_name, fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title(f'Vertical Profile: {variable_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def save_profile_data_to_netcdf(z, variable_name, data, output_file):
    """
    Calculate and save profile statistics to netCDF file.

    For each variable, saves:
    1. mean_profile: mean calculated over all x, y, t dimensions
    2. normalized_log_std: std of ln(data/mean_profile) across x, y, t

    Parameters
    ----------
    z : np.ndarray
        Height levels (m)
    variable_name : str
        Name of variable
    data : np.ndarray
        Data array with shape (nx, ny, nz, nt)
    output_file : str or Path
        Output file path for netCDF file
    """
    # Create or append to netCDF file
    ds = nc.Dataset(str(output_file), 'a', format='NETCDF4')

    # Calculate mean over all x, y, t dimensions -> shape (nz,)
    mean_profile = np.mean(data, axis=(0, 1, 3))

    # Calculate normalized_log_std
    # Shape of data is (nx, ny, nz, nt)
    nz = data.shape[2]

    # Avoid division by zero: only process where mean_profile > 0
    normalized_log_std = np.zeros(nz)

    for k in range(nz):
        if mean_profile[k] > 0:
            # Divide by mean at this level and take log
            normalized_data = data[:, :, k, :] / mean_profile[k]
            # Only take log where values are positive
            valid = normalized_data > 0
            if np.any(valid):
                log_data = np.log(normalized_data[valid])
                normalized_log_std[k] = np.std(log_data)
            else:
                normalized_log_std[k] = np.nan
        else:
            normalized_log_std[k] = np.nan

    # Create or get the height dimension
    if 'height' not in ds.dimensions:
        ds.createDimension('height', len(z))
        zvar = ds.createVariable('height', 'f4', ('height',))
        zvar[:] = z
        zvar.units = 'm'
        zvar.description = 'Height levels'

    # Save mean_profile
    if f'{variable_name}_mean_profile' not in ds.variables:
        var = ds.createVariable(f'{variable_name}_mean_profile', 'f4', ('height',))
        var.units = 'variable units'
        var.description = f'Mean of {variable_name} calculated over all x, y, t dimensions'
        var[:] = mean_profile
    else:
        ds.variables[f'{variable_name}_mean_profile'][:] = mean_profile

    # Save normalized_log_std
    if f'{variable_name}_normalized_log_std' not in ds.variables:
        var = ds.createVariable(f'{variable_name}_normalized_log_std', 'f4', ('height',))
        var.units = 'dimensionless'
        var.description = f'Standard deviation of ln({variable_name}/mean_profile) calculated over all x, y, t dimensions'
        var[:] = normalized_log_std
    else:
        ds.variables[f'{variable_name}_normalized_log_std'][:] = normalized_log_std

    ds.close()


def plot_profiles_SAM_COMBLE():
    """Plot vertical profiles for SAM COMBLE dataset (one plot per timestep)."""
    print("Plotting profiles: SAM_COMBLE...")

    output_dir = Path("profiles")
    output_dir.mkdir(exist_ok=True)

    variables = ['QV', 'QN', 'QI', 'TABS', 'W', 'U']

    # NetCDF output file
    nc_output = output_dir / "SAM_COMBLE_profiles.nc"
    if nc_output.exists():
        nc_output.unlink()  # Remove if exists to start fresh

    for var in variables:
        data, (x, y, z) = load_data.load_SAM_COMBLE(var)

        # Save profile statistics to netCDF
        save_profile_data_to_netcdf(z, var, data, nc_output)

        # Process all available timesteps
        for t_idx in range(data.shape[3]):
            data_slice = data[:, :, :, t_idx]
            output_file = output_dir / f"SAM_COMBLE_t{t_idx}_{var}.png"
            plot_profile(z, var, data_slice, output_file)
            print(f"  Saved {output_file}")

    # Calculate and save QT (sum of moisture variables)
    print("  Computing QT (QV + QN + QI)...", end=' ')
    data_qv, (x, y, z) = load_data.load_SAM_COMBLE('QV')
    data_qn, _ = load_data.load_SAM_COMBLE('QN')
    data_qi, _ = load_data.load_SAM_COMBLE('QI')
    data_qt = data_qv + data_qn + data_qi
    print("done")

    # Save QT to netCDF
    save_profile_data_to_netcdf(z, 'QT', data_qt, nc_output)

    # Plot QT
    for t_idx in range(data_qt.shape[3]):
        data_slice = data_qt[:, :, :, t_idx]
        output_file = output_dir / f"SAM_COMBLE_t{t_idx}_QT.png"
        plot_profile(z, 'QT', data_slice, output_file)
        print(f"  Saved {output_file}")

    print(f"  Saved {nc_output}")


def plot_profiles_SAM_DYCOMS():
    """Plot vertical profiles for SAM DYCOMS dataset (one plot per timestep)."""
    print("Plotting profiles: SAM_DYCOMS...")

    output_dir = Path("profiles")
    output_dir.mkdir(exist_ok=True)

    variables = ['QV', 'QN', 'TABS', 'W', 'U']

    # NetCDF output file
    nc_output = output_dir / "SAM_DYCOMS_profiles.nc"
    if nc_output.exists():
        nc_output.unlink()  # Remove if exists to start fresh

    for var in variables:
        data, (x, y, z) = load_data.load_SAM_DYCOMS(var)

        # Save profile statistics to netCDF
        save_profile_data_to_netcdf(z, var, data, nc_output)

        # Process all available timesteps
        for t_idx in range(data.shape[3]):
            data_slice = data[:, :, :, t_idx]
            output_file = output_dir / f"SAM_DYCOMS_t{t_idx}_{var}.png"
            plot_profile(z, var, data_slice, output_file)
            print(f"  Saved {output_file}")

    # Calculate and save QT (sum of moisture variables)
    print("  Computing QT (QV + QN)...", end=' ')
    data_qv, (x, y, z) = load_data.load_SAM_DYCOMS('QV')
    data_qn, _ = load_data.load_SAM_DYCOMS('QN')
    data_qt = data_qv + data_qn
    print("done")

    # Save QT to netCDF
    save_profile_data_to_netcdf(z, 'QT', data_qt, nc_output)

    # Plot QT
    for t_idx in range(data_qt.shape[3]):
        data_slice = data_qt[:, :, :, t_idx]
        output_file = output_dir / f"SAM_DYCOMS_t{t_idx}_QT.png"
        plot_profile(z, 'QT', data_slice, output_file)
        print(f"  Saved {output_file}")

    print(f"  Saved {nc_output}")


def plot_profiles_SAM_TWPICE():
    """Plot vertical profiles for SAM TWPICE dataset (one timestep)."""
    print("Plotting profiles: SAM_TWPICE...")

    output_dir = Path("profiles")
    output_dir.mkdir(exist_ok=True)

    variables = ['QV', 'QC', 'QI', 'TABS', 'W', 'U']

    # NetCDF output file
    nc_output = output_dir / "SAM_TWPICE_profiles.nc"
    if nc_output.exists():
        nc_output.unlink()  # Remove if exists to start fresh

    for var in variables:
        data, (x, y, z) = load_data.load_SAM_TWPICE(var, single_timestep=True)

        # Save profile statistics to netCDF
        save_profile_data_to_netcdf(z, var, data, nc_output)

        # Extract single timestep from shape (nx, ny, nz, 1)
        data_slice = data[:, :, :, 0]
        output_file = output_dir / f"SAM_TWPICE_t0_{var}.png"
        plot_profile(z, var, data_slice, output_file)
        print(f"  Saved {output_file}")

    # Calculate and save QT (sum of moisture variables)
    print("  Computing QT (QV + QC + QI)...", end=' ')
    data_qv, (x, y, z) = load_data.load_SAM_TWPICE('QV', single_timestep=True)
    data_qc, _ = load_data.load_SAM_TWPICE('QC', single_timestep=True)
    data_qi, _ = load_data.load_SAM_TWPICE('QI', single_timestep=True)
    data_qt = data_qv + data_qc + data_qi
    print("done")

    # Save QT to netCDF
    save_profile_data_to_netcdf(z, 'QT', data_qt, nc_output)

    # Plot QT
    data_slice = data_qt[:, :, :, 0]
    output_file = output_dir / f"SAM_TWPICE_t0_QT.png"
    plot_profile(z, 'QT', data_slice, output_file)
    print(f"  Saved {output_file}")

    print(f"  Saved {nc_output}")


def plot_profiles_SAM_RCEMIP():
    """Plot vertical profiles for SAM RCEMIP dataset (one plot per timestep)."""
    print("Plotting profiles: SAM_RCEMIP...")

    output_dir = Path("profiles")
    output_dir.mkdir(exist_ok=True)

    variables = ['QV', 'QN', 'TABS', 'W', 'U']

    # NetCDF output file
    nc_output = output_dir / "SAM_RCEMIP_profiles.nc"
    if nc_output.exists():
        nc_output.unlink()  # Remove if exists to start fresh

    for var in variables:
        data, (x, y, z) = load_data.load_SAM_RCEMIP(var)

        # Save profile statistics to netCDF
        save_profile_data_to_netcdf(z, var, data, nc_output)

        for t_idx in range(data.shape[3]):
            data_slice = data[:, :, :, t_idx]
            output_file = output_dir / f"SAM_RCEMIP_t{t_idx}_{var}.png"
            plot_profile(z, var, data_slice, output_file)
            print(f"  Saved {output_file}")

    # Calculate and save QT (sum of moisture variables)
    print("  Computing QT (QV + QN)...", end=' ')
    data_qv, (x, y, z) = load_data.load_SAM_RCEMIP('QV')
    data_qn, _ = load_data.load_SAM_RCEMIP('QN')
    data_qt = data_qv + data_qn
    print("done")

    # Save QT to netCDF
    save_profile_data_to_netcdf(z, 'QT', data_qt, nc_output)

    # Plot QT
    for t_idx in range(data_qt.shape[3]):
        data_slice = data_qt[:, :, :, t_idx]
        output_file = output_dir / f"SAM_RCEMIP_t{t_idx}_QT.png"
        plot_profile(z, 'QT', data_slice, output_file)
        print(f"  Saved {output_file}")

    print(f"  Saved {nc_output}")


def plot_profiles_CM1_RCEMIP():
    """Plot vertical profiles for CM1 RCEMIP dataset (one plot per timestep)."""
    print("Plotting profiles: CM1_RCEMIP...")

    output_dir = Path("profiles")
    output_dir.mkdir(exist_ok=True)

    variables = {'clw': 'CLW (g/g)', 'cli': 'CLI (g/g)', 'ta': 'Ta (K)', 'wa': 'W (m/s)', 'ua': 'U (m/s)'}

    # NetCDF output file
    nc_output = output_dir / "CM1_RCEMIP_profiles.nc"
    if nc_output.exists():
        nc_output.unlink()  # Remove if exists to start fresh

    for var, label in variables.items():
        data, (x, y, z) = load_data.load_CM1_RCEMIP(var)

        # Save profile statistics to netCDF
        save_profile_data_to_netcdf(z, var, data, nc_output)

        for t_idx in range(data.shape[3]):
            data_slice = data[:, :, :, t_idx]
            output_file = output_dir / f"CM1_RCEMIP_t{t_idx}_{var}.png"
            plot_profile(z, label, data_slice, output_file)
            print(f"  Saved {output_file}")

    # Calculate and save QT (sum of moisture variables: hus + clw + cli)
    print("  Computing QT (hus + clw + cli)...", end=' ')
    data_qv, (x, y, z) = load_data.load_CM1_RCEMIP('hus')
    data_clw, _ = load_data.load_CM1_RCEMIP('clw')
    data_cli, _ = load_data.load_CM1_RCEMIP('cli')
    data_qt = data_qv + data_clw + data_cli
    print("done")

    # Save QT to netCDF
    save_profile_data_to_netcdf(z, 'QT', data_qt, nc_output)

    # Plot QT
    for t_idx in range(data_qt.shape[3]):
        data_slice = data_qt[:, :, :, t_idx]
        output_file = output_dir / f"CM1_RCEMIP_t{t_idx}_QT.png"
        plot_profile(z, 'QT (hus+clw+cli)', data_slice, output_file)
        print(f"  Saved {output_file}")

    print(f"  Saved {nc_output}")


def plot_profiles_HRRR():
    """Plot vertical profiles for HRRR dataset (single timestep)."""
    print("Plotting profiles: HRRR...")

    output_dir = Path("profiles")
    output_dir.mkdir(exist_ok=True)

    # Variables available on isobaric levels
    variables = ['t', 'q', 'clwmr', 'u', 'w']

    # NetCDF output file
    nc_output = output_dir / "HRRR_profiles.nc"
    if nc_output.exists():
        nc_output.unlink()  # Remove if exists to start fresh

    for var in variables:
        try:
            data, (x, y, z) = load_data.load_HRRR(var, single_timestep=True)

            # Save profile statistics to netCDF
            save_profile_data_to_netcdf(z, var, data, nc_output)

            # Process single timestep
            data_slice = data[:, :, :, 0]
            output_file = output_dir / f"HRRR_t0_{var}.png"
            plot_profile(z, var, data_slice, output_file)
            print(f"  Saved {output_file}")
        except ValueError as e:
            print(f"  Warning: Could not load {var}: {e}")
            continue

    # Calculate and save total water (q + clwmr + snmr)
    print("  Computing QT (q + clwmr + snmr)...", end=' ')
    try:
        data_q, (x, y, z) = load_data.load_HRRR('q', single_timestep=True)
        data_clwmr, _ = load_data.load_HRRR('clwmr', single_timestep=True)
        data_snmr, _ = load_data.load_HRRR('snmr', single_timestep=True)
        data_qt = data_q + data_clwmr + data_snmr
        print("done")

        # Save QT to netCDF
        save_profile_data_to_netcdf(z, 'QT', data_qt, nc_output)

        # Plot QT
        data_slice = data_qt[:, :, :, 0]
        output_file = output_dir / "HRRR_t0_QT.png"
        plot_profile(z, 'QT', data_slice, output_file)
        print(f"  Saved {output_file}")
    except Exception as e:
        print(f"Could not compute QT: {e}")

    print(f"  Saved {nc_output}")


def main():
    """Process all datasets and create vertical profile visualizations."""
    print("=" * 60)
    print("Plotting vertical profiles for LES datasets")
    print("=" * 60)

    plot_profiles_SAM_COMBLE()
    plot_profiles_SAM_DYCOMS()
    # plot_profiles_SAM_TWPICE()
    plot_profiles_SAM_RCEMIP()
    plot_profiles_CM1_RCEMIP()
    plot_profiles_HRRR()

    print("=" * 60)
    print("All profile visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
