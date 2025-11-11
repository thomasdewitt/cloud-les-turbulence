"""
Plot optical depth for LES cloud datasets.

Loads each timestep of each simulation, calculates optical depth from cloud water and ice,
and saves visualizations to optical_depth_images/ directory.

Also generates vertical profiles showing mean and ±1std for all variables.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import netCDF4 as nc
import load_data


def calculate_optical_depth(data_qc, data_qi, heights, dx):
    """
    Calculate optical depth from cloud water and ice content.

    Parameters
    ----------
    data_qc : np.ndarray
        Cloud liquid water content (g/kg), shape (nx, ny, nz, nt)
    data_qi : np.ndarray
        Cloud ice content (g/kg), shape (nx, ny, nz, nt)
    heights : np.ndarray
        Height levels (m)
    dx : float
        Grid spacing (m)

    Returns
    -------
    optical_depth : np.ndarray
        Optical depth, shape (nx, ny, nt)
    """
    nx, ny, nz, nt = data_qc.shape

    # Physical constants
    g = 9.81
    R = 287.05
    assumed_T = 290
    scale_height = 7000
    p0 = 1013

    # Calculate pressure and density at each level
    pressures = p0 * np.exp(-heights / scale_height)  # (nz,)
    densities = (pressures * 100) / (R * assumed_T)  # (nz,)

    # Total condensate (combine QC and QI)
    # Shape: (nx, ny, nz, nt)
    total_condensate = data_qc + data_qi

    # Calculate water path at each grid point and level
    # densities shape needs to be broadcast: (1, 1, nz, 1)
    water_path = densities[np.newaxis, np.newaxis, :, np.newaxis] * total_condensate * dx

    # Effective radius assumptions
    re_liquid = 10  # micrometers
    re_ice = 30    # micrometers

    # Optical depth relationships
    # For liquid: LWP = 0.6292 * tau * re
    # For ice: use similar relationship
    fac_liquid = 1 / (0.6292 * re_liquid)
    fac_ice = 1 / (0.6292 * re_ice)

    # Calculate optical depths
    # Simple approach: use single effective radius for total condensate
    fac = fac_liquid  # Use liquid water effective radius
    optical_depth_levels = fac * water_path

    # Integrate over vertical dimension (sum along z-axis)
    optical_depth = optical_depth_levels.sum(axis=2)  # (nx, ny, nt)

    return optical_depth


def calculate_opacity(optical_depth):
    """
    Convert optical depth to opacity (0 to 1).

    Parameters
    ----------
    optical_depth : np.ndarray
        Optical depth

    Returns
    -------
    opacity : np.ndarray
        Opacity values (0 to 1)
    """
    return 1 - np.exp(-optical_depth)


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


def create_opacity_image(opacity, output_file, figsize=(8, 8)):
    """
    Create and save an opacity visualization.

    Parameters
    ----------
    opacity : np.ndarray
        Opacity array, shape (nx, ny)
    output_file : str or Path
        Output file path
    figsize : tuple
        Figure size
    """
    # Create custom colormap: blue (sky) to white (opaque cloud)
    sky_blue = '#3A4AA6'
    colors = [sky_blue, 'white']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('sky_to_cloud', colors, N=n_bins)

    # Create figure without axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(opacity.T, origin='lower', cmap=cmap, vmin=0, vmax=1,
              interpolation='nearest')

    # Remove all axes, ticks, labels
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save with no padding
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_SAM_COMBLE():
    """Process SAM COMBLE dataset."""
    print("Processing SAM_COMBLE...")

    # COMBLE has QN (total condensate) and QI (ice)
    # Use QN as liquid+ice and QI separately for better optical depth estimate
    data_qn, (x, y, z) = load_data.load_SAM_COMBLE('QN')
    data_qi, _ = load_data.load_SAM_COMBLE('QI')

    dx = x[1] - x[0]
    # For optical depth, we'll use the total condensate QN
    optical_depth = calculate_optical_depth(data_qn, np.zeros_like(data_qn), z, dx)
    opacity = calculate_opacity(optical_depth)

    output_dir = Path("optical_depth_images")
    output_dir.mkdir(exist_ok=True)

    # Process all available timesteps
    for t_idx in range(opacity.shape[2]):
        opacity_2d = opacity[:, :, t_idx]
        output_file = output_dir / f"SAM_COMBLE_t{t_idx}.png"
        create_opacity_image(opacity_2d, output_file)
        print(f"  Saved {output_file}")


def process_SAM_DYCOMS():
    """Process SAM DYCOMS dataset."""
    print("Processing SAM_DYCOMS...")

    # DYCOMS only has QN (total condensate), no separate QC and QI
    data_qn, (x, y, z) = load_data.load_SAM_DYCOMS('QN')

    # Treat QN as all liquid for optical depth calculation
    dx = x[1] - x[0]
    optical_depth = calculate_optical_depth(data_qn, np.zeros_like(data_qn), z, dx)
    opacity = calculate_opacity(optical_depth)

    output_dir = Path("optical_depth_images")
    output_dir.mkdir(exist_ok=True)

    # Process all available timesteps
    for t_idx in range(opacity.shape[2]):
        opacity_2d = opacity[:, :, t_idx]
        output_file = output_dir / f"SAM_DYCOMS_t{t_idx}.png"
        create_opacity_image(opacity_2d, output_file)
        print(f"  Saved {output_file}")


def process_SAM_TWPICE():
    """Process SAM TWPICE dataset."""
    print("Processing SAM_TWPICE...")

    output_dir = Path("optical_depth_images")
    output_dir.mkdir(exist_ok=True)

    timesteps = [150, 1800, 3450]
    for timestep in timesteps:
        data_qc, (x, y, z) = load_data.load_SAM_TWPICE('QC', single_timestep=True)
        data_qi, _ = load_data.load_SAM_TWPICE('QI', single_timestep=True)

        dx = x[1] - x[0]
        optical_depth = calculate_optical_depth(data_qc, data_qi, z, dx)
        opacity = calculate_opacity(optical_depth)

        opacity_2d = opacity[:, :, 0]
        output_file = output_dir / f"SAM_TWPICE_t{timestep}.png"
        create_opacity_image(opacity_2d, output_file)
        print(f"  Saved {output_file}")


def process_SAM_RCEMIP():
    """Process SAM RCEMIP dataset."""
    print("Processing SAM_RCEMIP...")

    # RCEMIP only has QN (total condensate), no separate QC and QI
    data_qn, (x, y, z) = load_data.load_SAM_RCEMIP('QN')

    dx = x[1] - x[0]
    optical_depth = calculate_optical_depth(data_qn, np.zeros_like(data_qn), z, dx)
    opacity = calculate_opacity(optical_depth)

    output_dir = Path("optical_depth_images")
    output_dir.mkdir(exist_ok=True)

    # Handle variable number of timesteps
    for t_idx in range(opacity.shape[2]):
        opacity_2d = opacity[:, :, t_idx]
        output_file = output_dir / f"SAM_RCEMIP_t{t_idx}.png"
        create_opacity_image(opacity_2d, output_file)
        print(f"  Saved {output_file}")


def process_CM1_RCEMIP():
    """Process CM1 RCEMIP dataset."""
    print("Processing CM1_RCEMIP...")

    data_clw, (x, y, z) = load_data.load_CM1_RCEMIP('clw')
    data_cli, _ = load_data.load_CM1_RCEMIP('cli')

    # Convert from g/g to g/kg for consistency
    data_qc = data_clw * 1000
    data_qi = data_cli * 1000

    dx = x[1] - x[0]
    optical_depth = calculate_optical_depth(data_qc, data_qi, z, dx)
    opacity = calculate_opacity(optical_depth)

    output_dir = Path("optical_depth_images")
    output_dir.mkdir(exist_ok=True)

    # Handle variable number of timesteps
    for t_idx in range(opacity.shape[2]):
        opacity_2d = opacity[:, :, t_idx]
        output_file = output_dir / f"CM1_RCEMIP_t{t_idx}.png"
        create_opacity_image(opacity_2d, output_file)
        print(f"  Saved {output_file}")


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

    print(f"  Saved {nc_output}")


def plot_profiles_SAM_TWPICE():
    """Plot vertical profiles for SAM TWPICE dataset (one timestep)."""
    print("Plotting profiles: SAM_TWPICE...")

    output_dir = Path("profiles")
    output_dir.mkdir(exist_ok=True)

    timestep = 150
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
        output_file = output_dir / f"SAM_TWPICE_t{timestep}_{var}.png"
        plot_profile(z, var, data_slice, output_file)
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

    print(f"  Saved {nc_output}")


def main():
    """Process all datasets and create optical depth visualizations and profiles."""
    print("=" * 60)
    print("Processing LES datasets for optical depth visualization")
    print("=" * 60)

    process_SAM_COMBLE()
    process_SAM_DYCOMS()
    process_SAM_RCEMIP()
    process_CM1_RCEMIP()
    process_SAM_TWPICE()

    print("=" * 60)
    print("Plotting vertical profiles")
    print("=" * 60)

    plot_profiles_SAM_COMBLE()
    plot_profiles_SAM_DYCOMS()
    plot_profiles_SAM_TWPICE()
    plot_profiles_SAM_RCEMIP()
    plot_profiles_CM1_RCEMIP()

    print("=" * 60)
    print("All visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
