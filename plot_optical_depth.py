"""
Plot optical depth for LES cloud datasets.

Loads each timestep of each simulation, calculates optical depth from cloud water and ice,
and saves visualizations to optical_depth_images/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
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

    data_qc, (x, y, z) = load_data.load_SAM_COMBLE('QC')
    data_qi, _ = load_data.load_SAM_COMBLE('QI')

    dx = x[1] - x[0]
    optical_depth = calculate_optical_depth(data_qc, data_qi, z, dx)
    opacity = calculate_opacity(optical_depth)

    output_dir = Path("optical_depth_images")
    output_dir.mkdir(exist_ok=True)

    timesteps = [3600, 5400, 7200]
    for t_idx, timestep in enumerate(timesteps):
        opacity_2d = opacity[:, :, t_idx]
        output_file = output_dir / f"SAM_COMBLE_t{timestep}.png"
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

    timesteps = [43200, 50400, 57600]
    for t_idx, timestep in enumerate(timesteps):
        opacity_2d = opacity[:, :, t_idx]
        output_file = output_dir / f"SAM_DYCOMS_t{timestep}.png"
        create_opacity_image(opacity_2d, output_file)
        print(f"  Saved {output_file}")


def process_SAM_TWPICE():
    """Process SAM TWPICE dataset."""
    print("Processing SAM_TWPICE...")

    data_qc, (x, y, z) = load_data.load_SAM_TWPICE('QC')
    data_qi, _ = load_data.load_SAM_TWPICE('QI')

    dx = x[1] - x[0]
    optical_depth = calculate_optical_depth(data_qc, data_qi, z, dx)
    opacity = calculate_opacity(optical_depth)

    output_dir = Path("optical_depth_images")
    output_dir.mkdir(exist_ok=True)

    timesteps = [150, 1800, 3450]
    for t_idx, timestep in enumerate(timesteps):
        opacity_2d = opacity[:, :, t_idx]
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

    timesteps = [86400, 777600, 1468800, 2160000]
    for t_idx, timestep in enumerate(timesteps):
        opacity_2d = opacity[:, :, t_idx]
        output_file = output_dir / f"SAM_RCEMIP_t{timestep}.png"
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

    hours = [0, 408, 816, 1200]
    for t_idx, hour in enumerate(hours):
        opacity_2d = opacity[:, :, t_idx]
        output_file = output_dir / f"CM1_RCEMIP_hour{hour:04d}.png"
        create_opacity_image(opacity_2d, output_file)
        print(f"  Saved {output_file}")


def main():
    """Process all datasets and create optical depth visualizations."""
    print("=" * 60)
    print("Processing LES datasets for optical depth visualization")
    print("=" * 60)

    try:
        process_SAM_COMBLE()
    except Exception as e:
        print(f"  Error processing SAM_COMBLE: {e}")

    try:
        process_SAM_DYCOMS()
    except Exception as e:
        print(f"  Error processing SAM_DYCOMS: {e}")

    try:
        process_SAM_TWPICE()
    except Exception as e:
        print(f"  Error processing SAM_TWPICE: {e}")

    try:
        process_SAM_RCEMIP()
    except Exception as e:
        print(f"  Error processing SAM_RCEMIP: {e}")

    try:
        process_CM1_RCEMIP()
    except Exception as e:
        print(f"  Error processing CM1_RCEMIP: {e}")

    print("=" * 60)
    print("All visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
