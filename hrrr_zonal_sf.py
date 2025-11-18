"""Minimal script: load HRRR u field at pressure level, compute zonal structure function, plot in loglog."""

import numpy as np
import xarray as xr
import scaleinvariance
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_HRRR_pressure_level(variable, pressure_level, single_timestep=False):
    """
    Load HRRR variable at specified pressure level (no height interpolation).

    Parameters
    ----------
    variable : str
        Variable name (e.g., 'u', 'v', 't', 'q')
    pressure_level : int
        Pressure level in hPa (e.g., 500)
    single_timestep : bool
        If True, return only first timestep

    Returns
    -------
    data : np.ndarray
        Shape (nx, ny, nt)
    coords : tuple
        (x, y) coordinate arrays
    """
    base_path = Path("/Volumes/BLUE/HRRR")

    # Find all grib2 files (f00 = analysis)
    matching_files = []
    for date_dir in sorted(base_path.glob("????????")):
        if date_dir.is_dir():
            for time_dir in sorted(date_dir.glob("t*z")):
                matching_files.extend(sorted(time_dir.glob("*_wrfprsf_f00_valid_*.grib2")))

    if not matching_files:
        raise FileNotFoundError("No HRRR files found")

    grib_files = matching_files[:1] if single_timestep else matching_files

    var_data_list = []
    for grib_file in grib_files:
        ds = xr.open_dataset(str(grib_file), engine='cfgrib',
                            filter_by_keys={'typeOfLevel': 'isobaricInhPa'},
                            decode_timedelta=False)

        # Check if requested pressure level exists
        available_levels = ds.isobaricInhPa.values
        if pressure_level not in available_levels:
            raise ValueError(f"Pressure level {pressure_level}hPa not found. Available: {available_levels}")

        # Select only the requested pressure level
        var_data = ds[variable].sel(isobaricInhPa=pressure_level).values.astype(np.float32)
        var_data_list.append(var_data)

        # Get coordinates from first file
        if len(var_data_list) == 1:
            lon = ds.longitude.values.astype(np.float32)
            lat = ds.latitude.values.astype(np.float32)

        ds.close()

    # Stack timesteps: (nt, y, x) -> transpose to (x, y, nt)
    data = np.stack(var_data_list, axis=0)  # (nt, y, x)
    data = np.transpose(data, (2, 1, 0))     # (x, y, nt)

    # Create coordinate arrays
    nx, ny = data.shape[0], data.shape[1]
    dx = np.mean(np.diff(lon[0, :]))
    dy = np.mean(np.diff(lat[:, 0]))
    x_coords = np.arange(nx, dtype=np.float32) * dx
    y_coords = np.arange(ny, dtype=np.float32) * dy

    return data, (x_coords, y_coords)


# Pressure level (default 500mb), overridable via command line
pressure_level = int(sys.argv[1]) if len(sys.argv) > 1 else 500

# Load HRRR u field at specified pressure level
u, (x, y) = load_HRRR_pressure_level('u', pressure_level, single_timestep=False)
print(f"Loaded u at {pressure_level}hPa, shape {u.shape}")

# Calculate zonal (x-direction) structure function
lags, sf = scaleinvariance.structure_function_analysis(u, axis=0, order=2)

# Filter out NaN and zero values
valid = (np.isfinite(sf)) & (sf > 0) & (np.isfinite(lags)) & (lags > 0)
lags_clean = lags[valid]
sf_clean = sf[valid]

# Fit regression between 10 and 100 grid points
mask = (lags_clean >= 10) & (lags_clean <= 100)
if np.sum(mask) > 1:
    lags_fit = lags_clean[mask]
    sf_fit = sf_clean[mask]

    # Linear regression in log-log space
    log_lags = np.log10(lags_fit)
    log_sf = np.log10(sf_fit)
    coeffs = np.polyfit(log_lags, log_sf, 1)
    slope = coeffs[0]
    H = slope / 2  # H = slope / 2 for structure functions
else:
    slope = None
    H = None

# Plot in loglog
plt.loglog(lags_clean, sf_clean, 'o-', linewidth=2, markersize=4, label='Data')

if slope is not None:
    # Plot regression line
    sf_fit_line = 10**(np.polyval(coeffs, log_lags))
    plt.loglog(lags_fit, sf_fit_line, '--', linewidth=2, color='red', label=f'Fit (10-100)')

    # Report slope and H on plot
    mid_idx = len(lags_fit) // 2
    plt.text(lags_fit[mid_idx], sf_fit_line[mid_idx] * 0.7,
            f'slope={slope:.3f}\n$H$={H:.3f}',
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    print(f"Slope (10-100): {slope:.4f}")
    print(f"Hurst exponent H: {H:.4f}")

plt.xlabel('Zonal separation (grid points)')
plt.ylabel('Structure Function S₂')
plt.title(f'HRRR Zonal Structure Function ({pressure_level}hPa)')
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
