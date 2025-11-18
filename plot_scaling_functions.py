"""
Plot scaling functions (structure functions and spectra) for LES datasets.

For each dataset and variable, creates a 2-subplot figure:
  - Left: Vertical and horizontal structure functions on log-log plot
  - Right: Vertical and horizontal power spectral densities on log-log plot

Includes reference lines with theoretical slopes and custom formatting.
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from pathlib import Path


# Color scheme
COLOR_HORIZONTAL = '#1f77b4'  # blue
COLOR_VERTICAL = '#ff7f0e'    # orange

# Figure styling
FIGSIZE = (12, 6)
FONTSIZE_SMALL = 10
FONTSIZE_LARGE = 12

# Spectral slope reference lines (for log-log plots)
# General variables: -5/3 (horizontal), -11/5 (vertical)
# W variable: -7/9 (horizontal), -1/5 (vertical)
SPECTRAL_SLOPES = {
    'general': {'horiz': -5/3, 'vert': -11/5},
    # 'W': {'horiz': -5/3, 'vert': -11/5},
    'W': {'horiz': -7/9, 'vert': -3/5},
}


def format_log_log_ax(ax, xlabel='', ylabel=''):
    """Format axis for log-log plotting with nice ticks."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LARGE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LARGE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_SMALL)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')


def plot_structure_functions(ax, dataset_name, variable_name, grp):
    """
    Plot vertical and horizontal structure functions on log-log axis with regression lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    dataset_name : str
        Name of dataset
    variable_name : str
        Name of variable (U, QV, QT, W)
    grp : netCDF4.Group
        netCDF group containing the data
    """
    plotted_horiz = False
    plotted_vert = False

    # Load horizontal structure function
    lags_h = np.array(grp.variables[f'{variable_name}_lags_horizontal'][:])
    sf_h = np.array(grp.variables[f'{variable_name}_sf_horizontal'][:])

    # Filter out NaN and zero values
    valid_h = (np.isfinite(sf_h)) & (sf_h > 0) & (np.isfinite(lags_h)) & (lags_h > 0)
    lags_h_clean = lags_h[valid_h]
    sf_h_clean = sf_h[valid_h]

    if len(lags_h_clean) > 0:
        ax.plot(lags_h_clean, sf_h_clean, color=COLOR_HORIZONTAL, linewidth=2,
                marker='o', markersize=4, label='Horizontal', alpha=0.8)
        plotted_horiz = True

        # Fit regression between 10 and 100 grid points
        mask_h = (lags_h_clean >= 10) & (lags_h_clean <= 100)
        if np.sum(mask_h) > 1:
            lags_h_fit = lags_h_clean[mask_h]
            sf_h_fit = sf_h_clean[mask_h]

            # Linear regression in log-log space
            log_lags_h = np.log10(lags_h_fit)
            log_sf_h = np.log10(sf_h_fit)
            coeffs_h = np.polyfit(log_lags_h, log_sf_h, 1)
            slope_h = coeffs_h[0]
            H_h = slope_h / 2  # H = slope / 2 for structure functions

            # Plot regression line over fit interval
            sf_h_fit_line = 10**(np.polyval(coeffs_h, log_lags_h))
            ax.plot(lags_h_fit, sf_h_fit_line, color=COLOR_HORIZONTAL, linewidth=1,
                   linestyle='--', alpha=0.7)

            # Label with H value
            mid_idx = len(lags_h_fit) // 2
            ax.text(lags_h_fit[mid_idx], sf_h_fit_line[mid_idx] * 0.8,
                   f'$H_h$={H_h:.2f}', fontsize=FONTSIZE_SMALL,
                   color=COLOR_HORIZONTAL, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Load vertical structure function
    lags_v = np.array(grp.variables[f'{variable_name}_lags_vertical'][:])
    sf_v = np.array(grp.variables[f'{variable_name}_sf_vertical'][:])

    # Filter out NaN and zero values
    valid_v = (np.isfinite(sf_v)) & (sf_v > 0) & (np.isfinite(lags_v)) & (lags_v > 0)
    lags_v_clean = lags_v[valid_v]
    sf_v_clean = sf_v[valid_v]

    if len(lags_v_clean) > 0:
        ax.plot(lags_v_clean, sf_v_clean, color=COLOR_VERTICAL, linewidth=2,
                marker='s', markersize=4, label='Vertical', alpha=0.8)
        plotted_vert = True

        # Fit regression between 1 and 10 grid points
        mask_v = (lags_v_clean >= 1) & (lags_v_clean <= 10)
        if np.sum(mask_v) > 1:
            lags_v_fit = lags_v_clean[mask_v]
            sf_v_fit = sf_v_clean[mask_v]

            # Linear regression in log-log space
            log_lags_v = np.log10(lags_v_fit)
            log_sf_v = np.log10(sf_v_fit)
            coeffs_v = np.polyfit(log_lags_v, log_sf_v, 1)
            slope_v = coeffs_v[0]
            H_v = slope_v / 2  # H = slope / 2 for structure functions

            # Plot regression line over fit interval
            sf_v_fit_line = 10**(np.polyval(coeffs_v, log_lags_v))
            ax.plot(lags_v_fit, sf_v_fit_line, color=COLOR_VERTICAL, linewidth=1,
                   linestyle='--', alpha=0.7)

            # Label with H value
            mid_idx = len(lags_v_fit) // 2
            ax.text(lags_v_fit[mid_idx], sf_v_fit_line[mid_idx] * 1.2,
                   f'$H_v$={H_v:.2f}', fontsize=FONTSIZE_SMALL,
                   color=COLOR_VERTICAL, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add reference lines (except for W variable)
    if variable_name != 'W':
        # Theoretical slopes for structure functions
        # For typical Kolmogorov: horizontal ~ lag^(2/3), vertical ~ lag^(6/5)
        anchor_idx = len(lags_h_clean) // 3 if plotted_horiz else 0

        if plotted_horiz and anchor_idx < len(lags_h_clean):
            # Horizontal reference line: slope 2/3
            slope_h = 2/3
            lag_min = np.log10(lags_h_clean[0])
            lag_max = np.log10(lags_h_clean[-1])
            lags_ref_h = np.logspace(lag_min, lag_max, 100)

            # Calculate intercept to pass through anchor point
            intercept_h = np.log10(sf_h_clean[anchor_idx]) - slope_h * np.log10(lags_h_clean[anchor_idx])
            sf_ref_h = 10**(slope_h * np.log10(lags_ref_h) + intercept_h)

            ax.plot(lags_ref_h, sf_ref_h, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.annotate(f"$H_h=1/3$", xy=(lags_ref_h[-20], sf_ref_h[-20]),
                       xytext=(lags_ref_h[-20]*0.7, sf_ref_h[-20]*1.5),
                       arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.5),
                       fontsize=FONTSIZE_SMALL)

        anchor_idx_v = len(lags_v_clean) // 3 if plotted_vert else 0

        if plotted_vert and anchor_idx_v < len(lags_v_clean):
            # Vertical reference line: slope 6/5
            # slope_v = 6/5
            slope_v = 1.6
            lag_min = np.log10(lags_v_clean[0])
            lag_max = np.log10(lags_v_clean[-1])
            lags_ref_v = np.logspace(lag_min, lag_max, 100)

            # Calculate intercept to pass through anchor point
            intercept_v = np.log10(sf_v_clean[anchor_idx_v]) - slope_v * np.log10(lags_v_clean[anchor_idx_v])
            sf_ref_v = 10**(slope_v * np.log10(lags_ref_v) + intercept_v)

            ax.plot(lags_ref_v, sf_ref_v, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.annotate(f"$H_v=.8$", xy=(lags_ref_v[20], sf_ref_v[20]),
                       xytext=(lags_ref_v[20]*1.5, sf_ref_v[20]*0.6),
                       arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.5),
                       fontsize=FONTSIZE_SMALL)

    if plotted_horiz or plotted_vert:
        ax.legend(fontsize=FONTSIZE_SMALL, frameon=True, loc='best')

    format_log_log_ax(ax, xlabel='Separation lag (grid points)',
                      ylabel=f'Structure Function S₂')


def plot_spectra(ax, dataset_name, variable_name, grp):
    """
    Plot vertical and horizontal power spectral densities on log-log axis with reference lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    dataset_name : str
        Name of dataset
    variable_name : str
        Name of variable (U, QV, QT, W)
    grp : netCDF4.Group
        netCDF group containing the data
    """
    plotted_horiz = False
    plotted_vert = False

    # Load horizontal spectrum
    freq_h = np.array(grp.variables[f'{variable_name}_frequencies_horizontal'][:])
    psd_h = np.array(grp.variables[f'{variable_name}_psd_horizontal'][:])

    # Filter out NaN and zero values
    valid_h = (np.isfinite(psd_h)) & (psd_h > 0) & (np.isfinite(freq_h)) & (freq_h > 0)
    freq_h_clean = freq_h[valid_h]
    psd_h_clean = psd_h[valid_h]

    if len(freq_h_clean) > 0:
        ax.plot(freq_h_clean, psd_h_clean, color=COLOR_HORIZONTAL, linewidth=2,
                marker='o', markersize=4, label='Horizontal', alpha=0.8)
        plotted_horiz = True

    # Load vertical spectrum
    freq_v = np.array(grp.variables[f'{variable_name}_frequencies_vertical'][:])
    psd_v = np.array(grp.variables[f'{variable_name}_psd_vertical'][:])

    # Filter out NaN and zero values
    valid_v = (np.isfinite(psd_v)) & (psd_v > 0) & (np.isfinite(freq_v)) & (freq_v > 0)
    freq_v_clean = freq_v[valid_v]
    psd_v_clean = psd_v[valid_v]

    if len(freq_v_clean) > 0:
        ax.plot(freq_v_clean, psd_v_clean, color=COLOR_VERTICAL, linewidth=2,
                marker='s', markersize=4, label='Vertical', alpha=0.8)
        plotted_vert = True

    # Add reference lines for spectra
    slopes = SPECTRAL_SLOPES['W'] if variable_name == 'W' else SPECTRAL_SLOPES['general']

    anchor_idx = len(freq_h_clean) // 3 if plotted_horiz else 0

    if plotted_horiz and anchor_idx < len(freq_h_clean):
        # Horizontal reference line
        slope_h = slopes['horiz']
        freq_min = np.log10(freq_h_clean[0])
        freq_max = np.log10(freq_h_clean[-1])
        freqs_ref_h = np.logspace(freq_min, freq_max, 100)

        # Calculate intercept to pass through anchor point
        intercept_h = np.log10(psd_h_clean[anchor_idx]) - slope_h * np.log10(freq_h_clean[anchor_idx])
        psd_ref_h = 10**(slope_h * np.log10(freqs_ref_h) + intercept_h)

        ax.plot(freqs_ref_h, psd_ref_h, color='black', linestyle='-', linewidth=1, alpha=0.5)
        slope_label = f"${slope_h:.2g}$"
        ax.annotate(f"slope={slope_label}", xy=(freqs_ref_h[-20], psd_ref_h[-20]),
                   xytext=(freqs_ref_h[-20]*0.7, psd_ref_h[-20]*1.5),
                   arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.5),
                   fontsize=FONTSIZE_SMALL)

    anchor_idx_v = len(freq_v_clean) // 3 if plotted_vert else 0

    if plotted_vert and anchor_idx_v < len(freq_v_clean):
        # Vertical reference line
        slope_v = slopes['vert']
        freq_min = np.log10(freq_v_clean[0])
        freq_max = np.log10(freq_v_clean[-1])
        freqs_ref_v = np.logspace(freq_min, freq_max, 100)

        # Calculate intercept to pass through anchor point
        intercept_v = np.log10(psd_v_clean[anchor_idx_v]) - slope_v * np.log10(freq_v_clean[anchor_idx_v])
        psd_ref_v = 10**(slope_v * np.log10(freqs_ref_v) + intercept_v)

        ax.plot(freqs_ref_v, psd_ref_v, color='black', linestyle='-', linewidth=1, alpha=0.5)
        slope_label = f"${slope_v:.2g}$"
        ax.annotate(f"slope={slope_label}", xy=(freqs_ref_v[20], psd_ref_v[20]),
                   xytext=(freqs_ref_v[20]*1.5, psd_ref_v[20]*0.6),
                   arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.5),
                   fontsize=FONTSIZE_SMALL)

    if plotted_horiz or plotted_vert:
        ax.legend(fontsize=FONTSIZE_SMALL, frameon=True, loc='best')

    format_log_log_ax(ax, xlabel='Frequency (1/grid points)',
                      ylabel='Power Spectral Density')


def main():
    """Load netCDF and create plots for each dataset/variable combination."""
    input_file = Path('scaling_functions') / 'scaling_functions_all_datasets.nc'
    output_dir = Path('scaling_functions')

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run compute_scaling_functions.py first.")
        return

    print("=" * 70)
    print("Plotting Scaling Functions")
    print("=" * 70)

    with nc.Dataset(str(input_file), 'r') as ds:
        ds.set_auto_mask(False)

        # Get list of datasets (groups)
        for dataset_name in sorted(ds.groups.keys()):
            print(f"\n{dataset_name}")
            print("-" * 70)

            grp = ds.groups[dataset_name]

            # Get list of variables (extract unique variable names)
            var_names = set()
            for var_key in grp.variables.keys():
                # Variable names appear in keys like '{var}_lags_vertical'
                for suffix in ['_lags_vertical', '_sf_vertical', '_lags_horizontal',
                               '_sf_horizontal', '_frequencies_vertical', '_psd_vertical',
                               '_frequencies_horizontal', '_psd_horizontal']:
                    if suffix in var_key:
                        var_name = var_key.replace(suffix, '')
                        var_names.add(var_name)
                        break

            for variable_name in sorted(var_names):
                print(f"  Processing {variable_name}...", end=' ')

                # Create figure with 2 subplots side by side
                fig, (ax_sf, ax_spec) = plt.subplots(1, 2, figsize=FIGSIZE)

                # Plot structure functions on left
                plot_structure_functions(ax_sf, dataset_name, variable_name, grp)

                # Plot spectra on right
                plot_spectra(ax_spec, dataset_name, variable_name, grp)

                # Overall title
                fig.suptitle(f'{dataset_name} - {variable_name}', fontsize=FONTSIZE_LARGE + 2)

                # Tight layout
                plt.tight_layout()

                # Save figure
                output_file = output_dir / f'{dataset_name}_{variable_name}_scaling.png'
                plt.savefig(str(output_file), dpi=150, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                print(f"✓ Saved {output_file.name}")
                plt.close()

    print("\n" + "=" * 70)
    print("✓ All plots saved to scaling_functions/")
    print("=" * 70)


if __name__ == '__main__':
    main()
