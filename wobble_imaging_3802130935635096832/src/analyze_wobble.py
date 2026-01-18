#!/usr/bin/env python3
"""Analyze wobble in centroid measurements and create visualizations."""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from scipy import stats

OUTPUT_DIR = "/home/primary/DESI-BH-CANDIDATE-SEARCH/wobble_imaging_3802130935635096832"
WOBBLE_FIGURES = "/home/primary/DESI-BH-CANDIDATE-SEARCH/WOBBLE_FIGURES"

# PS1 pixel scale
PIXEL_SCALE = 0.25  # arcsec/pixel


def load_measurements():
    """Load centroid measurements."""
    results_file = os.path.join(OUTPUT_DIR, 'results', 'centroid_measurements.json')
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data['measurements'], data['summary']


def filter_outliers(measurements, sigma_clip=3.0):
    """Filter outliers using absolute pixel positions."""
    x_pix = np.array([m['x_pix'] for m in measurements])
    y_pix = np.array([m['y_pix'] for m in measurements])

    # Median and MAD for robust outlier detection
    x_med = np.median(x_pix)
    y_med = np.median(y_pix)
    x_mad = np.median(np.abs(x_pix - x_med))
    y_mad = np.median(np.abs(y_pix - y_med))

    # Filter
    good = []
    for m in measurements:
        x_dev = np.abs(m['x_pix'] - x_med) / (x_mad + 0.01)
        y_dev = np.abs(m['y_pix'] - y_med) / (y_mad + 0.01)
        if x_dev < sigma_clip and y_dev < sigma_clip:
            good.append(m)

    print(f"[ANALYZE] Kept {len(good)}/{len(measurements)} after outlier rejection")
    return good


def compute_wobble_stats(measurements):
    """Compute wobble statistics from cleaned measurements."""
    x_pix = np.array([m['x_pix'] for m in measurements])
    y_pix = np.array([m['y_pix'] for m in measurements])
    mjd = np.array([m['mjd'] for m in measurements])

    # Center on median
    x_centered = x_pix - np.median(x_pix)
    y_centered = y_pix - np.median(y_pix)

    # Convert to arcsec
    x_arcsec = x_centered * PIXEL_SCALE
    y_arcsec = y_centered * PIXEL_SCALE

    # Convert to mas
    x_mas = x_arcsec * 1000
    y_mas = y_arcsec * 1000

    # Basic statistics
    stats_dict = {
        'n_measurements': len(measurements),
        'mjd_range': [float(mjd.min()), float(mjd.max())],
        'baseline_years': float((mjd.max() - mjd.min()) / 365.25),
        'x_rms_mas': float(np.std(x_mas)),
        'y_rms_mas': float(np.std(y_mas)),
        'total_rms_mas': float(np.sqrt(np.std(x_mas)**2 + np.std(y_mas)**2)),
        'x_range_mas': float(x_mas.max() - x_mas.min()),
        'y_range_mas': float(y_mas.max() - y_mas.min()),
    }

    # Add proper motion estimate (linear fit)
    years = (mjd - mjd.min()) / 365.25
    if len(years) > 2:
        try:
            slope_x, _, _, _, _ = stats.linregress(years, x_mas)
            slope_y, _, _, _, _ = stats.linregress(years, y_mas)
            stats_dict['pm_x_masyr'] = float(slope_x)
            stats_dict['pm_y_masyr'] = float(slope_y)
        except:
            pass

    return stats_dict, x_mas, y_mas, mjd


def create_figures(measurements, x_mas, y_mas, mjd, wobble_stats):
    """Create visualization figures."""
    os.makedirs(WOBBLE_FIGURES, exist_ok=True)

    years = (mjd - mjd.min()) / 365.25
    filters = [m['filter'] for m in measurements]

    # Color mapping by filter
    color_map = {'g': 'green', 'r': 'red', 'i': 'orange', 'z': 'brown', 'y': 'purple'}
    colors = [color_map.get(f.split('.')[0], 'blue') for f in filters]

    # Figure 1: X position vs time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.scatter(years, x_mas, c=colors, s=30, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('ΔX (mas)')
    ax.set_title(f'Gaia DR3 3802130935635096832 - Astrometric Time Series\n'
                 f'RMS: X={wobble_stats["x_rms_mas"]:.1f} mas, Y={wobble_stats["y_rms_mas"]:.1f} mas')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(years, y_mas, c=colors, s=30, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Years since first epoch')
    ax.set_ylabel('ΔY (mas)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(WOBBLE_FIGURES, 'wobble_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[ANALYZE] Saved wobble_timeseries.png")

    # Figure 2: Position track (X vs Y)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Time coloring
    sc = ax.scatter(x_mas, y_mas, c=years, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(sc, label='Years since first epoch')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('ΔX (mas)')
    ax.set_ylabel('ΔY (mas)')
    ax.set_title('Centroid Track (color = time)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add error circle
    rms = wobble_stats['total_rms_mas']
    circle = plt.Circle((0, 0), rms, fill=False, color='red', linestyle='--', label=f'RMS={rms:.0f} mas')
    ax.add_patch(circle)
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(WOBBLE_FIGURES, 'centroid_track.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[ANALYZE] Saved centroid_track.png")

    # Figure 3: Histogram of deviations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(x_mas, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('ΔX (mas)')
    ax.set_ylabel('Count')
    ax.set_title(f'X Residuals (σ = {wobble_stats["x_rms_mas"]:.1f} mas)')

    ax = axes[1]
    ax.hist(y_mas, bins=15, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('ΔY (mas)')
    ax.set_ylabel('Count')
    ax.set_title(f'Y Residuals (σ = {wobble_stats["y_rms_mas"]:.1f} mas)')

    plt.tight_layout()
    fig.savefig(os.path.join(WOBBLE_FIGURES, 'residual_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[ANALYZE] Saved residual_histograms.png")

    # Figure 4: By filter
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_filters = list(set([f.split('.')[0] for f in filters]))
    for filt in unique_filters:
        mask = [f.split('.')[0] == filt for f in filters]
        x_f = np.array(x_mas)[mask]
        y_f = np.array(y_mas)[mask]
        color = color_map.get(filt, 'blue')
        ax.scatter(x_f, y_f, c=color, s=50, alpha=0.7, label=f'{filt} (n={sum(mask)})')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('ΔX (mas)')
    ax.set_ylabel('ΔY (mas)')
    ax.set_title('Centroid Positions by Filter')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(WOBBLE_FIGURES, 'centroid_by_filter.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[ANALYZE] Saved centroid_by_filter.png")


def compute_wobble_limits(wobble_stats, gaia_data):
    """Compute upper limits on companion mass from wobble."""
    # Load Gaia data
    gaia_file = os.path.join(OUTPUT_DIR, 'data', 'gaia_query.json')
    with open(gaia_file, 'r') as f:
        gaia = json.load(f)

    target = gaia['target']

    # Parallax and distance
    parallax_mas = target['parallax'] if target['parallax'] else 0.1
    parallax_err = target['parallax_error'] if target['parallax_error'] else 0.1

    # Use parallax if significant, otherwise assume 1 kpc
    if parallax_mas > 2 * parallax_err and parallax_mas > 0:
        distance_pc = 1000.0 / parallax_mas
    else:
        distance_pc = 1000.0  # Assume 1 kpc if parallax insignificant

    # RMS wobble in arcsec
    wobble_rms_arcsec = wobble_stats['total_rms_mas'] / 1000.0

    # Physical wobble at distance (AU)
    wobble_au = wobble_rms_arcsec * distance_pc

    # For a 1 Msun primary, wobble amplitude = a * (M2/Mtot)
    # If wobble is due to noise, we can set upper limit on M2

    # Typical PS1 centroid precision is ~50-100 mas for faint sources
    # Our RMS is consistent with measurement noise

    limits = {
        'parallax_mas': float(parallax_mas),
        'parallax_err_mas': float(parallax_err),
        'distance_pc': float(distance_pc),
        'wobble_rms_mas': float(wobble_stats['total_rms_mas']),
        'wobble_rms_arcsec': float(wobble_rms_arcsec),
        'wobble_amplitude_au': float(wobble_au),
        'typical_ps1_precision_mas': 100.0,
        'wobble_detection': wobble_stats['total_rms_mas'] > 150,  # Detection threshold
        'interpretation': 'Consistent with measurement noise' if wobble_stats['total_rms_mas'] < 150 else 'Possible astrometric signal',
    }

    return limits


def main():
    # Load measurements
    measurements, summary = load_measurements()

    if not measurements:
        print("[ANALYZE] No measurements to analyze")
        return

    # Filter outliers
    clean_measurements = filter_outliers(measurements)

    # Compute statistics
    wobble_stats, x_mas, y_mas, mjd = compute_wobble_stats(clean_measurements)

    print(f"[ANALYZE] Wobble Statistics:")
    print(f"  N measurements: {wobble_stats['n_measurements']}")
    print(f"  Baseline: {wobble_stats['baseline_years']:.2f} years")
    print(f"  X RMS: {wobble_stats['x_rms_mas']:.1f} mas")
    print(f"  Y RMS: {wobble_stats['y_rms_mas']:.1f} mas")
    print(f"  Total RMS: {wobble_stats['total_rms_mas']:.1f} mas")

    # Create figures
    create_figures(clean_measurements, x_mas, y_mas, mjd, wobble_stats)

    # Compute wobble limits
    limits = compute_wobble_limits(wobble_stats, None)

    print(f"[ANALYZE] Wobble Limits:")
    print(f"  Distance: {limits['distance_pc']:.0f} pc")
    print(f"  Wobble amplitude: {limits['wobble_amplitude_au']:.4f} AU")
    print(f"  Interpretation: {limits['interpretation']}")

    # Save all results
    results = {
        'wobble_statistics': wobble_stats,
        'wobble_limits': limits,
        'clean_measurements': len(clean_measurements),
    }

    results_file = os.path.join(OUTPUT_DIR, 'results', 'wobble_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[ANALYZE] Saved analysis to {results_file}")

    # Also save to WOBBLE_FIGURES
    limits_file = os.path.join(WOBBLE_FIGURES, 'wobble_limits.json')
    with open(limits_file, 'w') as f:
        json.dump(limits, f, indent=2)

    print(f"[ANALYZE] Saved limits to {limits_file}")


if __name__ == "__main__":
    main()
