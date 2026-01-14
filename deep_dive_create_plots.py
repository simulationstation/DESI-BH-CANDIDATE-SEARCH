#!/usr/bin/env python3
"""
Create RV vs MJD plots for top 10 CLEAN_STELLAR_CANDIDATE targets.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Create output directory
PLOT_DIR = "data/derived/deep_dive_clean_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load data
with open('data/derived/deep_dive_clean_for_plot.json', 'r') as f:
    clean_data = json.load(f)

print(f"Creating plots for {len(clean_data)} targets...")

for i, target in enumerate(clean_data, 1):
    targetid = target['targetid']
    epochs = target['epochs']

    if not epochs:
        print(f"  [{i}] TARGETID {targetid}: No epochs, skipping")
        continue

    # Extract data
    mjds = np.array([e['mjd'] for e in epochs])
    rvs = np.array([e['rv'] for e in epochs])
    errs = np.array([e['rv_err'] for e in epochs])
    success = [e['success'] for e in epochs]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot points
    for j, (mjd, rv, err, succ) in enumerate(zip(mjds, rvs, errs, success)):
        color = 'green' if succ else 'orange'
        marker = 'o' if succ else 's'
        ax.errorbar(mjd, rv, yerr=err, fmt=marker, color=color, capsize=3,
                    markersize=10, markeredgecolor='black', markeredgewidth=1)

    # Add median line
    rv_median = np.median(rvs)
    ax.axhline(rv_median, color='gray', linestyle='--', alpha=0.7)

    # Labels
    ax.set_xlabel('MJD', fontsize=12)
    ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)

    simbad_type = target.get('simbad_type', 'NO_MATCH')
    title = (f"TARGETID {targetid}\n"
             f"N={target['N_epochs']}, ΔRV={target['delta_rv_kms']:.0f} km/s, "
             f"S_robust={target['S_robust']:.1f}, d_max={target['d_max']:.1f}")
    ax.set_title(title, fontsize=11)

    # Metrics box
    n_success = sum(success)
    n_total = len(success)
    textstr = (f"MJD span: {target['MJD_span']:.1f} days\n"
               f"SUCCESS: {n_success}/{n_total}\n"
               f"SIMBAD: {simbad_type}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, markeredgecolor='black', label='SUCCESS=True'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange',
               markersize=10, markeredgecolor='black', label='SUCCESS=False'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    outpath = os.path.join(PLOT_DIR, f"rv_deep_dive_{targetid}.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [{i}] Saved {outpath}")

print(f"\nDone. {len(clean_data)} plots saved to {PLOT_DIR}/")
