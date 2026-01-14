#!/usr/bin/env python3
"""
Create RV vs MJD plot for deep-dive target with quality annotations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load epoch data
with open('data/derived/deep_dive_epochs.json', 'r') as f:
    data = json.load(f)

epochs = data['epochs']
mjds = np.array([e['mjd'] for e in epochs])
rvs = np.array([e['rv'] for e in epochs])
errs = np.array([e['rv_err'] for e in epochs])
success = [e['success'] for e in epochs]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot points, coloring by SUCCESS flag
for i, (mjd, rv, err, succ) in enumerate(zip(mjds, rvs, errs, success)):
    color = 'green' if succ else 'orange'
    marker = 'o' if succ else 's'
    label = None
    if i == 0:
        label = 'SUCCESS=True' if succ else 'SUCCESS=False'
    elif succ and not any(success[:i]):
        label = 'SUCCESS=True'
    elif not succ and all(success[:i]):
        label = 'SUCCESS=False'

    ax.errorbar(mjd, rv, yerr=err, fmt=marker, color=color, capsize=3,
                markersize=10, markeredgecolor='black', markeredgewidth=1,
                label=label)

# Add horizontal line at median
rv_median = np.median(rvs)
ax.axhline(rv_median, color='gray', linestyle='--', alpha=0.7,
           label=f'Median RV = {rv_median:.1f} km/s')

# Add zero line
ax.axhline(0, color='black', linestyle='-', alpha=0.3)

# Annotations
ax.set_xlabel('MJD', fontsize=12)
ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)
ax.set_title('TARGETID 39628001431785529 | SDSS J235252.06+084235.4 (QSO)\n'
             f'N={len(epochs)}, ΔRV={data["delta_rv"]:.0f} km/s, S_robust={data["S_robust"]:.1f}',
             fontsize=11)

# Add text box with metrics
textstr = (f'MJD span: {data["mjd_span"]:.1f} days\n'
           f'd_max: {data["d_max"]:.1f}\n'
           f'SUCCESS=True: {sum(success)}/{len(success)}\n'
           f'RVS_WARN=8: {sum(1 for e in epochs if e.get("rvs_warn", 0) == 8)}/{len(epochs)}')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

# Warning box
warning = 'SIMBAD: QSO\nRV variability is from\nemission line shifts,\nNOT orbital motion'
props2 = dict(boxstyle='round', facecolor='lightyellow', edgecolor='red', alpha=0.9)
ax.text(0.98, 0.02, warning, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right', bbox=props2)

ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Tight layout
plt.tight_layout()

# Save
outpath = 'data/derived/deep_dive_39628001431785529.pdf'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved plot to {outpath}')

# Also save PNG for easy viewing
outpath_png = 'data/derived/deep_dive_39628001431785529.png'
plt.savefig(outpath_png, dpi=150, bbox_inches='tight')
print(f'Saved plot to {outpath_png}')
