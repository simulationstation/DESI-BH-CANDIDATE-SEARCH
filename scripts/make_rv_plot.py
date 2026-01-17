#!/usr/bin/env python3
"""Generate publication-quality RV plot for RNAAS submission."""

import matplotlib.pyplot as plt
import numpy as np

# Data points
mjd = np.array([59568.488, 59605.380, 59607.374, 59607.389])
rv = np.array([-86.4, 59.7, 26.4, 25.2])
err = np.array([0.6, 0.8, 1.1, 1.1])

# Set up publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
ax.set_facecolor('white')

# Dashed grey line connecting points
ax.plot(mjd, rv, linestyle='--', color='grey', linewidth=1, zorder=1)

# Black markers with error bars
ax.errorbar(mjd, rv, yerr=err, fmt='o', color='black', markersize=6,
            capsize=3, capthick=1, elinewidth=1, zorder=2)

# Labels and title
ax.set_xlabel('Time [MJD]')
ax.set_ylabel(r'Radial Velocity [km s$^{-1}$]')
ax.set_title('DESI RVs for Gaia DR3 3802130935635096832')

# Add horizontal line at RV=0 for reference
ax.axhline(0, color='lightgrey', linestyle=':', linewidth=0.8, zorder=0)

plt.tight_layout()
plt.savefig('/home/primary/DESI-BH-CANDIDATE-SEARCH/rv_plot.png', dpi=300,
            facecolor='white', edgecolor='none', bbox_inches='tight')
plt.close()

print("Saved rv_plot.png")
