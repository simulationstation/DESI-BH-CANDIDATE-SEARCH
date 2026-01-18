#!/usr/bin/env python3
"""
Create animated GIF showing the measured centroid wobble over time.
Uses only "good" data from same-night consistency test.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# Load same-night consistency results
data_dir = Path(__file__).parent.parent
results_file = data_dir / "results" / "same_night_consistency.json"

with open(results_file) as f:
    results = json.load(f)

# Load original measurements
measurements_file = data_dir / "results" / "centroid_measurements.json"
with open(measurements_file) as f:
    meas_data = json.load(f)

measurements = meas_data["measurements"]

# Get good indices
good_indices = set(results['good_indices'])

# Extract good measurements and sort by MJD
good_meas = [m for m in measurements if m['index'] in good_indices]
good_meas.sort(key=lambda x: x['mjd'])

# Calculate offsets relative to mean of good data
x_pix = np.array([m['x_pix'] for m in good_meas])
y_pix = np.array([m['y_pix'] for m in good_meas])
mjd = np.array([m['mjd'] for m in good_meas])
filters = [m['filter'].replace('.00000', '') for m in good_meas]

x_mean = x_pix.mean()
y_mean = y_pix.mean()

# Convert to mas
dx_mas = (x_pix - x_mean) * 250
dy_mas = (y_pix - y_mean) * 250

# Time in years from first observation
t_years = (mjd - mjd.min()) / 365.25

print(f"Creating wobble animation with {len(good_meas)} good observations")
print(f"Time span: {t_years.max():.2f} years")
print(f"RMS: X={dx_mas.std():.1f} mas, Y={dy_mas.std():.1f} mas")

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))

# Set up axes
lim = max(abs(dx_mas).max(), abs(dy_mas).max()) * 1.4
lim = max(lim, 100)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('RA offset (mas)')
ax.set_ylabel('Dec offset (mas)')

# Draw reference circle at RMS radius
rms = np.sqrt(dx_mas.std()**2 + dy_mas.std()**2)
circle = plt.Circle((0, 0), rms, fill=False, color='gray', linestyle='--', alpha=0.5)
ax.add_patch(circle)

# Initialize elements
trail, = ax.plot([], [], 'o-', color='lightblue', alpha=0.3, markersize=4, linewidth=1)
current_point, = ax.plot([], [], 'o', markersize=20, color='blue')
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace')
title_text = ax.set_title('PS1 Centroid Wobble (Good Data)')

# Filter colors
filter_colors = {'g': 'green', 'i': 'red'}

def init():
    trail.set_data([], [])
    current_point.set_data([], [])
    time_text.set_text('')
    return trail, current_point, time_text

def animate(frame):
    # Show trail of all previous points
    trail.set_data(dx_mas[:frame+1], dy_mas[:frame+1])

    # Current point
    current_point.set_data([dx_mas[frame]], [dy_mas[frame]])
    current_point.set_color(filter_colors.get(filters[frame], 'blue'))

    # Time text
    time_text.set_text(f'MJD {mjd[frame]:.2f}\nt = {t_years[frame]:.2f} yr\n{filters[frame]}-band')

    return trail, current_point, time_text

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(good_meas),
                     interval=300, blit=True, repeat=True)

# Save as GIF
output_file = data_dir.parent / "WOBBLE_FIGURES" / "wobble_timeseries.gif"
writer = PillowWriter(fps=3)
anim.save(output_file, writer=writer, dpi=100)
print(f"Animation saved to: {output_file}")

plt.close()

# Also create a version with interpolated smooth motion
fig2, ax2 = plt.subplots(figsize=(8, 8))

ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_aspect('equal')
ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax2.axvline(0, color='gray', linestyle='-', alpha=0.3)
ax2.set_xlabel('RA offset (mas)')
ax2.set_ylabel('Dec offset (mas)')

# RMS circle
circle2 = plt.Circle((0, 0), rms, fill=False, color='gray', linestyle='--', alpha=0.5)
ax2.add_patch(circle2)

# Plot all points as background
for i, (x, y, f) in enumerate(zip(dx_mas, dy_mas, filters)):
    ax2.plot(x, y, 'o', color=filter_colors.get(f, 'gray'), alpha=0.2, markersize=8)

# Animated elements
star_marker, = ax2.plot([], [], '*', markersize=25, color='gold', markeredgecolor='black', markeredgewidth=1)
time_text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=12,
                      verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.set_title('PS1 Measured Centroid Motion')

def init2():
    star_marker.set_data([], [])
    time_text2.set_text('')
    return star_marker, time_text2

def animate2(frame):
    star_marker.set_data([dx_mas[frame]], [dy_mas[frame]])
    time_text2.set_text(f'Epoch {frame+1}/{len(good_meas)}\nMJD {mjd[frame]:.1f}\n{filters[frame]}-band\ndx={dx_mas[frame]:.1f} mas\ndy={dy_mas[frame]:.1f} mas')
    return star_marker, time_text2

anim2 = FuncAnimation(fig2, animate2, init_func=init2, frames=len(good_meas),
                      interval=400, blit=True, repeat=True)

output_file2 = data_dir.parent / "WOBBLE_FIGURES" / "wobble_star.gif"
anim2.save(output_file2, writer=PillowWriter(fps=2.5), dpi=100)
print(f"Star animation saved to: {output_file2}")

plt.close()
