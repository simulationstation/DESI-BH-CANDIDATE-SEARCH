#!/usr/bin/env python3
"""
Same-Night Consistency Test for PS1 Astrometry

Applies the paper's methodology: observations taken on the same night
should give consistent centroid measurements. Data that doesn't meet
this criterion is flagged as problematic/systematic.

Key insight: We use RAW PIXEL positions (x_pix, y_pix) since these are
direct measurements. Within a same-night group, we identify outliers
as observations that deviate from the median by >3*MAD.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Load the centroid measurements
data_dir = Path(__file__).parent.parent
results_file = data_dir / "results" / "centroid_measurements.json"

with open(results_file) as f:
    data = json.load(f)

measurements = data["measurements"]

# Convert to DataFrame
df = pd.DataFrame(measurements)

# Extract filter name (remove .00000 suffix)
df['filt'] = df['filter'].str.replace('.00000', '', regex=False)

# Group by integer MJD (night) and filter
df['night'] = df['mjd'].astype(int)

print("=" * 60)
print("SAME-NIGHT CONSISTENCY TEST")
print("=" * 60)

# First pass: identify outliers within each night using pixel positions
df['is_outlier'] = False

for (night, filt), group in df.groupby(['night', 'filt']):
    if len(group) < 2:
        continue

    # Get pixel positions
    x_vals = group['x_pix'].values
    y_vals = group['y_pix'].values

    # Use median and MAD for robust statistics
    x_med = np.median(x_vals)
    y_med = np.median(y_vals)

    x_mad = np.median(np.abs(x_vals - x_med))
    y_mad = np.median(np.abs(y_vals - y_med))

    # Convert MAD to sigma equivalent (MAD * 1.4826 ≈ sigma for Gaussian)
    x_sigma = x_mad * 1.4826 if x_mad > 0 else np.std(x_vals)
    y_sigma = y_mad * 1.4826 if y_mad > 0 else np.std(y_vals)

    # Use a minimum threshold of 0.1 pixels (~25 mas) to avoid flagging everything
    x_sigma = max(x_sigma, 0.1)
    y_sigma = max(y_sigma, 0.1)

    # Flag outliers: >3 sigma from median
    for idx in group.index:
        x_dev = abs(df.loc[idx, 'x_pix'] - x_med) / x_sigma
        y_dev = abs(df.loc[idx, 'y_pix'] - y_med) / y_sigma
        if x_dev > 3 or y_dev > 3:
            df.loc[idx, 'is_outlier'] = True

# Report outliers
outliers = df[df['is_outlier']]
print(f"\nFound {len(outliers)} outliers using same-night consistency:")
for _, row in outliers.iterrows():
    print(f"  MJD {row['mjd']:.4f} ({row['filt']}): x_pix={row['x_pix']:.2f}, y_pix={row['y_pix']:.2f}")

# Now recalculate same-night statistics WITHOUT outliers
df_clean = df[~df['is_outlier']].copy()

night_results = []

for (night, filt), group in df_clean.groupby(['night', 'filt']):
    if len(group) < 2:
        continue

    # Calculate intra-night scatter in pixels and mas
    x_scatter_pix = group['x_pix'].std()
    y_scatter_pix = group['y_pix'].std()

    # Convert to mas (0.25 arcsec/pixel = 250 mas/pixel)
    x_scatter_mas = x_scatter_pix * 250
    y_scatter_mas = y_scatter_pix * 250
    total_scatter_mas = np.sqrt(x_scatter_mas**2 + y_scatter_mas**2)

    # Time span in minutes
    dt_min = (group['mjd'].max() - group['mjd'].min()) * 24 * 60

    night_results.append({
        'night': int(night),
        'filter': filt,
        'n_obs': len(group),
        'dt_min': float(dt_min),
        'x_scatter_mas': float(x_scatter_mas),
        'y_scatter_mas': float(y_scatter_mas),
        'total_scatter_mas': float(total_scatter_mas),
        'indices': [int(i) for i in group['index'].values]
    })

night_df = pd.DataFrame(night_results)

print(f"\n{len(night_df)} same-night groups after outlier removal:")
print()

# Sort by total scatter
night_df_sorted = night_df.sort_values('total_scatter_mas')

# Define threshold for "good" data
CONSISTENCY_THRESHOLD_MAS = 50.0

print(f"Consistency threshold: {CONSISTENCY_THRESHOLD_MAS} mas")
print()

for _, row in night_df_sorted.iterrows():
    status = "GOOD" if row['total_scatter_mas'] < CONSISTENCY_THRESHOLD_MAS else "BAD"
    print(f"Night {row['night']} ({row['filter']}): {row['n_obs']} obs over {row['dt_min']:.1f} min")
    print(f"  Scatter: {row['total_scatter_mas']:.1f} mas (X: {row['x_scatter_mas']:.1f}, Y: {row['y_scatter_mas']:.1f})")
    print(f"  Status: {status}")
    print()

# Identify good and bad observations
good_indices = set()
bad_indices = set()

for _, row in night_df.iterrows():
    if row['total_scatter_mas'] < CONSISTENCY_THRESHOLD_MAS:
        good_indices.update(row['indices'])
    else:
        bad_indices.update(row['indices'])

# Outliers are automatically "bad"
outlier_indices = set(df[df['is_outlier']]['index'].values)
bad_indices.update(outlier_indices)
good_indices -= outlier_indices

# Observations without same-night pairs are "unknown"
all_indices = set(df['index'].values)
paired_indices = good_indices | bad_indices
unpaired_indices = all_indices - paired_indices

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Good (consistent same-night pairs): {len(good_indices)} observations")
print(f"Bad (outliers + inconsistent pairs): {len(bad_indices)} observations")
print(f"Unknown (no same-night pair): {len(unpaired_indices)} observations")

# Create filtered dataset with only "good" observations
df_good = df[df['index'].isin(good_indices)].copy()

# Recalculate centroid offsets relative to mean of GOOD data only
if len(df_good) > 0:
    x_mean = df_good['x_pix'].mean()
    y_mean = df_good['y_pix'].mean()

    df_good['dx_mas'] = (df_good['x_pix'] - x_mean) * 250
    df_good['dy_mas'] = (df_good['y_pix'] - y_mean) * 250

    print(f"\n'Good' observations (offsets relative to good-data mean):")
    for _, row in df_good.iterrows():
        print(f"  MJD {row['mjd']:.4f} ({row['filt']}): dx={row['dx_mas']:.1f} mas, dy={row['dy_mas']:.1f} mas")

    # Calculate RMS with only good data
    x_rms_good = df_good['dx_mas'].std()
    y_rms_good = df_good['dy_mas'].std()
    total_rms_good = np.sqrt(x_rms_good**2 + y_rms_good**2)

    print(f"\nRMS with only 'good' data ({len(df_good)} observations):")
    print(f"  X: {x_rms_good:.1f} mas")
    print(f"  Y: {y_rms_good:.1f} mas")
    print(f"  Total: {total_rms_good:.1f} mas")
else:
    x_rms_good = None
    y_rms_good = None
    total_rms_good = None

# Save results
output = {
    'consistency_threshold_mas': CONSISTENCY_THRESHOLD_MAS,
    'night_groups': night_results,
    'good_indices': [int(i) for i in good_indices],
    'bad_indices': [int(i) for i in bad_indices],
    'unpaired_indices': [int(i) for i in unpaired_indices],
    'outlier_indices': [int(i) for i in outlier_indices],
    'rms_good_data': {
        'x_mas': float(x_rms_good) if x_rms_good is not None else None,
        'y_mas': float(y_rms_good) if y_rms_good is not None else None,
        'total_mas': float(total_rms_good) if total_rms_good is not None else None,
        'n_obs': len(df_good)
    }
}

output_file = data_dir / "results" / "same_night_consistency.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_file}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: scatter plot by night group
ax1 = axes[0]
colors = {'g': 'green', 'i': 'red'}
for _, row in night_df.iterrows():
    c = colors.get(row['filter'], 'gray')
    marker = 'o' if row['total_scatter_mas'] < CONSISTENCY_THRESHOLD_MAS else 'x'
    ax1.scatter(row['n_obs'], row['total_scatter_mas'], c=c, marker=marker, s=100)

ax1.axhline(CONSISTENCY_THRESHOLD_MAS, color='k', linestyle='--', label=f'Threshold = {CONSISTENCY_THRESHOLD_MAS} mas')
ax1.set_xlabel('Number of observations in same-night group')
ax1.set_ylabel('Intra-night scatter (mas)')
ax1.set_title('Same-Night Consistency Test (after outlier removal)')
ax1.legend()

# Right: centroid positions colored by status
ax2 = axes[1]

# Calculate offsets relative to good-data mean for all observations
if len(df_good) > 0:
    x_ref = df_good['x_pix'].mean()
    y_ref = df_good['y_pix'].mean()
else:
    x_ref = df['x_pix'].mean()
    y_ref = df['y_pix'].mean()

df['dx_plot'] = (df['x_pix'] - x_ref) * 250
df['dy_plot'] = (df['y_pix'] - y_ref) * 250

# Plot bad observations
df_bad = df[df['index'].isin(bad_indices)]
if len(df_bad) > 0:
    ax2.scatter(df_bad['dx_plot'], df_bad['dy_plot'],
                c='red', marker='x', s=50, alpha=0.7, label=f'Bad/outlier ({len(df_bad)})')

# Plot unpaired observations
df_unpaired = df[df['index'].isin(unpaired_indices)]
if len(df_unpaired) > 0:
    ax2.scatter(df_unpaired['dx_plot'], df_unpaired['dy_plot'],
                c='gray', marker='s', s=50, alpha=0.5, label=f'No pair ({len(df_unpaired)})')

# Plot good observations
if len(df_good) > 0:
    ax2.scatter(df_good['dx_mas'], df_good['dy_mas'],
                c='green', marker='o', s=80, label=f'Good ({len(df_good)})')

ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(0, color='k', linestyle='-', alpha=0.3)
ax2.set_xlabel('RA offset (mas)')
ax2.set_ylabel('Dec offset (mas)')
ax2.set_title('Centroid Positions by Quality')
ax2.legend()
ax2.set_aspect('equal')

# Limit axes to show good data region
if len(df_good) > 0:
    xlim = max(abs(df_good['dx_mas'].max()), abs(df_good['dx_mas'].min())) * 1.5
    ylim = max(abs(df_good['dy_mas'].max()), abs(df_good['dy_mas'].min())) * 1.5
    lim = max(xlim, ylim, 100)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)

plt.tight_layout()

fig_file = data_dir.parent / "WOBBLE_FIGURES" / "same_night_consistency.png"
plt.savefig(fig_file, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {fig_file}")

plt.close()

# Also create a zoomed-in version showing only the good data
if len(df_good) > 0:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by filter
    g_mask = df_good['filt'] == 'g'
    i_mask = df_good['filt'] == 'i'

    if g_mask.any():
        ax.scatter(df_good.loc[g_mask, 'dx_mas'], df_good.loc[g_mask, 'dy_mas'],
                   c='green', marker='o', s=80, alpha=0.7, label=f'g-band ({g_mask.sum()})')
    if i_mask.any():
        ax.scatter(df_good.loc[i_mask, 'dx_mas'], df_good.loc[i_mask, 'dy_mas'],
                   c='red', marker='o', s=80, alpha=0.7, label=f'i-band ({i_mask.sum()})')

    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('RA offset (mas)')
    ax.set_ylabel('Dec offset (mas)')
    ax.set_title(f"'Good' Centroid Positions ({len(df_good)} obs, RMS = {total_rms_good:.1f} mas)")
    ax.legend()
    ax.set_aspect('equal')

    # Set symmetric limits
    lim = max(abs(df_good['dx_mas'].max()), abs(df_good['dx_mas'].min()),
              abs(df_good['dy_mas'].max()), abs(df_good['dy_mas'].min())) * 1.3
    lim = max(lim, 100)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.tight_layout()
    fig_file2 = data_dir.parent / "WOBBLE_FIGURES" / "same_night_good_data.png"
    plt.savefig(fig_file2, dpi=150, bbox_inches='tight')
    print(f"Zoomed figure saved to: {fig_file2}")
    plt.close()
