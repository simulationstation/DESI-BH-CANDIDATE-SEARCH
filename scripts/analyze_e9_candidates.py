#!/usr/bin/env python3
"""Analyze E9 laser candidate results."""

import pandas as pd
import numpy as np

# Load candidates
df = pd.read_csv('data/e9_laser_candidates_sparcl.csv')

print("=" * 60)
print("E9 LASER SEARCH ANALYSIS")
print("=" * 60)

print(f"\nTotal candidates: {len(df)}")
print(f"Unique targets: {df['sparcl_id'].nunique()}")

# Statistics by spectype
print("\n--- By Spectype ---")
for spectype in df['spectype'].unique():
    sub = df[df['spectype'] == spectype]
    print(f"{spectype}: {len(sub)} candidates from {sub['sparcl_id'].nunique()} targets")

# Common wavelengths (likely emission lines)
print("\n--- Most Common Wavelength Ranges ---")
# Bin by 50 Angstrom
df['wave_bin'] = (df['wavelength_peak'] // 50) * 50
common_bins = df.groupby('wave_bin').size().sort_values(ascending=False).head(15)
print("Wavelength range, count:")
for wave_bin, count in common_bins.items():
    print(f"  {wave_bin:.0f}-{wave_bin+50:.0f} Å: {count} detections")

# Known emission lines for comparison
KNOWN_LINES = {
    'H-alpha': 6563,
    'H-beta': 4861,
    '[OIII]_5007': 5007,
    '[OIII]_4959': 4959,
    '[OII]_3727': 3727,
    '[NII]_6583': 6583,
    '[SII]_6717': 6717,
    '[SII]_6731': 6731,
    'Lyman-alpha': 1216,  # observed at higher wavelengths for high-z
}

# Check for lines NOT near known emission wavelengths (at z=0)
# Also check for lines in unusual wavelength ranges
print("\n--- Unusual Candidates (not near z=0 emission lines) ---")

def check_near_line(wave, tolerance=20):
    for name, ref_wave in KNOWN_LINES.items():
        if abs(wave - ref_wave) < tolerance:
            return True, name
    return False, ""

# Find candidates not near any known line
unusual = []
for _, row in df.iterrows():
    is_known, _ = check_near_line(row['wavelength_peak'])
    if not is_known and row['snr'] > 15:
        unusual.append(row)

unusual_df = pd.DataFrame(unusual)
if len(unusual_df) > 0:
    print(f"\nFound {len(unusual_df)} high-SNR candidates NOT near z=0 emission lines")
    print("Top 20 by SNR:")
    top = unusual_df.sort_values('snr', ascending=False).head(20)
    for _, row in top.iterrows():
        print(f"  λ={row['wavelength_peak']:.1f}Å SNR={row['snr']:.1f} "
              f"FWHM={row['fwhm_angstrom']:.2f}Å RA={row['ra']:.4f} DEC={row['dec']:.4f}")
else:
    print("No unusual candidates found")

# Very narrow lines (FWHM < 2 Angstrom) are most laser-like
print("\n--- Ultra-Narrow Lines (FWHM < 2 Angstrom) ---")
narrow = df[df['fwhm_angstrom'] < 2.0].sort_values('snr', ascending=False)
print(f"Found {len(narrow)} ultra-narrow candidates")
for _, row in narrow.head(10).iterrows():
    print(f"  λ={row['wavelength_peak']:.1f}Å SNR={row['snr']:.1f} "
          f"FWHM={row['fwhm_angstrom']:.2f}Å {row['spectype']} "
          f"RA={row['ra']:.4f} DEC={row['dec']:.4f}")

# Save the most interesting candidates
print("\n--- Saving Priority Candidates ---")
priority = df[(df['snr'] > 20) & (df['fwhm_angstrom'] < 3)].copy()
priority = priority.sort_values('snr', ascending=False)
priority.to_csv('data/e9_priority_candidates.csv', index=False)
print(f"Saved {len(priority)} priority candidates to data/e9_priority_candidates.csv")

print("\n" + "=" * 60)
print("NOTE: Most detections are real emission lines from galaxies.")
print("True laser candidates would be:")
print("  - In 'empty' sky fibers (not in SPARCL)")
print("  - Extremely narrow (< 1 Angstrom)")
print("  - At random wavelengths (not matching redshifted emission)")
print("  - Spatially isolated (only in one fiber)")
print("=" * 60)
