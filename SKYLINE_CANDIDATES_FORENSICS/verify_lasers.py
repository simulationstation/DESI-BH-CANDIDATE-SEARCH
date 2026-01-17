#!/usr/bin/env python3
"""
E9 PHASE 2: Forensic Verification of Laser Candidates

Distinguishes Cosmic Rays (needle profile) from Real Signals (Gaussian profile)
"""

import pandas as pd
import numpy as np
from sparcl.client import SparclClient
import sys

# 1. Connect and Load
print("Connecting to SPARCL...")
client = SparclClient()

try:
    df = pd.read_csv('data/e9_priority_candidates.csv')
    # Filter for objects classified as STAR
    stars = df[df['spectype'] == 'STAR'].copy()
    print(f"Found {len(stars)} STAR candidates to investigate.")
except FileNotFoundError:
    print("Error: e9_priority_candidates.csv not found.")
    sys.exit(1)

# 2. Visualization Function
def ascii_plot(wave, flux, center_wave, width_ang=20):
    # Slice the region around the laser line
    mask = (wave > center_wave - width_ang) & (wave < center_wave + width_ang)
    w_region = wave[mask]
    f_region = flux[mask]

    if len(f_region) == 0:
        return "No data in region"

    # Normalize for plotting
    f_min = np.min(f_region)
    f_max = np.max(f_region)
    if f_max == f_min:
        return "Flat line"

    print(f"\n   Zoom on {center_wave:.1f} A (Range: {f_min:.1f} to {f_max:.1f}):")
    print("   " + "-"*40)

    for w, f in zip(w_region, f_region):
        # Determine bar length (0 to 40 chars)
        norm_val = (f - f_min) / (f_max - f_min + 1e-9)
        bar_len = int(norm_val * 40)

        # ASCII Graphics
        row_str = "#" * bar_len
        # highlight the peak
        if f == f_max:
            row_str += f"  <-- PEAK ({f:.1f})"

        print(f"   {w:.1f} : {row_str}")
    print("   " + "-"*40)

def classify_profile(wave, flux, center_wave, width_ang=5):
    """Classify as COSMIC_RAY or REAL_SIGNAL based on profile shape."""
    mask = (wave > center_wave - width_ang) & (wave < center_wave + width_ang)
    w_region = wave[mask]
    f_region = flux[mask]

    if len(f_region) < 3:
        return "UNKNOWN", "Insufficient data"

    # Find peak
    peak_idx = np.argmax(f_region)
    peak_flux = f_region[peak_idx]

    # Check neighbors
    left_flux = f_region[peak_idx-1] if peak_idx > 0 else 0
    right_flux = f_region[peak_idx+1] if peak_idx < len(f_region)-1 else 0

    # Background level
    bg = np.median(f_region)

    # Calculate wing ratios
    peak_above_bg = peak_flux - bg
    left_above_bg = left_flux - bg
    right_above_bg = right_flux - bg

    if peak_above_bg <= 0:
        return "UNKNOWN", "No significant peak"

    left_ratio = left_above_bg / peak_above_bg if peak_above_bg > 0 else 0
    right_ratio = right_above_bg / peak_above_bg if peak_above_bg > 0 else 0

    # Classification logic:
    # Cosmic ray: very sharp, neighbors < 10% of peak
    # Real signal: Gaussian wings, neighbors > 20% of peak
    wing_avg = (left_ratio + right_ratio) / 2

    if wing_avg < 0.1:
        return "COSMIC_RAY", f"Sharp spike (wing ratio: {wing_avg:.2f})"
    elif wing_avg > 0.2:
        return "REAL_SIGNAL", f"Gaussian wings (wing ratio: {wing_avg:.2f})"
    else:
        return "AMBIGUOUS", f"Intermediate profile (wing ratio: {wing_avg:.2f})"

# 3. Execution Loop
print("\n" + "="*60)
print("E9 PHASE 2: FORENSIC VERIFICATION")
print("="*60)

results = []

for i, row in stars.iterrows():
    tid = row['targetid']
    wave_peak = row['wavelength_peak']
    print(f"\n{'='*60}")
    print(f"TARGET: {tid}")
    print(f"  > Signal: {row['snr']:.1f} sigma at {wave_peak:.1f} A")
    print(f"  > FWHM: {row['fwhm_angstrom']:.2f} A")
    print(f"  > RA: {row['ra']:.4f}, DEC: {row['dec']:.4f}")

    try:
        # Retrieve full spectrum
        res = client.retrieve(uuid_list=[row['sparcl_id']], include=['flux', 'wavelength', 'ivar', 'mask'])
        spec = res.records[0]

        flux = np.array(spec['flux'])
        wave = np.array(spec['wavelength'])
        mask_arr = np.array(spec['mask']) if 'mask' in spec else np.zeros_like(flux)

        # Check DESI pipeline flags
        idx = (np.abs(wave - wave_peak)).argmin()
        pixel_mask = mask_arr[idx]

        if pixel_mask > 0:
             print(f"    [!] FLAGGED: DESI Pipeline marked this pixel (Mask Val: {pixel_mask})")
             flagged = True
        else:
             print("    [+] CLEAN: Pipeline did not flag this.")
             flagged = False

        # Classify the profile
        verdict, reason = classify_profile(wave, flux, wave_peak)
        print(f"    >>> VERDICT: {verdict} - {reason}")

        # Visualize
        ascii_plot(wave, flux, wave_peak)

        print(f"  > Legacy Viewer: https://www.legacysurvey.org/viewer?ra={row['ra']}&dec={row['dec']}&layer=ls-dr9&zoom=16")

        results.append({
            'targetid': tid,
            'wavelength': wave_peak,
            'snr': row['snr'],
            'ra': row['ra'],
            'dec': row['dec'],
            'flagged': flagged,
            'verdict': verdict,
            'reason': reason
        })

    except Exception as e:
        print(f"  Error retrieving: {e}")
        results.append({
            'targetid': tid,
            'wavelength': wave_peak,
            'snr': row['snr'],
            'ra': row['ra'],
            'dec': row['dec'],
            'flagged': None,
            'verdict': 'ERROR',
            'reason': str(e)
        })

# Summary
print("\n" + "="*60)
print("FORENSICS SUMMARY")
print("="*60)

results_df = pd.DataFrame(results)
cosmic_rays = results_df[results_df['verdict'] == 'COSMIC_RAY']
real_signals = results_df[results_df['verdict'] == 'REAL_SIGNAL']
ambiguous = results_df[results_df['verdict'] == 'AMBIGUOUS']

print(f"\nTotal STAR candidates analyzed: {len(results_df)}")
print(f"  COSMIC_RAY (rejected):  {len(cosmic_rays)}")
print(f"  REAL_SIGNAL (keep):     {len(real_signals)}")
print(f"  AMBIGUOUS:              {len(ambiguous)}")

if len(real_signals) > 0:
    print("\n*** REAL SIGNAL CANDIDATES ***")
    for _, r in real_signals.iterrows():
        print(f"  {r['wavelength']:.1f} A | SNR {r['snr']:.1f} | RA {r['ra']:.4f} DEC {r['dec']:.4f}")
        print(f"    https://www.legacysurvey.org/viewer?ra={r['ra']}&dec={r['dec']}&layer=ls-dr9&zoom=16")

if len(ambiguous) > 0:
    print("\n*** AMBIGUOUS - NEEDS FURTHER INVESTIGATION ***")
    for _, r in ambiguous.iterrows():
        print(f"  {r['wavelength']:.1f} A | SNR {r['snr']:.1f} | {r['reason']}")

# Save results
results_df.to_csv('data/e9_forensic_results.csv', index=False)
print(f"\nResults saved to data/e9_forensic_results.csv")

print("\n" + "="*60)
print("FORENSICS COMPLETE")
print("="*60)
