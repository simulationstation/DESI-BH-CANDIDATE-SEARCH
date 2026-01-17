#!/usr/bin/env python3
"""
E9 PHASE 2: Forensic Verification of ALL Laser Candidates
Processes unprocessed candidates regardless of spectype (STAR or GALAXY)
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from sparcl.client import SparclClient
import sys
import time

def classify_profile(wave, flux, center_wave, width_ang=5):
    """Classify as COSMIC_RAY or REAL_SIGNAL based on profile shape."""
    mask = (wave > center_wave - width_ang) & (wave < center_wave + width_ang)
    w_region = wave[mask]
    f_region = flux[mask]

    if len(f_region) < 3:
        return "UNKNOWN", "Insufficient data", 0.0

    peak_idx = np.argmax(f_region)
    peak_flux = f_region[peak_idx]

    left_flux = f_region[peak_idx-1] if peak_idx > 0 else 0
    right_flux = f_region[peak_idx+1] if peak_idx < len(f_region)-1 else 0

    bg = np.median(f_region)
    peak_above_bg = peak_flux - bg
    left_above_bg = left_flux - bg
    right_above_bg = right_flux - bg

    if peak_above_bg <= 0:
        return "UNKNOWN", "No significant peak", 0.0

    left_ratio = left_above_bg / peak_above_bg if peak_above_bg > 0 else 0
    right_ratio = right_above_bg / peak_above_bg if peak_above_bg > 0 else 0
    wing_avg = (left_ratio + right_ratio) / 2

    if wing_avg < 0.1:
        return "COSMIC_RAY", f"Sharp spike (wing ratio: {wing_avg:.2f})", wing_avg
    elif wing_avg > 0.2:
        return "REAL_SIGNAL", f"Gaussian wings (wing ratio: {wing_avg:.2f})", wing_avg
    else:
        return "AMBIGUOUS", f"Intermediate profile (wing ratio: {wing_avg:.2f})", wing_avg

# Load data
print("Loading candidate data...")
priority = pd.read_csv('data/e9_priority_candidates.csv')
existing = pd.read_csv('data/e9_forensic_results.csv')

# Create keys for matching
priority['key'] = priority['targetid'].astype(str) + '_' + priority['wavelength_peak'].round(1).astype(str)
existing['key'] = existing['targetid'].astype(str) + '_' + existing['wavelength'].round(1).astype(str)

processed_keys = set(existing['key'])
unprocessed = priority[~priority['key'].isin(processed_keys)].copy()

print(f"Total priority candidates: {len(priority)}")
print(f"Already processed: {len(existing)}")
print(f"Remaining to process: {len(unprocessed)}")

if len(unprocessed) == 0:
    print("All candidates already processed!")
    sys.exit(0)

# Connect to SPARCL
print("\nConnecting to SPARCL...")
client = SparclClient()

print("\n" + "="*60)
print("E9 PHASE 2: FORENSIC VERIFICATION (ALL CANDIDATES)")
print("="*60)

results = []
errors = 0

for i, (idx, row) in enumerate(unprocessed.iterrows()):
    tid = row['targetid']
    wave_peak = row['wavelength_peak']
    spectype = row['spectype']

    print(f"\n[{i+1}/{len(unprocessed)}] TARGET: {tid} ({spectype})")
    print(f"  Signal: {row['snr']:.1f} sigma at {wave_peak:.1f} A")

    try:
        res = client.retrieve(uuid_list=[row['sparcl_id']], include=['flux', 'wavelength', 'ivar', 'mask'])
        spec = res.records[0]

        flux = np.array(spec['flux'])
        wave = np.array(spec['wavelength'])
        mask_arr = np.array(spec['mask']) if 'mask' in spec else np.zeros_like(flux)

        idx_peak = (np.abs(wave - wave_peak)).argmin()
        pixel_mask = mask_arr[idx_peak]
        flagged = pixel_mask > 0

        verdict, reason, wing_ratio = classify_profile(wave, flux, wave_peak)

        status = "FLAGGED" if flagged else "CLEAN"
        print(f"  [{status}] VERDICT: {verdict} - {reason}")

        results.append({
            'targetid': tid,
            'wavelength': wave_peak,
            'snr': row['snr'],
            'ra': row['ra'],
            'dec': row['dec'],
            'flagged': flagged,
            'verdict': verdict,
            'reason': reason,
            'spectype': spectype,
            'wing_ratio': wing_ratio
        })

        # Rate limit
        time.sleep(0.1)

    except Exception as e:
        print(f"  ERROR: {e}")
        errors += 1
        results.append({
            'targetid': tid,
            'wavelength': wave_peak,
            'snr': row['snr'],
            'ra': row['ra'],
            'dec': row['dec'],
            'flagged': None,
            'verdict': 'ERROR',
            'reason': str(e),
            'spectype': spectype,
            'wing_ratio': None
        })

# Combine with existing results
new_results = pd.DataFrame(results)

# Add missing columns to existing if needed
for col in ['spectype', 'wing_ratio']:
    if col not in existing.columns:
        existing[col] = None

combined = pd.concat([existing, new_results[existing.columns.tolist()]], ignore_index=True)
combined.to_csv('data/e9_forensic_results.csv', index=False)

# Summary
print("\n" + "="*60)
print("FORENSICS SUMMARY")
print("="*60)

print(f"\nNewly processed: {len(new_results)}")
print(f"Errors: {errors}")

new_cr = new_results[new_results['verdict'] == 'COSMIC_RAY']
new_real = new_results[new_results['verdict'] == 'REAL_SIGNAL']
new_amb = new_results[new_results['verdict'] == 'AMBIGUOUS']

print(f"\nNew results breakdown:")
print(f"  COSMIC_RAY:   {len(new_cr)}")
print(f"  REAL_SIGNAL:  {len(new_real)}")
print(f"  AMBIGUOUS:    {len(new_amb)}")
print(f"  ERROR/UNKNOWN: {len(new_results) - len(new_cr) - len(new_real) - len(new_amb)}")

if len(new_real) > 0:
    print("\n*** NEW REAL SIGNAL CANDIDATES ***")
    for _, r in new_real.iterrows():
        print(f"  {r['wavelength']:.1f} A | SNR {r['snr']:.1f} | {r['spectype']}")
        print(f"    RA {r['ra']:.4f} DEC {r['dec']:.4f}")

if len(new_amb) > 0:
    print("\n*** NEW AMBIGUOUS CANDIDATES ***")
    for _, r in new_amb.iterrows():
        print(f"  {r['wavelength']:.1f} A | SNR {r['snr']:.1f} | {r['spectype']} | {r['reason']}")

# Full combined summary
print("\n" + "="*60)
print("COMBINED TOTALS (all processed)")
print("="*60)
all_cr = combined[combined['verdict'] == 'COSMIC_RAY']
all_real = combined[combined['verdict'] == 'REAL_SIGNAL']
all_amb = combined[combined['verdict'] == 'AMBIGUOUS']

print(f"Total analyzed:   {len(combined)}")
print(f"  COSMIC_RAY:     {len(all_cr)}")
print(f"  REAL_SIGNAL:    {len(all_real)}")
print(f"  AMBIGUOUS:      {len(all_amb)}")

print(f"\nResults saved to data/e9_forensic_results.csv")
print("="*60)
