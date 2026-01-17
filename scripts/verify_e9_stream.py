#!/usr/bin/env python3
"""
E9 FORENSIC VERIFICATION - Streaming Laser Candidates

Analyzes top candidates from e9_stream_candidates.csv
Distinguishes Cosmic Rays (sharp spike) from Real Signals (Gaussian wings)
"""

import pandas as pd
import numpy as np
from sparcl.client import SparclClient
import warnings
warnings.filterwarnings('ignore')

# Known airglow/atmospheric lines to flag
AIRGLOW_LINES = [5577.34, 5889.95, 5895.92, 6300.30, 6363.78, 6533.04,
    6863.96, 7316.29, 7340.89, 7571.75, 7750.64, 7794.11,
    7913.71, 7993.33, 8344.60, 8399.17, 8430.17, 8827.10, 8885.85]

# H-alpha and other common emission
ASTROPHYSICAL_LINES = [6562.8, 6548.0, 6583.5, 4861.3, 4959.0, 5007.0]

def is_known_line(wavelength, tolerance=5.0):
    """Check if wavelength matches known atmospheric or astrophysical line."""
    for line in AIRGLOW_LINES:
        if abs(wavelength - line) < tolerance:
            return f"AIRGLOW ({line:.1f})"
    for line in ASTROPHYSICAL_LINES:
        if abs(wavelength - line) < tolerance:
            return f"ASTROPHYSICAL ({line:.1f})"
    return None

def classify_profile(wave, flux, center_wave, width_ang=5):
    """Classify as COSMIC_RAY or REAL_SIGNAL based on profile shape."""
    mask = (wave > center_wave - width_ang) & (wave < center_wave + width_ang)
    w_region = wave[mask]
    f_region = flux[mask]

    if len(f_region) < 5:
        return "UNKNOWN", "Insufficient data points"

    # Find peak
    peak_idx = np.argmax(f_region)
    peak_flux = f_region[peak_idx]

    # Check neighbors (1 and 2 pixels away)
    left1 = f_region[peak_idx-1] if peak_idx > 0 else 0
    right1 = f_region[peak_idx+1] if peak_idx < len(f_region)-1 else 0
    left2 = f_region[peak_idx-2] if peak_idx > 1 else 0
    right2 = f_region[peak_idx+2] if peak_idx < len(f_region)-2 else 0

    # Background level (edges of region)
    bg = np.median(np.concatenate([f_region[:3], f_region[-3:]]))

    # Calculate wing ratios
    peak_above_bg = peak_flux - bg
    if peak_above_bg <= 0:
        return "UNKNOWN", "No significant peak above background"

    # Immediate neighbors
    wing1_avg = ((left1 - bg) + (right1 - bg)) / 2 / peak_above_bg
    # Second neighbors
    wing2_avg = ((left2 - bg) + (right2 - bg)) / 2 / peak_above_bg

    # Classification logic:
    # Cosmic ray: very sharp, immediate neighbors < 15% of peak
    # Real signal: Gaussian wings, immediate neighbors > 25% of peak
    # Also check second neighbors for extended profile

    if wing1_avg < 0.15:
        return "COSMIC_RAY", f"Sharp spike (wing1={wing1_avg:.2f}, wing2={wing2_avg:.2f})"
    elif wing1_avg > 0.25 and wing2_avg > 0.05:
        return "REAL_SIGNAL", f"Gaussian profile (wing1={wing1_avg:.2f}, wing2={wing2_avg:.2f})"
    else:
        return "AMBIGUOUS", f"Intermediate (wing1={wing1_avg:.2f}, wing2={wing2_avg:.2f})"

def ascii_profile(wave, flux, center_wave, width_ang=10):
    """Create ASCII visualization of the peak region."""
    mask = (wave > center_wave - width_ang) & (wave < center_wave + width_ang)
    w_region = wave[mask]
    f_region = flux[mask]

    if len(f_region) == 0:
        return

    f_min, f_max = np.min(f_region), np.max(f_region)
    if f_max == f_min:
        print("    [Flat region]")
        return

    print(f"    Profile ({center_wave-width_ang:.0f} - {center_wave+width_ang:.0f} A):")
    for w, f in zip(w_region[::2], f_region[::2]):  # Every other point for compactness
        norm = (f - f_min) / (f_max - f_min)
        bar = "#" * int(norm * 30)
        marker = " <-- PEAK" if f == f_max else ""
        print(f"    {w:7.1f} |{bar}{marker}")

def main():
    print("=" * 60)
    print("E9 FORENSIC VERIFICATION - Streaming Laser Candidates")
    print("=" * 60)

    # Load candidates
    df = pd.read_csv('laser_candidates/e9_stream_candidates.csv')
    print(f"Total candidates: {len(df)}")

    # Sort by SNR and take top 50 for analysis
    top_n = 50
    top_cands = df.nlargest(top_n, 'snr').copy()
    print(f"Analyzing top {len(top_cands)} by SNR")

    # Connect to SPARCL
    print("\nConnecting to SPARCL...")
    client = SparclClient()

    results = []

    for i, row in top_cands.iterrows():
        tid = row['targetid']
        wave_peak = row['wavelength']
        snr = row['snr']
        ra, dec = row['ra'], row['dec']
        tile = row['tile']

        print(f"\n{'='*60}")
        print(f"[{len(results)+1}/{len(top_cands)}] TARGET: {tid}")
        print(f"  Signal: {snr:.1f}σ at {wave_peak:.1f} Å")
        print(f"  Position: RA={ra:.4f}, DEC={dec:.4f}")
        print(f"  Tile: {tile}")

        # Check for known lines first
        known = is_known_line(wave_peak)
        if known:
            print(f"  [!] KNOWN LINE: {known}")
            results.append({
                'targetid': tid, 'wavelength': wave_peak, 'snr': snr,
                'ra': ra, 'dec': dec, 'tile': tile,
                'verdict': 'KNOWN_LINE', 'reason': known
            })
            continue

        # Try to retrieve spectrum from SPARCL
        try:
            # Search by targetid
            found = client.find(outfields=['sparcl_id', 'targetid', 'ra', 'dec'],
                               constraints={'targetid': [int(tid)]})

            if len(found.records) == 0:
                # Try coordinate search
                found = client.find(outfields=['sparcl_id', 'targetid', 'ra', 'dec'],
                                   constraints={'ra': [ra-0.01, ra+0.01],
                                               'dec': [dec-0.01, dec+0.01]})

            if len(found.records) == 0:
                print(f"  [!] Not found in SPARCL")
                results.append({
                    'targetid': tid, 'wavelength': wave_peak, 'snr': snr,
                    'ra': ra, 'dec': dec, 'tile': tile,
                    'verdict': 'NOT_IN_SPARCL', 'reason': 'Target not found'
                })
                continue

            sparcl_id = found.records[0]['sparcl_id']

            # Retrieve spectrum
            spec = client.retrieve(uuid_list=[sparcl_id],
                                  include=['flux', 'wavelength', 'mask'])
            rec = spec.records[0]

            flux = np.array(rec['flux'])
            wave = np.array(rec['wavelength'])
            mask_arr = np.array(rec.get('mask', np.zeros_like(flux)))

            # Check if peak pixel is flagged
            idx = np.argmin(np.abs(wave - wave_peak))
            pixel_mask = mask_arr[idx] if idx < len(mask_arr) else 0

            if pixel_mask > 0:
                print(f"  [!] FLAGGED by pipeline (mask={pixel_mask})")
            else:
                print(f"  [+] Not flagged by pipeline")

            # Classify profile
            verdict, reason = classify_profile(wave, flux, wave_peak)
            print(f"  >>> VERDICT: {verdict}")
            print(f"      {reason}")

            # Show profile
            ascii_profile(wave, flux, wave_peak)

            results.append({
                'targetid': tid, 'wavelength': wave_peak, 'snr': snr,
                'ra': ra, 'dec': dec, 'tile': tile,
                'verdict': verdict, 'reason': reason,
                'flagged': pixel_mask > 0
            })

        except Exception as e:
            print(f"  [!] Error: {e}")
            results.append({
                'targetid': tid, 'wavelength': wave_peak, 'snr': snr,
                'ra': ra, 'dec': dec, 'tile': tile,
                'verdict': 'ERROR', 'reason': str(e)[:50]
            })

    # Summary
    print("\n" + "=" * 60)
    print("FORENSICS SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(results)

    verdicts = results_df['verdict'].value_counts()
    print(f"\nTotal analyzed: {len(results_df)}")
    for v, count in verdicts.items():
        print(f"  {v}: {count}")

    # Show real signals
    real = results_df[results_df['verdict'] == 'REAL_SIGNAL']
    if len(real) > 0:
        print(f"\n*** REAL SIGNAL CANDIDATES ({len(real)}) ***")
        for _, r in real.iterrows():
            print(f"  {r['wavelength']:.1f} Å | SNR {r['snr']:.1f} | RA {r['ra']:.4f} DEC {r['dec']:.4f}")
            print(f"    https://www.legacysurvey.org/viewer?ra={r['ra']}&dec={r['dec']}&layer=ls-dr9&zoom=16")

    # Show ambiguous
    ambig = results_df[results_df['verdict'] == 'AMBIGUOUS']
    if len(ambig) > 0:
        print(f"\n*** AMBIGUOUS ({len(ambig)}) - Need manual review ***")
        for _, r in ambig.iterrows():
            print(f"  {r['wavelength']:.1f} Å | SNR {r['snr']:.1f} | {r['reason']}")

    # Save
    results_df.to_csv('laser_candidates/e9_forensic_results.csv', index=False)
    print(f"\nResults saved to laser_candidates/e9_forensic_results.csv")

    print("\n" + "=" * 60)
    print("FORENSICS COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
