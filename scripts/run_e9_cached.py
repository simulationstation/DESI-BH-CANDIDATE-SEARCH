#!/usr/bin/env python3
"""
E9 LASER SEARCH - PROCESS CACHED FILES ONLY

Simple, fast processing of already-downloaded DESI files.
"""

import os
import sys
import glob
import numpy as np
from astropy.io import fits
from multiprocessing import Pool
import gc
from pathlib import Path
from datetime import datetime
import json
import csv

# Configuration
N_WORKERS = 8  # Can use more workers since files are already cached
SIGMA_THRESHOLD = 8.0
MAX_FWHM_PIXELS = 4.0
REJECTION_WINDOW = 2.5

OUTPUT_DIR = Path("data/e9_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Airglow lines
AIRGLOW = {
    5577.3, 6300.3, 6363.8, 5895.9, 5889.9, 4358.3, 5460.7,
    6834.0, 6871.0, 6950.0, 7244.0, 7316.0, 7524.0, 7621.0,
    7750.0, 7853.0, 7964.0, 8101.0, 8344.0, 8505.0, 8696.0, 8827.0
}


def is_airglow(w):
    for line in AIRGLOW:
        if abs(w - line) < REJECTION_WINDOW:
            return True
    return False


def process_file(filepath):
    """Process one FITS file."""
    candidates = []

    try:
        with fits.open(filepath, memmap=True) as hdu:
            fibermap = hdu['FIBERMAP'].data

            # Get sky fibers
            if 'OBJTYPE' in fibermap.dtype.names:
                sky_idx = np.where(fibermap['OBJTYPE'] == 'SKY')[0]
            else:
                sky_idx = np.where(fibermap['TARGETID'] < 0)[0]

            if len(sky_idx) < 3:
                return []

            # Process each arm
            for arm in ['B', 'R', 'Z']:
                if f'{arm}_WAVELENGTH' not in [h.name for h in hdu]:
                    continue

                wave = np.array(hdu[f'{arm}_WAVELENGTH'].data)
                flux = hdu[f'{arm}_FLUX'].data[sky_idx].copy()
                ivar = hdu[f'{arm}_IVAR'].data[sky_idx].copy()

                # Median sky
                median = np.nanmedian(flux, axis=0)
                mad = np.nanmedian(np.abs(flux - median), axis=0) * 1.4826

                # Search each fiber
                for i, fib in enumerate(sky_idx):
                    residual = flux[i] - median

                    with np.errstate(divide='ignore', invalid='ignore'):
                        sigma = np.where(ivar[i] > 0, 1.0/np.sqrt(ivar[i]), np.inf)
                        noise = np.sqrt(mad**2 + sigma**2)
                        snr = residual / noise

                    # Find peaks
                    peaks = np.where(snr > SIGMA_THRESHOLD)[0]

                    for pk in peaks:
                        if pk < 5 or pk > len(wave) - 5:
                            continue

                        w = wave[pk]
                        s = snr[pk]

                        # FWHM check
                        fwhm = 1
                        for off in range(1, 5):
                            if pk-off >= 0 and snr[pk-off] > SIGMA_THRESHOLD * 0.5:
                                fwhm += 1
                            if pk+off < len(snr) and snr[pk+off] > SIGMA_THRESHOLD * 0.5:
                                fwhm += 1

                        if fwhm > MAX_FWHM_PIXELS:
                            continue

                        # Neighbors check
                        n_neighbors = sum(1 for j in range(len(flux))
                                        if j != i and
                                        (flux[j,pk] - median[pk]) / noise[pk] > SIGMA_THRESHOLD * 0.5)

                        is_ag = is_airglow(w)
                        isolated = n_neighbors <= 1
                        passed = not is_ag and isolated

                        info = fibermap[fib]
                        candidates.append({
                            'FILE': os.path.basename(filepath),
                            'ARM': arm,
                            'FIBER': int(fib),
                            'RA': float(info['TARGET_RA']) if 'TARGET_RA' in info.dtype.names else 0,
                            'DEC': float(info['TARGET_DEC']) if 'TARGET_DEC' in info.dtype.names else 0,
                            'WAVELENGTH': float(w),
                            'SNR': float(s),
                            'FWHM': int(fwhm),
                            'IS_AIRGLOW': is_ag,
                            'ISOLATED': isolated,
                            'PASSED': passed
                        })

                del flux, ivar

        gc.collect()

    except Exception as e:
        print(f"  Error: {os.path.basename(filepath)}: {e}")

    return candidates


def main():
    print("=" * 60)
    print("E9 LASER SEARCH - PROCESSING CACHED FILES")
    print("=" * 60)

    start = datetime.now()

    # Find all cached files
    files = list(glob.glob("data/e9_cache/**/*.fits", recursive=True))
    print(f"\nFound {len(files)} cached FITS files")
    print(f"Using {N_WORKERS} workers")
    print(f"Threshold: {SIGMA_THRESHOLD}σ")

    if not files:
        print("No files to process!")
        return 1

    # Process in parallel
    print("\nProcessing...")

    with Pool(N_WORKERS) as pool:
        results = pool.map(process_file, files)

    # Collect results
    all_candidates = []
    for r in results:
        all_candidates.extend(r)

    runtime = (datetime.now() - start).total_seconds()

    # Statistics
    n_total = len(all_candidates)
    n_passed = sum(1 for c in all_candidates if c['PASSED'])
    n_airglow = sum(1 for c in all_candidates if c['IS_AIRGLOW'])

    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Files processed: {len(files)}")
    print(f"Total detections: {n_total}")
    print(f"Passed all cuts: {n_passed}")
    print(f"Rejected (airglow): {n_airglow}")
    print(f"Runtime: {runtime:.1f}s")

    # Save results
    if all_candidates:
        outfile = OUTPUT_DIR / "laser_candidates.csv"
        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_candidates[0].keys())
            writer.writeheader()
            writer.writerows(all_candidates)
        print(f"\nSaved to: {outfile}")

        if n_passed > 0:
            passfile = OUTPUT_DIR / "laser_candidates_passed.csv"
            passed = [c for c in all_candidates if c['PASSED']]
            with open(passfile, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=passed[0].keys())
                writer.writeheader()
                writer.writerows(passed)
            print(f"Passed saved to: {passfile}")

    # Show top candidates
    if n_passed > 0:
        print(f"\nTOP CANDIDATES (passed all cuts):")
        passed = sorted([c for c in all_candidates if c['PASSED']], key=lambda x: -x['SNR'])
        for i, c in enumerate(passed[:10]):
            print(f"  [{i+1}] λ={c['WAVELENGTH']:.2f}Å  SNR={c['SNR']:.1f}  RA={c['RA']:.4f}")
    else:
        print("\nNo candidates passed all cuts.")
        if all_candidates:
            print("Top rejected:")
            top = sorted(all_candidates, key=lambda x: -x['SNR'])[:5]
            for c in top:
                reason = "airglow" if c['IS_AIRGLOW'] else "not isolated"
                print(f"  λ={c['WAVELENGTH']:.1f}Å SNR={c['SNR']:.1f} [{reason}]")

    # Save stats
    stats = {
        'files': len(files),
        'detections': n_total,
        'passed': n_passed,
        'airglow': n_airglow,
        'runtime': runtime
    }
    with open(OUTPUT_DIR / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
