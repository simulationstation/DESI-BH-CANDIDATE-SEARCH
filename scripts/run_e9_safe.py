#!/usr/bin/env python3
"""
E9 INTERSTELLAR LASER SEARCH - MEMORY-SAFE VERSION

This script processes DESI DR1 sky fibers in batches with explicit memory management.
Uses limited workers and garbage collection to avoid OOM crashes.

Author: Claude Code
"""

import os
import sys
import glob
import numpy as np
from astropy.io import fits
from multiprocessing import Pool
import gc
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import json
import csv
import logging

# =============================================================================
# CONFIGURATION - MEMORY SAFE SETTINGS
# =============================================================================

N_WORKERS = 4          # Limited workers to avoid OOM
BATCH_SIZE = 10        # Process files in small batches
SIGMA_THRESHOLD = 8.0  # Detection threshold
MAX_FWHM_PIXELS = 4.0  # Maximum width for laser-like signals
REJECTION_WINDOW = 2.5 # Angstroms window around airglow lines

DESI_DR1_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
CACHE_DIR = Path("data/e9_cache")
OUTPUT_DIR = Path("data/e9_results")

# =============================================================================
# AIRGLOW LINES DATABASE
# =============================================================================

AIRGLOW_LINES = {
    # Oxygen forbidden lines
    'OI_5577': 5577.3,
    'OI_6300': 6300.3,
    'OI_6364': 6363.8,
    # Sodium D lines
    'NaD1': 5895.9,
    'NaD2': 5889.9,
    # Mercury
    'Hg_4358': 4358.3,
    'Hg_5461': 5460.7,
}

# Add OH forest lines (major bands)
for w in [6834, 6871, 6950, 7244, 7316, 7524, 7621, 7750, 7853, 7964, 8101, 8344, 8505, 8696, 8827]:
    AIRGLOW_LINES[f'OH_{w}'] = float(w)


def is_near_airglow(wavelength: float) -> tuple:
    """Check if wavelength is near known airglow line."""
    for name, line_wave in AIRGLOW_LINES.items():
        if abs(wavelength - line_wave) < REJECTION_WINDOW:
            return True, name
    return False, ""


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_file(healpix: int, survey: str = 'main', program: str = 'bright') -> Path:
    """Fetch a single DESI coadd file."""
    pix_dir = healpix // 100
    filename = f"coadd-{survey}-{program}-{healpix}.fits"
    url = f"{DESI_DR1_BASE}/healpix/{survey}/{program}/{pix_dir}/{healpix}/{filename}"

    local_dir = CACHE_DIR / "healpix" / survey / program / str(pix_dir) / str(healpix)
    local_path = local_dir / filename

    if local_path.exists():
        return local_path

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except:
        return None


def get_available_files(n_max: int = 50) -> list:
    """Discover available files by trying known healpix pixels."""
    # Known good healpix range
    healpix_list = list(range(9550, 9700)) + list(range(10000, 10100))

    available = []
    for hpx in healpix_list:
        if len(available) >= n_max:
            break
        path = fetch_file(hpx)
        if path is not None:
            available.append(path)
            print(f"  Downloaded: {path.name}")

    return available


# =============================================================================
# CORE DETECTION (MEMORY SAFE)
# =============================================================================

def process_single_file(filepath: str) -> list:
    """
    Process a single DESI file for laser candidates.

    Uses memmap to minimize memory usage.
    Returns list of candidate dictionaries.
    """
    candidates = []

    try:
        # Open with memmap=True to keep data on disk
        with fits.open(filepath, mode='readonly', memmap=True) as hdul:
            # Get fibermap
            if 'FIBERMAP' not in [h.name for h in hdul]:
                return []

            fibermap = hdul['FIBERMAP'].data

            # Select sky fibers
            if 'OBJTYPE' in fibermap.dtype.names:
                sky_mask = fibermap['OBJTYPE'] == 'SKY'
            else:
                sky_mask = fibermap['TARGETID'] < 0

            sky_indices = np.where(sky_mask)[0]
            n_sky = len(sky_indices)

            if n_sky < 3:
                return []

            # Process each spectral arm
            for arm in ['B', 'R', 'Z']:
                wave_ext = f'{arm}_WAVELENGTH'
                flux_ext = f'{arm}_FLUX'
                ivar_ext = f'{arm}_IVAR'

                if wave_ext not in [h.name for h in hdul]:
                    continue

                # Load wavelength (small array)
                wavelength = np.array(hdul[wave_ext].data)

                # Load only sky fiber data to minimize memory
                flux_all = hdul[flux_ext].data
                ivar_all = hdul[ivar_ext].data

                # Extract sky fiber spectra
                flux = flux_all[sky_mask].copy()
                ivar = ivar_all[sky_mask].copy()

                # Clear references to full arrays
                del flux_all, ivar_all

                # Compute median sky spectrum
                median_sky = np.nanmedian(flux, axis=0)
                mad = np.nanmedian(np.abs(flux - median_sky), axis=0) * 1.4826

                # Search each sky fiber for anomalies
                for i, fib_idx in enumerate(sky_indices):
                    fiber_flux = flux[i]
                    fiber_ivar = ivar[i]

                    # Compute residual
                    residual = fiber_flux - median_sky

                    # Compute noise (inverse variance based)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        fiber_sigma = np.where(fiber_ivar > 0, 1.0/np.sqrt(fiber_ivar), np.inf)
                        total_sigma = np.sqrt(mad**2 + fiber_sigma**2)
                        snr = residual / total_sigma

                    # Find pixels above threshold
                    peaks = np.where(snr > SIGMA_THRESHOLD)[0]

                    for peak_idx in peaks:
                        # Edge filtering
                        if peak_idx < 5 or peak_idx > len(wavelength) - 5:
                            continue

                        # Get wavelength
                        peak_wave = wavelength[peak_idx]
                        peak_snr = snr[peak_idx]

                        # Check airglow
                        is_airglow, airglow_name = is_near_airglow(peak_wave)

                        # Estimate FWHM (check neighboring pixels)
                        fwhm = 1
                        for offset in range(1, 5):
                            if peak_idx - offset >= 0 and snr[peak_idx - offset] > SIGMA_THRESHOLD * 0.5:
                                fwhm += 1
                            else:
                                break
                        for offset in range(1, 5):
                            if peak_idx + offset < len(snr) and snr[peak_idx + offset] > SIGMA_THRESHOLD * 0.5:
                                fwhm += 1
                            else:
                                break

                        if fwhm > MAX_FWHM_PIXELS:
                            continue

                        # Check spatial isolation (signal only in this fiber)
                        n_neighbors = 0
                        for j in range(len(flux)):
                            if j == i:
                                continue
                            other_residual = flux[j] - median_sky
                            with np.errstate(divide='ignore', invalid='ignore'):
                                other_sigma = np.where(ivar[j] > 0, 1.0/np.sqrt(ivar[j]), np.inf)
                                other_total = np.sqrt(mad**2 + other_sigma**2)
                                other_snr = other_residual / other_total
                            if other_snr[peak_idx] > SIGMA_THRESHOLD * 0.5:
                                n_neighbors += 1

                        is_isolated = n_neighbors <= 1
                        passed = not is_airglow and is_isolated

                        # Build candidate
                        fib_info = fibermap[fib_idx]
                        cand = {
                            'FILE': os.path.basename(filepath),
                            'ARM': arm,
                            'FIBER': int(fib_idx),
                            'TARGETID': int(fib_info['TARGETID']) if 'TARGETID' in fib_info.dtype.names else -1,
                            'RA': float(fib_info['TARGET_RA']) if 'TARGET_RA' in fib_info.dtype.names else 0,
                            'DEC': float(fib_info['TARGET_DEC']) if 'TARGET_DEC' in fib_info.dtype.names else 0,
                            'WAVELENGTH': float(peak_wave),
                            'FLUX': float(fiber_flux[peak_idx]),
                            'SNR': float(peak_snr),
                            'FWHM': int(fwhm),
                            'N_NEIGHBORS': n_neighbors,
                            'IS_AIRGLOW': is_airglow,
                            'AIRGLOW_LINE': airglow_name,
                            'IS_ISOLATED': is_isolated,
                            'PASSED': passed
                        }
                        candidates.append(cand)

                # Explicit cleanup
                del flux, ivar, median_sky, mad, wavelength

        # Force garbage collection after each file
        gc.collect()

    except Exception as e:
        print(f"  Error processing {os.path.basename(str(filepath))}: {e}")
        return []

    return candidates


# =============================================================================
# MAIN EXECUTION (BATCHED)
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("E9 INTERSTELLAR LASER SEARCH - MEMORY SAFE VERSION")
    print("=" * 70)

    start_time = datetime.now()

    # Setup directories
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Workers: {N_WORKERS} (limited for memory safety)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Detection threshold: {SIGMA_THRESHOLD} σ")
    print(f"  Max FWHM: {MAX_FWHM_PIXELS} pixels")

    # Step 1: Discover/download files
    print(f"\n[1] Discovering DESI DR1 files...")
    n_target = 30  # Target number of files
    files = get_available_files(n_target)

    if not files:
        # Fallback: use cached files
        files = list(CACHE_DIR.glob("**/*.fits"))

    print(f"  Found {len(files)} files to process")

    if not files:
        print("ERROR: No files available!")
        return 1

    # Step 2: Process in batches
    print(f"\n[2] Processing files in batches of {BATCH_SIZE}...")

    all_candidates = []
    total_sky_fibers = 0
    files_processed = 0

    for batch_start in range(0, len(files), BATCH_SIZE):
        batch = files[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} files)...")

        # Process batch in parallel with limited workers
        with Pool(processes=N_WORKERS) as pool:
            results = pool.map(process_single_file, [str(f) for f in batch])

        # Collect results
        batch_candidates = []
        for result in results:
            batch_candidates.extend(result)

        all_candidates.extend(batch_candidates)
        files_processed += len(batch)

        passed_in_batch = sum(1 for c in batch_candidates if c['PASSED'])
        print(f"    Candidates in batch: {len(batch_candidates)} (passed: {passed_in_batch})")

        # Force garbage collection between batches
        del results
        gc.collect()

    # Step 3: Report results
    runtime = (datetime.now() - start_time).total_seconds()

    n_total = len(all_candidates)
    n_passed = sum(1 for c in all_candidates if c['PASSED'])
    n_airglow = sum(1 for c in all_candidates if c['IS_AIRGLOW'])
    n_not_isolated = sum(1 for c in all_candidates if not c['IS_ISOLATED'])

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Files processed: {files_processed}")
    print(f"  Total detections: {n_total}")
    print(f"  Passed all cuts: {n_passed}")
    print(f"  Rejected (airglow): {n_airglow}")
    print(f"  Rejected (not isolated): {n_not_isolated}")
    print(f"  Runtime: {runtime:.1f} seconds")

    # Save all candidates
    if all_candidates:
        output_file = OUTPUT_DIR / "laser_candidates.csv"
        fieldnames = list(all_candidates[0].keys())

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in all_candidates:
                writer.writerow(c)

        print(f"\n  All candidates saved to: {output_file}")

        # Save passed candidates
        if n_passed > 0:
            passed_file = OUTPUT_DIR / "laser_candidates_passed.csv"
            passed = [c for c in all_candidates if c['PASSED']]
            with open(passed_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for c in passed:
                    writer.writerow(c)
            print(f"  Passed candidates saved to: {passed_file}")

    # Print top candidates
    if n_passed > 0:
        print(f"\n" + "=" * 70)
        print(f"TOP LASER CANDIDATES (>{SIGMA_THRESHOLD}σ, passed all cuts)")
        print("=" * 70)

        passed = sorted([c for c in all_candidates if c['PASSED']], key=lambda x: -x['SNR'])

        for i, c in enumerate(passed[:10]):
            print(f"\n  [{i+1}] λ = {c['WAVELENGTH']:.2f} Å")
            print(f"      SNR = {c['SNR']:.1f}")
            print(f"      RA, DEC = {c['RA']:.4f}, {c['DEC']:.4f}")
            print(f"      FWHM = {c['FWHM']} pixels")
            print(f"      File: {c['FILE']}")
    else:
        print("\n  No candidates passed all cuts.")

        if all_candidates:
            print("\n  Top rejected candidates:")
            top = sorted(all_candidates, key=lambda x: -x['SNR'])[:5]
            for c in top:
                reason = f"airglow ({c['AIRGLOW_LINE']})" if c['IS_AIRGLOW'] else "not isolated"
                print(f"    λ={c['WAVELENGTH']:.1f}Å SNR={c['SNR']:.1f} [{reason}]")

    # Save stats
    stats = {
        'files_processed': files_processed,
        'total_detections': n_total,
        'passed_all_cuts': n_passed,
        'rejected_airglow': n_airglow,
        'rejected_not_isolated': n_not_isolated,
        'runtime_seconds': runtime,
        'timestamp': datetime.now().isoformat()
    }

    stats_file = OUTPUT_DIR / "search_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Statistics saved to: {stats_file}")
    print("\n" + "=" * 70)
    print("E9 SEARCH COMPLETE")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
