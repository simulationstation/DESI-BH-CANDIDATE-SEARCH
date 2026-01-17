#!/usr/bin/env python3
"""
E9 INTERSTELLAR LASER SEARCH - COMPLETE PIPELINE

Downloads DESI DR1 sky fiber data sequentially (to avoid corruption),
then processes in parallel.

Usage: nohup python3 -u scripts/run_e9_complete.py > data/e9_results/e9.log 2>&1 &
"""

import os
import sys
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

# =============================================================================
# CONFIGURATION
# =============================================================================

N_DOWNLOAD = 15         # Number of files to download
N_WORKERS = 6           # Workers for processing (not downloading)
SIGMA_THRESHOLD = 8.0   # Detection threshold
MAX_FWHM = 4            # Max FWHM in pixels
AIRGLOW_WINDOW = 2.5    # Rejection window (Angstroms)

DESI_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
CACHE_DIR = Path("data/e9_cache")
OUTPUT_DIR = Path("data/e9_results")

# Known good healpix pixels
HEALPIX_LIST = list(range(9550, 9600))

# Airglow lines to reject
AIRGLOW = {
    5577.3, 6300.3, 6363.8, 5895.9, 5889.9, 4358.3, 5460.7,
    6834, 6871, 6950, 7244, 7316, 7524, 7621, 7750, 7853,
    7964, 8101, 8344, 8505, 8696, 8827
}

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(healpix: int, retries: int = 3) -> Path:
    """Download a single file with retries."""
    pix_dir = healpix // 100
    filename = f"coadd-main-bright-{healpix}.fits"
    url = f"{DESI_BASE}/healpix/main/bright/{pix_dir}/{healpix}/{filename}"

    local_dir = CACHE_DIR / "healpix" / "main" / "bright" / str(pix_dir) / str(healpix)
    local_path = local_dir / filename

    if local_path.exists():
        # Verify file is complete
        try:
            with fits.open(local_path) as hdu:
                _ = hdu['FIBERMAP'].data
            return local_path
        except:
            # File is corrupt, delete it
            local_path.unlink()

    local_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            print(f"  Downloading {filename}...", end=" ", flush=True)
            urllib.request.urlretrieve(url, local_path)

            # Verify download
            with fits.open(local_path) as hdu:
                n_ext = len(hdu)
            print(f"OK ({n_ext} extensions)")
            return local_path

        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            if local_path.exists():
                local_path.unlink()
            if attempt < retries - 1:
                print(f"    Retrying...")

    return None


def download_all(n_files: int) -> list:
    """Download files sequentially."""
    print(f"\n[PHASE 1] Downloading {n_files} DESI files...")

    paths = []
    for hpx in HEALPIX_LIST:
        if len(paths) >= n_files:
            break

        path = download_file(hpx)
        if path:
            paths.append(path)

    print(f"  Downloaded {len(paths)} files successfully")
    return paths


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def is_airglow(w):
    for line in AIRGLOW:
        if abs(w - line) < AIRGLOW_WINDOW:
            return True
    return False


def process_file(filepath: str) -> list:
    """Process one FITS file for laser candidates."""
    candidates = []

    try:
        with fits.open(filepath, memmap=True) as hdu:
            fibermap = hdu['FIBERMAP'].data

            # Get sky fibers
            if 'OBJTYPE' in fibermap.dtype.names:
                sky_idx = np.where(fibermap['OBJTYPE'] == 'SKY')[0]
            else:
                sky_idx = np.where(fibermap['TARGETID'] < 0)[0]

            n_sky = len(sky_idx)
            if n_sky < 3:
                return []

            # Process each spectral arm
            for arm in ['B', 'R', 'Z']:
                if f'{arm}_WAVELENGTH' not in [h.name for h in hdu]:
                    continue

                wave = np.array(hdu[f'{arm}_WAVELENGTH'].data)
                flux_all = hdu[f'{arm}_FLUX'].data
                ivar_all = hdu[f'{arm}_IVAR'].data

                flux = flux_all[sky_idx].copy()
                ivar = ivar_all[sky_idx].copy()

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

                        if fwhm > MAX_FWHM:
                            continue

                        # Neighbor check
                        n_neighbors = sum(
                            1 for j in range(len(flux))
                            if j != i and (flux[j,pk] - median[pk]) / noise[pk] > SIGMA_THRESHOLD * 0.5
                        )

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
                            'FLUX': float(flux[i, pk]),
                            'SNR': float(s),
                            'FWHM': int(fwhm),
                            'N_NEIGHBORS': n_neighbors,
                            'IS_AIRGLOW': is_ag,
                            'ISOLATED': isolated,
                            'PASSED': passed
                        })

                del flux, ivar, flux_all, ivar_all

        gc.collect()

    except Exception as e:
        print(f"  Error processing {os.path.basename(str(filepath))}: {e}")

    return candidates


def process_all(files: list) -> list:
    """Process all files in parallel."""
    print(f"\n[PHASE 2] Processing {len(files)} files with {N_WORKERS} workers...")

    with Pool(N_WORKERS) as pool:
        results = pool.map(process_file, [str(f) for f in files])

    all_candidates = []
    for r in results:
        all_candidates.extend(r)

    return all_candidates


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("E9 INTERSTELLAR LASER SEARCH")
    print("Searching for monochromatic anomalies in DESI sky fibers")
    print("=" * 65)

    start_time = datetime.now()

    # Setup
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Files to download: {N_DOWNLOAD}")
    print(f"  Processing workers: {N_WORKERS}")
    print(f"  Detection threshold: {SIGMA_THRESHOLD}σ")
    print(f"  Max FWHM: {MAX_FWHM} pixels")

    # Phase 1: Download
    files = download_all(N_DOWNLOAD)

    if not files:
        print("\nERROR: No files downloaded!")
        return 1

    # Phase 2: Process
    candidates = process_all(files)

    # Results
    runtime = (datetime.now() - start_time).total_seconds()

    n_total = len(candidates)
    n_passed = sum(1 for c in candidates if c['PASSED'])
    n_airglow = sum(1 for c in candidates if c['IS_AIRGLOW'])
    n_not_isolated = sum(1 for c in candidates if not c['ISOLATED'])

    print(f"\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Files processed: {len(files)}")
    print(f"  Total detections (>{SIGMA_THRESHOLD}σ): {n_total}")
    print(f"  Passed all cuts: {n_passed}")
    print(f"  Rejected (airglow): {n_airglow}")
    print(f"  Rejected (not isolated): {n_not_isolated}")
    print(f"  Runtime: {runtime:.1f} seconds")

    # Save all candidates
    if candidates:
        out_all = OUTPUT_DIR / "laser_candidates.csv"
        with open(out_all, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=candidates[0].keys())
            writer.writeheader()
            writer.writerows(candidates)
        print(f"\n  All candidates: {out_all}")

        # Save passed candidates
        passed = [c for c in candidates if c['PASSED']]
        if passed:
            out_passed = OUTPUT_DIR / "laser_candidates_passed.csv"
            with open(out_passed, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=passed[0].keys())
                writer.writeheader()
                writer.writerows(passed)
            print(f"  Passed candidates: {out_passed}")

    # Show top candidates
    if n_passed > 0:
        print(f"\n" + "=" * 65)
        print(f">>> {n_passed} LASER CANDIDATES DETECTED <<<")
        print("=" * 65)

        passed = sorted([c for c in candidates if c['PASSED']], key=lambda x: -x['SNR'])
        for i, c in enumerate(passed[:15]):
            print(f"\n  [{i+1}] Wavelength: {c['WAVELENGTH']:.2f} Å")
            print(f"      SNR: {c['SNR']:.1f}")
            print(f"      Position: RA={c['RA']:.5f}, DEC={c['DEC']:.5f}")
            print(f"      File: {c['FILE']}, Fiber: {c['FIBER']}")
    else:
        print("\n  No candidates passed all cuts (expected for normal sky)")

        if candidates:
            print("\n  Top rejected candidates (for reference):")
            top = sorted(candidates, key=lambda x: -x['SNR'])[:5]
            for c in top:
                reason = "airglow" if c['IS_AIRGLOW'] else f"not isolated ({c['N_NEIGHBORS']} neighbors)"
                print(f"    λ={c['WAVELENGTH']:.1f}Å  SNR={c['SNR']:.1f}  [{reason}]")

    # Save stats
    stats = {
        'files_processed': len(files),
        'total_detections': n_total,
        'passed_all_cuts': n_passed,
        'rejected_airglow': n_airglow,
        'rejected_not_isolated': n_not_isolated,
        'runtime_seconds': runtime,
        'timestamp': datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / "search_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n" + "=" * 65)
    print("E9 SEARCH COMPLETE")
    print("=" * 65)

    return 0


if __name__ == '__main__':
    sys.exit(main())
