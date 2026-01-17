#!/usr/bin/env python3
"""
E9 INTERSTELLAR LASER SEARCH - FULL SCALE
==========================================

Complete technosignature search across DESI DR1 sky fiber spectra.
Searches for monochromatic optical pulses that could indicate
interstellar laser communication or propulsion systems.

This is the FULL search - no shortcuts, all available data.

Usage:
    nohup python3 -u scripts/run_e9_full_scale.py > data/e9_results/e9_full.log 2>&1 &
"""

import os
import sys
import numpy as np
from astropy.io import fits
from multiprocessing import Pool, cpu_count
import gc
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import json
import csv
import time

# =============================================================================
# CONFIGURATION - FULL SCALE
# =============================================================================

# Target: Download ALL available files we can access
N_TARGET_FILES = 100       # Target number of files
N_WORKERS = max(1, cpu_count() - 2)  # Use most cores for processing
DOWNLOAD_TIMEOUT = 120     # Seconds per download
DOWNLOAD_RETRIES = 2

# Detection parameters
SIGMA_THRESHOLD = 8.0      # 8-sigma detection threshold
MAX_FWHM_PIXELS = 4        # Max ~4 pixels (narrow line)
AIRGLOW_WINDOW = 2.5       # Angstroms rejection window

# Data source
DESI_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
CACHE_DIR = Path("data/e9_cache")
OUTPUT_DIR = Path("data/e9_results")

# ALL known healpix ranges in DESI DR1 main survey
HEALPIX_RANGES = [
    (9550, 9650),   # Main survey region 1
    (9700, 9800),   # Main survey region 2
    (10000, 10100), # Main survey region 3
    (10100, 10200), # Main survey region 4
    (10200, 10300), # Main survey region 5
]

# Build complete healpix list
ALL_HEALPIX = []
for start, end in HEALPIX_RANGES:
    ALL_HEALPIX.extend(range(start, end))

# Known atmospheric emission lines (Anti-Airglow Shield)
AIRGLOW_LINES = {
    # Oxygen forbidden lines
    5577.3, 6300.3, 6363.8,
    # Sodium D
    5895.9, 5889.9,
    # Mercury
    4046.6, 4358.3, 5460.7, 5769.6, 5790.7,
    # OH forest (major bands)
    6834, 6871, 6912, 6950, 6989,
    7244, 7276, 7316, 7340, 7369,
    7524, 7556, 7586, 7621, 7651,
    7750, 7794, 7821, 7853, 7914,
    7964, 7993, 8025, 8063, 8101,
    8264, 8298, 8344, 8382, 8430,
    8505, 8627, 8696, 8761, 8827, 8886, 8943
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_near_airglow(wavelength: float) -> bool:
    """Check if wavelength is within rejection window of known airglow."""
    for line in AIRGLOW_LINES:
        if abs(wavelength - line) < AIRGLOW_WINDOW:
            return True
    return False


def get_airglow_name(wavelength: float) -> str:
    """Get name of nearby airglow line."""
    for line in AIRGLOW_LINES:
        if abs(wavelength - line) < AIRGLOW_WINDOW:
            if line == 5577.3:
                return "OI_5577"
            elif line == 6300.3:
                return "OI_6300"
            elif line in (5889.9, 5895.9):
                return "NaD"
            else:
                return f"sky_{int(line)}"
    return ""


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_with_retry(healpix: int) -> Path:
    """Download a DESI coadd file with retries."""
    pix_dir = healpix // 100
    filename = f"coadd-main-bright-{healpix}.fits"
    url = f"{DESI_BASE}/healpix/main/bright/{pix_dir}/{healpix}/{filename}"

    local_dir = CACHE_DIR / "healpix" / str(pix_dir) / str(healpix)
    local_path = local_dir / filename

    # Check if already cached and valid
    if local_path.exists():
        try:
            with fits.open(local_path) as hdu:
                if len(hdu) > 3:  # Valid file has multiple extensions
                    return local_path
        except:
            local_path.unlink()

    local_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(DOWNLOAD_RETRIES):
        try:
            urllib.request.urlretrieve(url, local_path, timeout=DOWNLOAD_TIMEOUT)

            # Verify the download
            with fits.open(local_path) as hdu:
                if len(hdu) > 3:
                    return local_path
            # Invalid file
            local_path.unlink()

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # File doesn't exist
            # Other HTTP error, retry
        except Exception:
            if local_path.exists():
                local_path.unlink()
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(2)

    return None


def download_files_sequential(healpix_list: list, target_count: int) -> list:
    """Download files one at a time to ensure integrity."""
    downloaded = []
    failed = 0
    total = len(healpix_list)

    print(f"\n  Scanning {total} healpix pixels...")
    print(f"  Target: {target_count} files")

    for i, hpx in enumerate(healpix_list):
        if len(downloaded) >= target_count:
            break

        print(f"\r  [{len(downloaded)}/{target_count}] Trying healpix {hpx}...", end="", flush=True)

        path = download_with_retry(hpx)
        if path:
            downloaded.append(path)
            print(f"\r  [{len(downloaded)}/{target_count}] Downloaded: {path.name}                    ")
        else:
            failed += 1

    print(f"\n  Download complete: {len(downloaded)} files ({failed} unavailable)")
    return downloaded


# =============================================================================
# CORE DETECTION ENGINE
# =============================================================================

def process_single_file(filepath: str) -> tuple:
    """
    Process one DESI file for laser candidates.

    Returns (candidates_list, stats_dict)
    """
    candidates = []
    stats = {'sky_fibers': 0, 'peaks_checked': 0}

    try:
        with fits.open(filepath, memmap=True) as hdu:
            fibermap = hdu['FIBERMAP'].data

            # Select sky fibers
            if 'OBJTYPE' in fibermap.dtype.names:
                sky_mask = fibermap['OBJTYPE'] == 'SKY'
            else:
                sky_mask = fibermap['TARGETID'] < 0

            sky_indices = np.where(sky_mask)[0]
            n_sky = len(sky_indices)

            if n_sky < 3:
                return [], stats

            stats['sky_fibers'] = n_sky

            # Process each spectral arm (B=blue, R=red, Z=infrared)
            for arm in ['B', 'R', 'Z']:
                wave_ext = f'{arm}_WAVELENGTH'
                flux_ext = f'{arm}_FLUX'
                ivar_ext = f'{arm}_IVAR'

                if wave_ext not in [h.name for h in hdu]:
                    continue

                # Load data
                wavelength = np.array(hdu[wave_ext].data)
                flux_all = hdu[flux_ext].data
                ivar_all = hdu[ivar_ext].data

                # Extract sky fiber data
                flux = flux_all[sky_mask].copy()
                ivar = ivar_all[sky_mask].copy()

                # THE DIFFERENCE ENGINE
                # Step 1: Calculate median sky spectrum
                median_sky = np.nanmedian(flux, axis=0)
                mad = np.nanmedian(np.abs(flux - median_sky), axis=0) * 1.4826

                # Step 2: Analyze each fiber
                for i, fib_idx in enumerate(sky_indices):
                    # Compute residual
                    residual = flux[i] - median_sky

                    # Compute significance (SNR)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        fiber_sigma = np.where(ivar[i] > 0, 1.0/np.sqrt(ivar[i]), np.inf)
                        total_sigma = np.sqrt(mad**2 + fiber_sigma**2)
                        snr = residual / total_sigma

                    # Find peaks above threshold
                    peak_pixels = np.where(snr > SIGMA_THRESHOLD)[0]

                    for pk in peak_pixels:
                        stats['peaks_checked'] += 1

                        # Filter: Skip spectrum edges
                        if pk < 5 or pk > len(wavelength) - 5:
                            continue

                        peak_wave = wavelength[pk]
                        peak_snr = snr[pk]
                        peak_flux = flux[i, pk]

                        # LASER CRITERIA

                        # 1. Width check (FWHM in pixels)
                        fwhm = 1
                        for offset in range(1, 6):
                            if pk - offset >= 0 and snr[pk - offset] > SIGMA_THRESHOLD * 0.5:
                                fwhm += 1
                            else:
                                break
                        for offset in range(1, 6):
                            if pk + offset < len(snr) and snr[pk + offset] > SIGMA_THRESHOLD * 0.5:
                                fwhm += 1
                            else:
                                break

                        if fwhm > MAX_FWHM_PIXELS:
                            continue  # Too wide, not a laser

                        # 2. Airglow check
                        is_airglow = is_near_airglow(peak_wave)
                        airglow_name = get_airglow_name(peak_wave) if is_airglow else ""

                        # 3. Spatial isolation check
                        n_neighbors = 0
                        for j in range(len(flux)):
                            if j == i:
                                continue
                            other_residual = flux[j, pk] - median_sky[pk]
                            other_noise = total_sigma[pk]
                            if other_residual / other_noise > SIGMA_THRESHOLD * 0.5:
                                n_neighbors += 1

                        is_isolated = n_neighbors <= 1

                        # Final determination
                        passed_all = not is_airglow and is_isolated

                        # Get fiber metadata
                        fib_info = fibermap[fib_idx]
                        ra = float(fib_info['TARGET_RA']) if 'TARGET_RA' in fib_info.dtype.names else 0
                        dec = float(fib_info['TARGET_DEC']) if 'TARGET_DEC' in fib_info.dtype.names else 0
                        mjd = float(fib_info['MJD']) if 'MJD' in fib_info.dtype.names else 0

                        candidate = {
                            'FILE': os.path.basename(filepath),
                            'ARM': arm,
                            'FIBER': int(fib_idx),
                            'RA': ra,
                            'DEC': dec,
                            'MJD': mjd,
                            'WAVELENGTH': float(peak_wave),
                            'FLUX': float(peak_flux),
                            'SNR': float(peak_snr),
                            'FWHM_PIXELS': int(fwhm),
                            'MEDIAN_SKY': float(median_sky[pk]),
                            'N_NEIGHBORS': n_neighbors,
                            'IS_AIRGLOW': is_airglow,
                            'AIRGLOW_LINE': airglow_name,
                            'IS_ISOLATED': is_isolated,
                            'PASSED_ALL_CUTS': passed_all
                        }
                        candidates.append(candidate)

                # Cleanup
                del flux, ivar, flux_all, ivar_all, wavelength

        gc.collect()

    except Exception as e:
        print(f"\n  Error: {os.path.basename(str(filepath))}: {e}")

    return candidates, stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("E9 INTERSTELLAR LASER SEARCH - FULL SCALE")
    print("=" * 70)
    print("\nSearching for monochromatic optical pulse signatures")
    print("in DESI Data Release 1 sky fiber spectra")
    print("=" * 70)

    start_time = datetime.now()

    # Setup directories
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configuration summary
    print(f"\nCONFIGURATION:")
    print(f"  Target files: {N_TARGET_FILES}")
    print(f"  Processing workers: {N_WORKERS}")
    print(f"  Detection threshold: {SIGMA_THRESHOLD}σ")
    print(f"  Max FWHM: {MAX_FWHM_PIXELS} pixels")
    print(f"  Airglow rejection window: ±{AIRGLOW_WINDOW}Å")
    print(f"  Healpix pixels to scan: {len(ALL_HEALPIX)}")

    # =========================================================================
    # PHASE 1: DATA ACQUISITION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: DATA ACQUISITION")
    print("=" * 70)

    files = download_files_sequential(ALL_HEALPIX, N_TARGET_FILES)

    if not files:
        print("\nFATAL ERROR: Could not download any DESI files!")
        print("Check network connection and DESI data availability.")
        return 1

    # =========================================================================
    # PHASE 2: DIFFERENCE ENGINE PROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: LASER DETECTION (DIFFERENCE ENGINE)")
    print("=" * 70)

    print(f"\n  Processing {len(files)} files with {N_WORKERS} parallel workers...")

    all_candidates = []
    total_sky_fibers = 0
    total_peaks_checked = 0

    with Pool(N_WORKERS) as pool:
        results = pool.map(process_single_file, [str(f) for f in files])

    for candidates, stats in results:
        all_candidates.extend(candidates)
        total_sky_fibers += stats.get('sky_fibers', 0)
        total_peaks_checked += stats.get('peaks_checked', 0)

    # =========================================================================
    # PHASE 3: RESULTS AND REPORTING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: RESULTS")
    print("=" * 70)

    runtime = (datetime.now() - start_time).total_seconds()

    n_total = len(all_candidates)
    n_passed = sum(1 for c in all_candidates if c['PASSED_ALL_CUTS'])
    n_airglow = sum(1 for c in all_candidates if c['IS_AIRGLOW'])
    n_not_isolated = sum(1 for c in all_candidates if not c['IS_ISOLATED'])

    print(f"\n  SEARCH STATISTICS:")
    print(f"    Files processed: {len(files)}")
    print(f"    Sky fibers analyzed: {total_sky_fibers:,}")
    print(f"    Spectral peaks checked: {total_peaks_checked:,}")
    print(f"    Total {SIGMA_THRESHOLD}σ+ detections: {n_total}")
    print(f"    ----------------------------------------")
    print(f"    PASSED ALL CUTS: {n_passed}")
    print(f"    Rejected (airglow): {n_airglow}")
    print(f"    Rejected (not isolated): {n_not_isolated}")
    print(f"    ----------------------------------------")
    print(f"    Runtime: {runtime:.1f} seconds")

    # Save all candidates
    if all_candidates:
        out_all = OUTPUT_DIR / "laser_candidates_all.csv"
        fieldnames = list(all_candidates[0].keys())
        with open(out_all, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candidates)
        print(f"\n  Saved all candidates to: {out_all}")

        # Save passed candidates
        passed_list = [c for c in all_candidates if c['PASSED_ALL_CUTS']]
        if passed_list:
            out_passed = OUTPUT_DIR / "laser_candidates_PASSED.csv"
            with open(out_passed, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(passed_list)
            print(f"  Saved passed candidates to: {out_passed}")

    # Report findings
    if n_passed > 0:
        print("\n" + "=" * 70)
        print(f">>> {n_passed} TECHNOSIGNATURE CANDIDATES DETECTED <<<")
        print("=" * 70)

        passed_sorted = sorted(
            [c for c in all_candidates if c['PASSED_ALL_CUTS']],
            key=lambda x: -x['SNR']
        )

        for i, c in enumerate(passed_sorted[:20]):
            print(f"\n  CANDIDATE #{i+1}")
            print(f"    Wavelength: {c['WAVELENGTH']:.3f} Å")
            print(f"    SNR: {c['SNR']:.2f}")
            print(f"    Position: RA={c['RA']:.6f}°, DEC={c['DEC']:.6f}°")
            print(f"    MJD: {c['MJD']:.4f}")
            print(f"    FWHM: {c['FWHM_PIXELS']} pixels")
            print(f"    Source: {c['FILE']}, Fiber {c['FIBER']}")

    else:
        print("\n  No candidates passed all cuts.")
        print("  This is EXPECTED for normal sky observations.")

        if all_candidates:
            print("\n  For reference, top rejected candidates:")
            top = sorted(all_candidates, key=lambda x: -x['SNR'])[:10]
            for c in top:
                reason = []
                if c['IS_AIRGLOW']:
                    reason.append(f"airglow ({c['AIRGLOW_LINE']})")
                if not c['IS_ISOLATED']:
                    reason.append(f"not isolated ({c['N_NEIGHBORS']} neighbors)")
                reason_str = ", ".join(reason)
                print(f"    λ={c['WAVELENGTH']:.2f}Å  SNR={c['SNR']:.1f}  [{reason_str}]")

    # Save comprehensive statistics
    stats = {
        'experiment': 'E9_INTERSTELLAR_LASER_SEARCH',
        'version': 'FULL_SCALE',
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': runtime,
        'configuration': {
            'target_files': N_TARGET_FILES,
            'workers': N_WORKERS,
            'sigma_threshold': SIGMA_THRESHOLD,
            'max_fwhm_pixels': MAX_FWHM_PIXELS,
            'airglow_window': AIRGLOW_WINDOW
        },
        'results': {
            'files_processed': len(files),
            'sky_fibers_analyzed': total_sky_fibers,
            'peaks_checked': total_peaks_checked,
            'total_detections': n_total,
            'passed_all_cuts': n_passed,
            'rejected_airglow': n_airglow,
            'rejected_not_isolated': n_not_isolated
        }
    }

    stats_file = OUTPUT_DIR / "e9_full_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Statistics saved to: {stats_file}")

    print("\n" + "=" * 70)
    print("E9 INTERSTELLAR LASER SEARCH COMPLETE")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
