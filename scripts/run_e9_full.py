#!/usr/bin/env python3
"""
E9 INTERSTELLAR LASER SEARCH - FULL DESI DR1 EXECUTION

This script runs the laser search on a large batch of DESI DR1 sky fibers.
Uses multiprocessing for parallel execution.
"""

import os
import sys
import logging
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

# Configure logging
os.makedirs('data/e9_results', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/e9_results/e9_full_search.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DESI_DR1_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
CACHE_DIR = Path("data/e9_cache")
OUTPUT_DIR = Path("data/e9_results")

# Detection parameters
SIGMA_THRESHOLD = 8.0  # Lower to catch more candidates
MAX_FWHM_PIXELS = 4.0
REJECTION_WINDOW = 2.0  # Angstroms

# Known airglow lines
AIRGLOW_LINES = {
    'OI_5577': 5577.3, 'OI_6300': 6300.3, 'OI_6364': 6363.8,
    'NaD1': 5895.9, 'NaD2': 5889.9,
    'Hg_4047': 4046.6, 'Hg_4358': 4358.3, 'Hg_5461': 5460.7,
}

# Add OH forest
for w in np.arange(6800, 9000, 50):
    AIRGLOW_LINES[f'OH_{int(w)}'] = w


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_spectra_file(healpix: int, survey: str = 'main', program: str = 'bright') -> Path:
    """Fetch a DESI coadd spectra file."""
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


def get_healpix_list(n_max: int = 100) -> list:
    """Get list of healpix pixels to process."""
    # These are known valid healpix pixels in DR1
    pixels = list(range(9550, 9700)) + list(range(10000, 10100))
    return pixels[:n_max]


# =============================================================================
# Detection Engine
# =============================================================================

def is_near_airglow(wavelength: float) -> tuple:
    """Check if wavelength is near airglow."""
    for name, line in AIRGLOW_LINES.items():
        if abs(wavelength - line) < REJECTION_WINDOW:
            return True, name
    return False, ""


def process_single_exposure(filepath: Path) -> list:
    """
    Process a single DESI spectra file for laser candidates.

    Returns list of candidate dictionaries.
    """
    candidates = []

    try:
        with fits.open(filepath) as hdu:
            fibermap = hdu['FIBERMAP'].data

            # Get sky fibers
            if 'OBJTYPE' in fibermap.dtype.names:
                sky_mask = fibermap['OBJTYPE'] == 'SKY'
            else:
                sky_mask = fibermap['TARGETID'] < 0

            n_sky = np.sum(sky_mask)
            if n_sky < 3:
                return []

            sky_fibermap = fibermap[sky_mask]

            # Process each spectral arm
            for arm in ['B', 'R', 'Z']:
                wave_key = f'{arm}_WAVELENGTH'
                flux_key = f'{arm}_FLUX'
                ivar_key = f'{arm}_IVAR'

                if wave_key not in [h.name for h in hdu]:
                    continue

                wavelength = hdu[wave_key].data
                flux = hdu[flux_key].data[sky_mask]
                ivar = hdu[ivar_key].data[sky_mask]

                # Compute median sky
                median_flux = np.nanmedian(flux, axis=0)
                mad = np.nanmedian(np.abs(flux - median_flux), axis=0) * 1.4826

                # Compute residuals
                residuals = flux - median_flux

                with np.errstate(invalid='ignore', divide='ignore'):
                    fiber_sigma = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf)
                    total_sigma = np.sqrt(mad**2 + fiber_sigma**2)
                    significance = residuals / total_sigma

                # Find peaks above threshold
                for fib_idx in range(n_sky):
                    sig = significance[fib_idx]

                    # Find pixels above threshold
                    above = sig > SIGMA_THRESHOLD
                    if not np.any(above):
                        continue

                    # Find contiguous regions
                    diff = np.diff(above.astype(int))
                    starts = np.where(diff == 1)[0] + 1
                    ends = np.where(diff == -1)[0] + 1

                    if above[0]:
                        starts = np.concatenate([[0], starts])
                    if above[-1]:
                        ends = np.concatenate([ends, [len(above)]])

                    for start, end in zip(starts, ends):
                        fwhm = end - start
                        if fwhm > MAX_FWHM_PIXELS:
                            continue

                        # Find peak
                        peak_idx = start + np.argmax(sig[start:end])
                        peak_wave = wavelength[peak_idx]
                        peak_sig = sig[peak_idx]
                        peak_flux = flux[fib_idx, peak_idx]

                        # Check airglow
                        is_airglow, airglow_name = is_near_airglow(peak_wave)

                        # Count neighboring fibers with similar detection
                        n_neighbors = 0
                        for other_fib in range(n_sky):
                            if other_fib == fib_idx:
                                continue
                            if significance[other_fib, peak_idx] > SIGMA_THRESHOLD * 0.5:
                                n_neighbors += 1

                        is_isolated = n_neighbors <= 1
                        passed = not is_airglow and is_isolated

                        # Get fiber info
                        fib_info = sky_fibermap[fib_idx]

                        candidate = {
                            'FILEPATH': str(filepath),
                            'ARM': arm,
                            'FIBER_IDX': fib_idx,
                            'TARGETID': int(fib_info['TARGETID']) if 'TARGETID' in fib_info.dtype.names else -1,
                            'RA': float(fib_info['TARGET_RA']) if 'TARGET_RA' in fib_info.dtype.names else 0,
                            'DEC': float(fib_info['TARGET_DEC']) if 'TARGET_DEC' in fib_info.dtype.names else 0,
                            'WAVELENGTH': float(peak_wave),
                            'FLUX': float(peak_flux),
                            'SNR': float(peak_sig),
                            'FWHM': float(fwhm),
                            'MEDIAN_SKY': float(median_flux[peak_idx]),
                            'N_NEIGHBORS': n_neighbors,
                            'IS_AIRGLOW': is_airglow,
                            'AIRGLOW_NAME': airglow_name,
                            'IS_ISOLATED': is_isolated,
                            'PASSED': passed
                        }
                        candidates.append(candidate)

        return candidates

    except Exception as e:
        logger.debug(f"Error processing {filepath}: {e}")
        return []


def process_healpix(healpix: int) -> tuple:
    """Process a single healpix pixel."""
    filepath = fetch_spectra_file(healpix)
    if filepath is None:
        return healpix, [], 0

    candidates = process_single_exposure(filepath)

    # Count sky fibers processed
    try:
        with fits.open(filepath) as hdu:
            fibermap = hdu['FIBERMAP'].data
            if 'OBJTYPE' in fibermap.dtype.names:
                n_sky = np.sum(fibermap['OBJTYPE'] == 'SKY')
            else:
                n_sky = np.sum(fibermap['TARGETID'] < 0)
    except:
        n_sky = 0

    return healpix, candidates, n_sky


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("E9 INTERSTELLAR LASER SEARCH - FULL DESI DR1 RUN")
    print("=" * 70)

    start_time = datetime.now()

    # Get healpix list
    n_files = 50  # Process 50 files
    healpix_list = get_healpix_list(n_files)

    print(f"\nConfiguration:")
    print(f"  Healpix pixels to process: {len(healpix_list)}")
    print(f"  Detection threshold: {SIGMA_THRESHOLD} σ")
    print(f"  Max FWHM: {MAX_FWHM_PIXELS} pixels")
    print(f"  Airglow rejection window: {REJECTION_WINDOW} Å")

    # Determine workers
    n_workers = max(1, cpu_count() - 1)
    print(f"  Workers: {n_workers}")

    print(f"\n[1] Processing DESI spectra files...")

    all_candidates = []
    total_fibers = 0
    files_processed = 0

    # Process in parallel
    with Pool(n_workers) as pool:
        results = pool.map(process_healpix, healpix_list)

    for healpix, candidates, n_fibers in results:
        if n_fibers > 0:
            files_processed += 1
            total_fibers += n_fibers
        all_candidates.extend(candidates)

        if files_processed % 10 == 0:
            logger.info(f"Processed {files_processed} files, {len(all_candidates)} candidates so far")

    # Summary statistics
    runtime = (datetime.now() - start_time).total_seconds()

    n_passed = sum(1 for c in all_candidates if c['PASSED'])
    n_airglow = sum(1 for c in all_candidates if c['IS_AIRGLOW'])
    n_not_isolated = sum(1 for c in all_candidates if not c['IS_ISOLATED'])

    print(f"\n[2] Results:")
    print(f"  Files processed: {files_processed}")
    print(f"  Sky fibers analyzed: {total_fibers}")
    print(f"  Initial detections: {len(all_candidates)}")
    print(f"  Passed all cuts: {n_passed}")
    print(f"  Rejected (airglow): {n_airglow}")
    print(f"  Rejected (not isolated): {n_not_isolated}")
    print(f"  Runtime: {runtime:.1f} seconds")

    # Save results
    output_file = OUTPUT_DIR / "laser_candidates.csv"

    if all_candidates:
        import csv
        fieldnames = list(all_candidates[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in all_candidates:
                writer.writerow(c)
        print(f"\n  Results saved to: {output_file}")

        # Also save passed candidates separately
        passed = [c for c in all_candidates if c['PASSED']]
        if passed:
            passed_file = OUTPUT_DIR / "laser_candidates_passed.csv"
            with open(passed_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for c in passed:
                    writer.writerow(c)
            print(f"  Passed candidates saved to: {passed_file}")

    # Print top candidates
    if n_passed > 0:
        print(f"\n" + "=" * 70)
        print(f"TOP {min(10, n_passed)} LASER CANDIDATES (>{SIGMA_THRESHOLD}σ, passed all cuts):")
        print("=" * 70)

        passed = sorted([c for c in all_candidates if c['PASSED']],
                       key=lambda x: -x['SNR'])

        for i, c in enumerate(passed[:10]):
            print(f"\n  [{i+1}] λ = {c['WAVELENGTH']:.2f} Å")
            print(f"      SNR = {c['SNR']:.1f}")
            print(f"      RA, DEC = {c['RA']:.4f}, {c['DEC']:.4f}")
            print(f"      Flux = {c['FLUX']:.2f}")
            print(f"      FWHM = {c['FWHM']:.0f} pixels")
    else:
        print("\n  No candidates passed all cuts.")

        # Show top candidates that were rejected
        if all_candidates:
            print("\n  Top rejected candidates (for inspection):")
            top_rejected = sorted(all_candidates, key=lambda x: -x['SNR'])[:5]
            for c in top_rejected:
                reason = "airglow" if c['IS_AIRGLOW'] else "not isolated"
                print(f"    λ={c['WAVELENGTH']:.1f}Å SNR={c['SNR']:.1f} ({reason}: {c.get('AIRGLOW_NAME', '')})")

    # Save statistics
    stats = {
        'files_processed': files_processed,
        'sky_fibers_analyzed': total_fibers,
        'initial_detections': len(all_candidates),
        'passed_all_cuts': n_passed,
        'rejected_airglow': n_airglow,
        'rejected_not_isolated': n_not_isolated,
        'runtime_seconds': runtime,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sigma_threshold': SIGMA_THRESHOLD,
            'max_fwhm_pixels': MAX_FWHM_PIXELS,
            'rejection_window': REJECTION_WINDOW
        }
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
    sys.exit(main())
