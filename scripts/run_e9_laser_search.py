#!/usr/bin/env python3
"""
E9 Interstellar Laser Search - Real DESI Data Execution Script

This script accesses DESI DR1 public data and runs the laser detection pipeline.

Usage:
    python3 scripts/run_e9_laser_search.py --mode test      # Test with 1 file
    python3 scripts/run_e9_laser_search.py --mode small     # Small batch (10 files)
    python3 scripts/run_e9_laser_search.py --mode full      # Full search
"""

import os
import sys
import logging
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import json
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rare_companions.experiments.e9_interstellar_laser import (
    LaserSearchPipeline,
    DESISkyFiberLoader,
    DifferenceEngine,
    run_smoke_test
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/e9_results/e9_search.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DESI DR1 File Discovery
# =============================================================================

DESI_DR1_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"

# Known healpix pixels with sky fibers in DR1
# These are from the main survey with good sky fiber coverage
SAMPLE_HEALPIX = [
    # Main survey bright program
    9557, 9558, 9559, 9560, 9561, 9562, 9563, 9564,
    9565, 9566, 9567, 9568, 9569, 9570, 9571, 9572,
    9573, 9574, 9575, 9576, 9577, 9578, 9579, 9580,
    9581, 9582, 9583, 9584, 9585, 9586, 9587, 9588,
    9589, 9590, 9591, 9592, 9593, 9594, 9595, 9596,
    10000, 10001, 10002, 10003, 10004, 10005, 10006,
    10007, 10008, 10009, 10010, 10011, 10012, 10013,
    10014, 10015, 10016, 10017, 10018, 10019, 10020,
]


def fetch_desi_spectra_file(healpix: int, survey: str = 'main',
                             program: str = 'bright',
                             cache_dir: str = 'data/e9_cache') -> Path:
    """
    Fetch a DESI coadd spectra file from DR1.

    Parameters
    ----------
    healpix : int
        Healpix pixel ID
    survey : str
        Survey name (main, sv1, sv2, sv3)
    program : str
        Program name (bright, dark)
    cache_dir : str
        Local cache directory

    Returns
    -------
    Path or None
        Path to downloaded file, or None if not found
    """
    # Construct URL
    pix_dir = healpix // 100
    filename = f"coadd-{survey}-{program}-{healpix}.fits"
    url = f"{DESI_DR1_BASE}/healpix/{survey}/{program}/{pix_dir}/{healpix}/{filename}"

    # Local path
    local_dir = Path(cache_dir) / "healpix" / survey / program / str(pix_dir) / str(healpix)
    local_path = local_dir / filename

    if local_path.exists():
        logger.debug(f"Using cached file: {local_path}")
        return local_path

    # Download
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading: {filename}")
        urllib.request.urlretrieve(url, local_path)
        logger.info(f"  Downloaded: {local_path}")
        return local_path
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug(f"Not found: {url}")
        else:
            logger.warning(f"HTTP error {e.code}: {url}")
        return None
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        return None


def discover_available_files(n_files: int = 10,
                             survey: str = 'main',
                             program: str = 'bright',
                             cache_dir: str = 'data/e9_cache') -> list:
    """
    Discover and download available DESI spectra files.

    Returns list of paths to successfully downloaded files.
    """
    logger.info(f"Discovering DESI DR1 files (target: {n_files} files)...")

    available_files = []

    for healpix in SAMPLE_HEALPIX:
        if len(available_files) >= n_files:
            break

        path = fetch_desi_spectra_file(healpix, survey, program, cache_dir)
        if path is not None:
            available_files.append(path)

    logger.info(f"Found {len(available_files)} available files")
    return available_files


def try_alternative_access(cache_dir: str = 'data/e9_cache') -> list:
    """
    Try alternative methods to access DESI spectral data.

    This tries different surveys and programs to find accessible data.
    """
    logger.info("Trying alternative data access methods...")

    available_files = []

    # Try different surveys and programs
    configs = [
        ('main', 'bright'),
        ('main', 'dark'),
        ('sv3', 'bright'),
        ('sv3', 'dark'),
        ('sv1', 'bright'),
    ]

    for survey, program in configs:
        logger.info(f"  Trying {survey}/{program}...")

        for healpix in SAMPLE_HEALPIX[:20]:  # Try first 20 healpix
            path = fetch_desi_spectra_file(healpix, survey, program, cache_dir)
            if path is not None:
                available_files.append(path)
                if len(available_files) >= 5:
                    logger.info(f"    Found {len(available_files)} files in {survey}/{program}")
                    return available_files

    return available_files


# =============================================================================
# Main Execution
# =============================================================================

def run_test_mode():
    """Run a minimal test with 1 file."""
    print("\n" + "=" * 70)
    print("E9 INTERSTELLAR LASER SEARCH - TEST MODE")
    print("=" * 70)

    # First run smoke test with synthetic data
    print("\n[1] Running smoke test with synthetic data...")
    smoke_ok = run_smoke_test()

    if not smoke_ok:
        print("ERROR: Smoke test failed!")
        return 1

    # Try to access real data
    print("\n[2] Attempting to access DESI DR1 data...")

    files = discover_available_files(n_files=1)

    if not files:
        print("\nCould not access DESI public data directly.")
        print("Trying alternative access methods...")
        files = try_alternative_access()

    if not files:
        print("\n" + "-" * 70)
        print("NOTICE: Could not access DESI public spectral data.")
        print("This may be due to network restrictions or data availability.")
        print("\nThe pipeline has been verified with synthetic data.")
        print("To run on real data, either:")
        print("  1. Download DESI DR1 spectra files manually")
        print("  2. Run from a machine with direct DESI data access")
        print("-" * 70)
        return 0

    # Process the file
    print(f"\n[3] Processing {len(files)} file(s)...")

    pipeline = LaserSearchPipeline(
        output_dir='data/e9_results',
        cache_dir='data/e9_cache',
        sigma_threshold=10.0,
        n_workers=1  # Single worker for test
    )

    candidates = pipeline.run_from_local_files(files)

    print(f"\n[4] Results:")
    print(f"    Files processed: {pipeline.stats['files_processed']}")
    print(f"    Initial detections: {pipeline.stats['initial_detections']}")
    print(f"    Passed all cuts: {pipeline.stats['passed_all_cuts']}")

    if candidates:
        pipeline.save_results("laser_candidates_test.csv")

        passed = [c for c in candidates if c.passed_all_cuts]
        if passed:
            print(f"\n    TOP CANDIDATES:")
            for c in sorted(passed, key=lambda x: -x.snr)[:5]:
                print(f"      λ={c.wavelength_peak:.2f}Å  SNR={c.snr:.1f}")

    print("\n" + "=" * 70)
    print("TEST MODE COMPLETE")
    print("=" * 70)

    return 0


def run_small_batch():
    """Run on a small batch of ~10 files."""
    print("\n" + "=" * 70)
    print("E9 INTERSTELLAR LASER SEARCH - SMALL BATCH MODE")
    print("=" * 70)

    print("\n[1] Discovering available DESI files...")
    files = discover_available_files(n_files=10)

    if len(files) < 3:
        files = try_alternative_access()

    if not files:
        print("ERROR: Could not find any accessible DESI files")
        return 1

    print(f"    Found {len(files)} files")

    # Determine number of workers
    import multiprocessing
    n_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"\n[2] Running laser search with {n_workers} workers...")

    pipeline = LaserSearchPipeline(
        output_dir='data/e9_results',
        cache_dir='data/e9_cache',
        sigma_threshold=10.0,
        n_workers=n_workers
    )

    candidates = pipeline.run_from_local_files(files)

    print(f"\n[3] Results:")
    print(f"    Files processed: {pipeline.stats['files_processed']}")
    print(f"    Initial detections: {pipeline.stats['initial_detections']}")
    print(f"    Passed all cuts: {pipeline.stats['passed_all_cuts']}")
    print(f"    Rejected (airglow): {pipeline.stats['rejected_airglow']}")
    print(f"    Rejected (not isolated): {pipeline.stats['rejected_spatial']}")

    pipeline.save_results("laser_candidates_small.csv")

    passed = [c for c in candidates if c.passed_all_cuts]
    if passed:
        print(f"\n    TOP 10 CANDIDATES (>10σ, passed all cuts):")
        for i, c in enumerate(sorted(passed, key=lambda x: -x.snr)[:10]):
            print(f"      {i+1}. λ={c.wavelength_peak:.2f}Å  SNR={c.snr:.1f}  "
                  f"RA={c.ra:.4f}  DEC={c.dec:.4f}")
    else:
        print("\n    No candidates passed all cuts (this is expected for small samples)")

    print("\n" + "=" * 70)
    print("SMALL BATCH COMPLETE")
    print("=" * 70)

    return 0


def run_full_search():
    """Run full search on all available data."""
    print("\n" + "=" * 70)
    print("E9 INTERSTELLAR LASER SEARCH - FULL MODE")
    print("=" * 70)
    print("WARNING: This will download and process large amounts of data!")

    print("\n[1] Discovering all available DESI files...")

    # Get as many files as possible
    all_files = []
    for survey in ['main', 'sv3', 'sv1']:
        for program in ['bright', 'dark']:
            files = []
            for healpix in SAMPLE_HEALPIX:
                path = fetch_desi_spectra_file(healpix, survey, program, 'data/e9_cache')
                if path:
                    files.append(path)

            if files:
                logger.info(f"Found {len(files)} files in {survey}/{program}")
                all_files.extend(files)

    if not all_files:
        print("ERROR: Could not find any accessible DESI files")
        return 1

    print(f"    Total files discovered: {len(all_files)}")

    # Remove duplicates
    all_files = list(set(all_files))
    print(f"    Unique files: {len(all_files)}")

    import multiprocessing
    n_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"\n[2] Running laser search with {n_workers} workers...")

    pipeline = LaserSearchPipeline(
        output_dir='data/e9_results',
        cache_dir='data/e9_cache',
        sigma_threshold=10.0,
        n_workers=n_workers
    )

    candidates = pipeline.run_from_local_files(all_files)

    print(f"\n[3] Final Results:")
    print(f"    Files processed: {pipeline.stats['files_processed']}")
    print(f"    Total sky fibers analyzed: {pipeline.stats.get('fibers_processed', 'N/A')}")
    print(f"    Initial detections: {pipeline.stats['initial_detections']}")
    print(f"    Passed all cuts: {pipeline.stats['passed_all_cuts']}")
    print(f"    Rejected (airglow): {pipeline.stats['rejected_airglow']}")
    print(f"    Rejected (not isolated): {pipeline.stats['rejected_spatial']}")

    pipeline.save_results("laser_candidates.csv")

    passed = [c for c in candidates if c.passed_all_cuts]
    if passed:
        print(f"\n" + "=" * 70)
        print(f"LASER CANDIDATES: {len(passed)} detections passed all cuts!")
        print("=" * 70)

        for i, c in enumerate(sorted(passed, key=lambda x: -x.snr)):
            print(f"\n  Candidate #{i+1}:")
            print(f"    Wavelength: {c.wavelength_peak:.2f} Å")
            print(f"    SNR: {c.snr:.1f}")
            print(f"    RA, DEC: {c.ra:.6f}, {c.dec:.6f}")
            print(f"    MJD: {c.mjd:.4f}")
            print(f"    Fiber ID: {c.fiber_id}")
            print(f"    FWHM: {c.fwhm_pixels:.1f} pixels")
    else:
        print("\n    No candidates passed all cuts.")
        print("    This could mean:")
        print("      - No laser signals present (expected for normal sky)")
        print("      - Detection threshold too high")
        print("      - Not enough data processed")

    print("\n" + "=" * 70)
    print("FULL SEARCH COMPLETE")
    print("=" * 70)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="E9 Interstellar Laser Search"
    )
    parser.add_argument('--mode', choices=['smoke', 'test', 'small', 'full'],
                        default='test', help='Execution mode')

    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs('data/e9_cache', exist_ok=True)
    os.makedirs('data/e9_results', exist_ok=True)

    if args.mode == 'smoke':
        success = run_smoke_test()
        return 0 if success else 1
    elif args.mode == 'test':
        return run_test_mode()
    elif args.mode == 'small':
        return run_small_batch()
    elif args.mode == 'full':
        return run_full_search()


if __name__ == '__main__':
    sys.exit(main())
