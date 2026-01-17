#!/usr/bin/env python3
"""
E9 Laser Search Pipeline - Comprehensive, Resumable, Space-Efficient

Features:
- Tracks processed healpix by ID (never re-downloads)
- Auto-deletes files after successful processing
- Parallel processing (max 8 workers)
- Fully resumable from any point
- Generates Globus batch files for unprocessed data
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple
from multiprocessing import Pool
import numpy as np
from astropy.io import fits
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from rare_companions.experiments.e9_interstellar_laser import AIRGLOW_LINES

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("/home/primary/MSOUP/data/e9_healpix")
OUTPUT_DIR = Path("data/e9_results")
MANIFEST_FILE = OUTPUT_DIR / "e9_manifest.json"
CANDIDATES_FILE = OUTPUT_DIR / "e9_all_candidates.csv"

DESI_PUBLIC = "6b4e1f6a-e600-11ed-9b9b-c9bb788c490e"
MSOUP_ENDPOINT = "d1294d08-e055-11f0-a4db-0213754b0ca1"

MAX_WORKERS = 8
SIGMA_THRESHOLD = 5.0
AIRGLOW_TOLERANCE = 5.0
BATCH_SIZE = 500  # Files per Globus transfer

airglow_waves = np.array(sorted(AIRGLOW_LINES.values()))

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Candidate:
    healpix: int
    survey: str
    program: str
    fiber: int
    targetid: int
    ra: float
    dec: float
    wavelength: float
    flux: float
    significance: float
    n_fibers: int
    fwhm_pixels: int
    band: str

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# =============================================================================
# MANIFEST MANAGEMENT
# =============================================================================

def load_manifest() -> Dict:
    """Load the processing manifest."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {
        "version": 2,
        "created": datetime.now().isoformat(),
        "processed_healpix": {},  # {healpix_id: {survey, program, timestamp, n_sky, n_candidates}}
        "stats": {
            "total_healpix": 0,
            "total_sky_fibers": 0,
            "total_candidates": 0,
            "total_resolved": 0
        }
    }

def save_manifest(manifest: Dict):
    """Save the processing manifest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest["last_update"] = datetime.now().isoformat()
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)

def get_processed_healpix(manifest: Dict, survey: str = "main", program: str = "bright") -> Set[int]:
    """Get set of already-processed healpix IDs for a survey/program."""
    processed = set()
    for hp_id, info in manifest["processed_healpix"].items():
        if info.get("survey") == survey and info.get("program") == program:
            processed.add(int(hp_id))
    return processed

def mark_healpix_processed(manifest: Dict, healpix: int, survey: str, program: str,
                           n_sky: int, n_candidates: int):
    """Mark a healpix as processed."""
    manifest["processed_healpix"][str(healpix)] = {
        "survey": survey,
        "program": program,
        "timestamp": datetime.now().isoformat(),
        "n_sky_fibers": n_sky,
        "n_candidates": n_candidates
    }
    manifest["stats"]["total_healpix"] += 1
    manifest["stats"]["total_sky_fibers"] += n_sky
    manifest["stats"]["total_candidates"] += n_candidates

# =============================================================================
# GLOBUS BATCH GENERATION
# =============================================================================

def get_available_healpix(survey: str = "main", program: str = "bright") -> List[int]:
    """Get list of available healpix from DESI endpoint."""
    base_path = f"/dr1/spectro/redux/iron/healpix/{survey}/{program}"

    result = subprocess.run(
        ["globus", "ls", f"{DESI_PUBLIC}:{base_path}/"],
        capture_output=True, text=True, timeout=60
    )

    if result.returncode != 0:
        print(f"Error listing {base_path}: {result.stderr}")
        return []

    groups = [line.rstrip('/') for line in result.stdout.strip().split('\n') if line.strip().isdigit()]

    all_healpix = []
    for group in groups:
        try:
            result = subprocess.run(
                ["globus", "ls", f"{DESI_PUBLIC}:{base_path}/{group}/"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                pixels = [int(p.rstrip('/')) for p in result.stdout.strip().split('\n') if p.strip().isdigit()]
                all_healpix.extend(pixels)
        except Exception as e:
            print(f"  Warning: Error listing group {group}: {e}")
            continue

    return sorted(set(all_healpix))

def generate_globus_batch(manifest: Dict, survey: str = "main", program: str = "bright",
                          max_files: int = BATCH_SIZE) -> Tuple[Path, int]:
    """Generate Globus batch file for unprocessed healpix."""
    print(f"Generating Globus batch for {survey}/{program}...")

    # Get already processed
    processed = get_processed_healpix(manifest, survey, program)
    print(f"  Already processed: {len(processed)} healpix")

    # Get available
    print(f"  Fetching available healpix from DESI...")
    available = get_available_healpix(survey, program)
    print(f"  Available: {len(available)} healpix")

    # Filter to unprocessed
    to_download = [hp for hp in available if hp not in processed]
    print(f"  Unprocessed: {len(to_download)} healpix")

    if not to_download:
        print("  All healpix already processed!")
        return None, 0

    # Limit batch size
    to_download = to_download[:max_files]
    print(f"  Batch size: {len(to_download)} healpix")

    # Generate batch file
    batch_lines = []
    base_src = f"/dr1/spectro/redux/iron/healpix/{survey}/{program}"
    base_dst = f"/home/primary/MSOUP/data/e9_healpix/{survey}/{program}"

    for hp in to_download:
        group = hp // 100
        filename = f"coadd-{survey}-{program}-{hp}.fits"
        src = f"{base_src}/{group}/{hp}/{filename}"
        dst = f"{base_dst}/{group}/{hp}/{filename}"
        batch_lines.append(f"{src} {dst}")

    batch_file = OUTPUT_DIR / f"globus_batch_{survey}_{program}.txt"
    with open(batch_file, 'w') as f:
        f.write('\n'.join(batch_lines))

    print(f"  Batch file: {batch_file}")
    return batch_file, len(to_download)

def submit_globus_transfer(batch_file: Path, label: str) -> Optional[str]:
    """Submit a Globus transfer and return task ID."""
    result = subprocess.run(
        ["globus", "transfer", DESI_PUBLIC, MSOUP_ENDPOINT,
         "--batch", str(batch_file), "--label", label],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Error submitting transfer: {result.stderr}")
        return None

    # Extract task ID
    match = re.search(r'Task ID: ([a-f0-9-]+)', result.stdout)
    if match:
        return match.group(1)
    return None

def check_transfer_status(task_id: str) -> Dict:
    """Check Globus transfer status."""
    result = subprocess.run(
        ["globus", "task", "show", task_id],
        capture_output=True, text=True
    )

    status = {"status": "UNKNOWN", "succeeded": 0, "pending": 0, "bytes": 0}

    for line in result.stdout.split('\n'):
        if 'Status:' in line:
            status["status"] = line.split(':')[1].strip()
        elif 'Subtasks Succeeded:' in line:
            status["succeeded"] = int(line.split(':')[1].strip())
        elif 'Subtasks Pending:' in line:
            status["pending"] = int(line.split(':')[1].strip())
        elif 'Bytes Transferred:' in line:
            status["bytes"] = int(line.split(':')[1].strip())

    return status

# =============================================================================
# FILE PROCESSING
# =============================================================================

def is_near_airglow(wave: float, tol: float = AIRGLOW_TOLERANCE) -> bool:
    return np.any(np.abs(airglow_waves - wave) < tol)

def process_single_file(args: Tuple[Path, str, str]) -> Tuple[int, int, int, List[Dict]]:
    """Process a single coadd file. Returns (healpix, n_sky, n_candidates, candidates)."""
    filepath, survey, program = args

    # Extract healpix from filename
    match = re.search(r'coadd-\w+-\w+-(\d+)\.fits', filepath.name)
    healpix = int(match.group(1)) if match else 0

    candidates = []
    n_sky = 0

    try:
        with fits.open(filepath) as hdul:
            ext_names = [h.name for h in hdul]

            if 'FIBERMAP' not in ext_names:
                return healpix, 0, 0, []

            fibermap = hdul['FIBERMAP'].data

            # Find sky fibers
            if hasattr(fibermap, 'dtype') and fibermap.dtype.names:
                if 'OBJTYPE' in fibermap.dtype.names:
                    sky_mask = fibermap['OBJTYPE'] == 'SKY'
                elif 'TARGETID' in fibermap.dtype.names:
                    sky_mask = fibermap['TARGETID'] < 0
                else:
                    return healpix, 0, 0, []
            else:
                return healpix, 0, 0, []

            n_sky = int(np.sum(sky_mask))
            if n_sky < 5:
                return healpix, n_sky, 0, []

            # Process each arm
            for arm in ['B', 'R', 'Z']:
                wave_key = f'{arm}_WAVELENGTH'
                flux_key = f'{arm}_FLUX'
                ivar_key = f'{arm}_IVAR'
                mask_key = f'{arm}_MASK'

                if wave_key not in ext_names:
                    continue

                wave = hdul[wave_key].data
                flux = hdul[flux_key].data[sky_mask]
                ivar = hdul[ivar_key].data[sky_mask]
                mask = hdul[mask_key].data[sky_mask] if mask_key in ext_names else np.zeros_like(flux)
                sky_fibermap = fibermap[sky_mask]

                # Compute significance
                median_flux = np.nanmedian(flux, axis=0)
                mad = np.nanmedian(np.abs(flux - median_flux), axis=0) * 1.4826
                mad = np.where(mad > 0, mad, 1e-6)
                significance = (flux - median_flux) / mad

                # Find peaks
                for fiber_idx in range(n_sky):
                    sig = significance[fiber_idx]
                    msk = mask[fiber_idx]
                    iv = ivar[fiber_idx]

                    good = (msk == 0) & (iv > 0) & np.isfinite(sig)
                    peaks = np.where((sig > SIGMA_THRESHOLD) & good)[0]

                    for p in peaks:
                        w = float(wave[p])

                        if is_near_airglow(w):
                            continue

                        # Check spatial isolation
                        other_peaks = np.sum((significance[:, p] > 3.0) & (mask[:, p] == 0))
                        if other_peaks > 2:
                            continue

                        # Compute FWHM
                        window = flux[fiber_idx, max(0, p-2):min(len(wave), p+3)]
                        peak_val = np.max(window)
                        half_max = peak_val / 2 if peak_val > 0 else 0
                        fwhm = int(np.sum(window > half_max)) if half_max > 0 else 0

                        fiber_info = sky_fibermap[fiber_idx]
                        fibermap_cols = fibermap.dtype.names if hasattr(fibermap, 'dtype') and fibermap.dtype.names else []
                        candidates.append(asdict(Candidate(
                            healpix=healpix,
                            survey=survey,
                            program=program,
                            fiber=int(fiber_info['FIBER']) if 'FIBER' in fibermap_cols else fiber_idx,
                            targetid=int(fiber_info['TARGETID']) if 'TARGETID' in fibermap_cols else -1,
                            ra=float(fiber_info['TARGET_RA']) if 'TARGET_RA' in fibermap_cols else 0,
                            dec=float(fiber_info['TARGET_DEC']) if 'TARGET_DEC' in fibermap_cols else 0,
                            wavelength=w,
                            flux=float(flux[fiber_idx, p]),
                            significance=float(sig[p]),
                            n_fibers=int(other_peaks),
                            fwhm_pixels=fwhm,
                            band=arm
                        )))

            return healpix, n_sky, len(candidates), candidates

    except Exception as e:
        print(f"    Error processing {filepath.name}: {e}")
        return healpix, 0, 0, []

def process_available_files(manifest: Dict, survey: str = "main", program: str = "bright",
                            delete_after: bool = True) -> int:
    """Process all available files, update manifest, optionally delete after."""
    data_path = DATA_DIR / survey / program

    if not data_path.exists():
        print(f"Data path does not exist: {data_path}")
        return 0

    # Find all complete coadd files (> 100MB to skip incomplete)
    all_files = list(data_path.glob("**/coadd-*.fits"))
    complete_files = [f for f in all_files if f.stat().st_size > 100_000_000]

    # Filter to unprocessed
    processed = get_processed_healpix(manifest, survey, program)
    to_process = []
    for f in complete_files:
        match = re.search(r'coadd-\w+-\w+-(\d+)\.fits', f.name)
        if match:
            hp = int(match.group(1))
            if hp not in processed:
                to_process.append(f)

    if not to_process:
        print(f"No new files to process in {survey}/{program}")
        return 0

    print(f"Processing {len(to_process)} new files from {survey}/{program}")

    # Process in parallel
    n_workers = min(MAX_WORKERS, len(to_process))
    args_list = [(f, survey, program) for f in to_process]

    all_candidates = []

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_file, args_list)
    else:
        results = [process_single_file(args) for args in args_list]

    # Update manifest and collect candidates
    files_processed = 0
    for filepath, (healpix, n_sky, n_cand, candidates) in zip(to_process, results):
        if healpix > 0:
            mark_healpix_processed(manifest, healpix, survey, program, n_sky, n_cand)
            all_candidates.extend(candidates)
            files_processed += 1

            if n_cand > 0:
                manifest["stats"]["total_resolved"] += sum(1 for c in candidates if c['fwhm_pixels'] > 1)

            # Delete file after successful processing
            if delete_after and filepath.exists():
                try:
                    filepath.unlink()
                    # Also remove empty parent directories
                    parent = filepath.parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                except Exception as e:
                    print(f"    Warning: Could not delete {filepath}: {e}")

    # Save candidates to CSV
    if all_candidates:
        import pandas as pd
        df = pd.DataFrame(all_candidates)

        # Append to existing or create new
        if CANDIDATES_FILE.exists():
            existing = pd.read_csv(CANDIDATES_FILE)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(CANDIDATES_FILE, index=False)

    # Save manifest
    save_manifest(manifest)

    print(f"  Processed: {files_processed} files")
    print(f"  Candidates found: {len(all_candidates)}")
    print(f"  Total in manifest: {manifest['stats']['total_healpix']} healpix")

    return files_processed

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(survey: str = "main", program: str = "bright",
                 generate_batch: bool = True, process_local: bool = True,
                 delete_after: bool = True):
    """Run the full pipeline."""
    print("=" * 70)
    print("E9 LASER SEARCH PIPELINE")
    print("=" * 70)
    print(f"Start: {datetime.now()}")
    print(f"Survey: {survey}, Program: {program}")
    print(f"Delete after processing: {delete_after}")
    print()

    # Load manifest
    manifest = load_manifest()
    processed = get_processed_healpix(manifest, survey, program)
    print(f"Already processed: {len(processed)} healpix")
    print(f"Total candidates so far: {manifest['stats']['total_candidates']}")
    print()

    # Process any available local files first
    if process_local:
        print("=" * 70)
        print("PROCESSING LOCAL FILES")
        print("=" * 70)
        n_processed = process_available_files(manifest, survey, program, delete_after)
        print()

    # Generate batch for more data
    if generate_batch:
        print("=" * 70)
        print("GENERATING GLOBUS BATCH")
        print("=" * 70)
        batch_file, n_files = generate_globus_batch(manifest, survey, program)

        if batch_file and n_files > 0:
            print(f"\nTo start transfer, run:")
            print(f"  globus transfer {DESI_PUBLIC} {MSOUP_ENDPOINT} \\")
            print(f"    --batch {batch_file} --label 'E9-{survey}-{program}'")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    manifest = load_manifest()  # Reload
    print(f"Total healpix processed: {manifest['stats']['total_healpix']}")
    print(f"Total sky fibers scanned: {manifest['stats']['total_sky_fibers']}")
    print(f"Total candidates: {manifest['stats']['total_candidates']}")
    print(f"Total resolved (FWHM>1): {manifest['stats']['total_resolved']}")
    print(f"\nManifest: {MANIFEST_FILE}")
    print(f"Candidates: {CANDIDATES_FILE}")
    print(f"\nEnd: {datetime.now()}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="E9 Laser Search Pipeline")
    parser.add_argument("--survey", default="main", help="Survey name")
    parser.add_argument("--program", default="bright", help="Program name")
    parser.add_argument("--no-batch", action="store_true", help="Skip batch generation")
    parser.add_argument("--no-process", action="store_true", help="Skip local processing")
    parser.add_argument("--keep-files", action="store_true", help="Don't delete after processing")
    parser.add_argument("--status", action="store_true", help="Just show status")

    args = parser.parse_args()

    if args.status:
        manifest = load_manifest()
        print(json.dumps(manifest["stats"], indent=2))
        print(f"\nProcessed healpix: {len(manifest['processed_healpix'])}")
        return

    run_pipeline(
        survey=args.survey,
        program=args.program,
        generate_batch=not args.no_batch,
        process_local=not args.no_process,
        delete_after=not args.keep_files
    )

if __name__ == "__main__":
    main()
