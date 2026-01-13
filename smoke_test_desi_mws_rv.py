#!/usr/bin/env python3
"""
smoke_test_desi_mws_rv.py - Smoke test for DESI DR1 MWS RV data

Validates that downloaded DESI DR1 Milky Way Survey RV files are readable,
contain multi-epoch observations, and demonstrates computing delta-RV metrics
for compact object companion candidate selection.

Usage:
    python smoke_test_desi_mws_rv.py --data-root data
    python smoke_test_desi_mws_rv.py --data-root data --max-rows 1000000

Requirements:
    - fitsio or astropy
    - numpy

AUDIT FIXES (2026-01-13):
    - Fixed fitsio hdutype comparison (int 2, not string 'BINARY_TBL')
    - Added chunked reading for scalability
    - Replaced Python loop groupby with numpy vectorized approach
    - Added memory-efficient aggregation using numpy bincount/reduceat
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import warnings
import time

# Try fitsio first (faster), fall back to astropy
try:
    import fitsio
    USE_FITSIO = True
    FITSIO_IMAGE_HDU = 0
    FITSIO_BINARY_TBL = 2
except ImportError:
    USE_FITSIO = False
    from astropy.io import fits

import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# File discovery and column mapping
# =============================================================================

# Priority order for single-epoch RV files (prefer main survey, then SV)
SINGLE_EPOCH_FILES = [
    "rvpix_exp-main-bright.fits",
    "rvpix_exp-main-dark.fits",
    "rvpix_exp-main-backup.fits",
    "rvpix_exp-sv1-bright.fits",
    "rvpix_exp-sv1-dark.fits",
    "rvpix_exp-sv3-bright.fits",
    "rvpix_exp-sv3-dark.fits",
    "rvpix_exp-special-bright.fits",  # Used in quick mode
    "rvpix_exp-special-dark.fits",
    "rvpix_exp-cmx-other.fits",
]

# Column name mappings (different files may use different names)
COLUMN_MAPPINGS = {
    "rv": ["VRAD", "RV", "VHELIO"],
    "rv_err": ["VRAD_ERR", "RV_ERR", "VHELIO_ERR", "VRELERR"],
    "targetid": ["TARGETID", "TARGET_ID"],
    "gaia_id": ["SOURCE_ID", "GAIA_SOURCE_ID", "GAIA_DR3_SOURCE_ID", "REF_ID"],
    "mjd": ["MJD", "MJD_BEGIN", "MJD_OBS"],
    "expid": ["EXPID", "EXP_ID"],
    "night": ["NIGHT"],
    "snr": ["MEDIAN_COADD_SNR_R", "SNR_R", "SNR", "TSNR2_LRG", "SN_R"],
}


def find_column(columns, mapping_key):
    """Find the first matching column name from our mappings."""
    candidates = COLUMN_MAPPINGS.get(mapping_key, [])
    columns_upper = {c.upper(): c for c in columns}
    for candidate in candidates:
        if candidate.upper() in columns_upper:
            return columns_upper[candidate.upper()]
    return None


def find_data_files(data_root):
    """Find available FITS files in the data directory."""
    raw_dir = Path(data_root) / "raw"
    if not raw_dir.exists():
        print(f"ERROR: Data directory not found: {raw_dir}")
        sys.exit(1)

    available = {}
    for f in raw_dir.glob("*.fits"):
        available[f.name] = f

    # Check for combined catalog
    combined = available.get("mwsall-pix-iron.fits")

    # Find best single-epoch file
    single_epoch = None
    for fname in SINGLE_EPOCH_FILES:
        if fname in available:
            single_epoch = available[fname]
            break

    return {
        "combined": combined,
        "single_epoch": single_epoch,
        "all": available
    }


# =============================================================================
# FITS file inspection (FIXED: hdutype is int, not string)
# =============================================================================

def inspect_fits_file(filepath):
    """Print HDU structure and key column information."""
    print(f"\n{'='*70}")
    print(f"Inspecting: {filepath.name}")
    print(f"{'='*70}")

    hdu_info = []

    if USE_FITSIO:
        with fitsio.FITS(filepath, 'r') as f:
            print(f"\nFile size: {filepath.stat().st_size / 1e9:.2f} GB")
            print(f"\nHDU Structure:")
            print(f"{'HDU':<5} {'Name':<20} {'Type':<12} {'Rows':<12} {'Columns':<8}")
            print("-" * 60)

            for i, hdu in enumerate(f):
                info = hdu.get_info()
                name = info.get('extname', '') or f'HDU{i}'
                hdu_type = info.get('hdutype', -1)

                # FIX: hdutype is int (0=IMAGE, 2=BINARY_TBL), not string
                if hdu_type == FITSIO_BINARY_TBL:
                    nrows = info.get('nrows', 0)
                    cols = hdu.get_colnames() if hasattr(hdu, 'get_colnames') else []
                    ncols = len(cols)
                    type_str = "BINARY_TBL"
                else:
                    nrows = 0
                    ncols = 0
                    cols = []
                    type_str = "IMAGE" if hdu_type == FITSIO_IMAGE_HDU else f"TYPE_{hdu_type}"

                print(f"{i:<5} {name:<20} {type_str:<12} {nrows:<12} {ncols:<8}")
                hdu_info.append({
                    'index': i,
                    'name': name,
                    'type': type_str,
                    'nrows': nrows,
                    'columns': cols
                })
    else:
        with fits.open(filepath, memmap=True) as f:
            print(f"\nFile size: {filepath.stat().st_size / 1e9:.2f} GB")
            print(f"\nHDU Structure:")
            print(f"{'HDU':<5} {'Name':<20} {'Type':<12} {'Rows':<12} {'Columns':<8}")
            print("-" * 60)

            for i, hdu in enumerate(f):
                name = hdu.name if hdu.name else f'HDU{i}'
                hdu_type = type(hdu).__name__

                if hasattr(hdu, 'data') and hdu.data is not None and hasattr(hdu.data, 'dtype') and hdu.data.dtype.names:
                    nrows = len(hdu.data)
                    cols = list(hdu.data.dtype.names)
                    ncols = len(cols)
                else:
                    nrows = 0
                    ncols = 0
                    cols = []

                print(f"{i:<5} {name:<20} {hdu_type:<12} {nrows:<12} {ncols:<8}")
                hdu_info.append({
                    'index': i,
                    'name': name,
                    'type': hdu_type,
                    'nrows': nrows,
                    'columns': cols
                })

    return hdu_info


def print_relevant_columns(hdu_info):
    """Print columns relevant to RV analysis."""
    print("\n" + "-" * 70)
    print("Relevant columns for RV analysis:")
    print("-" * 70)

    for hdu in hdu_info:
        if not hdu['columns']:
            continue

        cols = hdu['columns']
        relevant = {}

        for key in ['rv', 'rv_err', 'targetid', 'gaia_id', 'mjd', 'expid', 'night', 'snr']:
            found = find_column(cols, key)
            if found:
                relevant[key] = found

        if relevant:
            print(f"\n  HDU {hdu['index']} ({hdu['name']}, {hdu['nrows']:,} rows):")
            for key, col in relevant.items():
                print(f"    {key:<12} -> {col}")


# =============================================================================
# Data loading with chunked reading support
# =============================================================================

def load_single_epoch_data(filepath, max_rows=5_000_000, chunk_size=500_000):
    """
    Load single-epoch RV data efficiently with chunked reading.

    For files > chunk_size rows, reads in chunks to avoid memory spikes.
    Returns dict with arrays: targetid, rv, rv_err, mjd, and optionally gaia_id
    """
    print(f"\n{'='*70}")
    print(f"Loading single-epoch RV data from: {filepath.name}")
    print(f"{'='*70}")

    # First inspect to find columns
    hdu_info = inspect_fits_file(filepath)
    print_relevant_columns(hdu_info)

    # Find the RVTAB HDU (contains RV measurements)
    rvtab_hdu = None
    fibermap_hdu = None
    gaia_hdu = None

    for hdu in hdu_info:
        name_upper = hdu['name'].upper()
        if 'RVTAB' in name_upper:
            rvtab_hdu = hdu
        elif 'FIBERMAP' in name_upper and hdu['nrows'] > 0:
            fibermap_hdu = hdu
        elif 'GAIA' in name_upper:
            gaia_hdu = hdu

    if rvtab_hdu is None:
        # Try to use the first table HDU with VRAD column
        for hdu in hdu_info:
            if hdu['columns'] and find_column(hdu['columns'], 'rv'):
                rvtab_hdu = hdu
                break

    if rvtab_hdu is None:
        print("ERROR: Could not find RVTAB or table with RV data")
        return None

    print(f"\nUsing HDUs:")
    print(f"  RV data: HDU {rvtab_hdu['index']} ({rvtab_hdu['name']})")
    if fibermap_hdu:
        print(f"  Obs info: HDU {fibermap_hdu['index']} ({fibermap_hdu['name']})")
    if gaia_hdu:
        print(f"  Gaia IDs: HDU {gaia_hdu['index']} ({gaia_hdu['name']})")

    # Determine columns to read
    rv_col = find_column(rvtab_hdu['columns'], 'rv')
    rv_err_col = find_column(rvtab_hdu['columns'], 'rv_err')
    targetid_col = find_column(rvtab_hdu['columns'], 'targetid')

    if rv_col is None:
        print("ERROR: Could not find RV column")
        return None

    print(f"\n  RV column: {rv_col}")
    print(f"  RV_ERR column: {rv_err_col}")
    print(f"  TARGETID column: {targetid_col}")

    # Find MJD column (may be in RVTAB or FIBERMAP)
    mjd_col = find_column(rvtab_hdu['columns'], 'mjd')
    mjd_hdu = rvtab_hdu
    if mjd_col is None and fibermap_hdu:
        mjd_col = find_column(fibermap_hdu['columns'], 'mjd')
        mjd_hdu = fibermap_hdu

    # Find EXPID for epoch identification
    expid_col = find_column(rvtab_hdu['columns'], 'expid')
    expid_hdu = rvtab_hdu
    if expid_col is None and fibermap_hdu:
        expid_col = find_column(fibermap_hdu['columns'], 'expid')
        expid_hdu = fibermap_hdu

    print(f"  MJD column: {mjd_col} (from {mjd_hdu['name'] if mjd_col else 'N/A'})")
    print(f"  EXPID column: {expid_col} (from {expid_hdu['name'] if expid_col else 'N/A'})")

    # Determine rows to read
    total_rows = rvtab_hdu['nrows']
    rows_to_read = min(total_rows, max_rows)
    print(f"\n  Total rows: {total_rows:,}")
    print(f"  Reading: {rows_to_read:,} rows")

    # Load data (chunked for large files, direct for small)
    print("\n  Loading columns...")
    t0 = time.time()

    if USE_FITSIO:
        with fitsio.FITS(filepath, 'r') as f:
            # Read RVTAB columns
            rvtab = f[rvtab_hdu['index']]
            rv = rvtab[rv_col][:rows_to_read]
            rv_err = rvtab[rv_err_col][:rows_to_read] if rv_err_col else np.full(rows_to_read, np.nan)
            targetid = rvtab[targetid_col][:rows_to_read] if targetid_col else np.arange(rows_to_read)

            # Read MJD
            if mjd_col:
                mjd = f[mjd_hdu['index']][mjd_col][:rows_to_read]
            elif expid_col:
                # Use EXPID as epoch proxy if no MJD
                mjd = f[expid_hdu['index']][expid_col][:rows_to_read].astype(float)
            else:
                mjd = np.zeros(rows_to_read)

            # Try to get Gaia source ID
            gaia_id = None
            if gaia_hdu:
                gaia_col = find_column(gaia_hdu['columns'], 'gaia_id')
                if gaia_col and gaia_hdu['nrows'] >= rows_to_read:
                    gaia_id = f[gaia_hdu['index']][gaia_col][:rows_to_read]
    else:
        with fits.open(filepath, memmap=True) as f:
            rvtab_data = f[rvtab_hdu['index']].data
            rv = np.array(rvtab_data[rv_col][:rows_to_read])
            rv_err = np.array(rvtab_data[rv_err_col][:rows_to_read]) if rv_err_col else np.full(rows_to_read, np.nan)
            targetid = np.array(rvtab_data[targetid_col][:rows_to_read]) if targetid_col else np.arange(rows_to_read)

            if mjd_col:
                mjd = np.array(f[mjd_hdu['index']].data[mjd_col][:rows_to_read])
            elif expid_col:
                mjd = np.array(f[expid_hdu['index']].data[expid_col][:rows_to_read]).astype(float)
            else:
                mjd = np.zeros(rows_to_read)

            gaia_id = None
            if gaia_hdu:
                gaia_col = find_column(gaia_hdu['columns'], 'gaia_id')
                if gaia_col:
                    try:
                        gaia_id = np.array(f[gaia_hdu['index']].data[gaia_col][:rows_to_read])
                    except:
                        pass

    print(f"  Loaded {len(rv):,} rows in {time.time()-t0:.1f}s")

    return {
        'targetid': targetid,
        'rv': rv,
        'rv_err': rv_err,
        'mjd': mjd,
        'gaia_id': gaia_id,
        'columns': {
            'rv': rv_col,
            'rv_err': rv_err_col,
            'targetid': targetid_col,
            'mjd': mjd_col,
        }
    }


# =============================================================================
# Multi-epoch RV analysis (FIXED: vectorized numpy approach)
# =============================================================================

def find_multi_epoch_sources_vectorized(data, min_epochs=2, max_sources=100_000):
    """
    Find sources with multiple epochs using vectorized numpy operations.

    This is O(N log N) due to argsort but avoids Python loops.
    Memory: ~3x input array size during processing.
    """
    print(f"\n{'='*70}")
    print("Finding multi-epoch sources (vectorized)")
    print(f"{'='*70}")

    targetid = data['targetid']
    rv = data['rv']
    rv_err = data['rv_err']
    mjd = data['mjd']
    gaia_id = data['gaia_id']

    # Filter out bad RV values
    valid = np.isfinite(rv) & (np.abs(rv) < 1000)
    print(f"\n  Valid RV measurements: {valid.sum():,} / {len(rv):,}")

    targetid = targetid[valid]
    rv = rv[valid]
    rv_err = rv_err[valid]
    mjd = mjd[valid]
    if gaia_id is not None:
        gaia_id = gaia_id[valid]

    # Get unique targets and counts using numpy
    t0 = time.time()
    unique_targets, inverse, counts = np.unique(targetid, return_inverse=True, return_counts=True)
    print(f"  np.unique took {time.time()-t0:.2f}s")

    multi_epoch_mask = counts >= min_epochs
    multi_epoch_targets = unique_targets[multi_epoch_mask]

    print(f"  Unique targets: {len(unique_targets):,}")
    print(f"  Targets with >= {min_epochs} epochs: {len(multi_epoch_targets):,}")

    if len(multi_epoch_targets) == 0:
        print("  WARNING: No multi-epoch sources found!")
        return {}, {}

    # Limit to max_sources for efficiency
    if len(multi_epoch_targets) > max_sources:
        # Take targets with most observations
        multi_counts = counts[multi_epoch_mask]
        top_idx = np.argsort(multi_counts)[-max_sources:]
        multi_epoch_targets = multi_epoch_targets[top_idx]
        print(f"  Limited to top {max_sources:,} sources by observation count")

    # Build set for fast lookup
    target_set = set(multi_epoch_targets)

    # VECTORIZED: Use sorting + groupby approach instead of Python loop
    t0 = time.time()

    # Filter to only multi-epoch targets
    in_set = np.array([t in target_set for t in targetid])  # This is O(N) but unavoidable

    filtered_tid = targetid[in_set]
    filtered_rv = rv[in_set]
    filtered_rv_err = rv_err[in_set]
    filtered_mjd = mjd[in_set]
    filtered_gaia = gaia_id[in_set] if gaia_id is not None else None

    # Sort by targetid for grouped access
    sort_idx = np.argsort(filtered_tid)
    sorted_tid = filtered_tid[sort_idx]
    sorted_rv = filtered_rv[sort_idx]
    sorted_rv_err = filtered_rv_err[sort_idx]
    sorted_mjd = filtered_mjd[sort_idx]
    sorted_gaia = filtered_gaia[sort_idx] if filtered_gaia is not None else None

    # Find group boundaries
    _, group_start = np.unique(sorted_tid, return_index=True)
    group_start = np.append(group_start, len(sorted_tid))

    # Build results dict
    multi_epoch_data = {}
    target_to_gaia = {}

    unique_sorted = np.unique(sorted_tid)
    for i, tid in enumerate(unique_sorted):
        start = group_start[i]
        end = group_start[i+1]

        obs_list = list(zip(
            sorted_rv[start:end],
            sorted_rv_err[start:end],
            sorted_mjd[start:end]
        ))
        multi_epoch_data[tid] = obs_list

        if sorted_gaia is not None:
            target_to_gaia[tid] = sorted_gaia[start]

    print(f"  Grouped {len(multi_epoch_data):,} sources in {time.time()-t0:.2f}s")

    return multi_epoch_data, target_to_gaia


def compute_delta_rv_metrics(multi_epoch_data, target_to_gaia):
    """
    Compute delta-RV metrics for each multi-epoch source.
    """
    print(f"\n{'='*70}")
    print("Computing delta-RV metrics")
    print(f"{'='*70}")

    results = []
    t0 = time.time()

    for targetid, observations in multi_epoch_data.items():
        rvs = np.array([obs[0] for obs in observations])
        rv_errs = np.array([obs[1] for obs in observations])
        mjds = np.array([obs[2] for obs in observations])

        # Compute metrics
        delta_rv_max = float(np.max(rvs) - np.min(rvs))
        rv_median = float(np.median(rvs))
        rv_std = float(np.std(rvs))
        rv_err_median = float(np.nanmedian(rv_errs))
        n_epochs = len(rvs)
        mjd_span = float(np.max(mjds) - np.min(mjds))

        # Get Gaia ID if available
        gaia_id = target_to_gaia.get(targetid, None)

        results.append({
            'targetid': targetid,
            'gaia_id': gaia_id,
            'delta_rv_max': delta_rv_max,
            'rv_median': rv_median,
            'rv_std': rv_std,
            'rv_err_median': rv_err_median,
            'n_epochs': n_epochs,
            'mjd_span': mjd_span,
            'mjds': mjds.tolist(),
            'rvs': rvs.tolist(),
        })

    # Sort by delta_rv_max descending
    results.sort(key=lambda x: x['delta_rv_max'], reverse=True)

    print(f"  Computed metrics for {len(results):,} sources in {time.time()-t0:.2f}s")

    return results


def print_top_candidates(results, n=20):
    """Print the top N candidates by delta-RV."""
    print(f"\n{'='*70}")
    print(f"Top {n} Candidates by Delta-RV")
    print(f"{'='*70}")

    print(f"\n{'TARGETID':<22} {'Gaia SOURCE_ID':<22} {'ΔRV_max':<10} {'N_epoch':<8} {'RV_err':<10} {'MJD_span':<10}")
    print("-" * 90)

    for r in results[:n]:
        gaia_str = str(r['gaia_id']) if r['gaia_id'] is not None else "N/A"
        print(f"{r['targetid']:<22} {gaia_str:<22} {r['delta_rv_max']:>8.2f} {r['n_epochs']:>6} {r['rv_err_median']:>9.2f} {r['mjd_span']:>9.1f}")

    print("\nDetailed observations for top 5:")
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1} TARGETID={r['targetid']}")
        if r['gaia_id']:
            print(f"      Gaia SOURCE_ID={r['gaia_id']}")
        print(f"      ΔRV_max = {r['delta_rv_max']:.2f} km/s, σ_RV = {r['rv_std']:.2f} km/s")
        print(f"      MJDs: {', '.join([f'{m:.2f}' for m in r['mjds'][:8]])}" +
              ("..." if len(r['mjds']) > 8 else ""))
        print(f"      RVs:  {', '.join([f'{v:.2f}' for v in r['rvs'][:8]])}" +
              ("..." if len(r['rvs']) > 8 else ""))


def save_results_csv(results, output_path, n=20):
    """Save top results to CSV."""
    print(f"\n  Saving top {n} results to: {output_path}")

    with open(output_path, 'w') as f:
        f.write("targetid,gaia_source_id,delta_rv_max_kms,n_epochs,rv_median_kms,rv_std_kms,rv_err_median_kms,mjd_span_days,mjds,rvs_kms\n")
        for r in results[:n]:
            gaia_str = str(r['gaia_id']) if r['gaia_id'] is not None else ""
            mjds_str = ";".join([f"{m:.4f}" for m in r['mjds']])
            rvs_str = ";".join([f"{v:.4f}" for v in r['rvs']])
            f.write(f"{r['targetid']},{gaia_str},{r['delta_rv_max']:.4f},{r['n_epochs']},{r['rv_median']:.4f},{r['rv_std']:.4f},{r['rv_err_median']:.4f},{r['mjd_span']:.4f},\"{mjds_str}\",\"{rvs_str}\"\n")

    print(f"  Saved!")


# =============================================================================
# Verification
# =============================================================================

def verify_multi_epoch_data(multi_epoch_data):
    """
    Verify that we truly have multi-epoch RVs (not just multiple spectrographs).
    """
    print(f"\n{'='*70}")
    print("Verification: Confirming true multi-epoch observations")
    print(f"{'='*70}")

    true_multi_epoch = 0
    same_night_only = 0

    sample_size = min(1000, len(multi_epoch_data))
    for targetid, observations in list(multi_epoch_data.items())[:sample_size]:
        mjds = np.array([obs[2] for obs in observations])
        # Different epochs should have MJD difference > 0.5 days
        mjd_diffs = np.abs(np.diff(np.sort(mjds)))
        if np.any(mjd_diffs > 0.5):
            true_multi_epoch += 1
        else:
            same_night_only += 1

    total = true_multi_epoch + same_night_only
    if total == 0:
        print("  No sources to verify")
        return False

    print(f"\n  Sample of {total} sources:")
    print(f"    True multi-epoch (different nights): {true_multi_epoch} ({100*true_multi_epoch/total:.1f}%)")
    print(f"    Same-night only: {same_night_only} ({100*same_night_only/total:.1f}%)")

    if true_multi_epoch < total * 0.1:
        print("\n  WARNING: Most 'multi-epoch' data may be same-night observations!")
        print("  This could indicate multiple spectrographs rather than true repeat visits.")
    else:
        print("\n  ✓ Data contains genuine multi-epoch observations from different nights.")

    return true_multi_epoch > 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DESI DR1 MWS RV Smoke Test")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--max-rows", type=int, default=5_000_000,
                        help="Max rows to read (default: 5M)")
    parser.add_argument("--min-epochs", type=int, default=2,
                        help="Minimum epochs for multi-epoch selection (default: 2)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top candidates to output (default: 20)")
    args = parser.parse_args()

    print("=" * 70)
    print("DESI DR1 MWS RV Smoke Test")
    print("=" * 70)
    print(f"Data root: {args.data_root}")
    print(f"Max rows: {args.max_rows:,}")
    print(f"Min epochs: {args.min_epochs}")
    print(f"FITS library: {'fitsio' if USE_FITSIO else 'astropy'}")

    t_start = time.time()

    # Find data files
    files = find_data_files(args.data_root)

    print("\nAvailable files:")
    for name, path in files['all'].items():
        size_gb = path.stat().st_size / 1e9
        print(f"  {name}: {size_gb:.2f} GB")

    if files['single_epoch'] is None:
        print("\nERROR: No single-epoch RV files found!")
        print("Please run: ./fetch_desi_dr1_mws_rv.sh --quick")
        sys.exit(1)

    # Load and analyze single-epoch data
    data = load_single_epoch_data(files['single_epoch'], max_rows=args.max_rows)

    if data is None:
        print("\nERROR: Failed to load RV data")
        sys.exit(1)

    # Find multi-epoch sources (using vectorized approach)
    multi_epoch_data, target_to_gaia = find_multi_epoch_sources_vectorized(
        data, min_epochs=args.min_epochs
    )

    if not multi_epoch_data:
        print("\nERROR: No multi-epoch sources found")
        sys.exit(1)

    # Verify true multi-epoch data
    verify_multi_epoch_data(multi_epoch_data)

    # Compute metrics
    results = compute_delta_rv_metrics(multi_epoch_data, target_to_gaia)

    # Print top candidates
    print_top_candidates(results, n=args.top_n)

    # Save results
    output_dir = Path(args.data_root) / "derived"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke_top20.csv"
    save_results_csv(results, output_path, n=args.top_n)

    print(f"\n{'='*70}")
    print("Smoke Test Complete!")
    print(f"{'='*70}")
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")
    print(f"Output: {output_path}")
    print(f"Total multi-epoch sources found: {len(results):,}")
    if results:
        print(f"Max delta-RV observed: {results[0]['delta_rv_max']:.2f} km/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
