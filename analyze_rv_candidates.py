#!/usr/bin/env python3
"""
analyze_rv_candidates.py - Production-quality RV candidate analysis for DESI DR1 MWS

Extends the smoke test with:
1. Quality filters (VRAD_ERR < 10, |VRAD| < 500, MJD_span > 0.5)
2. Significance metric S = ΔRV_max / sqrt(sum(VRAD_ERR^2))

Usage:
    python analyze_rv_candidates.py --data-root data --max-rows 1000000
    python analyze_rv_candidates.py --data-root data --file rvpix_exp-main-bright.fits

Output:
    data/derived/top_candidates_1M.csv (top 200 candidates)
"""

import argparse
import sys
from pathlib import Path
import warnings
import time

try:
    import fitsio
    USE_FITSIO = True
    FITSIO_BINARY_TBL = 2
except ImportError:
    USE_FITSIO = False
    from astropy.io import fits

import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# QUALITY FILTER CONSTANTS
# =============================================================================

# Filter 1: Maximum allowed RV uncertainty (km/s)
# Rationale: VRAD_ERR > 10 km/s indicates poor spectral fit or low SNR.
# Such measurements are unreliable for detecting true RV variability.
MAX_VRAD_ERR = 10.0

# Filter 2: Maximum allowed absolute RV (km/s)
# Rationale: |VRAD| > 500 km/s is unphysical for bound MW stars.
# These are typically failed fits, hot pixels, or cosmic rays.
MAX_ABS_VRAD = 500.0

# Filter 3: Minimum MJD span for multi-epoch (days)
# Rationale: Observations on the same night (< 0.5 days apart) may be
# from different spectrographs or exposures, not true repeat visits.
# We require actual multi-night coverage to claim RV variability.
MIN_MJD_SPAN = 0.5


# =============================================================================
# Column mappings
# =============================================================================

COLUMN_MAPPINGS = {
    "rv": ["VRAD", "RV", "VHELIO"],
    "rv_err": ["VRAD_ERR", "RV_ERR", "VHELIO_ERR"],
    "targetid": ["TARGETID", "TARGET_ID"],
    "gaia_id": ["SOURCE_ID", "GAIA_SOURCE_ID"],
    "mjd": ["MJD", "MJD_BEGIN"],
    "expid": ["EXPID", "EXP_ID"],
}


def find_column(columns, mapping_key):
    """Find the first matching column name from mappings."""
    candidates = COLUMN_MAPPINGS.get(mapping_key, [])
    columns_upper = {c.upper(): c for c in columns}
    for candidate in candidates:
        if candidate.upper() in columns_upper:
            return columns_upper[candidate.upper()]
    return None


# =============================================================================
# Data loading
# =============================================================================

def load_rv_data(filepath, max_rows=None):
    """
    Load RV data from FITS file.

    Returns dict with arrays: targetid, rv, rv_err, mjd, gaia_id
    """
    print(f"\n{'='*70}")
    print(f"Loading: {filepath.name}")
    print(f"{'='*70}")

    if USE_FITSIO:
        with fitsio.FITS(filepath, 'r') as f:
            # Find HDUs
            rvtab_idx = None
            fibermap_idx = None
            gaia_idx = None

            for i, hdu in enumerate(f):
                info = hdu.get_info()
                name = info.get('extname', '').upper()
                if 'RVTAB' in name:
                    rvtab_idx = i
                elif 'FIBERMAP' in name:
                    fibermap_idx = i
                elif 'GAIA' in name:
                    gaia_idx = i

            if rvtab_idx is None:
                print("ERROR: RVTAB HDU not found")
                return None

            # Get column names
            rvtab_cols = f[rvtab_idx].get_colnames()
            fibermap_cols = f[fibermap_idx].get_colnames() if fibermap_idx else []
            gaia_cols = f[gaia_idx].get_colnames() if gaia_idx else []

            # Find columns
            rv_col = find_column(rvtab_cols, 'rv')
            rv_err_col = find_column(rvtab_cols, 'rv_err')
            targetid_col = find_column(rvtab_cols, 'targetid')
            mjd_col = find_column(fibermap_cols, 'mjd') if fibermap_cols else None
            gaia_col = find_column(gaia_cols, 'gaia_id') if gaia_cols else None

            if rv_col is None or rv_err_col is None:
                print(f"ERROR: Missing required columns. Found: {rvtab_cols[:10]}")
                return None

            # Determine row count
            total_rows = f[rvtab_idx].get_info()['nrows']
            rows_to_read = min(total_rows, max_rows) if max_rows else total_rows

            print(f"  Total rows: {total_rows:,}")
            print(f"  Reading: {rows_to_read:,} rows")
            print(f"  Columns: RV={rv_col}, RV_ERR={rv_err_col}, TARGETID={targetid_col}")
            print(f"           MJD={mjd_col}, GAIA={gaia_col}")

            # Load data
            t0 = time.time()
            rv = f[rvtab_idx][rv_col][:rows_to_read]
            rv_err = f[rvtab_idx][rv_err_col][:rows_to_read]
            targetid = f[rvtab_idx][targetid_col][:rows_to_read] if targetid_col else np.arange(rows_to_read)

            mjd = f[fibermap_idx][mjd_col][:rows_to_read] if mjd_col else np.zeros(rows_to_read)
            gaia_id = f[gaia_idx][gaia_col][:rows_to_read] if gaia_col else None

            print(f"  Loaded in {time.time()-t0:.1f}s")
    else:
        # Astropy fallback
        with fits.open(filepath, memmap=True) as f:
            # Similar logic for astropy
            rvtab_data = f['RVTAB'].data
            total_rows = len(rvtab_data)
            rows_to_read = min(total_rows, max_rows) if max_rows else total_rows

            rv = np.array(rvtab_data['VRAD'][:rows_to_read])
            rv_err = np.array(rvtab_data['VRAD_ERR'][:rows_to_read])
            targetid = np.array(rvtab_data['TARGETID'][:rows_to_read])
            mjd = np.array(f['FIBERMAP'].data['MJD'][:rows_to_read])
            gaia_id = np.array(f['GAIA'].data['SOURCE_ID'][:rows_to_read])

    return {
        'targetid': targetid,
        'rv': rv,
        'rv_err': rv_err,
        'mjd': mjd,
        'gaia_id': gaia_id,
        'n_total': rows_to_read
    }


# =============================================================================
# Quality filtering (TASK 1)
# =============================================================================

def apply_quality_filters(data):
    """
    Apply per-measurement quality filters.

    Filters applied:
    1. VRAD_ERR < 10 km/s (exclude unreliable measurements)
    2. |VRAD| < 500 km/s (exclude unphysical velocities)

    Note: MJD_span filter is applied per-target later.
    """
    print(f"\n{'='*70}")
    print("Applying per-measurement quality filters")
    print(f"{'='*70}")

    rv = data['rv']
    rv_err = data['rv_err']

    n_initial = len(rv)

    # Filter 1: VRAD_ERR < MAX_VRAD_ERR
    # Rationale: High RV errors indicate poor fits
    mask_err = rv_err < MAX_VRAD_ERR
    n_after_err = mask_err.sum()
    print(f"  Filter 1 (VRAD_ERR < {MAX_VRAD_ERR} km/s): {n_initial:,} -> {n_after_err:,} ({100*n_after_err/n_initial:.1f}%)")

    # Filter 2: |VRAD| < MAX_ABS_VRAD
    # Rationale: Unphysical for bound MW stars
    mask_vrad = np.abs(rv) < MAX_ABS_VRAD
    n_after_vrad = mask_vrad.sum()
    print(f"  Filter 2 (|VRAD| < {MAX_ABS_VRAD} km/s): {n_initial:,} -> {n_after_vrad:,} ({100*n_after_vrad/n_initial:.1f}%)")

    # Combined mask
    mask = mask_err & mask_vrad & np.isfinite(rv) & np.isfinite(rv_err)
    n_final = mask.sum()
    print(f"  Combined: {n_initial:,} -> {n_final:,} ({100*n_final/n_initial:.1f}%)")

    return {
        'targetid': data['targetid'][mask],
        'rv': data['rv'][mask],
        'rv_err': data['rv_err'][mask],
        'mjd': data['mjd'][mask],
        'gaia_id': data['gaia_id'][mask] if data['gaia_id'] is not None else None,
        'n_after_quality': n_final
    }


# =============================================================================
# Multi-epoch grouping with MJD span filter (TASK 1 continued)
# =============================================================================

def group_multi_epoch_targets(data, min_epochs=2):
    """
    Group observations by TARGETID and apply MJD span filter.

    Filter 3: MJD_span > MIN_MJD_SPAN days
    Rationale: Same-night observations may be from different spectrographs,
    not true repeat visits. Require actual multi-night coverage.
    """
    print(f"\n{'='*70}")
    print("Grouping by TARGETID and applying MJD span filter")
    print(f"{'='*70}")

    targetid = data['targetid']
    rv = data['rv']
    rv_err = data['rv_err']
    mjd = data['mjd']
    gaia_id = data['gaia_id']

    # Get unique targets and counts
    t0 = time.time()
    unique_targets, inverse, counts = np.unique(targetid, return_inverse=True, return_counts=True)

    # Filter to targets with >= min_epochs observations
    multi_mask = counts >= min_epochs
    multi_targets = unique_targets[multi_mask]

    print(f"  Unique targets: {len(unique_targets):,}")
    print(f"  Targets with >= {min_epochs} epochs: {len(multi_targets):,}")

    # Build target set for fast lookup
    target_set = set(multi_targets)

    # Filter arrays to multi-epoch targets
    in_set = np.array([t in target_set for t in targetid])

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
    unique_sorted = np.unique(sorted_tid)

    # Build candidate list with MJD span filter
    candidates = []
    n_mjd_filtered = 0

    for i, tid in enumerate(unique_sorted):
        start = group_start[i]
        end = group_start[i+1]

        obs_rv = sorted_rv[start:end]
        obs_rv_err = sorted_rv_err[start:end]
        obs_mjd = sorted_mjd[start:end]
        obs_gaia = sorted_gaia[start] if sorted_gaia is not None else None

        mjd_span = float(np.max(obs_mjd) - np.min(obs_mjd))

        # Filter 3: MJD_span > MIN_MJD_SPAN
        # Rationale: Require true multi-night observations
        if mjd_span < MIN_MJD_SPAN:
            n_mjd_filtered += 1
            continue

        candidates.append({
            'targetid': tid,
            'gaia_id': obs_gaia,
            'rv': obs_rv,
            'rv_err': obs_rv_err,
            'mjd': obs_mjd,
            'n_epochs': len(obs_rv),
            'mjd_span': mjd_span
        })

    print(f"  Filter 3 (MJD_span > {MIN_MJD_SPAN} days): removed {n_mjd_filtered:,} same-night targets")
    print(f"  Candidates surviving all filters: {len(candidates):,}")
    print(f"  Grouping took {time.time()-t0:.2f}s")

    return candidates


# =============================================================================
# Significance metric computation (TASK 2)
# =============================================================================

def compute_significance_metric(candidates):
    """
    Compute significance metric for each candidate.

    S = ΔRV_max / sqrt(sum(VRAD_ERR^2))

    This metric:
    - Rewards large RV amplitude (ΔRV_max)
    - Penalizes noisy measurements (high VRAD_ERR)
    - Naturally down-ranks pathological fits with large errors

    In one sentence: S measures how many combined sigma the RV varies by.
    """
    print(f"\n{'='*70}")
    print("Computing significance metric S = ΔRV_max / sqrt(sum(VRAD_ERR^2))")
    print(f"{'='*70}")

    t0 = time.time()

    for c in candidates:
        rv = c['rv']
        rv_err = c['rv_err']

        # ΔRV_max = max - min
        delta_rv = float(np.max(rv) - np.min(rv))

        # Combined error = sqrt(sum of squared errors)
        combined_err = float(np.sqrt(np.sum(rv_err**2)))

        # Significance S
        if combined_err > 0:
            S = delta_rv / combined_err
        else:
            S = 0.0

        c['delta_rv'] = delta_rv
        c['combined_err'] = combined_err
        c['S'] = S
        c['rv_err_median'] = float(np.median(rv_err))
        c['rv_median'] = float(np.median(rv))

    # Sort by S descending
    candidates.sort(key=lambda x: x['S'], reverse=True)

    print(f"  Computed S for {len(candidates):,} candidates in {time.time()-t0:.2f}s")

    # Report distribution
    S_values = np.array([c['S'] for c in candidates])
    n_gt_5 = np.sum(S_values > 5)
    n_gt_10 = np.sum(S_values > 10)
    n_gt_20 = np.sum(S_values > 20)

    print(f"\n  S distribution:")
    print(f"    S > 5:  {n_gt_5:,} candidates")
    print(f"    S > 10: {n_gt_10:,} candidates")
    print(f"    S > 20: {n_gt_20:,} candidates")

    return candidates, {'n_gt_5': n_gt_5, 'n_gt_10': n_gt_10, 'n_gt_20': n_gt_20}


# =============================================================================
# Output
# =============================================================================

def print_top_candidates(candidates, n=20):
    """Print top N candidates."""
    print(f"\n{'='*70}")
    print(f"Top {n} Candidates by Significance (S)")
    print(f"{'='*70}")

    print(f"\n{'TARGETID':<22} {'Gaia SOURCE_ID':<22} {'S':>8} {'ΔRV':>8} {'RV_err':>8} {'N_ep':>6} {'MJD_span':>10}")
    print("-" * 95)

    for c in candidates[:n]:
        gaia_str = str(c['gaia_id']) if c['gaia_id'] and c['gaia_id'] > 0 else "N/A"
        print(f"{c['targetid']:<22} {gaia_str:<22} {c['S']:>8.2f} {c['delta_rv']:>8.2f} {c['rv_err_median']:>8.2f} {c['n_epochs']:>6} {c['mjd_span']:>10.1f}")


def save_candidates_csv(candidates, output_path, n=200):
    """Save top N candidates to CSV."""
    print(f"\n  Saving top {n} candidates to: {output_path}")

    with open(output_path, 'w') as f:
        f.write("rank,targetid,gaia_source_id,S,delta_rv_kms,rv_err_median_kms,n_epochs,mjd_span_days,rv_median_kms\n")
        for i, c in enumerate(candidates[:n]):
            gaia_str = str(c['gaia_id']) if c['gaia_id'] and c['gaia_id'] > 0 else ""
            f.write(f"{i+1},{c['targetid']},{gaia_str},{c['S']:.4f},{c['delta_rv']:.4f},{c['rv_err_median']:.4f},{c['n_epochs']},{c['mjd_span']:.4f},{c['rv_median']:.4f}\n")

    print(f"  Saved!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DESI DR1 MWS RV Candidate Analysis")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--file", default=None, help="Specific FITS file to analyze")
    parser.add_argument("--max-rows", type=int, default=1_000_000, help="Max rows to read")
    parser.add_argument("--min-epochs", type=int, default=2, help="Minimum epochs required")
    parser.add_argument("--top-n", type=int, default=20, help="Top N to print")
    parser.add_argument("--save-n", type=int, default=200, help="Top N to save to CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("DESI DR1 MWS RV Candidate Analysis")
    print("=" * 70)
    print(f"Quality filters:")
    print(f"  - VRAD_ERR < {MAX_VRAD_ERR} km/s")
    print(f"  - |VRAD| < {MAX_ABS_VRAD} km/s")
    print(f"  - MJD_span > {MIN_MJD_SPAN} days")
    print(f"Ranking: S = ΔRV_max / sqrt(sum(VRAD_ERR^2))")

    t_start = time.time()

    # Find data file
    raw_dir = Path(args.data_root) / "raw"

    if args.file:
        filepath = raw_dir / args.file
    else:
        # Priority: main-bright > special-bright
        for fname in ["rvpix_exp-main-bright.fits", "rvpix_exp-special-bright.fits"]:
            if (raw_dir / fname).exists():
                filepath = raw_dir / fname
                break
        else:
            print(f"ERROR: No FITS files found in {raw_dir}")
            sys.exit(1)

    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    print(f"\nInput: {filepath}")
    print(f"Max rows: {args.max_rows:,}")

    # Load data
    data = load_rv_data(filepath, max_rows=args.max_rows)
    if data is None:
        sys.exit(1)

    # Apply quality filters
    filtered_data = apply_quality_filters(data)

    # Group by target and apply MJD span filter
    candidates = group_multi_epoch_targets(filtered_data, min_epochs=args.min_epochs)

    if not candidates:
        print("\nERROR: No candidates surviving filters")
        sys.exit(1)

    # Compute significance metric
    candidates, stats = compute_significance_metric(candidates)

    # Print top candidates
    print_top_candidates(candidates, n=args.top_n)

    # Save results
    output_dir = Path(args.data_root) / "derived"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "top_candidates_1M.csv"
    save_candidates_csv(candidates, output_path, n=args.save_n)

    # Summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")
    print(f"\nSummary:")
    print(f"  Total RV measurements processed: {data['n_total']:,}")
    print(f"  After per-measurement quality cuts: {filtered_data['n_after_quality']:,}")
    print(f"  Targets surviving all filters: {len(candidates):,}")
    print(f"  Candidates with S > 5: {stats['n_gt_5']:,}")
    print(f"  Candidates with S > 10: {stats['n_gt_10']:,}")
    print(f"  Candidates with S > 20: {stats['n_gt_20']:,}")
    print(f"\nOutput: {output_path}")

    if candidates:
        print(f"\nTop candidate: TARGETID={candidates[0]['targetid']}, S={candidates[0]['S']:.2f}, ΔRV={candidates[0]['delta_rv']:.2f} km/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
