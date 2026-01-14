#!/usr/bin/env python3
"""
triage_rv_candidates.py - Robust triage of RV variability candidates

This script performs leave-one-out analysis and outlier detection to separate
real RV variability from two-point flukes or single bad epochs.

DEFINITIONS (from the RV time series):
- RV_i   = the i-th radial velocity measurement for a target (km/s)
- σ_i    = the i-th RV uncertainty (VRAD_ERR) (km/s)
- MJD_i  = the i-th observation time (days)
- N      = number of epochs for that target

CURRENT SCORE (from previous analysis):
- ΔRV_max = max(RV_i) − min(RV_i)
- denom   = sqrt( Σ_{i=1..N} (σ_i)^2 )
- S       = ΔRV_max / denom

NEW ROBUST DIAGNOSTICS:
1) Leave-one-out stability:
   For each k in 1..N, compute S_{−k} using all epochs except epoch k.
   - S_min_LOO = min_k S_{−k}  (worst-case when you drop an epoch)
   - S_med_LOO = median_k S_{−k}
   A candidate driven by a single bad epoch will collapse when that epoch is dropped.

2) Outlier leverage:
   d_i = |RV_i − median(RV)| / σ_i   (normalized distance from median)
   d_max = max_i d_i
   A single epoch with huge normalized deviation suggests a bad fit or outlier.

3) Conservative ranking score:
   S_robust = min(S, S_min_LOO)   for N_epochs >= 3
   S_robust = 0                    for N_epochs == 2 (cannot compute LOO stability)

   Rationale: With only 2 epochs, dropping one leaves N=1 and no variability measure.
   These are kept in a separate Tier B for follow-up observations.

TIERS:
- Tier A: N_epochs >= 3, ranked by S_robust descending
- Tier B: N_epochs == 2, ranked by S descending, labeled "needs 1 more epoch"

Author: Claude (Anthropic)
Date: 2026-01-13
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import fitsio
    USE_FITSIO = True
except ImportError:
    from astropy.io import fits
    USE_FITSIO = False

# Optional matplotlib for plots
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# QUALITY FILTER CONSTANTS (must match analyze_rv_candidates.py)
# =============================================================================

# Filter 1: Maximum allowed RV uncertainty (km/s)
MAX_VRAD_ERR = 10.0

# Filter 2: Maximum allowed absolute RV (km/s)
MAX_ABS_VRAD = 500.0

# Sentinel value for missing Gaia SOURCE_ID in DESI files
GAIA_MISSING_SENTINEL = -9999


def compute_S(rv_arr, err_arr):
    """
    Compute significance score S for an RV time series.

    S = ΔRV_max / sqrt(Σ σ_i^2)

    Parameters
    ----------
    rv_arr : array-like
        Radial velocity measurements (km/s)
    err_arr : array-like
        RV uncertainties (km/s)

    Returns
    -------
    float
        Significance score S, or 0.0 if less than 2 valid epochs
    """
    rv = np.asarray(rv_arr)
    err = np.asarray(err_arr)

    if len(rv) < 2:
        return 0.0

    delta_rv = np.max(rv) - np.min(rv)
    denom = np.sqrt(np.sum(err**2))

    if denom <= 0:
        return 0.0

    return delta_rv / denom


def leave_one_out_scores(rv_arr, err_arr):
    """
    Compute leave-one-out S scores.

    For each epoch k, compute S_{-k} using all epochs except k.

    Parameters
    ----------
    rv_arr : array-like
        Radial velocity measurements
    err_arr : array-like
        RV uncertainties

    Returns
    -------
    tuple (S_min_LOO, S_med_LOO, S_all_LOO)
        - S_min_LOO: minimum S when any single epoch is dropped
        - S_med_LOO: median S across all leave-one-out trials
        - S_all_LOO: array of all LOO scores
    """
    rv = np.asarray(rv_arr)
    err = np.asarray(err_arr)
    n = len(rv)

    if n < 3:
        # Cannot compute meaningful LOO with only 2 epochs
        # Dropping one leaves only 1 epoch -> no variability measure
        return np.nan, np.nan, np.array([])

    loo_scores = []
    for k in range(n):
        # All indices except k
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        s_k = compute_S(rv[mask], err[mask])
        loo_scores.append(s_k)

    loo_scores = np.array(loo_scores)
    return np.min(loo_scores), np.median(loo_scores), loo_scores


def compute_outlier_leverage(rv_arr, err_arr):
    """
    Compute outlier leverage metric d_max.

    For each epoch i:
        d_i = |RV_i - median(RV)| / σ_i

    d_max = max_i d_i

    A large d_max indicates one epoch is far from the median in units of its error.

    Parameters
    ----------
    rv_arr : array-like
        Radial velocity measurements
    err_arr : array-like
        RV uncertainties

    Returns
    -------
    float
        Maximum normalized deviation d_max
    """
    rv = np.asarray(rv_arr)
    err = np.asarray(err_arr)

    if len(rv) < 2:
        return 0.0

    med_rv = np.median(rv)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.abs(rv - med_rv) / err
        d = np.where(np.isfinite(d), d, 0.0)

    return np.max(d)


def load_candidate_targetids(csv_path, top_n=5000):
    """
    Load target IDs from candidate CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to candidate CSV
    top_n : int
        Number of top candidates to load

    Returns
    -------
    numpy.ndarray
        Array of target IDs (int64)
    """
    # Simple CSV parsing without pandas
    targetids = []
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        tid_idx = header.index('targetid')

        for i, line in enumerate(f):
            if i >= top_n:
                break
            parts = line.strip().split(',')
            targetids.append(int(parts[tid_idx]))

    return np.array(targetids, dtype=np.int64)


def extract_epochs_for_targets(fits_path, targetids, chunk_size=500000):
    """
    Extract per-epoch RV data for specified targets from FITS file.

    IMPORTANT: This function applies per-measurement quality cuts to ensure
    consistency with analyze_rv_candidates.py. Epochs failing these cuts
    are excluded BEFORE computing diagnostics.

    Quality cuts applied per-measurement:
    - finite(RV) and finite(RV_ERR)
    - VRAD_ERR < 10 km/s
    - |VRAD| < 500 km/s

    Parameters
    ----------
    fits_path : str or Path
        Path to rvpix_exp FITS file
    targetids : array-like
        Array of target IDs to extract
    chunk_size : int
        Number of rows to process at a time

    Returns
    -------
    dict
        Dictionary mapping targetid -> {'mjd': [], 'rv': [], 'rv_err': [], 'gaia_source_id': int or None}
    """
    target_data = {tid: {'mjd': [], 'rv': [], 'rv_err': [], 'gaia_source_id': None}
                   for tid in targetids}

    if USE_FITSIO:
        # fitsio approach - read columns directly
        with fitsio.FITS(fits_path, 'r') as f:
            rvtab = f[1]  # RVTAB extension
            nrows = rvtab.get_nrows()
            fibermap = f[2]  # FIBERMAP for MJD

            # GAIA HDU (HDU 4) contains SOURCE_ID, row-aligned with RVTAB
            gaia_hdu = f[4] if len(f) > 4 else None
            has_gaia_hdu = gaia_hdu is not None and 'SOURCE_ID' in gaia_hdu.get_colnames()

            # Read in chunks
            for start in range(0, nrows, chunk_size):
                end = min(start + chunk_size, nrows)

                # Read columns for this chunk
                chunk_tid = rvtab['TARGETID'][start:end]
                chunk_rv = rvtab['VRAD'][start:end]
                chunk_err = rvtab['VRAD_ERR'][start:end]
                chunk_mjd = fibermap['MJD'][start:end]

                # Read SOURCE_ID from GAIA HDU (not RVTAB)
                if has_gaia_hdu:
                    chunk_gaia = gaia_hdu['SOURCE_ID'][start:end]
                else:
                    chunk_gaia = None

                # Find matching rows using numpy
                target_mask = np.isin(chunk_tid, targetids)

                if not np.any(target_mask):
                    continue

                # Apply per-measurement quality cuts (consistent with analyze_rv_candidates.py)
                quality_mask = (
                    np.isfinite(chunk_rv) &
                    np.isfinite(chunk_err) &
                    (chunk_err < MAX_VRAD_ERR) &
                    (np.abs(chunk_rv) < MAX_ABS_VRAD)
                )

                # Combined mask: target match AND quality pass
                combined_mask = target_mask & quality_mask

                if not np.any(combined_mask):
                    continue

                # Extract matching data
                matched_tid = chunk_tid[combined_mask]
                matched_rv = chunk_rv[combined_mask]
                matched_err = chunk_err[combined_mask]
                matched_mjd = chunk_mjd[combined_mask]
                matched_gaia = chunk_gaia[combined_mask] if chunk_gaia is not None else None

                # Populate target_data
                for i, tid in enumerate(matched_tid):
                    tid_int = int(tid)
                    if tid_int in target_data:
                        target_data[tid_int]['rv'].append(float(matched_rv[i]))
                        target_data[tid_int]['rv_err'].append(float(matched_err[i]))
                        target_data[tid_int]['mjd'].append(float(matched_mjd[i]))
                        # Set Gaia ID if available and not already set
                        # Treat GAIA_MISSING_SENTINEL (-9999) as missing
                        if matched_gaia is not None and target_data[tid_int]['gaia_source_id'] is None:
                            gaia_val = int(matched_gaia[i])
                            if gaia_val != GAIA_MISSING_SENTINEL:
                                target_data[tid_int]['gaia_source_id'] = gaia_val
    else:
        # astropy approach
        with fits.open(fits_path, memmap=True) as hdul:
            rvtab = hdul[1].data
            fibermap = hdul[2].data

            chunk_tid = rvtab['TARGETID']
            chunk_rv = rvtab['VRAD']
            chunk_err = rvtab['VRAD_ERR']
            chunk_mjd = fibermap['MJD']

            # Read SOURCE_ID from GAIA HDU (HDU 4)
            try:
                chunk_gaia = hdul[4].data['SOURCE_ID']
                has_gaia = True
            except (KeyError, IndexError):
                has_gaia = False
                chunk_gaia = None

            # Find matching rows
            target_mask = np.isin(chunk_tid, targetids)

            # Apply per-measurement quality cuts
            quality_mask = (
                np.isfinite(chunk_rv) &
                np.isfinite(chunk_err) &
                (chunk_err < MAX_VRAD_ERR) &
                (np.abs(chunk_rv) < MAX_ABS_VRAD)
            )

            combined_mask = target_mask & quality_mask

            matched_tid = chunk_tid[combined_mask]
            matched_rv = chunk_rv[combined_mask]
            matched_err = chunk_err[combined_mask]
            matched_mjd = chunk_mjd[combined_mask]
            matched_gaia = chunk_gaia[combined_mask] if has_gaia else None

            for i, tid in enumerate(matched_tid):
                tid_int = int(tid)
                if tid_int in target_data:
                    target_data[tid_int]['rv'].append(float(matched_rv[i]))
                    target_data[tid_int]['rv_err'].append(float(matched_err[i]))
                    target_data[tid_int]['mjd'].append(float(matched_mjd[i]))
                    if matched_gaia is not None and target_data[tid_int]['gaia_source_id'] is None:
                        gaia_val = int(matched_gaia[i])
                        if gaia_val != GAIA_MISSING_SENTINEL:
                            target_data[tid_int]['gaia_source_id'] = gaia_val

    # Convert lists to arrays
    for tid in target_data:
        target_data[tid]['rv'] = np.array(target_data[tid]['rv'])
        target_data[tid]['rv_err'] = np.array(target_data[tid]['rv_err'])
        target_data[tid]['mjd'] = np.array(target_data[tid]['mjd'])

    return target_data


def compute_diagnostics(target_data):
    """
    Compute all robust diagnostics for extracted targets.

    Parameters
    ----------
    target_data : dict
        Dictionary from extract_epochs_for_targets()

    Returns
    -------
    list of dict
        List of diagnostic records, one per target
    """
    results = []

    for tid, data in target_data.items():
        rv = data['rv']
        err = data['rv_err']
        mjd = data['mjd']
        gaia_id = data['gaia_source_id']

        n_epochs = len(rv)

        if n_epochs < 2:
            # Skip targets with insufficient epochs
            continue

        # Basic metrics
        mjd_span = np.max(mjd) - np.min(mjd)
        delta_rv = np.max(rv) - np.min(rv)
        rv_median = np.median(rv)
        rv_err_median = np.median(err)

        # Original S score
        S = compute_S(rv, err)

        # Leave-one-out stability
        S_min_LOO, S_med_LOO, _ = leave_one_out_scores(rv, err)

        # Outlier leverage
        d_max = compute_outlier_leverage(rv, err)

        # Conservative ranking score
        # For N >= 3: S_robust = min(S, S_min_LOO)
        # For N == 2: S_robust = 0 (cannot compute LOO, needs follow-up)
        if n_epochs >= 3:
            S_robust = min(S, S_min_LOO)
            tier = 'A'
        else:
            S_robust = 0.0
            tier = 'B'

        # Two-point flag
        two_point_only = (n_epochs == 2)

        results.append({
            'targetid': tid,
            'gaia_source_id': gaia_id,  # Keep as None if missing, don't coerce to 0
            'n_epochs': n_epochs,
            'mjd_span': mjd_span,
            'S': S,
            'S_min_LOO': S_min_LOO if not np.isnan(S_min_LOO) else 0.0,
            'S_med_LOO': S_med_LOO if not np.isnan(S_med_LOO) else 0.0,
            'S_robust': S_robust,
            'd_max': d_max,
            'delta_rv_kms': delta_rv,
            'rv_median_kms': rv_median,
            'rv_err_median_kms': rv_err_median,
            'tier': tier,
            'two_point_only': two_point_only,
            # Store raw data for plotting
            '_rv': rv,
            '_err': err,
            '_mjd': mjd
        })

    return results


def generate_notes(rec):
    """
    Generate a human-readable notes string explaining why this candidate passed.

    Parameters
    ----------
    rec : dict
        Diagnostic record for a target

    Returns
    -------
    str
        Notes string
    """
    notes_parts = []

    if rec['tier'] == 'A':
        notes_parts.append(f"N={rec['n_epochs']}")
        notes_parts.append(f"S_robust={rec['S_robust']:.1f}")
        notes_parts.append(f"d_max={rec['d_max']:.1f}")

        # Check LOO stability
        if rec['S_min_LOO'] > 0:
            stability_ratio = rec['S_min_LOO'] / rec['S'] if rec['S'] > 0 else 0
            if stability_ratio > 0.8:
                notes_parts.append("LOO-stable")
            elif stability_ratio > 0.5:
                notes_parts.append("LOO-moderate")
            else:
                notes_parts.append("LOO-drops")
    else:
        notes_parts.append("N=2")
        notes_parts.append(f"S={rec['S']:.1f}")
        notes_parts.append("needs_followup")

    return "; ".join(notes_parts)


def rank_and_filter(diagnostics, mjd_span_min=0.5):
    """
    Apply triage logic and rank candidates.

    Parameters
    ----------
    diagnostics : list of dict
        List of diagnostic records
    mjd_span_min : float
        Minimum MJD span required (days)

    Returns
    -------
    tuple (tier_a, tier_b)
        tier_a: List of Tier A candidates (N >= 3), ranked by S_robust
        tier_b: List of Tier B candidates (N == 2), ranked by S
    """
    # Filter by MJD span
    filtered = [d for d in diagnostics if d['mjd_span'] >= mjd_span_min]

    # Separate tiers
    tier_a = [d for d in filtered if d['tier'] == 'A']
    tier_b = [d for d in filtered if d['tier'] == 'B']

    # Rank Tier A by S_robust descending
    tier_a.sort(key=lambda x: x['S_robust'], reverse=True)

    # Rank Tier B by S descending
    tier_b.sort(key=lambda x: x['S'], reverse=True)

    return tier_a, tier_b


def write_triage_csv(filepath, records, top_n=200):
    """
    Write triage results to CSV.

    Parameters
    ----------
    filepath : str or Path
        Output CSV path
    records : list of dict
        Diagnostic records (already ranked)
    top_n : int
        Number of records to write
    """
    # Combine Tier A and Tier B, keeping Tier A first
    # The records should already be separated and ranked

    columns = [
        'rank', 'targetid', 'gaia_source_id', 'tier', 'n_epochs', 'mjd_span',
        'S', 'S_min_LOO', 'S_med_LOO', 'S_robust', 'd_max',
        'delta_rv_kms', 'rv_err_median_kms'
    ]

    with open(filepath, 'w') as f:
        f.write(','.join(columns) + '\n')

        for i, rec in enumerate(records[:top_n]):
            # Use empty string for missing Gaia IDs, not 0
            gaia_str = str(rec['gaia_source_id']) if rec['gaia_source_id'] is not None else ''
            row = [
                str(i + 1),
                str(rec['targetid']),
                gaia_str,
                rec['tier'],
                str(rec['n_epochs']),
                f"{rec['mjd_span']:.4f}",
                f"{rec['S']:.4f}",
                f"{rec['S_min_LOO']:.4f}",
                f"{rec['S_med_LOO']:.4f}",
                f"{rec['S_robust']:.4f}",
                f"{rec['d_max']:.4f}",
                f"{rec['delta_rv_kms']:.4f}",
                f"{rec['rv_err_median_kms']:.4f}"
            ]
            f.write(','.join(row) + '\n')

    print(f"  Wrote {min(len(records), top_n)} records to {filepath}")


def write_shortlist_csv(filepath, records, top_n=50):
    """
    Write shortlist with notes column.

    Parameters
    ----------
    filepath : str or Path
        Output CSV path
    records : list of dict
        Diagnostic records (Tier A only, already ranked)
    top_n : int
        Number of records to write
    """
    columns = [
        'rank', 'targetid', 'gaia_source_id', 'n_epochs', 'mjd_span',
        'S', 'S_robust', 'd_max', 'delta_rv_kms', 'notes'
    ]

    with open(filepath, 'w') as f:
        f.write(','.join(columns) + '\n')

        for i, rec in enumerate(records[:top_n]):
            notes = generate_notes(rec)
            # Escape notes for CSV (wrap in quotes if contains comma)
            if ',' in notes:
                notes = f'"{notes}"'

            # Use empty string for missing Gaia IDs
            gaia_str = str(rec['gaia_source_id']) if rec['gaia_source_id'] is not None else ''

            row = [
                str(i + 1),
                str(rec['targetid']),
                gaia_str,
                str(rec['n_epochs']),
                f"{rec['mjd_span']:.4f}",
                f"{rec['S']:.4f}",
                f"{rec['S_robust']:.4f}",
                f"{rec['d_max']:.4f}",
                f"{rec['delta_rv_kms']:.4f}",
                notes
            ]
            f.write(','.join(row) + '\n')

    print(f"  Wrote {min(len(records), top_n)} records to {filepath}")


def create_rv_plots(records, output_dir, top_n=20):
    """
    Create RV vs MJD plots for top candidates.

    Parameters
    ----------
    records : list of dict
        Diagnostic records with _rv, _err, _mjd fields
    output_dir : str or Path
        Output directory for plots
    top_n : int
        Number of plots to create
    """
    if not HAS_MATPLOTLIB:
        print("  Matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(records[:top_n]):
        tid = rec['targetid']
        rv = rec['_rv']
        err = rec['_err']
        mjd = rec['_mjd']

        # Sort by MJD for plotting
        sort_idx = np.argsort(mjd)
        mjd_sorted = mjd[sort_idx]
        rv_sorted = rv[sort_idx]
        err_sorted = err[sort_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(mjd_sorted, rv_sorted, yerr=err_sorted,
                    fmt='o', capsize=3, capthick=1, markersize=8,
                    color='#2E86AB', ecolor='#A23B72', markeredgecolor='black')

        ax.set_xlabel('MJD (days)', fontsize=12)
        ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)
        ax.set_title(f'TARGETID {tid}\n'
                     f'N={rec["n_epochs"]}, S={rec["S"]:.1f}, '
                     f'S_robust={rec["S_robust"]:.1f}, d_max={rec["d_max"]:.1f}',
                     fontsize=11)

        # Add horizontal line at median RV
        ax.axhline(rec['rv_median_kms'], color='gray', linestyle='--',
                   alpha=0.5, label=f'Median RV = {rec["rv_median_kms"]:.1f} km/s')
        ax.legend(loc='best')

        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save as PDF
        pdf_path = output_dir / f'rv_target_{tid}.pdf'
        plt.savefig(pdf_path, format='pdf', dpi=150)
        plt.close(fig)

    print(f"  Created {min(len(records), top_n)} plots in {output_dir}")


def run_triage(fits_path, candidates_csv, output_prefix, survey_name,
               top_candidates=5000, make_plots=True):
    """
    Run full triage pipeline for one survey.

    Parameters
    ----------
    fits_path : str or Path
        Path to rvpix_exp FITS file
    candidates_csv : str or Path
        Path to top candidates CSV from previous analysis
    output_prefix : str
        Prefix for output files (e.g., 'bright' or 'dark')
    survey_name : str
        Human-readable survey name for logging
    top_candidates : int
        Number of top candidates to triage
    make_plots : bool
        Whether to create RV plots

    Returns
    -------
    tuple (tier_a, tier_b)
        Ranked candidates by tier
    """
    print(f"\n{'='*60}")
    print(f"TRIAGE: {survey_name}")
    print(f"{'='*60}")

    # 1. Load candidate target IDs
    print(f"\n1. Loading top {top_candidates} candidates from {candidates_csv}")
    targetids = load_candidate_targetids(candidates_csv, top_n=top_candidates)
    print(f"   Loaded {len(targetids)} target IDs")

    # 2. Extract per-epoch data from FITS
    print(f"\n2. Extracting per-epoch RV data from {fits_path}")
    target_data = extract_epochs_for_targets(fits_path, targetids)
    valid_targets = sum(1 for d in target_data.values() if len(d['rv']) >= 2)
    print(f"   Extracted data for {valid_targets} targets with >= 2 epochs")

    # 3. Compute diagnostics
    print(f"\n3. Computing robust diagnostics...")
    diagnostics = compute_diagnostics(target_data)
    print(f"   Computed diagnostics for {len(diagnostics)} targets")

    # 4. Rank and filter
    print(f"\n4. Applying triage logic (MJD_span > 0.5 days)...")
    tier_a, tier_b = rank_and_filter(diagnostics, mjd_span_min=0.5)
    print(f"   Tier A (N >= 3): {len(tier_a)} candidates")
    print(f"   Tier B (N == 2): {len(tier_b)} candidates")

    # 5. Write outputs
    print(f"\n5. Writing output files...")

    derived_dir = Path('data/derived')
    derived_dir.mkdir(parents=True, exist_ok=True)

    # Combine tiers for full triage output (Tier A first)
    combined = tier_a + tier_b
    triage_path = derived_dir / f'triage_candidates_{output_prefix}.csv'
    write_triage_csv(triage_path, combined, top_n=200)

    # Shortlist (Tier A only)
    shortlist_path = derived_dir / f'shortlist_top50_{output_prefix}.csv'
    write_shortlist_csv(shortlist_path, tier_a, top_n=50)

    # 6. Create plots (Tier A only)
    if make_plots and len(tier_a) > 0:
        print(f"\n6. Creating RV plots for top 20...")
        plots_dir = derived_dir / 'plots' / output_prefix
        create_rv_plots(tier_a, plots_dir, top_n=20)

    # 7. Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {survey_name}")
    print(f"{'='*60}")
    print(f"Total candidates triaged: {len(diagnostics)}")
    print(f"Tier A (N >= 3, S_robust ranking): {len(tier_a)}")
    print(f"Tier B (N == 2, needs follow-up):  {len(tier_b)}")

    if len(tier_a) > 0:
        print(f"\nTop 10 Tier A candidates:")
        print("-" * 100)
        print(f"{'Rank':<6} {'TARGETID':<20} {'N':<4} {'MJD_span':<10} {'S':<10} {'S_robust':<10} {'S_min_LOO':<10} {'d_max':<8}")
        print("-" * 100)
        for i, rec in enumerate(tier_a[:10]):
            print(f"{i+1:<6} {rec['targetid']:<20} {rec['n_epochs']:<4} "
                  f"{rec['mjd_span']:<10.2f} {rec['S']:<10.2f} {rec['S_robust']:<10.2f} "
                  f"{rec['S_min_LOO']:<10.2f} {rec['d_max']:<8.2f}")

    return tier_a, tier_b


def main():
    parser = argparse.ArgumentParser(
        description='Robust triage of RV variability candidates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--survey', choices=['bright', 'dark', 'both'], default='both',
                        help='Which survey to process (default: both)')
    parser.add_argument('--top-candidates', type=int, default=5000,
                        help='Number of top candidates to triage (default: 5000)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')

    args = parser.parse_args()

    # Define paths
    base_dir = Path('.')
    fits_bright = base_dir / 'data/raw/rvpix_exp-main-bright.fits'
    fits_dark = base_dir / 'data/raw/rvpix_exp-main-dark.fits'
    csv_bright = base_dir / 'data/derived/top_candidates_main_bright_full.csv'
    csv_dark = base_dir / 'data/derived/top_candidates_main_dark_full.csv'

    make_plots = not args.no_plots

    if args.survey in ['bright', 'both']:
        if not fits_bright.exists():
            print(f"ERROR: {fits_bright} not found")
            sys.exit(1)
        if not csv_bright.exists():
            print(f"ERROR: {csv_bright} not found")
            sys.exit(1)
        run_triage(fits_bright, csv_bright, 'bright', 'main-bright',
                   args.top_candidates, make_plots)

    if args.survey in ['dark', 'both']:
        if not fits_dark.exists():
            print(f"ERROR: {fits_dark} not found")
            sys.exit(1)
        if not csv_dark.exists():
            print(f"ERROR: {csv_dark} not found")
            sys.exit(1)
        run_triage(fits_dark, csv_dark, 'dark', 'main-dark',
                   args.top_candidates, make_plots)

    print("\n" + "="*60)
    print("TRIAGE COMPLETE")
    print("="*60)
    print("\nOutput files:")
    print("  data/derived/triage_candidates_bright.csv")
    print("  data/derived/triage_candidates_dark.csv")
    print("  data/derived/shortlist_top50_bright.csv")
    print("  data/derived/shortlist_top50_dark.csv")
    print("  data/derived/plots/bright/*.pdf")
    print("  data/derived/plots/dark/*.pdf")


if __name__ == '__main__':
    main()
