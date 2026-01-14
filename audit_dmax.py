#!/usr/bin/env python3
"""
audit_dmax.py - Sanity audit of d_max calculation

This script re-extracts per-epoch RV data for top Tier A candidates
and validates the d_max computation.

DEFINITIONS:
- RV_i: i-th radial velocity measurement (km/s)
- σ_i: i-th RV uncertainty (km/s)
- d_i: normalized deviation = |RV_i − median(RV)| / σ_i (dimensionless)
- d_max: max_i d_i (dimensionless)
"""

import sys
from pathlib import Path
import numpy as np

try:
    import fitsio
    USE_FITSIO = True
except ImportError:
    from astropy.io import fits
    USE_FITSIO = False


def load_tier_a_targetids(csv_path, top_n=20):
    """Load top N Tier A target IDs from triage CSV."""
    targetids = []
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        tid_idx = header.index('targetid')
        tier_idx = header.index('tier')

        for line in f:
            parts = line.strip().split(',')
            if parts[tier_idx] == 'A':
                targetids.append(int(parts[tid_idx]))
                if len(targetids) >= top_n:
                    break
    return np.array(targetids, dtype=np.int64)


def extract_epochs(fits_path, targetids):
    """Extract per-epoch RV data for specified targets."""
    target_data = {tid: {'mjd': [], 'rv': [], 'rv_err': []} for tid in targetids}

    if USE_FITSIO:
        with fitsio.FITS(fits_path, 'r') as f:
            rvtab = f[1]
            fibermap = f[2]
            nrows = rvtab.get_nrows()

            chunk_size = 500000
            for start in range(0, nrows, chunk_size):
                end = min(start + chunk_size, nrows)
                chunk_tid = rvtab['TARGETID'][start:end]
                chunk_rv = rvtab['VRAD'][start:end]
                chunk_err = rvtab['VRAD_ERR'][start:end]
                chunk_mjd = fibermap['MJD'][start:end]

                mask = np.isin(chunk_tid, targetids)
                if not np.any(mask):
                    continue

                for i in np.where(mask)[0]:
                    tid = int(chunk_tid[i])
                    if tid in target_data:
                        target_data[tid]['rv'].append(float(chunk_rv[i]))
                        target_data[tid]['rv_err'].append(float(chunk_err[i]))
                        target_data[tid]['mjd'].append(float(chunk_mjd[i]))
    else:
        with fits.open(fits_path, memmap=True) as hdul:
            rvtab = hdul[1].data
            fibermap = hdul[2].data

            mask = np.isin(rvtab['TARGETID'], targetids)
            for i in np.where(mask)[0]:
                tid = int(rvtab['TARGETID'][i])
                if tid in target_data:
                    target_data[tid]['rv'].append(float(rvtab['VRAD'][i]))
                    target_data[tid]['rv_err'].append(float(rvtab['VRAD_ERR'][i]))
                    target_data[tid]['mjd'].append(float(fibermap['MJD'][i]))

    # Convert to arrays
    for tid in target_data:
        target_data[tid]['rv'] = np.array(target_data[tid]['rv'])
        target_data[tid]['rv_err'] = np.array(target_data[tid]['rv_err'])
        target_data[tid]['mjd'] = np.array(target_data[tid]['mjd'])

    return target_data


def audit_dmax(target_data, sigma_floor=0.0):
    """
    Audit d_max calculation for each target.

    Parameters
    ----------
    sigma_floor : float
        Minimum σ_i value (km/s). If > 0, σ_i_eff = max(σ_i, sigma_floor)
    """
    results = []

    for tid, data in target_data.items():
        rv = data['rv']
        err = data['rv_err']
        mjd = data['mjd']

        if len(rv) < 2:
            continue

        # Sort by MJD
        sort_idx = np.argsort(mjd)
        rv = rv[sort_idx]
        err = err[sort_idx]
        mjd = mjd[sort_idx]

        # Apply sigma floor if specified
        if sigma_floor > 0:
            err_eff = np.maximum(err, sigma_floor)
        else:
            err_eff = err

        med_rv = np.median(rv)

        # Compute d_i for each epoch
        d_i = np.abs(rv - med_rv) / err_eff
        d_max = np.max(d_i)

        # Find which epoch has d_max
        max_idx = np.argmax(d_i)

        # Check for tiny errors
        min_err = np.min(err)
        has_tiny_err = min_err < 0.05  # < 50 m/s is suspiciously small

        results.append({
            'targetid': tid,
            'n_epochs': len(rv),
            'rv': rv,
            'err': err,
            'err_eff': err_eff,
            'mjd': mjd,
            'med_rv': med_rv,
            'd_i': d_i,
            'd_max': d_max,
            'max_epoch_idx': max_idx,
            'min_err': min_err,
            'has_tiny_err': has_tiny_err
        })

    return results


def print_audit_report(results, survey_name, sigma_floor=0.0):
    """Print detailed audit report."""
    print(f"\n{'='*80}")
    print(f"D_MAX AUDIT: {survey_name}")
    if sigma_floor > 0:
        print(f"(with sigma_floor = {sigma_floor} km/s)")
    print(f"{'='*80}")

    tiny_err_count = sum(1 for r in results if r['has_tiny_err'])
    print(f"\nTargets with σ_i < 0.05 km/s: {tiny_err_count}/{len(results)}")

    for i, r in enumerate(results[:20]):
        print(f"\n{'-'*80}")
        print(f"Rank {i+1}: TARGETID {r['targetid']}")
        print(f"N_epochs = {r['n_epochs']}, median(RV) = {r['med_rv']:.4f} km/s")
        print(f"d_max = {r['d_max']:.2f} (epoch {r['max_epoch_idx']+1})")
        print(f"min(σ_i) = {r['min_err']:.4f} km/s {'⚠️ TINY!' if r['has_tiny_err'] else ''}")

        print(f"\n  {'Epoch':<6} {'MJD':<12} {'RV_i (km/s)':<15} {'σ_i (km/s)':<12} {'d_i':<10}")
        print(f"  {'-'*55}")
        for j in range(r['n_epochs']):
            marker = " ← MAX" if j == r['max_epoch_idx'] else ""
            print(f"  {j+1:<6} {r['mjd'][j]:<12.2f} {r['rv'][j]:<15.4f} {r['err'][j]:<12.4f} {r['d_i'][j]:<10.2f}{marker}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY OF ISSUES")
    print(f"{'='*80}")

    # Count by cause
    extreme_rv_count = 0
    tiny_err_count = 0
    both_count = 0

    for r in results:
        max_idx = r['max_epoch_idx']
        rv_deviation = abs(r['rv'][max_idx] - r['med_rv'])
        is_extreme_rv = rv_deviation > 50  # > 50 km/s from median
        is_tiny_err = r['err'][max_idx] < 0.1  # < 100 m/s

        if is_extreme_rv and is_tiny_err:
            both_count += 1
        elif is_extreme_rv:
            extreme_rv_count += 1
        elif is_tiny_err:
            tiny_err_count += 1

    print(f"High d_max caused by:")
    print(f"  - Extreme RV only:      {extreme_rv_count}")
    print(f"  - Tiny σ_i only:        {tiny_err_count}")
    print(f"  - Both extreme RV + tiny σ: {both_count}")

    # Find minimum σ_i across all targets
    all_errs = np.concatenate([r['err'] for r in results])
    print(f"\nσ_i distribution across all epochs:")
    print(f"  min:  {np.min(all_errs):.4f} km/s")
    print(f"  5%:   {np.percentile(all_errs, 5):.4f} km/s")
    print(f"  50%:  {np.percentile(all_errs, 50):.4f} km/s")
    print(f"  95%:  {np.percentile(all_errs, 95):.4f} km/s")
    print(f"  max:  {np.max(all_errs):.4f} km/s")


def main():
    base_dir = Path('.')

    # Process bright survey
    print("\n" + "="*80)
    print("LOADING MAIN-BRIGHT TOP 20 TIER A")
    print("="*80)

    bright_triage = base_dir / 'data/derived/triage_candidates_bright.csv'
    bright_fits = base_dir / 'data/raw/rvpix_exp-main-bright.fits'

    bright_tids = load_tier_a_targetids(bright_triage, top_n=20)
    print(f"Loaded {len(bright_tids)} Tier A target IDs")

    bright_data = extract_epochs(bright_fits, bright_tids)
    print(f"Extracted epoch data for {len(bright_data)} targets")

    bright_results = audit_dmax(bright_data, sigma_floor=0.0)
    print_audit_report(bright_results, "main-bright", sigma_floor=0.0)

    # Process dark survey
    print("\n" + "="*80)
    print("LOADING MAIN-DARK TOP 20 TIER A")
    print("="*80)

    dark_triage = base_dir / 'data/derived/triage_candidates_dark.csv'
    dark_fits = base_dir / 'data/raw/rvpix_exp-main-dark.fits'

    dark_tids = load_tier_a_targetids(dark_triage, top_n=20)
    print(f"Loaded {len(dark_tids)} Tier A target IDs")

    dark_data = extract_epochs(dark_fits, dark_tids)
    print(f"Extracted epoch data for {len(dark_data)} targets")

    dark_results = audit_dmax(dark_data, sigma_floor=0.0)
    print_audit_report(dark_results, "main-dark", sigma_floor=0.0)

    # Test with sigma floor
    print("\n" + "="*80)
    print("TESTING WITH SIGMA_FLOOR = 0.1 km/s")
    print("="*80)

    print("\n--- BRIGHT with sigma_floor ---")
    bright_results_floor = audit_dmax(bright_data, sigma_floor=0.1)
    d_max_before = [r['d_max'] for r in audit_dmax(bright_data, sigma_floor=0.0)]
    d_max_after = [r['d_max'] for r in bright_results_floor]

    print(f"{'TARGETID':<22} {'d_max (raw)':<15} {'d_max (floor)':<15} {'Change':<10}")
    print("-"*60)
    for i, r in enumerate(bright_results_floor[:10]):
        change = d_max_after[i] - d_max_before[i]
        print(f"{r['targetid']:<22} {d_max_before[i]:<15.2f} {d_max_after[i]:<15.2f} {change:+.2f}")

    print("\n--- DARK with sigma_floor ---")
    dark_results_floor = audit_dmax(dark_data, sigma_floor=0.1)
    d_max_before = [r['d_max'] for r in audit_dmax(dark_data, sigma_floor=0.0)]
    d_max_after = [r['d_max'] for r in dark_results_floor]

    print(f"{'TARGETID':<22} {'d_max (raw)':<15} {'d_max (floor)':<15} {'Change':<10}")
    print("-"*60)
    for i, r in enumerate(dark_results_floor[:10]):
        change = d_max_after[i] - d_max_before[i]
        print(f"{r['targetid']:<22} {d_max_before[i]:<15.2f} {d_max_after[i]:<15.2f} {change:+.2f}")


if __name__ == '__main__':
    main()
