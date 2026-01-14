#!/usr/bin/env python3
"""
Deep-dive analysis for top RV-variable candidate.
Target: TARGETID=39628001431785529, Gaia SOURCE_ID=2759088365339967488
"""

import numpy as np
from astropy.io import fits
import glob
import os

# Target identifiers
TARGET_ID = 39628001431785529
GAIA_SOURCE_ID = 2759088365339967488

# Quality cuts
MAX_VRAD_ERR = 10.0  # km/s
MAX_ABS_VRAD = 500.0  # km/s

# Find all FITS files
FITS_FILES = glob.glob("data/raw/*.fits")
print(f"Found {len(FITS_FILES)} FITS files:")
for f in FITS_FILES:
    print(f"  - {os.path.basename(f)}")
print()

# Collect all epochs
all_epochs = []

for fits_file in FITS_FILES:
    print(f"=== Searching {os.path.basename(fits_file)} ===")

    with fits.open(fits_file, memmap=True) as hdu:
        # Get tables
        rvtab = hdu['RVTAB'].data
        fibermap = hdu['FIBERMAP'].data
        scores = hdu['SCORES'].data if 'SCORES' in [h.name for h in hdu] else None

        # Try to get GAIA table
        gaia_data = None
        if 'GAIA' in [h.name for h in hdu]:
            gaia_data = hdu['GAIA'].data

        # Search by TARGETID
        targetid_mask = rvtab['TARGETID'] == TARGET_ID
        n_targetid = np.sum(targetid_mask)
        print(f"Matches by TARGETID: {n_targetid}")

        # Search by Gaia SOURCE_ID if GAIA table exists
        n_gaia = 0
        gaia_mask = np.zeros(len(rvtab), dtype=bool)
        if gaia_data is not None and 'SOURCE_ID' in gaia_data.dtype.names:
            gaia_mask = gaia_data['SOURCE_ID'] == GAIA_SOURCE_ID
            n_gaia = np.sum(gaia_mask)
            print(f"Matches by Gaia SOURCE_ID: {n_gaia}")

        # Combined mask
        combined_mask = targetid_mask | gaia_mask
        n_combined = np.sum(combined_mask)
        print(f"Combined matches: {n_combined}")

        if n_combined == 0:
            print("No matches found in this file.\n")
            continue

        # Get indices
        indices = np.where(combined_mask)[0]

        # Extract epochs
        for idx in indices:
            row_rv = rvtab[idx]
            row_fm = fibermap[idx]
            row_sc = scores[idx] if scores is not None else None

            epoch = {
                'file': os.path.basename(fits_file),
                'targetid': int(row_rv['TARGETID']),
                'rv': float(row_rv['VRAD']),
                'rv_err': float(row_rv['VRAD_ERR']),
                'mjd': float(row_fm['MJD']),
                'expid': int(row_rv['EXPID']),
                'success': bool(row_rv['SUCCESS']),
                'sn_b': float(row_rv['SN_B']) if 'SN_B' in rvtab.dtype.names else np.nan,
                'sn_r': float(row_rv['SN_R']) if 'SN_R' in rvtab.dtype.names else np.nan,
                'sn_z': float(row_rv['SN_Z']) if 'SN_Z' in rvtab.dtype.names else np.nan,
                'rvs_warn': int(row_rv['RVS_WARN']) if 'RVS_WARN' in rvtab.dtype.names else 0,
                'chisq_tot': float(row_rv['CHISQ_TOT']) if 'CHISQ_TOT' in rvtab.dtype.names else np.nan,
            }

            # Get TSNR2 from SCORES if available
            if row_sc is not None:
                if 'TSNR2_STAR' in scores.dtype.names:
                    epoch['tsnr2_star'] = float(row_sc['TSNR2_STAR'])
                else:
                    epoch['tsnr2_star'] = np.nan
            else:
                epoch['tsnr2_star'] = np.nan

            all_epochs.append(epoch)

        print()

print("=" * 70)
print(f"TASK 1 RESULTS: Total raw matches across all files: {len(all_epochs)}")
print("=" * 70)
print()

# Sort by MJD and show all raw epochs
all_epochs = sorted(all_epochs, key=lambda x: x['mjd'])

print("Raw epochs found (sorted by MJD):")
print(f"{'#':<3} {'File':<28} {'EXPID':<10} {'MJD':<14} {'RV':<10} {'RV_ERR':<8} {'SUCCESS':<8} {'SN_R':<8}")
print("-" * 100)
for i, e in enumerate(all_epochs, 1):
    success_str = "YES" if e['success'] else "NO"
    print(f"{i:<3} {e['file']:<28} {e['expid']:<10} {e['mjd']:<14.6f} {e['rv']:<10.2f} {e['rv_err']:<8.4f} {success_str:<8} {e['sn_r']:<8.2f}")
print()

# De-duplication
print("=== De-duplication ===")

unique_epochs = []
used_expids = set()

for epoch in all_epochs:
    expid = epoch['expid']
    mjd = epoch['mjd']

    # Check if this EXPID was already used
    if expid in used_expids and expid != -1:
        print(f"  Skipping duplicate EXPID {expid}")
        continue

    # Check if MJD is too close to an existing epoch
    is_duplicate = False
    for existing in unique_epochs:
        if abs(mjd - existing['mjd']) < 1e-3:
            # Keep the one with smaller RV_ERR
            if epoch['rv_err'] < existing['rv_err']:
                unique_epochs.remove(existing)
                unique_epochs.append(epoch)
                print(f"  Replaced epoch at MJD {mjd:.6f} (better RV_ERR)")
            else:
                print(f"  Skipping epoch at MJD {mjd:.6f} (worse RV_ERR)")
            is_duplicate = True
            break

    if not is_duplicate:
        unique_epochs.append(epoch)
        if expid != -1:
            used_expids.add(expid)

print()
print(f"Total unique epochs after de-duplication: {len(unique_epochs)}")
print()

# Apply quality cuts
print("=== Applying quality cuts ===")
quality_epochs = []
excluded_reasons = []

for e in unique_epochs:
    rv = e['rv']
    rv_err = e['rv_err']

    if not np.isfinite(rv):
        print(f"  Excluded EXPID {e['expid']} (MJD {e['mjd']:.6f}): non-finite RV")
        excluded_reasons.append(('non-finite RV', e))
        continue
    if not np.isfinite(rv_err):
        print(f"  Excluded EXPID {e['expid']} (MJD {e['mjd']:.6f}): non-finite RV_ERR")
        excluded_reasons.append(('non-finite RV_ERR', e))
        continue
    if rv_err >= MAX_VRAD_ERR:
        print(f"  Excluded EXPID {e['expid']} (MJD {e['mjd']:.6f}): RV_ERR={rv_err:.2f} >= {MAX_VRAD_ERR}")
        excluded_reasons.append(('RV_ERR too large', e))
        continue
    if abs(rv) >= MAX_ABS_VRAD:
        print(f"  Excluded EXPID {e['expid']} (MJD {e['mjd']:.6f}): |RV|={abs(rv):.2f} >= {MAX_ABS_VRAD}")
        excluded_reasons.append(('|RV| too large', e))
        continue

    quality_epochs.append(e)

print()
print(f"Epochs passing quality cuts: {len(quality_epochs)}")
print()

# TASK 2: Recompute RV diagnostics
print("=" * 70)
print("TASK 2: RECOMPUTED RV DIAGNOSTICS")
print("=" * 70)
print()

# Sort by MJD
quality_epochs = sorted(quality_epochs, key=lambda x: x['mjd'])

print("Final epoch table (quality-filtered):")
print(f"{'#':<4} {'MJD':<14} {'RV (km/s)':<12} {'RV_ERR':<8} {'SUCCESS':<8} {'SN_R':<8} {'EXPID':<10}")
print("-" * 75)
for i, e in enumerate(quality_epochs, 1):
    success_str = "YES" if e['success'] else "NO"
    print(f"{i:<4} {e['mjd']:<14.6f} {e['rv']:<12.2f} {e['rv_err']:<8.4f} {success_str:<8} {e['sn_r']:<8.2f} {e['expid']:<10}")
print()

# Compute metrics
if len(quality_epochs) >= 2:
    rvs = np.array([e['rv'] for e in quality_epochs])
    errs = np.array([e['rv_err'] for e in quality_epochs])
    mjds = np.array([e['mjd'] for e in quality_epochs])

    rv_min = np.min(rvs)
    rv_max = np.max(rvs)
    delta_rv = rv_max - rv_min
    mjd_span = np.max(mjds) - np.min(mjds)

    # S = delta_rv / sqrt(sum(sigma^2))
    sigma_combined = np.sqrt(np.sum(errs**2))
    S = delta_rv / sigma_combined

    # d_max = max |RV_i - median(RV)| / sigma_i
    rv_median = np.median(rvs)
    d_values = np.abs(rvs - rv_median) / errs
    d_max = np.max(d_values)
    d_max_idx = np.argmax(d_values)

    # Leave-one-out S values
    S_loo = []
    for i in range(len(rvs)):
        rvs_loo = np.delete(rvs, i)
        errs_loo = np.delete(errs, i)
        if len(rvs_loo) >= 2:
            delta_loo = np.max(rvs_loo) - np.min(rvs_loo)
            sigma_loo = np.sqrt(np.sum(errs_loo**2))
            S_loo.append(delta_loo / sigma_loo)
        else:
            S_loo.append(0)

    S_min_loo = np.min(S_loo)
    S_robust = min(S, S_min_loo)

    print("Computed metrics:")
    print(f"  N_epochs:     {len(quality_epochs)}")
    print(f"  MJD_span:     {mjd_span:.4f} days ({mjd_span:.1f} days)")
    print(f"  RV_min:       {rv_min:.2f} km/s")
    print(f"  RV_max:       {rv_max:.2f} km/s")
    print(f"  ΔRV_max:      {delta_rv:.2f} km/s")
    print(f"  S:            {S:.2f}")
    print(f"  S_min_LOO:    {S_min_loo:.2f}")
    print(f"  S_robust:     {S_robust:.2f}")
    print(f"  d_max:        {d_max:.2f}")
    print(f"  RV_median:    {rv_median:.2f} km/s")
    print()

    # Which epoch drives d_max?
    print("d_max analysis:")
    print(f"  d_max is driven by epoch #{d_max_idx + 1}")
    print(f"    MJD:    {quality_epochs[d_max_idx]['mjd']:.6f}")
    print(f"    RV:     {quality_epochs[d_max_idx]['rv']:.2f} km/s")
    print(f"    RV_ERR: {quality_epochs[d_max_idx]['rv_err']:.4f} km/s")
    print(f"    d_i:    {d_values[d_max_idx]:.2f}")
    print()

    # Show all d values
    print("Per-epoch d values (|RV - median| / sigma):")
    for i, (e, d) in enumerate(zip(quality_epochs, d_values), 1):
        marker = " <-- MAX" if i-1 == d_max_idx else ""
        print(f"  Epoch {i}: d = {d:.2f}{marker}")
    print()

    # Is RV swing dominated by one epoch?
    print("RV distribution analysis:")
    rv_range_without_max = np.max(np.delete(rvs, np.argmax(rvs))) - np.min(rvs)
    rv_range_without_min = np.max(rvs) - np.min(np.delete(rvs, np.argmin(rvs)))
    print(f"  Full ΔRV:                      {delta_rv:.2f} km/s")
    print(f"  ΔRV without max-RV epoch:      {rv_range_without_max:.2f} km/s ({100*rv_range_without_max/delta_rv:.1f}%)")
    print(f"  ΔRV without min-RV epoch:      {rv_range_without_min:.2f} km/s ({100*rv_range_without_min/delta_rv:.1f}%)")

    if rv_range_without_max < 0.5 * delta_rv and rv_range_without_min < 0.5 * delta_rv:
        print("  --> RV swing is DOMINATED by extreme epochs at BOTH ends")
    elif rv_range_without_max < 0.5 * delta_rv:
        print("  --> RV swing DEPENDS on the max-RV epoch")
    elif rv_range_without_min < 0.5 * delta_rv:
        print("  --> RV swing DEPENDS on the min-RV epoch")
    else:
        print("  --> RV swing is SPREAD across multiple epochs")
    print()

else:
    print(f"ERROR: Only {len(quality_epochs)} epochs - cannot compute diagnostics")
    print()

# TASK 3: Per-epoch quality context
print("=" * 70)
print("TASK 3: PER-EPOCH QUALITY CONTEXT")
print("=" * 70)
print()

print("Detailed quality metrics per epoch:")
print(f"{'#':<4} {'MJD':<14} {'RV':<10} {'RV_ERR':<8} {'SUCCESS':<8} {'SN_B':<8} {'SN_R':<8} {'SN_Z':<8} {'RVS_WARN':<10} {'CHISQ':<10}")
print("-" * 100)
for i, e in enumerate(quality_epochs, 1):
    success_str = "YES" if e['success'] else "NO"
    print(f"{i:<4} {e['mjd']:<14.6f} {e['rv']:<10.2f} {e['rv_err']:<8.4f} {success_str:<8} {e['sn_b']:<8.2f} {e['sn_r']:<8.2f} {e['sn_z']:<8.2f} {e['rvs_warn']:<10} {e['chisq_tot']:<10.2f}")
print()

# Check if extreme RV epochs have poor quality
if len(quality_epochs) >= 2:
    sn_r_values = [e['sn_r'] for e in quality_epochs if np.isfinite(e['sn_r'])]
    if sn_r_values:
        median_sn_r = np.median(sn_r_values)
        print(f"Median SN_R: {median_sn_r:.2f}")

        # Check extreme epochs
        max_rv_epoch = quality_epochs[np.argmax(rvs)]
        min_rv_epoch = quality_epochs[np.argmin(rvs)]

        print(f"\nMax RV epoch (RV={max_rv_epoch['rv']:.2f} km/s) at MJD {max_rv_epoch['mjd']:.6f}:")
        print(f"  SN_R = {max_rv_epoch['sn_r']:.2f} (median = {median_sn_r:.2f})")
        print(f"  SUCCESS = {max_rv_epoch['success']}")
        print(f"  RVS_WARN = {max_rv_epoch['rvs_warn']}")
        if max_rv_epoch['sn_r'] < 0.5 * median_sn_r:
            print("  --> POOR quality (SN_R < 50% of median)")
        elif max_rv_epoch['sn_r'] < 0.8 * median_sn_r:
            print("  --> MARGINAL quality (SN_R 50-80% of median)")
        else:
            print("  --> NORMAL quality")

        print(f"\nMin RV epoch (RV={min_rv_epoch['rv']:.2f} km/s) at MJD {min_rv_epoch['mjd']:.6f}:")
        print(f"  SN_R = {min_rv_epoch['sn_r']:.2f} (median = {median_sn_r:.2f})")
        print(f"  SUCCESS = {min_rv_epoch['success']}")
        print(f"  RVS_WARN = {min_rv_epoch['rvs_warn']}")
        if min_rv_epoch['sn_r'] < 0.5 * median_sn_r:
            print("  --> POOR quality (SN_R < 50% of median)")
        elif min_rv_epoch['sn_r'] < 0.8 * median_sn_r:
            print("  --> MARGINAL quality (SN_R 50-80% of median)")
        else:
            print("  --> NORMAL quality")

        # Check SUCCESS flag distribution
        print(f"\nSUCCESS flag distribution:")
        n_success = sum(1 for e in quality_epochs if e['success'])
        n_fail = len(quality_epochs) - n_success
        print(f"  SUCCESS=True:  {n_success}/{len(quality_epochs)}")
        print(f"  SUCCESS=False: {n_fail}/{len(quality_epochs)}")

        if n_fail > 0:
            print("\n  WARNING: Some quality-passing epochs have SUCCESS=False")
            print("  This indicates the RV pipeline flagged potential issues")
            for e in quality_epochs:
                if not e['success']:
                    print(f"    - EXPID {e['expid']}: RV={e['rv']:.2f}, RVS_WARN={e['rvs_warn']}")
    print()

# Store results for later
results = {
    'n_raw': len(all_epochs),
    'n_unique': len(unique_epochs),
    'n_quality': len(quality_epochs),
    'epochs': quality_epochs,
}

if len(quality_epochs) >= 2:
    results.update({
        'delta_rv': delta_rv,
        'S': S,
        'S_robust': S_robust,
        'd_max': d_max,
        'mjd_span': mjd_span,
        'd_max_epoch': d_max_idx,
    })

# Save for next script
import json
with open('data/derived/deep_dive_epochs.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)

print("Saved epoch data to data/derived/deep_dive_epochs.json")
