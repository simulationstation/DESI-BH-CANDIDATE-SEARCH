#!/usr/bin/env python3
"""
Deep-dive analysis for ALL targets in priorityA_followup_only.csv

Outputs:
- data/derived/deep_dive_all_followup.csv
- data/derived/deep_dive_clean_subset.csv
- data/derived/deep_dive_not_relevant.csv
"""

import csv
import glob
import os
import sys
import time
import numpy as np
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

# Quality cuts
MAX_VRAD_ERR = 10.0  # km/s
MAX_ABS_VRAD = 500.0  # km/s

# Input/output files
INPUT_CSV = "data/derived/priorityA_followup_only.csv"
OUTPUT_ALL = "data/derived/deep_dive_all_followup.csv"
OUTPUT_CLEAN = "data/derived/deep_dive_clean_subset.csv"
OUTPUT_NOT_RELEVANT = "data/derived/deep_dive_not_relevant.csv"

# Find all FITS files
FITS_FILES = glob.glob("data/raw/*.fits")
print(f"Found {len(FITS_FILES)} FITS files")

# Pre-load FITS file handles and data indices for efficiency
print("Pre-loading FITS file indices...")
fits_data = {}
for fits_file in FITS_FILES:
    fname = os.path.basename(fits_file)
    print(f"  Loading {fname}...")
    with fits.open(fits_file, memmap=True) as hdu:
        rvtab = hdu['RVTAB'].data
        fibermap = hdu['FIBERMAP'].data
        scores = hdu['SCORES'].data if 'SCORES' in [h.name for h in hdu] else None
        gaia_data = hdu['GAIA'].data if 'GAIA' in [h.name for h in hdu] else None

        # Store column names and data
        fits_data[fname] = {
            'path': fits_file,
            'targetids': np.array(rvtab['TARGETID']),
            'gaia_ids': np.array(gaia_data['SOURCE_ID']) if gaia_data is not None else None,
        }
print()


def extract_epochs_for_target(targetid, gaia_source_id):
    """Extract all epochs for a target across all FITS files."""
    all_epochs = []

    for fname, fdata in fits_data.items():
        # Find matching rows
        tid_mask = fdata['targetids'] == targetid
        gaia_mask = np.zeros_like(tid_mask)
        if fdata['gaia_ids'] is not None and gaia_source_id:
            gaia_mask = fdata['gaia_ids'] == gaia_source_id

        combined_mask = tid_mask | gaia_mask
        if not np.any(combined_mask):
            continue

        indices = np.where(combined_mask)[0]

        # Open file and extract data
        with fits.open(fdata['path'], memmap=True) as hdu:
            rvtab = hdu['RVTAB'].data
            fibermap = hdu['FIBERMAP'].data
            scores = hdu['SCORES'].data if 'SCORES' in [h.name for h in hdu] else None

            for idx in indices:
                row_rv = rvtab[idx]
                row_fm = fibermap[idx]

                epoch = {
                    'file': fname,
                    'targetid': int(row_rv['TARGETID']),
                    'rv': float(row_rv['VRAD']),
                    'rv_err': float(row_rv['VRAD_ERR']),
                    'mjd': float(row_fm['MJD']),
                    'expid': int(row_rv['EXPID']),
                    'success': bool(row_rv['SUCCESS']),
                    'rvs_warn': int(row_rv['RVS_WARN']) if 'RVS_WARN' in rvtab.dtype.names else 0,
                    'sn_r': float(row_rv['SN_R']) if 'SN_R' in rvtab.dtype.names else np.nan,
                }
                all_epochs.append(epoch)

    return all_epochs


def deduplicate_epochs(epochs):
    """De-duplicate epochs by EXPID or MJD proximity."""
    if not epochs:
        return []

    sorted_epochs = sorted(epochs, key=lambda x: x['mjd'])
    unique_epochs = []
    used_expids = set()

    for epoch in sorted_epochs:
        expid = epoch['expid']
        mjd = epoch['mjd']

        if expid in used_expids and expid != -1:
            continue

        is_duplicate = False
        for existing in unique_epochs:
            if abs(mjd - existing['mjd']) < 1e-3:
                if epoch['rv_err'] < existing['rv_err']:
                    unique_epochs.remove(existing)
                    unique_epochs.append(epoch)
                is_duplicate = True
                break

        if not is_duplicate:
            unique_epochs.append(epoch)
            if expid != -1:
                used_expids.add(expid)

    return unique_epochs


def apply_quality_cuts(epochs):
    """Apply per-epoch quality cuts."""
    quality_epochs = []
    for e in epochs:
        rv, rv_err = e['rv'], e['rv_err']
        if not np.isfinite(rv) or not np.isfinite(rv_err):
            continue
        if rv_err >= MAX_VRAD_ERR:
            continue
        if abs(rv) >= MAX_ABS_VRAD:
            continue
        quality_epochs.append(e)
    return quality_epochs


def compute_metrics(epochs):
    """Compute RV diagnostics from filtered epochs."""
    if len(epochs) < 2:
        return {
            'n_epochs': len(epochs),
            'mjd_span': 0,
            'delta_rv': 0,
            'S': 0,
            'S_robust': 0,
            'd_max': 0,
            'n_success_true': sum(1 for e in epochs if e['success']),
            'n_success_false': sum(1 for e in epochs if not e['success']),
            'max_warn_flag': max((e['rvs_warn'] for e in epochs), default=0),
        }

    rvs = np.array([e['rv'] for e in epochs])
    errs = np.array([e['rv_err'] for e in epochs])
    mjds = np.array([e['mjd'] for e in epochs])

    delta_rv = np.max(rvs) - np.min(rvs)
    mjd_span = np.max(mjds) - np.min(mjds)
    sigma_combined = np.sqrt(np.sum(errs**2))
    S = delta_rv / sigma_combined if sigma_combined > 0 else 0

    rv_median = np.median(rvs)
    d_values = np.abs(rvs - rv_median) / errs
    d_max = np.max(d_values) if len(d_values) > 0 else 0

    # Leave-one-out
    S_loo = []
    for i in range(len(rvs)):
        rvs_loo = np.delete(rvs, i)
        errs_loo = np.delete(errs, i)
        if len(rvs_loo) >= 2:
            delta_loo = np.max(rvs_loo) - np.min(rvs_loo)
            sigma_loo = np.sqrt(np.sum(errs_loo**2))
            S_loo.append(delta_loo / sigma_loo if sigma_loo > 0 else 0)
        else:
            S_loo.append(0)

    S_min_loo = np.min(S_loo) if S_loo else 0
    S_robust = min(S, S_min_loo)

    return {
        'n_epochs': len(epochs),
        'mjd_span': mjd_span,
        'delta_rv': delta_rv,
        'S': S,
        'S_robust': S_robust,
        'd_max': d_max,
        'n_success_true': sum(1 for e in epochs if e['success']),
        'n_success_false': sum(1 for e in epochs if not e['success']),
        'max_warn_flag': max((e['rvs_warn'] for e in epochs), default=0),
    }


def query_simbad(gaia_source_id):
    """Query SIMBAD for object classification."""
    try:
        from astroquery.simbad import Simbad
        Simbad.reset_votable_fields()
        Simbad.add_votable_fields('otype')

        gaia_id_str = f'Gaia DR3 {gaia_source_id}'
        result = Simbad.query_object(gaia_id_str)

        if result is not None and len(result) > 0:
            main_id = result['main_id'][0]
            otype = result['otype'][0]
            if hasattr(main_id, 'decode'):
                main_id = main_id.decode('utf-8')
            if hasattr(otype, 'decode'):
                otype = otype.decode('utf-8')
            return {'simbad_id': main_id, 'simbad_type': otype}
        else:
            return {'simbad_id': '', 'simbad_type': 'NO_MATCH'}
    except Exception as e:
        return {'simbad_id': '', 'simbad_type': 'NO_QUERY'}


def classify_verdict(metrics, simbad_info):
    """Classify the target based on metrics and SIMBAD info."""
    stype = simbad_info['simbad_type'].upper() if simbad_info['simbad_type'] else ''

    # Check SIMBAD type
    if 'QSO' in stype or 'AGN' in stype:
        return 'NOT_RELEVANT_AGN'
    if 'RR*' in stype or stype.startswith('RR'):
        return 'NOT_RELEVANT_PULSATOR'
    if 'EB*' in stype or 'BY*' in stype or 'SB*' in stype:
        return 'KNOWN_BINARY'

    # Check quality metrics
    n_epochs = metrics['n_epochs']
    S_robust = metrics['S_robust']
    n_success_true = metrics['n_success_true']
    n_success_false = metrics['n_success_false']
    max_warn = metrics['max_warn_flag']

    if n_epochs < 3:
        return 'NEEDS_HUMAN_REVIEW'
    if S_robust < 10:
        return 'NEEDS_HUMAN_REVIEW'

    # Majority SUCCESS=True and low warnings
    total = n_success_true + n_success_false
    if total > 0 and n_success_true / total >= 0.5 and max_warn <= 8:
        return 'CLEAN_STELLAR_CANDIDATE'
    else:
        return 'NEEDS_HUMAN_REVIEW'


def main():
    # Read input CSV
    print(f"Reading {INPUT_CSV}...")
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        targets = list(reader)
    print(f"Found {len(targets)} targets to analyze\n")

    results = []

    for i, target in enumerate(targets, 1):
        targetid = int(target['targetid'])
        gaia_id_str = target['gaia_source_id']
        gaia_source_id = int(gaia_id_str) if gaia_id_str.strip() else None
        rank = target.get('rank', i)

        print(f"[{i}/{len(targets)}] Processing TARGETID {targetid}...")

        # A) Extract epochs
        raw_epochs = extract_epochs_for_target(targetid, gaia_source_id)
        unique_epochs = deduplicate_epochs(raw_epochs)
        quality_epochs = apply_quality_cuts(unique_epochs)

        # B) Compute metrics
        metrics = compute_metrics(quality_epochs)

        # C) SIMBAD check
        if gaia_source_id:
            simbad_info = query_simbad(gaia_source_id)
        else:
            simbad_info = {'simbad_id': '', 'simbad_type': 'NO_GAIA_ID'}

        # D) Verdict
        verdict = classify_verdict(metrics, simbad_info)

        # Notes
        notes = []
        if metrics['n_success_false'] > 0:
            notes.append(f"{metrics['n_success_false']} epochs SUCCESS=False")
        if metrics['max_warn_flag'] > 0:
            notes.append(f"max_warn={metrics['max_warn_flag']}")
        if metrics['n_epochs'] < 3:
            notes.append("N<3")

        result = {
            'rank': rank,
            'targetid': targetid,
            'gaia_source_id': gaia_source_id if gaia_source_id else '',
            'N_epochs': metrics['n_epochs'],
            'MJD_span': round(metrics['mjd_span'], 4),
            'delta_rv_kms': round(metrics['delta_rv'], 2),
            'S': round(metrics['S'], 2),
            'S_robust': round(metrics['S_robust'], 2),
            'd_max': round(metrics['d_max'], 2),
            'n_success_true': metrics['n_success_true'],
            'n_success_false': metrics['n_success_false'],
            'max_warn_flag': metrics['max_warn_flag'],
            'simbad_type': simbad_info['simbad_type'],
            'simbad_id': simbad_info['simbad_id'],
            'verdict': verdict,
            'notes': '; '.join(notes) if notes else '',
        }
        results.append(result)

        # Store epochs for later plotting
        result['_epochs'] = quality_epochs

        print(f"    N={metrics['n_epochs']}, S_robust={metrics['S_robust']:.1f}, "
              f"SIMBAD={simbad_info['simbad_type']}, verdict={verdict}")

    print()

    # Write outputs
    fieldnames = ['rank', 'targetid', 'gaia_source_id', 'N_epochs', 'MJD_span',
                  'delta_rv_kms', 'S', 'S_robust', 'd_max', 'n_success_true',
                  'n_success_false', 'max_warn_flag', 'simbad_type', 'simbad_id',
                  'verdict', 'notes']

    # All results
    print(f"Writing {OUTPUT_ALL}...")
    with open(OUTPUT_ALL, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    # Clean subset
    clean = [r for r in results if r['verdict'] == 'CLEAN_STELLAR_CANDIDATE']
    clean.sort(key=lambda x: float(x['S_robust']), reverse=True)
    print(f"Writing {OUTPUT_CLEAN}... ({len(clean)} targets)")
    with open(OUTPUT_CLEAN, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(clean)

    # Not relevant
    not_relevant = [r for r in results if r['verdict'].startswith('NOT_RELEVANT')]
    print(f"Writing {OUTPUT_NOT_RELEVANT}... ({len(not_relevant)} targets)")
    with open(OUTPUT_NOT_RELEVANT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(not_relevant)

    # Summary
    print()
    print("=" * 70)
    print("DEEP-DIVE SUMMARY")
    print("=" * 70)
    print()

    # Counts by verdict
    verdict_counts = {}
    for r in results:
        v = r['verdict']
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    print("Counts by verdict:")
    for v in sorted(verdict_counts.keys()):
        print(f"  {v}: {verdict_counts[v]}")
    print()

    # Top 10 clean candidates
    if clean:
        print(f"Top {min(10, len(clean))} CLEAN_STELLAR_CANDIDATE targets:")
        print(f"{'Rank':<6}{'TARGETID':<22}{'N':<4}{'S_robust':<10}{'ΔRV':<10}{'d_max':<8}{'SIMBAD':<15}")
        print("-" * 80)
        for r in clean[:10]:
            stype = r['simbad_type'][:12] if r['simbad_type'] else 'NO_MATCH'
            print(f"{r['rank']:<6}{r['targetid']:<22}{r['N_epochs']:<4}{r['S_robust']:<10}"
                  f"{r['delta_rv_kms']:<10}{r['d_max']:<8.1f}{stype:<15}")
        print()

    # Surprises
    surprises = [r for r in results if r['verdict'].startswith('NOT_RELEVANT')]
    if surprises:
        print("Surprises (objects that should have been excluded earlier):")
        for r in surprises:
            print(f"  TARGETID {r['targetid']}: {r['simbad_type']} -> {r['verdict']}")
        print()

    # Store clean results with epochs for plotting
    return clean


if __name__ == "__main__":
    clean_results = main()

    # Save clean results for plotting
    import json
    clean_for_plot = []
    for r in clean_results[:10]:
        entry = {k: v for k, v in r.items() if not k.startswith('_')}
        entry['epochs'] = r.get('_epochs', [])
        clean_for_plot.append(entry)

    with open('data/derived/deep_dive_clean_for_plot.json', 'w') as f:
        json.dump(clean_for_plot, f, indent=2, default=float)

    print("Saved top 10 clean candidates with epochs to data/derived/deep_dive_clean_for_plot.json")
