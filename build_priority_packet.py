#!/usr/bin/env python3
"""
build_priority_packet.py - Build follow-up-ready Priority A candidate packet

This script applies data-driven gating thresholds to Tier A candidates and
produces a clean observer packet for spectroscopic follow-up.

GATING THRESHOLDS (data-driven from quantile analysis):
- Tier A only (N_epochs >= 3): already filtered
- S_robust >= 10: approximately 60th-80th percentile across surveys
- mjd_span >= 0.5 days: already enforced in triage
- No hard d_max cutoff: high d_max represents real large RV excursions
  (audit confirmed σ_i values are reasonable, not artificially small)

FLAGS:
- d_max > 100: "high-leverage epoch" - one epoch dominates the signal
  (could be real high-amplitude variability OR needs inspection)

OUTPUTS:
- data/derived/priorityA_bright.csv
- data/derived/priorityA_dark.csv
- data/derived/priorityA_master.csv (merged, deduped by gaia_source_id)
- data/derived/observer_packet/plots/*.pdf
- data/derived/observer_packet/README_packet.md

Author: Claude (Anthropic)
Date: 2026-01-13
"""

import argparse
import shutil
from pathlib import Path
import numpy as np

try:
    import fitsio
    USE_FITSIO = True
except ImportError:
    from astropy.io import fits
    USE_FITSIO = False

# Optional matplotlib for plot generation
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# GATING THRESHOLDS (data-driven)
# ============================================================================

# Minimum S_robust threshold
# Justification: ~60th-80th percentile across both surveys
# Bright 80th: 35.1, Dark 80th: 20.3 → use 10 to be inclusive
S_ROBUST_MIN = 10.0

# Minimum MJD span (days) - already enforced in triage
MJD_SPAN_MIN = 0.5

# d_max warning threshold (not a cutoff)
# Values above this are flagged as "high-leverage epoch"
D_MAX_WARNING = 100.0


# ============================================================================
# Data Loading
# ============================================================================

def load_triage_csv(csv_path):
    """Load triage CSV and return list of candidate dicts."""
    candidates = []
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        col_idx = {name: i for i, name in enumerate(header)}

        for line in f:
            parts = line.strip().split(',')

            # Handle missing Gaia ID (empty string in CSV)
            gaia_str = parts[col_idx['gaia_source_id']]
            gaia_id = int(gaia_str) if gaia_str.strip() else None

            candidates.append({
                'rank': int(parts[col_idx['rank']]),
                'targetid': int(parts[col_idx['targetid']]),
                'gaia_source_id': gaia_id,
                'tier': parts[col_idx['tier']],
                'n_epochs': int(parts[col_idx['n_epochs']]),
                'mjd_span': float(parts[col_idx['mjd_span']]),
                'S': float(parts[col_idx['S']]),
                'S_min_LOO': float(parts[col_idx['S_min_LOO']]),
                'S_med_LOO': float(parts[col_idx['S_med_LOO']]),
                'S_robust': float(parts[col_idx['S_robust']]),
                'd_max': float(parts[col_idx['d_max']]),
                'delta_rv_kms': float(parts[col_idx['delta_rv_kms']]),
                'rv_err_median_kms': float(parts[col_idx['rv_err_median_kms']])
            })
    return candidates


def apply_priority_gate(candidates, s_robust_min=S_ROBUST_MIN):
    """
    Apply Priority A gating thresholds.

    Returns filtered list of candidates meeting:
    - Tier == 'A' (N_epochs >= 3)
    - S_robust >= s_robust_min
    - mjd_span >= MJD_SPAN_MIN
    """
    priority = []
    for c in candidates:
        if c['tier'] != 'A':
            continue
        if c['S_robust'] < s_robust_min:
            continue
        if c['mjd_span'] < MJD_SPAN_MIN:
            continue
        priority.append(c)

    # Sort by S_robust descending
    priority.sort(key=lambda x: x['S_robust'], reverse=True)
    return priority


def generate_notes(c):
    """Generate notes string for a candidate."""
    notes = []

    # LOO stability assessment
    if c['S_robust'] > 0 and c['S'] > 0:
        stability = c['S_min_LOO'] / c['S']
        if stability > 0.8:
            notes.append("LOO-stable")
        elif stability > 0.5:
            notes.append("LOO-moderate")
        else:
            notes.append("LOO-drops")

    # High leverage warning
    if c['d_max'] > D_MAX_WARNING:
        notes.append(f"high-leverage(d={c['d_max']:.0f})")

    # Epoch count
    if c['n_epochs'] >= 5:
        notes.append(f"N={c['n_epochs']}")
    elif c['n_epochs'] == 3:
        notes.append("N=3(min)")

    return "; ".join(notes) if notes else "OK"


def write_priority_csv(filepath, candidates, survey):
    """Write Priority A CSV with notes."""
    columns = [
        'rank', 'targetid', 'gaia_source_id', 'survey',
        'n_epochs', 'mjd_span', 'S', 'S_robust', 'd_max',
        'delta_rv_kms', 'rv_err_median_kms', 'notes'
    ]

    with open(filepath, 'w') as f:
        f.write(','.join(columns) + '\n')

        for i, c in enumerate(candidates):
            notes = generate_notes(c)
            # Escape notes for CSV
            if ',' in notes:
                notes = f'"{notes}"'

            # Use empty string for missing Gaia IDs
            gaia_str = str(c['gaia_source_id']) if c['gaia_source_id'] is not None else ''

            row = [
                str(i + 1),
                str(c['targetid']),
                gaia_str,
                survey,
                str(c['n_epochs']),
                f"{c['mjd_span']:.4f}",
                f"{c['S']:.4f}",
                f"{c['S_robust']:.4f}",
                f"{c['d_max']:.4f}",
                f"{c['delta_rv_kms']:.4f}",
                f"{c['rv_err_median_kms']:.4f}",
                notes
            ]
            f.write(','.join(row) + '\n')

    print(f"  Wrote {len(candidates)} candidates to {filepath}")


def merge_and_dedup(bright_candidates, dark_candidates):
    """
    Merge bright and dark candidates, dedup by gaia_source_id.

    When duplicates exist, keep the one with higher S_robust.
    """
    # Combine all candidates with survey tag
    all_candidates = []
    for c in bright_candidates:
        c['survey'] = 'bright'
        all_candidates.append(c)
    for c in dark_candidates:
        c['survey'] = 'dark'
        all_candidates.append(c)

    # Dedup by gaia_source_id (keep higher S_robust)
    # If Gaia ID is missing (None), use TARGETID as unique key
    seen = {}
    for c in all_candidates:
        gaia_id = c['gaia_source_id']
        if gaia_id is None:
            # No Gaia ID - use targetid as key
            key = f"tid_{c['targetid']}"
        else:
            key = f"gaia_{gaia_id}"

        if key not in seen or c['S_robust'] > seen[key]['S_robust']:
            seen[key] = c

    # Convert back to list and sort by S_robust
    master = list(seen.values())
    master.sort(key=lambda x: x['S_robust'], reverse=True)

    return master


def write_master_csv(filepath, candidates):
    """Write master Priority A CSV."""
    columns = [
        'rank', 'targetid', 'gaia_source_id', 'survey',
        'n_epochs', 'mjd_span', 'S', 'S_robust', 'd_max',
        'delta_rv_kms', 'rv_err_median_kms', 'notes'
    ]

    with open(filepath, 'w') as f:
        f.write(','.join(columns) + '\n')

        for i, c in enumerate(candidates):
            notes = generate_notes(c)
            if ',' in notes:
                notes = f'"{notes}"'

            # Use empty string for missing Gaia IDs
            gaia_str = str(c['gaia_source_id']) if c['gaia_source_id'] is not None else ''

            row = [
                str(i + 1),
                str(c['targetid']),
                gaia_str,
                c['survey'],
                str(c['n_epochs']),
                f"{c['mjd_span']:.4f}",
                f"{c['S']:.4f}",
                f"{c['S_robust']:.4f}",
                f"{c['d_max']:.4f}",
                f"{c['delta_rv_kms']:.4f}",
                f"{c['rv_err_median_kms']:.4f}",
                notes
            ]
            f.write(','.join(row) + '\n')

    print(f"  Wrote {len(candidates)} candidates to {filepath}")


# ============================================================================
# Plot Management
# ============================================================================

def copy_plots(master_candidates, source_dirs, dest_dir, top_n=20):
    """
    Copy existing plots for top N master candidates to observer packet.

    Parameters
    ----------
    master_candidates : list
        Merged master candidate list
    source_dirs : dict
        {'bright': Path, 'dark': Path} to existing plot directories
    dest_dir : Path
        Destination directory for plots
    top_n : int
        Number of top candidates to include
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []

    for c in master_candidates[:top_n]:
        tid = c['targetid']
        survey = c['survey']

        # Look for existing plot
        src_path = source_dirs[survey] / f'rv_target_{tid}.pdf'

        if src_path.exists():
            dst_path = dest_dir / f'rv_target_{tid}.pdf'
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            missing.append((tid, survey))

    print(f"  Copied {copied} plots to {dest_dir}")
    if missing:
        print(f"  Missing plots for {len(missing)} targets (would need regeneration)")

    return missing


def extract_epochs(fits_path, targetids):
    """Extract per-epoch data for targets (for plot generation)."""
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

                mask = np.isin(chunk_tid, list(targetids))
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
            mask = np.isin(rvtab['TARGETID'], list(targetids))

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


def generate_missing_plots(missing_targets, fits_paths, dest_dir, candidates_lookup):
    """Generate plots for missing targets."""
    if not HAS_MATPLOTLIB:
        print("  Matplotlib not available, cannot generate missing plots")
        return

    # Group by survey
    by_survey = {'bright': [], 'dark': []}
    for tid, survey in missing_targets:
        by_survey[survey].append(tid)

    for survey, tids in by_survey.items():
        if not tids:
            continue

        fits_path = fits_paths[survey]
        print(f"  Extracting epoch data for {len(tids)} {survey} targets...")
        target_data = extract_epochs(fits_path, tids)

        for tid in tids:
            data = target_data[tid]
            if len(data['rv']) < 2:
                continue

            c = candidates_lookup.get(tid, {})

            # Sort by MJD
            sort_idx = np.argsort(data['mjd'])
            mjd = data['mjd'][sort_idx]
            rv = data['rv'][sort_idx]
            err = data['rv_err'][sort_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(mjd, rv, yerr=err, fmt='o', capsize=3, capthick=1,
                        markersize=8, color='#2E86AB', ecolor='#A23B72',
                        markeredgecolor='black')

            med_rv = np.median(rv)
            ax.axhline(med_rv, color='gray', linestyle='--', alpha=0.5,
                       label=f'Median RV = {med_rv:.1f} km/s')

            S_robust = c.get('S_robust', 0)
            d_max = c.get('d_max', 0)
            n_epochs = len(rv)

            ax.set_xlabel('MJD (days)', fontsize=12)
            ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)
            ax.set_title(f'TARGETID {tid}\n'
                         f'N={n_epochs}, S_robust={S_robust:.1f}, d_max={d_max:.1f}',
                         fontsize=11)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(dest_dir / f'rv_target_{tid}.pdf', format='pdf', dpi=150)
            plt.close(fig)


def write_readme(filepath, bright_count, dark_count, master_count):
    """Write README for observer packet."""
    content = f"""# Priority A RV Variable Candidates - Observer Packet

## Summary

This packet contains {master_count} Priority A candidates selected from DESI DR1
Milky Way Survey radial velocity measurements. These are stars showing
statistically significant RV variability that warrants spectroscopic follow-up.

**This is NOT a claim of compact object companions.** These are RV-variable
candidates only. The variability could be due to:
- Binary companions (stellar or sub-stellar)
- Pulsations or other intrinsic variability
- Systematic effects or bad spectral fits

## Candidate Counts

| Survey | Tier A Total | Priority A (S_robust >= 10) |
|--------|--------------|----------------------------|
| Bright | varies | {bright_count} |
| Dark | varies | {dark_count} |
| Master (deduped) | - | {master_count} |

## Selection Criteria

Candidates passed these data-driven gates:

1. **Tier A only**: N_epochs >= 3 (at least 3 RV measurements)
2. **S_robust >= 10**: Conservative significance score
   - S_robust = min(S, S_min_LOO) where S_min_LOO is the minimum
     significance when any single epoch is dropped
   - This guards against single-epoch flukes
3. **MJD_span >= 0.5 days**: Observations span at least half a day

## Metric Definitions

| Metric | Definition |
|--------|------------|
| S | ΔRV_max / sqrt(Σ σ_i²) - original significance |
| S_robust | min(S, S_min_LOO) - conservative significance |
| S_min_LOO | Minimum S when any single epoch is dropped |
| d_max | max_i |RV_i - median(RV)| / σ_i - outlier leverage |
| ΔRV | max(RV) - min(RV) in km/s |

## Flags in Notes Column

- **LOO-stable**: S_min_LOO / S > 0.8 (signal robust to dropping any epoch)
- **LOO-moderate**: 0.5 < S_min_LOO / S <= 0.8
- **LOO-drops**: S_min_LOO / S <= 0.5 (signal depends on one epoch)
- **high-leverage(d=X)**: d_max > 100 (one epoch far from median)
- **N=X**: Number of epochs (N=3 is minimum for Tier A)

## What "Needs Follow-up" Means

All candidates in this list need additional observations to:
1. Confirm the RV variability with independent measurements
2. Characterize the orbit (if binary)
3. Rule out systematic effects

High-priority targets for follow-up:
- **LOO-stable** with **N >= 4**: Most robust signals
- **Large MJD_span**: Better orbital phase coverage
- **Moderate d_max**: Less likely to be single-epoch artifacts

## Files in This Packet

- `priorityA_bright.csv`: Bright survey candidates
- `priorityA_dark.csv`: Dark survey candidates
- `priorityA_master.csv`: Merged and deduplicated master list
- `plots/*.pdf`: RV vs MJD plots for top 20 master candidates

## Data Provenance

- Source: DESI DR1 Milky Way Survey "iron" VAC
- Files: rvpix_exp-main-bright.fits, rvpix_exp-main-dark.fits
- URL: https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0/
- Processing: triage_rv_candidates.py → build_priority_packet.py

## Caveats

1. **No orbit fitting performed**: ΔRV is max-min, not orbital amplitude
2. **No stellar parameter cuts**: No filtering by Teff, logg, [Fe/H], etc.
3. **Missing Gaia IDs**: Some targets have no Gaia SOURCE_ID (empty in CSV)
4. **VRAD_ERR is quoted pipeline uncertainty**: May not reflect true errors
5. **Per-measurement quality cuts applied**: VRAD_ERR < 10 km/s, |VRAD| < 500 km/s

---
Generated by build_priority_packet.py
Date: 2026-01-13
"""

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  Wrote {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build Priority A candidate packet for follow-up',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--s-robust-min', type=float, default=S_ROBUST_MIN,
                        help=f'Minimum S_robust threshold (default: {S_ROBUST_MIN})')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot copying/generation')

    args = parser.parse_args()

    base_dir = Path('.')
    derived_dir = base_dir / 'data/derived'
    packet_dir = derived_dir / 'observer_packet'
    plots_dir = packet_dir / 'plots'

    print("="*70)
    print("BUILD PRIORITY A CANDIDATE PACKET")
    print("="*70)
    print(f"\nGating thresholds:")
    print(f"  S_robust >= {args.s_robust_min}")
    print(f"  MJD_span >= {MJD_SPAN_MIN} days")
    print(f"  d_max warning threshold: {D_MAX_WARNING}")

    # Load triage CSVs
    print("\n1. Loading triage candidates...")
    bright_all = load_triage_csv(derived_dir / 'triage_candidates_bright.csv')
    dark_all = load_triage_csv(derived_dir / 'triage_candidates_dark.csv')
    print(f"   Bright: {len(bright_all)} total, {sum(1 for c in bright_all if c['tier']=='A')} Tier A")
    print(f"   Dark:   {len(dark_all)} total, {sum(1 for c in dark_all if c['tier']=='A')} Tier A")

    # Apply gating
    print("\n2. Applying Priority A gate...")
    s_robust_min = args.s_robust_min

    bright_priority = apply_priority_gate(bright_all, s_robust_min)
    dark_priority = apply_priority_gate(dark_all, s_robust_min)
    print(f"   Bright Priority A: {len(bright_priority)}")
    print(f"   Dark Priority A:   {len(dark_priority)}")

    # Write per-survey CSVs
    print("\n3. Writing per-survey Priority A CSVs...")
    write_priority_csv(derived_dir / 'priorityA_bright.csv', bright_priority, 'bright')
    write_priority_csv(derived_dir / 'priorityA_dark.csv', dark_priority, 'dark')

    # Merge and dedup
    print("\n4. Merging and deduplicating...")
    master = merge_and_dedup(bright_priority, dark_priority)
    print(f"   Master list: {len(master)} unique candidates")

    # Write master CSV
    print("\n5. Writing master Priority A CSV...")
    write_master_csv(derived_dir / 'priorityA_master.csv', master)

    # Create observer packet
    print("\n6. Creating observer packet...")
    packet_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Copy existing plots
        source_dirs = {
            'bright': derived_dir / 'plots/bright',
            'dark': derived_dir / 'plots/dark'
        }
        missing = copy_plots(master, source_dirs, plots_dir, top_n=20)

        # Generate missing plots if needed
        if missing:
            fits_paths = {
                'bright': base_dir / 'data/raw/rvpix_exp-main-bright.fits',
                'dark': base_dir / 'data/raw/rvpix_exp-main-dark.fits'
            }
            candidates_lookup = {c['targetid']: c for c in master}
            generate_missing_plots(missing, fits_paths, plots_dir, candidates_lookup)

    # Write README
    write_readme(packet_dir / 'README_packet.md',
                 len(bright_priority), len(dark_priority), len(master))

    # Final report
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)

    print(f"\nTier A counts:")
    print(f"  Bright: {sum(1 for c in bright_all if c['tier']=='A')}")
    print(f"  Dark:   {sum(1 for c in dark_all if c['tier']=='A')}")

    print(f"\nPriority A survivors (S_robust >= {s_robust_min}):")
    print(f"  Bright: {len(bright_priority)}")
    print(f"  Dark:   {len(dark_priority)}")
    print(f"  Master (deduped): {len(master)}")

    print(f"\nTop 10 master candidates:")
    print("-"*110)
    print(f"{'Rank':<5} {'TARGETID':<20} {'Gaia_ID':<20} {'Survey':<7} {'N':<3} "
          f"{'MJD_span':<9} {'S':<8} {'S_robust':<9} {'d_max':<8} {'err_med':<8}")
    print("-"*110)

    for i, c in enumerate(master[:10]):
        gaia_str = str(c['gaia_source_id']) if c['gaia_source_id'] is not None else "N/A"
        print(f"{i+1:<5} {c['targetid']:<20} {gaia_str:<20} {c['survey']:<7} "
              f"{c['n_epochs']:<3} {c['mjd_span']:<9.1f} {c['S']:<8.1f} "
              f"{c['S_robust']:<9.1f} {c['d_max']:<8.1f} {c['rv_err_median_kms']:<8.2f}")

    # Red flags check
    print(f"\nRed flags check:")
    high_dmax = [c for c in master if c['d_max'] > 100]
    print(f"  Candidates with d_max > 100: {len(high_dmax)}/{len(master)}")

    loo_drops = [c for c in master if c['S_min_LOO'] / c['S'] <= 0.5 if c['S'] > 0]
    print(f"  Candidates with LOO-drops (signal depends on one epoch): {len(loo_drops)}/{len(master)}")

    n3_only = [c for c in master if c['n_epochs'] == 3]
    print(f"  Candidates with minimum N=3 epochs: {len(n3_only)}/{len(master)}")

    print("\n" + "="*70)
    print("OUTPUTS CREATED")
    print("="*70)
    print(f"  {derived_dir / 'priorityA_bright.csv'}")
    print(f"  {derived_dir / 'priorityA_dark.csv'}")
    print(f"  {derived_dir / 'priorityA_master.csv'}")
    print(f"  {packet_dir / 'README_packet.md'}")
    if not args.no_plots:
        print(f"  {plots_dir}/*.pdf")


if __name__ == '__main__':
    main()
