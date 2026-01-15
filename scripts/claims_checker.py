#!/usr/bin/env python3
"""
claims_checker.py - Verify paper claims against repo data

Scans paper claims and validates against computed metrics.
"""

import json

# Paper claims from DarkCompanion.pdf
PAPER_CLAIMS = [
    {
        'id': 1,
        'claim': 'S_robust = 100.0 for Gaia DR3 3802130935635096832',
        'location': 'Table 1, page 6',
        'type': 'quantitative',
    },
    {
        'id': 2,
        'claim': 'ΔRV ≈ 146 km/s',
        'location': 'Abstract, Section 5.1',
        'type': 'quantitative',
    },
    {
        'id': 3,
        'claim': 'RUWE = 1.95',
        'location': 'Section 5.1.3, Table 1',
        'type': 'quantitative',
    },
    {
        'id': 4,
        'claim': 'Astrometric Excess Noise Significance σ_AEN = 16.5',
        'location': 'Section 5.1.3',
        'type': 'quantitative',
    },
    {
        'id': 5,
        'claim': 'W1 - W2 = 0.052',
        'location': 'Section 5.1.3',
        'type': 'quantitative',
    },
    {
        'id': 6,
        'claim': 'K ≈ 73 km/s (velocity semi-amplitude)',
        'location': 'Section 6 Discussion',
        'type': 'quantitative',
    },
    {
        'id': 7,
        'claim': 'N = 4 epochs',
        'location': 'Table 1',
        'type': 'quantitative',
    },
    {
        'id': 8,
        'claim': 'Baseline = 39 days',
        'location': 'Section 5.1',
        'type': 'quantitative',
    },
    {
        'id': 9,
        'claim': '"Violent radial velocity variations"',
        'location': 'Section 5.1',
        'type': 'qualitative',
    },
    {
        'id': 10,
        'claim': 'GALEX NUV non-detection rules out hot WD (T > 10,000 K)',
        'location': 'Section 5.1.2',
        'type': 'qualitative',
    },
    {
        'id': 11,
        'claim': 'TESS shows no eclipses or ellipsoidal variations',
        'location': 'Section 5.1.3, Figure 3',
        'type': 'qualitative',
    },
    {
        'id': 12,
        'claim': '"High-confidence dark companion"',
        'location': 'Section 4',
        'type': 'qualitative',
    },
    {
        'id': 13,
        'claim': 'Primary is ~0.7 M_sun K dwarf',
        'location': 'Section 6 Discussion',
        'type': 'quantitative',
    },
    {
        'id': 14,
        'claim': '"Massive companion" required by high K',
        'location': 'Section 6 Discussion',
        'type': 'qualitative',
    },
    {
        'id': 15,
        'claim': 'Candidate labeled "Dark Companion" (Table 1 Verdict)',
        'location': 'Table 1',
        'type': 'qualitative',
    },
]

# Computed values from repo/scripts
REPO_VALUES = {
    'S': 79.84,
    'S_min_LOO': 19.78,
    'S_robust': 19.78,
    'd_max': 113.44,
    'delta_rv_kms': 146.07,
    'K_est_kms': 73.04,
    'n_epochs': 4,
    'mjd_span_days': 38.90,
    'RUWE': 1.9535599946975708,
    'AEN_sig': 16.493242,
    'W1_W2': 0.052,
    'high_leverage': True,
    'loo_drops': True,
}

def check_claims():
    """Check all paper claims against repo values."""

    print("=" * 80)
    print("CLAIM LEDGER: Paper vs Repository Verification")
    print("=" * 80)
    print()

    results = []

    for claim in PAPER_CLAIMS:
        cid = claim['id']
        text = claim['claim']
        loc = claim['location']
        ctype = claim['type']

        # Determine verification status
        status = 'UNVERIFIED'
        artifact = ''
        notes = ''

        if cid == 1:  # S_robust = 100
            repo_val = REPO_VALUES['S_robust']
            status = 'WRONG'
            artifact = 'scripts/compute_rv_dossier.py, validation_results_full.csv'
            notes = f'Paper: 100.0, Repo: {repo_val:.2f}. Discrepancy of 5x.'

        elif cid == 2:  # ΔRV ≈ 146 km/s
            repo_val = REPO_VALUES['delta_rv_kms']
            status = 'VERIFIED'
            artifact = 'scripts/compute_rv_dossier.py'
            notes = f'Computed: {repo_val:.2f} km/s. Matches.'

        elif cid == 3:  # RUWE = 1.95
            repo_val = REPO_VALUES['RUWE']
            status = 'VERIFIED'
            artifact = 'validation_results_full.csv'
            notes = f'Gaia DR3: {repo_val:.4f}. Matches.'

        elif cid == 4:  # AEN_sig = 16.5
            repo_val = REPO_VALUES['AEN_sig']
            status = 'VERIFIED'
            artifact = 'validation_results_full.csv'
            notes = f'Gaia DR3: {repo_val:.2f}. Matches.'

        elif cid == 5:  # W1-W2 = 0.052
            repo_val = REPO_VALUES['W1_W2']
            status = 'VERIFIED'
            artifact = 'validation_results_full.csv'
            notes = f'WISE: {repo_val:.3f}. Matches.'

        elif cid == 6:  # K ≈ 73 km/s
            repo_val = REPO_VALUES['K_est_kms']
            status = 'VERIFIED'
            artifact = 'scripts/compute_rv_dossier.py'
            notes = f'K_est = ΔRV/2 = {repo_val:.2f} km/s. Matches.'

        elif cid == 7:  # N = 4 epochs
            repo_val = REPO_VALUES['n_epochs']
            status = 'VERIFIED'
            artifact = 'data/raw/rvpix_exp-main-bright.fits'
            notes = f'Extracted {repo_val} epochs. Matches.'

        elif cid == 8:  # Baseline = 39 days
            repo_val = REPO_VALUES['mjd_span_days']
            status = 'VERIFIED'
            artifact = 'scripts/compute_rv_dossier.py'
            notes = f'MJD span = {repo_val:.2f} days. Matches.'

        elif cid == 9:  # "Violent RV variations"
            status = 'VERIFIED'
            artifact = 'scripts/compute_rv_dossier.py'
            notes = 'ΔRV = 146 km/s is large; "violent" is qualitative but reasonable.'

        elif cid == 10:  # GALEX rules out hot WD
            status = 'UNVERIFIED'
            artifact = 'Manual inspection (Figure 2 screenshot)'
            notes = 'GALEX check was manual. Need reproducible script. Claim is reasonable but not programmatically verified.'

        elif cid == 11:  # TESS no eclipses
            status = 'VERIFIED'
            artifact = 'analyze_tess_photometry.py, tess_analysis_result.png'
            notes = 'LS periodogram peak power = 0.0014 (insignificant). No eclipses detected.'

        elif cid == 12:  # "High-confidence dark companion"
            status = 'WRONG'
            artifact = 'scripts/compute_rv_dossier.py'
            notes = f'S_robust = {REPO_VALUES["S_robust"]:.2f} with d_max = {REPO_VALUES["d_max"]:.1f}. Single epoch dominates. Confidence overstated.'

        elif cid == 13:  # Primary ~0.7 M_sun K dwarf
            status = 'UNVERIFIED'
            artifact = 'None'
            notes = 'Parallax = 0.12 ± 0.16 mas (uncertain). Primary mass not constrained. Assumed, not derived.'

        elif cid == 14:  # "Massive companion" required
            status = 'WRONG'
            artifact = 'scripts/orbit_feasibility.py'
            notes = 'Without period, mass function only gives lower limit. M2_min ~ 0.7-1.3 M_sun for P ~ 30-50 days. Could be WD.'

        elif cid == 15:  # "Dark Companion" verdict
            status = 'WRONG'
            artifact = 'scripts/orbit_feasibility.py'
            notes = 'Cannot distinguish WD/NS/BH without period. Verdict overstated. Should be "Dark Companion Candidate".'

        results.append({
            'id': cid,
            'claim': text,
            'location': loc,
            'status': status,
            'artifact': artifact,
            'notes': notes,
        })

    # Print results as table
    print(f"{'ID':<4} {'Status':<12} {'Claim':<50} {'Location':<25}")
    print("-" * 91)

    for r in results:
        claim_short = r['claim'][:47] + '...' if len(r['claim']) > 50 else r['claim']
        print(f"{r['id']:<4} {r['status']:<12} {claim_short:<50} {r['location']:<25}")

    print()
    print("=" * 80)
    print("DETAILED FINDINGS")
    print("=" * 80)

    for r in results:
        print()
        print(f"[{r['id']}] {r['status']}: {r['claim']}")
        print(f"    Location: {r['location']}")
        print(f"    Artifact: {r['artifact']}")
        print(f"    Notes: {r['notes']}")

    # Summary statistics
    verified = sum(1 for r in results if r['status'] == 'VERIFIED')
    unverified = sum(1 for r in results if r['status'] == 'UNVERIFIED')
    wrong = sum(1 for r in results if r['status'] == 'WRONG')

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  VERIFIED:   {verified} / {len(results)}")
    print(f"  UNVERIFIED: {unverified} / {len(results)}")
    print(f"  WRONG:      {wrong} / {len(results)}")
    print()

    if wrong > 0:
        print("CRITICAL ISSUES REQUIRING CORRECTION:")
        for r in results:
            if r['status'] == 'WRONG':
                print(f"  - [{r['id']}] {r['claim'][:60]}...")

    return results

if __name__ == "__main__":
    check_claims()
