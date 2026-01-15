#!/usr/bin/env python3
"""
verify_candidates.py - Comprehensive validation of BH candidate companions

"Negative Space" validation: Looking for objects with GRAVITY (high Gaia RUWE)
but NO LIGHT (no infrared excess, no eclipses).

Queries:
1. Gaia DR3 - RUWE, proper motions, photometry
2. WISE/2MASS - Infrared colors (W1-W2)
3. SIMBAD - Known classifications

Author: Claude (Anthropic)
Date: 2026-01-14
"""

import os
import sys
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Astroquery imports
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.ipac.irsa import Irsa

# Lightkurve for TESS/Kepler data
import lightkurve as lk

# Configuration
INPUT_FILE = "data/derived/priorityA_followup_only.csv"
OUTPUT_FILE = "validation_results_full.csv"
PLOT_DIR = "validation_plots"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

print("=" * 70)
print("BH CANDIDATE VALIDATION PIPELINE")
print("=" * 70)
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# PHASE 1: Load Candidates
# =============================================================================
print("[PHASE 1] Loading candidate list...")

candidates = pd.read_csv(INPUT_FILE)
print(f"  Loaded {len(candidates)} candidates from {INPUT_FILE}")
print(f"  Columns: {list(candidates.columns)}")

# Extract Gaia source IDs
gaia_ids = candidates['gaia_source_id'].astype(str).tolist()
print(f"  Gaia Source IDs: {gaia_ids[:5]}... (showing first 5)")
print()

# =============================================================================
# PHASE 2: Query Functions
# =============================================================================

def query_gaia_single(source_id, retries=MAX_RETRIES):
    """Query Gaia DR3 for a single source."""
    for attempt in range(retries):
        try:
            query = f"""
            SELECT source_id, ra, dec, ruwe, pmra, pmdec, pmra_error, pmdec_error,
                   phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                   parallax, parallax_error, radial_velocity, radial_velocity_error,
                   astrometric_excess_noise, astrometric_excess_noise_sig,
                   ipd_gof_harmonic_amplitude, visibility_periods_used
            FROM gaiadr3.gaia_source
            WHERE source_id = {source_id}
            """
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            job = Gaia.launch_job(query)
            result = job.get_results()
            if len(result) > 0:
                return dict(result[0])
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [WARN] Gaia query failed for {source_id}: {e}")
                return None


def query_wise_single(ra, dec, radius_arcsec=5.0, retries=MAX_RETRIES):
    """Query AllWISE catalog for infrared photometry."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    for attempt in range(retries):
        try:
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
            result = Irsa.query_region(coord, catalog='allwise_p3as_psd',
                                        radius=radius_arcsec * u.arcsec)
            if len(result) > 0:
                # Return closest match
                row = result[0]
                return {
                    'w1mpro': float(row['w1mpro']) if row['w1mpro'] else None,
                    'w2mpro': float(row['w2mpro']) if row['w2mpro'] else None,
                    'w3mpro': float(row['w3mpro']) if row['w3mpro'] else None,
                    'w4mpro': float(row['w4mpro']) if row['w4mpro'] else None,
                    'j_m_2mass': float(row['j_m_2mass']) if 'j_m_2mass' in row.colnames and row['j_m_2mass'] else None,
                    'h_m_2mass': float(row['h_m_2mass']) if 'h_m_2mass' in row.colnames and row['h_m_2mass'] else None,
                    'k_m_2mass': float(row['k_m_2mass']) if 'k_m_2mass' in row.colnames and row['k_m_2mass'] else None,
                }
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [WARN] WISE query failed for ({ra}, {dec}): {e}")
                return None


def query_simbad_single(source_id, retries=MAX_RETRIES):
    """Query SIMBAD for object classification."""
    for attempt in range(retries):
        try:
            # Configure SIMBAD to return what we need
            custom_simbad = Simbad()
            custom_simbad.add_votable_fields('otype', 'otypes', 'rv_value', 'flux(V)')

            result = custom_simbad.query_object(f"Gaia DR3 {source_id}")
            if result is not None and len(result) > 0:
                row = result[0]
                return {
                    'simbad_main_id': str(row['MAIN_ID']) if row['MAIN_ID'] else None,
                    'simbad_otype': str(row['OTYPE']) if row['OTYPE'] else None,
                    'simbad_otypes': str(row['OTYPES']) if 'OTYPES' in row.colnames and row['OTYPES'] else None,
                }
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                # Not found is OK, just return None
                return None


def validate_candidate(row):
    """Validate a single candidate with all queries."""
    source_id = str(row['gaia_source_id'])
    targetid = row['targetid']

    result = {
        'targetid': targetid,
        'gaia_source_id': source_id,
        'rank': row['rank'],
        'S_robust': row['S_robust'],
        'delta_rv_kms': row['delta_rv_kms'],
        'n_epochs': row['n_epochs'],
        'mjd_span': row['mjd_span'],
    }

    # Query Gaia
    gaia_data = query_gaia_single(source_id)
    if gaia_data:
        result['ra'] = gaia_data.get('ra')
        result['dec'] = gaia_data.get('dec')
        result['ruwe'] = gaia_data.get('ruwe')
        result['pmra'] = gaia_data.get('pmra')
        result['pmdec'] = gaia_data.get('pmdec')
        result['phot_g_mean_mag'] = gaia_data.get('phot_g_mean_mag')
        result['phot_bp_mean_mag'] = gaia_data.get('phot_bp_mean_mag')
        result['phot_rp_mean_mag'] = gaia_data.get('phot_rp_mean_mag')
        result['parallax'] = gaia_data.get('parallax')
        result['parallax_error'] = gaia_data.get('parallax_error')
        result['gaia_rv'] = gaia_data.get('radial_velocity')
        result['gaia_rv_error'] = gaia_data.get('radial_velocity_error')
        result['astrometric_excess_noise'] = gaia_data.get('astrometric_excess_noise')
        result['astrometric_excess_noise_sig'] = gaia_data.get('astrometric_excess_noise_sig')
        result['ipd_gof_harmonic_amplitude'] = gaia_data.get('ipd_gof_harmonic_amplitude')
        result['gaia_status'] = 'OK'
    else:
        result['gaia_status'] = 'NO_DATA'

    # Query WISE (need RA/DEC from Gaia)
    if gaia_data and gaia_data.get('ra') and gaia_data.get('dec'):
        wise_data = query_wise_single(gaia_data['ra'], gaia_data['dec'])
        if wise_data:
            result['w1mpro'] = wise_data.get('w1mpro')
            result['w2mpro'] = wise_data.get('w2mpro')
            result['w3mpro'] = wise_data.get('w3mpro')
            result['w4mpro'] = wise_data.get('w4mpro')
            result['j_m_2mass'] = wise_data.get('j_m_2mass')
            result['h_m_2mass'] = wise_data.get('h_m_2mass')
            result['k_m_2mass'] = wise_data.get('k_m_2mass')

            # Calculate W1-W2 color
            if wise_data.get('w1mpro') and wise_data.get('w2mpro'):
                result['w1_w2_color'] = wise_data['w1mpro'] - wise_data['w2mpro']
            result['wise_status'] = 'OK'
        else:
            result['wise_status'] = 'NO_DATA'
    else:
        result['wise_status'] = 'NO_GAIA_COORDS'

    # Query SIMBAD
    simbad_data = query_simbad_single(source_id)
    if simbad_data:
        result['simbad_main_id'] = simbad_data.get('simbad_main_id')
        result['simbad_otype'] = simbad_data.get('simbad_otype')
        result['simbad_otypes'] = simbad_data.get('simbad_otypes')
        result['simbad_status'] = 'OK'
    else:
        result['simbad_status'] = 'NO_MATCH'

    return result


# =============================================================================
# PHASE 2: Run Parallel Queries
# =============================================================================
print("[PHASE 2] Querying Gaia DR3, WISE, and SIMBAD for all candidates...")
print("  (This may take a few minutes with parallel queries)")
print()

results = []
n_total = len(candidates)

# Use ThreadPoolExecutor for parallel I/O-bound queries
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(validate_candidate, row): i
               for i, row in candidates.iterrows()}

    for i, future in enumerate(as_completed(futures)):
        idx = futures[future]
        try:
            result = future.result()
            results.append(result)
            print(f"  [{i+1}/{n_total}] Validated Gaia {result['gaia_source_id']}: "
                  f"RUWE={result.get('ruwe', 'N/A')}, W1-W2={result.get('w1_w2_color', 'N/A')}")
        except Exception as e:
            print(f"  [{i+1}/{n_total}] ERROR: {e}")

print()

# =============================================================================
# PHASE 3: Scoring & Analysis
# =============================================================================
print("[PHASE 3] Scoring candidates...")

df = pd.DataFrame(results)

# Calculate BH_Probability score
def calculate_bh_score(row):
    """
    Score candidates based on "Negative Space" hypothesis:
    - HIGH if: ruwe > 1.4 AND (W1-W2) < 0.1 (gravity wobble, no IR excess)
    - LOW if: SIMBAD shows eclipsing binary OR (W1-W2) > 0.2 (red companion)
    """
    score = 50  # Neutral baseline
    reasons = []

    # Gravity check (RUWE)
    ruwe = row.get('ruwe')
    if ruwe is not None and not np.isnan(ruwe):
        if ruwe > 1.4:
            score += 25
            reasons.append(f"RUWE={ruwe:.2f}>1.4 (wobble)")
        elif ruwe > 1.2:
            score += 10
            reasons.append(f"RUWE={ruwe:.2f}>1.2 (mild wobble)")
        else:
            score -= 10
            reasons.append(f"RUWE={ruwe:.2f}<1.2 (stable)")

    # Astrometric excess noise (alternative gravity indicator)
    aen_sig = row.get('astrometric_excess_noise_sig')
    if aen_sig is not None and not np.isnan(aen_sig):
        if aen_sig > 2:
            score += 15
            reasons.append(f"AEN_sig={aen_sig:.1f}>2 (astrometric wobble)")

    # Darkness check (W1-W2 color)
    w1_w2 = row.get('w1_w2_color')
    if w1_w2 is not None and not np.isnan(w1_w2):
        if abs(w1_w2) < 0.1:
            score += 20
            reasons.append(f"W1-W2={w1_w2:.3f}~0 (no IR excess)")
        elif w1_w2 > 0.2:
            score -= 20
            reasons.append(f"W1-W2={w1_w2:.3f}>0.2 (possible red companion)")
        elif w1_w2 < -0.1:
            score += 10
            reasons.append(f"W1-W2={w1_w2:.3f}<-0.1 (blue, interesting)")

    # SIMBAD classification check
    otype = str(row.get('simbad_otype', '')).lower()
    otypes = str(row.get('simbad_otypes', '')).lower()
    combined_types = otype + ' ' + otypes

    # Penalize known variables/binaries
    if 'eclipsing' in combined_types or 'eb' in combined_types:
        score -= 30
        reasons.append("SIMBAD: Eclipsing Binary (BAD)")
    elif 'rrlyr' in combined_types or 'cepheid' in combined_types:
        score -= 25
        reasons.append("SIMBAD: Pulsating Variable (BAD)")
    elif 'agn' in combined_types or 'qso' in combined_types:
        score -= 30
        reasons.append("SIMBAD: AGN/QSO (BAD)")
    elif row.get('simbad_status') == 'NO_MATCH':
        score += 5
        reasons.append("SIMBAD: No match (unknown, interesting)")

    # RV variability from our analysis
    s_robust = row.get('S_robust')
    if s_robust is not None and not np.isnan(s_robust):
        if s_robust > 50:
            score += 15
            reasons.append(f"S_robust={s_robust:.1f}>50 (strong RV signal)")
        elif s_robust > 30:
            score += 10
            reasons.append(f"S_robust={s_robust:.1f}>30 (good RV signal)")

    # Clamp score
    score = max(0, min(100, score))

    return score, '; '.join(reasons)


# Apply scoring
scores = df.apply(calculate_bh_score, axis=1)
df['BH_Probability'] = [s[0] for s in scores]
df['BH_Reasons'] = [s[1] for s in scores]

# Sort by BH_Probability
df = df.sort_values('BH_Probability', ascending=False).reset_index(drop=True)
df['validation_rank'] = df.index + 1

print(f"  Scoring complete. Top 5 candidates:")
for i, row in df.head(5).iterrows():
    print(f"    #{row['validation_rank']}: Gaia {row['gaia_source_id']} | "
          f"BH_Prob={row['BH_Probability']} | RUWE={row.get('ruwe', 'N/A')}")

print()

# =============================================================================
# PHASE 4: Save Results
# =============================================================================
print("[PHASE 4] Saving results...")

# Reorder columns
output_cols = [
    'validation_rank', 'targetid', 'gaia_source_id', 'BH_Probability', 'BH_Reasons',
    'ra', 'dec', 'ruwe', 'astrometric_excess_noise', 'astrometric_excess_noise_sig',
    'pmra', 'pmdec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
    'parallax', 'parallax_error', 'gaia_rv', 'gaia_rv_error',
    'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro', 'w1_w2_color',
    'j_m_2mass', 'h_m_2mass', 'k_m_2mass',
    'simbad_main_id', 'simbad_otype', 'simbad_otypes',
    'S_robust', 'delta_rv_kms', 'n_epochs', 'mjd_span',
    'gaia_status', 'wise_status', 'simbad_status'
]
output_cols = [c for c in output_cols if c in df.columns]
df_out = df[output_cols]

df_out.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved full results to: {OUTPUT_FILE}")

# =============================================================================
# PHASE 5: Summary Table
# =============================================================================
print()
print("=" * 70)
print("TOP 3 BH CANDIDATES (Validation Ranking)")
print("=" * 70)
print()

top3 = df.head(3)
for i, row in top3.iterrows():
    print(f"RANK #{row['validation_rank']}: Gaia DR3 {row['gaia_source_id']}")
    print(f"  BH Probability Score: {row['BH_Probability']}/100")
    print(f"  RUWE: {row.get('ruwe', 'N/A')}")
    print(f"  W1-W2 Color: {row.get('w1_w2_color', 'N/A')}")
    print(f"  RV Amplitude: {row.get('delta_rv_kms', 'N/A'):.1f} km/s")
    print(f"  N_epochs: {row.get('n_epochs', 'N/A')}, MJD span: {row.get('mjd_span', 'N/A'):.1f} days")
    print(f"  SIMBAD: {row.get('simbad_otype', 'NO_MATCH')}")
    print(f"  Reasons: {row.get('BH_Reasons', '')}")
    print()

# =============================================================================
# PHASE 6: Deep Dive - Lightcurve for #1 Candidate
# =============================================================================
print("[PHASE 5] Deep Dive: Lightcurve for #1 Candidate...")

Path(PLOT_DIR).mkdir(exist_ok=True)

top_candidate = df.iloc[0]
top_gaia_id = top_candidate['gaia_source_id']
top_ra = top_candidate.get('ra')
top_dec = top_candidate.get('dec')

print(f"  Target: Gaia DR3 {top_gaia_id}")
print(f"  Coordinates: RA={top_ra}, DEC={top_dec}")

# Try to get TESS lightcurve
lightcurve_found = False
lc_source = None

if top_ra and top_dec:
    try:
        print("  Searching for TESS lightcurves...")
        search_result = lk.search_lightcurve(f"Gaia DR3 {top_gaia_id}", mission='TESS')

        if len(search_result) == 0:
            # Try coordinate search
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            coord = SkyCoord(ra=top_ra, dec=top_dec, unit=(u.deg, u.deg))
            search_result = lk.search_lightcurve(coord, mission='TESS', radius=10)

        if len(search_result) > 0:
            print(f"  Found {len(search_result)} TESS observation(s)")
            lc_collection = search_result.download_all()
            if lc_collection is not None and len(lc_collection) > 0:
                lc = lc_collection.stitch()
                lightcurve_found = True
                lc_source = 'TESS'
    except Exception as e:
        print(f"  TESS search failed: {e}")

# Try ZTF if TESS not available
if not lightcurve_found:
    try:
        print("  Searching for ZTF lightcurves...")
        search_result = lk.search_lightcurve(f"Gaia DR3 {top_gaia_id}", mission='ZTF')

        if len(search_result) > 0:
            print(f"  Found {len(search_result)} ZTF observation(s)")
            lc_collection = search_result.download_all()
            if lc_collection is not None and len(lc_collection) > 0:
                lc = lc_collection.stitch()
                lightcurve_found = True
                lc_source = 'ZTF'
    except Exception as e:
        print(f"  ZTF search failed: {e}")

# Try Kepler/K2 if still not found
if not lightcurve_found:
    try:
        print("  Searching for Kepler/K2 lightcurves...")
        for mission in ['Kepler', 'K2']:
            search_result = lk.search_lightcurve(f"Gaia DR3 {top_gaia_id}", mission=mission)
            if len(search_result) > 0:
                print(f"  Found {len(search_result)} {mission} observation(s)")
                lc_collection = search_result.download_all()
                if lc_collection is not None and len(lc_collection) > 0:
                    lc = lc_collection.stitch()
                    lightcurve_found = True
                    lc_source = mission
                    break
    except Exception as e:
        print(f"  Kepler/K2 search failed: {e}")

# Generate plot
plot_file = os.path.join(PLOT_DIR, "candidate_1_lightcurve.png")

if lightcurve_found:
    print(f"  Generating lightcurve plot from {lc_source}...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Full lightcurve
    ax1 = axes[0]
    lc.plot(ax=ax1, label=f'{lc_source} Lightcurve')
    ax1.set_title(f"Gaia DR3 {top_gaia_id} - {lc_source} Lightcurve")
    ax1.legend()

    # Flattened/detrended
    ax2 = axes[1]
    try:
        lc_flat = lc.flatten(window_length=1001)
        lc_flat.plot(ax=ax2, label='Flattened')

        # Calculate variability metrics
        flux_std = np.nanstd(lc_flat.flux.value)
        flux_median = np.nanmedian(lc_flat.flux.value)
        variability = flux_std / flux_median * 100  # percent

        ax2.set_title(f"Flattened Lightcurve (Variability: {variability:.3f}%)")
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline')
        ax2.legend()

        # Analysis
        print(f"  Lightcurve Variability: {variability:.3f}%")
        if variability < 1.0:
            print("  ANALYSIS: FLAT lightcurve - consistent with compact companion (GOOD)")
        elif variability < 5.0:
            print("  ANALYSIS: Low variability - possible ellipsoidal variations")
        else:
            print("  ANALYSIS: High variability - may be eclipsing or pulsating")

    except Exception as e:
        ax2.text(0.5, 0.5, f"Flattening failed: {e}", ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"  Saved: {plot_file}")

else:
    print("  No lightcurve data available from TESS/ZTF/Kepler")
    print("  Generating placeholder plot...")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5,
            f"No Lightcurve Data Available\n\nGaia DR3 {top_gaia_id}\n\n"
            f"RA: {top_ra:.5f}, DEC: {top_dec:.5f}\n"
            f"RUWE: {top_candidate.get('ruwe', 'N/A')}\n"
            f"W1-W2: {top_candidate.get('w1_w2_color', 'N/A')}\n"
            f"RV Amplitude: {top_candidate.get('delta_rv_kms', 'N/A'):.1f} km/s\n\n"
            f"This target requires ground-based photometric follow-up.",
            ha='center', va='center', fontsize=12, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"BH Candidate #1: Gaia DR3 {top_gaia_id}")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"  Saved: {plot_file}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print()
print(f"Total candidates validated: {len(df)}")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"Lightcurve plot: {plot_file}")
print()
print("Candidate Summary by BH Probability:")
print(f"  HIGH (>70): {len(df[df['BH_Probability'] > 70])}")
print(f"  MEDIUM (50-70): {len(df[(df['BH_Probability'] >= 50) & (df['BH_Probability'] <= 70)])}")
print(f"  LOW (<50): {len(df[df['BH_Probability'] < 50])}")
print()
print(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
