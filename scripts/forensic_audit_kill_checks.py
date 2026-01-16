#!/usr/bin/env python3
"""
FORENSIC AUDIT: Dark Companion Candidate Kill Checks
Target: Gaia DR3 3802130935635096832
RA: 164.523494, Dec: -1.660156
Suspect LAMOST ObsIDs: 437513049 and 870813030

Three "Kill Modes" to test:
1. LAMOST RV Version Control (DR7 vs DR9/DR10)
2. Blend/Contamination Audit (Gaia + Pan-STARRS)
3. Spectrum Reliability Check (SNR, subclass)
"""

import numpy as np
import requests
import json
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
import warnings
warnings.filterwarnings('ignore')

# Target info
GAIA_ID = 3802130935635096832
RA = 164.523494
DEC = -1.660156
LAMOST_OBSIDS = [437513049, 870813030]
ORIGINAL_DR7_RV = -49.36  # km/s from original analysis

print("=" * 70)
print("FORENSIC AUDIT: Dark Companion Kill Checks")
print("=" * 70)
print(f"Target: Gaia DR3 {GAIA_ID}")
print(f"Coordinates: RA={RA}, Dec={DEC}")
print(f"Original DR7 RV: {ORIGINAL_DR7_RV} km/s")
print("=" * 70)

results = {
    'task1_lamost_rv': {},
    'task2_blend_audit': {},
    'task3_spectrum_quality': {},
    'verdicts': {}
}

# =============================================================================
# TASK 1: LAMOST VERSION CONTROL CHECK
# =============================================================================
print("\n" + "=" * 70)
print("TASK 1: LAMOST VERSION CONTROL CHECK")
print("=" * 70)

def query_lamost_dr(obsid, dr_version):
    """Query LAMOST catalog via VizieR for a specific ObsID."""
    try:
        if dr_version == 'DR7':
            catalog = 'V/164/dr7'  # LAMOST DR7
        elif dr_version == 'DR9':
            catalog = 'V/164/dr9'  # LAMOST DR9
        elif dr_version == 'DR10':
            catalog = 'V/164/dr10'  # LAMOST DR10
        else:
            return None

        v = Vizier(columns=['*'], row_limit=10)
        v.ROW_LIMIT = 10

        # Query by coordinates with small radius
        coord = SkyCoord(ra=RA, dec=DEC, unit=(u.deg, u.deg))
        result = v.query_region(coord, radius=5*u.arcsec, catalog=catalog)

        if result and len(result) > 0:
            return result[0]
        return None
    except Exception as e:
        print(f"  VizieR query failed for {dr_version}: {e}")
        return None

def query_lamost_direct(obsid):
    """Query LAMOST directly via their API."""
    try:
        # LAMOST DR10 API
        url = f"http://www.lamost.org/dr10/v2.0/search?obsid={obsid}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

# Query each DR version
print("\nQuerying LAMOST catalogs via VizieR...")
print(f"Looking for ObsIDs: {LAMOST_OBSIDS}")

# Try VizieR for LAMOST data
coord = SkyCoord(ra=RA, dec=DEC, unit=(u.deg, u.deg))
v = Vizier(columns=['obsid', 'rv', 'rv_err', 'snrg', 'snri', 'snrr', 'subclass', 'class'], row_limit=50)

lamost_results = {}

# Query LAMOST DR7
print("\n--- LAMOST DR7 ---")
try:
    dr7 = v.query_region(coord, radius=5*u.arcsec, catalog='V/164/dr7')
    if dr7 and len(dr7) > 0:
        print(f"  Found {len(dr7[0])} entries")
        for row in dr7[0]:
            obsid = row['obsid'] if 'obsid' in row.colnames else 'N/A'
            rv = row['rv'] if 'rv' in row.colnames else np.nan
            rv_err = row['rv_err'] if 'rv_err' in row.colnames else np.nan
            snr = row['snrg'] if 'snrg' in row.colnames else np.nan
            subclass = row['subclass'] if 'subclass' in row.colnames else 'N/A'
            print(f"    ObsID: {obsid}, RV: {rv:.2f} +/- {rv_err:.2f} km/s, SNR_g: {snr:.1f}, Subclass: {subclass}")
            lamost_results[f'DR7_{obsid}'] = {'rv': float(rv), 'rv_err': float(rv_err), 'snr': float(snr), 'subclass': str(subclass)}
    else:
        print("  No entries found")
except Exception as e:
    print(f"  Query error: {e}")

# Query LAMOST DR9
print("\n--- LAMOST DR9 ---")
try:
    dr9 = v.query_region(coord, radius=5*u.arcsec, catalog='V/164/dr9')
    if dr9 and len(dr9) > 0:
        print(f"  Found {len(dr9[0])} entries")
        for row in dr9[0]:
            obsid = row['obsid'] if 'obsid' in row.colnames else 'N/A'
            rv = row['rv'] if 'rv' in row.colnames else np.nan
            rv_err = row['rv_err'] if 'rv_err' in row.colnames else np.nan
            snr = row['snrg'] if 'snrg' in row.colnames else np.nan
            subclass = row['subclass'] if 'subclass' in row.colnames else 'N/A'
            print(f"    ObsID: {obsid}, RV: {rv:.2f} +/- {rv_err:.2f} km/s, SNR_g: {snr:.1f}, Subclass: {subclass}")
            lamost_results[f'DR9_{obsid}'] = {'rv': float(rv), 'rv_err': float(rv_err), 'snr': float(snr), 'subclass': str(subclass)}
    else:
        print("  No entries found")
except Exception as e:
    print(f"  Query error: {e}")

# Query LAMOST DR10
print("\n--- LAMOST DR10 ---")
try:
    dr10 = v.query_region(coord, radius=5*u.arcsec, catalog='V/164/dr10')
    if dr10 and len(dr10) > 0:
        print(f"  Found {len(dr10[0])} entries")
        for row in dr10[0]:
            obsid = row['obsid'] if 'obsid' in row.colnames else 'N/A'
            rv = row['rv'] if 'rv' in row.colnames else np.nan
            rv_err = row['rv_err'] if 'rv_err' in row.colnames else np.nan
            snr = row['snrg'] if 'snrg' in row.colnames else np.nan
            subclass = row['subclass'] if 'subclass' in row.colnames else 'N/A'
            print(f"    ObsID: {obsid}, RV: {rv:.2f} +/- {rv_err:.2f} km/s, SNR_g: {snr:.1f}, Subclass: {subclass}")
            lamost_results[f'DR10_{obsid}'] = {'rv': float(rv), 'rv_err': float(rv_err), 'snr': float(snr), 'subclass': str(subclass)}
    else:
        print("  No entries found")
except Exception as e:
    print(f"  Query error: {e}")

# Also try direct LAMOST website query
print("\n--- Direct LAMOST Query (obsid search) ---")
for obsid in LAMOST_OBSIDS:
    try:
        url = f"http://www.lamost.org/dr10/v2.0/spectrum/view?obsid={obsid}"
        print(f"  ObsID {obsid}: Check manually at {url}")
    except:
        pass

results['task1_lamost_rv'] = lamost_results

# Analyze RV consistency
print("\n--- TASK 1 ANALYSIS ---")
if lamost_results:
    rvs = [v['rv'] for v in lamost_results.values() if not np.isnan(v['rv'])]
    if rvs:
        rv_mean = np.mean(rvs)
        rv_std = np.std(rvs)
        rv_range = max(rvs) - min(rvs)
        print(f"  RV values found: {rvs}")
        print(f"  Mean: {rv_mean:.2f} km/s, Std: {rv_std:.2f} km/s, Range: {rv_range:.2f} km/s")
        print(f"  Original DR7 value: {ORIGINAL_DR7_RV} km/s")

        # Check if DR9/DR10 significantly differs from DR7
        shift_from_dr7 = abs(rv_mean - ORIGINAL_DR7_RV)
        print(f"  Shift from original DR7: {shift_from_dr7:.2f} km/s")

        if shift_from_dr7 > 20:
            print("  *** WARNING: Significant RV shift detected! Pipeline artifact possible.")
            results['verdicts']['task1'] = 'KILL'
        else:
            print("  RV values appear consistent across DR versions.")
            results['verdicts']['task1'] = 'SURVIVE'
else:
    print("  No LAMOST data retrieved - unable to verify")
    results['verdicts']['task1'] = 'INCONCLUSIVE'

# =============================================================================
# TASK 2: BLEND/CONTAMINATION AUDIT
# =============================================================================
print("\n" + "=" * 70)
print("TASK 2: BLEND/CONTAMINATION AUDIT")
print("=" * 70)

# Query Gaia DR3 for the source and nearby sources
print("\n--- Gaia DR3 Query ---")
try:
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    # Query the target
    query = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag, parallax, parallax_error,
           ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
           ipd_frac_multi_peak, ipd_frac_odd_win, duplicated_source,
           phot_bp_rp_excess_factor
    FROM gaiadr3.gaia_source
    WHERE source_id = {GAIA_ID}
    """

    job = Gaia.launch_job(query)
    gaia_target = job.get_results()

    if len(gaia_target) > 0:
        row = gaia_target[0]
        print(f"  Source ID: {row['source_id']}")
        print(f"  G mag: {row['phot_g_mean_mag']:.2f}")
        print(f"  RUWE: {row['ruwe']:.3f}")
        print(f"  Astrometric Excess Noise: {row['astrometric_excess_noise']:.3f} mas")
        print(f"  IPD frac multi peak: {row['ipd_frac_multi_peak']}")
        print(f"  IPD frac odd win: {row['ipd_frac_odd_win']}")
        print(f"  Duplicated source: {row['duplicated_source']}")
        print(f"  BP-RP excess factor: {row['phot_bp_rp_excess_factor']:.3f}")

        results['task2_blend_audit']['gaia_target'] = {
            'source_id': int(row['source_id']),
            'g_mag': float(row['phot_g_mean_mag']),
            'ruwe': float(row['ruwe']),
            'aen': float(row['astrometric_excess_noise']),
            'ipd_frac_multi_peak': int(row['ipd_frac_multi_peak']) if row['ipd_frac_multi_peak'] else 0,
            'ipd_frac_odd_win': int(row['ipd_frac_odd_win']) if row['ipd_frac_odd_win'] else 0,
            'duplicated_source': bool(row['duplicated_source']),
            'bp_rp_excess': float(row['phot_bp_rp_excess_factor'])
        }

        # Check for elevated IPD fractions
        ipd_multi = row['ipd_frac_multi_peak'] if row['ipd_frac_multi_peak'] else 0
        ipd_odd = row['ipd_frac_odd_win'] if row['ipd_frac_odd_win'] else 0

        print(f"\n  --- IPD Fraction Analysis ---")
        print(f"  ipd_frac_multi_peak: {ipd_multi}% (threshold: >10% is concerning)")
        print(f"  ipd_frac_odd_win: {ipd_odd}% (threshold: >10% is concerning)")

        if ipd_multi > 10 or ipd_odd > 10:
            print("  *** WARNING: Elevated IPD fractions suggest possible blend!")
        else:
            print("  IPD fractions within normal range.")

except Exception as e:
    print(f"  Gaia query error: {e}")

# Query for nearby sources within 2 arcsec
print("\n--- Nearby Source Search (within 2\") ---")
try:
    query_neighbors = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag,
           DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {RA}, {DEC})) AS dist_arcsec
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {RA}, {DEC}, 0.000556))
    ORDER BY dist_arcsec
    """

    job = Gaia.launch_job(query_neighbors)
    neighbors = job.get_results()

    print(f"  Found {len(neighbors)} Gaia sources within 2\"")
    neighbor_list = []
    for row in neighbors:
        dist = row['dist_arcsec'] * 3600  # convert to arcsec
        print(f"    Source {row['source_id']}: G={row['phot_g_mean_mag']:.2f}, dist={dist:.3f}\"")
        neighbor_list.append({
            'source_id': int(row['source_id']),
            'g_mag': float(row['phot_g_mean_mag']),
            'dist_arcsec': float(dist)
        })

    results['task2_blend_audit']['gaia_neighbors'] = neighbor_list

    # Check for close companions (excluding target itself)
    close_companions = [n for n in neighbor_list if n['dist_arcsec'] > 0.01 and n['dist_arcsec'] < 1.5]
    if close_companions:
        print(f"\n  *** WARNING: {len(close_companions)} source(s) within 1.5\" could contaminate fiber!")
        for c in close_companions:
            print(f"      Source {c['source_id']}: G={c['g_mag']:.2f}, dist={c['dist_arcsec']:.3f}\"")
    else:
        print(f"\n  No sources within 1.5\" (excluding target) - fiber contamination unlikely from Gaia sources")

except Exception as e:
    print(f"  Neighbor query error: {e}")

# Query Pan-STARRS
print("\n--- Pan-STARRS DR2 Query ---")
try:
    v_ps = Vizier(columns=['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag'], row_limit=20)
    ps_result = v_ps.query_region(coord, radius=3*u.arcsec, catalog='II/349/ps1')

    if ps_result and len(ps_result) > 0:
        print(f"  Found {len(ps_result[0])} Pan-STARRS sources within 3\"")
        ps_sources = []
        for row in ps_result[0]:
            ps_coord = SkyCoord(ra=row['RAJ2000'], dec=row['DEJ2000'], unit=(u.deg, u.deg))
            sep = coord.separation(ps_coord).arcsec
            gmag = row['gmag'] if 'gmag' in row.colnames and not np.ma.is_masked(row['gmag']) else np.nan
            print(f"    objID {row['objID']}: g={gmag:.2f}, sep={sep:.3f}\"")
            ps_sources.append({'objID': str(row['objID']), 'gmag': float(gmag), 'sep_arcsec': float(sep)})
        results['task2_blend_audit']['panstarrs'] = ps_sources
    else:
        print("  No Pan-STARRS sources found within 3\"")
except Exception as e:
    print(f"  Pan-STARRS query error: {e}")

# TASK 2 Verdict
print("\n--- TASK 2 ANALYSIS ---")
blend_flags = []
if 'gaia_target' in results['task2_blend_audit']:
    gt = results['task2_blend_audit']['gaia_target']
    if gt['ipd_frac_multi_peak'] > 10:
        blend_flags.append(f"ipd_frac_multi_peak={gt['ipd_frac_multi_peak']}% (>10%)")
    if gt['ipd_frac_odd_win'] > 10:
        blend_flags.append(f"ipd_frac_odd_win={gt['ipd_frac_odd_win']}% (>10%)")
    if gt['duplicated_source']:
        blend_flags.append("duplicated_source=True")
    if gt['bp_rp_excess'] > 1.3:
        blend_flags.append(f"bp_rp_excess={gt['bp_rp_excess']:.2f} (>1.3)")

if 'gaia_neighbors' in results['task2_blend_audit']:
    close = [n for n in results['task2_blend_audit']['gaia_neighbors'] if n['dist_arcsec'] > 0.01 and n['dist_arcsec'] < 1.5]
    if close:
        blend_flags.append(f"{len(close)} Gaia source(s) within 1.5\"")

if blend_flags:
    print(f"  BLEND FLAGS RAISED:")
    for f in blend_flags:
        print(f"    - {f}")
    results['verdicts']['task2'] = 'KILL' if len(blend_flags) >= 2 else 'CAUTION'
else:
    print("  No significant blend flags detected.")
    results['verdicts']['task2'] = 'SURVIVE'

# =============================================================================
# TASK 3: SPECTRUM RELIABILITY CHECK
# =============================================================================
print("\n" + "=" * 70)
print("TASK 3: SPECTRUM RELIABILITY CHECK")
print("=" * 70)

print("\n--- SNR and Subclass Analysis ---")
if lamost_results:
    snrs = []
    subclasses = []
    for key, val in lamost_results.items():
        snrs.append(val['snr'])
        subclasses.append(val['subclass'])
        print(f"  {key}: SNR_g={val['snr']:.1f}, Subclass={val['subclass']}")

    results['task3_spectrum_quality']['snrs'] = snrs
    results['task3_spectrum_quality']['subclasses'] = subclasses

    # Check SNR threshold
    low_snr = [s for s in snrs if s < 5 and not np.isnan(s)]
    if low_snr:
        print(f"\n  *** WARNING: {len(low_snr)} observation(s) with SNR < 5!")
    else:
        print(f"\n  All observations have SNR >= 5")

    # Check subclass consistency
    unique_subclasses = list(set([s for s in subclasses if s != 'N/A' and s]))
    print(f"  Unique subclasses: {unique_subclasses}")

    # Check for wild inconsistencies (e.g., M0 vs Galaxy)
    stellar_types = [s for s in unique_subclasses if s and not s.startswith('Galaxy') and s != 'Unknown']
    galaxy_types = [s for s in unique_subclasses if s and s.startswith('Galaxy')]

    if stellar_types and galaxy_types:
        print("  *** CRITICAL: Mix of stellar and galaxy classifications!")
        results['verdicts']['task3'] = 'KILL'
    elif len(unique_subclasses) > 2:
        print("  *** WARNING: Multiple different subclass classifications")
        results['verdicts']['task3'] = 'CAUTION'
    else:
        print("  Subclass classifications appear consistent")
        results['verdicts']['task3'] = 'SURVIVE'
else:
    print("  No LAMOST data available for quality check")
    results['verdicts']['task3'] = 'INCONCLUSIVE'

# =============================================================================
# FINAL FORENSIC REPORT
# =============================================================================
print("\n" + "=" * 70)
print("FORENSIC REPORT: FINAL VERDICTS")
print("=" * 70)

print(f"""
Target: Gaia DR3 {GAIA_ID}
RA: {RA}, Dec: {DEC}

┌─────────────────────────────────────────────────────────────────────┐
│ KILL MODE                          │ VERDICT                       │
├─────────────────────────────────────────────────────────────────────┤
│ 1. LAMOST RV Version Control       │ {results['verdicts'].get('task1', 'N/A'):^29} │
│ 2. Blend/Contamination Audit       │ {results['verdicts'].get('task2', 'N/A'):^29} │
│ 3. Spectrum Reliability            │ {results['verdicts'].get('task3', 'N/A'):^29} │
└─────────────────────────────────────────────────────────────────────┘
""")

# Overall verdict
verdicts = [results['verdicts'].get('task1'), results['verdicts'].get('task2'), results['verdicts'].get('task3')]
kills = verdicts.count('KILL')
cautions = verdicts.count('CAUTION')

print("OVERALL ASSESSMENT:")
if kills >= 2:
    print("  *** CANDIDATE KILLED: Multiple fatal flags detected ***")
    overall = 'KILLED'
elif kills == 1:
    print("  *** CANDIDATE CRITICALLY WOUNDED: One fatal flag detected ***")
    overall = 'CRITICALLY_WOUNDED'
elif cautions >= 2:
    print("  *** CANDIDATE SUSPECT: Multiple caution flags - needs follow-up ***")
    overall = 'SUSPECT'
else:
    print("  *** CANDIDATE SURVIVES: No fatal issues found ***")
    overall = 'SURVIVES'

results['overall_verdict'] = overall

# Save results
with open('forensic_audit_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nDetailed results saved to forensic_audit_results.json")

print("\n" + "=" * 70)
print("FORENSIC AUDIT COMPLETE")
print("=" * 70)
