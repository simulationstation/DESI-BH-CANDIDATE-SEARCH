#!/usr/bin/env python3
"""
Gaia DR3 + LAMOST validation for Priority A follow-up candidates.

Outputs:
- data/derived/priorityA_followup_only_annotated_gaia_lamost.csv
- data/derived/VALIDATION_GAIA_LAMOST_REPORT.md
"""

import csv
import json
import time
import math
import urllib.request
import urllib.parse
import urllib.error
from io import StringIO

# Input/output files
INPUT_CSV = "data/derived/priorityA_followup_only.csv"
OUTPUT_CSV = "data/derived/priorityA_followup_only_annotated_gaia_lamost.csv"
OUTPUT_REPORT = "data/derived/VALIDATION_GAIA_LAMOST_REPORT.md"

# Gaia TAP endpoint
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

# VizieR TAP endpoint for LAMOST
VIZIER_TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def query_gaia_batch(source_ids):
    """
    Batch query Gaia DR3 for a list of source_ids.
    Returns dict mapping source_id -> row dict.
    """
    if not source_ids:
        return {}

    # Build ADQL query
    ids_str = ",".join(str(sid) for sid in source_ids)
    adql = f"""
    SELECT
        source_id,
        ra, dec, l, b,
        phot_g_mean_mag, bp_rp,
        parallax, parallax_error,
        pmra, pmdec,
        ruwe,
        astrometric_excess_noise,
        astrometric_params_solved,
        visibility_periods_used,
        phot_variable_flag,
        radial_velocity, radial_velocity_error,
        rv_nb_transits
    FROM gaiadr3.gaia_source
    WHERE source_id IN ({ids_str})
    """

    # Query parameters
    params = {
        'REQUEST': 'doQuery',
        'LANG': 'ADQL',
        'FORMAT': 'csv',
        'QUERY': adql.strip()
    }

    try:
        data = urllib.parse.urlencode(params).encode('utf-8')
        req = urllib.request.Request(GAIA_TAP_URL, data=data, method='POST')
        req.add_header('Content-Type', 'application/x-www-form-urlencoded')

        with urllib.request.urlopen(req, timeout=60) as response:
            result_text = response.read().decode('utf-8')

        # Parse CSV response
        reader = csv.DictReader(StringIO(result_text))
        results = {}
        for row in reader:
            sid = int(row['source_id'])
            results[sid] = row
        return results

    except Exception as e:
        print(f"  Gaia query failed: {e}")
        return {'_error': str(e)}


def query_lamost_by_coords(ra, dec, radius_arcsec=3.0):
    """
    Query VizieR for LAMOST DR5 sources near given coordinates.
    Returns dict with match info or None.
    """
    # LAMOST DR5 catalog in VizieR: V/164/dr5
    # Note: VizieR LAMOST table has 'z' (redshift) not separate 'RV' column
    # For stellar sources, RV ≈ c*z in km/s (valid for small z)
    url = (f'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?'
           f'-source=V/164/dr5&-c={ra}+{dec}&-c.rs={radius_arcsec}&'
           f'-out=ObsID,Target,z,snrg&-out.max=1&-sort=-snrg')

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as response:
            result_text = response.read().decode('utf-8')

        # Parse TSV response (skip comment lines and separator lines)
        lines = [l for l in result_text.strip().split('\n')
                 if l and not l.startswith('#') and not l.startswith('-')]

        # Need at least header + one data row (skip empty data rows)
        if len(lines) >= 2:
            headers = lines[0].split('\t')

            # Find first non-empty data row
            for line in lines[1:]:
                values = line.split('\t')
                # Check if this is actual data (not just whitespace)
                if any(v.strip() for v in values):
                    row = {}
                    for h, v in zip(headers, values):
                        row[h.strip()] = v.strip()

                    # Convert z to RV (c * z) for stellar sources
                    z_str = row.get('z', '')
                    rv_str = ''
                    if z_str:
                        try:
                            z_val = float(z_str)
                            rv_val = 299792.458 * z_val  # c in km/s
                            rv_str = f'{rv_val:.2f}'
                        except ValueError:
                            pass

                    return {
                        'lamost_match': 'YES',
                        'lamost_id': row.get('ObsID', ''),
                        'lamost_target': row.get('Target', ''),
                        'lamost_z': z_str,
                        'lamost_rv': rv_str,
                        'lamost_rv_err': '',  # Not available in this table
                        'lamost_snr': row.get('snrg', ''),
                        'lamost_obsdate': '',  # Not available in this table
                    }

        return {'lamost_match': 'NO'}

    except Exception as e:
        return {'lamost_match': 'QUERY_FAIL', 'lamost_error': str(e)}


def compute_gaia_flags(gaia_row):
    """
    Compute simple quality flags from Gaia data.
    """
    flags = {}

    # Astrometry flag
    try:
        ruwe_str = gaia_row.get('ruwe', '')
        excess_str = gaia_row.get('astrometric_excess_noise', '')

        if ruwe_str and excess_str:
            ruwe = float(ruwe_str)
            excess = float(excess_str)

            if ruwe <= 1.4 and excess < 2.0:
                flags['gaia_astrometry_flag'] = 'OK'
            else:
                flags['gaia_astrometry_flag'] = 'SUSPECT'
        else:
            flags['gaia_astrometry_flag'] = 'MISSING'
    except (ValueError, TypeError):
        flags['gaia_astrometry_flag'] = 'MISSING'

    # Photometric variability flag
    phot_var = gaia_row.get('phot_variable_flag', '')
    if phot_var and phot_var.upper() not in ('', 'NOT_AVAILABLE'):
        flags['gaia_photvar_flag'] = 'VAR'
    else:
        flags['gaia_photvar_flag'] = 'NOFLAG'

    return flags


def main():
    print("=" * 70)
    print("GAIA DR3 + LAMOST VALIDATION")
    print("=" * 70)
    print()

    # Read input CSV
    print(f"Reading {INPUT_CSV}...")
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        input_fieldnames = reader.fieldnames
        targets = list(reader)
    print(f"Found {len(targets)} targets\n")

    # Extract Gaia source IDs
    source_ids = []
    for t in targets:
        gaia_id_str = t.get('gaia_source_id', '')
        if gaia_id_str.strip():
            source_ids.append(int(gaia_id_str))

    print(f"Querying Gaia DR3 for {len(source_ids)} source IDs...")
    gaia_results = query_gaia_batch(source_ids)

    if '_error' in gaia_results:
        print(f"  ERROR: {gaia_results['_error']}")
        gaia_error = gaia_results['_error']
        gaia_results = {}
    else:
        gaia_error = None
        print(f"  Retrieved {len(gaia_results)} Gaia rows\n")

    # Query LAMOST for each target
    print("Querying LAMOST (VizieR) for each target...")
    lamost_results = {}

    for i, t in enumerate(targets, 1):
        gaia_id_str = t.get('gaia_source_id', '')
        if not gaia_id_str.strip():
            lamost_results[t['targetid']] = {'lamost_match': 'NO_GAIA_ID'}
            continue

        gaia_id = int(gaia_id_str)
        gaia_row = gaia_results.get(gaia_id, {})
        ra = gaia_row.get('ra', '')
        dec = gaia_row.get('dec', '')

        if ra and dec:
            try:
                ra_f = float(ra)
                dec_f = float(dec)
                print(f"  [{i}/{len(targets)}] Querying LAMOST for ({ra_f:.4f}, {dec_f:.4f})...")
                lamost_info = query_lamost_by_coords(ra_f, dec_f)
                lamost_results[t['targetid']] = lamost_info
                time.sleep(0.3)  # Rate limit
            except (ValueError, TypeError):
                lamost_results[t['targetid']] = {'lamost_match': 'BAD_COORDS'}
        else:
            lamost_results[t['targetid']] = {'lamost_match': 'NO_COORDS'}

    print()

    # Build output rows
    print("Building annotated output...")

    # Define new columns
    gaia_cols = [
        'gaia_ra', 'gaia_dec', 'gaia_l', 'gaia_b',
        'gaia_g_mag', 'gaia_bp_rp',
        'gaia_parallax', 'gaia_parallax_error',
        'gaia_pmra', 'gaia_pmdec',
        'gaia_ruwe', 'gaia_excess_noise',
        'gaia_phot_variable_flag',
        'gaia_rv', 'gaia_rv_error', 'gaia_rv_nb_transits',
        'gaia_astrometry_flag', 'gaia_photvar_flag'
    ]
    lamost_cols = [
        'lamost_match', 'lamost_id', 'lamost_target', 'lamost_z',
        'lamost_rv', 'lamost_snr'
    ]

    output_fieldnames = list(input_fieldnames) + gaia_cols + lamost_cols
    output_rows = []

    # Statistics for report
    stats = {
        'total': len(targets),
        'gaia_success': 0,
        'gaia_ruwe_high': 0,
        'gaia_photvar': 0,
        'gaia_has_rv': 0,
        'lamost_match': 0,
        'lamost_query_fail': 0,
    }

    for t in targets:
        row = dict(t)  # Copy original columns

        gaia_id_str = t.get('gaia_source_id', '')
        gaia_id = int(gaia_id_str) if gaia_id_str.strip() else None

        # Add Gaia columns
        if gaia_id and gaia_id in gaia_results:
            g = gaia_results[gaia_id]
            stats['gaia_success'] += 1

            row['gaia_ra'] = g.get('ra', '')
            row['gaia_dec'] = g.get('dec', '')
            row['gaia_l'] = g.get('l', '')
            row['gaia_b'] = g.get('b', '')
            row['gaia_g_mag'] = g.get('phot_g_mean_mag', '')
            row['gaia_bp_rp'] = g.get('bp_rp', '')
            row['gaia_parallax'] = g.get('parallax', '')
            row['gaia_parallax_error'] = g.get('parallax_error', '')
            row['gaia_pmra'] = g.get('pmra', '')
            row['gaia_pmdec'] = g.get('pmdec', '')
            row['gaia_ruwe'] = g.get('ruwe', '')
            row['gaia_excess_noise'] = g.get('astrometric_excess_noise', '')
            row['gaia_phot_variable_flag'] = g.get('phot_variable_flag', '')
            row['gaia_rv'] = g.get('radial_velocity', '')
            row['gaia_rv_error'] = g.get('radial_velocity_error', '')
            row['gaia_rv_nb_transits'] = g.get('rv_nb_transits', '')

            # Compute flags
            flags = compute_gaia_flags(g)
            row['gaia_astrometry_flag'] = flags['gaia_astrometry_flag']
            row['gaia_photvar_flag'] = flags['gaia_photvar_flag']

            # Update stats
            try:
                if float(g.get('ruwe', 0)) > 1.4:
                    stats['gaia_ruwe_high'] += 1
            except (ValueError, TypeError):
                pass

            if flags['gaia_photvar_flag'] == 'VAR':
                stats['gaia_photvar'] += 1

            if g.get('radial_velocity', '').strip():
                stats['gaia_has_rv'] += 1
        else:
            # No Gaia data
            for col in gaia_cols:
                row[col] = ''
            row['gaia_astrometry_flag'] = 'MISSING'
            row['gaia_photvar_flag'] = 'NOFLAG'

        # Add LAMOST columns
        lamost_info = lamost_results.get(t['targetid'], {})
        row['lamost_match'] = lamost_info.get('lamost_match', '')
        row['lamost_id'] = lamost_info.get('lamost_id', '')
        row['lamost_target'] = lamost_info.get('lamost_target', '')
        row['lamost_z'] = lamost_info.get('lamost_z', '')
        row['lamost_rv'] = lamost_info.get('lamost_rv', '')
        row['lamost_snr'] = lamost_info.get('lamost_snr', '')

        if lamost_info.get('lamost_match') == 'YES':
            stats['lamost_match'] += 1
        elif lamost_info.get('lamost_match') == 'QUERY_FAIL':
            stats['lamost_query_fail'] += 1

        output_rows.append(row)

    # Write output CSV
    print(f"Writing {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Write report
    print(f"Writing {OUTPUT_REPORT}...")
    report = f"""# Gaia DR3 + LAMOST Validation Report

## Summary

| Metric | Count |
|--------|-------|
| Total targets | {stats['total']} |
| Gaia queries succeeded | {stats['gaia_success']} |
| Gaia RUWE > 1.4 | {stats['gaia_ruwe_high']} |
| Gaia phot_variable_flag set | {stats['gaia_photvar']} |
| Gaia RVS radial_velocity available | {stats['gaia_has_rv']} |
| LAMOST match found | {stats['lamost_match']} |
| LAMOST query failures | {stats['lamost_query_fail']} |

---

## Gaia DR3 Annotation

### Fields Retrieved

| Field | Description |
|-------|-------------|
| ra, dec | Position (ICRS) |
| l, b | Galactic coordinates |
| phot_g_mean_mag | G-band magnitude |
| bp_rp | BP-RP color |
| parallax, parallax_error | Parallax (mas) |
| pmra, pmdec | Proper motion (mas/yr) |
| ruwe | Renormalised Unit Weight Error |
| astrometric_excess_noise | Excess astrometric noise (mas) |
| phot_variable_flag | Photometric variability flag |
| radial_velocity, radial_velocity_error | Gaia RVS RV (km/s) |
| rv_nb_transits | Number of RV transits |

### Derived Flags

- **gaia_astrometry_flag**:
  - `OK`: RUWE ≤ 1.4 AND excess_noise < 2.0 mas
  - `SUSPECT`: RUWE > 1.4 OR excess_noise ≥ 2.0 mas
  - `MISSING`: Data not available

- **gaia_photvar_flag**:
  - `VAR`: phot_variable_flag indicates variability
  - `NOFLAG`: No variability flag or NOT_AVAILABLE

---

## LAMOST Annotation

### Query Method

- Catalog: LAMOST DR5 (VizieR V/164/dr5)
- Search radius: 3 arcsec cone around Gaia position
- Selection: Best SNR match if multiple
- RV computed from redshift z as c*z (valid for stellar sources)

### Fields Retrieved

| Field | Description |
|-------|-------------|
| lamost_match | YES / NO / QUERY_FAIL |
| lamost_id | LAMOST observation ID |
| lamost_target | LAMOST target designation |
| lamost_z | Redshift from LAMOST |
| lamost_rv | RV computed from z (c*z, km/s) |
| lamost_snr | Signal-to-noise ratio (g-band) |

---

## Detailed Results

### Gaia Astrometry Flags

"""

    # Count by astrometry flag
    astro_counts = {}
    for row in output_rows:
        flag = row.get('gaia_astrometry_flag', 'MISSING')
        astro_counts[flag] = astro_counts.get(flag, 0) + 1

    report += "| Flag | Count |\n|------|-------|\n"
    for flag in ['OK', 'SUSPECT', 'MISSING']:
        report += f"| {flag} | {astro_counts.get(flag, 0)} |\n"

    report += "\n### Gaia Photometric Variability\n\n"
    photvar_counts = {}
    for row in output_rows:
        flag = row.get('gaia_photvar_flag', 'NOFLAG')
        photvar_counts[flag] = photvar_counts.get(flag, 0) + 1

    report += "| Flag | Count |\n|------|-------|\n"
    for flag in ['VAR', 'NOFLAG']:
        report += f"| {flag} | {photvar_counts.get(flag, 0)} |\n"

    report += "\n### LAMOST Matches\n\n"
    lamost_counts = {}
    for row in output_rows:
        match = row.get('lamost_match', 'NO')
        lamost_counts[match] = lamost_counts.get(match, 0) + 1

    report += "| Status | Count |\n|--------|-------|\n"
    for status in sorted(lamost_counts.keys()):
        report += f"| {status} | {lamost_counts[status]} |\n"

    # List targets with Gaia RV
    report += "\n### Targets with Gaia RVS Data\n\n"
    gaia_rv_targets = [r for r in output_rows if r.get('gaia_rv', '').strip()]
    if gaia_rv_targets:
        report += "| TARGETID | Gaia RV (km/s) | RV_err | N_transits |\n"
        report += "|----------|----------------|--------|------------|\n"
        for r in gaia_rv_targets:
            report += f"| {r['targetid']} | {r['gaia_rv']} | {r['gaia_rv_error']} | {r['gaia_rv_nb_transits']} |\n"
    else:
        report += "None of the targets have Gaia RVS radial velocity data.\n"

    # List LAMOST matches
    report += "\n### Targets with LAMOST Match\n\n"
    lamost_match_targets = [r for r in output_rows if r.get('lamost_match') == 'YES']
    if lamost_match_targets:
        report += "| TARGETID | LAMOST ID | LAMOST Target | z | RV (km/s) | SNR |\n"
        report += "|----------|-----------|---------------|---|-----------|-----|\n"
        for r in lamost_match_targets:
            report += f"| {r['targetid']} | {r['lamost_id']} | {r['lamost_target']} | {r['lamost_z']} | {r['lamost_rv']} | {r['lamost_snr']} |\n"
    else:
        report += "None of the targets have LAMOST matches within 3 arcsec.\n"

    report += f"""
---

## Caveats

1. **These are annotations only** — not confirmation of companion type or orbital parameters.
2. Gaia RUWE > 1.4 may indicate binarity or astrometric issues, but is not definitive.
3. Gaia phot_variable_flag may reflect intrinsic variability (pulsation) or eclipses.
4. LAMOST RVs are single-epoch unless explicitly noted; comparison to DESI requires care.
5. Lack of LAMOST match does not imply the target is unstudied — other surveys may have data.

---

## Query Errors

"""
    if gaia_error:
        report += f"- Gaia TAP query error: {gaia_error}\n"
    else:
        report += "- Gaia TAP: Success\n"

    if stats['lamost_query_fail'] > 0:
        report += f"- LAMOST VizieR: {stats['lamost_query_fail']} query failures\n"
    else:
        report += "- LAMOST VizieR: All queries succeeded\n"

    report += f"""
---

## Output Files

- `{OUTPUT_CSV}` — Annotated candidate list
- `{OUTPUT_REPORT}` — This report

---

*Generated by validate_gaia_lamost.py*
*Date: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Gaia queries succeeded: {stats['gaia_success']}/{stats['total']}")
    print(f"Gaia RUWE > 1.4: {stats['gaia_ruwe_high']}")
    print(f"Gaia phot_variable_flag: {stats['gaia_photvar']}")
    print(f"Gaia RVS available: {stats['gaia_has_rv']}")
    print(f"LAMOST matches: {stats['lamost_match']}")
    print()
    print(f"Output: {OUTPUT_CSV}")
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
