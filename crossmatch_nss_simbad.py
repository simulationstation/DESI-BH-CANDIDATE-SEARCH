#!/usr/bin/env python3
"""
crossmatch_nss_simbad.py - Cross-match Priority A candidates against Gaia DR3 NSS and SIMBAD

This script annotates RV-variable candidates with known binary/non-single status
from public catalogs. This is for validation and transparency, NOT discovery claims.

Author: Claude (Anthropic)
Date: 2026-01-13
"""

import csv
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path


def load_priority_candidates(csv_path):
    """Load Priority A candidates from CSV."""
    candidates = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle empty gaia_source_id
            gaia_id = row['gaia_source_id'].strip()
            candidates.append({
                'rank': int(row['rank']),
                'targetid': row['targetid'],
                'gaia_source_id': int(gaia_id) if gaia_id else None,
                'survey': row['survey'],
                'n_epochs': int(row['n_epochs']),
                'mjd_span': float(row['mjd_span']),
                'S': float(row['S']),
                'S_robust': float(row['S_robust']),
                'd_max': float(row['d_max']),
                'delta_rv_kms': float(row['delta_rv_kms']),
                'rv_err_median_kms': float(row['rv_err_median_kms']),
                'notes': row['notes']
            })
    return candidates


def query_gaia_tap(query, max_retries=3):
    """
    Query Gaia TAP service.

    Parameters
    ----------
    query : str
        ADQL query string
    max_retries : int
        Number of retry attempts

    Returns
    -------
    list of dict
        Query results as list of row dictionaries
    """
    tap_url = "https://gea.esac.esa.int/tap-server/tap/sync"

    params = {
        'REQUEST': 'doQuery',
        'LANG': 'ADQL',
        'FORMAT': 'csv',
        'QUERY': query
    }

    data = urllib.parse.urlencode(params).encode('utf-8')

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(tap_url, data=data)
            req.add_header('User-Agent', 'Python-urllib/DESI-RV-crossmatch')

            with urllib.request.urlopen(req, timeout=120) as response:
                content = response.read().decode('utf-8')

            # Parse CSV response
            lines = content.strip().split('\n')
            if len(lines) < 1:
                return []

            header = lines[0].split(',')
            results = []
            for line in lines[1:]:
                if line.strip():
                    values = line.split(',')
                    row = {header[i].strip(): values[i].strip() if i < len(values) else ''
                           for i in range(len(header))}
                    results.append(row)
            return results

        except Exception as e:
            print(f"  TAP query attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

    return []


def query_nss_tables(gaia_ids):
    """
    Query all Gaia DR3 NSS tables for given source IDs.

    Parameters
    ----------
    gaia_ids : list of int
        Gaia SOURCE_ID values to check

    Returns
    -------
    dict
        Mapping source_id -> list of NSS matches
    """
    # Filter out None values
    valid_ids = [gid for gid in gaia_ids if gid is not None]

    if not valid_ids:
        return {}

    # Format IDs for SQL IN clause
    id_list = ','.join(str(gid) for gid in valid_ids)

    nss_results = {gid: [] for gid in valid_ids}

    # Table 1: nss_two_body_orbit (main orbital solutions)
    print("  Querying gaiadr3.nss_two_body_orbit...")
    query = f"""
    SELECT source_id, nss_solution_type, period, period_error,
           eccentricity, semi_amplitude_primary, semi_amplitude_secondary
    FROM gaiadr3.nss_two_body_orbit
    WHERE source_id IN ({id_list})
    """
    try:
        results = query_gaia_tap(query)
        for row in results:
            sid = int(row['source_id'])
            if sid in nss_results:
                nss_results[sid].append({
                    'table': 'nss_two_body_orbit',
                    'solution_type': row.get('nss_solution_type', ''),
                    'period': row.get('period', ''),
                    'eccentricity': row.get('eccentricity', ''),
                    'K1': row.get('semi_amplitude_primary', ''),
                    'K2': row.get('semi_amplitude_secondary', '')
                })
        print(f"    Found {len(results)} matches")
    except Exception as e:
        print(f"    Error: {e}")

    time.sleep(1)  # Rate limiting

    # Table 2: nss_acceleration_astro (acceleration solutions)
    print("  Querying gaiadr3.nss_acceleration_astro...")
    query = f"""
    SELECT source_id, nss_solution_type
    FROM gaiadr3.nss_acceleration_astro
    WHERE source_id IN ({id_list})
    """
    try:
        results = query_gaia_tap(query)
        for row in results:
            sid = int(row['source_id'])
            if sid in nss_results:
                nss_results[sid].append({
                    'table': 'nss_acceleration_astro',
                    'solution_type': row.get('nss_solution_type', ''),
                })
        print(f"    Found {len(results)} matches")
    except Exception as e:
        print(f"    Error: {e}")

    time.sleep(1)

    # Table 3: nss_non_linear_spectro (non-linear spectroscopic)
    print("  Querying gaiadr3.nss_non_linear_spectro...")
    query = f"""
    SELECT source_id, nss_solution_type
    FROM gaiadr3.nss_non_linear_spectro
    WHERE source_id IN ({id_list})
    """
    try:
        results = query_gaia_tap(query)
        for row in results:
            sid = int(row['source_id'])
            if sid in nss_results:
                nss_results[sid].append({
                    'table': 'nss_non_linear_spectro',
                    'solution_type': row.get('nss_solution_type', ''),
                })
        print(f"    Found {len(results)} matches")
    except Exception as e:
        print(f"    Error: {e}")

    time.sleep(1)

    # Table 4: nss_vim_fl (variability-induced movers)
    print("  Querying gaiadr3.nss_vim_fl...")
    query = f"""
    SELECT source_id, nss_solution_type
    FROM gaiadr3.nss_vim_fl
    WHERE source_id IN ({id_list})
    """
    try:
        results = query_gaia_tap(query)
        for row in results:
            sid = int(row['source_id'])
            if sid in nss_results:
                nss_results[sid].append({
                    'table': 'nss_vim_fl',
                    'solution_type': row.get('nss_solution_type', ''),
                })
        print(f"    Found {len(results)} matches")
    except Exception as e:
        print(f"    Error: {e}")

    return nss_results


def query_simbad(gaia_ids):
    """
    Query SIMBAD for object types using Gaia DR3 identifiers.

    Uses SIMBAD TAP service.

    Parameters
    ----------
    gaia_ids : list of int
        Gaia SOURCE_ID values

    Returns
    -------
    dict
        Mapping source_id -> SIMBAD info
    """
    valid_ids = [gid for gid in gaia_ids if gid is not None]

    if not valid_ids:
        return {}

    simbad_results = {gid: None for gid in valid_ids}

    # SIMBAD TAP URL
    tap_url = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

    # Query in batches to avoid timeout
    batch_size = 20

    for batch_start in range(0, len(valid_ids), batch_size):
        batch = valid_ids[batch_start:batch_start + batch_size]

        # Build SIMBAD query using Gaia DR3 identifiers
        # SIMBAD uses 'Gaia DR3 XXXXXXXXX' format
        id_conditions = ' OR '.join([f"ident.id = 'Gaia DR3 {gid}'" for gid in batch])

        query = f"""
        SELECT DISTINCT ident.id AS gaia_id, basic.main_id, basic.otype, basic.otype_txt
        FROM ident
        JOIN basic ON ident.oidref = basic.oid
        WHERE ({id_conditions})
        """

        params = {
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'FORMAT': 'csv',
            'QUERY': query
        }

        try:
            data = urllib.parse.urlencode(params).encode('utf-8')
            req = urllib.request.Request(tap_url, data=data)
            req.add_header('User-Agent', 'Python-urllib/DESI-RV-crossmatch')

            with urllib.request.urlopen(req, timeout=60) as response:
                content = response.read().decode('utf-8')

            # Parse CSV response
            lines = content.strip().split('\n')
            if len(lines) > 1:
                header = [h.strip().strip('"') for h in lines[0].split(',')]
                for line in lines[1:]:
                    if line.strip():
                        # Handle quoted CSV fields
                        values = []
                        in_quote = False
                        current = ''
                        for char in line:
                            if char == '"':
                                in_quote = not in_quote
                            elif char == ',' and not in_quote:
                                values.append(current.strip().strip('"'))
                                current = ''
                            else:
                                current += char
                        values.append(current.strip().strip('"'))

                        if len(values) >= 4:
                            gaia_id_str = values[0]
                            # Extract numeric ID from 'Gaia DR3 XXXXX'
                            if 'Gaia DR3' in gaia_id_str:
                                try:
                                    gid = int(gaia_id_str.replace('Gaia DR3', '').strip())
                                    if gid in simbad_results:
                                        simbad_results[gid] = {
                                            'main_id': values[1] if len(values) > 1 else '',
                                            'otype': values[2] if len(values) > 2 else '',
                                            'otype_txt': values[3] if len(values) > 3 else ''
                                        }
                                except ValueError:
                                    pass

        except Exception as e:
            print(f"    SIMBAD batch query error: {e}")

        time.sleep(0.5)  # Rate limiting

    return simbad_results


def is_binary_otype(otype):
    """Check if SIMBAD object type indicates a binary or non-single system."""
    if not otype:
        return False

    otype = otype.upper()

    # Binary indicators in SIMBAD object types
    binary_types = [
        'SB', 'SB*',  # Spectroscopic binary
        'EB', 'EB*',  # Eclipsing binary
        'AL', 'AL*',  # Algol-type
        'BY', 'BY*',  # BY Dra variable (often binaries)
        'RS', 'RS*',  # RS CVn variable (binaries)
        'WD', 'WD*',  # White dwarf (could be WD+companion)
        'CV', 'CV*',  # Cataclysmic variable
        'XB', 'XRB',  # X-ray binary
        'LXB', 'HXB', # Low/High mass X-ray binary
        '**',         # Double/multiple star
        'PM*',        # Proper motion binary
        'SY*',        # Symbiotic star
    ]

    for bt in binary_types:
        if bt in otype:
            return True

    return False


def format_nss_type(matches):
    """Format NSS match information for output."""
    if not matches:
        return 'NO', '', ''

    tables = set(m['table'] for m in matches)
    types = set(m.get('solution_type', '') for m in matches if m.get('solution_type'))

    notes_parts = []
    for m in matches:
        tbl_short = m['table'].replace('nss_', '')
        sol_type = m.get('solution_type', '')
        period = m.get('period', '')

        if period and period not in ['', 'nan']:
            notes_parts.append(f"{tbl_short}:{sol_type}(P={float(period):.2f}d)")
        elif sol_type:
            notes_parts.append(f"{tbl_short}:{sol_type}")
        else:
            notes_parts.append(tbl_short)

    return 'YES', '; '.join(types) if types else 'unknown', '; '.join(notes_parts)


def write_annotated_csv(candidates, nss_results, simbad_results, output_path):
    """Write annotated Priority A CSV."""

    with open(output_path, 'w', newline='') as f:
        # Extended columns
        fieldnames = [
            'rank', 'targetid', 'gaia_source_id', 'survey', 'n_epochs',
            'mjd_span', 'S', 'S_robust', 'd_max', 'delta_rv_kms',
            'rv_err_median_kms', 'notes',
            'gaia_nss_flag', 'gaia_nss_type', 'gaia_nss_notes',
            'simbad_main_id', 'simbad_otype', 'simbad_binary_flag'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for c in candidates:
            gid = c['gaia_source_id']

            # NSS annotation
            nss_matches = nss_results.get(gid, []) if gid else []
            nss_flag, nss_type, nss_notes = format_nss_type(nss_matches)

            # SIMBAD annotation
            simbad_info = simbad_results.get(gid) if gid else None
            if simbad_info:
                simbad_main_id = simbad_info.get('main_id', '')
                simbad_otype = simbad_info.get('otype', '')
                simbad_binary = 'YES' if is_binary_otype(simbad_otype) else 'NO'
            else:
                simbad_main_id = ''
                simbad_otype = ''
                simbad_binary = 'NO_MATCH'

            row = {
                'rank': c['rank'],
                'targetid': c['targetid'],
                'gaia_source_id': c['gaia_source_id'] if c['gaia_source_id'] else '',
                'survey': c['survey'],
                'n_epochs': c['n_epochs'],
                'mjd_span': f"{c['mjd_span']:.4f}",
                'S': f"{c['S']:.4f}",
                'S_robust': f"{c['S_robust']:.4f}",
                'd_max': f"{c['d_max']:.4f}",
                'delta_rv_kms': f"{c['delta_rv_kms']:.4f}",
                'rv_err_median_kms': f"{c['rv_err_median_kms']:.4f}",
                'notes': c['notes'],
                'gaia_nss_flag': nss_flag,
                'gaia_nss_type': nss_type,
                'gaia_nss_notes': nss_notes,
                'simbad_main_id': simbad_main_id,
                'simbad_otype': simbad_otype,
                'simbad_binary_flag': simbad_binary
            }
            writer.writerow(row)

    print(f"  Wrote {output_path}")


def generate_report(candidates, nss_results, simbad_results, output_path):
    """Generate cross-check report."""

    total = len(candidates)

    # Count NSS matches
    nss_any = 0
    nss_spectro = 0
    nss_accel = 0
    nss_orbit = 0
    nss_none = 0

    for c in candidates:
        gid = c['gaia_source_id']
        if gid is None:
            nss_none += 1
            continue

        matches = nss_results.get(gid, [])
        if not matches:
            nss_none += 1
        else:
            nss_any += 1
            tables = set(m['table'] for m in matches)
            types = set(m.get('solution_type', '') for m in matches)

            if 'nss_two_body_orbit' in tables:
                nss_orbit += 1
                if any('SB' in str(t) for t in types):
                    nss_spectro += 1
            if 'nss_acceleration_astro' in tables:
                nss_accel += 1

    # Count SIMBAD matches
    simbad_any = 0
    simbad_binary = 0

    for c in candidates:
        gid = c['gaia_source_id']
        if gid is None:
            continue

        info = simbad_results.get(gid)
        if info:
            simbad_any += 1
            if is_binary_otype(info.get('otype', '')):
                simbad_binary += 1

    report = f"""# Gaia DR3 NSS and SIMBAD Cross-Check Report

## Summary

**Input**: {total} Priority A RV-variable candidates from DESI DR1 MWS
**Date**: 2026-01-13

---

## Part A: Gaia DR3 Non-Single Star (NSS) Cross-Match

### Results

| Category | Count | Percentage |
|----------|-------|------------|
| Total candidates | {total} | 100% |
| With Gaia NSS entry | {nss_any} | {100*nss_any/total:.1f}% |
| With orbital solution (nss_two_body_orbit) | {nss_orbit} | {100*nss_orbit/total:.1f}% |
| Flagged as spectroscopic binary (SB*) | {nss_spectro} | {100*nss_spectro/total:.1f}% |
| With acceleration solution only | {nss_accel - nss_orbit if nss_accel > nss_orbit else 0} | — |
| No NSS entry found | {nss_none} | {100*nss_none/total:.1f}% |

### NSS Tables Queried

1. `gaiadr3.nss_two_body_orbit` — astrometric and spectroscopic binary orbital solutions
2. `gaiadr3.nss_acceleration_astro` — proper motion acceleration solutions
3. `gaiadr3.nss_non_linear_spectro` — non-linear spectroscopic solutions
4. `gaiadr3.nss_vim_fl` — variability-induced movers

---

## Part B: SIMBAD Cross-Check

### Results

| Category | Count | Percentage |
|----------|-------|------------|
| With SIMBAD match | {simbad_any} | {100*simbad_any/total:.1f}% |
| SIMBAD binary classification | {simbad_binary} | {100*simbad_binary/total:.1f}% |
| No SIMBAD match | {total - simbad_any} | {100*(total-simbad_any)/total:.1f}% |

---

## Part C: Interpretation

### Key Points

1. **Gaia NSS is incomplete**: The Gaia DR3 NSS catalog is known to be incomplete,
   especially for:
   - Short-period binaries (P < few days)
   - Long-period binaries (P > few years)
   - Faint sources with low RV precision
   - Systems with unfavorable orbital inclinations

2. **Lack of NSS flag does NOT imply singleness**: A source not appearing in
   Gaia NSS may still be a binary. The DESI RV data provides independent
   evidence of variability.

3. **Overlap validates the pipeline**: The {nss_any} candidates ({100*nss_any/total:.1f}%)
   with Gaia NSS entries demonstrate that the RV-based selection is recovering
   known non-single systems, validating the methodology.

4. **New candidates are expected**: The {nss_none} candidates ({100*nss_none/total:.1f}%)
   without NSS entries are NOT claimed as "new discoveries" — they are simply
   RV-variable sources that warrant follow-up observation.

### Caveats

- This cross-match is for **annotation only**, not for filtering candidates.
- SIMBAD classifications may be incomplete or outdated.
- Object types in SIMBAD may reflect one aspect of a complex system.
- The absence of a binary classification does not prove single-star nature.

---

## Output Files

- `data/derived/priorityA_master_nss_annotated.csv` — Annotated candidate list
- `data/derived/NSS_SIMBAD_CROSSCHECK_REPORT.md` — This report

---

*Generated by crossmatch_nss_simbad.py*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"  Wrote {output_path}")
    print("")
    print(report)


def main():
    base_dir = Path('.')
    input_csv = base_dir / 'data/derived/priorityA_master.csv'
    output_csv = base_dir / 'data/derived/priorityA_master_nss_annotated.csv'
    report_path = base_dir / 'data/derived/NSS_SIMBAD_CROSSCHECK_REPORT.md'

    print("="*70)
    print("GAIA DR3 NSS AND SIMBAD CROSS-CHECK")
    print("="*70)

    # Load candidates
    print("\n1. Loading Priority A candidates...")
    candidates = load_priority_candidates(input_csv)
    print(f"   Loaded {len(candidates)} candidates")

    # Extract Gaia IDs
    gaia_ids = [c['gaia_source_id'] for c in candidates]
    valid_gaia_ids = [gid for gid in gaia_ids if gid is not None]
    print(f"   {len(valid_gaia_ids)} have valid Gaia SOURCE_ID")

    # Query Gaia NSS
    print("\n2. Querying Gaia DR3 NSS tables...")
    nss_results = query_nss_tables(gaia_ids)
    nss_matches = sum(1 for gid in valid_gaia_ids if nss_results.get(gid))
    print(f"   Found {nss_matches} candidates with NSS entries")

    # Query SIMBAD
    print("\n3. Querying SIMBAD...")
    simbad_results = query_simbad(gaia_ids)
    simbad_matches = sum(1 for gid in valid_gaia_ids if simbad_results.get(gid))
    print(f"   Found {simbad_matches} candidates with SIMBAD entries")

    # Write annotated CSV
    print("\n4. Writing annotated CSV...")
    write_annotated_csv(candidates, nss_results, simbad_results, output_csv)

    # Generate report
    print("\n5. Generating report...")
    generate_report(candidates, nss_results, simbad_results, report_path)

    print("\n" + "="*70)
    print("CROSS-CHECK COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
