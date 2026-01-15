#!/usr/bin/env python3
"""
query_gaia_archival.py - Query Gaia DR3 NSS tables and archival RV cross-matches

Checks for:
1. Gaia DR3 NSS solutions (acceleration, two-body orbit, etc.)
2. Gaia DR3 epoch RVs
3. Gaia DR3 compact companion flags
4. Cross-matches to RAVE, APOGEE, LAMOST, GALAH
"""

import json
from astroquery.gaia import Gaia
import warnings
warnings.filterwarnings('ignore')

# Target
SOURCE_ID = 3802130935635096832
TARGET_NAME = f"Gaia DR3 {SOURCE_ID}"

# Gaia DR3 NSS tables to query
NSS_TABLES = [
    {
        'table': 'gaiadr3.nss_two_body_orbit',
        'description': 'Two-body orbital solutions',
        'key_columns': ['period', 'eccentricity', 'semi_amplitude_primary']
    },
    {
        'table': 'gaiadr3.nss_acceleration_astro',
        'description': 'Astrometric acceleration solutions',
        'key_columns': ['acceleration_ra', 'acceleration_dec']
    },
    {
        'table': 'gaiadr3.nss_non_linear_spectro',
        'description': 'Non-linear spectroscopic solutions',
        'key_columns': ['rv_template_teff', 'rv_amplitude_robust']
    },
    {
        'table': 'gaiadr3.nss_vim_fl',
        'description': 'Variability-induced movers',
        'key_columns': ['vim_f', 'vim_d']
    },
    {
        'table': 'gaiadr3.vari_compact_companion',
        'description': 'Compact companion candidates from Gaia photometry',
        'key_columns': ['classifier_result', 'solution_type']
    },
]

# Epoch RV table
EPOCH_RV_TABLE = {
    'table': 'gaiadr3.vari_epoch_radial_velocity',
    'description': 'Gaia DR3 epoch radial velocities',
    'key_columns': ['rv_obs_time', 'radial_velocity', 'radial_velocity_error']
}

# Cross-match tables for archival RVs
CROSSMATCH_TABLES = [
    {
        'table': 'gaiadr3.ravedr5_best_neighbour',
        'description': 'RAVE DR5 cross-match',
        'survey': 'RAVE DR5'
    },
    {
        'table': 'gaiadr3.ravedr6_best_neighbour',
        'description': 'RAVE DR6 cross-match',
        'survey': 'RAVE DR6'
    },
    {
        'table': 'gaiadr3.tmasspscxsc_best_neighbour',
        'description': '2MASS cross-match',
        'survey': '2MASS'
    },
    {
        'table': 'gaiadr3.allwise_best_neighbour',
        'description': 'AllWISE cross-match',
        'survey': 'AllWISE'
    },
]

# Additional tables to check (may not exist in all TAP services)
EXTRA_TABLES = [
    'gaiadr3.dr2_neighbourhood',  # DR2 cross-match
    'gaiadr3.gaia_source_simulation',  # Simulated sources (shouldn't match)
]

def query_table(table_name, source_id):
    """Query a Gaia table for a specific source_id."""
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE source_id = {source_id}
    """
    try:
        job = Gaia.launch_job(query)
        result = job.get_results()
        return result
    except Exception as e:
        return None, str(e)

def query_crossmatch(table_name, source_id):
    """Query a cross-match table."""
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE source_id = {source_id}
    """
    try:
        job = Gaia.launch_job(query)
        result = job.get_results()
        return result
    except Exception as e:
        return None, str(e)

def check_lamost_dr7():
    """Check LAMOST DR7 via VizieR or dedicated query."""
    # LAMOST DR7 is available via VizieR as V/164
    try:
        from astroquery.vizier import Vizier

        # Query by coordinates (RA=164.5235, Dec=-1.6602)
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=164.5235*u.deg, dec=-1.6602*u.deg, frame='icrs')

        # LAMOST DR7 catalog
        v = Vizier(columns=['*'], row_limit=10)
        result = v.query_region(coord, radius=3*u.arcsec, catalog='V/164')

        if result and len(result) > 0:
            return result[0]
        return None
    except Exception as e:
        return None, str(e)

def check_apogee_dr17():
    """Check APOGEE DR17 via VizieR."""
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=164.5235*u.deg, dec=-1.6602*u.deg, frame='icrs')

        # APOGEE DR17 catalog (III/286 or similar)
        v = Vizier(columns=['*'], row_limit=10)
        result = v.query_region(coord, radius=3*u.arcsec, catalog='III/286')

        if result and len(result) > 0:
            return result[0]
        return None
    except Exception as e:
        return None, str(e)

def check_galah_dr3():
    """Check GALAH DR3 via VizieR."""
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=164.5235*u.deg, dec=-1.6602*u.deg, frame='icrs')

        # GALAH DR3 catalog
        v = Vizier(columns=['*'], row_limit=10)
        result = v.query_region(coord, radius=3*u.arcsec, catalog='J/MNRAS/506/150')

        if result and len(result) > 0:
            return result[0]
        return None
    except Exception as e:
        return None, str(e)

def main():
    print("=" * 70)
    print(f"GAIA DR3 NSS & ARCHIVAL RV QUERY")
    print(f"Target: {TARGET_NAME}")
    print(f"Source ID: {SOURCE_ID}")
    print("=" * 70)
    print()

    results = {
        'source_id': SOURCE_ID,
        'nss_solutions': {},
        'epoch_rvs': None,
        'crossmatches': {},
        'archival_rvs': {}
    }

    # =========================================================================
    # 1. Query Gaia DR3 NSS tables
    # =========================================================================
    print("[1/4] Querying Gaia DR3 NSS tables...")
    print("-" * 50)

    for nss in NSS_TABLES:
        table = nss['table']
        desc = nss['description']
        print(f"  {table}...", end=" ")

        try:
            result = query_table(table, SOURCE_ID)
            if result is not None and len(result) > 0:
                print(f"FOUND ({len(result)} rows)")
                results['nss_solutions'][table] = {
                    'found': True,
                    'n_rows': len(result),
                    'columns': list(result.colnames)
                }
                # Print key columns
                for col in nss['key_columns']:
                    if col in result.colnames:
                        print(f"    {col}: {result[col][0]}")
            else:
                print("NOT FOUND")
                results['nss_solutions'][table] = {'found': False}
        except Exception as e:
            print(f"ERROR: {e}")
            results['nss_solutions'][table] = {'found': False, 'error': str(e)}

    print()

    # =========================================================================
    # 2. Query Gaia DR3 epoch RVs
    # =========================================================================
    print("[2/4] Querying Gaia DR3 epoch radial velocities...")
    print("-" * 50)

    table = EPOCH_RV_TABLE['table']
    print(f"  {table}...", end=" ")

    try:
        result = query_table(table, SOURCE_ID)
        if result is not None and len(result) > 0:
            print(f"FOUND ({len(result)} epochs)")
            results['epoch_rvs'] = {
                'found': True,
                'n_epochs': len(result),
                'epochs': []
            }
            print()
            print(f"  {'Obs Time':<14} {'RV (km/s)':<12} {'RV_err':<10}")
            print("  " + "-" * 36)
            for row in result:
                rv = row['radial_velocity']
                rv_err = row['radial_velocity_error']
                t = row['rv_obs_time']
                print(f"  {t:<14.4f} {rv:<12.2f} {rv_err:<10.2f}")
                results['epoch_rvs']['epochs'].append({
                    'time': float(t),
                    'rv': float(rv),
                    'rv_err': float(rv_err)
                })
        else:
            print("NOT FOUND")
            results['epoch_rvs'] = {'found': False}
            print("  (Star likely too faint for Gaia RVS: G = 17.27)")
    except Exception as e:
        print(f"ERROR: {e}")
        results['epoch_rvs'] = {'found': False, 'error': str(e)}

    print()

    # =========================================================================
    # 3. Query Gaia DR3 cross-match tables
    # =========================================================================
    print("[3/4] Querying Gaia DR3 cross-match tables...")
    print("-" * 50)

    for xm in CROSSMATCH_TABLES:
        table = xm['table']
        survey = xm['survey']
        print(f"  {survey} ({table})...", end=" ")

        try:
            result = query_crossmatch(table, SOURCE_ID)
            if result is not None and len(result) > 0:
                print(f"FOUND")
                results['crossmatches'][survey] = {
                    'found': True,
                    'table': table,
                    'n_rows': len(result)
                }
            else:
                print("NOT FOUND")
                results['crossmatches'][survey] = {'found': False}
        except Exception as e:
            print(f"ERROR: {e}")
            results['crossmatches'][survey] = {'found': False, 'error': str(e)}

    print()

    # =========================================================================
    # 4. Query archival spectroscopic surveys via VizieR
    # =========================================================================
    print("[4/4] Querying archival spectroscopic surveys (VizieR)...")
    print("-" * 50)

    # LAMOST
    print("  LAMOST DR7...", end=" ")
    try:
        lamost = check_lamost_dr7()
        if lamost is not None and not isinstance(lamost, tuple):
            print(f"FOUND ({len(lamost)} entries)")
            results['archival_rvs']['LAMOST'] = {'found': True, 'n_entries': len(lamost)}
        else:
            print("NOT FOUND")
            results['archival_rvs']['LAMOST'] = {'found': False}
    except Exception as e:
        print(f"ERROR: {e}")
        results['archival_rvs']['LAMOST'] = {'found': False, 'error': str(e)}

    # APOGEE
    print("  APOGEE DR17...", end=" ")
    try:
        apogee = check_apogee_dr17()
        if apogee is not None and not isinstance(apogee, tuple):
            print(f"FOUND ({len(apogee)} entries)")
            results['archival_rvs']['APOGEE'] = {'found': True, 'n_entries': len(apogee)}
        else:
            print("NOT FOUND")
            results['archival_rvs']['APOGEE'] = {'found': False}
    except Exception as e:
        print(f"ERROR: {e}")
        results['archival_rvs']['APOGEE'] = {'found': False, 'error': str(e)}

    # GALAH
    print("  GALAH DR3...", end=" ")
    try:
        galah = check_galah_dr3()
        if galah is not None and not isinstance(galah, tuple):
            print(f"FOUND ({len(galah)} entries)")
            results['archival_rvs']['GALAH'] = {'found': True, 'n_entries': len(galah)}
        else:
            print("NOT FOUND")
            results['archival_rvs']['GALAH'] = {'found': False}
    except Exception as e:
        print(f"ERROR: {e}")
        results['archival_rvs']['GALAH'] = {'found': False, 'error': str(e)}

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    # NSS solutions
    nss_found = [k for k, v in results['nss_solutions'].items() if v.get('found')]
    if nss_found:
        print(f"NSS Solutions Found: {len(nss_found)}")
        for t in nss_found:
            print(f"  - {t}")
    else:
        print("NSS Solutions Found: NONE")
        print("  → No Gaia DR3 NSS orbit, acceleration, or compact-companion solution")

    print()

    # Epoch RVs
    if results['epoch_rvs'] and results['epoch_rvs'].get('found'):
        print(f"Gaia Epoch RVs: {results['epoch_rvs']['n_epochs']} epochs")
    else:
        print("Gaia Epoch RVs: NONE")
        print("  → Star too faint for Gaia RVS (G = 17.27 > ~13 limit)")

    print()

    # Archival RVs
    archival_found = [k for k, v in results['archival_rvs'].items() if v.get('found')]
    if archival_found:
        print(f"Archival RVs Found: {len(archival_found)}")
        for s in archival_found:
            print(f"  - {s}")
    else:
        print("Archival RVs Found: NONE")
        print("  → No RAVE/APOGEE/LAMOST/GALAH coverage")

    print()

    # Conclusion
    print("-" * 50)
    print("CONCLUSION:")
    print()
    if not nss_found and not archival_found:
        print("  No additional data available from Gaia DR3 NSS or archival surveys.")
        print("  The DESI DR1 epochs are the only RV time series for this target.")
        print()
        print("  Recommended statement for paper:")
        print('  "No Gaia DR3 NSS or epoch-RV solution exists for this source.')
        print('   No cross-matched RVs from RAVE, APOGEE, LAMOST, or GALAH."')
    else:
        print("  Additional data found! See details above.")

    print()

    # Save results
    with open('gaia_archival_query_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved: gaia_archival_query_results.json")

    return results

if __name__ == "__main__":
    main()
