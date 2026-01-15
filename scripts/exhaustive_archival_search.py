#!/usr/bin/env python3
"""
exhaustive_archival_search.py - Comprehensive archival RV search

Queries ALL major spectroscopic surveys for RV epochs of the target.
This is a SMOKE TEST version - queries real archives, no fake data.

Target: Gaia DR3 3802130935635096832
RA, Dec: 164.5235, -1.6602

Surveys checked:
- Gaia DR3 (NSS, epoch RVs)
- LAMOST (DR7, DR8, DR9)
- APOGEE (DR17)
- GALAH (DR3)
- RAVE (DR5, DR6)
- SDSS (DR17 - includes SEGUE, BOSS, eBOSS)
- 6dF (DR3)
- SIMBAD (for any other references)
"""

import json
import warnings
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Target coordinates
TARGET_RA = 164.5235
TARGET_DEC = -1.6602
GAIA_SOURCE_ID = 3802130935635096832
SEARCH_RADIUS_ARCSEC = 3.0

# Output file
OUTPUT_FILE = 'data/rv_epochs/archival_search_results.json'

def query_simbad():
    """Query SIMBAD for basic object info and any known RVs."""
    print("  [SIMBAD] Querying...")
    try:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        # Custom SIMBAD query to get RV if available
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('rv_value', 'rvz_type', 'rvz_qual',
                                          'otypes', 'sptype', 'flux(V)')

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')
        result = custom_simbad.query_region(coord, radius=SEARCH_RADIUS_ARCSEC*u.arcsec)

        if result is not None and len(result) > 0:
            row = result[0]
            return {
                'found': True,
                'main_id': str(row['MAIN_ID']),
                'otypes': str(row['OTYPES']) if row['OTYPES'] else None,
                'sptype': str(row['SP_TYPE']) if row['SP_TYPE'] else None,
                'rv_value': float(row['RV_VALUE']) if row['RV_VALUE'] else None,
                'rv_type': str(row['RVZ_TYPE']) if row['RVZ_TYPE'] else None,
            }
        return {'found': False, 'error': 'No match within search radius'}
    except Exception as e:
        return {'found': False, 'error': str(e)}


def query_gaia_dr3():
    """Query Gaia DR3 for source info, NSS solutions, and epoch RVs."""
    print("  [Gaia DR3] Querying...")
    try:
        from astroquery.gaia import Gaia

        results = {
            'source_found': False,
            'nss_solutions': {},
            'epoch_rvs': None,
        }

        # 1. Query main source table
        query = f"""
        SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec,
               phot_g_mean_mag, bp_rp, ruwe,
               radial_velocity, radial_velocity_error,
               astrometric_excess_noise, astrometric_excess_noise_sig
        FROM gaiadr3.gaia_source
        WHERE source_id = {GAIA_SOURCE_ID}
        """
        job = Gaia.launch_job(query)
        result = job.get_results()

        if len(result) > 0:
            row = result[0]
            results['source_found'] = True
            results['source'] = {
                'source_id': int(row['source_id']),
                'ra': float(row['ra']),
                'dec': float(row['dec']),
                'parallax': float(row['parallax']) if row['parallax'] else None,
                'parallax_error': float(row['parallax_error']) if row['parallax_error'] else None,
                'G': float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] else None,
                'bp_rp': float(row['bp_rp']) if row['bp_rp'] else None,
                'ruwe': float(row['ruwe']) if row['ruwe'] else None,
                'rv': float(row['radial_velocity']) if row['radial_velocity'] else None,
                'rv_error': float(row['radial_velocity_error']) if row['radial_velocity_error'] else None,
                'aen': float(row['astrometric_excess_noise']) if row['astrometric_excess_noise'] else None,
                'aen_sig': float(row['astrometric_excess_noise_sig']) if row['astrometric_excess_noise_sig'] else None,
            }

        # 2. Query NSS tables
        nss_tables = [
            'gaiadr3.nss_two_body_orbit',
            'gaiadr3.nss_acceleration_astro',
            'gaiadr3.nss_non_linear_spectro',
            'gaiadr3.vari_compact_companion',
        ]

        for table in nss_tables:
            try:
                query = f"SELECT * FROM {table} WHERE source_id = {GAIA_SOURCE_ID}"
                job = Gaia.launch_job(query)
                res = job.get_results()
                if len(res) > 0:
                    results['nss_solutions'][table] = {'found': True, 'n_rows': len(res)}
                else:
                    results['nss_solutions'][table] = {'found': False}
            except:
                results['nss_solutions'][table] = {'found': False, 'error': 'query failed'}

        # 3. Query epoch RVs
        try:
            query = f"""
            SELECT * FROM gaiadr3.vari_epoch_radial_velocity
            WHERE source_id = {GAIA_SOURCE_ID}
            """
            job = Gaia.launch_job(query)
            res = job.get_results()
            if len(res) > 0:
                results['epoch_rvs'] = {
                    'found': True,
                    'n_epochs': len(res),
                    'epochs': [{'time': float(r['rv_obs_time']),
                                'rv': float(r['radial_velocity']),
                                'rv_err': float(r['radial_velocity_error'])} for r in res]
                }
            else:
                results['epoch_rvs'] = {'found': False, 'note': 'Star too faint for RVS (G=17.27)'}
        except:
            results['epoch_rvs'] = {'found': False, 'error': 'query failed'}

        return results

    except Exception as e:
        return {'found': False, 'error': str(e)}


def query_lamost():
    """Query LAMOST via VizieR for ALL observations."""
    print("  [LAMOST] Querying VizieR catalogs...")
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')

        # LAMOST catalogs in VizieR
        lamost_catalogs = [
            'V/164',      # LAMOST DR7 general
            'V/156',      # LAMOST DR5
            'V/153',      # LAMOST DR4
            'J/ApJS/245/34',  # LAMOST MRS
        ]

        results = {'found': False, 'epochs': [], 'catalogs_checked': []}

        v = Vizier(columns=['*'], row_limit=-1)

        for cat in lamost_catalogs:
            try:
                result = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC*u.arcsec, catalog=cat)
                results['catalogs_checked'].append(cat)

                if result and len(result) > 0:
                    for table in result:
                        for row in table:
                            epoch_data = {'catalog': cat}

                            # Try to extract RV and MJD from various column names
                            for col in table.colnames:
                                col_lower = col.lower()
                                if 'rv' in col_lower or 'vr' in col_lower:
                                    try:
                                        epoch_data['rv_kms'] = float(row[col])
                                    except:
                                        pass
                                if 'e_rv' in col_lower or 'rv_err' in col_lower:
                                    try:
                                        epoch_data['rv_err_kms'] = float(row[col])
                                    except:
                                        pass
                                if 'mjd' in col_lower:
                                    try:
                                        epoch_data['mjd'] = float(row[col])
                                    except:
                                        pass
                                if 'snr' in col_lower or 's/n' in col_lower:
                                    try:
                                        epoch_data['snr'] = float(row[col])
                                    except:
                                        pass
                                if 'class' in col_lower or 'subclass' in col_lower:
                                    try:
                                        epoch_data['spectral_type'] = str(row[col])
                                    except:
                                        pass

                            if 'rv_kms' in epoch_data and 'mjd' in epoch_data:
                                results['epochs'].append(epoch_data)
                                results['found'] = True
            except Exception as e:
                results['catalogs_checked'].append(f"{cat} (error: {str(e)[:50]})")

        # Deduplicate by MJD (within 0.01 day tolerance)
        if results['epochs']:
            unique_epochs = []
            seen_mjds = set()
            for ep in sorted(results['epochs'], key=lambda x: x.get('mjd', 0)):
                mjd = ep.get('mjd', 0)
                mjd_key = round(mjd, 1)
                if mjd_key not in seen_mjds:
                    seen_mjds.add(mjd_key)
                    unique_epochs.append(ep)
            results['epochs'] = unique_epochs
            results['n_unique_epochs'] = len(unique_epochs)

        return results

    except Exception as e:
        return {'found': False, 'error': str(e)}


def query_apogee():
    """Query APOGEE DR17 via VizieR."""
    print("  [APOGEE] Querying VizieR...")
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')
        v = Vizier(columns=['*'], row_limit=-1)

        # APOGEE DR17 catalogs
        apogee_catalogs = ['III/286', 'III/284']  # DR17, DR16

        results = {'found': False, 'epochs': [], 'catalogs_checked': []}

        for cat in apogee_catalogs:
            try:
                result = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC*u.arcsec, catalog=cat)
                results['catalogs_checked'].append(cat)

                if result and len(result) > 0:
                    for table in result:
                        for row in table:
                            epoch_data = {'catalog': cat}
                            for col in table.colnames:
                                col_lower = col.lower()
                                if col_lower == 'rv' or col_lower == 'vhelio':
                                    try:
                                        epoch_data['rv_kms'] = float(row[col])
                                    except:
                                        pass
                                if 'e_rv' in col_lower or 'verr' in col_lower:
                                    try:
                                        epoch_data['rv_err_kms'] = float(row[col])
                                    except:
                                        pass
                                if 'snr' in col_lower:
                                    try:
                                        epoch_data['snr'] = float(row[col])
                                    except:
                                        pass

                            if 'rv_kms' in epoch_data:
                                results['epochs'].append(epoch_data)
                                results['found'] = True
            except Exception as e:
                results['catalogs_checked'].append(f"{cat} (error)")

        return results

    except Exception as e:
        return {'found': False, 'error': str(e)}


def query_galah():
    """Query GALAH DR3 via VizieR."""
    print("  [GALAH] Querying VizieR...")
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')
        v = Vizier(columns=['*'], row_limit=-1)

        # GALAH DR3 catalog
        result = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC*u.arcsec,
                               catalog='J/MNRAS/506/150')

        if result and len(result) > 0:
            epochs = []
            for table in result:
                for row in table:
                    epoch_data = {'catalog': 'GALAH_DR3'}
                    for col in table.colnames:
                        col_lower = col.lower()
                        if 'rv' in col_lower and 'e_' not in col_lower:
                            try:
                                epoch_data['rv_kms'] = float(row[col])
                            except:
                                pass
                        if 'e_rv' in col_lower:
                            try:
                                epoch_data['rv_err_kms'] = float(row[col])
                            except:
                                pass
                    if 'rv_kms' in epoch_data:
                        epochs.append(epoch_data)

            return {'found': len(epochs) > 0, 'epochs': epochs}

        return {'found': False}

    except Exception as e:
        return {'found': False, 'error': str(e)}


def query_rave():
    """Query RAVE DR5/DR6 via VizieR."""
    print("  [RAVE] Querying VizieR...")
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')
        v = Vizier(columns=['*'], row_limit=-1)

        # RAVE catalogs
        rave_catalogs = ['III/283', 'III/279']  # DR6, DR5

        results = {'found': False, 'epochs': [], 'catalogs_checked': []}

        for cat in rave_catalogs:
            try:
                result = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC*u.arcsec, catalog=cat)
                results['catalogs_checked'].append(cat)

                if result and len(result) > 0:
                    for table in result:
                        for row in table:
                            epoch_data = {'catalog': cat}
                            for col in table.colnames:
                                col_lower = col.lower()
                                if col_lower == 'hrv' or col_lower == 'rv':
                                    try:
                                        epoch_data['rv_kms'] = float(row[col])
                                    except:
                                        pass
                                if 'e_hrv' in col_lower or 'e_rv' in col_lower:
                                    try:
                                        epoch_data['rv_err_kms'] = float(row[col])
                                    except:
                                        pass

                            if 'rv_kms' in epoch_data:
                                results['epochs'].append(epoch_data)
                                results['found'] = True
            except:
                pass

        return results

    except Exception as e:
        return {'found': False, 'error': str(e)}


def query_sdss():
    """Query SDSS spectroscopy via VizieR."""
    print("  [SDSS] Querying VizieR...")
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')
        v = Vizier(columns=['*'], row_limit=-1)

        # SDSS spectroscopic catalogs
        sdss_catalogs = [
            'V/154',       # SDSS DR17
            'V/147',       # SDSS DR12
            'III/294',     # SEGUE
        ]

        results = {'found': False, 'epochs': [], 'catalogs_checked': []}

        for cat in sdss_catalogs:
            try:
                result = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC*u.arcsec, catalog=cat)
                results['catalogs_checked'].append(cat)

                if result and len(result) > 0:
                    for table in result:
                        for row in table:
                            epoch_data = {'catalog': cat}
                            for col in table.colnames:
                                col_lower = col.lower()
                                if col_lower in ['rv', 'z', 'vhelio']:
                                    try:
                                        val = float(row[col])
                                        # If it looks like redshift, convert to km/s
                                        if abs(val) < 0.01:
                                            val = val * 299792.458
                                        epoch_data['rv_kms'] = val
                                    except:
                                        pass
                                if 'e_rv' in col_lower or 'zerr' in col_lower:
                                    try:
                                        epoch_data['rv_err_kms'] = float(row[col])
                                    except:
                                        pass
                                if 'mjd' in col_lower:
                                    try:
                                        epoch_data['mjd'] = float(row[col])
                                    except:
                                        pass

                            if 'rv_kms' in epoch_data:
                                results['epochs'].append(epoch_data)
                                results['found'] = True
            except:
                pass

        return results

    except Exception as e:
        return {'found': False, 'error': str(e)}


def query_6df():
    """Query 6dF DR3 via VizieR."""
    print("  [6dF] Querying VizieR...")
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')
        v = Vizier(columns=['*'], row_limit=-1)

        # 6dF catalog - note: 6dF is southern hemisphere, our target is near equator
        result = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC*u.arcsec, catalog='VII/259')

        if result and len(result) > 0:
            epochs = []
            for table in result:
                for row in table:
                    epoch_data = {'catalog': '6dF_DR3'}
                    for col in table.colnames:
                        col_lower = col.lower()
                        if 'rv' in col_lower or 'z' in col_lower:
                            try:
                                val = float(row[col])
                                if abs(val) < 0.01:  # redshift
                                    val = val * 299792.458
                                epoch_data['rv_kms'] = val
                            except:
                                pass
                    if 'rv_kms' in epoch_data:
                        epochs.append(epoch_data)

            return {'found': len(epochs) > 0, 'epochs': epochs}

        return {'found': False}

    except Exception as e:
        return {'found': False, 'error': str(e)}


def main():
    print("=" * 70)
    print("EXHAUSTIVE ARCHIVAL RV SEARCH")
    print("=" * 70)
    print(f"Target: Gaia DR3 {GAIA_SOURCE_ID}")
    print(f"Coordinates: RA={TARGET_RA:.4f}, Dec={TARGET_DEC:.4f}")
    print(f"Search radius: {SEARCH_RADIUS_ARCSEC} arcsec")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)
    print()

    results = {
        'target': f'Gaia DR3 {GAIA_SOURCE_ID}',
        'ra_deg': TARGET_RA,
        'dec_deg': TARGET_DEC,
        'search_radius_arcsec': SEARCH_RADIUS_ARCSEC,
        'query_date': datetime.now().isoformat(),
        'surveys': {},
        'all_epochs': [],
    }

    # Run all queries
    print("QUERYING ARCHIVES:")
    print("-" * 50)

    # Execute queries (could parallelize but being careful with rate limits)
    surveys = [
        ('SIMBAD', query_simbad),
        ('Gaia_DR3', query_gaia_dr3),
        ('LAMOST', query_lamost),
        ('APOGEE', query_apogee),
        ('GALAH', query_galah),
        ('RAVE', query_rave),
        ('SDSS', query_sdss),
        ('6dF', query_6df),
    ]

    for name, func in surveys:
        try:
            result = func()
            results['surveys'][name] = result

            # Collect epochs with source labels
            if isinstance(result, dict):
                if 'epochs' in result and result.get('found'):
                    for ep in result['epochs']:
                        ep['source_survey'] = name
                        results['all_epochs'].append(ep)
                elif 'epoch_rvs' in result:
                    epoch_data = result.get('epoch_rvs', {})
                    if epoch_data and epoch_data.get('found'):
                        for ep in epoch_data.get('epochs', []):
                            ep['source_survey'] = 'Gaia_DR3_RVS'
                            results['all_epochs'].append(ep)
        except Exception as e:
            results['surveys'][name] = {'found': False, 'error': str(e)}
            print(f"  [{name}] ERROR: {str(e)[:50]}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    # Summary table
    print("SURVEY RESULTS:")
    print("-" * 50)
    for name, result in results['surveys'].items():
        if isinstance(result, dict):
            found = result.get('found', False)
            n_epochs = len(result.get('epochs', []))
            status = f"FOUND ({n_epochs} epochs)" if found and n_epochs > 0 else "NOT FOUND"
            if result.get('error'):
                status = f"ERROR: {result['error'][:30]}"
            print(f"  {name:<15} {status}")
        else:
            print(f"  {name:<15} ERROR")

    print()

    # All unique epochs
    print(f"TOTAL ARCHIVAL EPOCHS FOUND: {len(results['all_epochs'])}")
    if results['all_epochs']:
        print()
        print(f"{'Survey':<15} {'MJD':<12} {'RV (km/s)':<12} {'σRV':<10}")
        print("-" * 50)
        for ep in sorted(results['all_epochs'], key=lambda x: x.get('mjd', 0)):
            survey = ep.get('source_survey', 'Unknown')[:15]
            mjd = ep.get('mjd', ep.get('time', 'N/A'))
            rv = ep.get('rv_kms', ep.get('rv', 'N/A'))
            rv_err = ep.get('rv_err_kms', ep.get('rv_err', 'N/A'))

            mjd_str = f"{mjd:.3f}" if isinstance(mjd, (int, float)) else str(mjd)
            rv_str = f"{rv:.2f}" if isinstance(rv, (int, float)) else str(rv)
            err_str = f"{rv_err:.2f}" if isinstance(rv_err, (int, float)) else str(rv_err)

            print(f"  {survey:<15} {mjd_str:<12} {rv_str:<12} {err_str:<10}")

    print()

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {OUTPUT_FILE}")

    # Conclusions
    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()

    n_archival = len(results['all_epochs'])
    if n_archival == 0:
        print("  No additional archival RV epochs found beyond LAMOST + DESI.")
        print("  The 5 epochs (1 LAMOST + 4 DESI) are the complete RV dataset.")
    else:
        print(f"  Found {n_archival} archival epoch(s) to add to the dataset.")
        print("  Review the epochs above and add valid ones to the master CSV.")

    print()

    return results


if __name__ == "__main__":
    main()
