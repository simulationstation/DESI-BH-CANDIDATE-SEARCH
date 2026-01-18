#!/usr/bin/env python3
"""Query Gaia DR3 for source and neighbors."""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

SOURCE_ID = 3802130935635096832
OUTPUT_DIR = "/home/primary/DESI-BH-CANDIDATE-SEARCH/wobble_imaging_3802130935635096832"

def query_gaia_source():
    """Query Gaia DR3 for the target source."""
    print(f"[GAIA] Querying source_id = {SOURCE_ID}")

    query = f"""
    SELECT
        source_id, ra, dec, parallax, parallax_error,
        pmra, pmra_error, pmdec, pmdec_error,
        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
        ipd_gof_harmonic_amplitude, ipd_frac_multi_peak,
        non_single_star, rv_amplitude_robust, rv_nb_transits
    FROM gaiadr3.gaia_source
    WHERE source_id = {SOURCE_ID}
    """

    job = Gaia.launch_job(query)
    result = job.get_results()

    if len(result) == 0:
        print("[ERROR] Source not found in Gaia DR3!")
        return None

    row = result[0]
    source_data = {
        'source_id': int(row['source_id']),
        'ra': float(row['ra']),
        'dec': float(row['dec']),
        'parallax': float(row['parallax']) if row['parallax'] else None,
        'parallax_error': float(row['parallax_error']) if row['parallax_error'] else None,
        'pmra': float(row['pmra']) if row['pmra'] else None,
        'pmra_error': float(row['pmra_error']) if row['pmra_error'] else None,
        'pmdec': float(row['pmdec']) if row['pmdec'] else None,
        'pmdec_error': float(row['pmdec_error']) if row['pmdec_error'] else None,
        'phot_g_mean_mag': float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] else None,
        'phot_bp_mean_mag': float(row['phot_bp_mean_mag']) if row['phot_bp_mean_mag'] else None,
        'phot_rp_mean_mag': float(row['phot_rp_mean_mag']) if row['phot_rp_mean_mag'] else None,
        'ruwe': float(row['ruwe']) if row['ruwe'] else None,
        'astrometric_excess_noise': float(row['astrometric_excess_noise']) if row['astrometric_excess_noise'] else None,
        'astrometric_excess_noise_sig': float(row['astrometric_excess_noise_sig']) if row['astrometric_excess_noise_sig'] else None,
        'ipd_gof_harmonic_amplitude': float(row['ipd_gof_harmonic_amplitude']) if row['ipd_gof_harmonic_amplitude'] else None,
        'ipd_frac_multi_peak': float(row['ipd_frac_multi_peak']) if row['ipd_frac_multi_peak'] else None,
    }

    # Handle non_single_star which may be integer
    try:
        source_data['non_single_star'] = int(row['non_single_star']) if row['non_single_star'] is not None else None
    except:
        source_data['non_single_star'] = None

    # Handle RV fields
    try:
        source_data['rv_amplitude_robust'] = float(row['rv_amplitude_robust']) if row['rv_amplitude_robust'] else None
    except:
        source_data['rv_amplitude_robust'] = None
    try:
        source_data['rv_nb_transits'] = int(row['rv_nb_transits']) if row['rv_nb_transits'] else None
    except:
        source_data['rv_nb_transits'] = None

    print(f"[GAIA] Found: RA={source_data['ra']:.6f}, Dec={source_data['dec']:.6f}")
    print(f"[GAIA] G={source_data['phot_g_mean_mag']:.2f}, parallax={source_data['parallax']:.3f}±{source_data['parallax_error']:.3f} mas")
    print(f"[GAIA] pmra={source_data['pmra']:.3f}±{source_data['pmra_error']:.3f}, pmdec={source_data['pmdec']:.3f}±{source_data['pmdec_error']:.3f} mas/yr")
    print(f"[GAIA] RUWE={source_data['ruwe']:.3f}, astrometric_excess_noise_sig={source_data['astrometric_excess_noise_sig']:.1f}")

    return source_data


def query_neighbors(ra, dec, radius_arcsec=60):
    """Query Gaia DR3 for neighbors within radius."""
    print(f"[GAIA] Querying neighbors within {radius_arcsec} arcsec")

    query = f"""
    SELECT
        source_id, ra, dec, phot_g_mean_mag, pmra, pmdec, parallax
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600.0})
    )
    AND source_id != {SOURCE_ID}
    ORDER BY phot_g_mean_mag
    """

    job = Gaia.launch_job(query)
    result = job.get_results()

    neighbors = []
    for row in result:
        neighbors.append({
            'source_id': int(row['source_id']),
            'ra': float(row['ra']),
            'dec': float(row['dec']),
            'phot_g_mean_mag': float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] else None,
            'pmra': float(row['pmra']) if row['pmra'] else None,
            'pmdec': float(row['pmdec']) if row['pmdec'] else None,
            'parallax': float(row['parallax']) if row['parallax'] else None,
        })

    print(f"[GAIA] Found {len(neighbors)} neighbors")
    return neighbors


def main():
    # Query main source
    source = query_gaia_source()
    if source is None:
        return

    # Query neighbors
    neighbors = query_neighbors(source['ra'], source['dec'])

    # Save results
    output = {
        'target': source,
        'neighbors': neighbors
    }

    outfile = os.path.join(OUTPUT_DIR, 'data', 'gaia_query.json')
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[GAIA] Saved to {outfile}")

    # Also save just the coordinates for easy access
    coords_file = os.path.join(OUTPUT_DIR, 'data', 'target_coords.json')
    with open(coords_file, 'w') as f:
        json.dump({
            'source_id': source['source_id'],
            'ra': source['ra'],
            'dec': source['dec'],
            'phot_g_mean_mag': source['phot_g_mean_mag']
        }, f, indent=2)

    print(f"[GAIA] Target coords saved to {coords_file}")


if __name__ == "__main__":
    main()
