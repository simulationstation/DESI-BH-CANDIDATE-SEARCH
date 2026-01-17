#!/usr/bin/env python3
"""
Gaia Sanity Check for BH Candidates
Queries Gaia DR3 for RUWE and non_single_star flags.

Usage:
    python gaia_sanity_check.py <ra> <dec>
    python gaia_sanity_check.py --targetid <desi_targetid>  # if coords in CSV
"""

import sys
import requests
from io import StringIO

def query_gaia(ra, dec, radius_arcsec=3.0):
    """Query Gaia DR3 for a source at given coordinates."""

    # Gaia TAP service
    tap_url = "https://gea.esac.esa.int/tap-server/tap/sync"

    query = f"""
    SELECT
        source_id,
        ra, dec,
        parallax, parallax_error,
        pmra, pmdec,
        phot_g_mean_mag,
        ruwe,
        non_single_star,
        astrometric_excess_noise,
        astrometric_excess_noise_sig
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600})
    ) = 1
    ORDER BY phot_g_mean_mag ASC
    LIMIT 5
    """

    params = {
        'REQUEST': 'doQuery',
        'LANG': 'ADQL',
        'FORMAT': 'csv',
        'QUERY': query
    }

    print(f"Querying Gaia DR3 at RA={ra:.6f}, Dec={dec:.6f}...")
    response = requests.get(tap_url, params=params, timeout=60)

    if response.status_code != 200:
        print(f"Query failed: HTTP {response.status_code}")
        return None

    import pandas as pd
    df = pd.read_csv(StringIO(response.text))

    if len(df) == 0:
        print("No Gaia sources found within search radius")
        return None

    return df


def analyze_gaia_result(df):
    """Analyze Gaia results for BH candidate validation."""

    print("\n" + "=" * 60)
    print("GAIA DR3 RESULTS")
    print("=" * 60)

    # Take brightest source (most likely match)
    source = df.iloc[0]

    print(f"\nSource ID: {source['source_id']}")
    print(f"RA, Dec: {source['ra']:.6f}, {source['dec']:.6f}")
    print(f"G mag: {source['phot_g_mean_mag']:.2f}")
    print(f"Parallax: {source['parallax']:.3f} ± {source['parallax_error']:.3f} mas")

    ruwe = source['ruwe']
    nss = source['non_single_star']
    aen = source['astrometric_excess_noise']
    aen_sig = source['astrometric_excess_noise_sig']

    print(f"\n--- BINARY INDICATORS ---")
    print(f"RUWE: {ruwe:.3f}")
    print(f"Non-single-star flag: {nss}")
    print(f"Astrometric excess noise: {aen:.3f}")
    print(f"AEN significance: {aen_sig:.1f}")

    print("\n--- INTERPRETATION ---")

    # RUWE check
    if ruwe < 1.4:
        print(f"⚠️  RUWE = {ruwe:.2f} < 1.4: NO astrometric wobble detected")
        print("    This is SUSPICIOUS for a massive companion.")
        print("    A 9 M☉ BH should cause detectable wobble.")
        ruwe_verdict = "FAIL"
    else:
        print(f"✓  RUWE = {ruwe:.2f} > 1.4: Astrometric wobble DETECTED")
        print("    Consistent with unseen massive companion.")
        ruwe_verdict = "PASS"

    # Non-single-star check
    if nss == 0:
        print(f"\n⚠️  non_single_star = {nss}: Gaia sees it as SINGLE")
        nss_verdict = "FAIL"
    else:
        print(f"\n✓  non_single_star = {nss}: Gaia flags it as BINARY")
        nss_verdict = "PASS"

    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    if ruwe_verdict == "FAIL" and nss_verdict == "FAIL":
        print("❌ LIKELY FALSE POSITIVE")
        print("   No evidence of massive companion in Gaia astrometry.")
        print("   Probably an SB2 or measurement artifact.")
    elif ruwe_verdict == "PASS" and nss_verdict == "PASS":
        print("✅ CONSISTENT WITH BH CANDIDATE")
        print("   Gaia sees astrometric wobble and flags as non-single.")
        print("   Worth further investigation!")
    else:
        print("⚡ INCONCLUSIVE")
        print("   Mixed signals - needs spectroscopic follow-up.")

    return {
        'source_id': source['source_id'],
        'ruwe': ruwe,
        'non_single_star': nss,
        'ruwe_verdict': ruwe_verdict,
        'nss_verdict': nss_verdict
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python gaia_sanity_check.py <ra> <dec>")
        print("       python gaia_sanity_check.py --targetid <desi_targetid>")
        sys.exit(1)

    if sys.argv[1] == '--targetid':
        # Look up coordinates from CSV
        import pandas as pd
        targetid = int(sys.argv[2])

        csv_path = '/home/primary/DESI-BH-CANDIDATE-SEARCH/e5-7-results/all_unique_candidates.csv'
        df = pd.read_csv(csv_path)

        match = df[df['targetid'] == targetid]
        if len(match) == 0:
            print(f"TargetID {targetid} not found in CSV")
            sys.exit(1)

        # Check if coordinates are valid
        ra = match.iloc[0].get('ra', 0)
        dec = match.iloc[0].get('dec', 0)

        if ra == 0 and dec == 0:
            print(f"ERROR: Coordinates are 0.0 for this target (metadata bug)")
            print("You need to manually look up the coordinates from DESI.")
            sys.exit(1)

    else:
        ra = float(sys.argv[1])
        dec = float(sys.argv[2])

    # Query Gaia
    result = query_gaia(ra, dec)

    if result is not None:
        analyze_gaia_result(result)


if __name__ == "__main__":
    main()
