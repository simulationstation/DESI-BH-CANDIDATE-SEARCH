#!/usr/bin/env python3
"""
Fetch and verify Gaia DR3 astrometric metrics for the BH candidate.

Queries RUWE, astrometric_excess_noise, and G magnitude for SOURCE_ID 3802130935635096832.
"""

from astroquery.gaia import Gaia
import warnings
warnings.filterwarnings('ignore')

# Target Gaia source ID
SOURCE_ID = 3802130935635096832

# RUWE threshold for binary detection (Lindegren+ 2021)
RUWE_BINARY_THRESHOLD = 1.4

def fetch_gaia_metrics(source_id):
    """
    Query Gaia DR3 for astrometric metrics.

    Returns dict with ruwe, astrometric_excess_noise, phot_g_mean_mag
    """
    query = f"""
    SELECT
        source_id,
        ruwe,
        astrometric_excess_noise,
        astrometric_excess_noise_sig,
        phot_g_mean_mag,
        parallax,
        parallax_error,
        pmra,
        pmdec
    FROM gaiadr3.gaia_source
    WHERE source_id = {source_id}
    """

    print(f"Querying Gaia DR3 for SOURCE_ID {source_id}...")

    job = Gaia.launch_job(query)
    results = job.get_results()

    if len(results) == 0:
        print(f"\033[91mERROR: Source {source_id} not found in Gaia DR3!\033[0m")
        return None

    row = results[0]
    return {
        'source_id': int(row['source_id']),
        'ruwe': float(row['ruwe']),
        'astrometric_excess_noise': float(row['astrometric_excess_noise']),
        'astrometric_excess_noise_sig': float(row['astrometric_excess_noise_sig']),
        'phot_g_mean_mag': float(row['phot_g_mean_mag']),
        'parallax': float(row['parallax']),
        'parallax_error': float(row['parallax_error']),
        'pmra': float(row['pmra']),
        'pmdec': float(row['pmdec']),
    }


def main():
    print("=" * 60)
    print("Gaia DR3 Astrometric Metrics Verification")
    print("=" * 60)

    metrics = fetch_gaia_metrics(SOURCE_ID)

    if metrics is None:
        return

    print(f"\nSource: Gaia DR3 {metrics['source_id']}")
    print("-" * 60)

    # RUWE
    ruwe = metrics['ruwe']
    print(f"\nRUWE (Renormalized Unit Weight Error):")
    print(f"  Value: {ruwe:.3f}")

    if ruwe > RUWE_BINARY_THRESHOLD:
        print(f"  \033[93mFLAG: RUWE > {RUWE_BINARY_THRESHOLD} (binary threshold)\033[0m")
        print(f"  Interpretation: Astrometric solution shows excess scatter,")
        print(f"                  consistent with unresolved orbital motion.")
    else:
        print(f"  Status: Below binary threshold ({RUWE_BINARY_THRESHOLD})")

    # Astrometric excess noise
    aen = metrics['astrometric_excess_noise']
    aen_sig = metrics['astrometric_excess_noise_sig']
    print(f"\nAstrometric Excess Noise:")
    print(f"  Value: {aen:.4f} mas")
    print(f"  Significance: {aen_sig:.2f} sigma")

    if aen_sig > 2:
        print(f"  \033[93mFLAG: Significant excess noise detected\033[0m")

    # Photometry
    print(f"\nPhotometry:")
    print(f"  G magnitude: {metrics['phot_g_mean_mag']:.3f}")

    # Astrometry
    print(f"\nAstrometry:")
    print(f"  Parallax: {metrics['parallax']:.4f} +/- {metrics['parallax_error']:.4f} mas")
    if metrics['parallax'] > 0:
        dist_pc = 1000 / metrics['parallax']
        print(f"  Distance: ~{dist_pc:.1f} pc")
    print(f"  Proper motion (RA):  {metrics['pmra']:.3f} mas/yr")
    print(f"  Proper motion (Dec): {metrics['pmdec']:.3f} mas/yr")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    flags = []
    if ruwe > RUWE_BINARY_THRESHOLD:
        flags.append(f"RUWE={ruwe:.2f} exceeds binary threshold")
    if aen_sig > 2:
        flags.append(f"Astrometric excess noise significant ({aen_sig:.1f}σ)")

    if flags:
        print("\nAstrometric flags raised:")
        for f in flags:
            print(f"  - {f}")
        print("\nThese metrics support the presence of an unseen companion.")
    else:
        print("\nNo astrometric flags raised.")
        print("Note: Short-period binaries may not show elevated RUWE.")


if __name__ == "__main__":
    main()
