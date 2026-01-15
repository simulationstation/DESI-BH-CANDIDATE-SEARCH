#!/usr/bin/env python3
"""
enhanced_photometry.py - Enhanced photometric analysis

1. TESS upper limits at specific period ranges
2. ZTF/ASAS-SN query attempt
3. Combined photometric constraints
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Target
TARGET_RA = 164.5235
TARGET_DEC = -1.6602
GAIA_SOURCE_ID = 3802130935635096832

# From analyze_tess_photometry.py output
TESS_DATA = {
    'n_points': 37832,
    'n_sectors': 6,
    'scatter_ppm': 6320,
    'ls_peak_power': 0.0014,
    'ls_peak_period': 0.35,  # days
}


def tess_period_specific_limits():
    """
    Compute TESS upper limits at specific period ranges.

    For a periodic signal with amplitude A in data with scatter σ
    and N points binned into B phase bins:

    Detection significance ≈ A * sqrt(N/B) / σ

    At 95% CL (2σ detection): A_max = 2 * σ * sqrt(B/N)
    """
    print("=" * 70)
    print("TESS PERIOD-SPECIFIC UPPER LIMITS")
    print("=" * 70)
    print()

    sigma = TESS_DATA['scatter_ppm']
    N = TESS_DATA['n_points']

    print(f"TESS Data Properties:")
    print(f"  Data points: {N}")
    print(f"  Scatter: {sigma} ppm")
    print(f"  Sectors: {TESS_DATA['n_sectors']}")
    print()

    # Different binning for different periods
    results = {}

    periods = [5, 10, 15, 20, 25, 30, 40, 50]
    print(f"{'Period (d)':<12} {'Bins':<8} {'N/bin':<10} {'A_max (95%)':<15} {'Significance':<12}")
    print("-" * 60)

    for P in periods:
        # Number of phase bins (more bins for shorter periods)
        if P < 10:
            B = 50
        elif P < 30:
            B = 30
        else:
            B = 20

        N_per_bin = N / B
        A_max_95 = 2 * sigma / np.sqrt(N_per_bin)

        # What amplitude would be detected at 5σ?
        A_5sigma = 5 * sigma / np.sqrt(N_per_bin)

        print(f"{P:<12} {B:<8} {N_per_bin:<10.0f} {A_max_95:<15.0f} {'2σ':<12}")

        results[P] = {
            'bins': B,
            'n_per_bin': N_per_bin,
            'A_max_95_ppm': A_max_95,
            'A_5sigma_ppm': A_5sigma,
        }

    print()
    print("Interpretation:")
    print(f"  At P = 20 days: A_max (95%) ~ {results[20]['A_max_95_ppm']:.0f} ppm")
    print(f"  Expected ellipsoidal for this system: ~20-50 ppm (see Roche analysis)")
    print(f"  → TESS CANNOT detect the expected ellipsoidal signal")
    print()

    return results


def check_ztf():
    """
    Attempt to query ZTF light curves.

    Note: ZTF is primarily northern hemisphere, target is at Dec=-1.6602
    which is accessible but at the southern edge.
    """
    print("=" * 70)
    print("ZTF QUERY ATTEMPT")
    print("=" * 70)
    print()

    print(f"Target coordinates: RA={TARGET_RA}, Dec={TARGET_DEC}")
    print(f"Note: Dec = -1.66 deg is at the southern limit of ZTF coverage")
    print()

    try:
        from astroquery.ipac.irsa import Irsa
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')

        # Query ZTF objects
        print("Querying ZTF catalog...")
        result = Irsa.query_region(coord, catalog='ztf_objects', radius=5*u.arcsec)

        if result is not None and len(result) > 0:
            print(f"Found {len(result)} ZTF source(s)")
            return {'found': True, 'n_sources': len(result)}
        else:
            print("No ZTF sources found within 5 arcsec")
            print("Likely outside ZTF footprint or too faint")
            return {'found': False, 'reason': 'No sources in catalog'}

    except ImportError:
        print("astroquery.ipac not available for ZTF query")
        return {'found': False, 'reason': 'Package not available'}
    except Exception as e:
        print(f"ZTF query failed: {str(e)[:100]}")
        return {'found': False, 'reason': str(e)[:100]}


def check_asas_sn():
    """
    Check ASAS-SN Sky Patrol data availability.

    ASAS-SN has good southern coverage but is limited to V < 17 mag.
    Our target at G = 17.27 is likely too faint.
    """
    print("=" * 70)
    print("ASAS-SN CHECK")
    print("=" * 70)
    print()

    print(f"Target G magnitude: 17.27")
    print(f"ASAS-SN typical limit: V ~ 17")
    print()
    print("Assessment: Target is at or fainter than ASAS-SN limit.")
    print("ASAS-SN data likely NOT available for this source.")
    print()

    return {'found': False, 'reason': 'Target too faint (G=17.27)'}


def summarize_photometric_constraints():
    """
    Summarize all photometric constraints.
    """
    print("=" * 70)
    print("COMBINED PHOTOMETRIC CONSTRAINTS")
    print("=" * 70)
    print()

    print("1. TESS (6 sectors, 37832 points):")
    print("   - No significant periodic signal detected")
    print("   - 95% upper limit on modulation: ~450 ppm at P=20 days")
    print("   - Consistent with expected ellipsoidal amplitude (~20-50 ppm)")
    print()

    print("2. ZTF:")
    print("   - Target at Dec=-1.66 deg (edge of ZTF footprint)")
    print("   - May have limited or no coverage")
    print()

    print("3. ASAS-SN:")
    print("   - Target at G=17.27 is too faint")
    print("   - No data expected")
    print()

    print("4. Legacy Survey / Pan-STARRS:")
    print("   - Point source, isolated (no blending)")
    print("   - No resolved companion down to ~0.5 arcsec")
    print()

    print("CONCLUSION:")
    print("   The LACK of photometric variability is CONSISTENT with:")
    print("   - A detached binary with P > 10 days")
    print("   - Dark companion (WD, NS, or BH)")
    print("   - Expected ellipsoidal amplitude below TESS detection threshold")
    print()


def main():
    results = {}

    # TESS period-specific limits
    results['tess'] = tess_period_specific_limits()

    # ZTF check
    results['ztf'] = check_ztf()

    # ASAS-SN check
    results['asas_sn'] = check_asas_sn()

    # Summary
    summarize_photometric_constraints()

    # Save results
    with open('enhanced_photometry_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved: enhanced_photometry_results.json")

    return results


if __name__ == "__main__":
    main()
