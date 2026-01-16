#!/usr/bin/env python3
"""
Additional verification checks for the dark companion candidate.
"""

import numpy as np
from astroquery.mast import Tesscut, Catalogs
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

RA = 164.523494
DEC = -1.660156
GAIA_ID = 3802130935635096832

print("=" * 70)
print("ADDITIONAL VERIFICATION CHECKS")
print("=" * 70)

# =============================================================================
# CHECK 1: Re-fit orbit with corrected LAMOST RV
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 1: ORBITAL FIT SENSITIVITY TO LAMOST RV")
print("=" * 70)

# Original epochs
epochs_original = [
    {"mjd": 57457.0, "rv": -49.36, "rv_err": 4.10},  # LAMOST with systematic floor
    {"mjd": 59568.488, "rv": -86.39, "rv_err": 0.55},  # DESI
    {"mjd": 59605.380, "rv": 59.68, "rv_err": 0.83},   # DESI
    {"mjd": 59607.374, "rv": 26.43, "rv_err": 1.06},   # DESI
    {"mjd": 59607.389, "rv": 25.16, "rv_err": 1.11},   # DESI
]

# Corrected epochs (LAMOST RV = -39.44 from M-dwarf catalog)
epochs_corrected = [
    {"mjd": 57457.0, "rv": -39.44, "rv_err": 4.10},  # Corrected LAMOST
    {"mjd": 59568.488, "rv": -86.39, "rv_err": 0.55},
    {"mjd": 59605.380, "rv": 59.68, "rv_err": 0.83},
    {"mjd": 59607.374, "rv": 26.43, "rv_err": 1.06},
    {"mjd": 59607.389, "rv": 25.16, "rv_err": 1.11},
]

def compute_rv_stats(epochs):
    rvs = np.array([e['rv'] for e in epochs])
    errs = np.array([e['rv_err'] for e in epochs])
    weights = 1/errs**2

    mean_rv = np.average(rvs, weights=weights)
    delta_rv = rvs.max() - rvs.min()

    # Chi-squared for constant model
    chi2 = np.sum(weights * (rvs - mean_rv)**2)

    return {
        'mean_rv': mean_rv,
        'delta_rv': delta_rv,
        'rv_min': rvs.min(),
        'rv_max': rvs.max(),
        'chi2': chi2
    }

print("\nOriginal (LAMOST RV = -49.36):")
stats_orig = compute_rv_stats(epochs_original)
print(f"  RV range: {stats_orig['rv_min']:.1f} to {stats_orig['rv_max']:.1f} km/s")
print(f"  Delta RV: {stats_orig['delta_rv']:.1f} km/s")
print(f"  Chi2(constant): {stats_orig['chi2']:.1f}")

print("\nCorrected (LAMOST RV = -39.44):")
stats_corr = compute_rv_stats(epochs_corrected)
print(f"  RV range: {stats_corr['rv_min']:.1f} to {stats_corr['rv_max']:.1f} km/s")
print(f"  Delta RV: {stats_corr['delta_rv']:.1f} km/s")
print(f"  Chi2(constant): {stats_corr['chi2']:.1f}")

print("\n  Impact: Delta RV changes by only ~10 km/s")
print("  The 4 DESI epochs alone span -86 to +60 = 146 km/s")
print("  --> LAMOST correction does NOT kill the variability signal")

# =============================================================================
# CHECK 2: TESS Observations
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 2: TESS PHOTOMETRY SEARCH")
print("=" * 70)

coord = SkyCoord(ra=RA, dec=DEC, unit=(u.deg, u.deg))

print(f"\nSearching TESS for observations at ({RA}, {DEC})...")

try:
    # Check what TESS sectors observed this target
    sector_table = Tesscut.get_sectors(coord)

    if len(sector_table) > 0:
        print(f"  Found {len(sector_table)} TESS sector(s)!")
        for row in sector_table:
            print(f"    Sector {row['sector']}: Camera {row['camera']}, CCD {row['ccd']}")

        print("\n  TESS data available - could check for:")
        print("    - Ellipsoidal variations (confirms close binary)")
        print("    - Eclipses (would constrain inclination)")
        print("    - Stellar activity")
    else:
        print("  No TESS observations found for this target")

except Exception as e:
    print(f"  TESS query error: {e}")

# =============================================================================
# CHECK 3: What would kill the candidate definitively?
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 3: DEFINITIVE TESTS NEEDED")
print("=" * 70)

print("""
To DEFINITIVELY KILL this candidate, we would need:

1. LAMOST SPECTRUM INSPECTION
   - Download actual spectrum from LAMOST
   - Check if cross-correlation function (CCF) shows double peaks
   - Verify RV measurement quality
   - URL: http://www.lamost.org/dr10/v2.0/spectrum/view?obsid=437513049

2. DESI SPECTRUM INSPECTION
   - Check DESI CCF for SB2 signatures
   - Look for asymmetric line profiles
   - Verify no fiber contamination from neighbor

3. TESS PHOTOMETRY ANALYSIS
   - Download light curve if available
   - Search for ellipsoidal variations at ~22 day period
   - Non-detection at expected amplitude would be concerning

4. HIGH-RESOLUTION FOLLOW-UP
   - New spectra at different orbital phases
   - Would immediately confirm/refute RV variability
   - Best telescope: Keck/HIRES, VLT/UVES, Gemini/GRACES

To DEFINITIVELY CONFIRM this candidate:

1. Additional RV epochs showing same ~22-day periodicity
2. Ellipsoidal variations in TESS matching orbital period
3. Consistent spectral type across all epochs
4. No SB2 signatures (ruling out luminous companion)
""")

# =============================================================================
# CHECK 4: LAMOST Web Access
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 4: LAMOST DIRECT ACCESS")
print("=" * 70)

print("""
LAMOST spectrum can be viewed/downloaded at:

ObsID 437513049 (from archival search):
  http://www.lamost.org/dr10/v2.0/spectrum/view?obsid=437513049

ObsID 579613097 (from our dossier):
  http://www.lamost.org/dr10/v2.0/spectrum/view?obsid=579613097

ObsID 870813030 (forum-suggested second epoch):
  http://www.lamost.org/dr10/v2.0/spectrum/view?obsid=870813030

You can also search by coordinates:
  http://www.lamost.org/dr10/v2.0/search?ra=164.523494&dec=-1.660156&radius=5

NOTE: Two different ObsIDs appear in our files (437513049 vs 579613097)
This needs investigation - possibly two different LAMOST observations!
""")

print("=" * 70)
print("CHECKS COMPLETE")
print("=" * 70)
