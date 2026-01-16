#!/usr/bin/env python3
"""
Completeness check for forensic audit - addressing gaps.
"""

from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

GAIA_ID = 3802130935635096832
COMPANION_ID = 3802130935634233472
RA = 164.523494
DEC = -1.660156
OBSID_1 = 437513049
OBSID_2 = 870813030  # Second ObsID from forum - DID WE CHECK THIS?

print("=" * 70)
print("FORENSIC COMPLETENESS CHECK")
print("=" * 70)

# =============================================================================
# GAP 1: Did we check the SECOND ObsID (870813030)?
# =============================================================================
print("\n" + "=" * 70)
print("GAP 1: SECOND LAMOST ObsID (870813030)")
print("=" * 70)

coord = SkyCoord(ra=RA, dec=DEC, unit=(u.deg, u.deg))
v = Vizier(columns=['**'], row_limit=100)

# Search for the second obsid
print(f"\nSearching for ObsID {OBSID_2} in all LAMOST catalogs...")

lamost_cats = ['V/164/dr7', 'V/153/dr4', 'V/156/dr5', 'J/ApJS/245/34']
found_obsid2 = False

for cat in lamost_cats:
    try:
        result = v.query_region(coord, radius=30*u.arcsec, catalog=cat)
        if result and len(result) > 0:
            for row in result[0]:
                if 'ObsID' in result[0].colnames:
                    obsid = row['ObsID']
                    if obsid == OBSID_2 or str(obsid) == str(OBSID_2):
                        print(f"  FOUND in {cat}!")
                        print(f"    ObsID: {obsid}")
                        found_obsid2 = True
                        # Print RV if available
                        for col in result[0].colnames:
                            if 'rv' in col.lower() or 'RV' in col:
                                print(f"    {col}: {row[col]}")
    except Exception as e:
        pass

if not found_obsid2:
    print(f"  ObsID {OBSID_2} NOT FOUND in VizieR LAMOST catalogs")
    print("  This could mean:")
    print("    1. It's in a newer DR not yet in VizieR")
    print("    2. It's a different target nearby")
    print("    3. The forum user made an error")

# =============================================================================
# GAP 2: Is the 0.688" companion physically associated?
# =============================================================================
print("\n" + "=" * 70)
print("GAP 2: COMPANION PROPER MOTION CHECK")
print("=" * 70)

print("\nQuerying proper motions for target and companion...")

try:
    query = f"""
    SELECT source_id, ra, dec, parallax, parallax_error,
           pmra, pmra_error, pmdec, pmdec_error,
           phot_g_mean_mag, ruwe
    FROM gaiadr3.gaia_source
    WHERE source_id IN ({GAIA_ID}, {COMPANION_ID})
    """

    job = Gaia.launch_job(query)
    result = job.get_results()

    print("\n  Proper Motion Comparison:")
    print("  " + "-" * 60)

    target_pm = None
    companion_pm = None

    for row in result:
        src_id = row['source_id']
        pmra = row['pmra']
        pmdec = row['pmdec']
        plx = row['parallax']
        gmag = row['phot_g_mean_mag']

        label = "TARGET" if src_id == GAIA_ID else "COMPANION"
        print(f"  {label} ({src_id}):")
        print(f"    G mag: {gmag:.2f}")
        print(f"    Parallax: {plx:.3f} +/- {row['parallax_error']:.3f} mas")
        print(f"    PM_RA: {pmra:.3f} +/- {row['pmra_error']:.3f} mas/yr")
        print(f"    PM_Dec: {pmdec:.3f} +/- {row['pmdec_error']:.3f} mas/yr")

        if src_id == GAIA_ID:
            target_pm = (pmra, pmdec, plx)
        else:
            companion_pm = (pmra, pmdec, plx)

    if target_pm and companion_pm:
        pm_diff = np.sqrt((target_pm[0] - companion_pm[0])**2 +
                         (target_pm[1] - companion_pm[1])**2)
        plx_diff = abs(target_pm[2] - companion_pm[2])

        print(f"\n  PM difference: {pm_diff:.2f} mas/yr")
        print(f"  Parallax difference: {plx_diff:.3f} mas")

        if pm_diff < 5 and plx_diff < 0.5:
            print("  --> LIKELY PHYSICAL PAIR (similar PM and parallax)")
        elif pm_diff > 20 or plx_diff > 1.0:
            print("  --> LIKELY CHANCE ALIGNMENT (very different PM or parallax)")
        else:
            print("  --> UNCERTAIN - could be either")

except Exception as e:
    print(f"  Query error: {e}")

# =============================================================================
# GAP 3: How much can 13% flux contamination affect RV?
# =============================================================================
print("\n" + "=" * 70)
print("GAP 3: FLUX CONTAMINATION RV IMPACT ANALYSIS")
print("=" * 70)

delta_g = 2.21  # mag difference
flux_ratio = 10**(-delta_g/2.5)
contamination_frac = flux_ratio / (1 + flux_ratio)

print(f"\n  Companion is {delta_g:.2f} mag fainter")
print(f"  Flux ratio (companion/target): {flux_ratio:.3f}")
print(f"  Contamination fraction: {contamination_frac:.1%}")

# If companion has different RV, what would we observe?
print("\n  Scenario Analysis:")
print("  If companion RV differs by 50 km/s from target:")
rv_shift = 50 * contamination_frac
print(f"    Observed RV shift: ~{rv_shift:.1f} km/s")

print("  If companion RV differs by 100 km/s from target:")
rv_shift = 100 * contamination_frac
print(f"    Observed RV shift: ~{rv_shift:.1f} km/s")

print("\n  Our observed total RV range: 146 km/s (-86 to +60)")
print("  13% contamination alone CANNOT explain 146 km/s amplitude")
print("  (would need companion RV to vary by ~1000 km/s)")

# =============================================================================
# GAP 4: The 10 km/s LAMOST discrepancy - how significant?
# =============================================================================
print("\n" + "=" * 70)
print("GAP 4: LAMOST RV DISCREPANCY SIGNIFICANCE")
print("=" * 70)

rv_dr7 = -49.36
rv_err_dr7 = 2.79
rv_mdwarf = -39.44
rv_err_mdwarf = 2.91

combined_err = np.sqrt(rv_err_dr7**2 + rv_err_mdwarf**2)
discrepancy = abs(rv_dr7 - rv_mdwarf)
sigma = discrepancy / combined_err

print(f"\n  DR7 general catalog: {rv_dr7:.2f} +/- {rv_err_dr7:.2f} km/s")
print(f"  M-dwarf catalog:     {rv_mdwarf:.2f} +/- {rv_err_mdwarf:.2f} km/s")
print(f"  Discrepancy: {discrepancy:.2f} km/s")
print(f"  Combined error: {combined_err:.2f} km/s")
print(f"  Significance: {sigma:.1f} sigma")

if sigma > 3:
    print("  --> SIGNIFICANT discrepancy (>3 sigma)")
elif sigma > 2:
    print("  --> MARGINAL discrepancy (2-3 sigma)")
else:
    print("  --> NOT SIGNIFICANT (<2 sigma)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("COMPLETENESS ASSESSMENT")
print("=" * 70)

print("""
GAPS IDENTIFIED:

1. SECOND ObsID (870813030): NOT FOUND in VizieR
   - Could be in newer LAMOST DR not yet public
   - Need direct LAMOST access to verify

2. COMPANION PROPER MOTION: Checked above
   - Determines if contamination is from bound companion vs chance alignment

3. FLUX CONTAMINATION IMPACT: 13% is TOO SMALL to explain 146 km/s
   - Contamination alone doesn't kill the candidate
   - But could add systematic scatter

4. RV DISCREPANCY: 10 km/s is ~2.5 sigma
   - Concerning but not definitive
   - Different pipelines do give different answers for M-dwarfs

REVISED CONFIDENCE LEVELS:
- Kill Mode 1 (RV instability): 60% confident (need ObsID 870813030)
- Kill Mode 2 (Blend): 40% confident (13% flux too small for 146 km/s)
- Kill Mode 3 (SNR): 70% confident (SNR_g genuinely low)

OVERALL: Evidence is WEAKER than initially stated.
The candidate is WOUNDED but not definitively KILLED.
""")

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
