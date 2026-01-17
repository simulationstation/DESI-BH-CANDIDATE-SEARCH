#!/usr/bin/env python3
"""
Deep LAMOST query via VizieR - multiple catalogs and approaches.
"""

from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

RA = 164.523494
DEC = -1.660156
OBSIDS = [437513049, 870813030, 579613097]  # All mentioned ObsIDs

coord = SkyCoord(ra=RA, dec=DEC, unit=(u.deg, u.deg))

print("=" * 70)
print("DEEP LAMOST VizieR QUERY")
print("=" * 70)
print(f"Target: RA={RA}, Dec={DEC}")
print(f"ObsIDs to check: {OBSIDS}")

# List of LAMOST catalogs in VizieR
lamost_catalogs = [
    'V/164/dr7',      # DR7 general
    'V/164/dr7lrs',   # DR7 LRS
    'V/156/dr5',      # DR5
    'V/153/dr4',      # DR4
    'V/149/dr2',      # DR2
    'J/ApJS/245/34',  # LAMOST M-dwarfs
    'J/ApJ/900/164',  # M dwarf catalog
]

# Configure Vizier for comprehensive queries
v = Vizier(columns=['**'], row_limit=50)

print("\n--- Searching all LAMOST-related catalogs ---\n")

for cat in lamost_catalogs:
    print(f"Catalog: {cat}")
    try:
        result = v.query_region(coord, radius=10*u.arcsec, catalog=cat)
        if result and len(result) > 0:
            print(f"  Found {len(result[0])} entries!")
            for row in result[0]:
                # Print all columns that might be relevant
                cols = result[0].colnames
                print(f"    Columns: {cols[:10]}...")  # Show first 10 columns

                # Look for RV-related columns
                rv_cols = [c for c in cols if 'rv' in c.lower() or 'vel' in c.lower()]
                if rv_cols:
                    print(f"    RV columns: {rv_cols}")
                    for rc in rv_cols:
                        val = row[rc]
                        if not np.ma.is_masked(val):
                            print(f"      {rc}: {val}")

                # Look for SNR columns
                snr_cols = [c for c in cols if 'snr' in c.lower()]
                if snr_cols:
                    print(f"    SNR columns: {snr_cols}")
                    for sc in snr_cols:
                        val = row[sc]
                        if not np.ma.is_masked(val):
                            print(f"      {sc}: {val}")

                # Look for obsid
                obsid_cols = [c for c in cols if 'obsid' in c.lower() or 'obs_id' in c.lower()]
                if obsid_cols:
                    print(f"    ObsID columns: {obsid_cols}")
                    for oc in obsid_cols:
                        val = row[oc]
                        print(f"      {oc}: {val}")

                # Subclass
                sub_cols = [c for c in cols if 'subclass' in c.lower() or 'class' in c.lower() or 'sp' in c.lower()]
                if sub_cols:
                    for sc in sub_cols[:3]:
                        val = row[sc]
                        if not np.ma.is_masked(val):
                            print(f"      {sc}: {val}")

                print()
        else:
            print("  No entries found")
    except Exception as e:
        print(f"  Query error: {e}")
    print()

# Also search general stellar surveys that might have LAMOST cross-matches
print("\n--- Additional stellar catalogs ---\n")

extra_cats = [
    'I/350/gaiaedr3',  # Gaia EDR3
    'I/355/gaiadr3',   # Gaia DR3
]

for cat in extra_cats:
    print(f"Catalog: {cat}")
    try:
        result = v.query_region(coord, radius=2*u.arcsec, catalog=cat)
        if result and len(result) > 0:
            print(f"  Found {len(result[0])} entries")
            for row in result[0]:
                if 'Source' in result[0].colnames:
                    print(f"    Source ID: {row['Source']}")
                if 'RV' in result[0].colnames:
                    print(f"    RV: {row['RV']}")
        else:
            print("  No entries")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 70)
print("QUERY COMPLETE")
print("=" * 70)
