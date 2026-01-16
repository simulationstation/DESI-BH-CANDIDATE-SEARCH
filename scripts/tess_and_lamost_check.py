#!/usr/bin/env python3
"""
TESS and LAMOST direct checks.
"""

import requests
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

RA = 164.523494
DEC = -1.660156

print("=" * 70)
print("TESS AND LAMOST DIRECT CHECKS")
print("=" * 70)

# =============================================================================
# TESS Check via MAST API
# =============================================================================
print("\n--- TESS SECTOR CHECK ---")

try:
    # Use MAST API directly
    url = "https://mast.stsci.edu/api/v0.1/tesscut/pointsector"
    params = {"ra": RA, "dec": DEC}
    resp = requests.get(url, params=params, timeout=30)

    if resp.status_code == 200:
        data = resp.json()
        if 'sectors' in data and len(data['sectors']) > 0:
            print(f"  Found TESS coverage in {len(data['sectors'])} sector(s)!")
            for s in data['sectors'][:10]:
                print(f"    Sector {s}")
        else:
            print("  No TESS sectors found")
            print(f"  Response: {data}")
    else:
        print(f"  API error: {resp.status_code}")
except Exception as e:
    print(f"  Error: {e}")

# Alternative: Check TIC catalog
print("\n--- TIC CATALOG CHECK ---")
try:
    tic_url = f"https://mast.stsci.edu/api/v0.1/ticposition?ra={RA}&dec={DEC}&radius=0.01"
    resp = requests.get(tic_url, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        if 'data' in data and len(data['data']) > 0:
            print(f"  Found {len(data['data'])} TIC source(s)")
            for src in data['data'][:3]:
                print(f"    TIC ID: {src.get('ID', 'N/A')}, Tmag: {src.get('Tmag', 'N/A')}")
        else:
            print("  No TIC sources found")
    else:
        print(f"  TIC query failed: {resp.status_code}")
except Exception as e:
    print(f"  TIC error: {e}")

# =============================================================================
# LAMOST ObsID Investigation
# =============================================================================
print("\n" + "=" * 70)
print("LAMOST ObsID INVESTIGATION")
print("=" * 70)

obsids = [437513049, 579613097, 870813030]

for obsid in obsids:
    print(f"\n--- ObsID {obsid} ---")

    # Try to fetch metadata from LAMOST
    url = f"http://www.lamost.org/dr10/v2.0/api/object?obsid={obsid}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            try:
                data = resp.json()
                if data:
                    print(f"  Found in LAMOST DR10!")
                    # Print relevant fields
                    for key in ['ra', 'dec', 'class', 'subclass', 'rv', 'rv_err', 'snrg', 'snrr']:
                        if key in data:
                            print(f"    {key}: {data[key]}")
                else:
                    print("  Empty response")
            except:
                print(f"  Response (not JSON): {resp.text[:200]}")
        else:
            print(f"  HTTP {resp.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

    # Alternative URL format
    alt_url = f"http://www.lamost.org/dr10/v2.0/spectrum/fits/{obsid}"
    print(f"  FITS URL: {alt_url}")

# =============================================================================
# Web search URL
# =============================================================================
print("\n" + "=" * 70)
print("MANUAL VERIFICATION URLS")
print("=" * 70)

print(f"""
LAMOST coordinate search:
  http://www.lamost.org/dr10/search?ra={RA}&dec={DEC}&radius=10

LAMOST spectrum viewer:
  http://www.lamost.org/dr10/v2.0/spectrum/view?obsid=437513049
  http://www.lamost.org/dr10/v2.0/spectrum/view?obsid=579613097
  http://www.lamost.org/dr10/v2.0/spectrum/view?obsid=870813030

Legacy Survey viewer (check for blend visually):
  https://www.legacysurvey.org/viewer?ra={RA}&dec={DEC}&layer=ls-dr10&zoom=16

Aladin Lite (multi-survey):
  https://aladin.cds.unistra.fr/AladinLite/?target={RA}%20{DEC}&fov=0.05&survey=CDS%2FP%2FDSS2%2Fcolor
""")

print("=" * 70)
print("COMPLETE")
print("=" * 70)
