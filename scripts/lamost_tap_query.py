#!/usr/bin/env python3
"""
Direct LAMOST TAP Query for RV version control check.
"""

import requests
import pandas as pd
from io import StringIO
import numpy as np

RA = 164.523494
DEC = -1.660156
OBSIDS = [437513049, 870813030]

print("=" * 70)
print("LAMOST DIRECT TAP QUERY")
print("=" * 70)

# LAMOST TAP endpoint
TAP_URL = "http://www.lamost.org/dr10/v2.0/tap/sync"

# Query by coordinates
query = f"""
SELECT obsid, ra, dec, class, subclass, rv, rv_err, snrg, snrr, snri, mjd
FROM dr10_v2_0.medrs_general
WHERE 1=CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {RA}, {DEC}, 0.002))
ORDER BY mjd
"""

print(f"\nQuerying LAMOST DR10 TAP at coordinates ({RA}, {DEC})...")

try:
    params = {
        'request': 'doQuery',
        'lang': 'ADQL',
        'format': 'csv',
        'query': query
    }
    resp = requests.post(TAP_URL, data=params, timeout=60)

    if resp.status_code == 200 and 'obsid' in resp.text.lower():
        df = pd.read_csv(StringIO(resp.text))
        print(f"Found {len(df)} LAMOST observations")
        print("\nLAMOST DR10 Results:")
        print(df.to_string())

        for _, row in df.iterrows():
            print(f"\n  ObsID: {row['obsid']}")
            print(f"    RV: {row['rv']:.2f} +/- {row['rv_err']:.2f} km/s")
            print(f"    SNR (g/r/i): {row['snrg']:.1f} / {row['snrr']:.1f} / {row['snri']:.1f}")
            print(f"    Class/Subclass: {row['class']} / {row['subclass']}")
            print(f"    MJD: {row['mjd']}")
    else:
        print(f"No results or error. Status: {resp.status_code}")
        print(f"Response preview: {resp.text[:500]}")
except Exception as e:
    print(f"TAP query failed: {e}")

# Also try low-resolution spectra catalog
print("\n" + "=" * 70)
print("Trying LAMOST LRS catalog...")
print("=" * 70)

query_lrs = f"""
SELECT obsid, ra, dec, class, subclass, rv, rv_err, snrg, snrr, snri, mjd
FROM dr10_v2_0.lrs_stellar
WHERE 1=CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {RA}, {DEC}, 0.002))
ORDER BY mjd
"""

try:
    params = {
        'request': 'doQuery',
        'lang': 'ADQL',
        'format': 'csv',
        'query': query_lrs
    }
    resp = requests.post(TAP_URL, data=params, timeout=60)

    if resp.status_code == 200 and 'obsid' in resp.text.lower():
        df = pd.read_csv(StringIO(resp.text))
        print(f"Found {len(df)} LRS observations")
        print("\nLAMOST LRS Results:")
        for _, row in df.iterrows():
            print(f"\n  ObsID: {row['obsid']}")
            print(f"    RV: {row['rv']:.2f} +/- {row['rv_err']:.2f} km/s")
            print(f"    SNR (g/r/i): {row['snrg']:.1f} / {row['snrr']:.1f} / {row['snri']:.1f}")
            print(f"    Class/Subclass: {row['class']} / {row['subclass']}")
            print(f"    MJD: {row['mjd']}")
    else:
        print(f"No LRS results. Status: {resp.status_code}")
except Exception as e:
    print(f"LRS query failed: {e}")

# Query specific obsids
print("\n" + "=" * 70)
print("Querying specific ObsIDs...")
print("=" * 70)

for obsid in OBSIDS:
    print(f"\n--- ObsID {obsid} ---")

    query_obsid = f"""
    SELECT obsid, ra, dec, class, subclass, rv, rv_err, snrg, snrr, snri, mjd
    FROM dr10_v2_0.lrs_stellar
    WHERE obsid = {obsid}
    """

    try:
        params = {
            'request': 'doQuery',
            'lang': 'ADQL',
            'format': 'csv',
            'query': query_obsid
        }
        resp = requests.post(TAP_URL, data=params, timeout=60)

        if resp.status_code == 200 and 'obsid' in resp.text.lower() and len(resp.text) > 100:
            df = pd.read_csv(StringIO(resp.text))
            if len(df) > 0:
                row = df.iloc[0]
                print(f"  Found in DR10!")
                print(f"  RV: {row['rv']:.2f} +/- {row['rv_err']:.2f} km/s")
                print(f"  SNR (g/r/i): {row['snrg']:.1f} / {row['snrr']:.1f} / {row['snri']:.1f}")
                print(f"  Class/Subclass: {row['class']} / {row['subclass']}")
            else:
                print("  Not found in DR10 lrs_stellar")
        else:
            print("  Not found or query error")
    except Exception as e:
        print(f"  Query failed: {e}")

print("\n" + "=" * 70)
print("Query complete")
print("=" * 70)
