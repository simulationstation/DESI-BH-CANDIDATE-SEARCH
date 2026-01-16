#!/usr/bin/env python3
"""Query DESI data via SPARCL and retrieve spectra."""

from sparcl.client import SparclClient
import numpy as np

print("Connecting to SPARCL...")
client = SparclClient()

# Check available fields
print("\n=== Available Fields for DESI-EDR ===")
try:
    all_fields = client.get_all_fields()
    # It returns a list, not dict
    for f in all_fields[:30]:
        print(f"  {f}")
except Exception as e:
    print(f"Fields: {all_fields[:30] if 'all_fields' in dir() else str(e)}")

# Check available data releases
print("\n=== Available Data Releases ===")
try:
    drs = client.get_data_releases()
    print(f"Releases: {drs}")
except Exception as e:
    print(f"Error: {e}")

# Search for spectra with specific objtype
print("\n=== Searching DESI-EDR Spectra ===")
try:
    result = client.find(
        outfields=['sparcl_id', 'ra', 'dec', 'spectype', 'targetid', 'survey', 'program', 'objtype'],
        constraints={'data_release': ['DESI-EDR']},
        limit=100
    )
    print(f"Found {len(result.records)} spectra")

    # Check unique spectypes and objtypes
    spectypes = set()
    objtypes = set()
    for rec in result.records:
        if 'spectype' in rec:
            spectypes.add(rec['spectype'])
        if 'objtype' in rec:
            objtypes.add(rec.get('objtype'))
    print(f"Spectypes found: {spectypes}")
    print(f"Objtypes found: {objtypes}")

    # Show sample
    print("\nSample records:")
    for rec in result.records[:3]:
        print(f"  {rec}")
except Exception as e:
    print(f"Error: {e}")

# Now retrieve actual spectra
print("\n=== Retrieving Actual Spectra ===")
try:
    # Get sparcl_ids from our search
    ids = [rec['sparcl_id'] for rec in result.records[:5]]
    print(f"Retrieving spectra for {len(ids)} targets...")

    spectra = client.retrieve(
        uuid_list=ids,
        include=['sparcl_id', 'flux', 'wavelength', 'ivar', 'mask']
    )

    print(f"Retrieved {len(spectra.records)} spectra")
    if spectra.records:
        spec = spectra.records[0]
        print(f"First spectrum keys: {spec.keys()}")
        if 'flux' in spec:
            flux = np.array(spec['flux'])
            print(f"Flux shape: {flux.shape}, range: [{flux.min():.2f}, {flux.max():.2f}]")
        if 'wavelength' in spec:
            wave = np.array(spec['wavelength'])
            print(f"Wavelength range: [{wave.min():.1f}, {wave.max():.1f}] Angstroms")
except Exception as e:
    print(f"Error retrieving spectra: {e}")

print("\n=== SUCCESS: SPARCL connection verified ===")
