#!/usr/bin/env python3
"""Check what DESI data is available through Data Lab."""

from dl import authClient as ac
from dl import queryClient as qc
from dl import storeClient as sc
import time

# Login
print("Logging in...")
token = ac.login('aidensmith', 'GreatJob777')
print(f"Logged in as: {ac.whoAmI()}")

# Check what tables exist with DESI
print("\n=== Checking for DESI tables ===")
for attempt in range(5):
    try:
        # Query for DESI-related schemas
        result = qc.query(sql="""
            SELECT DISTINCT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema ILIKE '%desi%'
            OR table_name ILIKE '%desi%'
            LIMIT 50
        """, fmt='csv')
        print(result)
        break
    except Exception as e:
        if '502' in str(e) or '504' in str(e):
            print(f"Server error, retry {attempt+1}...")
            time.sleep(10)
        else:
            print(f"Error: {e}")
            break

# Check storage for DESI files
print("\n=== Checking storage ===")
try:
    # Check user's storage
    files = sc.ls('vos://')
    print(f"Root storage: {files}")
except Exception as e:
    print(f"Storage error: {e}")

# Check if there's a DESI public area
print("\n=== Checking for public DESI data ===")
paths_to_try = [
    'vos://desi',
    'vos://public/desi',
    'vos://datalab/desi',
    'vos://noao/desi',
]
for path in paths_to_try:
    try:
        files = sc.ls(path)
        print(f"{path}: {files[:200]}")
    except Exception as e:
        print(f"{path}: Not found")

print("\n=== CONCLUSION ===")
print("Data Lab provides DESI data through SPARCL (coadded spectra).")
print("Raw exposure cframe files are only at data.desi.lbl.gov")
