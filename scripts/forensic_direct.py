#!/usr/bin/env python3
"""
E9 DIRECT FORENSIC VERIFICATION

Downloads coadd tiles via Globus and analyzes beam profiles directly.
Sky fibers aren't in SPARCL, so we need direct file access.
"""

import pandas as pd
import numpy as np
from astropy.io import fits
from pathlib import Path
import subprocess
import re
import warnings
warnings.filterwarnings('ignore')

# Config
DESI_ENDPOINT = "6b4e1f6a-e600-11ed-9b9b-c9bb788c490e"
LOCAL_ENDPOINT = "d1294d08-e055-11f0-a4db-0213754b0ca1"
DESI_BASE = "/dr1/spectro/redux/iron/tiles/pernight"
TEMP_DIR = Path("temp_data")
OUTPUT_DIR = Path("laser_candidates")

# Known lines
AIRGLOW_LINES = [5577.34, 5889.95, 5895.92, 6300.30, 6363.78, 6533.04,
    6863.96, 7316.29, 7340.89, 7571.75, 7750.64, 7794.11,
    7913.71, 7993.33, 8344.60, 8399.17, 8430.17, 8827.10, 8885.85]
ASTROPHYSICAL_LINES = [6562.8, 6548.0, 6583.5, 4861.3, 4959.0, 5007.0]

def is_known_line(wavelength, tolerance=5.0):
    for line in AIRGLOW_LINES:
        if abs(wavelength - line) < tolerance:
            return f"AIRGLOW_{line:.0f}"
    for line in ASTROPHYSICAL_LINES:
        if abs(wavelength - line) < tolerance:
            return f"ASTRO_{line:.0f}"
    return None

def globus_ls(path, timeout=60):
    cmd = f'globus ls "{DESI_ENDPOINT}:{path}" 2>/dev/null'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        return [x for x in result.stdout.strip().split('\n') if x]
    except:
        return None

def globus_transfer(src_path, local_name, timeout=300):
    local_path = TEMP_DIR / local_name
    if local_path.exists():
        return local_path  # Already have it

    # Also check if search script already downloaded it with different name
    # Search uses: coadd-{tile_id}.fits, forensics uses: forensic-{tile_id}.fits
    if local_name.startswith('forensic-'):
        tile_id = local_name.replace('forensic-', '').replace('.fits', '')
        alt_path = TEMP_DIR / f"coadd-{tile_id}.fits"
        if alt_path.exists():
            return alt_path  # Use file from search

    cmd = f'globus transfer "{DESI_ENDPOINT}:{src_path}" "{LOCAL_ENDPOINT}:~/DESI-BH-CANDIDATE-SEARCH/{local_path}" --label "E9-forensic"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        match = re.search(r'Task ID: ([a-f0-9-]+)', result.stdout)
        if not match:
            return None
        task_id = match.group(1)
        subprocess.run(f'globus task wait {task_id}', shell=True, timeout=timeout)
        if local_path.exists() and local_path.stat().st_size > 1000:
            return local_path
    except Exception as e:
        print(f"Transfer error: {e}")
    return None

def classify_profile(flux_region):
    """Classify based on beam profile shape."""
    if len(flux_region) < 5:
        return "UNKNOWN", "Insufficient data"

    peak_idx = np.argmax(flux_region)
    peak = flux_region[peak_idx]

    # Get neighbors
    left1 = flux_region[peak_idx-1] if peak_idx > 0 else 0
    right1 = flux_region[peak_idx+1] if peak_idx < len(flux_region)-1 else 0
    left2 = flux_region[peak_idx-2] if peak_idx > 1 else 0
    right2 = flux_region[peak_idx+2] if peak_idx < len(flux_region)-2 else 0

    # Background from edges
    bg = np.median(np.concatenate([flux_region[:2], flux_region[-2:]]))
    peak_above = peak - bg

    if peak_above <= 0:
        return "UNKNOWN", "No peak"

    # Wing ratios
    wing1 = ((left1 - bg) + (right1 - bg)) / 2 / peak_above
    wing2 = ((left2 - bg) + (right2 - bg)) / 2 / peak_above

    # Classification
    if wing1 < 0.15:
        return "COSMIC_RAY", f"Sharp (w1={wing1:.2f})"
    elif wing1 > 0.25 and wing2 > 0.05:
        return "REAL_SIGNAL", f"Gaussian (w1={wing1:.2f}, w2={wing2:.2f})"
    else:
        return "AMBIGUOUS", f"Intermediate (w1={wing1:.2f}, w2={wing2:.2f})"

def analyze_tile(tile_id, candidates_in_tile, filepath):
    """Analyze all candidates from a single tile."""
    results = []

    try:
        with fits.open(filepath) as hdul:
            fibermap = hdul['FIBERMAP'].data
            r_wave = hdul['R_WAVELENGTH'].data
            r_flux = hdul['R_FLUX'].data

            for _, cand in candidates_in_tile.iterrows():
                tid = cand['targetid']
                wave_peak = cand['wavelength']
                snr = cand['snr']

                # Check known lines
                known = is_known_line(wave_peak)
                if known:
                    results.append({
                        'targetid': tid, 'wavelength': wave_peak, 'snr': snr,
                        'ra': cand['ra'], 'dec': cand['dec'], 'tile': tile_id,
                        'verdict': 'KNOWN_LINE', 'reason': known
                    })
                    continue

                # Find fiber
                fiber_mask = fibermap['TARGETID'] == tid
                if not np.any(fiber_mask):
                    results.append({
                        'targetid': tid, 'wavelength': wave_peak, 'snr': snr,
                        'ra': cand['ra'], 'dec': cand['dec'], 'tile': tile_id,
                        'verdict': 'ERROR', 'reason': 'Fiber not found'
                    })
                    continue

                fiber_idx = np.where(fiber_mask)[0][0]
                flux = r_flux[fiber_idx]

                # Find wavelength index
                wave_idx = np.argmin(np.abs(r_wave - wave_peak))

                # Extract region around peak (+/- 10 pixels)
                start = max(0, wave_idx - 10)
                end = min(len(flux), wave_idx + 11)
                flux_region = flux[start:end]

                # Subtract median sky for this analysis
                sky_mask = fibermap['OBJTYPE'] == 'SKY'
                if np.sum(sky_mask) > 1:
                    median_sky = np.nanmedian(r_flux[sky_mask], axis=0)
                    flux_region = flux[start:end] - median_sky[start:end]

                verdict, reason = classify_profile(flux_region)

                results.append({
                    'targetid': tid, 'wavelength': wave_peak, 'snr': snr,
                    'ra': cand['ra'], 'dec': cand['dec'], 'tile': tile_id,
                    'verdict': verdict, 'reason': reason
                })

    except Exception as e:
        for _, cand in candidates_in_tile.iterrows():
            results.append({
                'targetid': cand['targetid'], 'wavelength': cand['wavelength'],
                'snr': cand['snr'], 'ra': cand['ra'], 'dec': cand['dec'],
                'tile': tile_id, 'verdict': 'ERROR', 'reason': str(e)[:30]
            })

    return results

def main():
    print("=" * 60)
    print("E9 DIRECT FORENSIC VERIFICATION (INCREMENTAL)")
    print("=" * 60)

    TEMP_DIR.mkdir(exist_ok=True)
    RESULTS_FILE = OUTPUT_DIR / 'e9_forensic_results.csv'

    # Load candidates
    df = pd.read_csv('laser_candidates/e9_stream_candidates.csv')
    print(f"Total candidates: {len(df)}")

    # Load existing results to skip already-analyzed
    already_analyzed = set()
    existing_results = []
    if RESULTS_FILE.exists():
        existing_df = pd.read_csv(RESULTS_FILE)
        # Create unique key from targetid + wavelength
        for _, row in existing_df.iterrows():
            key = f"{row['targetid']}_{row['wavelength']:.1f}"
            already_analyzed.add(key)
        existing_results = existing_df.to_dict('records')
        print(f"Already analyzed: {len(already_analyzed)} candidates (skipping)")

    # Take top 100 by SNR, excluding already analyzed
    df['_key'] = df.apply(lambda r: f"{r['targetid']}_{r['wavelength']:.1f}", axis=1)
    df_new = df[~df['_key'].isin(already_analyzed)]

    top_n = 100
    top_cands = df_new.nlargest(top_n, 'snr').copy()
    print(f"New candidates to analyze: {len(top_cands)}")

    if len(top_cands) == 0:
        print("No new candidates to analyze. Exiting.")
        return

    # Group by tile
    tiles = top_cands['tile'].unique()
    print(f"Unique tiles to download: {len(tiles)}")

    all_results = []

    for i, tile_id in enumerate(tiles):
        tile_cands = top_cands[top_cands['tile'] == tile_id]
        print(f"\n[{i+1}/{len(tiles)}] Tile {tile_id} ({len(tile_cands)} candidates)")

        # Find and download coadd file
        tile_path = f"{DESI_BASE}/{tile_id}/"
        dates = globus_ls(tile_path)

        if not dates:
            print(f"  Tile not found")
            for _, c in tile_cands.iterrows():
                all_results.append({
                    'targetid': c['targetid'], 'wavelength': c['wavelength'],
                    'snr': c['snr'], 'ra': c['ra'], 'dec': c['dec'],
                    'tile': tile_id, 'verdict': 'TILE_NOT_FOUND', 'reason': ''
                })
            continue

        valid_dates = [d.rstrip('/') for d in dates if d.rstrip('/').isdigit()]
        if not valid_dates:
            continue
        date = sorted(valid_dates)[-1]

        files = globus_ls(f"{tile_path}{date}/")
        coadd = [f for f in files if f.startswith('coadd-0-') and f.endswith('.fits')]
        if not coadd:
            continue

        src_path = f"{tile_path}{date}/{coadd[0]}"
        local_name = f"forensic-{tile_id}.fits"

        print(f"  Downloading {coadd[0]}...")
        local_path = globus_transfer(src_path, local_name)

        if not local_path:
            print(f"  Download failed")
            continue

        print(f"  Analyzing {len(tile_cands)} candidates...")
        results = analyze_tile(tile_id, tile_cands, local_path)
        all_results.extend(results)

        # Show results for this tile
        for r in results:
            symbol = "+" if r['verdict'] == 'REAL_SIGNAL' else \
                     "x" if r['verdict'] == 'COSMIC_RAY' else \
                     "?" if r['verdict'] == 'AMBIGUOUS' else \
                     "~" if r['verdict'] == 'KNOWN_LINE' else "!"
            print(f"    [{symbol}] {r['wavelength']:.1f}A SNR={r['snr']:.1f} -> {r['verdict']}")

        # Clean up
        try:
            local_path.unlink()
        except:
            pass

    # Merge existing + new results
    all_results = existing_results + all_results

    # Summary
    print("\n" + "=" * 60)
    print("FORENSICS SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(all_results)

    print(f"\nTotal analyzed (all time): {len(results_df)}")
    for verdict, count in results_df['verdict'].value_counts().items():
        print(f"  {verdict}: {count}")

    # Real signals
    real = results_df[results_df['verdict'] == 'REAL_SIGNAL']
    if len(real) > 0:
        print(f"\n*** REAL SIGNAL CANDIDATES ({len(real)}) ***")
        for _, r in real.iterrows():
            print(f"  {r['wavelength']:.1f} A | SNR {r['snr']:.1f}")
            print(f"    RA={r['ra']:.4f} DEC={r['dec']:.4f}")
            print(f"    https://www.legacysurvey.org/viewer?ra={r['ra']}&dec={r['dec']}&layer=ls-dr9&zoom=16")

    # Ambiguous
    ambig = results_df[results_df['verdict'] == 'AMBIGUOUS']
    if len(ambig) > 0:
        print(f"\n*** AMBIGUOUS ({len(ambig)}) ***")
        for _, r in ambig.iterrows():
            print(f"  {r['wavelength']:.1f} A | SNR {r['snr']:.1f} | {r['reason']}")

    # Save
    results_df.to_csv('laser_candidates/e9_forensic_results.csv', index=False)
    print(f"\nResults saved to laser_candidates/e9_forensic_results.csv")

    print("\n" + "=" * 60)
    print("FORENSICS COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
