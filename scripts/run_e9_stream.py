#!/usr/bin/env python3
"""
E9 STREAMING LASER SEARCH - Raw DESI Sky Fiber Scanner via Globus

Downloads -> Scans -> Deletes coadd files one at a time.
Searches SKY fibers for monochromatic optical pulses.
RESUMABLE - saves progress and continues from where it left off.
"""

import os
import subprocess
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
from datetime import datetime
import time
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DESI_ENDPOINT = "6b4e1f6a-e600-11ed-9b9b-c9bb788c490e"
LOCAL_ENDPOINT = "d1294d08-e055-11f0-a4db-0213754b0ca1"
DESI_BASE = "/dr1/spectro/redux/iron/tiles/pernight"

TEMP_DIR = Path("temp_data")
OUTPUT_DIR = Path("laser_candidates")
CANDIDATES_FILE = OUTPUT_DIR / "e9_stream_candidates.csv"
PROGRESS_FILE = OUTPUT_DIR / "e9_progress.txt"
PROCESSED_FILE = OUTPUT_DIR / "e9_processed_tiles.txt"

SNR_THRESHOLD = 10.0
MAX_WIDTH_PIXELS = 3
AIRGLOW_LINES = [5577.34, 5889.95, 5895.92, 6300.30, 6363.78, 6533.04,
    6863.96, 7316.29, 7340.89, 7571.75, 7750.64, 7794.11,
    7913.71, 7993.33, 8344.60, 8399.17, 8430.17, 8827.10, 8885.85]

# Pre-defined tile ranges - covers ALL DESI DR1 pernight tiles
# Total: 716 tiles (1-999 small survey + 80xxx-82633 main survey)
TILE_RANGES = [
    # Small survey/commissioning tiles (1-999)
    (1, 100), (100, 200), (200, 300), (300, 400), (400, 500),
    (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000),
    # Main survey tiles (80000-83000)
    (80000, 80500), (80500, 81000), (81000, 81500), (81500, 82000),
    (82000, 82500), (82500, 83000),
]

# ============================================================================
# GLOBUS FUNCTIONS
# ============================================================================

def globus_ls(path, timeout=60):
    """List directory via Globus."""
    cmd = f'globus ls "{DESI_ENDPOINT}:{path}" 2>/dev/null'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        return [x for x in result.stdout.strip().split('\n') if x]
    except:
        return None

def globus_transfer(src_path, local_name, timeout=300):
    """Transfer file and wait for completion."""
    local_path = TEMP_DIR / local_name

    # Remove existing file
    if local_path.exists():
        local_path.unlink()

    cmd = f'globus transfer "{DESI_ENDPOINT}:{src_path}" "{LOCAL_ENDPOINT}:~/DESI-BH-CANDIDATE-SEARCH/{local_path}" --label "E9-laser"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        # Extract task ID
        match = re.search(r'Task ID: ([a-f0-9-]+)', result.stdout)
        if not match:
            return None
        task_id = match.group(1)

        # Wait for transfer
        wait_cmd = f'globus task wait {task_id}'
        subprocess.run(wait_cmd, shell=True, timeout=timeout)

        if local_path.exists() and local_path.stat().st_size > 1000:
            return local_path
    except Exception as e:
        print(f"Transfer error: {e}")
    return None

# ============================================================================
# LASER DETECTION
# ============================================================================

def is_airglow(w):
    return any(abs(w - l) < 3.0 for l in AIRGLOW_LINES)

def search_spectrum(wave, flux, ivar):
    """Find narrow peaks in spectrum."""
    from scipy.ndimage import gaussian_filter1d

    cont = gaussian_filter1d(flux, 50)
    resid = flux - cont
    noise = np.maximum(np.nanmedian(np.abs(resid)) * 1.4826, 0.01)

    with np.errstate(invalid='ignore', divide='ignore'):
        ivar_sigma = np.where(ivar > 0, 1.0/np.sqrt(ivar), np.inf)
    total_noise = np.sqrt(noise**2 + np.minimum(ivar_sigma, 100)**2)
    snr = resid / total_noise

    peaks = []
    above = snr > SNR_THRESHOLD
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            idx = start + np.argmax(snr[start:i])
            half = resid[idx] / 2
            l, r = idx, idx
            while l > 0 and resid[l] > half: l -= 1
            while r < len(resid)-1 and resid[r] > half: r += 1
            width = r - l
            if width <= MAX_WIDTH_PIXELS and not is_airglow(wave[idx]):
                peaks.append({'wavelength': wave[idx], 'snr': snr[idx], 'flux': resid[idx], 'width': width})
        i += 1
    return peaks

def process_coadd(filepath, tile_id):
    """Process coadd file for laser candidates."""
    candidates = []
    n_sky = 0

    try:
        with fits.open(filepath) as hdul:
            fibermap = hdul['FIBERMAP'].data
            sky_mask = fibermap['OBJTYPE'] == 'SKY'
            n_sky = np.sum(sky_mask)

            if n_sky == 0:
                return 0, candidates

            r_wave = hdul['R_WAVELENGTH'].data
            r_flux = hdul['R_FLUX'].data
            r_ivar = hdul['R_IVAR'].data

            sky_flux = r_flux[sky_mask]
            sky_ivar = r_ivar[sky_mask]
            sky_map = fibermap[sky_mask]

            median_sky = np.nanmedian(sky_flux, axis=0)

            for i in range(n_sky):
                resid = sky_flux[i] - median_sky
                peaks = search_spectrum(r_wave, resid, sky_ivar[i])

                for p in peaks:
                    candidates.append({
                        'targetid': int(sky_map['TARGETID'][i]),
                        'fiber': int(sky_map['FIBER'][i]),
                        'ra': float(sky_map['TARGET_RA'][i]),
                        'dec': float(sky_map['TARGET_DEC'][i]),
                        'wavelength': float(p['wavelength']),
                        'flux': float(p['flux']),
                        'snr': float(p['snr']),
                        'width_pixels': int(p['width']),
                        'tile': tile_id,
                        'camera': 'R',
                    })

    except Exception as e:
        print(f" Error: {e}")

    return n_sky, candidates

# ============================================================================
# RESUMABLE STATE
# ============================================================================

def load_processed():
    """Load set of already processed tiles."""
    if PROCESSED_FILE.exists():
        return set(PROCESSED_FILE.read_text().strip().split('\n'))
    return set()

def save_processed(tile_id, processed_set):
    """Add tile to processed set."""
    processed_set.add(str(tile_id))
    with open(PROCESSED_FILE, 'a') as f:
        f.write(f"{tile_id}\n")

def save_progress(tile_id, total_processed, total_sky, total_cands):
    """Save progress checkpoint."""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(f"{tile_id},{total_processed},{total_sky},{total_cands},{datetime.now().isoformat()}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("E9 STREAMING LASER SEARCH - Globus/DESI Sky Fibers")
    print("=" * 60)

    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Init CSV
    if not CANDIDATES_FILE.exists():
        pd.DataFrame(columns=[
            'targetid', 'fiber', 'ra', 'dec', 'wavelength',
            'flux', 'snr', 'width_pixels', 'tile', 'camera'
        ]).to_csv(CANDIDATES_FILE, index=False)

    # Load already processed tiles
    processed_tiles = load_processed()
    print(f"Already processed: {len(processed_tiles)} tiles")

    total_sky = 0
    total_cands = 0
    total_processed = len(processed_tiles)

    # Iterate through tile ranges
    for start, end in TILE_RANGES:
        for tile_id in range(start, end):
            if str(tile_id) in processed_tiles:
                continue

            tile_path = f"{DESI_BASE}/{tile_id}/"

            # Check if tile exists and get dates
            dates = globus_ls(tile_path, timeout=30)
            if not dates:
                continue

            # Get latest date
            valid_dates = [d.rstrip('/') for d in dates if d.rstrip('/').isdigit()]
            if not valid_dates:
                continue
            date = sorted(valid_dates)[-1]

            # List files in tile/date
            files = globus_ls(f"{tile_path}{date}/", timeout=30)
            if not files:
                continue

            # Find coadd-0 file
            coadd = [f for f in files if f.startswith('coadd-0-') and f.endswith('.fits')]
            if not coadd:
                continue

            src_path = f"{tile_path}{date}/{coadd[0]}"
            local_name = f"coadd-{tile_id}.fits"

            print(f"[{tile_id}] ", end="", flush=True)

            # Transfer
            local_path = globus_transfer(src_path, local_name)
            if not local_path:
                print("SKIP")
                continue

            # Process
            n_sky, cands = process_coadd(local_path, tile_id)
            total_sky += n_sky
            total_processed += 1

            # Save candidates
            if cands:
                df = pd.DataFrame(cands)
                df.to_csv(CANDIDATES_FILE, mode='a', header=False, index=False)
                total_cands += len(cands)

            # Keep file for forensics (under 250GB total, saves re-downloading)
            # Files stored in temp_data/coadd-{tile_id}.fits

            # Mark as processed
            save_processed(tile_id, processed_tiles)
            save_progress(tile_id, total_processed, total_sky, total_cands)

            print(f"{n_sky} sky, {len(cands)} hits | Total: {total_processed} tiles, {total_cands} cands")

            for c in cands[:2]:
                print(f"  >>> {c['wavelength']:.1f}Å SNR={c['snr']:.1f} RA={c['ra']:.3f} DEC={c['dec']:.3f}")

    print("\n" + "=" * 60)
    print(f"DONE: {total_processed} tiles, {total_sky} sky fibers, {total_cands} candidates")
    print(f"Results: {CANDIDATES_FILE}")
    print("=" * 60)

if __name__ == '__main__':
    import fcntl
    LOCK_FILE = OUTPUT_DIR / "e9_stream.lock"
    OUTPUT_DIR.mkdir(exist_ok=True)

    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Another instance is already running. Exiting.")
        exit(1)

    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for f in TEMP_DIR.glob('*'):
            try: f.unlink()
            except: pass
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
