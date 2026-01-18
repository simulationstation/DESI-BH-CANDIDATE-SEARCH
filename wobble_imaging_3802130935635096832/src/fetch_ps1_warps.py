#!/usr/bin/env python3
"""Fetch Pan-STARRS1 warp (single-epoch) images for the target."""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import requests
import numpy as np
from astropy.io import fits
from astropy.time import Time
import time as pytime

OUTPUT_DIR = "/home/primary/DESI-BH-CANDIDATE-SEARCH/wobble_imaging_3802130935635096832"
WOBBLE_FIGURES = "/home/primary/DESI-BH-CANDIDATE-SEARCH/WOBBLE_FIGURES"

# PS1 image cutout service
PS1_CUTOUT_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PS1_FITSCUT_URL = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"


def get_ps1_images_list(ra, dec, filters='grizy'):
    """Query PS1 for available images at position."""
    print(f"[PS1] Querying image list at RA={ra:.6f}, Dec={dec:.6f}")

    params = {
        'ra': ra,
        'dec': dec,
        'filters': filters,
        'type': 'warp',  # single-epoch images
    }

    try:
        response = requests.get(PS1_CUTOUT_URL, params=params, timeout=60)
        response.raise_for_status()
    except Exception as e:
        print(f"[PS1] Error querying image list: {e}")
        return []

    lines = response.text.strip().split('\n')
    if len(lines) <= 1:
        print("[PS1] No warp images found")
        return []

    # Parse header and data
    header = lines[0].split()
    images = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= len(header):
            img = dict(zip(header, parts))
            images.append(img)

    print(f"[PS1] Found {len(images)} warp images")
    return images


def download_ps1_cutout(ra, dec, filename, size=240, output_format='fits'):
    """Download a cutout from a PS1 warp image."""
    # Extract filter from filename
    params = {
        'ra': ra,
        'dec': dec,
        'size': size,
        'format': output_format,
        'red': filename,
    }

    try:
        response = requests.get(PS1_FITSCUT_URL, params=params, timeout=120)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"[PS1] Error downloading cutout: {e}")
        return None


def parse_mjd_from_filename(filename):
    """Extract MJD from PS1 warp filename."""
    # PS1 warp filenames contain the skycell and other info
    # The MJD is embedded in the filename pattern
    # Example: rings.v3.skycell.1234.056.wrp.g.12345.fits
    # We need to query the metadata for exact MJD
    return None  # Will get from FITS header


def main():
    # Load target coordinates
    coords_file = os.path.join(OUTPUT_DIR, 'data', 'target_coords.json')
    with open(coords_file, 'r') as f:
        coords = json.load(f)

    ra = coords['ra']
    dec = coords['dec']

    # Get list of available PS1 warp images
    images = get_ps1_images_list(ra, dec, filters='gri')

    if not images:
        print("[PS1] No images available, trying stacked images instead")
        # Fall back to stacked images if no warps
        params = {
            'ra': ra,
            'dec': dec,
            'filters': 'gri',
            'type': 'stack',
        }
        response = requests.get(PS1_CUTOUT_URL, params=params, timeout=60)
        lines = response.text.strip().split('\n')
        if len(lines) > 1:
            header = lines[0].split()
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= len(header):
                    img = dict(zip(header, parts))
                    images.append(img)
            print(f"[PS1] Found {len(images)} stacked images")

    # Download cutouts
    data_dir = os.path.join(OUTPUT_DIR, 'data', 'ps1_images')
    os.makedirs(data_dir, exist_ok=True)

    downloaded = []
    for i, img in enumerate(images[:50]):  # Limit to 50 images
        filename = img.get('filename', '')
        if not filename:
            continue

        outfile = os.path.join(data_dir, f"cutout_{i:03d}.fits")

        if os.path.exists(outfile):
            print(f"[PS1] Already have {outfile}")
            downloaded.append({'index': i, 'filename': filename, 'local': outfile})
            continue

        print(f"[PS1] Downloading {i+1}/{min(len(images), 50)}: {filename[:50]}...")
        data = download_ps1_cutout(ra, dec, filename, size=120)

        if data:
            with open(outfile, 'wb') as f:
                f.write(data)
            downloaded.append({'index': i, 'filename': filename, 'local': outfile})
            pytime.sleep(0.5)  # Rate limiting

    print(f"[PS1] Downloaded {len(downloaded)} cutouts")

    # Extract metadata from downloaded FITS files
    image_metadata = []
    for item in downloaded:
        try:
            with fits.open(item['local']) as hdu:
                header = hdu[0].header
                meta = {
                    'index': item['index'],
                    'filename': item['filename'],
                    'local': item['local'],
                    'mjd': header.get('MJD-OBS', header.get('MJD', None)),
                    'filter': header.get('FILTER', header.get('HIERARCH FPA.FILTERID', 'unknown')),
                    'exptime': header.get('EXPTIME', None),
                    'seeing': header.get('SEEING', header.get('PSF_FWHM', None)),
                }
                if meta['mjd']:
                    meta['date'] = Time(meta['mjd'], format='mjd').iso
                image_metadata.append(meta)
        except Exception as e:
            print(f"[PS1] Error reading {item['local']}: {e}")

    # Sort by MJD
    image_metadata.sort(key=lambda x: x.get('mjd', 0) or 0)

    # Save metadata
    meta_file = os.path.join(OUTPUT_DIR, 'data', 'ps1_metadata.json')
    with open(meta_file, 'w') as f:
        json.dump(image_metadata, f, indent=2)

    print(f"[PS1] Saved metadata to {meta_file}")

    # Summary
    if image_metadata:
        mjds = [m['mjd'] for m in image_metadata if m['mjd']]
        if mjds:
            print(f"[PS1] MJD range: {min(mjds):.2f} to {max(mjds):.2f}")
            baseline_days = max(mjds) - min(mjds)
            print(f"[PS1] Baseline: {baseline_days:.1f} days ({baseline_days/365.25:.2f} years)")


if __name__ == "__main__":
    main()
