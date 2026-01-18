#!/usr/bin/env python3
"""Measure centroids in PS1 warp images and build wobble time-series."""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_2dg, centroid_com
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/home/primary/DESI-BH-CANDIDATE-SEARCH/wobble_imaging_3802130935635096832"
WOBBLE_FIGURES = "/home/primary/DESI-BH-CANDIDATE-SEARCH/WOBBLE_FIGURES"


def find_target_centroid(data, x_init, y_init, box_size=15):
    """Find centroid of target star near initial position."""
    # Extract subimage around expected position
    ny, nx = data.shape
    x0 = max(0, int(x_init) - box_size)
    x1 = min(nx, int(x_init) + box_size)
    y0 = max(0, int(y_init) - box_size)
    y1 = min(ny, int(y_init) + box_size)

    subdata = data[y0:y1, x0:x1].copy()

    # Handle NaN/inf
    subdata = np.nan_to_num(subdata, nan=0, posinf=0, neginf=0)

    if subdata.size == 0 or np.all(subdata == 0):
        return None, None, None

    # Background subtract
    mean, median, std = sigma_clipped_stats(subdata, sigma=3.0)
    subdata_bgsub = subdata - median

    # Find centroid using 2D Gaussian fit
    try:
        x_cen, y_cen = centroid_2dg(subdata_bgsub)
        if np.isnan(x_cen) or np.isnan(y_cen):
            raise ValueError("NaN centroid")
    except:
        # Fall back to center of mass
        try:
            x_cen, y_cen = centroid_com(subdata_bgsub)
        except:
            return None, None, None

    # Convert back to full image coords
    x_full = x_cen + x0
    y_full = y_cen + y0

    # Estimate uncertainty from SNR
    peak = np.max(subdata_bgsub)
    snr = peak / std if std > 0 else 1
    # Centroid uncertainty ~ FWHM / SNR (rough approximation)
    fwhm_pix = 3.0  # typical PS1 seeing
    sigma_cen = fwhm_pix / snr if snr > 1 else fwhm_pix

    return x_full, y_full, sigma_cen


def find_reference_stars(data, exclude_center=True, threshold_sigma=5):
    """Find reference stars in image for alignment."""
    data_clean = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

    mean, median, std = sigma_clipped_stats(data_clean, sigma=3.0)

    # DAOStarFinder to detect stars
    daofind = DAOStarFinder(fwhm=3.0, threshold=threshold_sigma*std)
    try:
        sources = daofind(data_clean - median)
    except:
        return []

    if sources is None or len(sources) == 0:
        return []

    # Exclude center (target) region
    ny, nx = data.shape
    cx, cy = nx/2, ny/2

    stars = []
    for row in sources:
        x, y = row['xcentroid'], row['ycentroid']

        # Skip if too close to center
        if exclude_center:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < 15:
                continue

        stars.append({
            'x': float(x),
            'y': float(y),
            'flux': float(row['flux']),
        })

    # Sort by flux, take brightest
    stars.sort(key=lambda s: s['flux'], reverse=True)
    return stars[:10]


def compute_differential_centroid(target_x, target_y, ref_stars):
    """Compute differential centroid relative to reference stars."""
    if len(ref_stars) < 2:
        return target_x, target_y, None

    # Use mean position of reference stars as reference frame
    ref_x = np.mean([s['x'] for s in ref_stars])
    ref_y = np.mean([s['y'] for s in ref_stars])

    # Differential position
    dx = target_x - ref_x
    dy = target_y - ref_y

    # Reference scatter as error estimate
    ref_scatter = np.std([s['x'] for s in ref_stars])

    return dx, dy, ref_scatter


def main():
    # Load metadata
    meta_file = os.path.join(OUTPUT_DIR, 'data', 'ps1_metadata.json')
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    # Load target coords
    coords_file = os.path.join(OUTPUT_DIR, 'data', 'target_coords.json')
    with open(coords_file, 'r') as f:
        coords = json.load(f)

    results = []

    print(f"[CENTROID] Processing {len(metadata)} images")

    for i, meta in enumerate(metadata):
        filepath = meta.get('local')
        if not filepath or not os.path.exists(filepath):
            continue

        try:
            with fits.open(filepath) as hdu:
                data = hdu[0].data
                header = hdu[0].header

                if data is None:
                    print(f"[CENTROID] No data in {filepath}")
                    continue

                # Get WCS to find target position
                try:
                    wcs = WCS(header)
                    x_init, y_init = wcs.world_to_pixel_values(coords['ra'], coords['dec'])
                except:
                    # Fall back to image center
                    ny, nx = data.shape
                    x_init, y_init = nx/2, ny/2

                # Find target centroid
                x_cen, y_cen, sigma_cen = find_target_centroid(data, x_init, y_init)

                if x_cen is None:
                    print(f"[CENTROID] Failed to find target in image {i}")
                    continue

                # Find reference stars
                ref_stars = find_reference_stars(data)

                # Compute differential position
                dx, dy, ref_scatter = compute_differential_centroid(x_cen, y_cen, ref_stars)

                mjd = meta.get('mjd')
                filt = meta.get('filter', 'unknown')

                result = {
                    'index': i,
                    'mjd': mjd,
                    'filter': filt,
                    'x_pix': float(x_cen),
                    'y_pix': float(y_cen),
                    'sigma_pix': float(sigma_cen) if sigma_cen else None,
                    'dx_diff': float(dx),
                    'dy_diff': float(dy),
                    'n_ref_stars': len(ref_stars),
                    'ref_scatter': float(ref_scatter) if ref_scatter else None,
                }
                results.append(result)

                if (i + 1) % 10 == 0:
                    print(f"[CENTROID] Processed {i+1}/{len(metadata)} images")

        except Exception as e:
            print(f"[CENTROID] Error processing {filepath}: {e}")
            continue

    # Filter out images with no valid measurements
    results = [r for r in results if r['mjd'] is not None]

    # Sort by MJD
    results.sort(key=lambda r: r['mjd'])

    # Compute statistics
    if results:
        dx_vals = [r['dx_diff'] for r in results]
        dy_vals = [r['dy_diff'] for r in results]

        # Remove outliers
        dx_med = np.median(dx_vals)
        dy_med = np.median(dy_vals)
        dx_std = np.std(dx_vals)
        dy_std = np.std(dy_vals)

        # Subtract median to center
        for r in results:
            r['dx_centered'] = r['dx_diff'] - dx_med
            r['dy_centered'] = r['dy_diff'] - dy_med

        # PS1 pixel scale ~ 0.25 arcsec/pix
        pixel_scale = 0.25  # arcsec/pix

        # Convert to arcsec
        for r in results:
            r['dx_arcsec'] = r['dx_centered'] * pixel_scale
            r['dy_arcsec'] = r['dy_centered'] * pixel_scale

        dx_arcsec = [r['dx_arcsec'] for r in results]
        dy_arcsec = [r['dy_arcsec'] for r in results]

        summary = {
            'n_images': len(results),
            'mjd_range': [min(r['mjd'] for r in results), max(r['mjd'] for r in results)],
            'baseline_years': (max(r['mjd'] for r in results) - min(r['mjd'] for r in results)) / 365.25,
            'dx_rms_arcsec': float(np.std(dx_arcsec)),
            'dy_rms_arcsec': float(np.std(dy_arcsec)),
            'total_rms_arcsec': float(np.sqrt(np.std(dx_arcsec)**2 + np.std(dy_arcsec)**2)),
            'pixel_scale': pixel_scale,
        }

        print(f"[CENTROID] Summary:")
        print(f"  N images: {summary['n_images']}")
        print(f"  Baseline: {summary['baseline_years']:.2f} years")
        print(f"  RMS scatter: dx={summary['dx_rms_arcsec']*1000:.1f} mas, dy={summary['dy_rms_arcsec']*1000:.1f} mas")
        print(f"  Total RMS: {summary['total_rms_arcsec']*1000:.1f} mas")

    else:
        summary = {'n_images': 0, 'error': 'No valid measurements'}

    # Save results
    results_file = os.path.join(OUTPUT_DIR, 'results', 'centroid_measurements.json')
    with open(results_file, 'w') as f:
        json.dump({'summary': summary, 'measurements': results}, f, indent=2)

    print(f"[CENTROID] Saved to {results_file}")

    # Save as CSV for easy analysis
    csv_file = os.path.join(OUTPUT_DIR, 'results', 'astrometry_timeseries.csv')
    with open(csv_file, 'w') as f:
        f.write("mjd,filter,dx_arcsec,dy_arcsec,x_pix,y_pix,n_ref_stars\n")
        for r in results:
            f.write(f"{r['mjd']},{r['filter']},{r['dx_arcsec']:.6f},{r['dy_arcsec']:.6f},"
                    f"{r['x_pix']:.3f},{r['y_pix']:.3f},{r['n_ref_stars']}\n")

    print(f"[CENTROID] Saved CSV to {csv_file}")


if __name__ == "__main__":
    main()
