#!/usr/bin/env python3
"""
DESI BLEND-AWARE RV RE-MEASUREMENT

Comprehensive analysis of DESI DR1 spectra for target Gaia DR3 3802130935635096832
(DESI TARGETID 39627745210139276) to determine whether the observed RV swing
(-86 to +60 km/s) is robust to blending with the confirmed 0.688" neighbor.

Tests performed:
1. Single-template RV measurement with multiple wavelength masks
2. CCF peak multiplicity / component switching detection
3. Wavelength-split RV consistency test
4. Two-template (blend/SB2) fit comparison

Author: Claude Code
Date: 2026-01-16
"""

import numpy as np
import json
import hashlib
import os
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_ID = 39627745210139276
GAIA_SOURCE_ID = 3802130935635096832

# Known neighbor properties
NEIGHBOR_SEP_ARCSEC = 0.68793
NEIGHBOR_DELTA_G = 2.21099  # mag fainter
NEIGHBOR_FLUX_FRAC = 10**(-0.4 * NEIGHBOR_DELTA_G)  # ~0.13 in G-band

# DESI fiber aperture
DESI_FIBER_DIAMETER_ARCSEC = 1.5

# Paths
BASE_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH")
SPECTRA_DIR = BASE_DIR / "data" / "desi_spectra"
OUTPUT_DIR = BASE_DIR / "outputs" / "desi_blend_v1"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Epoch metadata (from rvpix files)
EPOCHS = [
    {'mjd': 59568.48825, 'tile': 20268, 'night': 20211219, 'petal': 2,
     'fiber': 1377, 'rv_catalog': -86.39, 'rv_err_catalog': 0.55,
     'coadd_file': 'coadd_20268_20211219_p2.fits', 'label': 'Epoch1 (MJD 59568)'},
    {'mjd': 59605.38003, 'tile': 24976, 'night': 20220125, 'petal': 7,
     'fiber': 3816, 'rv_catalog': 59.68, 'rv_err_catalog': 0.83,
     'coadd_file': 'coadd_24976_20220125_p7.fits', 'label': 'Epoch2 (MJD 59605)'},
    {'mjd': 59607.38, 'tile': 23137, 'night': 20220127, 'petal': 0,
     'fiber': 68, 'rv_catalog': 25.8, 'rv_err_catalog': 0.8,  # avg of 26.43 and 25.16
     'coadd_file': 'coadd_23137_20220127_p0.fits', 'label': 'Epoch3 (MJD 59607, coadd)'},
]

# Telluric regions to mask (Angstroms)
TELLURIC_REGIONS = [
    (6270, 6330),   # O2 B-band
    (6860, 6960),   # O2 A-band (strong)
    (7160, 7340),   # H2O
    (7590, 7700),   # O2 A-band extended
    (8130, 8350),   # H2O
    (8940, 9200),   # H2O (strong)
    (9300, 9800),   # H2O (very strong)
]

# Wavelength masks for RV extraction
WAVELENGTH_MASKS = {
    'TiO-6500': (6400, 6800),     # TiO-rich red optical
    'TiO-7200': (7000, 7400),     # TiO/metal region (excluding O2 A-band)
    'FarRed': (7650, 8100),       # Far red (between tellurics)
    'Combined': (6400, 8100),     # Broad combined red
}

# CCF parameters
CCF_V_RANGE = (-300, 300)  # km/s
CCF_V_STEP = 1.0  # km/s
C_KMS = 299792.458  # speed of light

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_file_hash(filepath):
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def apply_telluric_mask(wavelength, flux, ivar):
    """Mask telluric regions by setting ivar to 0."""
    mask = np.ones(len(wavelength), dtype=bool)
    for wl_min, wl_max in TELLURIC_REGIONS:
        mask &= ~((wavelength >= wl_min) & (wavelength <= wl_max))
    ivar_masked = ivar.copy()
    ivar_masked[~mask] = 0
    return ivar_masked

def continuum_normalize(wavelength, flux, ivar, window=100):
    """
    Continuum normalize spectrum using running median.
    Returns normalized flux and updated ivar.
    """
    good = (ivar > 0) & np.isfinite(flux)
    if np.sum(good) < 100:
        return flux, ivar

    # Running median for continuum
    continuum = np.ones_like(flux)
    for i in range(len(wavelength)):
        wl_min = wavelength[i] - window/2
        wl_max = wavelength[i] + window/2
        in_window = (wavelength >= wl_min) & (wavelength <= wl_max) & good
        if np.sum(in_window) > 10:
            continuum[i] = np.median(flux[in_window])
        else:
            continuum[i] = 1.0

    # Smooth continuum
    continuum = gaussian_filter1d(continuum, sigma=20)
    continuum[continuum <= 0] = 1.0

    flux_norm = flux / continuum
    ivar_norm = ivar * continuum**2

    return flux_norm, ivar_norm

def resample_to_log_wavelength(wavelength, flux, ivar, wl_min, wl_max, n_pixels=2048):
    """Resample spectrum to logarithmic wavelength grid."""
    log_wl_min = np.log(wl_min)
    log_wl_max = np.log(wl_max)
    log_wl = np.linspace(log_wl_min, log_wl_max, n_pixels)
    wl_new = np.exp(log_wl)

    # Interpolate
    good = (ivar > 0) & np.isfinite(flux)
    if np.sum(good) < 100:
        return wl_new, np.zeros(n_pixels), np.zeros(n_pixels)

    f_flux = interp1d(wavelength[good], flux[good], bounds_error=False, fill_value=np.nan)
    f_ivar = interp1d(wavelength[good], ivar[good], bounds_error=False, fill_value=0)

    flux_new = f_flux(wl_new)
    ivar_new = f_ivar(wl_new)

    # Handle NaNs
    flux_new = np.nan_to_num(flux_new, nan=0.0)
    ivar_new = np.nan_to_num(ivar_new, nan=0.0)

    return wl_new, flux_new, ivar_new

def gaussian(x, amp, center, sigma, offset):
    """Gaussian function for CCF peak fitting."""
    return amp * np.exp(-0.5 * ((x - center) / sigma)**2) + offset

def compute_ccf(template_flux, target_flux, target_ivar, v_range, v_step, wl_log=None):
    """
    Compute cross-correlation function between template and target.

    Uses FFT-based cross-correlation for efficiency and accuracy.

    Returns:
        velocities: array of velocity shifts
        ccf: cross-correlation values
    """
    n = len(template_flux)

    # Velocity array
    velocities = np.arange(v_range[0], v_range[1] + v_step, v_step)
    ccf = np.zeros(len(velocities))

    # Get valid pixels
    good_t = np.isfinite(template_flux) & (template_flux != 0)
    good_s = (target_ivar > 0) & np.isfinite(target_flux)
    good = good_t & good_s

    if np.sum(good) < 100:
        return velocities, ccf

    # Normalize spectra
    template_norm = template_flux.copy()
    target_norm = target_flux.copy()

    # Subtract mean and normalize
    template_norm[good] = (template_flux[good] - np.mean(template_flux[good]))
    target_norm[good] = (target_flux[good] - np.mean(target_flux[good]))

    # Set bad pixels to 0
    template_norm[~good] = 0
    target_norm[~good] = 0

    # Weights from ivar
    weights = np.sqrt(target_ivar)
    weights[~good] = 0

    # For log-wavelength grid, velocity shift = constant * pixel shift
    # v/c = d(ln λ) = (ln λ_max - ln λ_min) / n_pixels * Δpix
    # So: Δpix = v/c * n_pixels / (ln λ_max - ln λ_min)

    if wl_log is not None:
        ln_range = np.log(wl_log[-1]) - np.log(wl_log[0])
    else:
        ln_range = np.log(8100) - np.log(6400)  # default

    pix_per_kms = n / (ln_range * C_KMS)

    # Compute CCF via direct shift and correlation
    for i, v in enumerate(velocities):
        shift_pix = v * pix_per_kms

        # Interpolate shifted template
        x_orig = np.arange(n)
        x_shifted = x_orig - shift_pix  # Shift template to the right for positive v

        # Linear interpolation
        template_shifted = np.interp(x_orig, x_shifted, template_norm, left=0, right=0)

        # Weighted cross-correlation
        w = weights * weights  # both template and target weights
        num = np.sum(w * template_shifted * target_norm)
        denom1 = np.sqrt(np.sum(w * template_shifted**2))
        denom2 = np.sqrt(np.sum(w * target_norm**2))

        if denom1 > 0 and denom2 > 0:
            ccf[i] = num / (denom1 * denom2)

    return velocities, ccf

def fit_ccf_peak(velocities, ccf, n_peaks=3):
    """
    Fit CCF peaks and return peak parameters.

    Returns:
        peaks: list of dicts with 'velocity', 'height', 'fwhm', 'fit_quality'
    """
    results = []

    # Find peaks
    ccf_smooth = gaussian_filter1d(ccf, sigma=2)
    peak_indices, properties = find_peaks(ccf_smooth, height=0.1, distance=20)

    if len(peak_indices) == 0:
        return results

    # Sort by height
    heights = ccf_smooth[peak_indices]
    sorted_idx = np.argsort(heights)[::-1]
    peak_indices = peak_indices[sorted_idx]

    for i, peak_idx in enumerate(peak_indices[:n_peaks]):
        v_peak = velocities[peak_idx]
        h_peak = ccf[peak_idx]

        # Fit Gaussian around peak
        idx_range = 30  # pixels
        idx_min = max(0, peak_idx - idx_range)
        idx_max = min(len(velocities), peak_idx + idx_range)

        v_fit = velocities[idx_min:idx_max]
        ccf_fit = ccf[idx_min:idx_max]

        try:
            popt, pcov = curve_fit(
                gaussian, v_fit, ccf_fit,
                p0=[h_peak, v_peak, 20, 0],
                bounds=([0, v_peak-50, 1, -1], [2, v_peak+50, 200, 1]),
                maxfev=1000
            )

            amp, center, sigma, offset = popt
            fwhm = 2.355 * sigma

            # Estimate error from covariance
            try:
                v_err = np.sqrt(pcov[1, 1])
            except:
                v_err = 5.0  # default

            results.append({
                'velocity': center,
                'velocity_err': v_err,
                'height': amp + offset,
                'fwhm': fwhm,
                'fit_quality': 'GOOD' if fwhm < 150 else 'BROAD'
            })
        except Exception as e:
            results.append({
                'velocity': v_peak,
                'velocity_err': 10.0,
                'height': h_peak,
                'fwhm': np.nan,
                'fit_quality': 'FAILED'
            })

    return results

# =============================================================================
# SPECTRUM LOADING
# =============================================================================

def load_epoch_spectrum(epoch_info):
    """
    Load and preprocess DESI spectrum for an epoch.

    Returns dict with wavelength, flux, ivar for each band and combined.
    """
    fpath = SPECTRA_DIR / epoch_info['coadd_file']

    if not fpath.exists():
        raise FileNotFoundError(f"Coadd file not found: {fpath}")

    result = {
        'file': str(fpath),
        'file_hash': compute_file_hash(fpath),
        'mjd': epoch_info['mjd'],
        'rv_catalog': epoch_info['rv_catalog'],
        'label': epoch_info['label'],
    }

    with fits.open(fpath, memmap=True) as hdu:
        fmap = hdu['FIBERMAP'].data
        mask = fmap['TARGETID'] == TARGET_ID

        if not np.any(mask):
            raise ValueError(f"Target not found in {fpath}")

        idx = np.where(mask)[0][0]
        result['fiber_idx'] = int(idx)

        # Load each band
        bands = {}
        for band in ['B', 'R', 'Z']:
            wl = hdu[f'{band}_WAVELENGTH'].data
            fl = hdu[f'{band}_FLUX'].data[idx]
            iv = hdu[f'{band}_IVAR'].data[idx]

            # Apply telluric mask
            iv = apply_telluric_mask(wl, fl, iv)

            # Continuum normalize
            fl_norm, iv_norm = continuum_normalize(wl, fl, iv)

            bands[band] = {
                'wavelength': wl,
                'flux': fl,
                'flux_norm': fl_norm,
                'ivar': iv,
                'ivar_norm': iv_norm,
            }

        result['bands'] = bands

        # Create combined R+Z spectrum for red analysis
        wl_r = bands['R']['wavelength']
        wl_z = bands['Z']['wavelength']
        fl_r = bands['R']['flux_norm']
        fl_z = bands['Z']['flux_norm']
        iv_r = bands['R']['ivar_norm']
        iv_z = bands['Z']['ivar_norm']

        # Combine where they overlap
        combined_wl = np.concatenate([wl_r[wl_r < 7520], wl_z])
        combined_fl = np.concatenate([fl_r[wl_r < 7520], fl_z])
        combined_iv = np.concatenate([iv_r[wl_r < 7520], iv_z])

        result['combined'] = {
            'wavelength': combined_wl,
            'flux_norm': combined_fl,
            'ivar_norm': combined_iv,
        }

    return result

# =============================================================================
# SINGLE-TEMPLATE RV ANALYSIS
# =============================================================================

def analyze_single_template_rv(spectra):
    """
    Perform single-template RV analysis using self-template approach.
    Uses highest-SNR epoch as template.

    Returns dict with per-epoch, per-mask RV results.
    """
    print("\n" + "="*60)
    print("PART 1: SINGLE-TEMPLATE RV ANALYSIS")
    print("="*60)

    results = {
        'method': 'Self-template cross-correlation',
        'epochs': [],
        'masks': list(WAVELENGTH_MASKS.keys()),
    }

    # Use Epoch 1 (highest SNR, most extreme RV) as template
    template_spec = spectra[0]
    print(f"\nUsing {template_spec['label']} as template")

    for i, spec in enumerate(spectra):
        print(f"\n--- {spec['label']} ---")
        print(f"  Catalog RV: {spec['rv_catalog']:.2f} km/s")

        epoch_results = {
            'label': spec['label'],
            'mjd': spec['mjd'],
            'rv_catalog': spec['rv_catalog'],
            'masks': {},
        }

        for mask_name, (wl_min, wl_max) in WAVELENGTH_MASKS.items():
            # Extract wavelength region from combined spectrum
            wl = spec['combined']['wavelength']
            fl = spec['combined']['flux_norm']
            iv = spec['combined']['ivar_norm']

            in_mask = (wl >= wl_min) & (wl <= wl_max)

            # Also extract template
            wl_t = template_spec['combined']['wavelength']
            fl_t = template_spec['combined']['flux_norm']
            iv_t = template_spec['combined']['ivar_norm']

            in_mask_t = (wl_t >= wl_min) & (wl_t <= wl_max)

            # Resample to common log-wavelength grid
            wl_log, fl_log, iv_log = resample_to_log_wavelength(
                wl[in_mask], fl[in_mask], iv[in_mask], wl_min, wl_max
            )
            wl_t_log, fl_t_log, iv_t_log = resample_to_log_wavelength(
                wl_t[in_mask_t], fl_t[in_mask_t], iv_t[in_mask_t], wl_min, wl_max
            )

            # Compute CCF
            velocities, ccf = compute_ccf(
                fl_t_log, fl_log, iv_log, CCF_V_RANGE, CCF_V_STEP, wl_log
            )

            # Fit peaks
            peaks = fit_ccf_peak(velocities, ccf)

            if peaks:
                primary = peaks[0]
                rv_measured = primary['velocity']
                rv_err = primary['velocity_err']
                fwhm = primary['fwhm']
                fit_quality = primary['fit_quality']

                # For self-template, RV is relative to template epoch
                # Add template's catalog RV to get absolute RV
                # But since template is epoch 0, relative RV = RV difference from template

                print(f"  {mask_name}: ΔRV = {rv_measured:+.1f} ± {rv_err:.1f} km/s, FWHM = {fwhm:.0f} km/s [{fit_quality}]")
            else:
                rv_measured = np.nan
                rv_err = np.nan
                fwhm = np.nan
                fit_quality = 'NO_PEAK'
                print(f"  {mask_name}: NO PEAK DETECTED")

            epoch_results['masks'][mask_name] = {
                'rv_relative': rv_measured,
                'rv_err': rv_err,
                'fwhm': fwhm,
                'fit_quality': fit_quality,
                'ccf': ccf.tolist(),
                'velocities': velocities.tolist(),
                'peaks': peaks,
            }

        # Compute mask-to-mask consistency
        rvs = [epoch_results['masks'][m]['rv_relative'] for m in WAVELENGTH_MASKS.keys()]
        rvs = [r for r in rvs if np.isfinite(r)]
        if len(rvs) >= 2:
            rv_scatter = np.std(rvs)
            rv_mean = np.mean(rvs)
            epoch_results['rv_mean_relative'] = rv_mean
            epoch_results['rv_scatter'] = rv_scatter
            epoch_results['mask_consistency'] = 'GOOD' if rv_scatter < 10 else 'POOR'
            print(f"  Mean ΔRV: {rv_mean:+.1f} km/s, Scatter: {rv_scatter:.1f} km/s [{epoch_results['mask_consistency']}]")
        else:
            epoch_results['rv_mean_relative'] = np.nan
            epoch_results['rv_scatter'] = np.nan
            epoch_results['mask_consistency'] = 'INSUFFICIENT_DATA'

        results['epochs'].append(epoch_results)

    # Summary: compute epoch-to-epoch RV swing from our measurements
    print("\n--- RV SWING SUMMARY ---")
    rvs_measured = [e['rv_mean_relative'] for e in results['epochs'] if np.isfinite(e.get('rv_mean_relative', np.nan))]
    if len(rvs_measured) >= 2:
        rv_swing = max(rvs_measured) - min(rvs_measured)
        print(f"Measured ΔRV swing (relative): {rv_swing:.1f} km/s")
        results['rv_swing_measured'] = rv_swing

    # Compare to catalog
    rvs_catalog = [e['rv_catalog'] for e in EPOCHS]
    rv_swing_catalog = max(rvs_catalog) - min(rvs_catalog)
    print(f"Catalog ΔRV swing: {rv_swing_catalog:.1f} km/s")
    results['rv_swing_catalog'] = rv_swing_catalog

    return results

# =============================================================================
# CCF PEAK MULTIPLICITY ANALYSIS
# =============================================================================

def analyze_ccf_multiplicity(spectra, single_template_results):
    """
    Analyze CCF peak structure for signs of multi-component/switching.
    """
    print("\n" + "="*60)
    print("PART 2: CCF PEAK MULTIPLICITY ANALYSIS")
    print("="*60)

    results = {
        'epochs': [],
        'switching_detected': False,
        'multipeak_epochs': [],
    }

    for i, (spec, st_result) in enumerate(zip(spectra, single_template_results['epochs'])):
        print(f"\n--- {spec['label']} ---")

        epoch_result = {
            'label': spec['label'],
            'mjd': spec['mjd'],
            'masks': {},
            'multipeak': False,
            'switching': False,
        }

        for mask_name in WAVELENGTH_MASKS.keys():
            mask_data = st_result['masks'][mask_name]
            peaks = mask_data['peaks']

            if len(peaks) >= 2:
                p1 = peaks[0]
                p2 = peaks[1]

                height_ratio = p2['height'] / p1['height'] if p1['height'] > 0 else 0
                v_separation = abs(p2['velocity'] - p1['velocity'])

                multipeak = (height_ratio > 0.3) and (v_separation > 20)

                epoch_result['masks'][mask_name] = {
                    'n_peaks': len(peaks),
                    'peak1_v': p1['velocity'],
                    'peak2_v': p2['velocity'],
                    'height_ratio': height_ratio,
                    'v_separation': v_separation,
                    'multipeak': multipeak,
                }

                if multipeak:
                    epoch_result['multipeak'] = True
                    print(f"  {mask_name}: MULTIPEAK - v1={p1['velocity']:.1f}, v2={p2['velocity']:.1f}, ratio={height_ratio:.2f}")
                else:
                    print(f"  {mask_name}: single peak at v={p1['velocity']:.1f} km/s")
            else:
                epoch_result['masks'][mask_name] = {
                    'n_peaks': len(peaks),
                    'multipeak': False,
                }
                if peaks:
                    print(f"  {mask_name}: single peak at v={peaks[0]['velocity']:.1f} km/s")
                else:
                    print(f"  {mask_name}: no clear peak")

        if epoch_result['multipeak']:
            results['multipeak_epochs'].append(spec['label'])

        results['epochs'].append(epoch_result)

    # Check for switching between epochs
    print("\n--- COMPONENT SWITCHING CHECK ---")
    primary_vs = []
    for e in results['epochs']:
        vs = []
        for m, data in e['masks'].items():
            if 'peak1_v' in data:
                vs.append(data['peak1_v'])
        if vs:
            primary_vs.append(np.mean(vs))

    if len(primary_vs) >= 2:
        v_changes = np.diff(primary_vs)
        max_change = np.max(np.abs(v_changes))
        print(f"Primary peak velocity changes: {v_changes}")
        print(f"Maximum velocity change: {max_change:.1f} km/s")

        # Check if large changes could be explained by switching to secondary
        if max_change > 50:
            print("  Large velocity change detected - checking for component switching...")
            results['switching_detected'] = True

    results['verdict'] = 'MULTIPEAK_DETECTED' if results['multipeak_epochs'] else 'SINGLE_PEAK'
    print(f"\nVerdict: {results['verdict']}")

    return results

# =============================================================================
# WAVELENGTH-SPLIT RV TEST
# =============================================================================

def analyze_wavelength_split(spectra):
    """
    Test for wavelength-dependent RV (signature of blending).
    """
    print("\n" + "="*60)
    print("PART 3: WAVELENGTH-SPLIT RV TEST")
    print("="*60)

    # Define blue and red regions
    blue_region = (5800, 6400)  # B-R overlap region
    red_region = (7400, 8800)   # Far red Z-band

    results = {
        'blue_region': blue_region,
        'red_region': red_region,
        'epochs': [],
    }

    # Use highest-SNR epoch as template
    template_spec = spectra[0]

    for spec in spectra:
        print(f"\n--- {spec['label']} ---")

        epoch_result = {
            'label': spec['label'],
            'mjd': spec['mjd'],
        }

        # Blue region RV
        wl = spec['combined']['wavelength']
        fl = spec['combined']['flux_norm']
        iv = spec['combined']['ivar_norm']

        wl_t = template_spec['combined']['wavelength']
        fl_t = template_spec['combined']['flux_norm']
        iv_t = template_spec['combined']['ivar_norm']

        for region_name, (wl_min, wl_max) in [('blue', blue_region), ('red', red_region)]:
            in_mask = (wl >= wl_min) & (wl <= wl_max)
            in_mask_t = (wl_t >= wl_min) & (wl_t <= wl_max)

            n_valid = np.sum((iv[in_mask] > 0) & np.isfinite(fl[in_mask]))
            if n_valid < 100:
                print(f"  {region_name}: insufficient data ({n_valid} pixels)")
                epoch_result[f'rv_{region_name}'] = np.nan
                continue

            wl_log, fl_log, iv_log = resample_to_log_wavelength(
                wl[in_mask], fl[in_mask], iv[in_mask], wl_min, wl_max
            )
            wl_t_log, fl_t_log, iv_t_log = resample_to_log_wavelength(
                wl_t[in_mask_t], fl_t[in_mask_t], iv_t[in_mask_t], wl_min, wl_max
            )

            velocities, ccf = compute_ccf(fl_t_log, fl_log, iv_log, CCF_V_RANGE, CCF_V_STEP, wl_log)
            peaks = fit_ccf_peak(velocities, ccf)

            if peaks:
                rv = peaks[0]['velocity']
                epoch_result[f'rv_{region_name}'] = rv
                print(f"  {region_name} ({wl_min}-{wl_max} Å): ΔRV = {rv:+.1f} km/s")
            else:
                epoch_result[f'rv_{region_name}'] = np.nan
                print(f"  {region_name}: no peak")

        # Compute split
        if np.isfinite(epoch_result.get('rv_blue', np.nan)) and np.isfinite(epoch_result.get('rv_red', np.nan)):
            split = epoch_result['rv_red'] - epoch_result['rv_blue']
            epoch_result['rv_split'] = split
            print(f"  ΔRV_split (red - blue): {split:+.1f} km/s")
        else:
            epoch_result['rv_split'] = np.nan

        results['epochs'].append(epoch_result)

    # Summary
    print("\n--- WAVELENGTH SPLIT SUMMARY ---")
    splits = [e['rv_split'] for e in results['epochs'] if np.isfinite(e.get('rv_split', np.nan))]
    if len(splits) >= 2:
        split_mean = np.mean(splits)
        split_std = np.std(splits)
        print(f"Mean split: {split_mean:+.1f} km/s")
        print(f"Split scatter: {split_std:.1f} km/s")

        # Flag if split is large and inconsistent
        if np.abs(split_mean) > 10:
            results['verdict'] = 'LARGE_SPLIT'
            print("WARNING: Large wavelength-dependent RV detected")
        elif split_std > 10:
            results['verdict'] = 'INCONSISTENT_SPLIT'
            print("WARNING: Inconsistent splits across epochs")
        else:
            results['verdict'] = 'CONSISTENT'
            print("Wavelength split is small and consistent")

        results['split_mean'] = split_mean
        results['split_std'] = split_std
    else:
        results['verdict'] = 'INSUFFICIENT_DATA'

    return results

# =============================================================================
# TWO-TEMPLATE BLEND FIT
# =============================================================================

def two_template_fit(spectra):
    """
    Test whether a two-component spectral model is preferred.
    """
    print("\n" + "="*60)
    print("PART 4: TWO-TEMPLATE BLEND FIT")
    print("="*60)

    results = {
        'method': 'Two-template least squares fit',
        'neighbor_flux_fraction_expected': NEIGHBOR_FLUX_FRAC,
        'epochs': [],
    }

    # For two-template fit, we use the extreme epochs (epoch 0 and epoch 1)
    # as potential templates representing primary at different phases
    # and look for a secondary component

    # Actually, for a proper blend test, we should use:
    # Template 1: M0 dwarf (primary)
    # Template 2: later-type dwarf (neighbor)
    # But we don't have external templates, so we use self-template approach:
    # Test if the spectrum is better fit as a single component or as
    # a sum of two components with different velocity shifts

    print("\nUsing self-template approach: fit each epoch as single vs two-component model")
    print(f"Expected neighbor flux fraction: {NEIGHBOR_FLUX_FRAC:.3f}")

    # Use the combined epoch (highest SNR) as the base template
    template = spectra[0]

    wl_min, wl_max = 6400, 8000  # Combined red region

    for spec in spectra:
        print(f"\n--- {spec['label']} ---")

        # Extract data
        wl = spec['combined']['wavelength']
        fl = spec['combined']['flux_norm']
        iv = spec['combined']['ivar_norm']

        in_mask = (wl >= wl_min) & (wl <= wl_max) & (iv > 0)
        wl_data = wl[in_mask]
        fl_data = fl[in_mask]
        iv_data = iv[in_mask]
        n_pix = len(wl_data)

        # Get template
        wl_t = template['combined']['wavelength']
        fl_t = template['combined']['flux_norm']
        iv_t = template['combined']['ivar_norm']

        in_mask_t = (wl_t >= wl_min) & (wl_t <= wl_max) & (iv_t > 0)

        # Interpolate template to data wavelengths
        f_template = interp1d(wl_t[in_mask_t], fl_t[in_mask_t],
                              bounds_error=False, fill_value=1.0)
        template_on_data = f_template(wl_data)

        # Single-template fit: F = a * T(v1)
        def single_model(params):
            a, v1 = params
            # Shift wavelengths
            wl_shifted = wl_data * (1 - v1 / C_KMS)
            t_shifted = f_template(wl_shifted)
            model = a * t_shifted
            residual = (fl_data - model) * np.sqrt(iv_data)
            return np.sum(residual**2)

        # Two-template fit: F = a * T(v1) + b * T(v2)
        def two_model(params):
            a, v1, b, v2 = params
            wl_shifted1 = wl_data * (1 - v1 / C_KMS)
            wl_shifted2 = wl_data * (1 - v2 / C_KMS)
            t1 = f_template(wl_shifted1)
            t2 = f_template(wl_shifted2)
            model = a * t1 + b * t2
            residual = (fl_data - model) * np.sqrt(iv_data)
            return np.sum(residual**2)

        # Fit single template
        try:
            result_single = minimize(single_model, [1.0, 0.0],
                                     bounds=[(0.5, 1.5), (-200, 200)],
                                     method='L-BFGS-B')
            chi2_single = result_single.fun
            a_single, v1_single = result_single.x
            print(f"  Single: a={a_single:.3f}, v1={v1_single:+.1f} km/s, χ²={chi2_single:.0f}")
        except Exception as e:
            print(f"  Single fit failed: {e}")
            chi2_single = np.inf
            v1_single = 0

        # Fit two templates
        try:
            # Try multiple starting points
            best_chi2_two = np.inf
            best_params = None

            for v2_init in [-100, 0, 100]:
                for b_init in [0.05, 0.13, 0.3]:
                    try:
                        result = minimize(two_model, [0.87, v1_single, b_init, v2_init],
                                         bounds=[(0.5, 1.0), (-200, 200), (0.01, 0.5), (-200, 200)],
                                         method='L-BFGS-B')
                        if result.fun < best_chi2_two:
                            best_chi2_two = result.fun
                            best_params = result.x
                    except:
                        pass

            if best_params is not None:
                chi2_two = best_chi2_two
                a_two, v1_two, b_two, v2_two = best_params
                print(f"  Two: a={a_two:.3f}, v1={v1_two:+.1f}, b={b_two:.3f}, v2={v2_two:+.1f} km/s, χ²={chi2_two:.0f}")
            else:
                chi2_two = np.inf
                a_two = v1_two = b_two = v2_two = np.nan
        except Exception as e:
            print(f"  Two-template fit failed: {e}")
            chi2_two = np.inf
            a_two = v1_two = b_two = v2_two = np.nan

        # Compute BIC
        k_single = 2  # a, v1
        k_two = 4     # a, v1, b, v2

        bic_single = chi2_single + k_single * np.log(n_pix)
        bic_two = chi2_two + k_two * np.log(n_pix)
        delta_bic = bic_two - bic_single

        print(f"  ΔBIC = {delta_bic:.1f} (negative favors two-template)")

        epoch_result = {
            'label': spec['label'],
            'mjd': spec['mjd'],
            'n_pixels': n_pix,
            'single': {
                'a': float(a_single) if np.isfinite(chi2_single) else None,
                'v1': float(v1_single) if np.isfinite(chi2_single) else None,
                'chi2': float(chi2_single),
                'bic': float(bic_single),
            },
            'two': {
                'a': float(a_two) if np.isfinite(a_two) else None,
                'v1': float(v1_two) if np.isfinite(v1_two) else None,
                'b': float(b_two) if np.isfinite(b_two) else None,
                'v2': float(v2_two) if np.isfinite(v2_two) else None,
                'chi2': float(chi2_two),
                'bic': float(bic_two),
            },
            'delta_bic': float(delta_bic),
            'prefers_two': delta_bic < -10,
        }

        results['epochs'].append(epoch_result)

    # Summary
    print("\n--- TWO-TEMPLATE FIT SUMMARY ---")
    prefers_two = [e for e in results['epochs'] if e['prefers_two']]
    print(f"Epochs preferring two-template: {len(prefers_two)}/{len(results['epochs'])}")

    if prefers_two:
        for e in prefers_two:
            print(f"  {e['label']}: ΔBIC = {e['delta_bic']:.1f}")
        results['verdict'] = 'TWO_TEMPLATE_PREFERRED'
    else:
        results['verdict'] = 'SINGLE_TEMPLATE_SUFFICIENT'
        print("Single-template model is sufficient for all epochs")

    return results

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("="*60)
    print("DESI BLEND-AWARE RV RE-MEASUREMENT")
    print("="*60)
    print(f"\nTarget: Gaia DR3 {GAIA_SOURCE_ID}")
    print(f"DESI TARGETID: {TARGET_ID}")
    print(f"\nConfirmed neighbor: sep={NEIGHBOR_SEP_ARCSEC:.3f}\", ΔG={NEIGHBOR_DELTA_G:.2f} mag")
    print(f"Expected flux contamination: {NEIGHBOR_FLUX_FRAC:.1%}")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load spectra
    print("\n" + "="*60)
    print("LOADING SPECTRA")
    print("="*60)

    spectra = []
    file_hashes = {}

    for epoch in EPOCHS:
        print(f"\nLoading {epoch['label']}...")
        try:
            spec = load_epoch_spectrum(epoch)
            spectra.append(spec)
            file_hashes[spec['file']] = spec['file_hash']
            print(f"  File: {spec['file']}")
            print(f"  Hash: {spec['file_hash'][:16]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            return

    print(f"\nLoaded {len(spectra)} epoch spectra")

    # Run analyses
    single_template_results = analyze_single_template_rv(spectra)
    ccf_multiplicity_results = analyze_ccf_multiplicity(spectra, single_template_results)
    wavelength_split_results = analyze_wavelength_split(spectra)
    two_template_results = two_template_fit(spectra)

    # =================================================================
    # GENERATE FIGURES
    # =================================================================
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)

    # Figure 1: Epoch spectra overlays
    print("\nGenerating epoch_spectra_overlays.png...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = ['C0', 'C1', 'C2']
    for i, spec in enumerate(spectra):
        wl = spec['combined']['wavelength']
        fl = spec['combined']['flux_norm']
        iv = spec['combined']['ivar_norm']
        good = iv > 0

        label = f"{spec['label']} (RV={spec['rv_catalog']:+.0f})"
        axes[0].plot(wl[good], fl[good], alpha=0.7, label=label, color=colors[i])

    axes[0].set_xlabel('Wavelength (Å)')
    axes[0].set_ylabel('Normalized Flux')
    axes[0].set_title('DESI Epoch Spectra (Continuum Normalized)')
    axes[0].legend()
    axes[0].set_xlim(6000, 8500)
    axes[0].set_ylim(0.5, 1.5)

    # Zoom on TiO region
    for i, spec in enumerate(spectra):
        wl = spec['combined']['wavelength']
        fl = spec['combined']['flux_norm']
        iv = spec['combined']['ivar_norm']
        good = iv > 0

        axes[1].plot(wl[good], fl[good], alpha=0.7, color=colors[i])

    axes[1].set_xlabel('Wavelength (Å)')
    axes[1].set_ylabel('Normalized Flux')
    axes[1].set_title('Zoom: TiO Region (6400-7000 Å)')
    axes[1].set_xlim(6400, 7000)
    axes[1].set_ylim(0.3, 1.3)

    # Mark telluric regions
    for ax in axes:
        for wl_min, wl_max in TELLURIC_REGIONS:
            ax.axvspan(wl_min, wl_max, alpha=0.1, color='gray')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'epoch_spectra_overlays.png', dpi=150)
    plt.close()

    # Figure 2: CCF peaks by epoch
    print("Generating ccf_peaks_by_epoch.png...")
    fig, axes = plt.subplots(len(spectra), len(WAVELENGTH_MASKS), figsize=(16, 12))

    for i, epoch_result in enumerate(single_template_results['epochs']):
        for j, mask_name in enumerate(WAVELENGTH_MASKS.keys()):
            ax = axes[i, j] if len(spectra) > 1 else axes[j]

            mask_data = epoch_result['masks'][mask_name]
            v = np.array(mask_data['velocities'])
            ccf = np.array(mask_data['ccf'])

            ax.plot(v, ccf, 'b-', lw=0.5)

            # Mark peaks
            for k, peak in enumerate(mask_data['peaks'][:3]):
                color = 'r' if k == 0 else 'orange'
                ax.axvline(peak['velocity'], color=color, ls='--', alpha=0.7)
                ax.text(peak['velocity'], ax.get_ylim()[1]*0.9,
                       f"v={peak['velocity']:.0f}", fontsize=7, ha='center')

            if i == 0:
                ax.set_title(mask_name)
            if j == 0:
                ax.set_ylabel(epoch_result['label'])
            if i == len(spectra) - 1:
                ax.set_xlabel('Velocity (km/s)')

    plt.suptitle('CCF Peaks by Epoch and Wavelength Mask')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ccf_peaks_by_epoch.png', dpi=150)
    plt.close()

    # Figure 3: RV by method comparison
    print("Generating rv_by_method.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_labels = [e['label'] for e in single_template_results['epochs']]
    x = np.arange(len(epochs_labels))
    width = 0.15

    # Catalog RVs
    catalog_rvs = [e['rv_catalog'] for e in single_template_results['epochs']]
    ax.bar(x - 2*width, catalog_rvs, width, label='Catalog', alpha=0.8)

    # Per-mask RVs (add template RV to get absolute scale for relative measurements)
    template_rv = catalog_rvs[0]  # First epoch is template
    for i, mask_name in enumerate(WAVELENGTH_MASKS.keys()):
        rvs = []
        for e in single_template_results['epochs']:
            rv_rel = e['masks'][mask_name]['rv_relative']
            # Convert relative to "absolute" by noting template is at catalog RV
            # For epoch 0 (template), relative RV = 0
            # For other epochs, absolute = template_rv + relative
            if np.isfinite(rv_rel):
                rvs.append(template_rv + rv_rel)
            else:
                rvs.append(np.nan)
        ax.bar(x + (i-1)*width, rvs, width, label=mask_name, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([e.split()[0] for e in epochs_labels])
    ax.set_ylabel('RV (km/s)')
    ax.set_title('RV Measurements by Method')
    ax.legend(loc='upper right')
    ax.axhline(0, color='k', ls='-', lw=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rv_by_method.png', dpi=150)
    plt.close()

    # Figure 4: Wavelength split RV
    print("Generating wavelength_split_rv.png...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs_labels = [e['label'] for e in wavelength_split_results['epochs']]
    x = np.arange(len(epochs_labels))

    # Blue vs Red RV
    blue_rvs = [e.get('rv_blue', np.nan) for e in wavelength_split_results['epochs']]
    red_rvs = [e.get('rv_red', np.nan) for e in wavelength_split_results['epochs']]

    ax = axes[0]
    ax.bar(x - 0.2, blue_rvs, 0.4, label='Blue (5800-6400 Å)')
    ax.bar(x + 0.2, red_rvs, 0.4, label='Red (7400-8800 Å)')
    ax.set_xticks(x)
    ax.set_xticklabels([e.split()[0] for e in epochs_labels])
    ax.set_ylabel('Relative RV (km/s)')
    ax.set_title('RV by Wavelength Region')
    ax.legend()

    # Split
    ax = axes[1]
    splits = [e.get('rv_split', np.nan) for e in wavelength_split_results['epochs']]
    ax.bar(x, splits, color='purple')
    ax.set_xticks(x)
    ax.set_xticklabels([e.split()[0] for e in epochs_labels])
    ax.set_ylabel('ΔRV (Red - Blue) km/s')
    ax.set_title('Wavelength-Dependent RV Split')
    ax.axhline(0, color='k', ls='--')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'wavelength_split_rv.png', dpi=150)
    plt.close()

    # Figure 5: Two-template fit residuals
    print("Generating two_template_fit_residuals.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Show extreme epoch (epoch 0) and normal epoch (epoch 1)
    for idx, (spec, tt_result) in enumerate([(spectra[0], two_template_results['epochs'][0]),
                                              (spectra[1], two_template_results['epochs'][1])]):
        ax_spec = axes[idx, 0]
        ax_resid = axes[idx, 1]

        wl = spec['combined']['wavelength']
        fl = spec['combined']['flux_norm']
        iv = spec['combined']['ivar_norm']

        wl_min, wl_max = 6400, 8000
        in_mask = (wl >= wl_min) & (wl <= wl_max) & (iv > 0)

        ax_spec.plot(wl[in_mask], fl[in_mask], 'k-', lw=0.5, alpha=0.7, label='Data')
        ax_spec.set_title(f"{spec['label']} (Catalog RV: {spec['rv_catalog']:+.0f} km/s)")
        ax_spec.set_xlabel('Wavelength (Å)')
        ax_spec.set_ylabel('Normalized Flux')
        ax_spec.set_ylim(0.3, 1.3)
        ax_spec.legend()

        # Show ΔBIC
        ax_resid.text(0.5, 0.5, f"ΔBIC = {tt_result['delta_bic']:.1f}\n" +
                     (f"Prefers two-template" if tt_result['prefers_two'] else "Single template sufficient"),
                     ha='center', va='center', fontsize=14, transform=ax_resid.transAxes)
        ax_resid.set_title('Model Comparison')
        ax_resid.axis('off')

    plt.suptitle('Two-Template Fit Analysis')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'two_template_fit_residuals.png', dpi=150)
    plt.close()

    print("All figures generated.")

    # =================================================================
    # FINAL VERDICT
    # =================================================================
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)

    # Compile verdicts
    verdicts = {}

    # 1. Single-template stability
    scatters = [e['rv_scatter'] for e in single_template_results['epochs']
                if np.isfinite(e.get('rv_scatter', np.nan))]
    if scatters:
        max_scatter = max(scatters)
        if max_scatter < 5:
            verdicts['single_template_stability'] = 'PASS'
        elif max_scatter < 15:
            verdicts['single_template_stability'] = 'INCONCLUSIVE'
        else:
            verdicts['single_template_stability'] = 'FAIL'
    else:
        verdicts['single_template_stability'] = 'INSUFFICIENT_DATA'

    # 2. CCF multiplicity
    if ccf_multiplicity_results['multipeak_epochs']:
        verdicts['ccf_multiplicity'] = 'FAIL'
    elif ccf_multiplicity_results['switching_detected']:
        verdicts['ccf_multiplicity'] = 'INCONCLUSIVE'
    else:
        verdicts['ccf_multiplicity'] = 'PASS'

    # 3. Wavelength split
    if wavelength_split_results['verdict'] == 'CONSISTENT':
        verdicts['wavelength_split'] = 'PASS'
    elif wavelength_split_results['verdict'] == 'INSUFFICIENT_DATA':
        verdicts['wavelength_split'] = 'INCONCLUSIVE'
    else:
        verdicts['wavelength_split'] = 'FAIL'

    # 4. Two-template preference
    if two_template_results['verdict'] == 'SINGLE_TEMPLATE_SUFFICIENT':
        verdicts['two_template_preference'] = 'PASS'
    else:
        verdicts['two_template_preference'] = 'FAIL'

    # 5. Can blending explain 146 km/s?
    # Physics argument: 13% blend can shift RV by at most ~13 km/s
    # The observed swing is 146 km/s
    max_blend_effect = NEIGHBOR_FLUX_FRAC * 100  # ~13 km/s assuming 100 km/s neighbor offset
    observed_swing = single_template_results.get('rv_swing_catalog', 146)

    if observed_swing > max_blend_effect * 5:
        verdicts['blend_explains_amplitude'] = 'PASS'  # Blending cannot explain
        blend_explanation = f"Blending (max ~{max_blend_effect:.0f} km/s effect) cannot explain {observed_swing:.0f} km/s swing"
    else:
        verdicts['blend_explains_amplitude'] = 'INCONCLUSIVE'
        blend_explanation = "Amplitude could potentially be affected by blending"

    # Print verdict table
    print("\n| Check | Result | Notes |")
    print("|-------|--------|-------|")
    print(f"| Single-template RV stability | **{verdicts['single_template_stability']}** | Max scatter: {max_scatter:.1f} km/s |")
    print(f"| CCF peak multiplicity | **{verdicts['ccf_multiplicity']}** | {ccf_multiplicity_results['verdict']} |")
    print(f"| Wavelength-split consistency | **{verdicts['wavelength_split']}** | {wavelength_split_results['verdict']} |")
    print(f"| Two-template preference | **{verdicts['two_template_preference']}** | {two_template_results['verdict']} |")
    print(f"| Blend explains amplitude | **{verdicts['blend_explains_amplitude']}** | {blend_explanation} |")

    # Overall verdict
    n_pass = sum(1 for v in verdicts.values() if v == 'PASS')
    n_fail = sum(1 for v in verdicts.values() if v == 'FAIL')
    n_inconclusive = sum(1 for v in verdicts.values() if v == 'INCONCLUSIVE')

    print(f"\nSummary: {n_pass} PASS, {n_fail} FAIL, {n_inconclusive} INCONCLUSIVE")

    if n_fail == 0 and n_inconclusive <= 1:
        overall_verdict = "ROBUST"
        bottom_line = "DESI RV swing remains robust under blend-aware tests; blend unlikely to explain amplitude."
    elif n_fail >= 2:
        overall_verdict = "COMPROMISED"
        bottom_line = "Evidence suggests blend effects may compromise RV measurements; candidate requires further investigation."
    else:
        overall_verdict = "INCONCLUSIVE"
        bottom_line = "Results are mixed; higher-resolution spectroscopy recommended to definitively assess blending."

    print(f"\n**OVERALL VERDICT: {overall_verdict}**")
    print(f"\n{bottom_line}")

    # =================================================================
    # SAVE OUTPUTS
    # =================================================================
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)

    # JSON: epoch RV refit
    rv_refit = {
        'target_id': TARGET_ID,
        'gaia_source_id': GAIA_SOURCE_ID,
        'file_hashes': file_hashes,
        'method': single_template_results['method'],
        'epochs': single_template_results['epochs'],
        'rv_swing_catalog': single_template_results.get('rv_swing_catalog'),
        'rv_swing_measured': single_template_results.get('rv_swing_measured'),
    }

    # Define JSON serializer
    def json_default(x):
        if isinstance(x, (np.integer, np.int64)):
            return int(x)
        if isinstance(x, (np.floating, np.float64)):
            if np.isfinite(x):
                return float(x)
            return None
        if isinstance(x, np.ndarray):
            return x.tolist()
        return None

    # Remove large CCF arrays from results to keep file size manageable
    for e in rv_refit['epochs']:
        for mask_name, mask_data in e['masks'].items():
            mask_data.pop('ccf', None)
            mask_data.pop('velocities', None)

    with open(OUTPUT_DIR / 'desi_epoch_rv_refit.json', 'w') as f:
        json.dump(rv_refit, f, indent=2, default=json_default)
    print(f"Saved: {OUTPUT_DIR / 'desi_epoch_rv_refit.json'}")

    # JSON: CCF diagnostics - create a simplified version without circular refs
    ccf_diag = {
        'target_id': TARGET_ID,
        'verdict': ccf_multiplicity_results['verdict'],
        'switching_detected': ccf_multiplicity_results['switching_detected'],
        'multipeak_epochs': ccf_multiplicity_results['multipeak_epochs'],
        'epochs': []
    }
    for e in ccf_multiplicity_results['epochs']:
        ccf_diag['epochs'].append({
            'label': e['label'],
            'mjd': e['mjd'],
            'multipeak': e['multipeak'],
            'switching': e['switching'],
            'masks': {k: {kk: vv for kk, vv in v.items() if kk != 'ccf' and kk != 'velocities'}
                     for k, v in e['masks'].items()}
        })

    with open(OUTPUT_DIR / 'desi_ccf_diagnostics.json', 'w') as f:
        json.dump(ccf_diag, f, indent=2, default=json_default)
    print(f"Saved: {OUTPUT_DIR / 'desi_ccf_diagnostics.json'}")

    # JSON: two-template fit
    with open(OUTPUT_DIR / 'desi_two_template_fit.json', 'w') as f:
        json.dump(two_template_results, f, indent=2, default=json_default)
    print(f"Saved: {OUTPUT_DIR / 'desi_two_template_fit.json'}")

    # =================================================================
    # WRITE REPORT
    # =================================================================
    print("\nWriting report...")

    report = f"""# DESI BLEND-AWARE RV RE-MEASUREMENT REPORT

**Date:** 2026-01-16
**Target:** Gaia DR3 {GAIA_SOURCE_ID}
**DESI TARGETID:** {TARGET_ID}

---

## Executive Summary

This report presents a blend-aware analysis of DESI DR1 spectra to determine whether the observed RV swing (-86 to +60 km/s) is robust to potential contamination from a confirmed close neighbor.

### Key Facts

| Property | Value |
|----------|-------|
| Neighbor separation | {NEIGHBOR_SEP_ARCSEC:.3f}" |
| Neighbor ΔG | {NEIGHBOR_DELTA_G:.2f} mag |
| Expected flux contamination | {NEIGHBOR_FLUX_FRAC:.1%} |
| DESI fiber diameter | {DESI_FIBER_DIAMETER_ARCSEC}" |
| Catalog RV swing | {observed_swing:.0f} km/s |

### Verdict Table

| Check | Result | Notes |
|-------|--------|-------|
| Single-template RV stability | **{verdicts['single_template_stability']}** | Max mask scatter: {max_scatter:.1f} km/s |
| CCF peak multiplicity | **{verdicts['ccf_multiplicity']}** | {ccf_multiplicity_results['verdict']} |
| Wavelength-split consistency | **{verdicts['wavelength_split']}** | {wavelength_split_results['verdict']} |
| Two-template preference | **{verdicts['two_template_preference']}** | {two_template_results['verdict']} |
| Blend explains amplitude | **{verdicts['blend_explains_amplitude']}** | {blend_explanation} |

**OVERALL: {overall_verdict}**

---

## Part 1: Data Provenance

### DESI Spectra Used

| Epoch | MJD | Tile | File | SHA256 |
|-------|-----|------|------|--------|
"""

    for spec in spectra:
        report += f"| {spec['label']} | {spec['mjd']:.3f} | {EPOCHS[[e['label'] for e in EPOCHS].index(spec['label'])]['tile']} | {Path(spec['file']).name} | {spec['file_hash'][:16]}... |\n"

    report += f"""
### Epoch RV Reference Values (from DESI catalog)

| Epoch | Catalog RV (km/s) | σ (km/s) |
|-------|-------------------|----------|
"""

    for e in EPOCHS:
        report += f"| {e['label']} | {e['rv_catalog']:+.2f} | {e['rv_err_catalog']:.2f} |\n"

    report += f"""
---

## Part 2: Single-Template RV Analysis

### Method

- Template: Epoch 1 spectrum (highest SNR, extreme RV)
- Cross-correlation over v ∈ [{CCF_V_RANGE[0]}, {CCF_V_RANGE[1]}] km/s
- Telluric bands masked
- Continuum normalized with running median

### Wavelength Masks

| Mask | Range (Å) |
|------|-----------|
"""

    for name, (wl_min, wl_max) in WAVELENGTH_MASKS.items():
        report += f"| {name} | {wl_min}-{wl_max} |\n"

    report += f"""
### Results by Epoch

"""

    for e in single_template_results['epochs']:
        report += f"**{e['label']}** (Catalog RV: {e['rv_catalog']:+.1f} km/s)\n\n"
        report += "| Mask | ΔRV (km/s) | Error | FWHM | Quality |\n"
        report += "|------|------------|-------|------|--------|\n"
        for mask_name, mask_data in e['masks'].items():
            rv = mask_data['rv_relative']
            err = mask_data['rv_err']
            fwhm = mask_data['fwhm']
            qual = mask_data['fit_quality']
            rv_str = f"{rv:+.1f}" if np.isfinite(rv) else "N/A"
            err_str = f"{err:.1f}" if np.isfinite(err) else "N/A"
            fwhm_str = f"{fwhm:.0f}" if np.isfinite(fwhm) else "N/A"
            report += f"| {mask_name} | {rv_str} | {err_str} | {fwhm_str} | {qual} |\n"

        scatter = e.get('rv_scatter', np.nan)
        scatter_str = f"{scatter:.1f}" if np.isfinite(scatter) else "N/A"
        report += f"\nMask-to-mask scatter: {scatter_str} km/s ({e.get('mask_consistency', 'N/A')})\n\n"

    report += f"""
### Single-Template Verdict

**{verdicts['single_template_stability']}** — The RV measurements are {'consistent' if verdicts['single_template_stability'] == 'PASS' else 'somewhat variable'} across wavelength masks.

---

## Part 3: CCF Peak Multiplicity Analysis

### Purpose

Detect whether the CCF shows multiple peaks that could indicate component switching between epochs.

### Results

"""

    for e in ccf_multiplicity_results['epochs']:
        mp_str = "MULTIPEAK" if e['multipeak'] else "single peak"
        report += f"- **{e['label']}**: {mp_str}\n"

    report += f"""
### CCF Multiplicity Verdict

**{verdicts['ccf_multiplicity']}** — {ccf_multiplicity_results['verdict']}

---

## Part 4: Wavelength-Split RV Test

### Purpose

Test for wavelength-dependent RV shifts, which are a signature of spectral blending.

### Results

| Epoch | Blue RV | Red RV | Split (R-B) |
|-------|---------|--------|-------------|
"""

    for e in wavelength_split_results['epochs']:
        blue = e.get('rv_blue', np.nan)
        red = e.get('rv_red', np.nan)
        split = e.get('rv_split', np.nan)
        blue_str = f"{blue:+.1f}" if np.isfinite(blue) else "N/A"
        red_str = f"{red:+.1f}" if np.isfinite(red) else "N/A"
        split_str = f"{split:+.1f}" if np.isfinite(split) else "N/A"
        report += f"| {e['label']} | {blue_str} | {red_str} | {split_str} |\n"

    split_mean = wavelength_split_results.get('split_mean', np.nan)
    split_std = wavelength_split_results.get('split_std', np.nan)

    report += f"""
Mean split: {split_mean:+.1f} km/s (if finite)
Split scatter: {split_std:.1f} km/s (if finite)

### Wavelength Split Verdict

**{verdicts['wavelength_split']}** — {wavelength_split_results['verdict']}

---

## Part 5: Two-Template Blend Fit

### Purpose

Test whether a two-component spectral model (primary + contaminant) is statistically preferred over a single-star model.

### Model

- Single: F(λ) = a × T(λ shifted by v₁)
- Two: F(λ) = a × T(λ shifted by v₁) + b × T(λ shifted by v₂)

Model comparison via ΔBIC = BIC_two - BIC_single (negative favors two-template).

### Results

| Epoch | χ²_single | χ²_two | ΔBIC | Prefers Two? |
|-------|-----------|--------|------|--------------|
"""

    for e in two_template_results['epochs']:
        chi_s = e['single']['chi2']
        chi_t = e['two']['chi2']
        dbic = e['delta_bic']
        pref = "YES" if e['prefers_two'] else "no"
        report += f"| {e['label']} | {chi_s:.0f} | {chi_t:.0f} | {dbic:+.1f} | {pref} |\n"

    report += f"""
### Two-Template Verdict

**{verdicts['two_template_preference']}** — {two_template_results['verdict']}

---

## Part 6: Can Blending Explain the 146 km/s Amplitude?

### Physics Argument

A blend can shift measured RV by at most:

ΔRV_max ≈ f_contaminant × v_offset

Where f_contaminant = {NEIGHBOR_FLUX_FRAC:.2f} (from ΔG = {NEIGHBOR_DELTA_G:.2f})

Even with an extreme 100 km/s offset between primary and contaminant:

ΔRV_max ≈ {NEIGHBOR_FLUX_FRAC:.2f} × 100 ≈ {NEIGHBOR_FLUX_FRAC * 100:.0f} km/s

**The observed swing is {observed_swing:.0f} km/s — {observed_swing / (NEIGHBOR_FLUX_FRAC * 100):.0f}× larger than the maximum blend effect.**

### Verdict

**{verdicts['blend_explains_amplitude']}** — {blend_explanation}

---

## Final Summary

```
╔════════════════════════════════════════════════════════════════════╗
║              DESI BLEND-AWARE ANALYSIS RESULTS                     ║
╠════════════════════════════════════════════════════════════════════╣
║ Single-template stability:    {verdicts['single_template_stability']:<12}                          ║
║ CCF peak multiplicity:        {verdicts['ccf_multiplicity']:<12}                          ║
║ Wavelength-split consistency: {verdicts['wavelength_split']:<12}                          ║
║ Two-template preference:      {verdicts['two_template_preference']:<12}                          ║
║ Blend explains amplitude:     {verdicts['blend_explains_amplitude']:<12}                          ║
╠════════════════════════════════════════════════════════════════════╣
║ OVERALL VERDICT:              {overall_verdict:<12}                          ║
╚════════════════════════════════════════════════════════════════════╝
```

### Bottom Line

**{bottom_line}**

---

## Output Files

| File | Description |
|------|-------------|
| `desi_epoch_rv_refit.json` | Per-epoch RV measurements |
| `desi_ccf_diagnostics.json` | CCF multiplicity analysis |
| `desi_two_template_fit.json` | Two-template fit results |
| `figures/epoch_spectra_overlays.png` | Epoch spectra comparison |
| `figures/ccf_peaks_by_epoch.png` | CCF structure |
| `figures/rv_by_method.png` | RV by wavelength mask |
| `figures/wavelength_split_rv.png` | Blue vs red RV |
| `figures/two_template_fit_residuals.png` | Model comparison |

---

**Report generated:** 2026-01-16
**Analysis by:** Claude Code
"""

    with open(OUTPUT_DIR / 'DESI_BLEND_AWARE_REPORT.md', 'w') as f:
        f.write(report)

    print(f"Saved: {OUTPUT_DIR / 'DESI_BLEND_AWARE_REPORT.md'}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
