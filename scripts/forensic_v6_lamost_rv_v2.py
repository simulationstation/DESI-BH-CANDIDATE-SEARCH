#!/usr/bin/env python3
"""
FORENSIC V6.1: Proper LAMOST RV Re-measurement

Fixes from v6.0:
- Telluric masking
- Robust continuum normalization with sigma-clipping
- CCF peak-fitting with FWHM sanity checks (reject >300 km/s)
- Realistic error estimation from CCF curvature
- Self-template approach: cross-correlate epoch1 vs epoch2 directly
"""

import numpy as np
import json
import hashlib
from pathlib import Path
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

C_KMS = 299792.458
FITS_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH/data/lamost")
OUTPUT_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH/outputs/forensic_v6")
FIG_DIR = OUTPUT_DIR / "figures"

OBSID_1 = 437513049
OBSID_2 = 870813030

# Telluric absorption regions to mask (Angstroms)
# Based on standard atmospheric absorption bands
TELLURIC_REGIONS = [
    (6270, 6330),   # O2 B-band
    (6860, 6960),   # O2 A-band (strong)
    (7160, 7340),   # H2O
    (7590, 7700),   # O2 A-band extended
    (8130, 8350),   # H2O
    (8940, 9200),   # H2O (strong)
]

# M-dwarf friendly wavelength regions (avoiding tellurics)
GOOD_REGIONS = [
    (6350, 6850, "TiO-6500"),      # TiO bands, avoiding O2
    (6980, 7150, "TiO-7050"),      # Between telluric bands
    (7350, 7580, "TiO-7450"),      # Strong TiO
    (7710, 8100, "TiO-7900"),      # TiO + some molecular features
    (8360, 8600, "CaII-triplet"),  # Ca II triplet region
    (6350, 7150, "Combined-Red"),  # Combined red region
]

def sha256_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            sha256.update(block)
    return sha256.hexdigest()

def load_lamost_spectrum(filepath):
    """Load LAMOST spectrum from FITS."""
    with fits.open(filepath) as hdul:
        header = hdul[0].header

        if len(hdul) > 1 and 'COADD' in [h.name for h in hdul]:
            coadd = hdul['COADD'].data
            flux = coadd['FLUX'][0]
            wavelength = coadd['WAVELENGTH'][0]
            ivar = coadd['IVAR'][0] if 'IVAR' in coadd.names else None
        else:
            raise ValueError(f"Cannot parse FITS: {filepath}")

        info = {
            'obsid': header.get('OBSID'),
            'date_obs': header.get('DATE-OBS'),
            'mjd': header.get('MJD'),
            'helio_rv': header.get('HELIO_RV'),
            'z': header.get('Z'),
            'z_err': header.get('Z_ERR'),
            'snr_g': header.get('SNRG'),
            'snr_r': header.get('SNRR'),
            'snr_i': header.get('SNRI'),
            'subclass': header.get('SUBCLASS'),
        }

    return wavelength, flux, ivar, info

def mask_tellurics(wavelength, mask=None):
    """Create mask for telluric regions."""
    if mask is None:
        mask = np.ones(len(wavelength), dtype=bool)

    for wl_min, wl_max in TELLURIC_REGIONS:
        mask &= ~((wavelength >= wl_min) & (wavelength <= wl_max))

    return mask

def robust_continuum_normalize(wavelength, flux, poly_order=4, sigma_clip=2.5, iterations=5):
    """
    Robust continuum normalization with iterative sigma-clipping.
    Uses running median + polynomial fit.
    """
    good = np.isfinite(flux) & np.isfinite(wavelength) & (flux > 0)
    if np.sum(good) < 100:
        return flux / np.nanmedian(flux), np.ones_like(flux, dtype=bool)

    wl = wavelength[good]
    fl = flux[good]

    # Running median to get continuum estimate
    window = max(50, len(fl) // 50)
    continuum_est = np.zeros_like(fl)
    for i in range(len(fl)):
        i_min = max(0, i - window)
        i_max = min(len(fl), i + window)
        continuum_est[i] = np.percentile(fl[i_min:i_max], 90)

    # Iterative polynomial fit with sigma clipping
    mask = np.ones(len(wl), dtype=bool)
    for _ in range(iterations):
        if np.sum(mask) < poly_order + 1:
            break
        coeffs = np.polyfit(wl[mask], continuum_est[mask], poly_order)
        fit = np.polyval(coeffs, wl)
        residuals = continuum_est - fit
        std = np.std(residuals[mask])
        mask = np.abs(residuals) < sigma_clip * std

    # Final continuum
    coeffs = np.polyfit(wl[mask], continuum_est[mask], poly_order)
    full_continuum = np.polyval(coeffs, wavelength)
    full_continuum[full_continuum <= 0] = np.nanmedian(flux)

    normalized = flux / full_continuum
    good_norm = np.isfinite(normalized) & (normalized > 0) & (normalized < 3)

    return normalized, good_norm

def cross_correlate_clean(wl1, fl1, wl2, fl2, v_grid, good1, good2):
    """
    Clean cross-correlation with proper normalization.

    Returns CCF normalized to [-1, 1] range.
    """
    # Find common wavelength range
    wl_min = max(wl1[good1].min(), wl2[good2].min())
    wl_max = min(wl1[good1].max(), wl2[good2].max())

    # Create log-spaced wavelength grid for uniform velocity sampling
    dv = 1.0  # km/s per pixel
    n_pix = int(np.log(wl_max / wl_min) / np.log(1 + dv / C_KMS))
    wl_common = np.geomspace(wl_min, wl_max, n_pix)

    # Interpolate spectra
    interp1 = interp1d(wl1, fl1, kind='linear', bounds_error=False, fill_value=np.nan)
    interp2 = interp1d(wl2, fl2, kind='linear', bounds_error=False, fill_value=np.nan)

    fl1_c = interp1(wl_common)
    fl2_c = interp2(wl_common)

    # Create combined mask
    mask1 = interp1d(wl1, good1.astype(float), kind='nearest',
                     bounds_error=False, fill_value=0)(wl_common) > 0.5
    mask2 = interp1d(wl2, good2.astype(float), kind='nearest',
                     bounds_error=False, fill_value=0)(wl_common) > 0.5
    good = mask1 & mask2 & np.isfinite(fl1_c) & np.isfinite(fl2_c)

    if np.sum(good) < 50:
        return v_grid, np.zeros_like(v_grid)

    # Subtract mean and normalize
    fl1_c = fl1_c - np.nanmean(fl1_c[good])
    fl2_c = fl2_c - np.nanmean(fl2_c[good])

    fl1_c[~good] = 0
    fl2_c[~good] = 0

    norm1 = np.sqrt(np.sum(fl1_c[good]**2))
    norm2 = np.sqrt(np.sum(fl2_c[good]**2))

    if norm1 == 0 or norm2 == 0:
        return v_grid, np.zeros_like(v_grid)

    fl1_c = fl1_c / norm1
    fl2_c = fl2_c / norm2

    # Compute CCF
    ccf = np.zeros(len(v_grid))

    for i, v in enumerate(v_grid):
        # Doppler shift
        shift = v / C_KMS
        wl_shifted = wl_common * (1 + shift)

        # Interpolate shifted spectrum
        interp_s = interp1d(wl_shifted, fl2_c, kind='linear',
                           bounds_error=False, fill_value=0)
        fl2_shifted = interp_s(wl_common)

        # Correlation
        ccf[i] = np.sum(fl1_c * fl2_shifted)

    return v_grid, ccf

def fit_ccf_gaussian(v_grid, ccf, max_fwhm=300):
    """
    Fit CCF peak with Gaussian, rejecting unphysical fits.

    max_fwhm: reject fits with FWHM > this value (km/s)
    """
    # Find peak
    peak_idx = np.argmax(ccf)
    peak_v = v_grid[peak_idx]
    peak_h = ccf[peak_idx]

    if peak_h <= 0:
        return None

    # Select fitting region (±100 km/s around peak)
    fit_width = 100
    fit_mask = np.abs(v_grid - peak_v) < fit_width
    v_fit = v_grid[fit_mask]
    ccf_fit = ccf[fit_mask]

    if len(v_fit) < 10:
        return None

    # Gaussian model
    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset

    try:
        # Initial guess
        p0 = [peak_h - ccf_fit.min(), peak_v, 30, ccf_fit.min()]
        bounds = ([0, peak_v - 50, 5, -1], [2, peak_v + 50, 200, 1])

        popt, pcov = curve_fit(gaussian, v_fit, ccf_fit, p0=p0,
                               bounds=bounds, maxfev=5000)

        rv = popt[1]
        sigma = abs(popt[2])
        fwhm = 2.355 * sigma

        # Sanity checks
        if fwhm > max_fwhm:
            return None
        if fwhm < 5:  # Too narrow is also suspicious
            return None

        # Error from covariance
        if np.isfinite(pcov[1, 1]):
            rv_err = np.sqrt(pcov[1, 1])
        else:
            rv_err = fwhm / (2 * np.sqrt(peak_h * len(v_fit)))

        # Minimum error floor based on pixel scale
        rv_err = max(rv_err, 0.5)

        return {
            'rv': rv,
            'rv_err': rv_err,
            'fwhm': fwhm,
            'peak_height': peak_h,
            'fit_quality': 'GOOD'
        }

    except Exception as e:
        # Fallback: parabola fit
        try:
            coeffs = np.polyfit(v_fit, ccf_fit, 2)
            if coeffs[0] >= 0:  # Not a maximum
                return None
            rv = -coeffs[1] / (2 * coeffs[0])

            # Estimate FWHM from parabola
            # At half max: a*x^2 + b*x + c = peak/2
            peak_val = np.polyval(coeffs, rv)
            half_max = peak_val / 2
            # Solve quadratic
            a, b, c = coeffs[0], coeffs[1], coeffs[2] - half_max
            disc = b**2 - 4*a*c
            if disc > 0:
                x1 = (-b + np.sqrt(disc)) / (2*a)
                x2 = (-b - np.sqrt(disc)) / (2*a)
                fwhm = abs(x2 - x1)
            else:
                fwhm = 50  # Default

            if fwhm > max_fwhm or fwhm < 5:
                return None

            return {
                'rv': rv,
                'rv_err': max(2.0, fwhm / 10),
                'fwhm': fwhm,
                'peak_height': peak_h,
                'fit_quality': 'PARABOLA'
            }
        except:
            return None

def check_for_double_peak(v_grid, ccf, primary_rv, primary_fwhm):
    """
    Check for secondary CCF peaks (SB2 indicator).
    """
    # Mask primary peak region
    mask_width = max(primary_fwhm * 1.5, 50)
    secondary_mask = np.abs(v_grid - primary_rv) > mask_width

    if np.sum(secondary_mask) < 20:
        return {'secondary_detected': False, 'ratio': 0}

    ccf_masked = ccf.copy()
    ccf_masked[~secondary_mask] = ccf[secondary_mask].min()

    # Find peaks in masked CCF
    peaks, properties = find_peaks(ccf_masked, height=0.1, distance=20)

    if len(peaks) > 0:
        secondary_height = ccf_masked[peaks].max()
        primary_height = ccf.max()
        ratio = secondary_height / primary_height

        return {
            'secondary_detected': ratio > 0.3,
            'ratio': ratio,
            'n_peaks': len(peaks)
        }

    return {'secondary_detected': False, 'ratio': 0, 'n_peaks': 0}

def main():
    print("=" * 70)
    print("FORENSIC V6.1: PROPER LAMOST RV RE-MEASUREMENT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load spectra
    fits1 = FITS_DIR / f"lamost_{OBSID_1}.fits"
    fits2 = FITS_DIR / f"lamost_{OBSID_2}.fits"

    if not fits1.exists() or not fits2.exists():
        print(f"ERROR: Missing FITS files")
        return

    hash1 = sha256_file(fits1)
    hash2 = sha256_file(fits2)

    print(f"\nFile hashes:")
    print(f"  {OBSID_1}: {hash1[:16]}...")
    print(f"  {OBSID_2}: {hash2[:16]}...")

    wl1, fl1, ivar1, info1 = load_lamost_spectrum(fits1)
    wl2, fl2, ivar2, info2 = load_lamost_spectrum(fits2)

    print(f"\nEpoch 1 (ObsID {OBSID_1}):")
    print(f"  Date: {info1['date_obs']}, MJD: {info1['mjd']}")
    print(f"  HELIO_RV (header): {info1['helio_rv']:.2f} km/s")
    print(f"  SNR_i: {info1['snr_i']:.1f}")

    print(f"\nEpoch 2 (ObsID {OBSID_2}):")
    print(f"  Date: {info2['date_obs']}, MJD: {info2['mjd']}")
    print(f"  HELIO_RV (header): {info2['helio_rv']:.2f} km/s")
    print(f"  SNR_i: {info2['snr_i']:.1f}")

    print(f"\nHeader ΔRV: {info2['helio_rv'] - info1['helio_rv']:.2f} km/s")

    # Normalize spectra
    print("\nNormalizing spectra...")
    fl1_norm, good1 = robust_continuum_normalize(wl1, fl1)
    fl2_norm, good2 = robust_continuum_normalize(wl2, fl2)

    # Apply telluric mask
    print("Applying telluric mask...")
    good1 = good1 & mask_tellurics(wl1)
    good2 = good2 & mask_tellurics(wl2)

    print(f"  Epoch 1: {np.sum(good1)} good pixels")
    print(f"  Epoch 2: {np.sum(good2)} good pixels")

    # Velocity grid
    v_grid = np.arange(-200, 201, 0.5)

    # Cross-correlate each clean region
    print("\n" + "=" * 70)
    print("CROSS-CORRELATION BY WAVELENGTH REGION")
    print("=" * 70)

    region_results = []

    for wl_min, wl_max, region_name in GOOD_REGIONS:
        # Select region
        mask1 = good1 & (wl1 >= wl_min) & (wl1 <= wl_max)
        mask2 = good2 & (wl2 >= wl_min) & (wl2 <= wl_max)

        n_pix = min(np.sum(mask1), np.sum(mask2))
        if n_pix < 50:
            print(f"\n{region_name}: SKIPPED (only {n_pix} pixels)")
            continue

        # Cross-correlate
        v, ccf = cross_correlate_clean(wl1, fl1_norm, wl2, fl2_norm, v_grid, mask1, mask2)

        # Fit peak
        fit = fit_ccf_gaussian(v, ccf, max_fwhm=250)

        if fit is None:
            print(f"\n{region_name}: FIT FAILED (unphysical parameters)")
            continue

        # Check for double peaks
        dbl = check_for_double_peak(v, ccf, fit['rv'], fit['fwhm'])

        result = {
            'region': region_name,
            'wl_range': [wl_min, wl_max],
            'n_pixels': n_pix,
            'delta_rv': fit['rv'],
            'delta_rv_err': fit['rv_err'],
            'fwhm': fit['fwhm'],
            'peak_height': fit['peak_height'],
            'fit_quality': fit['fit_quality'],
            'secondary_peak_ratio': dbl['ratio'],
            'sb2_flag': dbl['secondary_detected'],
            'ccf_v': v.tolist(),
            'ccf': ccf.tolist()
        }
        region_results.append(result)

        print(f"\n{region_name} ({wl_min}-{wl_max} Å):")
        print(f"  ΔRV = {fit['rv']:.2f} ± {fit['rv_err']:.2f} km/s")
        print(f"  FWHM = {fit['fwhm']:.1f} km/s")
        print(f"  Peak height = {fit['peak_height']:.3f}")
        print(f"  Fit: {fit['fit_quality']}")
        if dbl['secondary_detected']:
            print(f"  WARNING: Secondary peak detected (ratio={dbl['ratio']:.2f})")

    # Combine results
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)

    if len(region_results) == 0:
        print("\nERROR: No valid regions!")
        return

    # Weighted average with outlier rejection
    rvs = np.array([r['delta_rv'] for r in region_results])
    errs = np.array([r['delta_rv_err'] for r in region_results])

    # Reject outliers (>3 sigma from median)
    median_rv = np.median(rvs)
    mad = np.median(np.abs(rvs - median_rv))
    good_fit = np.abs(rvs - median_rv) < 3 * mad * 1.4826

    if np.sum(good_fit) < 2:
        good_fit = np.ones(len(rvs), dtype=bool)

    rvs_clean = rvs[good_fit]
    errs_clean = errs[good_fit]

    # Weighted average
    weights = 1 / errs_clean**2
    delta_rv_weighted = np.sum(weights * rvs_clean) / np.sum(weights)
    delta_rv_err_internal = 1 / np.sqrt(np.sum(weights))

    # Add systematic error from scatter
    scatter = np.std(rvs_clean)
    delta_rv_err_total = np.sqrt(delta_rv_err_internal**2 + (scatter / np.sqrt(len(rvs_clean)))**2)

    # More conservative: use scatter directly if larger
    delta_rv_err_final = max(delta_rv_err_total, scatter / np.sqrt(len(rvs_clean)), 1.0)

    print(f"\nValid regions: {np.sum(good_fit)} / {len(region_results)}")
    print(f"Individual ΔRVs: {rvs_clean}")
    print(f"Scatter: {scatter:.2f} km/s")
    print(f"\nWeighted ΔRV: {delta_rv_weighted:.2f} ± {delta_rv_err_final:.2f} km/s")

    # Compare to header
    header_delta = info2['helio_rv'] - info1['helio_rv']
    print(f"Header ΔRV: {header_delta:.2f} km/s")
    print(f"Difference (refit - header): {delta_rv_weighted - header_delta:.2f} km/s")

    # Significance
    significance = abs(delta_rv_weighted) / delta_rv_err_final
    print(f"\nSignificance: {significance:.1f}σ")

    # Verdict
    if significance > 3 and abs(delta_rv_weighted) > 5:
        verdict = "DETECTED"
        reason = f"ΔRV = {delta_rv_weighted:.1f} ± {delta_rv_err_final:.1f} km/s at {significance:.1f}σ"
    elif significance > 2:
        verdict = "MARGINAL"
        reason = f"ΔRV = {delta_rv_weighted:.1f} ± {delta_rv_err_final:.1f} km/s at {significance:.1f}σ"
    else:
        verdict = "NOT DETECTED"
        reason = f"ΔRV = {delta_rv_weighted:.1f} ± {delta_rv_err_final:.1f} km/s ({significance:.1f}σ)"

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"Reason: {reason}")
    print(f"{'='*70}")

    # Check consistency with header
    header_consistent = abs(delta_rv_weighted - header_delta) < 3 * delta_rv_err_final
    print(f"\nConsistent with header ΔRV: {'YES' if header_consistent else 'NO'}")

    # Save results
    results = {
        'method': 'Self-template cross-correlation v2 (with telluric masking)',
        'file_hashes': {
            f'lamost_{OBSID_1}.fits': hash1,
            f'lamost_{OBSID_2}.fits': hash2
        },
        'epoch1': {
            'obsid': OBSID_1,
            'date_obs': info1['date_obs'],
            'mjd': info1['mjd'],
            'helio_rv_header_kms': info1['helio_rv'],
            'snr_i': info1['snr_i']
        },
        'epoch2': {
            'obsid': OBSID_2,
            'date_obs': info2['date_obs'],
            'mjd': info2['mjd'],
            'helio_rv_header_kms': info2['helio_rv'],
            'snr_i': info2['snr_i']
        },
        'telluric_regions_masked': TELLURIC_REGIONS,
        'wavelength_regions_used': [[int(r['wl_range'][0]), int(r['wl_range'][1]), r['region']]
                                    for r in region_results],
        'region_results': [
            {
                'region': r['region'],
                'wl_range': [int(r['wl_range'][0]), int(r['wl_range'][1])],
                'n_pixels': int(r['n_pixels']),
                'delta_rv_kms': r['delta_rv'],
                'delta_rv_err_kms': r['delta_rv_err'],
                'fwhm_kms': r['fwhm'],
                'peak_height': r['peak_height'],
                'fit_quality': r['fit_quality'],
                'sb2_flag': r['sb2_flag']
            }
            for r in region_results
        ],
        'combined': {
            'delta_rv_kms': float(delta_rv_weighted),
            'delta_rv_err_kms': float(delta_rv_err_final),
            'scatter_kms': float(scatter),
            'n_regions_used': int(np.sum(good_fit)),
            'significance_sigma': float(significance)
        },
        'header_comparison': {
            'header_delta_rv_kms': float(header_delta),
            'refit_minus_header_kms': float(delta_rv_weighted - header_delta),
            'consistent': bool(header_consistent)
        },
        'verdict': verdict,
        'verdict_reason': reason
    }

    output_json = OUTPUT_DIR / "lamost_rv_refit_v2.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_json}")

    # Create figures
    print("\nGenerating figures...")

    # Figure 1: CCF diagnostics
    n_regions = len(region_results)
    n_cols = min(3, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_regions == 1:
        axes1 = [axes1]
    else:
        axes1 = axes1.flatten()

    for i, r in enumerate(region_results):
        ax = axes1[i]
        v = np.array(r['ccf_v'])
        ccf = np.array(r['ccf'])

        ax.plot(v, ccf, 'b-', lw=1.5)
        ax.axvline(r['delta_rv'], color='r', ls='--', lw=2,
                  label=f"ΔRV={r['delta_rv']:.1f}±{r['delta_rv_err']:.1f}")
        ax.axvline(0, color='gray', ls=':', alpha=0.5)

        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('CCF')
        ax.set_title(f"{r['region']}\nFWHM={r['fwhm']:.0f} km/s, {r['fit_quality']}")
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-150, 150)
        ax.grid(True, alpha=0.3)

    for i in range(len(region_results), len(axes1)):
        axes1[i].set_visible(False)

    fig1.suptitle(f'CCF Diagnostics (Telluric-Masked)\n'
                  f'Combined ΔRV = {delta_rv_weighted:.1f} ± {delta_rv_err_final:.1f} km/s '
                  f'({significance:.1f}σ) | {verdict}', fontsize=11)
    plt.tight_layout()
    fig1.savefig(FIG_DIR / 'lamost_ccf_diagnostics_v2.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'lamost_ccf_diagnostics_v2.png'}")
    plt.close(fig1)

    # Figure 2: Spectrum comparison with masked regions
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes2[0]
    ax.plot(wl1, fl1_norm, 'b-', lw=0.5, alpha=0.7, label='Epoch 1')
    ax.plot(wl2, fl2_norm, 'r-', lw=0.5, alpha=0.7, label='Epoch 2')

    # Shade telluric regions
    for wl_min, wl_max in TELLURIC_REGIONS:
        ax.axvspan(wl_min, wl_max, alpha=0.2, color='gray')

    # Mark good regions
    for wl_min, wl_max, name in GOOD_REGIONS:
        ax.axvspan(wl_min, wl_max, alpha=0.1, color='green')

    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Spectra with Telluric Masks (gray) and Analysis Regions (green)')
    ax.legend()
    ax.set_xlim(6000, 9000)
    ax.set_ylim(0, 2)

    # Summary panel
    ax = axes2[1]
    ax.axis('off')

    summary_text = f"""
LAMOST RV Re-measurement Summary (v2 - Telluric Masked)
{'='*60}

Epoch 1: ObsID {OBSID_1}, MJD {info1['mjd']}, SNR_i={info1['snr_i']:.1f}
Epoch 2: ObsID {OBSID_2}, MJD {info2['mjd']}, SNR_i={info2['snr_i']:.1f}

Header HELIO_RV:
  Epoch 1: {info1['helio_rv']:+.2f} km/s
  Epoch 2: {info2['helio_rv']:+.2f} km/s
  Header ΔRV: {header_delta:.2f} km/s

Cross-Correlation Refit:
  Regions used: {np.sum(good_fit)} / {len(region_results)}
  ΔRV = {delta_rv_weighted:.2f} ± {delta_rv_err_final:.2f} km/s
  Scatter between regions: {scatter:.2f} km/s
  Significance: {significance:.1f}σ

VERDICT: {verdict}
{reason}

Consistent with header: {'YES' if header_consistent else 'NO'}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    fig2.savefig(FIG_DIR / 'lamost_epoch_overlay_v2.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'lamost_epoch_overlay_v2.png'}")
    plt.close(fig2)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results

if __name__ == '__main__':
    results = main()
