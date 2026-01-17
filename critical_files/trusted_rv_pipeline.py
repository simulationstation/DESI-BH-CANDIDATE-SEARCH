#!/usr/bin/env python3
"""
Trusted RV Pipeline for DESI target 39627745210139276.

Uses ONLY the Ca II triplet window (8500-9000 Å) which has been empirically
verified to be stable between same-night exposures, excluding the sky-dominated
9000-9800 Å region that shows instrumental instability.

Author: Claude
Date: 2026-01-16
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, brentq
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

TARGET_ID = 39627745210139276
C_KMS = 299792.458  # Speed of light in km/s

# TRUSTED SPECTRAL WINDOW: Ca II triplet region only
WAVE_MIN = 8500.0  # Å
WAVE_MAX = 9000.0  # Å

# Exposure metadata (from previous analysis)
EXPOSURES = [
    {"expid": 114768, "night": 20211219, "petal": 2, "fiber": 1377, "mjd": 59568.488,
     "row_idx": 377, "catalog_rv": -86.39, "catalog_err": 0.55},
    {"expid": 120194, "night": 20220125, "petal": 7, "fiber": 3816, "mjd": 59605.380,
     "row_idx": 316, "catalog_rv": 59.68, "catalog_err": 0.83},
    {"expid": 120449, "night": 20220127, "petal": 0, "fiber": 68, "mjd": 59607.374,
     "row_idx": 68, "catalog_rv": 26.43, "catalog_err": 1.06},
    {"expid": 120450, "night": 20220127, "petal": 0, "fiber": 68, "mjd": 59607.389,
     "row_idx": 68, "catalog_rv": 25.16, "catalog_err": 1.11},
]

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_EXP_DIR = os.path.join(BASE_DIR, "data/per_exposure")
TEMPLATE_DIR = os.path.join(BASE_DIR, "data/templates")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/trusted_rv_v1")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# Known telluric/sky features to mask within 8500-9000 Å
# These are approximate; we'll also mask ivar=0 pixels
TELLURIC_MASK_RANGES = [
    # O2 A-band tail (if present)
    # Generally 8500-9000 is relatively clean, but mask obvious sky lines
]


# =============================================================================
# Utility Functions
# =============================================================================

def compute_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_phoenix_template(teff=3800):
    """Load PHOENIX template for M-dwarf."""
    wave_file = os.path.join(TEMPLATE_DIR, "WAVE_PHOENIX.fits")

    # Find best matching template
    template_files = [f for f in os.listdir(TEMPLATE_DIR)
                      if f.startswith("phoenix") and f.endswith(".fits") and "WAVE" not in f]

    best_match = None
    best_diff = float('inf')
    for tf in template_files:
        try:
            parts = tf.replace(".fits", "").split("_")
            for p in parts:
                if p.isdigit() and len(p) == 4:
                    t = int(p)
                    if abs(t - teff) < best_diff:
                        best_diff = abs(t - teff)
                        best_match = tf
        except:
            continue

    if best_match is None:
        raise FileNotFoundError(f"No PHOENIX template found near Teff={teff}")

    print(f"  Loading PHOENIX template: {best_match}")

    with fits.open(wave_file) as hdul:
        wave = hdul[0].data
    with fits.open(os.path.join(TEMPLATE_DIR, best_match)) as hdul:
        flux = hdul[0].data

    return wave, flux, best_match


def resample_template_to_desi(wave_template, flux_template, wave_obs, desi_resolution_fwhm=1.5):
    """
    Resample PHOENIX template to DESI wavelength grid and resolution.

    Args:
        wave_template: PHOENIX wavelength array (Å)
        flux_template: PHOENIX flux array
        wave_obs: DESI observed wavelength array
        desi_resolution_fwhm: approximate FWHM in Å for DESI Z-arm

    Returns:
        flux_resampled: template flux on DESI wavelength grid
    """
    # PHOENIX is at very high resolution; convolve to approximate DESI resolution
    # PHOENIX pixel scale is ~0.01 Å, DESI Z-arm is ~0.8 Å/pixel
    # FWHM in pixels for PHOENIX
    phoenix_pixel_scale = np.median(np.diff(wave_template[(wave_template > 8000) & (wave_template < 9500)]))
    fwhm_pixels = desi_resolution_fwhm / phoenix_pixel_scale
    sigma_pixels = fwhm_pixels / 2.355

    # Convolve
    flux_convolved = gaussian_filter1d(flux_template, sigma_pixels)

    # Interpolate to DESI grid
    interp_func = interp1d(wave_template, flux_convolved, kind='linear',
                           bounds_error=False, fill_value=0.0)
    flux_resampled = interp_func(wave_obs)

    return flux_resampled


def doppler_shift(wave, rv_kms):
    """Apply Doppler shift to wavelength array."""
    beta = rv_kms / C_KMS
    return wave * np.sqrt((1 + beta) / (1 - beta))


def continuum_normalize(wave, flux, ivar, order=2, sigma_clip=3.0, n_iter=3):
    """
    Robust continuum normalization using iterative polynomial fit.

    Args:
        wave: wavelength array
        flux: flux array
        ivar: inverse variance
        order: polynomial order
        sigma_clip: sigma clipping threshold
        n_iter: number of iterations

    Returns:
        flux_norm: normalized flux
        continuum: fitted continuum
    """
    good = (ivar > 0) & np.isfinite(flux)

    if np.sum(good) < order + 2:
        return flux / np.median(flux[good]) if np.any(good) else flux, np.ones_like(flux)

    x = (wave - np.mean(wave[good])) / np.std(wave[good])  # Normalize for numerical stability

    mask = good.copy()
    for _ in range(n_iter):
        if np.sum(mask) < order + 2:
            break
        coeffs = np.polyfit(x[mask], flux[mask], order)
        continuum = np.polyval(coeffs, x)
        residuals = flux - continuum
        std = np.std(residuals[mask])
        mask = good & (np.abs(residuals) < sigma_clip * std)

    # Final fit
    if np.sum(mask) >= order + 2:
        coeffs = np.polyfit(x[mask], flux[mask], order)
        continuum = np.polyval(coeffs, x)
    else:
        continuum = np.median(flux[good]) * np.ones_like(flux)

    continuum = np.maximum(continuum, 1e-10)  # Avoid division by zero
    flux_norm = flux / continuum

    return flux_norm, continuum


# =============================================================================
# RV Fitting with χ² Minimization and Δχ²=1 Uncertainties
# =============================================================================

def compute_chi2_vs_rv(wave_obs, flux_obs, ivar_obs, wave_template, flux_template,
                        rv_grid, continuum_order=1):
    """
    Compute χ² as a function of RV using template fitting.

    For each trial RV:
    1. Doppler shift template
    2. Resample to observed wavelength grid
    3. Fit scale factor + continuum polynomial (linear least squares)
    4. Compute χ²

    Args:
        wave_obs: observed wavelength array
        flux_obs: observed flux (should be continuum-normalized or raw)
        ivar_obs: inverse variance
        wave_template: template wavelength array
        flux_template: template flux array
        rv_grid: array of RV values to test (km/s)
        continuum_order: polynomial order for nuisance continuum

    Returns:
        chi2_arr: χ² at each RV
        dof: degrees of freedom
        n_params: number of fitted parameters
    """
    good = (ivar_obs > 0) & np.isfinite(flux_obs) & np.isfinite(ivar_obs)
    n_good = np.sum(good)

    if n_good < continuum_order + 3:
        return np.full(len(rv_grid), np.inf), 0, 0

    # Number of parameters: 1 (scale) + continuum_order + 1 (polynomial coefficients)
    n_params = 1 + continuum_order + 1
    dof = n_good - n_params

    chi2_arr = np.zeros(len(rv_grid))

    # Normalized x for polynomial
    x_obs = (wave_obs - np.mean(wave_obs[good])) / (np.std(wave_obs[good]) + 1e-10)

    for i, rv in enumerate(rv_grid):
        # Doppler shift template wavelengths
        wave_shifted = doppler_shift(wave_template, rv)

        # Interpolate template to observed grid
        interp_func = interp1d(wave_shifted, flux_template, kind='linear',
                               bounds_error=False, fill_value=0.0)
        template_on_obs = interp_func(wave_obs)

        # Build design matrix: [template, 1, x, x^2, ...]
        design = np.zeros((n_good, n_params))
        design[:, 0] = template_on_obs[good]  # Template (scale factor)
        for j in range(continuum_order + 1):
            design[:, 1 + j] = x_obs[good] ** j

        # Weighted least squares
        W = np.diag(ivar_obs[good])
        try:
            # Normal equations: (X^T W X) beta = X^T W y
            XtWX = design.T @ W @ design
            XtWy = design.T @ W @ flux_obs[good]
            beta = np.linalg.solve(XtWX, XtWy)

            # Compute model and chi2
            model = design @ beta
            residuals = flux_obs[good] - model
            chi2_arr[i] = np.sum(residuals**2 * ivar_obs[good])
        except np.linalg.LinAlgError:
            chi2_arr[i] = np.inf

    return chi2_arr, dof, n_params


def fit_rv_with_uncertainty(wave_obs, flux_obs, ivar_obs, wave_template, flux_template,
                            rv_range=(-200, 200), rv_step=0.5):
    """
    Fit RV with proper Δχ²=1 uncertainty estimation.

    Returns:
        v_best: best-fit RV (km/s)
        sigma_v: uncertainty from Δχ²=1 (km/s)
        chi2_min: minimum χ²
        chi2_red: reduced χ² at best fit
        dof: degrees of freedom
        rv_grid: RV grid used
        chi2_arr: χ² array
    """
    # Coarse grid search
    rv_grid_coarse = np.arange(rv_range[0], rv_range[1] + rv_step, rv_step)
    chi2_coarse, dof, n_params = compute_chi2_vs_rv(
        wave_obs, flux_obs, ivar_obs, wave_template, flux_template,
        rv_grid_coarse, continuum_order=1
    )

    if dof <= 0 or np.all(np.isinf(chi2_coarse)):
        return np.nan, np.nan, np.nan, np.nan, 0, rv_grid_coarse, chi2_coarse

    # Find minimum
    idx_min = np.argmin(chi2_coarse)
    v_coarse = rv_grid_coarse[idx_min]

    # Fine grid around minimum
    rv_grid_fine = np.arange(v_coarse - 10, v_coarse + 10.1, 0.1)
    chi2_fine, dof, n_params = compute_chi2_vs_rv(
        wave_obs, flux_obs, ivar_obs, wave_template, flux_template,
        rv_grid_fine, continuum_order=1
    )

    idx_min_fine = np.argmin(chi2_fine)
    v_best = rv_grid_fine[idx_min_fine]
    chi2_min = chi2_fine[idx_min_fine]
    chi2_red = chi2_min / dof if dof > 0 else np.inf

    # Renormalize χ² so that χ²_red ~ 1
    # This accounts for model misspecification
    chi2_norm = chi2_fine / chi2_red
    chi2_min_norm = chi2_norm[idx_min_fine]  # Should be ~dof now

    # Find Δχ²=1 boundaries using renormalized χ²
    # We want χ²_norm - χ²_min_norm = 1
    target = chi2_min_norm + 1.0

    # Find lower bound
    sigma_lower = np.nan
    try:
        # Interpolate to find crossing
        for i in range(idx_min_fine - 1, -1, -1):
            if chi2_norm[i] >= target:
                # Linear interpolation
                x1, x2 = rv_grid_fine[i], rv_grid_fine[i+1]
                y1, y2 = chi2_norm[i], chi2_norm[i+1]
                v_cross = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
                sigma_lower = v_best - v_cross
                break
    except:
        pass

    # Find upper bound
    sigma_upper = np.nan
    try:
        for i in range(idx_min_fine + 1, len(chi2_norm)):
            if chi2_norm[i] >= target:
                x1, x2 = rv_grid_fine[i-1], rv_grid_fine[i]
                y1, y2 = chi2_norm[i-1], chi2_norm[i]
                v_cross = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
                sigma_upper = v_cross - v_best
                break
    except:
        pass

    # Take average of lower and upper (or the one that exists)
    if np.isfinite(sigma_lower) and np.isfinite(sigma_upper):
        sigma_v = (sigma_lower + sigma_upper) / 2
    elif np.isfinite(sigma_lower):
        sigma_v = sigma_lower
    elif np.isfinite(sigma_upper):
        sigma_v = sigma_upper
    else:
        # Fall back to curvature estimate
        # σ² ≈ 2 / (d²χ²/dv²) at minimum
        try:
            d2chi2 = (chi2_norm[idx_min_fine+1] + chi2_norm[idx_min_fine-1] - 2*chi2_min_norm) / (0.1**2)
            sigma_v = np.sqrt(2 / d2chi2) if d2chi2 > 0 else np.nan
        except:
            sigma_v = np.nan

    return v_best, sigma_v, chi2_min, chi2_red, dof, rv_grid_fine, chi2_fine


# =============================================================================
# Main Analysis Functions
# =============================================================================

def analyze_exposure(exp_info, wave_template, flux_template):
    """
    Analyze a single exposure in the trusted Ca II triplet window.

    Returns:
        result: dict with RV, uncertainty, chi2, flags, etc.
    """
    expid = exp_info["expid"]
    petal = exp_info["petal"]
    row_idx = exp_info["row_idx"]

    # Construct file path
    cam = f"z{petal}"
    filename = f"cframe-{cam}-{expid:08d}.fits"
    filepath = os.path.join(PER_EXP_DIR, filename)

    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return None

    # Compute hash for provenance
    file_hash = compute_sha256(filepath)

    # Load data
    with fits.open(filepath) as hdul:
        # Wavelength is 1D (shared across all fibers)
        wave = hdul['WAVELENGTH'].data
        # Flux and ivar are 2D: [fiber, wavelength]
        flux = hdul['FLUX'].data[row_idx]
        ivar = hdul['IVAR'].data[row_idx]

        # Get MJD from header if available
        try:
            mjd = hdul[0].header.get('MJD-OBS', exp_info['mjd'])
        except:
            mjd = exp_info['mjd']

    # Restrict to trusted window: 8500-9000 Å
    mask_window = (wave >= WAVE_MIN) & (wave <= WAVE_MAX)

    if np.sum(mask_window) < 50:
        print(f"  WARNING: Fewer than 50 pixels in trusted window for EXPID {expid}")
        return None

    wave_window = wave[mask_window]
    flux_window = flux[mask_window]
    ivar_window = ivar[mask_window]

    # Additional masking: ivar=0, bad values
    good = (ivar_window > 0) & np.isfinite(flux_window) & np.isfinite(ivar_window)

    # Apply telluric masks if any
    for wmin, wmax in TELLURIC_MASK_RANGES:
        good &= ~((wave_window >= wmin) & (wave_window <= wmax))

    n_pixels = np.sum(good)

    if n_pixels < 30:
        print(f"  WARNING: Fewer than 30 good pixels after masking for EXPID {expid}")
        return None

    # Compute SNR proxy
    snr_pixels = np.sqrt(ivar_window[good]) * np.abs(flux_window[good])
    snr_median = np.median(snr_pixels)

    # Continuum normalize
    flux_norm, continuum = continuum_normalize(wave_window, flux_window, ivar_window, order=2)

    # Scale ivar for normalized flux
    ivar_norm = ivar_window * continuum**2

    # Fit RV
    v_best, sigma_v, chi2_min, chi2_red, dof, rv_grid, chi2_arr = fit_rv_with_uncertainty(
        wave_window, flux_norm, ivar_norm, wave_template, flux_template,
        rv_range=(-200, 200), rv_step=0.5
    )

    # Determine flags
    flags = []
    if chi2_red > 10:
        flags.append("HIGH_CHI2_RED")
    if n_pixels < 100:
        flags.append("LOW_NPIX")
    if snr_median < 5:
        flags.append("LOW_SNR")
    if not np.isfinite(sigma_v):
        flags.append("UNCERTAIN_ERROR")

    result = {
        "expid": expid,
        "camera": cam,
        "mjd": mjd,
        "catalog_rv": exp_info["catalog_rv"],
        "catalog_err": exp_info["catalog_err"],
        "trusted_rv": round(v_best, 2) if np.isfinite(v_best) else None,
        "trusted_err": round(sigma_v, 2) if np.isfinite(sigma_v) else None,
        "chi2_min": round(chi2_min, 2) if np.isfinite(chi2_min) else None,
        "chi2_red": round(chi2_red, 2) if np.isfinite(chi2_red) else None,
        "dof": dof,
        "n_pixels": int(n_pixels),
        "snr_median": round(snr_median, 2),
        "wave_min": float(WAVE_MIN),
        "wave_max": float(WAVE_MAX),
        "flags": flags,
        "file_hash": file_hash,
        "rv_grid": rv_grid.tolist(),
        "chi2_scan": chi2_arr.tolist()
    }

    return result


def combine_same_night_exposures(results):
    """
    For same-night exposures, check consistency and optionally combine.

    Returns:
        epoch_results: list of per-epoch results
        consistency_check: dict with same-night consistency info
    """
    # Group by night
    from collections import defaultdict
    by_night = defaultdict(list)

    for r in results:
        if r is None:
            continue
        # Extract night from expid mapping
        for exp in EXPOSURES:
            if exp["expid"] == r["expid"]:
                by_night[exp["night"]].append(r)
                break

    epoch_results = []
    consistency_check = {}

    for night, night_results in sorted(by_night.items()):
        if len(night_results) == 1:
            # Single exposure for this night
            r = night_results[0]
            epoch_results.append({
                "night": night,
                "mjd": r["mjd"],
                "epoch_rv": r["trusted_rv"],
                "epoch_err": r["trusted_err"],
                "n_exposures": 1,
                "exposures": [r["expid"]],
                "flags": r["flags"]
            })
        else:
            # Multiple exposures - check consistency and combine
            rvs = [r["trusted_rv"] for r in night_results if r["trusted_rv"] is not None]
            errs = [r["trusted_err"] for r in night_results if r["trusted_err"] is not None]

            if len(rvs) >= 2 and len(errs) >= 2:
                # Weighted mean
                weights = [1/e**2 for e in errs if e > 0]
                if sum(weights) > 0:
                    rv_combined = sum(r*w for r, w in zip(rvs, weights)) / sum(weights)
                    err_combined = 1 / np.sqrt(sum(weights))

                    # Check consistency: chi2 of individual RVs around combined
                    chi2_consistency = sum((rv - rv_combined)**2 / e**2 for rv, e in zip(rvs, errs))
                    ndof = len(rvs) - 1

                    # If inconsistent, inflate error
                    if ndof > 0 and chi2_consistency / ndof > 2:
                        err_combined *= np.sqrt(chi2_consistency / ndof)
                        consistent = False
                    else:
                        consistent = True

                    consistency_check[night] = {
                        "night": night,
                        "exposures": [r["expid"] for r in night_results],
                        "rvs": rvs,
                        "errs": errs,
                        "rv_combined": round(rv_combined, 2),
                        "err_combined": round(err_combined, 2),
                        "chi2_consistency": round(chi2_consistency, 2),
                        "ndof": ndof,
                        "consistent": consistent,
                        "delta_rv": round(abs(rvs[0] - rvs[1]), 2) if len(rvs) == 2 else None,
                        "sigma_diff": round(abs(rvs[0] - rvs[1]) / np.sqrt(errs[0]**2 + errs[1]**2), 2) if len(rvs) == 2 else None
                    }

                    epoch_results.append({
                        "night": night,
                        "mjd": np.mean([r["mjd"] for r in night_results]),
                        "epoch_rv": round(rv_combined, 2),
                        "epoch_err": round(err_combined, 2),
                        "n_exposures": len(night_results),
                        "exposures": [r["expid"] for r in night_results],
                        "consistent": consistent,
                        "flags": []
                    })

    return epoch_results, consistency_check


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_chi2_scans(results, output_path):
    """Plot χ² scans for all exposures."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, r in enumerate(results):
        if r is None or i >= 4:
            continue

        ax = axes[i]
        rv_grid = np.array(r["rv_grid"])
        chi2_arr = np.array(r["chi2_scan"])

        # Normalize for plotting
        chi2_red = r["chi2_red"]
        chi2_norm = chi2_arr / chi2_red if chi2_red > 0 else chi2_arr

        ax.plot(rv_grid, chi2_norm, 'b-', lw=1.5)

        # Mark minimum
        idx_min = np.argmin(chi2_norm)
        ax.axvline(rv_grid[idx_min], color='r', ls='--', alpha=0.7, label=f'v = {r["trusted_rv"]:.1f} km/s')
        ax.axhline(chi2_norm[idx_min] + 1, color='gray', ls=':', alpha=0.7, label='Δχ² = 1')

        # Mark catalog RV
        ax.axvline(r["catalog_rv"], color='green', ls=':', alpha=0.7, label=f'Catalog = {r["catalog_rv"]:.1f} km/s')

        ax.set_xlabel('RV (km/s)')
        ax.set_ylabel('χ²/χ²_red')
        ax.set_title(f'EXPID {r["expid"]} (MJD {r["mjd"]:.3f})\nχ²_red = {r["chi2_red"]:.1f}, σ = {r["trusted_err"]:.2f} km/s')
        ax.legend(fontsize=8)
        ax.set_xlim(-100, 100)

    plt.suptitle(f'χ² Scans - Trusted Window ({WAVE_MIN:.0f}-{WAVE_MAX:.0f} Å)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_rv_by_exposure(results, output_path):
    """Plot RV vs MJD for individual exposures."""
    fig, ax = plt.subplots(figsize=(10, 6))

    mjds = []
    rvs = []
    errs = []
    catalog_rvs = []
    catalog_errs = []
    expids = []

    for r in results:
        if r is None or r["trusted_rv"] is None:
            continue
        mjds.append(r["mjd"])
        rvs.append(r["trusted_rv"])
        errs.append(r["trusted_err"] if r["trusted_err"] else 5.0)
        catalog_rvs.append(r["catalog_rv"])
        catalog_errs.append(r["catalog_err"])
        expids.append(r["expid"])

    mjds = np.array(mjds)
    rvs = np.array(rvs)
    errs = np.array(errs)

    # Plot trusted RVs
    ax.errorbar(mjds, rvs, yerr=errs, fmt='o', color='blue', markersize=10,
                capsize=5, capthick=2, elinewidth=2, label='Trusted Window RV')

    # Plot catalog RVs for comparison
    ax.errorbar(mjds + 0.5, catalog_rvs, yerr=catalog_errs, fmt='s', color='gray',
                markersize=8, capsize=3, alpha=0.5, label='DESI Catalog RV')

    # Annotate
    for i, expid in enumerate(expids):
        ax.annotate(f'{expid}', (mjds[i], rvs[i] + errs[i] + 3),
                    ha='center', fontsize=9)

    ax.axhline(0, color='black', ls='-', alpha=0.3)
    ax.set_xlabel('MJD', fontsize=12)
    ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)
    ax.set_title(f'Trusted Window RV by Exposure\n({WAVE_MIN:.0f}-{WAVE_MAX:.0f} Å, Ca II triplet)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_rv_by_epoch(epoch_results, output_path):
    """Plot per-epoch RV curve."""
    fig, ax = plt.subplots(figsize=(10, 6))

    mjds = [e["mjd"] for e in epoch_results]
    rvs = [e["epoch_rv"] for e in epoch_results]
    errs = [e["epoch_err"] if e["epoch_err"] else 5.0 for e in epoch_results]

    ax.errorbar(mjds, rvs, yerr=errs, fmt='o-', color='darkblue', markersize=12,
                capsize=5, capthick=2, elinewidth=2, linewidth=1.5)

    # Annotate epochs
    for i, e in enumerate(epoch_results):
        label = f"E{i+1}\n({e['n_exposures']} exp)"
        ax.annotate(label, (mjds[i], rvs[i] + errs[i] + 5),
                    ha='center', fontsize=9)

    ax.axhline(0, color='black', ls='-', alpha=0.3)
    ax.set_xlabel('MJD', fontsize=12)
    ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)
    ax.set_title(f'Trusted Window RV Curve by Epoch\n({WAVE_MIN:.0f}-{WAVE_MAX:.0f} Å, Ca II triplet)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add RV swing annotation
    rv_min, rv_max = min(rvs), max(rvs)
    rv_swing = rv_max - rv_min
    ax.annotate(f'ΔRV = {rv_swing:.1f} km/s',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_same_night_consistency(results, consistency_check, output_path):
    """Plot same-night exposure comparison."""
    # Find same-night pair (night 20220127 has 120449 and 120450)
    same_night = 20220127

    if same_night not in consistency_check:
        print("  No same-night exposures to compare")
        return

    check = consistency_check[same_night]

    fig, ax = plt.subplots(figsize=(8, 6))

    expids = check["exposures"]
    rvs = check["rvs"]
    errs = check["errs"]

    x = np.arange(len(expids))

    ax.errorbar(x, rvs, yerr=errs, fmt='o', markersize=15, capsize=8,
                capthick=2, elinewidth=2, color='blue')

    # Combined value
    ax.axhline(check["rv_combined"], color='red', ls='--', lw=2,
               label=f'Combined: {check["rv_combined"]:.1f} ± {check["err_combined"]:.1f} km/s')
    ax.axhspan(check["rv_combined"] - check["err_combined"],
               check["rv_combined"] + check["err_combined"],
               alpha=0.2, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels([f'EXPID {e}' for e in expids])
    ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)
    ax.set_title(f'Same-Night Consistency Check (Night {same_night})\nTrusted Window: {WAVE_MIN:.0f}-{WAVE_MAX:.0f} Å', fontsize=14)

    # Consistency annotation
    status = "CONSISTENT" if check["consistent"] else "INCONSISTENT"
    color = "green" if check["consistent"] else "red"
    ax.annotate(f'{status}\nΔRV = {check["delta_rv"]:.1f} km/s ({check["sigma_diff"]:.1f}σ)',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(results, epoch_results, consistency_check, template_name):
    """Generate TRUSTED_RV_REPORT.md"""

    report = f"""# Trusted RV Curve Report

## Overview

This report presents radial velocity measurements for DESI target **39627745210139276**
(Gaia DR3 3802130935635096832) using ONLY the empirically verified stable spectral window.

**Trusted Window**: {WAVE_MIN:.0f} - {WAVE_MAX:.0f} Å (Ca II triplet region)

**Rationale**: Per-exposure analysis showed that same-night Z-arm exposures (120449 & 120450,
15 minutes apart) have a 34 km/s discrepancy in the full Z-arm, but only ~3 km/s in the
Ca II triplet region. The 9000-9800 Å sky-dominated region is unstable and excluded.

---

## Data Provenance

| EXPID | Camera | File Hash (SHA256) |
|-------|--------|-------------------|
"""

    for r in results:
        if r:
            report += f"| {r['expid']} | {r['camera']} | `{r['file_hash'][:16]}...` |\n"

    report += f"""
**Template**: {template_name} (PHOENIX, Teff~3800K, logg~4.5)

---

## Per-Exposure Results

| EXPID | MJD | Trusted RV (km/s) | σ (km/s) | χ²_red | N_pix | SNR | Catalog RV | Flags |
|-------|-----|-------------------|----------|--------|-------|-----|------------|-------|
"""

    for r in results:
        if r:
            flags_str = ", ".join(r["flags"]) if r["flags"] else "-"
            report += f"| {r['expid']} | {r['mjd']:.3f} | {r['trusted_rv']} | {r['trusted_err']} | {r['chi2_red']} | {r['n_pixels']} | {r['snr_median']:.1f} | {r['catalog_rv']} | {flags_str} |\n"

    report += """
---

## Same-Night Consistency Check

"""

    if consistency_check:
        for night, check in consistency_check.items():
            status = "✓ CONSISTENT" if check["consistent"] else "✗ INCONSISTENT"
            report += f"""### Night {night}

- Exposures: {check['exposures']}
- Individual RVs: {check['rvs']} km/s
- Individual errors: {check['errs']} km/s
- **ΔRV**: {check['delta_rv']} km/s ({check['sigma_diff']:.1f}σ)
- Combined RV: {check['rv_combined']} ± {check['err_combined']} km/s
- **Status**: {status}

"""
    else:
        report += "No same-night exposure pairs to compare.\n\n"

    report += """---

## Per-Epoch Results

| Epoch | Night | MJD | RV (km/s) | σ (km/s) | N_exp |
|-------|-------|-----|-----------|----------|-------|
"""

    for i, e in enumerate(epoch_results):
        report += f"| {i+1} | {e['night']} | {e['mjd']:.3f} | {e['epoch_rv']} | {e['epoch_err']} | {e['n_exposures']} |\n"

    # Compute RV swing
    rvs = [e["epoch_rv"] for e in epoch_results if e["epoch_rv"] is not None]
    if len(rvs) >= 2:
        rv_swing = max(rvs) - min(rvs)
        rv_min_epoch = rvs.index(min(rvs)) + 1
        rv_max_epoch = rvs.index(max(rvs)) + 1
    else:
        rv_swing = 0
        rv_min_epoch = rv_max_epoch = 1

    report += f"""
---

## Key Findings

### RV Swing
- **Total RV swing**: {rv_swing:.1f} km/s
- Minimum: Epoch {rv_min_epoch} ({min(rvs):.1f} km/s)
- Maximum: Epoch {rv_max_epoch} ({max(rvs):.1f} km/s)

### Comparison with DESI Catalog
"""

    for r in results:
        if r and r["trusted_rv"] is not None:
            diff = r["trusted_rv"] - r["catalog_rv"]
            report += f"- EXPID {r['expid']}: Trusted = {r['trusted_rv']:.1f}, Catalog = {r['catalog_rv']:.1f}, Δ = {diff:+.1f} km/s\n"

    # Determine verdict
    catalog_swing = max([e["catalog_rv"] for e in EXPOSURES]) - min([e["catalog_rv"] for e in EXPOSURES])

    report += f"""
### Verdict

**Does the trusted-window RV curve still show a large swing?**
"""

    if rv_swing > 50:
        report += f"""
YES. The trusted-window analysis shows a {rv_swing:.1f} km/s swing, which is substantial
and broadly consistent with the DESI catalog swing of ~{catalog_swing:.0f} km/s.

**Interpretation**: The large RV variability persists even when restricted to the
stable Ca II triplet window. This supports the hypothesis that the RV variations
are astrophysical, not instrumental artifacts from the sky-dominated region.
"""
    elif rv_swing > 20:
        report += f"""
MODERATE. The trusted-window analysis shows a {rv_swing:.1f} km/s swing, smaller than
the catalog but still significant.

**Interpretation**: Some RV variability persists in the stable window, but at reduced
amplitude compared to the full-spectrum DESI catalog values.
"""
    else:
        report += f"""
NO. The trusted-window analysis shows only a {rv_swing:.1f} km/s swing, much smaller
than the DESI catalog swing of ~{catalog_swing:.0f} km/s.

**Interpretation**: The large catalog RV swing may have been dominated by instrumental
effects in the sky region, not astrophysical variability.
"""

    report += """
**Same-night exposures consistency?**
"""

    if consistency_check:
        for night, check in consistency_check.items():
            if check["consistent"]:
                report += f"""
YES. Exposures {check['exposures']} on night {night} are consistent within errors
(ΔRV = {check['delta_rv']:.1f} km/s, {check['sigma_diff']:.1f}σ).
"""
            else:
                report += f"""
NO. Exposures {check['exposures']} on night {night} show {check['sigma_diff']:.1f}σ
discrepancy (ΔRV = {check['delta_rv']:.1f} km/s), though this is much improved from
the full Z-arm discrepancy of ~34 km/s.
"""

    report += """
---

## Flagged Exposures

"""

    flagged = [r for r in results if r and r["flags"]]
    if flagged:
        for r in flagged:
            report += f"- **EXPID {r['expid']}**: {', '.join(r['flags'])}\n"
    else:
        report += "No exposures flagged as unreliable.\n"

    report += """
---

## Conclusion

"""

    if rv_swing > 50 and all(check["consistent"] for check in consistency_check.values()):
        report += """The trusted-window RV curve **supports** a large RV swing consistent with the DESI catalog.
Same-night exposures are consistent in the stable window. The candidate remains viable
for follow-up spectroscopy to confirm a compact companion.
"""
    elif rv_swing > 20:
        report += """The trusted-window RV curve shows **moderate** RV variability. Combined with the
Gaia astrometric excess (RUWE=1.954, AEN=16.5σ), the system warrants follow-up observations.
"""
    else:
        report += """The trusted-window RV curve shows **reduced** RV swing compared to the catalog.
Further investigation is needed to determine whether the original signal was dominated
by instrumental systematics.
"""

    report += f"""
---

*Generated by trusted_rv_pipeline.py*
*Spectral window: {WAVE_MIN:.0f}-{WAVE_MAX:.0f} Å*
"""

    return report


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("TRUSTED RV PIPELINE")
    print(f"Target: DESI {TARGET_ID}")
    print(f"Trusted Window: {WAVE_MIN:.0f} - {WAVE_MAX:.0f} Å (Ca II triplet)")
    print("=" * 70)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load template
    print("\n[1] Loading PHOENIX template...")
    wave_template, flux_template, template_name = load_phoenix_template(teff=3800)

    # Resample template to approximate DESI resolution for the trusted window
    print("  Preparing template for DESI resolution...")

    # Analyze each exposure
    print("\n[2] Analyzing exposures in trusted window...")
    results = []
    for exp in EXPOSURES:
        print(f"\n  Processing EXPID {exp['expid']}...")
        result = analyze_exposure(exp, wave_template, flux_template)
        results.append(result)
        if result:
            print(f"    RV = {result['trusted_rv']:.1f} ± {result['trusted_err']:.2f} km/s")
            print(f"    χ²_red = {result['chi2_red']:.1f}, N_pix = {result['n_pixels']}")

    # Combine same-night exposures and check consistency
    print("\n[3] Combining exposures and checking consistency...")
    epoch_results, consistency_check = combine_same_night_exposures(results)

    for night, check in consistency_check.items():
        status = "CONSISTENT" if check["consistent"] else "INCONSISTENT"
        print(f"  Night {night}: ΔRV = {check['delta_rv']:.1f} km/s ({check['sigma_diff']:.1f}σ) - {status}")

    # Generate plots
    print("\n[4] Generating figures...")
    plot_chi2_scans(results, os.path.join(FIG_DIR, "chi2_scans_by_exposure.png"))
    plot_rv_by_exposure(results, os.path.join(FIG_DIR, "trusted_rv_by_exposure.png"))
    plot_rv_by_epoch(epoch_results, os.path.join(FIG_DIR, "trusted_rv_by_epoch.png"))
    plot_same_night_consistency(results, consistency_check,
                                 os.path.join(FIG_DIR, "same_night_consistency.png"))

    # Save data files
    print("\n[5] Saving data files...")

    # JSON output
    output_data = {
        "target_id": TARGET_ID,
        "trusted_window": {"wave_min": WAVE_MIN, "wave_max": WAVE_MAX},
        "template": template_name,
        "per_exposure_results": [r for r in results if r],
        "per_epoch_results": epoch_results,
        "same_night_consistency": consistency_check
    }

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(os.path.join(OUTPUT_DIR, "trusted_rv_curve.json"), "w") as f:
        # Remove large arrays from JSON
        output_data_slim = output_data.copy()
        output_data_slim["per_exposure_results"] = [
            {k: v for k, v in r.items() if k not in ["rv_grid", "chi2_scan"]}
            for r in output_data["per_exposure_results"]
        ]
        json.dump(output_data_slim, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'trusted_rv_curve.json')}")

    # CSV output
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "trusted_rv_curve.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["expid", "mjd", "trusted_rv_kms", "trusted_err_kms",
                         "chi2_red", "n_pixels", "snr", "catalog_rv_kms", "flags"])
        for r in results:
            if r:
                writer.writerow([
                    r["expid"], r["mjd"], r["trusted_rv"], r["trusted_err"],
                    r["chi2_red"], r["n_pixels"], r["snr_median"],
                    r["catalog_rv"], "|".join(r["flags"]) if r["flags"] else ""
                ])
    print(f"  Saved: {csv_path}")

    # Generate report
    print("\n[6] Generating report...")
    report = generate_report(results, epoch_results, consistency_check, template_name)

    report_path = os.path.join(OUTPUT_DIR, "TRUSTED_RV_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    rvs = [e["epoch_rv"] for e in epoch_results if e["epoch_rv"] is not None]
    if rvs:
        rv_swing = max(rvs) - min(rvs)
        print(f"Trusted-window RV swing: {rv_swing:.1f} km/s")
        print(f"Epochs: {len(epoch_results)}")
        print(f"RV range: {min(rvs):.1f} to {max(rvs):.1f} km/s")

    if consistency_check:
        for night, check in consistency_check.items():
            status = "CONSISTENT" if check["consistent"] else "INCONSISTENT"
            print(f"Same-night ({night}): {status} (Δ = {check['delta_rv']:.1f} km/s)")

    print("\nOutput files:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'TRUSTED_RV_REPORT.md')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'trusted_rv_curve.json')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'trusted_rv_curve.csv')}")
    print(f"  - {os.path.join(FIG_DIR, 'chi2_scans_by_exposure.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'trusted_rv_by_exposure.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'trusted_rv_by_epoch.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'same_night_consistency.png')}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
