#!/usr/bin/env python3
"""
DESI BLEND-AWARE RV RE-MEASUREMENT v2

Proper methodology using:
- PHOENIX model templates (M0 for primary, M5 for neighbor)
- χ² minimization in flux space with ivar weights
- Flux-ratio prior for the known Gaia neighbor
- Cross-epoch constant-v2 test

Target: Gaia DR3 3802130935635096832 / DESI TARGETID 39627745210139276
Neighbor: sep=0.688", ΔG=2.21, expected flux ratio ~0.13

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
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp1d
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
NEIGHBOR_DELTA_G = 2.21099
NEIGHBOR_FLUX_FRAC = 10**(-0.4 * NEIGHBOR_DELTA_G)  # ~0.13

# Paths
BASE_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH")
SPECTRA_DIR = BASE_DIR / "data" / "desi_spectra"
TEMPLATE_DIR = BASE_DIR / "data" / "templates"
OUTPUT_DIR = BASE_DIR / "outputs" / "desi_blend_v2"
FIGURES_DIR = OUTPUT_DIR / "figures"

# DESI epochs
EPOCHS = [
    {'mjd': 59568.48825, 'tile': 20268, 'night': 20211219, 'petal': 2,
     'fiber': 1377, 'rv_catalog': -86.39, 'rv_err_catalog': 0.55,
     'coadd_file': 'coadd_20268_20211219_p2.fits', 'label': 'Epoch1'},
    {'mjd': 59605.38003, 'tile': 24976, 'night': 20220125, 'petal': 7,
     'fiber': 3816, 'rv_catalog': 59.68, 'rv_err_catalog': 0.83,
     'coadd_file': 'coadd_24976_20220125_p7.fits', 'label': 'Epoch2'},
    {'mjd': 59607.38, 'tile': 23137, 'night': 20220127, 'petal': 0,
     'fiber': 68, 'rv_catalog': 25.8, 'rv_err_catalog': 0.8,
     'coadd_file': 'coadd_23137_20220127_p0.fits', 'label': 'Epoch3'},
]

# Telluric regions to mask
TELLURIC_REGIONS = [
    (6270, 6330),   # O2 B-band
    (6860, 6960),   # O2 A-band
    (7160, 7340),   # H2O
    (7590, 7700),   # O2 extended
    (8130, 8350),   # H2O
    (8940, 9200),   # H2O strong
    (9300, 9800),   # H2O very strong
]

# Analysis wavelength range (R+Z bands, avoiding tellurics)
WL_MIN = 6000
WL_MAX = 8800

# RV search parameters
C_KMS = 299792.458
V_RANGE = (-200, 200)  # km/s
V_STEP = 1.0  # km/s for grid search

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_file_hash(filepath):
    """Compute SHA256 hash."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def apply_telluric_mask(wavelength, ivar):
    """Set ivar to 0 in telluric regions."""
    mask = np.ones(len(wavelength), dtype=bool)
    for wl_min, wl_max in TELLURIC_REGIONS:
        mask &= ~((wavelength >= wl_min) & (wavelength <= wl_max))
    ivar_out = ivar.copy()
    ivar_out[~mask] = 0
    return ivar_out

def continuum_normalize(wavelength, flux, ivar, window=150):
    """Robust continuum normalization preserving ivar."""
    good = (ivar > 0) & np.isfinite(flux) & (flux > 0)
    if np.sum(good) < 100:
        return flux, ivar

    # Running percentile (75th) for continuum
    continuum = np.ones_like(flux)
    for i in range(len(wavelength)):
        wl_lo = wavelength[i] - window/2
        wl_hi = wavelength[i] + window/2
        in_win = (wavelength >= wl_lo) & (wavelength <= wl_hi) & good
        if np.sum(in_win) > 20:
            continuum[i] = np.percentile(flux[in_win], 75)

    # Smooth
    continuum = gaussian_filter1d(continuum, sigma=30)
    continuum[continuum <= 0] = 1.0

    flux_norm = flux / continuum
    # ivar transforms as ivar_norm = ivar * continuum^2
    ivar_norm = ivar * continuum**2

    return flux_norm, ivar_norm, continuum

# =============================================================================
# TEMPLATE LOADING
# =============================================================================

def load_phoenix_templates():
    """Load PHOENIX M0 and M5 templates."""
    print("Loading PHOENIX templates...")

    # Load wavelength grid
    with fits.open(TEMPLATE_DIR / "WAVE_PHOENIX.fits") as hdu:
        wave_full = hdu[0].data  # Angstrom, vacuum

    # Load M0 template (Teff=3800K, logg=4.5)
    with fits.open(TEMPLATE_DIR / "phoenix_m0_3800_4.5.fits") as hdu:
        flux_m0_full = hdu[0].data

    # Load M5 template (Teff=3000K, logg=5.0)
    with fits.open(TEMPLATE_DIR / "phoenix_m5_3000_5.0.fits") as hdu:
        flux_m5_full = hdu[0].data

    # Trim to DESI range with margin
    margin = 500  # Å for RV shifts
    mask = (wave_full >= WL_MIN - margin) & (wave_full <= WL_MAX + margin)

    wave = wave_full[mask]
    flux_m0 = flux_m0_full[mask]
    flux_m5 = flux_m5_full[mask]

    # Normalize templates to have median ~1 in the range
    flux_m0 = flux_m0 / np.median(flux_m0)
    flux_m5 = flux_m5 / np.median(flux_m5)

    print(f"  M0 template: {len(wave)} pixels, λ=[{wave[0]:.0f}, {wave[-1]:.0f}] Å")
    print(f"  M5 template: {len(wave)} pixels")

    return {
        'wavelength': wave,
        'M0': flux_m0,
        'M5': flux_m5,
    }

# =============================================================================
# SPECTRUM LOADING
# =============================================================================

def load_desi_spectrum(epoch_info):
    """Load and preprocess DESI spectrum."""
    fpath = SPECTRA_DIR / epoch_info['coadd_file']

    result = {
        'file': str(fpath),
        'file_hash': compute_file_hash(fpath),
        'label': epoch_info['label'],
        'mjd': epoch_info['mjd'],
        'rv_catalog': epoch_info['rv_catalog'],
    }

    with fits.open(fpath) as hdu:
        fmap = hdu['FIBERMAP'].data
        mask = fmap['TARGETID'] == TARGET_ID
        idx = np.where(mask)[0][0]

        # Combine R and Z bands
        wl_r = hdu['R_WAVELENGTH'].data
        fl_r = hdu['R_FLUX'].data[idx]
        iv_r = hdu['R_IVAR'].data[idx]

        wl_z = hdu['Z_WAVELENGTH'].data
        fl_z = hdu['Z_FLUX'].data[idx]
        iv_z = hdu['Z_IVAR'].data[idx]

        # Apply telluric mask
        iv_r = apply_telluric_mask(wl_r, iv_r)
        iv_z = apply_telluric_mask(wl_z, iv_z)

        # Concatenate (R ends at 7620, Z starts at 7520)
        # Use non-overlapping regions
        r_mask = wl_r < 7550
        z_mask = wl_z >= 7550

        wl = np.concatenate([wl_r[r_mask], wl_z[z_mask]])
        fl = np.concatenate([fl_r[r_mask], fl_z[z_mask]])
        iv = np.concatenate([iv_r[r_mask], iv_z[z_mask]])

        # Continuum normalize
        fl_norm, iv_norm, continuum = continuum_normalize(wl, fl, iv)

        # Trim to analysis range
        in_range = (wl >= WL_MIN) & (wl <= WL_MAX)

        result['wavelength'] = wl[in_range]
        result['flux'] = fl[in_range]
        result['flux_norm'] = fl_norm[in_range]
        result['ivar'] = iv[in_range]
        result['ivar_norm'] = iv_norm[in_range]
        result['continuum'] = continuum[in_range]

        # SNR estimate
        good = iv_norm[in_range] > 0
        if np.sum(good) > 0:
            result['snr_median'] = np.median(fl_norm[in_range][good] * np.sqrt(iv_norm[in_range][good]))
        else:
            result['snr_median'] = 0

    return result

# =============================================================================
# CHI-SQUARED RV FITTING
# =============================================================================

def shift_template(template_wave, template_flux, velocity_kms, target_wave):
    """
    Shift template by velocity and interpolate to target wavelength grid.

    Doppler shift: λ_obs = λ_rest * (1 + v/c)
    So to shift template to observed frame: λ_rest = λ_obs / (1 + v/c)
    """
    # Shift factor
    z = velocity_kms / C_KMS

    # Template wavelengths in observed frame
    wave_shifted = template_wave * (1 + z)

    # Interpolate to target grid
    f_interp = interp1d(wave_shifted, template_flux, kind='linear',
                        bounds_error=False, fill_value=np.nan)

    return f_interp(target_wave)

def chi2_single_template(params, data_wave, data_flux, data_ivar,
                          template_wave, template_flux):
    """
    χ² for single-template model: F_model = A * T(v)
    params: [velocity_kms, amplitude]
    """
    v, A = params

    # Shift template
    model = A * shift_template(template_wave, template_flux, v, data_wave)

    # Valid pixels
    good = (data_ivar > 0) & np.isfinite(model) & np.isfinite(data_flux)

    if np.sum(good) < 100:
        return 1e10

    # χ²
    resid = data_flux[good] - model[good]
    chi2 = np.sum(resid**2 * data_ivar[good])

    return chi2

def fit_single_template(data, template, v_grid=None):
    """
    Fit single-template model to data.
    Returns best v, σ_v, amplitude, χ².
    """
    if v_grid is None:
        v_grid = np.arange(V_RANGE[0], V_RANGE[1] + V_STEP, V_STEP)

    data_wave = data['wavelength']
    data_flux = data['flux_norm']
    data_ivar = data['ivar_norm']

    template_wave = template['wavelength']
    template_flux = template['M0']

    # Grid search for initial guess
    chi2_grid = np.zeros(len(v_grid))

    for i, v in enumerate(v_grid):
        # Optimal amplitude for this velocity
        model = shift_template(template_wave, template_flux, v, data_wave)
        good = (data_ivar > 0) & np.isfinite(model) & np.isfinite(data_flux)

        if np.sum(good) < 100:
            chi2_grid[i] = 1e10
            continue

        # Weighted least squares for amplitude: A = sum(w*d*m) / sum(w*m^2)
        w = data_ivar[good]
        d = data_flux[good]
        m = model[good]

        A_opt = np.sum(w * d * m) / np.sum(w * m**2)

        resid = d - A_opt * m
        chi2_grid[i] = np.sum(resid**2 * w)

    # Find minimum
    i_min = np.argmin(chi2_grid)
    v_best = v_grid[i_min]
    chi2_min = chi2_grid[i_min]

    # Refine with optimizer
    def cost(params):
        return chi2_single_template(params, data_wave, data_flux, data_ivar,
                                    template_wave, template_flux)

    # Get optimal amplitude at v_best
    model_best = shift_template(template_wave, template_flux, v_best, data_wave)
    good = (data_ivar > 0) & np.isfinite(model_best)
    A_init = np.sum(data_ivar[good] * data_flux[good] * model_best[good]) / \
             np.sum(data_ivar[good] * model_best[good]**2)

    result = minimize(cost, [v_best, A_init], method='L-BFGS-B',
                     bounds=[(-200, 200), (0.5, 2.0)])

    v_fit = result.x[0]
    A_fit = result.x[1]
    chi2_fit = result.fun

    # Compute velocity error from χ² curvature (Δχ²=1 for 1σ)
    # Numerical estimate
    dv = 0.5
    chi2_plus = cost([v_fit + dv, A_fit])
    chi2_minus = cost([v_fit - dv, A_fit])

    d2chi2_dv2 = (chi2_plus - 2*chi2_fit + chi2_minus) / dv**2

    if d2chi2_dv2 > 0:
        sigma_v = np.sqrt(2.0 / d2chi2_dv2)
    else:
        sigma_v = 10.0  # fallback

    # Number of valid pixels
    good = (data_ivar > 0)
    n_pix = np.sum(good)
    n_params = 2
    dof = n_pix - n_params
    chi2_reduced = chi2_fit / dof if dof > 0 else np.inf

    return {
        'v': v_fit,
        'v_err': sigma_v,
        'amplitude': A_fit,
        'chi2': chi2_fit,
        'chi2_reduced': chi2_reduced,
        'n_pix': n_pix,
        'dof': dof,
        'chi2_grid': chi2_grid,
        'v_grid': v_grid,
    }

# =============================================================================
# TWO-TEMPLATE BLEND MODEL
# =============================================================================

def chi2_two_template(params, data_wave, data_flux, data_ivar,
                       template_wave, template_m0, template_m5,
                       flux_ratio_prior=None):
    """
    χ² for two-template model: F_model = A * [T_M0(v1) + b * T_M5(v2)]
    params: [v1, v2, A, b]

    flux_ratio_prior: if provided, add prior penalty on b
    """
    v1, v2, A, b = params

    # Shift templates
    t_m0 = shift_template(template_wave, template_m0, v1, data_wave)
    t_m5 = shift_template(template_wave, template_m5, v2, data_wave)

    model = A * (t_m0 + b * t_m5)

    # Valid pixels
    good = (data_ivar > 0) & np.isfinite(model) & np.isfinite(data_flux)

    if np.sum(good) < 100:
        return 1e10

    # χ²
    resid = data_flux[good] - model[good]
    chi2 = np.sum(resid**2 * data_ivar[good])

    # Add flux ratio prior if provided (lognormal)
    if flux_ratio_prior is not None:
        b_expected, b_sigma_log = flux_ratio_prior
        # lognormal prior: -2*ln(P) = ((ln(b) - ln(b_expected))/sigma)^2
        if b > 0:
            prior_penalty = ((np.log(b) - np.log(b_expected)) / b_sigma_log)**2
            chi2 += prior_penalty
        else:
            chi2 += 1e6

    return chi2

def fit_two_template(data, template, v1_init, flux_ratio_prior=None):
    """
    Fit two-template blend model.

    flux_ratio_prior: (b_expected, sigma_log) for lognormal prior on b
    """
    data_wave = data['wavelength']
    data_flux = data['flux_norm']
    data_ivar = data['ivar_norm']

    template_wave = template['wavelength']
    template_m0 = template['M0']
    template_m5 = template['M5']

    def cost(params):
        return chi2_two_template(params, data_wave, data_flux, data_ivar,
                                  template_wave, template_m0, template_m5,
                                  flux_ratio_prior)

    # Initial guess: v1 from single template, v2=0, A=1, b=0.13
    v2_init = 0.0
    A_init = 1.0
    b_init = NEIGHBOR_FLUX_FRAC

    # Bounds
    bounds = [(-200, 200), (-200, 200), (0.5, 2.0), (0.01, 0.5)]

    result = minimize(cost, [v1_init, v2_init, A_init, b_init],
                     method='L-BFGS-B', bounds=bounds)

    v1_fit, v2_fit, A_fit, b_fit = result.x
    chi2_fit = result.fun

    # Compute v1 error
    dv = 0.5
    chi2_plus = cost([v1_fit + dv, v2_fit, A_fit, b_fit])
    chi2_minus = cost([v1_fit - dv, v2_fit, A_fit, b_fit])
    d2chi2 = (chi2_plus - 2*chi2_fit + chi2_minus) / dv**2
    sigma_v1 = np.sqrt(2.0 / d2chi2) if d2chi2 > 0 else 10.0

    # Number of parameters
    good = (data_ivar > 0)
    n_pix = np.sum(good)
    n_params = 4
    dof = n_pix - n_params
    chi2_reduced = chi2_fit / dof if dof > 0 else np.inf

    return {
        'v1': v1_fit,
        'v1_err': sigma_v1,
        'v2': v2_fit,
        'A': A_fit,
        'b': b_fit,
        'chi2': chi2_fit,
        'chi2_reduced': chi2_reduced,
        'n_pix': n_pix,
        'dof': dof,
    }

# =============================================================================
# CROSS-EPOCH CONSTANT-V2 MODEL
# =============================================================================

def fit_constant_v2_model(all_data, template, flux_ratio_prior=None):
    """
    Fit a model where v2 (neighbor RV) is constant across all epochs,
    but v1 (primary RV) varies per epoch.

    This tests whether the apparent RV swing could be explained by
    component switching with a background/constant neighbor.
    """
    template_wave = template['wavelength']
    template_m0 = template['M0']
    template_m5 = template['M5']

    n_epochs = len(all_data)

    def total_chi2(params):
        """
        params: [v1_ep1, v1_ep2, v1_ep3, v2_shared, A1, A2, A3, b]
        """
        v1_list = params[:n_epochs]
        v2 = params[n_epochs]
        A_list = params[n_epochs+1:2*n_epochs+1]
        b = params[2*n_epochs+1]

        chi2_total = 0.0

        for i, data in enumerate(all_data):
            data_wave = data['wavelength']
            data_flux = data['flux_norm']
            data_ivar = data['ivar_norm']

            t_m0 = shift_template(template_wave, template_m0, v1_list[i], data_wave)
            t_m5 = shift_template(template_wave, template_m5, v2, data_wave)

            model = A_list[i] * (t_m0 + b * t_m5)

            good = (data_ivar > 0) & np.isfinite(model) & np.isfinite(data_flux)
            if np.sum(good) < 100:
                chi2_total += 1e8
                continue

            resid = data_flux[good] - model[good]
            chi2_total += np.sum(resid**2 * data_ivar[good])

        # Flux ratio prior
        if flux_ratio_prior is not None:
            b_expected, b_sigma_log = flux_ratio_prior
            if b > 0:
                chi2_total += ((np.log(b) - np.log(b_expected)) / b_sigma_log)**2

        return chi2_total

    # Initial guesses from single-template fits
    v1_inits = [d['rv_catalog'] for d in all_data]  # Use catalog as starting point
    v2_init = 0.0
    A_inits = [1.0] * n_epochs
    b_init = NEIGHBOR_FLUX_FRAC

    x0 = v1_inits + [v2_init] + A_inits + [b_init]

    # Bounds
    bounds = [(-200, 200)] * n_epochs + \
             [(-200, 200)] + \
             [(0.5, 2.0)] * n_epochs + \
             [(0.01, 0.5)]

    result = minimize(total_chi2, x0, method='L-BFGS-B', bounds=bounds)

    # Parse results
    v1_fits = result.x[:n_epochs]
    v2_fit = result.x[n_epochs]
    A_fits = result.x[n_epochs+1:2*n_epochs+1]
    b_fit = result.x[2*n_epochs+1]
    chi2_total = result.fun

    # Count total DOF
    n_pix_total = sum(np.sum(d['ivar_norm'] > 0) for d in all_data)
    n_params = 2 * n_epochs + 2  # v1*3 + v2 + A*3 + b
    dof_total = n_pix_total - n_params

    return {
        'v1_per_epoch': list(v1_fits),
        'v2_shared': v2_fit,
        'A_per_epoch': list(A_fits),
        'b': b_fit,
        'chi2_total': chi2_total,
        'chi2_reduced': chi2_total / dof_total if dof_total > 0 else np.inf,
        'n_pix_total': n_pix_total,
        'n_params': n_params,
        'dof': dof_total,
    }

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("="*70)
    print("DESI BLEND-AWARE RV RE-MEASUREMENT v2")
    print("="*70)
    print(f"\nTarget: Gaia DR3 {GAIA_SOURCE_ID}")
    print(f"DESI TARGETID: {TARGET_ID}")
    print(f"\nNeighbor: sep={NEIGHBOR_SEP_ARCSEC:.3f}\", ΔG={NEIGHBOR_DELTA_G:.2f}")
    print(f"Expected flux ratio: {NEIGHBOR_FLUX_FRAC:.3f}")

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load templates
    template = load_phoenix_templates()

    # Load spectra
    print("\nLoading DESI spectra...")
    spectra = []
    file_hashes = {}

    for epoch in EPOCHS:
        print(f"  {epoch['label']}: {epoch['coadd_file']}")
        spec = load_desi_spectrum(epoch)
        spectra.append(spec)
        file_hashes[spec['file']] = spec['file_hash']
        print(f"    SNR ~ {spec['snr_median']:.1f}, {len(spec['wavelength'])} pixels")

    # =================================================================
    # PART 1: SINGLE-TEMPLATE RV FITS
    # =================================================================
    print("\n" + "="*70)
    print("PART 1: SINGLE-TEMPLATE (M0) RV FITTING")
    print("="*70)

    single_results = []

    for spec in spectra:
        print(f"\n{spec['label']} (Catalog RV: {spec['rv_catalog']:+.1f} km/s)")

        result = fit_single_template(spec, template)
        result['label'] = spec['label']
        result['mjd'] = spec['mjd']
        result['rv_catalog'] = spec['rv_catalog']

        print(f"  Fitted RV: {result['v']:+.1f} ± {result['v_err']:.1f} km/s")
        print(f"  Amplitude: {result['amplitude']:.3f}")
        print(f"  χ²/dof: {result['chi2']:.1f}/{result['dof']} = {result['chi2_reduced']:.2f}")

        single_results.append(result)

    # Compare fitted vs catalog RVs
    print("\n--- Single-Template RV Comparison ---")
    print(f"{'Epoch':<10} {'Catalog':>10} {'Fitted':>12} {'Diff':>10}")
    for r in single_results:
        diff = r['v'] - r['rv_catalog']
        print(f"{r['label']:<10} {r['rv_catalog']:>+10.1f} {r['v']:>+10.1f}±{r['v_err']:.1f} {diff:>+10.1f}")

    # RV swing from fitted values
    rv_fitted = [r['v'] for r in single_results]
    rv_swing_fitted = max(rv_fitted) - min(rv_fitted)
    rv_swing_catalog = max(r['rv_catalog'] for r in single_results) - \
                       min(r['rv_catalog'] for r in single_results)

    print(f"\nCatalog RV swing: {rv_swing_catalog:.1f} km/s")
    print(f"Fitted RV swing:  {rv_swing_fitted:.1f} km/s")

    # =================================================================
    # PART 2: TWO-TEMPLATE BLEND FITS (PER-EPOCH)
    # =================================================================
    print("\n" + "="*70)
    print("PART 2: TWO-TEMPLATE (M0+M5) BLEND FITS")
    print("="*70)

    # Flux ratio prior: lognormal centered on 0.13 with 0.5 dex width
    # This allows range roughly 0.04 to 0.4
    flux_ratio_prior = (NEIGHBOR_FLUX_FRAC, 0.5)
    print(f"\nFlux ratio prior: b ~ lognormal({NEIGHBOR_FLUX_FRAC:.3f}, σ_log=0.5)")
    print(f"  Effective range: {NEIGHBOR_FLUX_FRAC * np.exp(-1):.3f} to {NEIGHBOR_FLUX_FRAC * np.exp(1):.3f}")

    blend_results = []

    for i, spec in enumerate(spectra):
        print(f"\n{spec['label']}")

        v1_init = single_results[i]['v']
        result = fit_two_template(spec, template, v1_init, flux_ratio_prior)
        result['label'] = spec['label']

        print(f"  v1 (primary): {result['v1']:+.1f} ± {result['v1_err']:.1f} km/s")
        print(f"  v2 (neighbor): {result['v2']:+.1f} km/s")
        print(f"  Amplitude A: {result['A']:.3f}")
        print(f"  Flux ratio b: {result['b']:.3f} (expected {NEIGHBOR_FLUX_FRAC:.3f})")
        print(f"  χ²/dof: {result['chi2']:.1f}/{result['dof']} = {result['chi2_reduced']:.2f}")

        blend_results.append(result)

    # Model comparison: single vs two-template
    print("\n--- Model Comparison (per-epoch) ---")
    print(f"{'Epoch':<10} {'χ²_single':>12} {'χ²_blend':>12} {'Δχ²':>10} {'ΔBIC':>10}")

    for sr, br in zip(single_results, blend_results):
        delta_chi2 = sr['chi2'] - br['chi2']
        # BIC = χ² + k*ln(n)
        k_single = 2
        k_blend = 4
        n = sr['n_pix']
        bic_single = sr['chi2'] + k_single * np.log(n)
        bic_blend = br['chi2'] + k_blend * np.log(n)
        delta_bic = bic_blend - bic_single

        print(f"{sr['label']:<10} {sr['chi2']:>12.1f} {br['chi2']:>12.1f} {delta_chi2:>+10.1f} {delta_bic:>+10.1f}")

        sr['bic'] = bic_single
        br['bic'] = bic_blend
        br['delta_chi2'] = delta_chi2
        br['delta_bic'] = delta_bic

    # =================================================================
    # PART 3: CONSTANT-V2 CROSS-EPOCH MODEL
    # =================================================================
    print("\n" + "="*70)
    print("PART 3: CROSS-EPOCH CONSTANT-v2 MODEL")
    print("="*70)

    print("\nTesting: v2 (neighbor) is constant across all epochs")
    print("         v1 (primary) varies freely per epoch")

    const_v2_result = fit_constant_v2_model(spectra, template, flux_ratio_prior)

    print(f"\nResults:")
    print(f"  v2 (shared): {const_v2_result['v2_shared']:+.1f} km/s")
    print(f"  Flux ratio b: {const_v2_result['b']:.3f}")
    for i, (v1, A) in enumerate(zip(const_v2_result['v1_per_epoch'],
                                     const_v2_result['A_per_epoch'])):
        print(f"  {spectra[i]['label']}: v1={v1:+.1f} km/s, A={A:.3f}")

    print(f"\n  Total χ²: {const_v2_result['chi2_total']:.1f}")
    print(f"  χ²/dof: {const_v2_result['chi2_reduced']:.2f}")

    # Compare to sum of per-epoch single-template fits
    chi2_single_total = sum(r['chi2'] for r in single_results)
    dof_single_total = sum(r['dof'] for r in single_results)
    n_params_single = 2 * len(spectra)  # v, A per epoch

    print(f"\n  For comparison:")
    print(f"    Sum of single-template χ²: {chi2_single_total:.1f}")
    print(f"    Δχ² (single - const_v2): {chi2_single_total - const_v2_result['chi2_total']:+.1f}")

    # Is constant-v2 model favored?
    const_v2_result['chi2_single_total'] = chi2_single_total
    const_v2_result['delta_chi2_vs_single'] = chi2_single_total - const_v2_result['chi2_total']

    # =================================================================
    # PART 4: VERDICT
    # =================================================================
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    verdicts = {}

    # Check 1: Single-template RV consistency with catalog
    rv_diffs = [r['v'] - r['rv_catalog'] for r in single_results]
    max_diff = max(abs(d) for d in rv_diffs)
    rms_diff = np.sqrt(np.mean([d**2 for d in rv_diffs]))

    print(f"\n1. Single-template RV vs catalog:")
    print(f"   Max |Δ|: {max_diff:.1f} km/s, RMS: {rms_diff:.1f} km/s")

    if max_diff < 10 and rms_diff < 5:
        verdicts['single_template_consistency'] = 'PASS'
        print("   → PASS: Fitted RVs consistent with DESI catalog")
    elif max_diff < 30:
        verdicts['single_template_consistency'] = 'INCONCLUSIVE'
        print("   → INCONCLUSIVE: Moderate differences")
    else:
        verdicts['single_template_consistency'] = 'FAIL'
        print("   → FAIL: Large RV discrepancies")

    # Check 2: Two-template model preference
    n_prefer_blend = sum(1 for br in blend_results if br['delta_bic'] < -6)

    print(f"\n2. Two-template model preference (ΔBIC < -6):")
    print(f"   Epochs preferring blend: {n_prefer_blend}/{len(blend_results)}")

    if n_prefer_blend == 0:
        verdicts['two_template_preference'] = 'PASS'
        print("   → PASS: Single-template sufficient for all epochs")
    elif n_prefer_blend == len(blend_results):
        verdicts['two_template_preference'] = 'FAIL'
        print("   → FAIL: All epochs prefer blend model")
    else:
        verdicts['two_template_preference'] = 'INCONCLUSIVE'
        print("   → INCONCLUSIVE: Mixed results")

    # Check 3: Flux ratio consistency
    b_values = [br['b'] for br in blend_results]
    b_mean = np.mean(b_values)
    b_std = np.std(b_values)

    print(f"\n3. Fitted flux ratios (expected ~{NEIGHBOR_FLUX_FRAC:.3f}):")
    print(f"   Mean b: {b_mean:.3f} ± {b_std:.3f}")

    if abs(np.log(b_mean / NEIGHBOR_FLUX_FRAC)) < 0.5:  # within factor of ~1.6
        verdicts['flux_ratio_physical'] = 'CONSISTENT'
        print("   → CONSISTENT with known neighbor")
    else:
        verdicts['flux_ratio_physical'] = 'INCONSISTENT'
        print("   → INCONSISTENT with known neighbor flux ratio")

    # Check 4: Constant-v2 model
    delta_chi2_const = const_v2_result['delta_chi2_vs_single']

    print(f"\n4. Constant-v2 (background neighbor) model:")
    print(f"   Δχ² improvement vs single-template: {delta_chi2_const:+.1f}")

    # Extra DOF for constant-v2: +1 (v2) + 1 (b) = 2 extra params
    # But it's a more constrained model (v2 shared), so comparison is complex
    if delta_chi2_const > 10:
        verdicts['constant_v2_model'] = 'FAVORED'
        print("   → Model slightly favored, but...")
    else:
        verdicts['constant_v2_model'] = 'NOT_FAVORED'
        print("   → NOT strongly favored")

    # Check 5: Can 13% blend explain 146 km/s swing?
    # Physics: ΔRV_max ≈ b * |v1 - v2|_max
    # For b=0.13 and observed swing 146 km/s in catalog,
    # would need |v1 - v2| ~ 1100 km/s, which is unphysical for a neighbor

    print(f"\n5. Physics check: Can ~13% blend explain 146 km/s swing?")
    max_plausible_blend_effect = NEIGHBOR_FLUX_FRAC * 200  # assume max 200 km/s neighbor offset
    print(f"   Max blend-induced shift (b=0.13, v_offset=200): ~{max_plausible_blend_effect:.0f} km/s")
    print(f"   Observed catalog swing: {rv_swing_catalog:.0f} km/s")

    if rv_swing_catalog > 5 * max_plausible_blend_effect:
        verdicts['blend_explains_swing'] = 'NO'
        print("   → NO: Blend cannot explain the amplitude")
    else:
        verdicts['blend_explains_swing'] = 'POSSIBLY'
        print("   → Possibly could contribute")

    # Overall verdict
    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)

    n_pass = sum(1 for v in verdicts.values() if v in ['PASS', 'NO', 'NOT_FAVORED', 'CONSISTENT'])
    n_fail = sum(1 for v in verdicts.values() if v in ['FAIL', 'YES', 'FAVORED', 'INCONSISTENT'])

    print("\nVerdict Summary:")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")

    if verdicts['blend_explains_swing'] == 'NO' and \
       verdicts['two_template_preference'] in ['PASS', 'INCONCLUSIVE']:
        overall_verdict = 'ROBUST'
        bottom_line = "DESI RV swing is ROBUST: the ~13% neighbor blend cannot explain the 146 km/s amplitude."
    elif verdicts['two_template_preference'] == 'FAIL' and \
         verdicts['flux_ratio_physical'] == 'CONSISTENT':
        overall_verdict = 'COMPROMISED'
        bottom_line = "DESI RV swing may be COMPROMISED by blending; further investigation required."
    else:
        overall_verdict = 'INCONCLUSIVE'
        bottom_line = "Results are INCONCLUSIVE; high-resolution spectroscopy recommended."

    print(f"\n**OVERALL: {overall_verdict}**")
    print(f"\n{bottom_line}")

    # =================================================================
    # GENERATE FIGURES
    # =================================================================
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    # Figure 1: χ² vs velocity for each epoch
    print("\nGenerating chi2_vs_v_by_epoch.png...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (sr, spec) in enumerate(zip(single_results, spectra)):
        ax = axes[i]
        v_grid = sr['v_grid']
        chi2_grid = sr['chi2_grid']

        # Normalize to minimum
        chi2_min = np.min(chi2_grid)
        delta_chi2 = chi2_grid - chi2_min

        ax.plot(v_grid, delta_chi2, 'b-', lw=1)
        ax.axvline(sr['v'], color='r', ls='--', label=f"v={sr['v']:.1f} km/s")
        ax.axvline(sr['rv_catalog'], color='g', ls=':', label=f"catalog={sr['rv_catalog']:.0f}")
        ax.axhline(1, color='gray', ls=':', alpha=0.5)
        ax.axhline(4, color='gray', ls=':', alpha=0.5)

        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Δχ²')
        ax.set_title(f"{spec['label']} (MJD {spec['mjd']:.1f})")
        ax.set_xlim(-150, 150)
        ax.set_ylim(0, min(50, np.max(delta_chi2[np.abs(v_grid) < 150])))
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'chi2_vs_v_by_epoch.png', dpi=150)
    plt.close()

    # Figure 2: RV comparison
    print("Generating rv_by_method_v2.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = [spec['label'] for spec in spectra]
    x = np.arange(len(epochs))
    width = 0.25

    rv_catalog = [spec['rv_catalog'] for spec in spectra]
    rv_single = [r['v'] for r in single_results]
    rv_blend = [r['v1'] for r in blend_results]

    ax.bar(x - width, rv_catalog, width, label='DESI Catalog', alpha=0.8)
    ax.bar(x, rv_single, width, label='Single-template (M0)', alpha=0.8)
    ax.bar(x + width, rv_blend, width, label='Two-template v1', alpha=0.8)

    # Error bars for single-template
    ax.errorbar(x, rv_single, yerr=[r['v_err'] for r in single_results],
                fmt='none', color='k', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.set_ylabel('RV (km/s)')
    ax.set_title('RV Comparison: Catalog vs Template Fits')
    ax.legend()
    ax.axhline(0, color='k', ls='-', lw=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rv_by_method_v2.png', dpi=150)
    plt.close()

    # Figure 3: Two-template residuals
    print("Generating two_template_residuals_v2.png...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for i, (spec, sr, br) in enumerate(zip(spectra, single_results, blend_results)):
        ax_spec = axes[i, 0]
        ax_resid = axes[i, 1]

        wl = spec['wavelength']
        fl = spec['flux_norm']
        iv = spec['ivar_norm']
        good = iv > 0

        # Single-template model
        model_single = sr['amplitude'] * shift_template(
            template['wavelength'], template['M0'], sr['v'], wl)

        # Two-template model
        model_m0 = shift_template(template['wavelength'], template['M0'], br['v1'], wl)
        model_m5 = shift_template(template['wavelength'], template['M5'], br['v2'], wl)
        model_blend = br['A'] * (model_m0 + br['b'] * model_m5)

        ax_spec.plot(wl[good], fl[good], 'k-', lw=0.3, alpha=0.5, label='Data')
        ax_spec.plot(wl[good], model_single[good], 'b-', lw=0.5, alpha=0.7, label='Single')
        ax_spec.plot(wl[good], model_blend[good], 'r-', lw=0.5, alpha=0.7, label='Blend')
        ax_spec.set_ylabel('Normalized Flux')
        ax_spec.set_title(f"{spec['label']}: v1={br['v1']:.0f}, v2={br['v2']:.0f}, b={br['b']:.3f}")
        ax_spec.legend(fontsize=8)
        ax_spec.set_xlim(6500, 7500)
        ax_spec.set_ylim(0.3, 1.3)

        # Residuals
        resid_single = fl - model_single
        resid_blend = fl - model_blend

        ax_resid.plot(wl[good], resid_single[good], 'b-', lw=0.3, alpha=0.5, label='Single')
        ax_resid.plot(wl[good], resid_blend[good], 'r-', lw=0.3, alpha=0.5, label='Blend')
        ax_resid.axhline(0, color='k', ls='--', lw=0.5)
        ax_resid.set_ylabel('Residual')
        ax_resid.legend(fontsize=8)
        ax_resid.set_xlim(6500, 7500)
        ax_resid.set_ylim(-0.3, 0.3)

        if i == 2:
            ax_spec.set_xlabel('Wavelength (Å)')
            ax_resid.set_xlabel('Wavelength (Å)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'two_template_residuals_v2.png', dpi=150)
    plt.close()

    # Figure 4: CCF-like visualization (χ² scan normalized)
    print("Generating ccf_peak_shapes_v2.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['C0', 'C1', 'C2']
    for i, sr in enumerate(single_results):
        v_grid = sr['v_grid']
        chi2_grid = sr['chi2_grid']

        # Convert to CCF-like (1 - normalized χ²)
        chi2_norm = (chi2_grid - np.min(chi2_grid)) / (np.max(chi2_grid) - np.min(chi2_grid))
        ccf_like = 1 - chi2_norm

        ax.plot(v_grid, ccf_like, '-', color=colors[i], lw=1.5,
                label=f"{sr['label']} (v={sr['v']:.0f})")
        ax.axvline(sr['v'], color=colors[i], ls='--', alpha=0.5)

    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('1 - normalized Δχ² (CCF-like)')
    ax.set_title('RV Sensitivity (χ² scan converted to CCF-like)')
    ax.set_xlim(-150, 150)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ccf_peak_shapes_v2.png', dpi=150)
    plt.close()

    print("All figures generated.")

    # =================================================================
    # SAVE JSON OUTPUTS
    # =================================================================
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    def json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            if np.isfinite(obj):
                return float(obj)
            return None
        return obj

    # Epoch RV refit
    rv_refit = {
        'target_id': TARGET_ID,
        'gaia_source_id': GAIA_SOURCE_ID,
        'file_hashes': file_hashes,
        'template_info': {
            'primary': 'PHOENIX M0 (Teff=3800K, logg=4.5)',
            'neighbor': 'PHOENIX M5 (Teff=3000K, logg=5.0)',
        },
        'single_template_results': [],
        'two_template_results': [],
        'rv_swing': {
            'catalog': rv_swing_catalog,
            'fitted': rv_swing_fitted,
        }
    }

    for sr in single_results:
        rv_refit['single_template_results'].append({
            'label': sr['label'],
            'mjd': sr['mjd'],
            'rv_catalog': sr['rv_catalog'],
            'rv_fitted': sr['v'],
            'rv_err': sr['v_err'],
            'amplitude': sr['amplitude'],
            'chi2': sr['chi2'],
            'chi2_reduced': sr['chi2_reduced'],
            'dof': sr['dof'],
        })

    for br in blend_results:
        rv_refit['two_template_results'].append({
            'label': br['label'],
            'v1': br['v1'],
            'v1_err': br['v1_err'],
            'v2': br['v2'],
            'A': br['A'],
            'b': br['b'],
            'chi2': br['chi2'],
            'chi2_reduced': br['chi2_reduced'],
            'delta_bic': br['delta_bic'],
        })

    with open(OUTPUT_DIR / 'desi_epoch_rv_refit_v2.json', 'w') as f:
        json.dump(rv_refit, f, indent=2, default=json_safe)
    print(f"Saved: {OUTPUT_DIR / 'desi_epoch_rv_refit_v2.json'}")

    # Model comparison
    model_compare = {
        'target_id': TARGET_ID,
        'flux_ratio_prior': {
            'expected': NEIGHBOR_FLUX_FRAC,
            'sigma_log': 0.5,
        },
        'per_epoch_comparison': [],
        'constant_v2_model': const_v2_result,
        'verdicts': verdicts,
        'overall_verdict': overall_verdict,
    }

    for sr, br in zip(single_results, blend_results):
        model_compare['per_epoch_comparison'].append({
            'label': sr['label'],
            'chi2_single': sr['chi2'],
            'chi2_blend': br['chi2'],
            'delta_chi2': br['delta_chi2'],
            'delta_bic': br['delta_bic'],
            'blend_preferred': br['delta_bic'] < -6,
        })

    with open(OUTPUT_DIR / 'desi_blend_model_compare_v2.json', 'w') as f:
        json.dump(model_compare, f, indent=2, default=json_safe)
    print(f"Saved: {OUTPUT_DIR / 'desi_blend_model_compare_v2.json'}")

    # =================================================================
    # WRITE REPORT
    # =================================================================
    print("\nWriting report...")

    report = f"""# DESI BLEND-AWARE RV RE-MEASUREMENT REPORT v2

**Date:** 2026-01-16
**Target:** Gaia DR3 {GAIA_SOURCE_ID}
**DESI TARGETID:** {TARGET_ID}

---

## Executive Summary

This report presents a rigorous blend-aware analysis of DESI DR1 spectra using **proper methodology**:
- PHOENIX model templates (M0 for primary, M5 for neighbor)
- χ² minimization in flux space with inverse-variance weights
- Flux-ratio prior informed by the known Gaia neighbor (b ~ 0.13)
- Cross-epoch constant-v2 model test

### Key Facts

| Property | Value |
|----------|-------|
| Neighbor separation | {NEIGHBOR_SEP_ARCSEC:.3f}" |
| Neighbor ΔG | {NEIGHBOR_DELTA_G:.2f} mag |
| Expected flux ratio | {NEIGHBOR_FLUX_FRAC:.3f} |
| Catalog RV swing | {rv_swing_catalog:.0f} km/s |
| Fitted RV swing | {rv_swing_fitted:.0f} km/s |

### Verdict Table

| Check | Result |
|-------|--------|
| Single-template RV consistency | **{verdicts['single_template_consistency']}** |
| Two-template preference | **{verdicts['two_template_preference']}** |
| Flux ratio physical | **{verdicts['flux_ratio_physical']}** |
| Constant-v2 model | **{verdicts['constant_v2_model']}** |
| Blend explains 146 km/s | **{verdicts['blend_explains_swing']}** |

**OVERALL: {overall_verdict}**

{bottom_line}

---

## Part 1: Data Provenance

### DESI Spectra Used

| Epoch | MJD | File | SHA256 |
|-------|-----|------|--------|
"""

    for spec in spectra:
        report += f"| {spec['label']} | {spec['mjd']:.3f} | {Path(spec['file']).name} | {spec['file_hash'][:16]}... |\n"

    report += f"""
### Templates Used

| Template | Source | Parameters |
|----------|--------|------------|
| Primary (M0) | PHOENIX-ACES | Teff=3800K, logg=4.5, [Fe/H]=0.0 |
| Neighbor (M5) | PHOENIX-ACES | Teff=3000K, logg=5.0, [Fe/H]=0.0 |

---

## Part 2: Single-Template RV Fits

Method: χ² minimization with PHOENIX M0 template, inverse-variance weights.

| Epoch | Catalog RV | Fitted RV | σ_v | χ²/dof |
|-------|------------|-----------|-----|--------|
"""

    for sr in single_results:
        report += f"| {sr['label']} | {sr['rv_catalog']:+.1f} | {sr['v']:+.1f} | {sr['v_err']:.1f} | {sr['chi2_reduced']:.2f} |\n"

    report += f"""
**RV swing (fitted): {rv_swing_fitted:.0f} km/s** (catalog: {rv_swing_catalog:.0f} km/s)

---

## Part 3: Two-Template Blend Fits

Method: Fit F = A × [T_M0(v1) + b × T_M5(v2)] with lognormal prior on b centered at {NEIGHBOR_FLUX_FRAC:.3f}.

| Epoch | v1 (km/s) | v2 (km/s) | b (flux) | ΔBIC | Prefer blend? |
|-------|-----------|-----------|----------|------|---------------|
"""

    for br in blend_results:
        prefer = "YES" if br['delta_bic'] < -6 else "no"
        report += f"| {br['label']} | {br['v1']:+.1f} | {br['v2']:+.1f} | {br['b']:.3f} | {br['delta_bic']:+.1f} | {prefer} |\n"

    report += f"""
### Interpretation

- **Flux ratio prior:** b ~ lognormal(0.13, σ_log=0.5), allowing range ~0.05-0.35
- **ΔBIC < -6:** Strong evidence for blend model
- **ΔBIC > -2:** No evidence for blend model

---

## Part 4: Cross-Epoch Constant-v2 Model

This tests whether the neighbor has constant RV across epochs (as expected for a background star).

| Parameter | Value |
|-----------|-------|
| v2 (shared) | {const_v2_result['v2_shared']:+.1f} km/s |
| Flux ratio b | {const_v2_result['b']:.3f} |
| Total χ² | {const_v2_result['chi2_total']:.1f} |
| Δχ² vs single-template | {const_v2_result['delta_chi2_vs_single']:+.1f} |

Per-epoch v1 values:
"""

    for i, v1 in enumerate(const_v2_result['v1_per_epoch']):
        report += f"- {spectra[i]['label']}: v1 = {v1:+.1f} km/s\n"

    report += f"""
---

## Part 5: Physics Check

### Can the known neighbor blend explain 146 km/s?

The maximum RV shift from blending is approximately:

$$\\Delta RV_{{max}} \\approx b \\times |v_1 - v_2|_{{max}}$$

For b = {NEIGHBOR_FLUX_FRAC:.2f} and a reasonable maximum velocity offset of 200 km/s:

$$\\Delta RV_{{max}} \\approx {NEIGHBOR_FLUX_FRAC:.2f} \\times 200 \\approx {NEIGHBOR_FLUX_FRAC * 200:.0f} \\text{{ km/s}}$$

**The observed swing is {rv_swing_catalog:.0f} km/s — {rv_swing_catalog / (NEIGHBOR_FLUX_FRAC * 200):.0f}× larger than the maximum blend effect.**

---

## Final Summary

```
╔══════════════════════════════════════════════════════════════════════╗
║                DESI BLEND-AWARE ANALYSIS v2 RESULTS                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ Single-template consistency:  {verdicts['single_template_consistency']:<12}                           ║
║ Two-template preference:      {verdicts['two_template_preference']:<12}                           ║
║ Flux ratio physical:          {verdicts['flux_ratio_physical']:<12}                           ║
║ Constant-v2 model:            {verdicts['constant_v2_model']:<12}                           ║
║ Blend explains 146 km/s:      {verdicts['blend_explains_swing']:<12}                           ║
╠══════════════════════════════════════════════════════════════════════╣
║ OVERALL VERDICT:              {overall_verdict:<12}                           ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Bottom Line

**{bottom_line}**

---

## Output Files

| File | Description |
|------|-------------|
| `desi_epoch_rv_refit_v2.json` | Per-epoch RV measurements |
| `desi_blend_model_compare_v2.json` | Model comparison results |
| `figures/chi2_vs_v_by_epoch.png` | χ² vs velocity scans |
| `figures/rv_by_method_v2.png` | RV comparison plot |
| `figures/two_template_residuals_v2.png` | Spectral fits and residuals |
| `figures/ccf_peak_shapes_v2.png` | CCF-like visualization |

---

**Report generated:** 2026-01-16
**Analysis by:** Claude Code (v2 methodology)
"""

    with open(OUTPUT_DIR / 'DESI_BLEND_AWARE_REPORT_v2.md', 'w') as f:
        f.write(report)
    print(f"Saved: {OUTPUT_DIR / 'DESI_BLEND_AWARE_REPORT_v2.md'}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
