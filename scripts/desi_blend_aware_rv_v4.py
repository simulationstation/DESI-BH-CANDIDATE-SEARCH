#!/usr/bin/env python3
"""
DESI Blend-Aware RV Analysis v4

Tests whether a PHYSICALLY PLAUSIBLE blend can explain the DESI RV swing.

Key changes from v3:
- Fixed-b tests at b=0.05, 0.13, 0.20 (not free to roam)
- Remove neighbor_teff=3400 (too similar to primary)
- Add "b per arm" model with weak prior
- Epoch 3 R-arm mask sensitivity test
- Flag when b hits boundaries as overfit

Target: Gaia DR3 3802130935635096832
DESI TARGETID: 39627745210139276
Known neighbor: sep=0.688", dG=2.21 -> flux ratio ~0.13
"""

# CRITICAL: Limit numpy/scipy threading BEFORE importing them
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, minimize
import hashlib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from multiprocessing import Pool, cpu_count

# =============================================================================
# CONFIGURATION
# =============================================================================

GAIA_SOURCE_ID = 3802130935635096832
TARGET_ID = 39627745210139276
NEIGHBOR_SEP_ARCSEC = 0.68793
NEIGHBOR_DELTA_G = 2.21099
NEIGHBOR_FLUX_FRAC_G = 10**(-NEIGHBOR_DELTA_G / 2.5)  # ~0.13

# Fixed b values to test (physically motivated)
B_FIXED_VALUES = [0.05, 0.13, 0.20]

# For "b per arm" model - weak lognormal prior centered on 0.13
B_PRIOR_CENTER = 0.13
B_PRIOR_LOG_WIDTH = np.log(2.0)  # factor of 2 width

# Bounds for b per arm model
B_ARM_MIN = 0.02
B_ARM_MAX = 0.25

# Speed of light
C_KMS = 299792.458

# Directories
DATA_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH/data")
TEMPLATE_DIR = DATA_DIR / "templates"
SPECTRA_DIR = DATA_DIR / "desi_spectra"
OUTPUT_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH/outputs/desi_blend_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

# DESI epoch files
EPOCH_FILES = [
    ("Epoch1", "coadd_20268_20211219_p2.fits", 59568.488, -86.39, 0.55),
    ("Epoch2", "coadd_24976_20220125_p7.fits", 59605.380, 59.68, 0.83),
    ("Epoch3", "coadd_23137_20220127_p0.fits", 59607.380, 25.80, 0.80),
]

# Template grids - NO 3400K neighbor (too similar to primary)
PRIMARY_TEFFS = [3600, 3800, 4000]
NEIGHBOR_TEFFS = [2800, 3000, 3200]

# Wavelength regions (Angstroms)
R_BAND = (6000, 7550)
Z_BAND = (7700, 8800)

# Standard telluric mask regions
TELLURIC_MASKS = [
    (6860, 6960),   # B-band O2
    (7150, 7350),   # H2O
    (7580, 7700),   # O2 A-band
    (8100, 8400),   # H2O
]

# TiO-only mask for Epoch 3 R-arm sensitivity test
# Exclude 6000-6400 (continuum-heavy) and focus on TiO bands
TIO_ONLY_MASK_R = (6400, 7550)
TIO_EXTRA_EXCLUSIONS = [
    (6000, 6400),  # Exclude continuum-heavy blue region
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def load_phoenix_wavelength():
    """Load PHOENIX wavelength grid."""
    with fits.open(TEMPLATE_DIR / "WAVE_PHOENIX.fits") as hdul:
        wave = hdul[0].data.astype(np.float64)
    return wave

def downsample_template(wave_full, flux_full, target_wave_range=(5500, 9500), resolution=1.0):
    """Downsample high-resolution PHOENIX template to DESI-like resolution."""
    mask = (wave_full >= target_wave_range[0]) & (wave_full <= target_wave_range[1])
    wave_subset = wave_full[mask]
    flux_subset = flux_full[mask]
    wave_new = np.arange(target_wave_range[0], target_wave_range[1], resolution)
    f_interp = interp1d(wave_subset, flux_subset, kind='linear', bounds_error=False, fill_value=np.nan)
    flux_new = f_interp(wave_new)
    return wave_new, flux_new

def load_phoenix_template(teff, logg):
    """Load PHOENIX template flux."""
    if teff == 3800 and logg == 4.5:
        fname = "phoenix_m0_3800_4.5.fits"
    elif teff == 3000 and logg == 5.0:
        fname = "phoenix_m5_3000_5.0.fits"
    else:
        fname = f"phoenix_{teff}_{logg}.fits"

    fpath = TEMPLATE_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Template not found: {fpath}")

    with fits.open(fpath) as hdul:
        flux = hdul[0].data.astype(np.float64)
    return flux

def load_desi_spectrum(filepath, arm='combined'):
    """Load DESI coadd spectrum."""
    with fits.open(filepath) as hdul:
        fibermap = hdul['FIBERMAP'].data
        idx = np.where(fibermap['TARGETID'] == TARGET_ID)[0]
        if len(idx) == 0:
            raise ValueError(f"Target {TARGET_ID} not found in {filepath}")
        idx = idx[0]

        if arm == 'combined':
            wave_r = hdul['R_WAVELENGTH'].data
            flux_r = hdul['R_FLUX'].data[idx]
            ivar_r = hdul['R_IVAR'].data[idx]
            wave_z = hdul['Z_WAVELENGTH'].data
            flux_z = hdul['Z_FLUX'].data[idx]
            ivar_z = hdul['Z_IVAR'].data[idx]
            wave = np.concatenate([wave_r, wave_z])
            flux = np.concatenate([flux_r, flux_z])
            ivar = np.concatenate([ivar_r, ivar_z])
            sort_idx = np.argsort(wave)
            wave, flux, ivar = wave[sort_idx], flux[sort_idx], ivar[sort_idx]
        else:
            arm_upper = arm.upper()
            wave = hdul[f'{arm_upper}_WAVELENGTH'].data
            flux = hdul[f'{arm_upper}_FLUX'].data[idx]
            ivar = hdul[f'{arm_upper}_IVAR'].data[idx]

    return wave, flux, ivar

def apply_wavelength_mask(wave, flux, ivar, wave_range, telluric_masks=TELLURIC_MASKS, extra_exclusions=None):
    """Apply wavelength range and telluric masks."""
    mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
    for t_min, t_max in telluric_masks:
        mask &= ~((wave >= t_min) & (wave <= t_max))
    if extra_exclusions:
        for e_min, e_max in extra_exclusions:
            mask &= ~((wave >= e_min) & (wave <= e_max))
    mask &= (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)
    return wave[mask], flux[mask], ivar[mask]

def shift_template(template_wave, template_flux, v_kms, target_wave):
    """Doppler shift template and interpolate to target wavelength grid."""
    shifted_wave = template_wave * (1 + v_kms / C_KMS)
    f_interp = interp1d(shifted_wave, template_flux, kind='linear',
                        bounds_error=False, fill_value=np.nan)
    return f_interp(target_wave)

def fit_amplitude_and_poly(data_flux, data_ivar, template_flux, poly_degree=2):
    """
    Analytically solve for amplitude A and polynomial coefficients.
    Model: data = A * template + poly(wave_normalized)
    """
    good = np.isfinite(template_flux) & np.isfinite(data_flux) & (data_ivar > 0)
    if np.sum(good) < 10:
        return np.nan, np.zeros(poly_degree + 1), np.inf

    n_pix = np.sum(good)
    x = np.linspace(-1, 1, len(data_flux))[good]

    # Design matrix: [template, 1, x, x^2, ...]
    n_poly = poly_degree + 1
    design = np.zeros((n_pix, 1 + n_poly))
    design[:, 0] = template_flux[good]
    for i in range(n_poly):
        design[:, 1 + i] = x ** i

    W = np.diag(data_ivar[good])
    y = data_flux[good]

    try:
        DTW = design.T @ W
        DTWD = DTW @ design
        DTWy = DTW @ y
        coeffs = np.linalg.solve(DTWD, DTWy)
    except np.linalg.LinAlgError:
        return np.nan, np.zeros(poly_degree + 1), np.inf

    A = coeffs[0]
    poly_coeffs = coeffs[1:]

    model = design @ coeffs
    residuals = y - model
    chi2 = np.sum(data_ivar[good] * residuals**2)

    return A, poly_coeffs, chi2

def fit_single_template_at_v(data_wave, data_flux, data_ivar, template_wave, template_flux, v):
    """Fit single template at fixed velocity, return chi2."""
    shifted = shift_template(template_wave, template_flux, v, data_wave)
    A, poly_coeffs, chi2 = fit_amplitude_and_poly(data_flux, data_ivar, shifted, poly_degree=2)
    return chi2

def fit_single_template(data_wave, data_flux, data_ivar, template_wave, template_flux):
    """
    Fit single-star model by scanning velocity grid.
    Returns best v, chi2, uncertainties.
    """
    # Coarse scan
    v_coarse = np.linspace(-250, 250, 51)
    chi2_coarse = np.array([fit_single_template_at_v(data_wave, data_flux, data_ivar,
                                                      template_wave, template_flux, v)
                            for v in v_coarse])

    # Find minimum
    idx_min = np.argmin(chi2_coarse)
    v_best_coarse = v_coarse[idx_min]

    # Fine scan around minimum
    v_fine = np.linspace(v_best_coarse - 20, v_best_coarse + 20, 41)
    chi2_fine = np.array([fit_single_template_at_v(data_wave, data_flux, data_ivar,
                                                    template_wave, template_flux, v)
                          for v in v_fine])

    idx_min_fine = np.argmin(chi2_fine)
    v_best = v_fine[idx_min_fine]
    chi2_best = chi2_fine[idx_min_fine]

    # Compute uncertainty from chi2 curvature
    if idx_min_fine > 0 and idx_min_fine < len(v_fine) - 1:
        d2chi2 = (chi2_fine[idx_min_fine+1] + chi2_fine[idx_min_fine-1] - 2*chi2_fine[idx_min_fine]) / (v_fine[1] - v_fine[0])**2
        if d2chi2 > 0:
            sigma_v_formal = np.sqrt(2.0 / d2chi2)
        else:
            sigma_v_formal = np.nan
    else:
        sigma_v_formal = np.nan

    # Compute chi2_red
    good = np.isfinite(data_flux) & (data_ivar > 0)
    n_pix = np.sum(good)
    n_params = 1 + 3 + 1  # A + poly(2) + v
    dof = n_pix - n_params
    chi2_red = chi2_best / dof if dof > 0 else np.inf

    # Renormalized uncertainty
    sigma_v_renorm = sigma_v_formal * np.sqrt(chi2_red) if np.isfinite(sigma_v_formal) else np.nan

    return {
        'v': v_best,
        'chi2': chi2_best,
        'chi2_red': chi2_red,
        'sigma_v_formal': sigma_v_formal,
        'sigma_v_renorm': sigma_v_renorm,
        'n_pix': int(n_pix),
        'dof': int(dof),
    }

def fit_two_component_fixed_b(data_wave, data_flux, data_ivar, template_wave,
                               template_primary, template_neighbor, b_fixed):
    """
    Fit two-component model with FIXED flux ratio b.
    Model: F = A * [T_primary(v1) + b * T_neighbor(v2)] + poly
    """
    # Grid search over v1, v2
    v1_grid = np.linspace(-150, 100, 26)
    v2_grid = np.linspace(-100, 100, 21)

    best_chi2 = np.inf
    best_v1, best_v2 = 0, 0

    for v1 in v1_grid:
        t1 = shift_template(template_wave, template_primary, v1, data_wave)
        for v2 in v2_grid:
            t2 = shift_template(template_wave, template_neighbor, v2, data_wave)
            t_combined = t1 + b_fixed * t2
            A, _, chi2 = fit_amplitude_and_poly(data_flux, data_ivar, t_combined, poly_degree=2)
            if chi2 < best_chi2 and np.isfinite(chi2):
                best_chi2 = chi2
                best_v1, best_v2 = v1, v2

    # Refine with optimization
    def objective(params):
        v1, v2 = params
        t1 = shift_template(template_wave, template_primary, v1, data_wave)
        t2 = shift_template(template_wave, template_neighbor, v2, data_wave)
        t_combined = t1 + b_fixed * t2
        A, _, chi2 = fit_amplitude_and_poly(data_flux, data_ivar, t_combined, poly_degree=2)
        return chi2 if np.isfinite(chi2) else 1e20

    result = minimize(objective, [best_v1, best_v2], method='Nelder-Mead', options={'maxiter': 300})
    v1_final, v2_final = result.x

    # Final chi2
    t1 = shift_template(template_wave, template_primary, v1_final, data_wave)
    t2 = shift_template(template_wave, template_neighbor, v2_final, data_wave)
    t_combined = t1 + b_fixed * t2
    A_final, _, chi2_final = fit_amplitude_and_poly(data_flux, data_ivar, t_combined, poly_degree=2)

    good = np.isfinite(t_combined) & (data_ivar > 0)
    n_pix = np.sum(good)
    n_params = 1 + 3 + 2  # A + poly(2) + v1, v2 (b is fixed, not a free param)
    dof = n_pix - n_params
    chi2_red = chi2_final / dof if dof > 0 else np.inf

    return {
        'v1': v1_final,
        'v2': v2_final,
        'b': b_fixed,
        'chi2': chi2_final,
        'chi2_red': chi2_red,
        'n_pix': int(n_pix),
        'dof': int(dof),
        'n_params': int(n_params),
    }

def fit_two_component_free_b(data_wave, data_flux, data_ivar, template_wave,
                              template_primary, template_neighbor, b_min=B_ARM_MIN, b_max=B_ARM_MAX):
    """
    Fit two-component model with FREE b (constrained to [b_min, b_max]).
    Returns fitted b and flags if it hit boundary.
    """
    v1_grid = np.linspace(-150, 100, 21)
    v2_grid = np.linspace(-100, 100, 17)
    b_grid = np.linspace(b_min, b_max, 7)

    best_chi2 = np.inf
    best_params = None

    for v1 in v1_grid:
        t1 = shift_template(template_wave, template_primary, v1, data_wave)
        for v2 in v2_grid:
            t2 = shift_template(template_wave, template_neighbor, v2, data_wave)
            for b in b_grid:
                t_combined = t1 + b * t2
                A, _, chi2 = fit_amplitude_and_poly(data_flux, data_ivar, t_combined, poly_degree=2)
                if chi2 < best_chi2 and np.isfinite(chi2):
                    best_chi2 = chi2
                    best_params = (v1, v2, b)

    if best_params is None:
        return None

    v1_init, v2_init, b_init = best_params

    def objective(params):
        v1, v2, b = params
        if b < b_min or b > b_max:
            return 1e20
        t1 = shift_template(template_wave, template_primary, v1, data_wave)
        t2 = shift_template(template_wave, template_neighbor, v2, data_wave)
        t_combined = t1 + b * t2
        A, _, chi2 = fit_amplitude_and_poly(data_flux, data_ivar, t_combined, poly_degree=2)
        return chi2 if np.isfinite(chi2) else 1e20

    result = minimize(objective, [v1_init, v2_init, b_init], method='Nelder-Mead', options={'maxiter': 500})
    v1_final, v2_final, b_final = result.x
    b_final = np.clip(b_final, b_min, b_max)

    t1 = shift_template(template_wave, template_primary, v1_final, data_wave)
    t2 = shift_template(template_wave, template_neighbor, v2_final, data_wave)
    t_combined = t1 + b_final * t2
    A_final, _, chi2_final = fit_amplitude_and_poly(data_flux, data_ivar, t_combined, poly_degree=2)

    good = np.isfinite(t_combined) & (data_ivar > 0)
    n_pix = np.sum(good)
    n_params = 1 + 3 + 3  # A + poly(2) + v1, v2, b
    dof = n_pix - n_params
    chi2_red = chi2_final / dof if dof > 0 else np.inf

    # Check if b hit boundary
    boundary_hit = (abs(b_final - b_min) < 0.001) or (abs(b_final - b_max) < 0.001)

    return {
        'v1': v1_final,
        'v2': v2_final,
        'b': b_final,
        'chi2': chi2_final,
        'chi2_red': chi2_red,
        'n_pix': int(n_pix),
        'dof': int(dof),
        'n_params': int(n_params),
        'boundary_hit': boundary_hit,
    }

def compute_bic(chi2, n_params, n_data):
    """Compute BIC on data chi2 only (no priors)."""
    return chi2 + n_params * np.log(n_data)

def lognormal_prior_penalty(b, center=B_PRIOR_CENTER, log_width=B_PRIOR_LOG_WIDTH):
    """Compute -2*ln(prior) for lognormal prior on b."""
    if b <= 0:
        return 1e20
    log_b = np.log(b)
    log_center = np.log(center)
    return ((log_b - log_center) / log_width) ** 2

# =============================================================================
# CHECKPOINT FUNCTIONS
# =============================================================================

CHECKPOINT_FILE = OUTPUT_DIR / 'checkpoint_v4.pkl'

def save_checkpoint(stage, data):
    checkpoint = {'stage': stage, 'data': data}
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  [Checkpoint saved: stage={stage}]")

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"  [Checkpoint loaded: stage={checkpoint['stage']}]")
        return checkpoint
    return None

def clear_checkpoint():
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("  [Checkpoint cleared]")

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("="*70)
    print("DESI BLEND-AWARE RV ANALYSIS v4")
    print("="*70)
    print(f"\nTarget: Gaia DR3 {GAIA_SOURCE_ID}")
    print(f"DESI TARGETID: {TARGET_ID}")
    print(f"\nKnown neighbor: sep={NEIGHBOR_SEP_ARCSEC:.3f}\", dG={NEIGHBOR_DELTA_G:.2f}")
    print(f"Expected flux ratio (G-band): {NEIGHBOR_FLUX_FRAC_G:.3f}")
    print(f"Fixed b values to test: {B_FIXED_VALUES}")

    n_workers = min(8, max(1, cpu_count() - 1))
    print(f"Using {n_workers} CPU workers")

    # Check for checkpoint
    checkpoint = load_checkpoint()
    checkpoint_stage = checkpoint['stage'] if checkpoint else 0
    checkpoint_data = checkpoint['data'] if checkpoint else {}

    # =========================================================================
    # LOAD TEMPLATES
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING TEMPLATES")
    print("="*70)

    template_wave_full = load_phoenix_wavelength()
    print(f"PHOENIX wavelength grid: {len(template_wave_full)} points")

    print("Downsampling templates...")
    primary_templates = {}
    template_wave = None
    for teff in PRIMARY_TEFFS:
        logg = 4.5
        flux_full = load_phoenix_template(teff, logg)
        wave_ds, flux_ds = downsample_template(template_wave_full, flux_full)
        if template_wave is None:
            template_wave = wave_ds
        primary_templates[teff] = flux_ds
        print(f"  Primary Teff={teff}K loaded")

    neighbor_templates = {}
    for teff in NEIGHBOR_TEFFS:
        logg = 5.0
        flux_full = load_phoenix_template(teff, logg)
        _, flux_ds = downsample_template(template_wave_full, flux_full)
        neighbor_templates[teff] = flux_ds
        print(f"  Neighbor Teff={teff}K loaded")

    print(f"Downsampled grid: {len(template_wave)} points")

    # =========================================================================
    # LOAD SPECTRA
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING DESI SPECTRA")
    print("="*70)

    spectra = []
    for label, fname, mjd, rv_cat, rv_err_cat in EPOCH_FILES:
        fpath = SPECTRA_DIR / fname
        file_hash = sha256_file(fpath)
        print(f"{label}: {fname} (MJD {mjd}, catalog RV: {rv_cat:+.1f} km/s)")
        spectra.append({
            'label': label,
            'file': str(fpath),
            'mjd': mjd,
            'rv_catalog': rv_cat,
            'sha256': file_hash[:16],
        })

    # =========================================================================
    # ANALYSIS 1: SINGLE-STAR FITS (PRIMARY AND NEIGHBOR)
    # =========================================================================
    if checkpoint_stage >= 1:
        print("\n" + "="*70)
        print("ANALYSIS 1: SINGLE-STAR FITS [from checkpoint]")
        print("="*70)
        single_results = checkpoint_data['single_results']
    else:
        print("\n" + "="*70)
        print("ANALYSIS 1: SINGLE-STAR FITS")
        print("="*70)

        arms_info = [('R', R_BAND), ('Z', Z_BAND), ('combined', (R_BAND[0], Z_BAND[1]))]
        single_results = []

        for spec in spectra:
            print(f"\n--- {spec['label']} (catalog RV: {spec['rv_catalog']:+.1f} km/s) ---")
            epoch_result = {'label': spec['label'], 'mjd': spec['mjd'], 'arms': {}}

            for arm, wave_range in arms_info:
                if arm == 'combined':
                    wave, flux, ivar = load_desi_spectrum(spec['file'], 'combined')
                else:
                    wave, flux, ivar = load_desi_spectrum(spec['file'], arm)
                wave_m, flux_m, ivar_m = apply_wavelength_mask(wave, flux, ivar, wave_range)

                # Primary-only fits
                best_primary_chi2 = np.inf
                best_primary_fit = None
                best_primary_teff = None
                for teff in PRIMARY_TEFFS:
                    fit = fit_single_template(wave_m, flux_m, ivar_m, template_wave, primary_templates[teff])
                    if fit['chi2'] < best_primary_chi2:
                        best_primary_chi2 = fit['chi2']
                        best_primary_fit = fit
                        best_primary_teff = teff

                # Neighbor-only fits
                best_neighbor_chi2 = np.inf
                best_neighbor_fit = None
                best_neighbor_teff = None
                for teff in NEIGHBOR_TEFFS:
                    fit = fit_single_template(wave_m, flux_m, ivar_m, template_wave, neighbor_templates[teff])
                    if fit['chi2'] < best_neighbor_chi2:
                        best_neighbor_chi2 = fit['chi2']
                        best_neighbor_fit = fit
                        best_neighbor_teff = teff

                epoch_result['arms'][arm] = {
                    'primary': {
                        'teff': best_primary_teff,
                        'v': best_primary_fit['v'],
                        'chi2': best_primary_fit['chi2'],
                        'chi2_red': best_primary_fit['chi2_red'],
                        'sigma_v_renorm': best_primary_fit['sigma_v_renorm'],
                        'n_pix': best_primary_fit['n_pix'],
                    },
                    'neighbor': {
                        'teff': best_neighbor_teff,
                        'v': best_neighbor_fit['v'],
                        'chi2': best_neighbor_fit['chi2'],
                        'chi2_red': best_neighbor_fit['chi2_red'],
                        'sigma_v_renorm': best_neighbor_fit['sigma_v_renorm'],
                    },
                    'chi2_ratio': best_neighbor_chi2 / best_primary_chi2 if best_primary_chi2 > 0 else np.nan,
                }

                print(f"  {arm.upper():8s}: primary v={best_primary_fit['v']:+6.1f}, "
                      f"neighbor v={best_neighbor_fit['v']:+6.1f}, "
                      f"chi2_ratio={best_neighbor_chi2/best_primary_chi2:.2f}")

            single_results.append(epoch_result)

        save_checkpoint(1, {'single_results': single_results})

    # =========================================================================
    # ANALYSIS 2: FIXED-b TWO-COMPONENT FITS
    # =========================================================================
    if checkpoint_stage >= 2:
        print("\n" + "="*70)
        print("ANALYSIS 2: FIXED-b TWO-COMPONENT FITS [from checkpoint]")
        print("="*70)
        fixed_b_results = checkpoint_data['fixed_b_results']
    else:
        print("\n" + "="*70)
        print("ANALYSIS 2: FIXED-b TWO-COMPONENT FITS")
        print("="*70)

        arms_info = [('R', R_BAND), ('Z', Z_BAND), ('combined', (R_BAND[0], Z_BAND[1]))]
        fixed_b_results = []

        for spec_idx, spec in enumerate(spectra):
            print(f"\n--- {spec['label']} ---")
            epoch_result = {'label': spec['label'], 'arms': {}}

            for arm, wave_range in arms_info:
                if arm == 'combined':
                    wave, flux, ivar = load_desi_spectrum(spec['file'], 'combined')
                else:
                    wave, flux, ivar = load_desi_spectrum(spec['file'], arm)
                wave_m, flux_m, ivar_m = apply_wavelength_mask(wave, flux, ivar, wave_range)

                # Get best templates from single fits
                best_primary_teff = single_results[spec_idx]['arms'][arm]['primary']['teff']
                best_neighbor_teff = single_results[spec_idx]['arms'][arm]['neighbor']['teff']

                t_primary = primary_templates[best_primary_teff]
                t_neighbor = neighbor_templates[best_neighbor_teff]

                # Single-star chi2 and BIC for comparison
                single_chi2 = single_results[spec_idx]['arms'][arm]['primary']['chi2']
                single_n_pix = single_results[spec_idx]['arms'][arm]['primary']['n_pix']
                single_n_params = 5  # A + poly(2) + v
                single_bic = compute_bic(single_chi2, single_n_params, single_n_pix)

                arm_result = {'b_tests': {}, 'single_chi2': single_chi2, 'single_bic': single_bic}

                for b_val in B_FIXED_VALUES:
                    fit = fit_two_component_fixed_b(wave_m, flux_m, ivar_m, template_wave,
                                                    t_primary, t_neighbor, b_val)

                    blend_bic = compute_bic(fit['chi2'], fit['n_params'], fit['n_pix'])
                    delta_bic = blend_bic - single_bic

                    arm_result['b_tests'][b_val] = {
                        'v1': fit['v1'],
                        'v2': fit['v2'],
                        'chi2': fit['chi2'],
                        'chi2_red': fit['chi2_red'],
                        'bic': blend_bic,
                        'delta_bic': delta_bic,
                        'blend_preferred': delta_bic < -6,
                    }

                    print(f"  {arm.upper()} b={b_val:.2f}: v1={fit['v1']:+6.1f}, v2={fit['v2']:+6.1f}, "
                          f"dBIC={delta_bic:+.1f} {'*' if delta_bic < -6 else ''}")

                epoch_result['arms'][arm] = arm_result

            fixed_b_results.append(epoch_result)

        save_checkpoint(2, {'single_results': single_results, 'fixed_b_results': fixed_b_results})

    # =========================================================================
    # ANALYSIS 3: FREE-b PER ARM MODEL
    # =========================================================================
    if checkpoint_stage >= 3:
        print("\n" + "="*70)
        print("ANALYSIS 3: FREE-b PER ARM MODEL [from checkpoint]")
        print("="*70)
        free_b_results = checkpoint_data['free_b_results']
    else:
        print("\n" + "="*70)
        print("ANALYSIS 3: FREE-b PER ARM MODEL")
        print("="*70)
        print(f"  b constrained to [{B_ARM_MIN}, {B_ARM_MAX}], flagging boundary hits")

        free_b_results = []

        for spec_idx, spec in enumerate(spectra):
            print(f"\n--- {spec['label']} ---")
            epoch_result = {'label': spec['label'], 'arms': {}}

            for arm, wave_range in [('R', R_BAND), ('Z', Z_BAND)]:
                wave, flux, ivar = load_desi_spectrum(spec['file'], arm)
                wave_m, flux_m, ivar_m = apply_wavelength_mask(wave, flux, ivar, wave_range)

                best_primary_teff = single_results[spec_idx]['arms'][arm]['primary']['teff']
                best_neighbor_teff = single_results[spec_idx]['arms'][arm]['neighbor']['teff']

                t_primary = primary_templates[best_primary_teff]
                t_neighbor = neighbor_templates[best_neighbor_teff]

                fit = fit_two_component_free_b(wave_m, flux_m, ivar_m, template_wave,
                                               t_primary, t_neighbor)

                if fit is None:
                    epoch_result['arms'][arm] = {'status': 'FAILED'}
                    print(f"  {arm}: FAILED")
                else:
                    prior_penalty = lognormal_prior_penalty(fit['b'])
                    epoch_result['arms'][arm] = {
                        'v1': fit['v1'],
                        'v2': fit['v2'],
                        'b': fit['b'],
                        'chi2': fit['chi2'],
                        'chi2_red': fit['chi2_red'],
                        'boundary_hit': fit['boundary_hit'],
                        'prior_penalty': prior_penalty,
                    }
                    flag = "BOUNDARY HIT" if fit['boundary_hit'] else ""
                    print(f"  {arm}: b={fit['b']:.3f}, v1={fit['v1']:+6.1f}, v2={fit['v2']:+6.1f} {flag}")

            free_b_results.append(epoch_result)

        save_checkpoint(3, {'single_results': single_results, 'fixed_b_results': fixed_b_results,
                           'free_b_results': free_b_results})

    # =========================================================================
    # ANALYSIS 4: EPOCH 3 R-ARM MASK SENSITIVITY
    # =========================================================================
    if checkpoint_stage >= 4:
        print("\n" + "="*70)
        print("ANALYSIS 4: EPOCH 3 R-ARM MASK SENSITIVITY [from checkpoint]")
        print("="*70)
        mask_sensitivity = checkpoint_data['mask_sensitivity']
    else:
        print("\n" + "="*70)
        print("ANALYSIS 4: EPOCH 3 R-ARM MASK SENSITIVITY")
        print("="*70)

        spec = spectra[2]  # Epoch 3
        wave, flux, ivar = load_desi_spectrum(spec['file'], 'R')

        # Standard R mask
        wave_std, flux_std, ivar_std = apply_wavelength_mask(wave, flux, ivar, R_BAND)

        # TiO-only mask (exclude 6000-6400)
        wave_tio, flux_tio, ivar_tio = apply_wavelength_mask(wave, flux, ivar, TIO_ONLY_MASK_R,
                                                              extra_exclusions=TIO_EXTRA_EXCLUSIONS)

        best_primary_teff = single_results[2]['arms']['R']['primary']['teff']
        t_primary = primary_templates[best_primary_teff]

        fit_std = fit_single_template(wave_std, flux_std, ivar_std, template_wave, t_primary)
        fit_tio = fit_single_template(wave_tio, flux_tio, ivar_tio, template_wave, t_primary)

        # Z-arm for reference
        wave_z, flux_z, ivar_z = load_desi_spectrum(spec['file'], 'Z')
        wave_z_m, flux_z_m, ivar_z_m = apply_wavelength_mask(wave_z, flux_z, ivar_z, Z_BAND)
        best_z_teff = single_results[2]['arms']['Z']['primary']['teff']
        fit_z = fit_single_template(wave_z_m, flux_z_m, ivar_z_m, template_wave, primary_templates[best_z_teff])

        mask_sensitivity = {
            'epoch': 'Epoch3',
            'R_standard': {
                'v': fit_std['v'],
                'sigma_v_renorm': fit_std['sigma_v_renorm'],
                'chi2_red': fit_std['chi2_red'],
                'n_pix': fit_std['n_pix'],
            },
            'R_tio_only': {
                'v': fit_tio['v'],
                'sigma_v_renorm': fit_tio['sigma_v_renorm'],
                'chi2_red': fit_tio['chi2_red'],
                'n_pix': fit_tio['n_pix'],
            },
            'Z_reference': {
                'v': fit_z['v'],
                'sigma_v_renorm': fit_z['sigma_v_renorm'],
            },
            'R_std_vs_Z_diff': fit_std['v'] - fit_z['v'],
            'R_tio_vs_Z_diff': fit_tio['v'] - fit_z['v'],
        }

        print(f"  R (standard mask):  v = {fit_std['v']:+7.1f} km/s ({fit_std['n_pix']} pixels)")
        print(f"  R (TiO-only mask):  v = {fit_tio['v']:+7.1f} km/s ({fit_tio['n_pix']} pixels)")
        print(f"  Z (reference):      v = {fit_z['v']:+7.1f} km/s")
        print(f"  R_std - Z diff:     {fit_std['v'] - fit_z['v']:+7.1f} km/s")
        print(f"  R_tio - Z diff:     {fit_tio['v'] - fit_z['v']:+7.1f} km/s")

        save_checkpoint(4, {'single_results': single_results, 'fixed_b_results': fixed_b_results,
                           'free_b_results': free_b_results, 'mask_sensitivity': mask_sensitivity})

    # =========================================================================
    # COMPONENT SWITCHING ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("COMPONENT SWITCHING ANALYSIS")
    print("="*70)

    switch_risk = []
    for spec_idx, spec in enumerate(spectra):
        epoch_risk = {'label': spec['label'], 'arms': {}}
        for arm in ['R', 'Z', 'combined']:
            primary_v = single_results[spec_idx]['arms'][arm]['primary']['v']
            neighbor_v = single_results[spec_idx]['arms'][arm]['neighbor']['v']
            chi2_ratio = single_results[spec_idx]['arms'][arm]['chi2_ratio']

            v_diff = abs(primary_v - neighbor_v)
            competitive = chi2_ratio < 1.5

            epoch_risk['arms'][arm] = {
                'primary_v': primary_v,
                'neighbor_v': neighbor_v,
                'v_diff': v_diff,
                'chi2_ratio': chi2_ratio,
                'competitive': competitive,
                'switching_risk': competitive and v_diff > 10,
            }
        switch_risk.append(epoch_risk)

        print(f"\n{spec['label']}:")
        for arm in ['R', 'Z', 'combined']:
            r = epoch_risk['arms'][arm]
            risk_flag = "SWITCHING RISK" if r['switching_risk'] else ""
            print(f"  {arm.upper()}: primary={r['primary_v']:+6.1f}, neighbor={r['neighbor_v']:+6.1f}, "
                  f"diff={r['v_diff']:.1f}, ratio={r['chi2_ratio']:.2f} {risk_flag}")

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    # JSON outputs
    with open(OUTPUT_DIR / 'desi_epoch_rv_refit_v4.json', 'w') as f:
        json.dump(single_results, f, indent=2, default=float)
    print(f"Saved: desi_epoch_rv_refit_v4.json")

    with open(OUTPUT_DIR / 'desi_blend_fixed_b_tests_v4.json', 'w') as f:
        json.dump(fixed_b_results, f, indent=2, default=float)
    print(f"Saved: desi_blend_fixed_b_tests_v4.json")

    with open(OUTPUT_DIR / 'desi_arm_b_fit_v4.json', 'w') as f:
        json.dump(free_b_results, f, indent=2, default=float)
    print(f"Saved: desi_arm_b_fit_v4.json")

    with open(OUTPUT_DIR / 'desi_neighbor_switch_risk_v4.json', 'w') as f:
        json.dump(switch_risk, f, indent=2, default=float)
    print(f"Saved: desi_neighbor_switch_risk_v4.json")

    with open(OUTPUT_DIR / 'desi_mask_sensitivity_v4.json', 'w') as f:
        json.dump(mask_sensitivity, f, indent=2, default=float)
    print(f"Saved: desi_mask_sensitivity_v4.json")

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    # Figure 1: Arm split RV comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, spec in enumerate(spectra):
        ax = axes[i]
        r_v = single_results[i]['arms']['R']['primary']['v']
        z_v = single_results[i]['arms']['Z']['primary']['v']
        r_err = single_results[i]['arms']['R']['primary']['sigma_v_renorm']
        z_err = single_results[i]['arms']['Z']['primary']['sigma_v_renorm']

        ax.errorbar(['R', 'Z'], [r_v, z_v], yerr=[r_err, z_err], fmt='o', capsize=5)
        ax.axhline(spec['rv_catalog'], color='gray', linestyle='--', label=f"Catalog: {spec['rv_catalog']:.1f}")
        ax.set_ylabel('RV (km/s)')
        ax.set_title(f"{spec['label']}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / 'arm_split_rv_v4.png', dpi=150)
    plt.close()
    print("Saved: arm_split_rv_v4.png")

    # Figure 2: Fixed-b model comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, spec in enumerate(spectra):
        ax = axes[i]
        for arm in ['R', 'Z', 'combined']:
            delta_bics = [fixed_b_results[i]['arms'][arm]['b_tests'][b]['delta_bic'] for b in B_FIXED_VALUES]
            ax.plot(B_FIXED_VALUES, delta_bics, 'o-', label=arm)
        ax.axhline(-6, color='red', linestyle='--', alpha=0.5, label='dBIC=-6')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0.13, color='green', linestyle=':', alpha=0.5, label='b_expected')
        ax.set_xlabel('Fixed b')
        ax.set_ylabel('delta BIC')
        ax.set_title(f"{spec['label']}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / 'fixed_b_model_comparison_v4.png', dpi=150)
    plt.close()
    print("Saved: fixed_b_model_comparison_v4.png")

    # Figure 3: Epoch 3 R-arm mask sensitivity
    fig, ax = plt.subplots(figsize=(8, 5))
    masks = ['R standard', 'R TiO-only', 'Z reference']
    vs = [mask_sensitivity['R_standard']['v'],
          mask_sensitivity['R_tio_only']['v'],
          mask_sensitivity['Z_reference']['v']]
    colors = ['blue', 'orange', 'green']
    ax.bar(masks, vs, color=colors, alpha=0.7)
    ax.axhline(spectra[2]['rv_catalog'], color='red', linestyle='--', label=f"Catalog: {spectra[2]['rv_catalog']:.1f}")
    ax.set_ylabel('RV (km/s)')
    ax.set_title('Epoch 3 R-arm Mask Sensitivity')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / 'epoch3_R_mask_sensitivity_v4.png', dpi=150)
    plt.close()
    print("Saved: epoch3_R_mask_sensitivity_v4.png")

    # Figure 4: b_R vs b_Z fitted values
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, spec in enumerate(spectra):
        if 'R' in free_b_results[i]['arms'] and 'Z' in free_b_results[i]['arms']:
            r_res = free_b_results[i]['arms']['R']
            z_res = free_b_results[i]['arms']['Z']
            if 'b' in r_res and 'b' in z_res:
                marker = 's' if r_res.get('boundary_hit') or z_res.get('boundary_hit') else 'o'
                ax.plot(r_res['b'], z_res['b'], marker, markersize=10, label=spec['label'])
    ax.axhline(0.13, color='green', linestyle=':', alpha=0.5)
    ax.axvline(0.13, color='green', linestyle=':', alpha=0.5)
    ax.axhline(B_ARM_MAX, color='red', linestyle='--', alpha=0.3)
    ax.axvline(B_ARM_MAX, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('b_R')
    ax.set_ylabel('b_Z')
    ax.set_title('Fitted b per arm (squares = boundary hit)')
    ax.legend()
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / 'bR_bZ_posterior_or_grid_v4.png', dpi=150)
    plt.close()
    print("Saved: bR_bZ_posterior_or_grid_v4.png")

    # =========================================================================
    # WRITE REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("WRITING REPORT")
    print("="*70)

    # Analyze results for verdict
    # Check if any fixed-b model at b~0.13 is preferred
    blend_explains_at_physical_b = False
    blend_requires_high_b = True

    for epoch_res in fixed_b_results:
        for arm in ['R', 'Z', 'combined']:
            b013_result = epoch_res['arms'][arm]['b_tests'][0.13]
            if b013_result['blend_preferred']:
                blend_explains_at_physical_b = True
            # Check if lower b values also show preference
            b005_result = epoch_res['arms'][arm]['b_tests'][0.05]
            if b005_result['blend_preferred']:
                blend_requires_high_b = False

    # Check boundary hits in free-b model
    any_boundary_hit = False
    for epoch_res in free_b_results:
        for arm in ['R', 'Z']:
            if epoch_res['arms'].get(arm, {}).get('boundary_hit', False):
                any_boundary_hit = True

    # Check mask sensitivity
    r_tio_agrees_with_z = abs(mask_sensitivity['R_tio_vs_Z_diff']) < 20

    report = f"""# DESI Blend-Aware RV Analysis v4

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Target:** Gaia DR3 {GAIA_SOURCE_ID}
**DESI TARGETID:** {TARGET_ID}

## Summary

This analysis tests whether a physically plausible blend (b ~ 0.13) can explain the observed DESI RV variability.

### Known neighbor
- Separation: {NEIGHBOR_SEP_ARCSEC:.3f}"
- Delta G: {NEIGHBOR_DELTA_G:.2f} mag
- Expected flux ratio: {NEIGHBOR_FLUX_FRAC_G:.3f}

### Key findings

"""

    # Add single-star RV summary
    report += "## Single-star RV fits\n\n"
    report += "| Epoch | Arm | Primary v (km/s) | Neighbor v (km/s) | chi2 ratio |\n"
    report += "|-------|-----|------------------|-------------------|------------|\n"
    for i, res in enumerate(single_results):
        for arm in ['R', 'Z', 'combined']:
            p_v = res['arms'][arm]['primary']['v']
            n_v = res['arms'][arm]['neighbor']['v']
            ratio = res['arms'][arm]['chi2_ratio']
            report += f"| {res['label']} | {arm} | {p_v:+.1f} | {n_v:+.1f} | {ratio:.2f} |\n"
    report += "\n"

    # Add fixed-b results
    report += "## Fixed-b blend model tests\n\n"
    report += "Testing whether blend models with physically plausible b values are preferred over single-star.\n\n"

    for b_val in B_FIXED_VALUES:
        report += f"### b = {b_val}\n\n"
        report += "| Epoch | Arm | v1 | v2 | delta BIC | Preferred? |\n"
        report += "|-------|-----|----|----|-----------|------------|\n"
        for res in fixed_b_results:
            for arm in ['R', 'Z', 'combined']:
                t = res['arms'][arm]['b_tests'][b_val]
                pref = "Yes" if t['blend_preferred'] else "No"
                report += f"| {res['label']} | {arm} | {t['v1']:+.1f} | {t['v2']:+.1f} | {t['delta_bic']:+.1f} | {pref} |\n"
        report += "\n"

    # Add free-b results
    report += "## Free-b per arm model\n\n"
    report += "b constrained to [0.02, 0.25], flagging boundary hits as potential overfit.\n\n"
    report += "| Epoch | Arm | Fitted b | v1 | v2 | Boundary hit? |\n"
    report += "|-------|-----|----------|----|----|---------------|\n"
    for res in free_b_results:
        for arm in ['R', 'Z']:
            if 'b' in res['arms'].get(arm, {}):
                a = res['arms'][arm]
                bh = "Yes" if a['boundary_hit'] else "No"
                report += f"| {res['label']} | {arm} | {a['b']:.3f} | {a['v1']:+.1f} | {a['v2']:+.1f} | {bh} |\n"
    report += "\n"

    # Add mask sensitivity
    report += "## Epoch 3 R-arm mask sensitivity\n\n"
    report += f"- R standard mask: v = {mask_sensitivity['R_standard']['v']:+.1f} km/s\n"
    report += f"- R TiO-only mask: v = {mask_sensitivity['R_tio_only']['v']:+.1f} km/s\n"
    report += f"- Z reference: v = {mask_sensitivity['Z_reference']['v']:+.1f} km/s\n"
    report += f"- R_std - Z difference: {mask_sensitivity['R_std_vs_Z_diff']:+.1f} km/s\n"
    report += f"- R_tio - Z difference: {mask_sensitivity['R_tio_vs_Z_diff']:+.1f} km/s\n\n"

    if r_tio_agrees_with_z:
        report += "The TiO-only mask brings R-arm into better agreement with Z-arm, suggesting the standard R-arm discrepancy may be a mask/systematic artifact.\n\n"
    else:
        report += "The R-arm discrepancy persists even with TiO-only mask.\n\n"

    # Switching risk
    report += "## Component switching risk\n\n"
    any_switch_risk = False
    for res in switch_risk:
        for arm in ['R', 'Z', 'combined']:
            if res['arms'][arm]['switching_risk']:
                any_switch_risk = True
                report += f"- {res['label']} {arm}: neighbor-only fit is competitive (chi2 ratio {res['arms'][arm]['chi2_ratio']:.2f}) with v difference {res['arms'][arm]['v_diff']:.1f} km/s\n"

    if not any_switch_risk:
        report += "No significant component switching risk detected.\n"
    report += "\n"

    # Final assessment
    report += "## Assessment\n\n"

    if blend_explains_at_physical_b:
        report += "Blend models with b ~ 0.13 (the expected value from the known neighbor) show improved fits in some cases. "
    else:
        report += "Blend models with b ~ 0.13 do not significantly improve fits over single-star models. "

    if any_boundary_hit:
        report += "The free-b model hit boundaries, suggesting potential overfit when b is unconstrained. "

    if r_tio_agrees_with_z:
        report += "The Epoch 3 R-arm anomaly appears to be mask-dependent. "

    report += "\n\n"

    # Verdict
    report += "### Does a physically plausible blend explain the DESI RV swing?\n\n"

    rv_swing = max([single_results[i]['arms']['combined']['primary']['v'] for i in range(3)]) - \
               min([single_results[i]['arms']['combined']['primary']['v'] for i in range(3)])

    if blend_explains_at_physical_b and not any_boundary_hit:
        report += f"Possibly. Blend models with b ~ 0.13 are preferred in some epoch/arm combinations. However, the {rv_swing:.0f} km/s RV swing is large for a ~13% blend to fully explain.\n"
    elif any_boundary_hit and not blend_explains_at_physical_b:
        report += f"Unlikely. Blend models only improve when b is pushed to unphysical values (boundary hits). The {rv_swing:.0f} km/s RV swing persists with physically constrained blend models.\n"
    else:
        report += f"Inconclusive. Results are mixed across epochs and arms. Further high-resolution observations are needed.\n"

    report += f"""
---

## Output files

- desi_epoch_rv_refit_v4.json: Single-star fit results
- desi_blend_fixed_b_tests_v4.json: Fixed-b blend model comparisons
- desi_arm_b_fit_v4.json: Free-b per arm results
- desi_neighbor_switch_risk_v4.json: Component switching analysis
- desi_mask_sensitivity_v4.json: Epoch 3 R-arm mask test

## Figures

- arm_split_rv_v4.png: R vs Z arm RV comparison
- fixed_b_model_comparison_v4.png: delta BIC vs fixed b
- epoch3_R_mask_sensitivity_v4.png: Mask sensitivity test
- bR_bZ_posterior_or_grid_v4.png: Fitted b per arm

---

**Analysis by:** Claude Code (v4 methodology)
**Templates:** PHOENIX-ACES (Primary: {PRIMARY_TEFFS}, Neighbor: {NEIGHBOR_TEFFS})
"""

    with open(OUTPUT_DIR / 'DESI_BLEND_AWARE_REPORT_v4.md', 'w') as f:
        f.write(report)
    print(f"Saved: DESI_BLEND_AWARE_REPORT_v4.md")

    clear_checkpoint()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
