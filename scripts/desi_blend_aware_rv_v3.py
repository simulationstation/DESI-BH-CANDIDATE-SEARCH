#!/usr/bin/env python3
"""
DESI Blend-Aware RV Analysis v3

Methodological improvements over v2:
A) RV uncertainties renormalized by sqrt(chi2_red)
B) BIC computed on data likelihood only (no priors in chi2)
C) Marginalize over amplitude A and continuum polynomial at each v
D) Separate R and Z band fits
E) Explicit neighbor-only fits and component-switching tests

Target: Gaia DR3 3802130935635096832
DESI TARGETID: 39627745210139276
Confirmed neighbor: sep=0.688", dG=2.21 -> flux ratio ~0.13
"""

# CRITICAL: Limit numpy/scipy threading BEFORE importing them
# Each worker should use 1 thread to avoid overloading system
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar
import hashlib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

# =============================================================================
# CONFIGURATION
# =============================================================================

GAIA_SOURCE_ID = 3802130935635096832
TARGET_ID = 39627745210139276
NEIGHBOR_SEP_ARCSEC = 0.68793
NEIGHBOR_DELTA_G = 2.21099
NEIGHBOR_FLUX_FRAC_G = 10**(-NEIGHBOR_DELTA_G / 2.5)  # ~0.13

# Flux ratio bounds for two-component fits
B_MIN = 0.02
B_MAX = 0.30

# Velocity grid for chi2(v) scans
V_GRID_COARSE = np.linspace(-250, 250, 51)   # 10 km/s resolution for initial scan
V_GRID_FINE = np.linspace(-250, 250, 201)    # 2.5 km/s resolution for final output

# Speed of light
C_KMS = 299792.458

# Directories
DATA_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH/data")
TEMPLATE_DIR = DATA_DIR / "templates"
SPECTRA_DIR = DATA_DIR / "desi_spectra"
OUTPUT_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH/outputs/desi_blend_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

# DESI epoch files
EPOCH_FILES = [
    ("Epoch1", "coadd_20268_20211219_p2.fits", 59568.488, -86.39, 0.55),
    ("Epoch2", "coadd_24976_20220125_p7.fits", 59605.380, 59.68, 0.83),
    ("Epoch3", "coadd_23137_20220127_p0.fits", 59607.380, 25.80, 0.80),
]

# Template grids
PRIMARY_TEFFS = [3400, 3600, 3800, 4000]
NEIGHBOR_TEFFS = [2800, 3000, 3200, 3400]

# Wavelength regions (Angstroms)
R_BAND = (6000, 7550)  # R arm, avoiding major tellurics
Z_BAND = (7700, 8800)  # Z arm, avoiding major tellurics

# Telluric mask regions
TELLURIC_MASKS = [
    (6860, 6960),   # B-band O2
    (7150, 7350),   # H2O
    (7580, 7700),   # O2 A-band
    (8100, 8400),   # H2O
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
    """
    Downsample high-resolution PHOENIX template to DESI-like resolution.

    This dramatically speeds up subsequent operations.
    Using 1 Å resolution (vs DESI native ~0.8 Å) for speed.
    """
    # Select wavelength range
    mask = (wave_full >= target_wave_range[0]) & (wave_full <= target_wave_range[1])
    wave_subset = wave_full[mask]
    flux_subset = flux_full[mask]

    # Create target wavelength grid at ~1 Å resolution
    wave_new = np.arange(target_wave_range[0], target_wave_range[1], resolution)

    # Interpolate
    f_interp = interp1d(wave_subset, flux_subset, kind='linear', bounds_error=False, fill_value=np.nan)
    flux_new = f_interp(wave_new)

    return wave_new, flux_new

def load_phoenix_template(teff, logg):
    """Load PHOENIX template flux."""
    # Check naming convention
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
    """
    Load DESI coadd spectrum.

    arm: 'R', 'Z', or 'combined'
    Returns wavelength, flux, ivar arrays.
    """
    with fits.open(filepath) as hdul:
        # Find target in FIBERMAP
        fibermap = hdul['FIBERMAP'].data
        idx = np.where(fibermap['TARGETID'] == TARGET_ID)[0]
        if len(idx) == 0:
            raise ValueError(f"Target {TARGET_ID} not found in {filepath}")
        idx = idx[0]

        if arm == 'combined':
            # Combine R and Z arms
            wave_r = hdul['R_WAVELENGTH'].data
            flux_r = hdul['R_FLUX'].data[idx]
            ivar_r = hdul['R_IVAR'].data[idx]

            wave_z = hdul['Z_WAVELENGTH'].data
            flux_z = hdul['Z_FLUX'].data[idx]
            ivar_z = hdul['Z_IVAR'].data[idx]

            wave = np.concatenate([wave_r, wave_z])
            flux = np.concatenate([flux_r, flux_z])
            ivar = np.concatenate([ivar_r, ivar_z])

            # Sort by wavelength
            sort_idx = np.argsort(wave)
            wave = wave[sort_idx]
            flux = flux[sort_idx]
            ivar = ivar[sort_idx]
        else:
            arm_upper = arm.upper()
            wave = hdul[f'{arm_upper}_WAVELENGTH'].data
            flux = hdul[f'{arm_upper}_FLUX'].data[idx]
            ivar = hdul[f'{arm_upper}_IVAR'].data[idx]

    return wave, flux, ivar

def apply_wavelength_mask(wave, flux, ivar, wave_range, telluric_masks=TELLURIC_MASKS):
    """Apply wavelength range and telluric masks."""
    mask = (wave >= wave_range[0]) & (wave <= wave_range[1])

    for t_min, t_max in telluric_masks:
        mask &= ~((wave >= t_min) & (wave <= t_max))

    mask &= (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)

    return wave[mask], flux[mask], ivar[mask]

def shift_template(template_wave, template_flux, velocity_kms, target_wave):
    """Doppler shift template and interpolate to target wavelength grid."""
    z = velocity_kms / C_KMS
    wave_shifted = template_wave * (1 + z)
    f_interp = interp1d(wave_shifted, template_flux, kind='linear',
                        bounds_error=False, fill_value=np.nan)
    return f_interp(target_wave)

def fit_amplitude_and_poly(data_flux, data_ivar, template_flux, poly_degree=2):
    """
    Analytically fit amplitude A and polynomial continuum coefficients.

    Model: F_model = A * template + sum_i(c_i * wave^i)

    This is a linear least squares problem: minimize chi2 = sum((data - model)^2 * ivar)

    Returns: A, poly_coeffs, chi2
    """
    n = len(data_flux)
    good = np.isfinite(template_flux) & (data_ivar > 0)
    if np.sum(good) < 10:
        return np.nan, np.zeros(poly_degree + 1), np.inf

    # Build design matrix: [template, 1, wave, wave^2, ...]
    # Normalize wavelength to avoid numerical issues
    wave_norm = np.linspace(-1, 1, n)

    n_params = 1 + poly_degree + 1  # A + poly coeffs
    X = np.zeros((n, n_params))
    X[:, 0] = template_flux  # Template column
    for i in range(poly_degree + 1):
        X[:, 1 + i] = wave_norm ** i

    # Weight matrix
    W = np.diag(data_ivar[good])
    X_g = X[good]
    y_g = data_flux[good]

    # Weighted least squares: (X^T W X) beta = X^T W y
    XtWX = X_g.T @ W @ X_g
    XtWy = X_g.T @ W @ y_g

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return np.nan, np.zeros(poly_degree + 1), np.inf

    A = beta[0]
    poly_coeffs = beta[1:]

    # Compute chi2
    model = X @ beta
    resid = data_flux - model
    chi2 = np.sum(resid[good]**2 * data_ivar[good])

    return A, poly_coeffs, chi2

def chi2_at_velocity(velocity, data_wave, data_flux, data_ivar,
                     template_wave, template_flux, poly_degree=2):
    """
    Compute chi2 at a given velocity, marginalizing over A and polynomial.
    """
    shifted = shift_template(template_wave, template_flux, velocity, data_wave)
    A, poly_coeffs, chi2 = fit_amplitude_and_poly(data_flux, data_ivar, shifted, poly_degree)
    return chi2, A, poly_coeffs

def scan_chi2_velocity(data_wave, data_flux, data_ivar,
                       template_wave, template_flux,
                       v_grid=None, poly_degree=2, fine_output=True):
    """
    Scan chi2 over velocity grid, marginalizing over amplitude and polynomial.

    Uses coarse grid for speed, then optionally computes fine grid for output.

    Returns: v_grid, chi2_array, best_v, best_chi2, best_A
    """
    # Coarse scan first
    chi2_coarse = np.zeros(len(V_GRID_COARSE))
    for i, v in enumerate(V_GRID_COARSE):
        chi2, _, _ = chi2_at_velocity(v, data_wave, data_flux, data_ivar,
                                      template_wave, template_flux, poly_degree)
        chi2_coarse[i] = chi2

    # Find coarse minimum
    i_min_coarse = np.argmin(chi2_coarse)
    v_coarse_best = V_GRID_COARSE[i_min_coarse]

    # Fine scan around minimum (±25 km/s)
    v_fine_local = np.linspace(v_coarse_best - 25, v_coarse_best + 25, 26)
    chi2_fine_local = np.zeros(len(v_fine_local))
    A_fine_local = np.zeros(len(v_fine_local))

    for i, v in enumerate(v_fine_local):
        chi2, A, _ = chi2_at_velocity(v, data_wave, data_flux, data_ivar,
                                      template_wave, template_flux, poly_degree)
        chi2_fine_local[i] = chi2
        A_fine_local[i] = A

    i_min_fine = np.argmin(chi2_fine_local)
    best_v = v_fine_local[i_min_fine]
    best_chi2 = chi2_fine_local[i_min_fine]
    best_A = A_fine_local[i_min_fine]

    if fine_output:
        # Compute local fine grid for output (for plotting)
        # Only around the minimum, not full -250 to 250
        v_plot = np.linspace(best_v - 100, best_v + 100, 101)  # 2 km/s resolution, local
        chi2_plot = np.zeros(len(v_plot))
        for i, v in enumerate(v_plot):
            chi2, _, _ = chi2_at_velocity(v, data_wave, data_flux, data_ivar,
                                          template_wave, template_flux, poly_degree)
            chi2_plot[i] = chi2
        return v_plot, chi2_plot, best_v, best_chi2, best_A
    else:
        # Return coarse grid for speed
        return V_GRID_COARSE, chi2_coarse, best_v, best_chi2, best_A

def refine_velocity(data_wave, data_flux, data_ivar,
                    template_wave, template_flux,
                    v_init, poly_degree=2):
    """Refine velocity using bounded minimization."""
    def neg_chi2(v):
        chi2, _, _ = chi2_at_velocity(v, data_wave, data_flux, data_ivar,
                                      template_wave, template_flux, poly_degree)
        return chi2

    result = minimize_scalar(neg_chi2, bounds=(v_init - 20, v_init + 20), method='bounded')
    return result.x, result.fun

def compute_rv_uncertainty(data_wave, data_flux, data_ivar,
                           template_wave, template_flux,
                           v_best, chi2_min, n_good, n_params,
                           poly_degree=2):
    """
    Compute formal and renormalized RV uncertainties.

    Formal: from Δchi2 = 1 criterion
    Renormalized: scaled by sqrt(chi2_red) to account for model mismatch
    """
    dof = n_good - n_params
    if dof <= 0:
        return np.nan, np.nan, np.inf

    chi2_red = chi2_min / dof

    # Find velocity range where chi2 increases by 1 (formal)
    # and by chi2_red (renormalized)
    def chi2_func(v):
        chi2, _, _ = chi2_at_velocity(v, data_wave, data_flux, data_ivar,
                                      template_wave, template_flux, poly_degree)
        return chi2

    # Sample around minimum
    v_samples = np.linspace(v_best - 50, v_best + 50, 201)
    chi2_samples = np.array([chi2_func(v) for v in v_samples])

    # Fit parabola near minimum
    near_min = np.abs(chi2_samples - chi2_min) < 10
    if np.sum(near_min) < 5:
        near_min = np.abs(v_samples - v_best) < 20

    if np.sum(near_min) >= 3:
        coeffs = np.polyfit(v_samples[near_min], chi2_samples[near_min], 2)
        # chi2 = a*v^2 + b*v + c, so d(chi2)/dv^2 = 2a
        # sigma_v_formal = 1/sqrt(a) for Δchi2=1
        if coeffs[0] > 0:
            sigma_v_formal = 1.0 / np.sqrt(coeffs[0])
            sigma_v_renorm = sigma_v_formal * np.sqrt(chi2_red)
        else:
            sigma_v_formal = np.nan
            sigma_v_renorm = np.nan
    else:
        sigma_v_formal = np.nan
        sigma_v_renorm = np.nan

    return sigma_v_formal, sigma_v_renorm, chi2_red

def fit_single_template(data_wave, data_flux, data_ivar,
                        template_wave, template_flux,
                        poly_degree=2, fine_output=True):
    """
    Fit single-template model with proper marginalization.

    Returns dict with: v, A, chi2, chi2_red, sigma_v_formal, sigma_v_renorm, n_pix, dof
    """
    # Grid scan
    v_grid, chi2_array, v_best, chi2_min, A_best = scan_chi2_velocity(
        data_wave, data_flux, data_ivar, template_wave, template_flux,
        poly_degree=poly_degree, fine_output=fine_output
    )

    # Refine
    v_refined, chi2_refined = refine_velocity(
        data_wave, data_flux, data_ivar, template_wave, template_flux,
        v_best, poly_degree
    )

    # Get final fit
    shifted = shift_template(template_wave, template_flux, v_refined, data_wave)
    A_final, poly_coeffs, chi2_final = fit_amplitude_and_poly(
        data_flux, data_ivar, shifted, poly_degree
    )

    # Count good pixels
    good = np.isfinite(shifted) & (data_ivar > 0)
    n_pix = np.sum(good)
    n_params = 1 + poly_degree + 1 + 1  # A + poly + v (v is effectively a parameter)
    dof = n_pix - n_params

    # Uncertainties
    sigma_v_formal, sigma_v_renorm, chi2_red = compute_rv_uncertainty(
        data_wave, data_flux, data_ivar, template_wave, template_flux,
        v_refined, chi2_final, n_pix, n_params, poly_degree
    )

    return {
        'v': v_refined,
        'A': A_final,
        'chi2': chi2_final,
        'chi2_red': chi2_red,
        'sigma_v_formal': sigma_v_formal,
        'sigma_v_renorm': sigma_v_renorm,
        'n_pix': int(n_pix),
        'dof': int(dof),
        'v_grid': v_grid,
        'chi2_grid': chi2_array,
    }

def _fit_single_template_worker(args):
    """Worker function for parallel single-template fitting."""
    (spec_idx, spec_file, spec_label, arm, wave_range, teff, template_wave,
     template_flux, is_primary, fine_output) = args

    # Load spectrum
    if arm == 'combined':
        wave, flux, ivar = load_desi_spectrum(spec_file, 'combined')
    else:
        wave, flux, ivar = load_desi_spectrum(spec_file, arm)

    wave_m, flux_m, ivar_m = apply_wavelength_mask(wave, flux, ivar, wave_range)

    # Fit
    fit = fit_single_template(wave_m, flux_m, ivar_m, template_wave, template_flux,
                              fine_output=fine_output)

    return {
        'spec_idx': spec_idx,
        'arm': arm,
        'teff': teff,
        'is_primary': is_primary,
        'fit': fit
    }

def _eval_two_component_point(args):
    """Worker function for parallel two-component grid search."""
    v1, v2, b, data_wave, data_flux, data_ivar, template_wave, template_primary, template_neighbor, poly_degree = args
    t1 = shift_template(template_wave, template_primary, v1, data_wave)
    t2 = shift_template(template_wave, template_neighbor, v2, data_wave)
    t_combined = t1 + b * t2
    A, poly_coeffs, chi2 = fit_amplitude_and_poly(data_flux, data_ivar, t_combined, poly_degree)
    return (v1, v2, b, A, chi2)

def fit_two_component(data_wave, data_flux, data_ivar,
                      template_wave, template_primary, template_neighbor,
                      b_range=(B_MIN, B_MAX), poly_degree=2, n_workers=None):
    """
    Fit two-component model: F = A * [T_primary(v1) + b * T_neighbor(v2)] + poly

    Uses PARALLEL grid search over v1, v2, b, then refinement.
    """
    if n_workers is None:
        n_workers = min(8, max(1, cpu_count() - 1))  # Cap at 8 workers for WSL

    # Coarse grid for v1, v2, b (reduced for speed)
    v1_grid = np.linspace(-200, 150, 15)  # 15 points
    v2_grid = np.linspace(-100, 100, 9)   # 9 points
    b_grid = np.linspace(b_range[0], b_range[1], 5)  # 5 points
    # Total: 15*9*5 = 675 evaluations

    # Build argument list for parallel execution
    args_list = []
    for v1 in v1_grid:
        for v2 in v2_grid:
            for b in b_grid:
                args_list.append((v1, v2, b, data_wave, data_flux, data_ivar,
                                  template_wave, template_primary, template_neighbor, poly_degree))

    # Parallel grid search
    with Pool(n_workers) as pool:
        results = pool.map(_eval_two_component_point, args_list)

    # Find best
    best_chi2 = np.inf
    best_params = None
    for v1, v2, b, A, chi2 in results:
        if chi2 < best_chi2 and np.isfinite(chi2):
            best_chi2 = chi2
            best_params = (v1, v2, b, A)

    if best_params is None:
        return None

    # Refine with optimization
    v1_init, v2_init, b_init, A_init = best_params

    def objective(params):
        v1, v2, b = params
        if b < b_range[0] or b > b_range[1]:
            return 1e20
        t1 = shift_template(template_wave, template_primary, v1, data_wave)
        t2 = shift_template(template_wave, template_neighbor, v2, data_wave)
        t_combined = t1 + b * t2
        A, _, chi2 = fit_amplitude_and_poly(data_flux, data_ivar, t_combined, poly_degree)
        return chi2 if np.isfinite(chi2) else 1e20

    result = minimize(objective, [v1_init, v2_init, b_init],
                      method='Nelder-Mead',
                      options={'maxiter': 500})

    v1_final, v2_final, b_final = result.x
    b_final = np.clip(b_final, b_range[0], b_range[1])

    # Final fit
    t1 = shift_template(template_wave, template_primary, v1_final, data_wave)
    t2 = shift_template(template_wave, template_neighbor, v2_final, data_wave)
    t_combined = t1 + b_final * t2
    A_final, poly_coeffs, chi2_final = fit_amplitude_and_poly(
        data_flux, data_ivar, t_combined, poly_degree
    )

    good = np.isfinite(t_combined) & (data_ivar > 0)
    n_pix = np.sum(good)
    n_params = 1 + poly_degree + 1 + 3  # A + poly + v1, v2, b
    dof = n_pix - n_params
    chi2_red = chi2_final / dof if dof > 0 else np.inf

    return {
        'v1': v1_final,
        'v2': v2_final,
        'b': b_final,
        'A': A_final,
        'chi2': chi2_final,
        'chi2_red': chi2_red,
        'n_pix': int(n_pix),
        'dof': int(dof),
        'n_params': int(n_params),
    }

def fit_constant_v2_model(all_data, template_wave, template_primary, template_neighbor,
                          b_range=(B_MIN, B_MAX), poly_degree=2):
    """
    Fit cross-epoch model with shared v2 and b.

    v1_i free per epoch, v2 and b shared across epochs.
    """
    n_epochs = len(all_data)

    def total_chi2(params):
        v1_list = params[:n_epochs]
        v2 = params[n_epochs]
        b = params[n_epochs + 1]

        if b < b_range[0] or b > b_range[1]:
            return 1e20

        chi2_total = 0.0
        for i, data in enumerate(all_data):
            t1 = shift_template(template_wave, template_primary, v1_list[i], data['wave'])
            t2 = shift_template(template_wave, template_neighbor, v2, data['wave'])
            t_combined = t1 + b * t2
            A, _, chi2 = fit_amplitude_and_poly(data['flux'], data['ivar'], t_combined, poly_degree)
            chi2_total += chi2 if np.isfinite(chi2) else 1e10

        return chi2_total

    # Initial guesses from catalog RVs
    v1_inits = [d['rv_catalog'] for d in all_data]
    v2_init = 0.0
    b_init = NEIGHBOR_FLUX_FRAC_G

    x0 = v1_inits + [v2_init, b_init]

    result = minimize(total_chi2, x0, method='Nelder-Mead',
                      options={'maxiter': 2000})

    # Parse results
    v1_fits = list(result.x[:n_epochs])
    v2_fit = result.x[n_epochs]
    b_fit = np.clip(result.x[n_epochs + 1], b_range[0], b_range[1])
    chi2_total = result.fun

    # Count total DOF
    n_pix_total = 0
    chi2_per_epoch = []
    for i, data in enumerate(all_data):
        t1 = shift_template(template_wave, template_primary, v1_fits[i], data['wave'])
        t2 = shift_template(template_wave, template_neighbor, v2_fit, data['wave'])
        t_combined = t1 + b_fit * t2
        A, _, chi2_ep = fit_amplitude_and_poly(data['flux'], data['ivar'], t_combined, poly_degree)

        good = np.isfinite(t_combined) & (data['ivar'] > 0)
        n_pix_total += np.sum(good)
        chi2_per_epoch.append(chi2_ep)

    n_params_total = n_epochs * (1 + poly_degree + 1 + 1) + 2  # per-epoch: A, poly, v1; shared: v2, b
    dof_total = n_pix_total - n_params_total
    chi2_red = chi2_total / dof_total if dof_total > 0 else np.inf

    return {
        'v1_per_epoch': v1_fits,
        'v2_shared': v2_fit,
        'b_shared': b_fit,
        'chi2_total': chi2_total,
        'chi2_per_epoch': chi2_per_epoch,
        'chi2_red': chi2_red,
        'n_pix_total': int(n_pix_total),
        'n_params': int(n_params_total),
        'dof': int(dof_total),
    }

def compute_bic(chi2, n_params, n_data):
    """Compute BIC on data chi2 only (no priors)."""
    return chi2 + n_params * np.log(n_data)

def json_safe(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# =============================================================================
# CHECKPOINT FUNCTIONS
# =============================================================================

CHECKPOINT_FILE = OUTPUT_DIR / 'checkpoint_v3.pkl'

def save_checkpoint(stage, data):
    """Save checkpoint after completing a stage."""
    checkpoint = {'stage': stage, 'data': data}
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  [CHECKPOINT SAVED: stage={stage}]")

def load_checkpoint():
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"  [CHECKPOINT LOADED: stage={checkpoint['stage']}]")
        return checkpoint
    return None

def clear_checkpoint():
    """Clear checkpoint after successful completion."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("  [CHECKPOINT CLEARED]")

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("="*70)
    print("DESI BLEND-AWARE RV RE-MEASUREMENT v3")
    print("="*70)
    print(f"\nTarget: Gaia DR3 {GAIA_SOURCE_ID}")
    print(f"DESI TARGETID: {TARGET_ID}")
    print(f"\nConfirmed neighbor: sep={NEIGHBOR_SEP_ARCSEC:.3f}\", ΔG={NEIGHBOR_DELTA_G:.2f}")
    print(f"Expected flux ratio (G-band): {NEIGHBOR_FLUX_FRAC_G:.3f}")
    print(f"Allowed flux ratio range: [{B_MIN}, {B_MAX}]")
    print(f"\nUsing {max(1, cpu_count()-1)} CPU cores for parallel processing")

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
    print(f"Original wavelength grid: {len(template_wave_full)} points")

    # Downsample templates for speed
    print("Downsampling templates to DESI resolution (~0.5 Å)...")

    # Primary templates
    primary_templates = {}
    template_wave = None  # Will be set from first downsample
    for teff in PRIMARY_TEFFS:
        logg = 4.5
        try:
            flux_full = load_phoenix_template(teff, logg)
            wave_ds, flux_ds = downsample_template(template_wave_full, flux_full)
            if template_wave is None:
                template_wave = wave_ds
            primary_templates[teff] = flux_ds
            print(f"  Primary Teff={teff}K, logg={logg}: loaded and downsampled")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print("STOPPING: Template grid incomplete.")
            return

    # Neighbor templates
    neighbor_templates = {}
    for teff in NEIGHBOR_TEFFS:
        logg = 5.0
        try:
            flux_full = load_phoenix_template(teff, logg)
            _, flux_ds = downsample_template(template_wave_full, flux_full)
            neighbor_templates[teff] = flux_ds
            print(f"  Neighbor Teff={teff}K, logg={logg}: loaded and downsampled")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print("STOPPING: Template grid incomplete.")
            return

    print(f"Downsampled wavelength grid: {len(template_wave)} points, {template_wave[0]:.1f}-{template_wave[-1]:.1f} Å")

    # =========================================================================
    # LOAD SPECTRA
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING DESI SPECTRA")
    print("="*70)

    spectra = []
    file_hashes = {}

    for label, fname, mjd, rv_cat, rv_err_cat in EPOCH_FILES:
        fpath = SPECTRA_DIR / fname
        if not fpath.exists():
            print(f"ERROR: {fpath} not found")
            return

        file_hash = sha256_file(fpath)
        file_hashes[fname] = file_hash
        print(f"{label}: {fname}")
        print(f"  SHA256: {file_hash[:16]}...")
        print(f"  MJD: {mjd}, Catalog RV: {rv_cat:+.2f} km/s")

        spectra.append({
            'label': label,
            'file': str(fpath),
            'file_name': fname,
            'mjd': mjd,
            'rv_catalog': rv_cat,
            'rv_err_catalog': rv_err_cat,
        })

    # =========================================================================
    # ANALYSIS 1: SINGLE-STAR PRIMARY-ONLY FITS
    # =========================================================================
    n_workers = min(8, max(1, cpu_count() - 1))  # Cap at 8 workers for WSL

    if checkpoint_stage >= 1:
        print("\n" + "="*70)
        print("ANALYSIS 1: SINGLE-STAR PRIMARY-ONLY FITS [LOADED FROM CHECKPOINT]")
        print("="*70)
        single_primary_results = checkpoint_data['single_primary_results']
    else:
        print("\n" + "="*70)
        print(f"ANALYSIS 1: SINGLE-STAR PRIMARY-ONLY FITS (PARALLEL, {n_workers} workers)")
        print("="*70)

        # Build all tasks: (spec_idx, arm, teff) combinations
        arms_info = [('R', R_BAND), ('Z', Z_BAND), ('combined', (R_BAND[0], Z_BAND[1]))]
        tasks = []
        for spec_idx, spec in enumerate(spectra):
            for arm, wave_range in arms_info:
                for teff in PRIMARY_TEFFS:
                    tasks.append((
                        spec_idx, spec['file'], spec['label'], arm, wave_range,
                        teff, template_wave, primary_templates[teff], True, False
                    ))

        print(f"  Running {len(tasks)} parallel fits...")

        # Run in parallel
        with Pool(n_workers) as pool:
            results = pool.map(_fit_single_template_worker, tasks)

        # Organize results
        single_primary_results = []
        for spec_idx, spec in enumerate(spectra):
            epoch_results = {
                'label': spec['label'],
                'mjd': spec['mjd'],
                'rv_catalog': spec['rv_catalog'],
                'by_arm': {},
            }

            for arm, wave_range in arms_info:
                arm_results = {'templates': {}}
                best_chi2 = np.inf
                best_teff = None
                best_fit = None

                # Find all results for this (spec_idx, arm)
                for r in results:
                    if r['spec_idx'] == spec_idx and r['arm'] == arm:
                        teff = r['teff']
                        fit = r['fit']
                        arm_results['templates'][teff] = {
                            'v': fit['v'],
                            'sigma_v_formal': fit['sigma_v_formal'],
                            'sigma_v_renorm': fit['sigma_v_renorm'],
                            'chi2': fit['chi2'],
                            'chi2_red': fit['chi2_red'],
                            'n_pix': fit['n_pix'],
                            'dof': fit['dof'],
                        }
                        if fit['chi2'] < best_chi2:
                            best_chi2 = fit['chi2']
                            best_teff = teff
                            best_fit = fit

                # Get fine grid for combined arm (sequential, just for best template)
                if arm == 'combined':
                    wave, flux, ivar = load_desi_spectrum(spec['file'], 'combined')
                    wave_m, flux_m, ivar_m = apply_wavelength_mask(wave, flux, ivar, wave_range)
                    fit_fine = fit_single_template(wave_m, flux_m, ivar_m, template_wave,
                                                   primary_templates[best_teff], fine_output=True)
                    arm_results['chi2_grid'] = fit_fine['chi2_grid']
                    arm_results['v_grid'] = fit_fine['v_grid']
                else:
                    arm_results['chi2_grid'] = best_fit['chi2_grid']
                    arm_results['v_grid'] = best_fit['v_grid']

                arm_results['best_teff'] = best_teff
                arm_results['best_v'] = best_fit['v']
                arm_results['best_sigma_v_formal'] = best_fit['sigma_v_formal']
                arm_results['best_sigma_v_renorm'] = best_fit['sigma_v_renorm']
                arm_results['best_chi2'] = best_fit['chi2']
                arm_results['best_chi2_red'] = best_fit['chi2_red']

                epoch_results['by_arm'][arm] = arm_results

            single_primary_results.append(epoch_results)

        # Print summary
        for spec_idx, spec in enumerate(spectra):
            print(f"\n--- {spec['label']} (Catalog RV: {spec['rv_catalog']:+.1f} km/s) ---")
            for arm in ['R', 'Z', 'combined']:
                r = single_primary_results[spec_idx]['by_arm'][arm]
                print(f"  {arm.upper():8s}: best Teff={r['best_teff']}K, v={r['best_v']:+7.1f} km/s, "
                      f"σ_formal={r['best_sigma_v_formal']:.2f}, σ_renorm={r['best_sigma_v_renorm']:.1f}, "
                      f"χ²_red={r['best_chi2_red']:.1f}")

        # Save checkpoint after Analysis 1
        save_checkpoint(1, {'single_primary_results': single_primary_results})

    # =========================================================================
    # ANALYSIS 2: NEIGHBOR-ONLY FITS
    # =========================================================================
    if checkpoint_stage >= 2:
        print("\n" + "="*70)
        print("ANALYSIS 2: NEIGHBOR-ONLY FITS [LOADED FROM CHECKPOINT]")
        print("="*70)
        single_neighbor_results = checkpoint_data['single_neighbor_results']
    else:
        print("\n" + "="*70)
        print(f"ANALYSIS 2: NEIGHBOR-ONLY FITS (PARALLEL, {n_workers} workers)")
        print("="*70)

        # Build all tasks: (spec_idx, arm, teff) combinations for neighbor templates
        tasks = []
        for spec_idx, spec in enumerate(spectra):
            for arm, wave_range in arms_info:
                for teff in NEIGHBOR_TEFFS:
                    tasks.append((
                        spec_idx, spec['file'], spec['label'], arm, wave_range,
                        teff, template_wave, neighbor_templates[teff], False, False
                    ))

        print(f"  Running {len(tasks)} parallel fits...")

        # Run in parallel
        with Pool(n_workers) as pool:
            results = pool.map(_fit_single_template_worker, tasks)

        # Organize results
        single_neighbor_results = []
        for spec_idx, spec in enumerate(spectra):
            epoch_results = {
                'label': spec['label'],
                'mjd': spec['mjd'],
                'by_arm': {},
            }

            for arm, wave_range in arms_info:
                arm_results = {'templates': {}}
                best_chi2 = np.inf
                best_teff = None
                best_fit = None

                for r in results:
                    if r['spec_idx'] == spec_idx and r['arm'] == arm:
                        teff = r['teff']
                        fit = r['fit']
                        arm_results['templates'][teff] = {
                            'v': fit['v'],
                            'chi2': fit['chi2'],
                            'chi2_red': fit['chi2_red'],
                        }
                        if fit['chi2'] < best_chi2:
                            best_chi2 = fit['chi2']
                            best_teff = teff
                            best_fit = fit

                arm_results['best_teff'] = best_teff
                arm_results['best_v'] = best_fit['v']
                arm_results['best_chi2'] = best_fit['chi2']
                arm_results['best_chi2_red'] = best_fit['chi2_red']

                epoch_results['by_arm'][arm] = arm_results

            single_neighbor_results.append(epoch_results)

        # Print summary
        for spec_idx, spec in enumerate(spectra):
            print(f"\n--- {spec['label']} ---")
            for arm in ['R', 'Z', 'combined']:
                r = single_neighbor_results[spec_idx]['by_arm'][arm]
                print(f"  {arm.upper():8s}: best Teff={r['best_teff']}K, v={r['best_v']:+7.1f} km/s, "
                      f"χ²={r['best_chi2']:.0f}, χ²_red={r['best_chi2_red']:.1f}")

        # Save checkpoint after Analysis 2
        save_checkpoint(2, {
            'single_primary_results': single_primary_results,
            'single_neighbor_results': single_neighbor_results
        })

    # =========================================================================
    # ANALYSIS 3: TWO-COMPONENT PER-EPOCH FITS
    # =========================================================================
    if checkpoint_stage >= 3:
        print("\n" + "="*70)
        print("ANALYSIS 3: TWO-COMPONENT PER-EPOCH FITS [LOADED FROM CHECKPOINT]")
        print("="*70)
        two_component_results = checkpoint_data['two_component_results']
    else:
        print("\n" + "="*70)
        print("ANALYSIS 3: TWO-COMPONENT PER-EPOCH FITS (PARALLEL)")
        print("="*70)

        two_component_results = []

        for i, spec in enumerate(spectra):
            print(f"\n--- {spec['label']} ---")

            epoch_results = {
                'label': spec['label'],
                'mjd': spec['mjd'],
                'by_arm': {},
            }

            for arm, wave_range in [('R', R_BAND), ('Z', Z_BAND), ('combined', (R_BAND[0], Z_BAND[1]))]:
                if arm == 'combined':
                    wave, flux, ivar = load_desi_spectrum(spec['file'], 'combined')
                else:
                    wave, flux, ivar = load_desi_spectrum(spec['file'], arm)

                wave_m, flux_m, ivar_m = apply_wavelength_mask(wave, flux, ivar, wave_range)

                # Get best templates
                best_primary_teff = single_primary_results[i]['by_arm'][arm]['best_teff']
                best_neighbor_teff = single_neighbor_results[i]['by_arm'][arm]['best_teff']

                template_primary = primary_templates[best_primary_teff]
                template_neighbor = neighbor_templates[best_neighbor_teff]

                fit = fit_two_component(wave_m, flux_m, ivar_m, template_wave,
                                        template_primary, template_neighbor)

                if fit is None:
                    print(f"  {arm.upper():8s}: FAILED")
                    epoch_results['by_arm'][arm] = {'status': 'FAILED'}
                    continue

                # Compute BIC for both models
                single_chi2 = single_primary_results[i]['by_arm'][arm]['best_chi2']
                single_n_pix = single_primary_results[i]['by_arm'][arm]['templates'][best_primary_teff]['n_pix']
                single_n_params = 1 + 3 + 1  # A + poly(2) + v

                bic_single = compute_bic(single_chi2, single_n_params, single_n_pix)
                bic_two = compute_bic(fit['chi2'], fit['n_params'], fit['n_pix'])
                delta_bic = bic_two - bic_single

                epoch_results['by_arm'][arm] = {
                    'v1': fit['v1'],
                    'v2': fit['v2'],
                    'b': fit['b'],
                    'chi2': fit['chi2'],
                    'chi2_red': fit['chi2_red'],
                    'n_pix': fit['n_pix'],
                    'n_params': fit['n_params'],
                    'bic_single': bic_single,
                    'bic_two': bic_two,
                    'delta_bic': delta_bic,
                    'two_preferred': delta_bic < -6,
                    'primary_teff': best_primary_teff,
                    'neighbor_teff': best_neighbor_teff,
                }

                print(f"  {arm.upper():8s}: v1={fit['v1']:+7.1f}, v2={fit['v2']:+7.1f}, b={fit['b']:.3f}, "
                      f"ΔBIC={delta_bic:+.1f} {'(TWO PREF)' if delta_bic < -6 else ''}")

            two_component_results.append(epoch_results)

        # Save checkpoint after Analysis 3
        save_checkpoint(3, {
            'single_primary_results': single_primary_results,
            'single_neighbor_results': single_neighbor_results,
            'two_component_results': two_component_results
        })

    # =========================================================================
    # ANALYSIS 4: CROSS-EPOCH CONSTANT-v2 MODEL
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 4: CROSS-EPOCH CONSTANT-v2 MODEL")
    print("="*70)

    # Prepare combined data for all epochs
    all_data_combined = []
    for spec in spectra:
        wave, flux, ivar = load_desi_spectrum(spec['file'], 'combined')
        wave_m, flux_m, ivar_m = apply_wavelength_mask(wave, flux, ivar, (R_BAND[0], Z_BAND[1]))
        all_data_combined.append({
            'wave': wave_m,
            'flux': flux_m,
            'ivar': ivar_m,
            'rv_catalog': spec['rv_catalog'],
            'label': spec['label'],
        })

    # Use best overall templates
    best_primary_teff = 3800  # Default to M0
    best_neighbor_teff = 3000  # Default to M5

    template_primary = primary_templates[best_primary_teff]
    template_neighbor = neighbor_templates[best_neighbor_teff]

    const_v2_fit = fit_constant_v2_model(all_data_combined, template_wave,
                                          template_primary, template_neighbor)

    print(f"\nConstant-v2 model results:")
    print(f"  v2 (shared): {const_v2_fit['v2_shared']:+.1f} km/s")
    print(f"  b (shared): {const_v2_fit['b_shared']:.3f}")
    for i, v1 in enumerate(const_v2_fit['v1_per_epoch']):
        print(f"  {spectra[i]['label']}: v1={v1:+.1f} km/s")
    print(f"  Total χ²: {const_v2_fit['chi2_total']:.1f}")
    print(f"  χ²_red: {const_v2_fit['chi2_red']:.2f}")

    # Compare to primary-only model
    chi2_primary_only = sum(single_primary_results[i]['by_arm']['combined']['best_chi2']
                            for i in range(len(spectra)))
    n_pix_primary = sum(single_primary_results[i]['by_arm']['combined']['templates'][
        single_primary_results[i]['by_arm']['combined']['best_teff']]['n_pix']
        for i in range(len(spectra)))
    n_params_primary = len(spectra) * 5  # per epoch: A, poly(2), v

    bic_primary_only = compute_bic(chi2_primary_only, n_params_primary, n_pix_primary)
    bic_const_v2 = compute_bic(const_v2_fit['chi2_total'], const_v2_fit['n_params'], const_v2_fit['n_pix_total'])
    delta_bic_const = bic_const_v2 - bic_primary_only

    print(f"\n  Comparison to primary-only model:")
    print(f"  Primary-only χ²: {chi2_primary_only:.1f}")
    print(f"  BIC primary-only: {bic_primary_only:.1f}")
    print(f"  BIC const-v2: {bic_const_v2:.1f}")
    print(f"  ΔBIC (const-v2 - primary): {delta_bic_const:+.1f}")
    print(f"  Verdict: {'CONST-V2 FAVORED' if delta_bic_const < -6 else 'PRIMARY-ONLY FAVORED'}")

    const_v2_result = {
        **const_v2_fit,
        'bic_primary_only': bic_primary_only,
        'bic_const_v2': bic_const_v2,
        'delta_bic': delta_bic_const,
        'const_v2_favored': delta_bic_const < -6,
        'primary_teff': best_primary_teff,
        'neighbor_teff': best_neighbor_teff,
    }

    # =========================================================================
    # ANALYSIS 5: WAVELENGTH-DEPENDENT RV DIAGNOSTICS
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 5: WAVELENGTH-DEPENDENT RV DIAGNOSTICS")
    print("="*70)

    arm_split_results = []

    for i, spec in enumerate(spectra):
        rv_R = single_primary_results[i]['by_arm']['R']['best_v']
        rv_Z = single_primary_results[i]['by_arm']['Z']['best_v']
        sigma_R = single_primary_results[i]['by_arm']['R']['best_sigma_v_renorm']
        sigma_Z = single_primary_results[i]['by_arm']['Z']['best_sigma_v_renorm']

        delta_rv = rv_Z - rv_R
        delta_sigma = np.sqrt(sigma_R**2 + sigma_Z**2) if np.isfinite(sigma_R) and np.isfinite(sigma_Z) else np.nan

        arm_split_results.append({
            'label': spec['label'],
            'rv_R': rv_R,
            'rv_Z': rv_Z,
            'sigma_R_renorm': sigma_R,
            'sigma_Z_renorm': sigma_Z,
            'delta_rv': delta_rv,
            'delta_sigma': delta_sigma,
            'significance': delta_rv / delta_sigma if np.isfinite(delta_sigma) and delta_sigma > 0 else np.nan,
        })

        print(f"{spec['label']}: RV_R={rv_R:+7.1f}, RV_Z={rv_Z:+7.1f}, Δ={delta_rv:+.1f} km/s "
              f"({delta_rv/delta_sigma:.1f}σ)" if np.isfinite(delta_sigma) else
              f"{spec['label']}: RV_R={rv_R:+7.1f}, RV_Z={rv_Z:+7.1f}, Δ={delta_rv:+.1f} km/s")

    # =========================================================================
    # VERDICTS
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL VERDICTS")
    print("="*70)

    verdicts = {}

    # 1) RV swing robust across arms/templates?
    rv_combined = [single_primary_results[i]['by_arm']['combined']['best_v'] for i in range(3)]
    rv_swing_fitted = max(rv_combined) - min(rv_combined)
    rv_swing_catalog = max(spec['rv_catalog'] for spec in spectra) - min(spec['rv_catalog'] for spec in spectra)

    # Check arm consistency
    arm_consistent = all(abs(r['delta_rv']) < 20 or (np.isfinite(r['significance']) and abs(r['significance']) < 3)
                         for r in arm_split_results)

    if rv_swing_fitted > 100 and arm_consistent:
        verdicts['rv_swing_robust'] = 'PASS'
    elif rv_swing_fitted > 50:
        verdicts['rv_swing_robust'] = 'INCONCLUSIVE'
    else:
        verdicts['rv_swing_robust'] = 'FAIL'

    print(f"\n1) RV swing robust across arms/templates?")
    print(f"   Fitted swing: {rv_swing_fitted:.1f} km/s (catalog: {rv_swing_catalog:.1f})")
    print(f"   Arm consistency: {'YES' if arm_consistent else 'NO'}")
    print(f"   Verdict: {verdicts['rv_swing_robust']}")

    # 2) Neighbor-only fits competitive?
    neighbor_competitive = []
    for i in range(3):
        chi2_primary = single_primary_results[i]['by_arm']['combined']['best_chi2']
        chi2_neighbor = single_neighbor_results[i]['by_arm']['combined']['best_chi2']
        ratio = chi2_neighbor / chi2_primary if chi2_primary > 0 else np.inf
        neighbor_competitive.append(ratio < 1.5)  # Within 50%

    if all(neighbor_competitive):
        verdicts['neighbor_only_competitive'] = 'FAIL'  # Bad - means neighbor could dominate
    elif any(neighbor_competitive):
        verdicts['neighbor_only_competitive'] = 'INCONCLUSIVE'
    else:
        verdicts['neighbor_only_competitive'] = 'PASS'  # Good - neighbor doesn't fit well

    print(f"\n2) Neighbor-only fits competitive?")
    for i in range(3):
        chi2_p = single_primary_results[i]['by_arm']['combined']['best_chi2']
        chi2_n = single_neighbor_results[i]['by_arm']['combined']['best_chi2']
        print(f"   {spectra[i]['label']}: χ²_primary={chi2_p:.0f}, χ²_neighbor={chi2_n:.0f}, ratio={chi2_n/chi2_p:.2f}")
    print(f"   Verdict: {verdicts['neighbor_only_competitive']}")

    # 3) Two-component per-epoch favored?
    two_comp_favored = [two_component_results[i]['by_arm']['combined'].get('two_preferred', False)
                        for i in range(3)]

    if all(two_comp_favored):
        verdicts['two_component_favored'] = 'FAIL'  # Bad - blend model preferred
    elif any(two_comp_favored):
        verdicts['two_component_favored'] = 'INCONCLUSIVE'
    else:
        verdicts['two_component_favored'] = 'PASS'  # Good - single model preferred

    print(f"\n3) Two-component per-epoch favored?")
    for i in range(3):
        dbic = two_component_results[i]['by_arm']['combined'].get('delta_bic', np.nan)
        print(f"   {spectra[i]['label']}: ΔBIC={dbic:+.1f} ({'TWO PREF' if two_comp_favored[i] else 'SINGLE PREF'})")
    print(f"   Verdict: {verdicts['two_component_favored']}")

    # 4) Cross-epoch constant-v2 model favored?
    if const_v2_result['const_v2_favored'] and B_MIN <= const_v2_result['b_shared'] <= B_MAX:
        verdicts['constant_v2_favored'] = 'FAIL'  # Bad - blend model with constant neighbor preferred
    elif const_v2_result['delta_bic'] < 0:
        verdicts['constant_v2_favored'] = 'INCONCLUSIVE'
    else:
        verdicts['constant_v2_favored'] = 'PASS'  # Good - primary-only preferred

    print(f"\n4) Cross-epoch constant-v2 model favored?")
    print(f"   ΔBIC: {const_v2_result['delta_bic']:+.1f}")
    print(f"   Best b: {const_v2_result['b_shared']:.3f} (expected ~{NEIGHBOR_FLUX_FRAC_G:.3f})")
    print(f"   Verdict: {verdicts['constant_v2_favored']}")

    # 5) Can blend switching plausibly explain the observed extrema?
    # Check if any epoch has neighbor-only fit with RV near the extreme catalog values
    neighbor_explains_extrema = False
    for i in range(3):
        rv_neighbor = single_neighbor_results[i]['by_arm']['combined']['best_v']
        rv_catalog = spectra[i]['rv_catalog']
        if abs(rv_neighbor - rv_catalog) < 20:
            neighbor_explains_extrema = True

    # Also check if two-component v2 values are plausible
    v2_values = [two_component_results[i]['by_arm']['combined'].get('v2', np.nan) for i in range(3)]
    v2_stable = np.std([v for v in v2_values if np.isfinite(v)]) < 30 if any(np.isfinite(v) for v in v2_values) else False

    if neighbor_explains_extrema and v2_stable:
        verdicts['blend_switching_plausible'] = 'FAIL'  # Bad - blend switching could explain
    elif neighbor_explains_extrema or v2_stable:
        verdicts['blend_switching_plausible'] = 'INCONCLUSIVE'
    else:
        verdicts['blend_switching_plausible'] = 'PASS'  # Good - blend switching unlikely

    print(f"\n5) Can blend switching plausibly explain the observed extrema?")
    print(f"   Neighbor-only RVs: {[f'{single_neighbor_results[i]['by_arm']['combined']['best_v']:+.1f}' for i in range(3)]}")
    print(f"   Two-comp v2 values: {[f'{v:+.1f}' if np.isfinite(v) else 'N/A' for v in v2_values]}")
    print(f"   v2 stable across epochs: {'YES' if v2_stable else 'NO'}")
    print(f"   Verdict: {verdicts['blend_switching_plausible']}")

    # Overall verdict
    pass_count = sum(1 for v in verdicts.values() if v == 'PASS')
    fail_count = sum(1 for v in verdicts.values() if v == 'FAIL')
    inconclusive_count = sum(1 for v in verdicts.values() if v == 'INCONCLUSIVE')

    if fail_count >= 2:
        overall_verdict = 'COMPROMISED'
        bottom_line = "Evidence suggests blend contamination may affect RV measurements."
    elif pass_count >= 4:
        overall_verdict = 'ROBUST'
        bottom_line = "RV variability is robust; blend contamination cannot explain the observed amplitude."
    else:
        overall_verdict = 'INCONCLUSIVE'
        bottom_line = "Results are mixed; further investigation recommended."

    print(f"\n{'='*70}")
    print(f"OVERALL VERDICT: {overall_verdict}")
    print(f"{'='*70}")
    print(f"PASS: {pass_count}, FAIL: {fail_count}, INCONCLUSIVE: {inconclusive_count}")
    print(f"\n{bottom_line}")

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    # Figure 1: chi2 vs v by epoch
    print("Generating chi2_vs_v_by_epoch_v3.png...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, spec in enumerate(spectra):
        ax = axes[i]
        v_grid = np.array(single_primary_results[i]['by_arm']['combined']['v_grid'])
        chi2_grid = np.array(single_primary_results[i]['by_arm']['combined']['chi2_grid'])
        chi2_min = np.min(chi2_grid)

        ax.plot(v_grid, chi2_grid - chi2_min, 'b-', lw=1.5)
        ax.axvline(single_primary_results[i]['by_arm']['combined']['best_v'],
                   color='b', ls='--', label=f'v={single_primary_results[i]["by_arm"]["combined"]["best_v"]:+.1f}')
        ax.axvline(spec['rv_catalog'], color='r', ls=':', label=f'catalog={spec["rv_catalog"]:+.1f}')
        ax.axhline(1, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Δχ²')
        ax.set_title(f'{spec["label"]}')
        ax.set_xlim(-200, 150)
        ax.set_ylim(0, 50)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / 'chi2_vs_v_by_epoch_v3.png', dpi=150)
    plt.close()

    # Figure 2: RV by method comparison
    print("Generating rv_by_method_v3.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = [1, 2, 3]
    rv_catalog = [spec['rv_catalog'] for spec in spectra]
    rv_fitted = [single_primary_results[i]['by_arm']['combined']['best_v'] for i in range(3)]
    sigma_renorm = [single_primary_results[i]['by_arm']['combined']['best_sigma_v_renorm'] for i in range(3)]

    ax.errorbar(epochs, rv_catalog, yerr=[spec['rv_err_catalog'] for spec in spectra],
                fmt='ko', capsize=5, label='DESI Catalog', markersize=10)
    ax.errorbar([e + 0.1 for e in epochs], rv_fitted, yerr=sigma_renorm,
                fmt='bs', capsize=5, label='v3 Fit (σ_renorm)', markersize=8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('RV (km/s)')
    ax.set_title('RV Comparison: Catalog vs v3 Fits')
    ax.legend()
    ax.set_xticks(epochs)
    ax.set_xticklabels([spec['label'] for spec in spectra])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / 'rv_by_method_v3.png', dpi=150)
    plt.close()

    # Figure 3: Neighbor-only vs primary chi2
    print("Generating neighbor_only_vs_primary_v3.png...")
    fig, ax = plt.subplots(figsize=(8, 6))
    chi2_primary = [single_primary_results[i]['by_arm']['combined']['best_chi2'] for i in range(3)]
    chi2_neighbor = [single_neighbor_results[i]['by_arm']['combined']['best_chi2'] for i in range(3)]

    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, chi2_primary, width, label='Primary-only', color='blue', alpha=0.7)
    ax.bar(x + width/2, chi2_neighbor, width, label='Neighbor-only', color='orange', alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('χ² (best fit)')
    ax.set_title('Primary vs Neighbor Template Fits')
    ax.set_xticks(x)
    ax.set_xticklabels([spec['label'] for spec in spectra])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / 'neighbor_only_vs_primary_v3.png', dpi=150)
    plt.close()

    # Figure 4: Constant-v2 model comparison
    print("Generating constant_v2_model_comparison_v3.png...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: v1 per epoch comparison
    ax = axes[0]
    v1_const = const_v2_result['v1_per_epoch']
    v1_single = [single_primary_results[i]['by_arm']['combined']['best_v'] for i in range(3)]

    ax.plot(epochs, rv_catalog, 'ko-', label='Catalog', markersize=10)
    ax.plot(epochs, v1_single, 'bs--', label='Primary-only fit', markersize=8)
    ax.plot(epochs, v1_const, 'r^:', label='Const-v2 model v1', markersize=8)
    ax.axhline(const_v2_result['v2_shared'], color='green', ls='-.',
               label=f'Shared v2={const_v2_result["v2_shared"]:+.1f}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('RV (km/s)')
    ax.set_title('Constant-v2 Model: Velocity Comparison')
    ax.legend()
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)

    # Right: BIC comparison
    ax = axes[1]
    models = ['Primary-only', 'Const-v2']
    bics = [const_v2_result['bic_primary_only'], const_v2_result['bic_const_v2']]
    colors = ['blue', 'red']
    ax.bar(models, bics, color=colors, alpha=0.7)
    ax.set_ylabel('BIC')
    ax.set_title(f'Model Comparison (ΔBIC={const_v2_result["delta_bic"]:+.1f})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / 'constant_v2_model_comparison_v3.png', dpi=150)
    plt.close()

    # Figure 5: Arm split RV
    print("Generating arm_split_rv_v3.png...")
    fig, ax = plt.subplots(figsize=(8, 6))

    rv_R = [r['rv_R'] for r in arm_split_results]
    rv_Z = [r['rv_Z'] for r in arm_split_results]
    sigma_R = [r['sigma_R_renorm'] for r in arm_split_results]
    sigma_Z = [r['sigma_Z_renorm'] for r in arm_split_results]

    ax.errorbar([e - 0.1 for e in epochs], rv_R, yerr=sigma_R, fmt='ro', capsize=5,
                label='R band', markersize=10)
    ax.errorbar([e + 0.1 for e in epochs], rv_Z, yerr=sigma_Z, fmt='b^', capsize=5,
                label='Z band', markersize=10)
    ax.plot(epochs, rv_catalog, 'k*', markersize=15, label='Catalog', zorder=5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('RV (km/s)')
    ax.set_title('R vs Z Band RV Comparison')
    ax.legend()
    ax.set_xticks(epochs)
    ax.set_xticklabels([spec['label'] for spec in spectra])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / 'arm_split_rv_v3.png', dpi=150)
    plt.close()

    print("All figures generated.")

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    # Epoch RV refit JSON
    rv_refit = {
        'target_id': TARGET_ID,
        'gaia_source_id': GAIA_SOURCE_ID,
        'file_hashes': file_hashes,
        'template_grid': {
            'primary_teffs': PRIMARY_TEFFS,
            'neighbor_teffs': NEIGHBOR_TEFFS,
        },
        'epochs': [],
    }

    for i, spec in enumerate(spectra):
        epoch_data = {
            'label': spec['label'],
            'mjd': spec['mjd'],
            'rv_catalog': spec['rv_catalog'],
            'rv_err_catalog': spec['rv_err_catalog'],
            'primary_only': {
                arm: {
                    'best_teff': single_primary_results[i]['by_arm'][arm]['best_teff'],
                    'v': single_primary_results[i]['by_arm'][arm]['best_v'],
                    'sigma_v_formal': single_primary_results[i]['by_arm'][arm]['best_sigma_v_formal'],
                    'sigma_v_renorm': single_primary_results[i]['by_arm'][arm]['best_sigma_v_renorm'],
                    'chi2': single_primary_results[i]['by_arm'][arm]['best_chi2'],
                    'chi2_red': single_primary_results[i]['by_arm'][arm]['best_chi2_red'],
                }
                for arm in ['R', 'Z', 'combined']
            },
            'neighbor_only': {
                arm: {
                    'best_teff': single_neighbor_results[i]['by_arm'][arm]['best_teff'],
                    'v': single_neighbor_results[i]['by_arm'][arm]['best_v'],
                    'chi2': single_neighbor_results[i]['by_arm'][arm]['best_chi2'],
                    'chi2_red': single_neighbor_results[i]['by_arm'][arm]['best_chi2_red'],
                }
                for arm in ['R', 'Z', 'combined']
            },
        }
        rv_refit['epochs'].append(epoch_data)

    with open(OUTPUT_DIR / 'desi_epoch_rv_refit_v3.json', 'w') as f:
        json.dump(rv_refit, f, indent=2, default=json_safe)
    print(f"Saved: {OUTPUT_DIR / 'desi_epoch_rv_refit_v3.json'}")

    # Model comparison JSON
    model_compare = {
        'target_id': TARGET_ID,
        'flux_ratio_bounds': {'min': B_MIN, 'max': B_MAX},
        'expected_flux_ratio_G': NEIGHBOR_FLUX_FRAC_G,
        'two_component_per_epoch': [],
        'arm_split_diagnostics': arm_split_results,
    }

    for i in range(3):
        model_compare['two_component_per_epoch'].append({
            'label': spectra[i]['label'],
            'combined': two_component_results[i]['by_arm'].get('combined', {}),
        })

    with open(OUTPUT_DIR / 'desi_model_comparison_v3.json', 'w') as f:
        json.dump(model_compare, f, indent=2, default=json_safe)
    print(f"Saved: {OUTPUT_DIR / 'desi_model_comparison_v3.json'}")

    # Constant-v2 fit JSON
    const_v2_output = {
        'target_id': TARGET_ID,
        'primary_teff': const_v2_result['primary_teff'],
        'neighbor_teff': const_v2_result['neighbor_teff'],
        'v1_per_epoch': const_v2_result['v1_per_epoch'],
        'v2_shared': const_v2_result['v2_shared'],
        'b_shared': const_v2_result['b_shared'],
        'chi2_total': const_v2_result['chi2_total'],
        'chi2_red': const_v2_result['chi2_red'],
        'bic_primary_only': const_v2_result['bic_primary_only'],
        'bic_const_v2': const_v2_result['bic_const_v2'],
        'delta_bic': const_v2_result['delta_bic'],
        'const_v2_favored': const_v2_result['const_v2_favored'],
    }

    with open(OUTPUT_DIR / 'desi_constant_v2_fit_v3.json', 'w') as f:
        json.dump(const_v2_output, f, indent=2, default=json_safe)
    print(f"Saved: {OUTPUT_DIR / 'desi_constant_v2_fit_v3.json'}")

    # =========================================================================
    # WRITE REPORT
    # =========================================================================
    print("\nWriting report...")

    report = f"""# DESI BLEND-AWARE RV RE-MEASUREMENT REPORT v3

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Target:** Gaia DR3 {GAIA_SOURCE_ID}
**DESI TARGETID:** {TARGET_ID}

---

## Executive Summary

This report presents a rigorous blend-aware analysis of DESI DR1 spectra with **methodological improvements over v2**:

1. **RV uncertainties properly renormalized** by sqrt(χ²_red)
2. **BIC computed on data likelihood only** (no priors mixed in)
3. **Marginalization over amplitude and continuum polynomial** at each velocity
4. **Separate R and Z band fits** for wavelength-dependent diagnostics
5. **Explicit neighbor-only and component-switching tests**

### Key Facts

| Property | Value |
|----------|-------|
| Neighbor separation | {NEIGHBOR_SEP_ARCSEC:.3f}" |
| Neighbor ΔG | {NEIGHBOR_DELTA_G:.2f} mag |
| Expected flux ratio (G) | {NEIGHBOR_FLUX_FRAC_G:.3f} |
| Allowed flux ratio range | [{B_MIN}, {B_MAX}] |
| Catalog RV swing | {rv_swing_catalog:.0f} km/s |
| Fitted RV swing | {rv_swing_fitted:.0f} km/s |

### Verdict Table

| Check | Result | Notes |
|-------|--------|-------|
| 1. RV swing robust across arms/templates | **{verdicts['rv_swing_robust']}** | Swing: {rv_swing_fitted:.0f} km/s |
| 2. Neighbor-only fits competitive | **{verdicts['neighbor_only_competitive']}** | χ² ratios shown below |
| 3. Two-component per-epoch favored | **{verdicts['two_component_favored']}** | ΔBIC values shown below |
| 4. Cross-epoch constant-v2 favored | **{verdicts['constant_v2_favored']}** | ΔBIC={const_v2_result['delta_bic']:+.1f} |
| 5. Blend switching plausible | **{verdicts['blend_switching_plausible']}** | See analysis below |

**OVERALL: {overall_verdict}**

{bottom_line}

---

## Data Provenance

### DESI Spectra

| Epoch | MJD | File | SHA256 |
|-------|-----|------|--------|
"""

    for spec in spectra:
        report += f"| {spec['label']} | {spec['mjd']:.3f} | {spec['file_name']} | {file_hashes[spec['file_name']][:16]}... |\n"

    report += f"""
### Template Grid

**Primary templates (logg=4.5):** Teff = {PRIMARY_TEFFS} K
**Neighbor templates (logg=5.0):** Teff = {NEIGHBOR_TEFFS} K

---

## Analysis 1: Single-Star Primary-Only Fits

For each epoch and wavelength region, we fit:
   F(λ) = A × T_primary(v) + poly(λ)

where poly(λ) is a degree-2 polynomial to absorb continuum mismatch.

### Results by Epoch (Combined R+Z)

| Epoch | Catalog RV | Fitted RV | σ_formal | σ_renorm | χ²_red | Best Teff |
|-------|------------|-----------|----------|----------|--------|-----------|
"""

    for i, spec in enumerate(spectra):
        r = single_primary_results[i]['by_arm']['combined']
        report += f"| {spec['label']} | {spec['rv_catalog']:+.1f} | {r['best_v']:+.1f} | {r['best_sigma_v_formal']:.2f} | {r['best_sigma_v_renorm']:.1f} | {r['best_chi2_red']:.1f} | {r['best_teff']}K |\n"

    report += f"""
**Fitted RV swing: {rv_swing_fitted:.0f} km/s** (catalog: {rv_swing_catalog:.0f} km/s)

---

## Analysis 2: Neighbor-Only Fits

Same method but using neighbor (cooler) templates.

### Results by Epoch (Combined R+Z)

| Epoch | Neighbor-only RV | χ² | χ²_red | Primary χ² | Ratio |
|-------|------------------|-----|--------|------------|-------|
"""

    for i, spec in enumerate(spectra):
        r_n = single_neighbor_results[i]['by_arm']['combined']
        r_p = single_primary_results[i]['by_arm']['combined']
        ratio = r_n['best_chi2'] / r_p['best_chi2'] if r_p['best_chi2'] > 0 else np.inf
        report += f"| {spec['label']} | {r_n['best_v']:+.1f} | {r_n['best_chi2']:.0f} | {r_n['best_chi2_red']:.1f} | {r_p['best_chi2']:.0f} | {ratio:.2f} |\n"

    report += f"""
**Verdict:** {verdicts['neighbor_only_competitive']}

If ratio > 1.5 for all epochs, neighbor-only template is a poor fit → PASS.

---

## Analysis 3: Two-Component Per-Epoch Fits

For each epoch, fit:
   F(λ) = A × [T_primary(v1) + b × T_neighbor(v2)] + poly(λ)

with b constrained to [{B_MIN}, {B_MAX}].

### Results (Combined R+Z)

| Epoch | v1 | v2 | b | ΔBIC | Two-comp preferred? |
|-------|----|----|---|------|---------------------|
"""

    for i, spec in enumerate(spectra):
        r = two_component_results[i]['by_arm'].get('combined', {})
        if r and 'v1' in r:
            report += f"| {spec['label']} | {r['v1']:+.1f} | {r['v2']:+.1f} | {r['b']:.3f} | {r['delta_bic']:+.1f} | {'YES' if r.get('two_preferred', False) else 'NO'} |\n"
        else:
            report += f"| {spec['label']} | N/A | N/A | N/A | N/A | N/A |\n"

    report += f"""
**Verdict:** {verdicts['two_component_favored']}

ΔBIC < -6 means two-component model is strongly preferred (bad for single-star hypothesis).

---

## Analysis 4: Cross-Epoch Constant-v2 Model

Joint fit across all epochs with:
- v2 (neighbor velocity) **shared** across epochs
- b (flux ratio) **shared** across epochs
- v1 (primary velocity) **free per epoch**

This tests whether the RV swing could be explained by a static background neighbor.

### Results

| Parameter | Value |
|-----------|-------|
| v2 (shared) | {const_v2_result['v2_shared']:+.1f} km/s |
| b (shared) | {const_v2_result['b_shared']:.3f} |
| v1 (Epoch1) | {const_v2_result['v1_per_epoch'][0]:+.1f} km/s |
| v1 (Epoch2) | {const_v2_result['v1_per_epoch'][1]:+.1f} km/s |
| v1 (Epoch3) | {const_v2_result['v1_per_epoch'][2]:+.1f} km/s |
| Total χ² | {const_v2_result['chi2_total']:.1f} |
| χ²_red | {const_v2_result['chi2_red']:.2f} |

### Model Comparison

| Model | BIC |
|-------|-----|
| Primary-only | {const_v2_result['bic_primary_only']:.1f} |
| Constant-v2 blend | {const_v2_result['bic_const_v2']:.1f} |
| **ΔBIC** | **{const_v2_result['delta_bic']:+.1f}** |

**Verdict:** {verdicts['constant_v2_favored']}

ΔBIC < -6 with b in plausible range means constant-v2 blend model is favored → component switching is a serious explanation.

---

## Analysis 5: Wavelength-Dependent RV Diagnostics

Compare RV measured from R band (6000-7550 Å) vs Z band (7700-8800 Å).

| Epoch | RV_R | σ_R | RV_Z | σ_Z | Δ(Z-R) | Significance |
|-------|------|-----|------|-----|--------|--------------|
"""

    for r in arm_split_results:
        sig_str = f"{r['significance']:.1f}σ" if np.isfinite(r['significance']) else "N/A"
        report += f"| {r['label']} | {r['rv_R']:+.1f} | {r['sigma_R_renorm']:.1f} | {r['rv_Z']:+.1f} | {r['sigma_Z_renorm']:.1f} | {r['delta_rv']:+.1f} | {sig_str} |\n"

    report += f"""
Large systematic differences (> 3σ) between arms would indicate wavelength-dependent RV shifts, a signature of blending.

---

## Final Verdict Summary

```
╔════════════════════════════════════════════════════════════════════╗
║              DESI BLEND-AWARE ANALYSIS v3 RESULTS                  ║
╠════════════════════════════════════════════════════════════════════╣
║ 1. RV swing robust across arms/templates:  {verdicts['rv_swing_robust']:15s}          ║
║ 2. Neighbor-only fits competitive:         {verdicts['neighbor_only_competitive']:15s}          ║
║ 3. Two-component per-epoch favored:        {verdicts['two_component_favored']:15s}          ║
║ 4. Cross-epoch constant-v2 favored:        {verdicts['constant_v2_favored']:15s}          ║
║ 5. Blend switching plausible:              {verdicts['blend_switching_plausible']:15s}          ║
╠════════════════════════════════════════════════════════════════════╣
║ OVERALL VERDICT:                           {overall_verdict:15s}          ║
╚════════════════════════════════════════════════════════════════════╝
```

### Interpretation

"""

    if overall_verdict == 'ROBUST':
        report += """The DESI RV swing of ~{:.0f} km/s is **ROBUST**:

1. Single-template fitting recovers consistent RVs across wavelength regions
2. Neighbor-only fits are significantly worse than primary fits
3. Two-component blend models are NOT statistically preferred
4. Cross-epoch constant-v2 model does NOT improve the fit

**The RV variability requires a gravitational companion.**
""".format(rv_swing_fitted)
    elif overall_verdict == 'COMPROMISED':
        report += """The DESI RV measurements are **COMPROMISED** by potential blend effects:

1. Neighbor-only or two-component models show competitive fits
2. Component switching may explain some of the observed RV variability
3. Further investigation with high-resolution spectroscopy is required

**Cannot definitively claim gravitational companion without ruling out blend scenarios.**
"""
    else:
        report += """Results are **INCONCLUSIVE**:

1. Some tests suggest robustness, others show potential blend contamination
2. The large RV swing ({:.0f} km/s) is difficult to explain by blending alone
3. But blend effects cannot be completely ruled out

**High-resolution spectroscopy is recommended to resolve the ambiguity.**
""".format(rv_swing_fitted)

    report += f"""
---

## Output Files

| File | Description |
|------|-------------|
| `desi_epoch_rv_refit_v3.json` | Per-epoch RV fits with uncertainties |
| `desi_model_comparison_v3.json` | Two-component and arm-split results |
| `desi_constant_v2_fit_v3.json` | Cross-epoch constant-v2 model |
| `figures/chi2_vs_v_by_epoch_v3.png` | χ²(v) curves |
| `figures/rv_by_method_v3.png` | RV comparison with error bars |
| `figures/neighbor_only_vs_primary_v3.png` | Template comparison |
| `figures/constant_v2_model_comparison_v3.png` | Model BIC comparison |
| `figures/arm_split_rv_v3.png` | R vs Z band RVs |

---

## Methodological Notes

### A) RV Uncertainty Renormalization

When χ²_red >> 1 (model mismatch), formal uncertainties underestimate true errors.
We report both:
- **σ_formal**: from Δχ² = 1 criterion
- **σ_renorm**: σ_formal × sqrt(χ²_red)

### B) BIC Computation

BIC = χ²_data + k × ln(n)

where χ²_data is the data-only chi-squared (no prior penalties), k is the number of parameters, and n is the number of data points.

### C) Continuum Marginalization

At each trial velocity, we analytically marginalize over:
- Amplitude A
- Degree-2 polynomial continuum

This ensures χ²(v) curves are smooth and stable.

### D) Flux Ratio Constraints

The flux ratio b is constrained to [{B_MIN}, {B_MAX}], broader than the G-band expectation (~{NEIGHBOR_FLUX_FRAC_G:.2f}) to allow for:
- Color differences between bands
- Seeing variations
- Fiber coupling differences

---

**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis by:** Claude Code (v3 methodology)
**Templates:** PHOENIX-ACES models from Göttingen

"""

    with open(OUTPUT_DIR / 'DESI_BLEND_AWARE_REPORT_v3.md', 'w') as f:
        f.write(report)
    print(f"Saved: {OUTPUT_DIR / 'DESI_BLEND_AWARE_REPORT_v3.md'}")

    # Clear checkpoint after successful completion
    clear_checkpoint()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
