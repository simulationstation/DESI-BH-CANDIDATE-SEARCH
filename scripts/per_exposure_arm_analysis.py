#!/usr/bin/env python3
"""
Per-exposure arm-split RV analysis for DESI target 39627745210139276.

Implements GPT's recommendations:
1. Per-exposure, per-camera arm-split check
2. Renormalize uncertainties by χ²_red
3. Δχ²=1 uncertainty estimation (not constant σ)

Author: Claude
"""

import os
import json
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

# Exposure metadata extracted from coadd files
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

PER_EXP_DIR = "data/per_exposure"
TEMPLATE_DIR = "data/templates"


# =============================================================================
# Template Loading
# =============================================================================

def load_phoenix_template(teff=3800):
    """Load PHOENIX template for M-dwarf."""
    wave_file = os.path.join(TEMPLATE_DIR, "WAVE_PHOENIX.fits")

    # Find best matching template
    template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.startswith("phoenix") and f.endswith(".fits")]

    best_match = None
    best_diff = float('inf')
    for tf in template_files:
        if "WAVE" in tf:
            continue
        # Parse Teff from filename
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

    print(f"Loading PHOENIX template: {best_match}")

    with fits.open(wave_file) as hdul:
        wave = hdul[0].data
    with fits.open(os.path.join(TEMPLATE_DIR, best_match)) as hdul:
        flux = hdul[0].data

    return wave, flux


# =============================================================================
# RV Fitting with Proper Uncertainty Estimation
# =============================================================================

def compute_chi2_vs_rv(wave_obs, flux_obs, ivar_obs, wave_template, flux_template,
                        rv_grid, continuum_order=3):
    """
    Compute χ² as a function of RV.

    Returns:
        rv_grid: array of RV values
        chi2_arr: χ² at each RV
        dof: degrees of freedom
    """
    # Good pixels
    good = (ivar_obs > 0) & np.isfinite(flux_obs) & np.isfinite(ivar_obs)
    n_good = good.sum()

    if n_good < 100:
        return rv_grid, np.full_like(rv_grid, np.nan), 0

    # Interpolate template
    template_interp = interp1d(wave_template, flux_template, kind='linear',
                                bounds_error=False, fill_value=np.nan)

    chi2_arr = np.zeros(len(rv_grid))

    for i, rv in enumerate(rv_grid):
        # Doppler shift template
        doppler = 1 + rv / C_KMS
        wave_shifted = wave_obs / doppler
        template_shifted = template_interp(wave_shifted)

        # Skip if too many NaNs
        valid = good & np.isfinite(template_shifted) & (template_shifted > 0)
        if valid.sum() < 100:
            chi2_arr[i] = np.nan
            continue

        # Fit continuum polynomial to flux/template ratio
        x = wave_obs[valid]
        x_norm = (x - x.mean()) / x.std()
        y = flux_obs[valid] / template_shifted[valid]
        w = ivar_obs[valid] * template_shifted[valid]**2

        # Polynomial fit
        try:
            coeffs = np.polyfit(x_norm, y, continuum_order, w=np.sqrt(w))
            continuum = np.polyval(coeffs, x_norm)
        except:
            chi2_arr[i] = np.nan
            continue

        # Model = template * continuum
        model = template_shifted[valid] * continuum

        # χ²
        residuals = flux_obs[valid] - model
        chi2 = np.sum(residuals**2 * ivar_obs[valid])
        chi2_arr[i] = chi2

    # Degrees of freedom: n_pixels - n_params (RV + continuum coeffs)
    dof = n_good - (1 + continuum_order + 1)

    return rv_grid, chi2_arr, dof


def fit_rv_with_uncertainties(wave_obs, flux_obs, ivar_obs, wave_template, flux_template,
                               rv_init=0, rv_range=(-300, 300), rv_step=1):
    """
    Fit RV and compute proper uncertainties via Δχ²=1.

    Returns:
        rv_best: best-fit RV
        rv_err_raw: raw uncertainty from Δχ²=1
        rv_err_scaled: uncertainty scaled by sqrt(χ²_red)
        chi2_min: minimum χ²
        chi2_red: reduced χ² at minimum
        rv_grid: RV grid
        chi2_arr: χ² vs RV
    """
    # Coarse grid search
    rv_grid = np.arange(rv_range[0], rv_range[1] + rv_step, rv_step)
    rv_grid_coarse, chi2_coarse, dof = compute_chi2_vs_rv(
        wave_obs, flux_obs, ivar_obs, wave_template, flux_template, rv_grid
    )

    if np.all(np.isnan(chi2_coarse)):
        return np.nan, np.nan, np.nan, np.nan, np.nan, rv_grid, chi2_coarse

    # Find minimum
    valid = np.isfinite(chi2_coarse)
    idx_min = np.nanargmin(chi2_coarse)
    rv_coarse = rv_grid_coarse[idx_min]
    chi2_min_coarse = chi2_coarse[idx_min]

    # Fine grid around minimum
    rv_fine = np.linspace(rv_coarse - 5, rv_coarse + 5, 101)
    rv_grid_fine, chi2_fine, dof = compute_chi2_vs_rv(
        wave_obs, flux_obs, ivar_obs, wave_template, flux_template, rv_fine
    )

    # Find best minimum
    valid_fine = np.isfinite(chi2_fine)
    if valid_fine.sum() < 3:
        return rv_coarse, np.nan, np.nan, chi2_min_coarse, chi2_min_coarse/dof if dof > 0 else np.nan, rv_grid_coarse, chi2_coarse

    idx_min = np.nanargmin(chi2_fine)
    rv_best = rv_fine[idx_min]
    chi2_min = chi2_fine[idx_min]
    chi2_red = chi2_min / dof if dof > 0 else np.nan

    # Compute Δχ²=1 uncertainty (raw)
    # Interpolate χ² curve
    chi2_interp = interp1d(rv_fine[valid_fine], chi2_fine[valid_fine],
                            kind='quadratic', bounds_error=False, fill_value=np.inf)

    # Find RV where χ² = χ²_min + 1
    target_chi2 = chi2_min + 1.0

    try:
        # Search for lower bound
        rv_search_lo = rv_fine[valid_fine][rv_fine[valid_fine] < rv_best]
        if len(rv_search_lo) > 0:
            rv_lo = brentq(lambda x: chi2_interp(x) - target_chi2,
                           rv_search_lo.min(), rv_best)
        else:
            rv_lo = rv_best - 10

        # Search for upper bound
        rv_search_hi = rv_fine[valid_fine][rv_fine[valid_fine] > rv_best]
        if len(rv_search_hi) > 0:
            rv_hi = brentq(lambda x: chi2_interp(x) - target_chi2,
                           rv_best, rv_search_hi.max())
        else:
            rv_hi = rv_best + 10

        rv_err_raw = (rv_hi - rv_lo) / 2
    except:
        # Fallback: parabolic approximation
        # χ² ≈ χ²_min + (rv - rv_best)² / (2σ²)
        # So σ² = 1 / (d²χ²/drv²)
        try:
            # Fit parabola near minimum
            mask = np.abs(rv_fine - rv_best) < 3
            if mask.sum() >= 3:
                coeffs = np.polyfit(rv_fine[mask], chi2_fine[mask], 2)
                curvature = 2 * coeffs[0]  # d²χ²/drv²
                if curvature > 0:
                    rv_err_raw = np.sqrt(1 / curvature)
                else:
                    rv_err_raw = np.nan
            else:
                rv_err_raw = np.nan
        except:
            rv_err_raw = np.nan

    # Scale uncertainty by sqrt(χ²_red) if χ²_red > 1
    if np.isfinite(chi2_red) and chi2_red > 1:
        rv_err_scaled = rv_err_raw * np.sqrt(chi2_red)
    else:
        rv_err_scaled = rv_err_raw

    return rv_best, rv_err_raw, rv_err_scaled, chi2_min, chi2_red, rv_grid_coarse, chi2_coarse


# =============================================================================
# Per-Exposure Analysis
# =============================================================================

def analyze_exposure(exp_info, wave_template, flux_template):
    """
    Analyze a single exposure for R and Z arm RVs.
    """
    expid = exp_info['expid']
    petal = exp_info['petal']
    row_idx = exp_info['row_idx']

    results = {
        'expid': expid,
        'mjd': exp_info['mjd'],
        'catalog_rv': exp_info['catalog_rv'],
        'catalog_err': exp_info['catalog_err'],
    }

    for arm in ['r', 'z']:
        camera = f"{arm}{petal}"
        filename = f"cframe-{camera}-{expid:08d}.fits"
        filepath = os.path.join(PER_EXP_DIR, filename)

        if not os.path.exists(filepath):
            print(f"  WARNING: {filename} not found")
            results[f'{arm}_rv'] = np.nan
            results[f'{arm}_err_raw'] = np.nan
            results[f'{arm}_err_scaled'] = np.nan
            results[f'{arm}_chi2_red'] = np.nan
            continue

        # Load data
        with fits.open(filepath) as hdul:
            wave = hdul['WAVELENGTH'].data
            flux = hdul['FLUX'].data[row_idx]
            ivar = hdul['IVAR'].data[row_idx]

        # Fit RV
        rv_best, rv_err_raw, rv_err_scaled, chi2_min, chi2_red, rv_grid, chi2_arr = fit_rv_with_uncertainties(
            wave, flux, ivar, wave_template, flux_template,
            rv_init=exp_info['catalog_rv']
        )

        results[f'{arm}_rv'] = rv_best
        results[f'{arm}_err_raw'] = rv_err_raw
        results[f'{arm}_err_scaled'] = rv_err_scaled
        results[f'{arm}_chi2_min'] = chi2_min
        results[f'{arm}_chi2_red'] = chi2_red
        results[f'{arm}_rv_grid'] = rv_grid
        results[f'{arm}_chi2_arr'] = chi2_arr

        print(f"    {arm.upper()}-arm: RV = {rv_best:+7.1f} km/s, "
              f"σ_raw = {rv_err_raw:.1f}, σ_scaled = {rv_err_scaled:.1f}, χ²_red = {chi2_red:.1f}")

    # Compute arm split
    if np.isfinite(results.get('r_rv', np.nan)) and np.isfinite(results.get('z_rv', np.nan)):
        delta_rz = results['r_rv'] - results['z_rv']

        # Combined error (use scaled errors)
        err_r = results.get('r_err_scaled', np.nan)
        err_z = results.get('z_err_scaled', np.nan)
        if np.isfinite(err_r) and np.isfinite(err_z):
            delta_err = np.sqrt(err_r**2 + err_z**2)
            delta_sigma = abs(delta_rz) / delta_err if delta_err > 0 else np.nan
        else:
            delta_err = np.nan
            delta_sigma = np.nan

        results['delta_rz'] = delta_rz
        results['delta_rz_err'] = delta_err
        results['delta_rz_sigma'] = delta_sigma

        print(f"    Δ(R-Z) = {delta_rz:+7.1f} ± {delta_err:.1f} km/s ({delta_sigma:.1f}σ)")

    return results


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("=" * 70)
    print("Per-Exposure Arm-Split RV Analysis")
    print("Target: DESI TARGETID", TARGET_ID)
    print("=" * 70)

    # Load template
    print("\nLoading PHOENIX template...")
    try:
        wave_template, flux_template = load_phoenix_template(teff=3800)
        print(f"  Template wave range: {wave_template.min():.0f} - {wave_template.max():.0f} Å")
    except Exception as e:
        print(f"ERROR loading template: {e}")
        return

    # Analyze each exposure
    print("\n" + "-" * 70)
    print("Per-Exposure Analysis")
    print("-" * 70)

    all_results = []

    for exp in EXPOSURES:
        print(f"\nEXPID {exp['expid']} (MJD {exp['mjd']:.3f}, Catalog RV = {exp['catalog_rv']:+.1f} km/s):")
        results = analyze_exposure(exp, wave_template, flux_template)
        all_results.append(results)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print(f"\n{'EXPID':<10} {'MJD':<12} {'Cat RV':<10} {'R-arm RV':<12} {'Z-arm RV':<12} {'Δ(R-Z)':<12} {'Signif':<8}")
    print("-" * 80)

    for r in all_results:
        cat_rv = f"{r['catalog_rv']:+.1f}"
        r_rv = f"{r.get('r_rv', np.nan):+.1f} ± {r.get('r_err_scaled', np.nan):.1f}" if np.isfinite(r.get('r_rv', np.nan)) else "N/A"
        z_rv = f"{r.get('z_rv', np.nan):+.1f} ± {r.get('z_err_scaled', np.nan):.1f}" if np.isfinite(r.get('z_rv', np.nan)) else "N/A"
        delta = f"{r.get('delta_rz', np.nan):+.1f}" if np.isfinite(r.get('delta_rz', np.nan)) else "N/A"
        sig = f"{r.get('delta_rz_sigma', np.nan):.1f}σ" if np.isfinite(r.get('delta_rz_sigma', np.nan)) else "N/A"

        print(f"{r['expid']:<10} {r['mjd']:<12.3f} {cat_rv:<10} {r_rv:<12} {z_rv:<12} {delta:<12} {sig:<8}")

    # Check χ²_red values
    print("\n" + "-" * 70)
    print("χ²_red Values (ideally ~1.0; if >> 1, errors are underestimated)")
    print("-" * 70)

    for r in all_results:
        print(f"EXPID {r['expid']}: R-arm χ²_red = {r.get('r_chi2_red', np.nan):.1f}, "
              f"Z-arm χ²_red = {r.get('z_chi2_red', np.nan):.1f}")

    # Key finding: compare EXPID 120449 and 120450 (same-night pair)
    print("\n" + "-" * 70)
    print("Same-Night Consistency Check (EXPIDs 120449 & 120450)")
    print("-" * 70)

    exp449 = next((r for r in all_results if r['expid'] == 120449), None)
    exp450 = next((r for r in all_results if r['expid'] == 120450), None)

    if exp449 and exp450:
        for arm in ['r', 'z']:
            rv1 = exp449.get(f'{arm}_rv', np.nan)
            rv2 = exp450.get(f'{arm}_rv', np.nan)
            err1 = exp449.get(f'{arm}_err_scaled', np.nan)
            err2 = exp450.get(f'{arm}_err_scaled', np.nan)

            if np.isfinite(rv1) and np.isfinite(rv2):
                delta = rv1 - rv2
                delta_err = np.sqrt(err1**2 + err2**2) if np.isfinite(err1) and np.isfinite(err2) else np.nan
                delta_sig = abs(delta) / delta_err if delta_err > 0 else np.nan

                print(f"  {arm.upper()}-arm: {rv1:+.1f} vs {rv2:+.1f} = Δ = {delta:+.1f} ± {delta_err:.1f} km/s ({delta_sig:.1f}σ)")

    # Create diagnostic plot
    print("\n" + "-" * 70)
    print("Generating diagnostic plot...")
    print("-" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {114768: 'red', 120194: 'blue', 120449: 'green', 120450: 'purple'}

    # Panel 1: R-arm χ² curves
    ax = axes[0, 0]
    for r in all_results:
        if 'r_rv_grid' in r and 'r_chi2_arr' in r:
            rv_grid = r['r_rv_grid']
            chi2 = r['r_chi2_arr']
            if chi2 is not None and np.any(np.isfinite(chi2)):
                # Normalize to minimum
                chi2_norm = chi2 - np.nanmin(chi2)
                ax.plot(rv_grid, chi2_norm, color=colors[r['expid']],
                        label=f"EXPID {r['expid']}", alpha=0.8)
                ax.axvline(r['r_rv'], color=colors[r['expid']], linestyle='--', alpha=0.5)

    ax.axhline(1.0, color='black', linestyle=':', label='Δχ²=1')
    ax.set_xlabel('RV [km/s]')
    ax.set_ylabel('Δχ² (from minimum)')
    ax.set_title('R-arm χ² vs RV')
    ax.set_ylim(0, 50)
    ax.set_xlim(-200, 150)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Z-arm χ² curves
    ax = axes[0, 1]
    for r in all_results:
        if 'z_rv_grid' in r and 'z_chi2_arr' in r:
            rv_grid = r['z_rv_grid']
            chi2 = r['z_chi2_arr']
            if chi2 is not None and np.any(np.isfinite(chi2)):
                chi2_norm = chi2 - np.nanmin(chi2)
                ax.plot(rv_grid, chi2_norm, color=colors[r['expid']],
                        label=f"EXPID {r['expid']}", alpha=0.8)
                ax.axvline(r['z_rv'], color=colors[r['expid']], linestyle='--', alpha=0.5)

    ax.axhline(1.0, color='black', linestyle=':', label='Δχ²=1')
    ax.set_xlabel('RV [km/s]')
    ax.set_ylabel('Δχ² (from minimum)')
    ax.set_title('Z-arm χ² vs RV')
    ax.set_ylim(0, 50)
    ax.set_xlim(-200, 150)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: R vs Z arm RV comparison
    ax = axes[1, 0]
    for r in all_results:
        r_rv = r.get('r_rv', np.nan)
        z_rv = r.get('z_rv', np.nan)
        r_err = r.get('r_err_scaled', np.nan)
        z_err = r.get('z_err_scaled', np.nan)

        if np.isfinite(r_rv) and np.isfinite(z_rv):
            ax.errorbar(r_rv, z_rv, xerr=r_err if np.isfinite(r_err) else None,
                        yerr=z_err if np.isfinite(z_err) else None,
                        fmt='o', color=colors[r['expid']], markersize=10,
                        label=f"EXPID {r['expid']}")

    # 1:1 line
    lims = [-150, 100]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1')
    ax.set_xlabel('R-arm RV [km/s]')
    ax.set_ylabel('Z-arm RV [km/s]')
    ax.set_title('R-arm vs Z-arm RV')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Panel 4: Arm split by exposure
    ax = axes[1, 1]
    expids = [r['expid'] for r in all_results]
    deltas = [r.get('delta_rz', np.nan) for r in all_results]
    delta_errs = [r.get('delta_rz_err', np.nan) for r in all_results]

    x = np.arange(len(expids))
    ax.bar(x, deltas, color=[colors[e] for e in expids], alpha=0.7)
    ax.errorbar(x, deltas, yerr=delta_errs, fmt='none', color='black', capsize=5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in expids])
    ax.set_xlabel('EXPID')
    ax.set_ylabel('Δ(R-Z) [km/s]')
    ax.set_title('Arm Split by Exposure (with scaled uncertainties)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('per_exposure_arm_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: per_exposure_arm_analysis.png")
    plt.close()

    # Save results to JSON
    results_for_json = []
    for r in all_results:
        r_clean = {k: v for k, v in r.items()
                   if not isinstance(v, np.ndarray) and not k.endswith('_grid') and not k.endswith('_arr')}
        # Convert numpy types
        for k, v in r_clean.items():
            if isinstance(v, (np.floating, np.integer)):
                r_clean[k] = float(v) if np.isfinite(v) else None
        results_for_json.append(r_clean)

    with open('per_exposure_arm_analysis.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print("Saved: per_exposure_arm_analysis.json")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
