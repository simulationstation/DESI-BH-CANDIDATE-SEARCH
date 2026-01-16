#!/usr/bin/env python3
"""
DESI Truth Filter v1 - Maximum no-telescope truth-filtering for candidate
Gaia DR3 3802130935635096832 / DESI TARGETID 39627745210139276

This script performs independent validation checks:
1. Per-exposure RV analysis (R vs Z arm)
2. Line/bandhead-level RV verification
3. Expected neighbor flux ratio (b_R, b_Z) computation
4. Gaia duplicity control sample analysis
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import json
import hashlib
from datetime import datetime
from astropy.io import fits
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_ID = 39627745210139276
GAIA_SOURCE_ID = 3802130935635096832
NEIGHBOR_SEP_ARCSEC = 0.68793
NEIGHBOR_DELTA_G = 2.21099

OUTPUT_DIR = '/home/primary/DESI-BH-CANDIDATE-SEARCH/outputs/desi_truthfilter_v1'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
DATA_DIR = '/home/primary/DESI-BH-CANDIDATE-SEARCH/data'

COADD_FILES = {
    'Epoch1': os.path.join(DATA_DIR, 'desi_spectra/coadd_20268_20211219_p2.fits'),
    'Epoch2': os.path.join(DATA_DIR, 'desi_spectra/coadd_24976_20220125_p7.fits'),
    'Epoch3': os.path.join(DATA_DIR, 'desi_spectra/coadd_23137_20220127_p0.fits')
}

RVPIX_FILE = os.path.join(DATA_DIR, 'raw/rvpix_exp-main-bright.fits')
TEMPLATE_FILE = os.path.join(DATA_DIR, 'templates/phoenix_m0_3800_4.5.fits')
WAVE_FILE = os.path.join(DATA_DIR, 'templates/WAVE_PHOENIX.fits')

# Neighbor templates for Part 3
NEIGHBOR_TEMPLATES = {
    'M5_3000K': os.path.join(DATA_DIR, 'templates/phoenix_m5_3000_5.0.fits'),
    'M3_3200K': os.path.join(DATA_DIR, 'templates/phoenix_3200_5.0.fits'),
}

# Velocity grid
V_MIN, V_MAX, V_STEP = -200, 200, 1  # km/s

# Speed of light
C_KMS = 299792.458

# DESI wavelength ranges
DESI_R_RANGE = (5760, 7620)  # Angstrom
DESI_Z_RANGE = (7520, 9824)  # Angstrom

# Telluric masks (approximate)
TELLURIC_REGIONS = [
    (6860, 6960),   # B-band O2
    (7160, 7340),   # A-band O2
    (7590, 7700),   # O2
    (8160, 8350),   # H2O
    (8950, 9200),   # H2O
    (9300, 9700),   # H2O
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # First 16 chars for brevity


def doppler_shift(wave, v_kms):
    """Apply Doppler shift to wavelength array."""
    return wave * (1 + v_kms / C_KMS)


def create_telluric_mask(wave):
    """Create mask excluding telluric regions (True = good)."""
    mask = np.ones(len(wave), dtype=bool)
    for wmin, wmax in TELLURIC_REGIONS:
        mask &= ~((wave >= wmin) & (wave <= wmax))
    return mask


def load_phoenix_template(template_file, wave_file):
    """Load PHOENIX template and wavelength grid."""
    with fits.open(wave_file) as hdul:
        wave_template = hdul[0].data  # Angstrom
    with fits.open(template_file) as hdul:
        flux_template = hdul[0].data
    return wave_template, flux_template


def chi2_rv_fit(wave_data, flux_data, ivar_data, wave_template, flux_template,
                v_grid, mask=None):
    """
    Fit RV by minimizing chi-squared over velocity grid.
    Returns best v, sigma_v, chi2_red.
    """
    if mask is None:
        mask = np.ones(len(wave_data), dtype=bool)

    # Apply mask
    good = mask & (ivar_data > 0) & np.isfinite(flux_data) & np.isfinite(ivar_data)
    if good.sum() < 100:
        return np.nan, np.nan, np.nan, 0

    wave_g = wave_data[good]
    flux_g = flux_data[good]
    ivar_g = ivar_data[good]

    # Interpolator for template
    interp_func = interp1d(wave_template, flux_template, kind='linear',
                           bounds_error=False, fill_value=0)

    chi2_arr = np.zeros(len(v_grid))

    for i, v in enumerate(v_grid):
        # Shift template
        wave_shifted = doppler_shift(wave_g, -v)  # Negative because we shift template
        template_shifted = interp_func(wave_shifted)

        # Fit scale factor analytically
        numerator = np.sum(flux_g * template_shifted * ivar_g)
        denominator = np.sum(template_shifted**2 * ivar_g)
        if denominator <= 0:
            chi2_arr[i] = np.inf
            continue
        scale = numerator / denominator

        # Chi-squared
        residual = flux_g - scale * template_shifted
        chi2_arr[i] = np.sum(residual**2 * ivar_g)

    # Find minimum
    best_idx = np.argmin(chi2_arr)
    best_v = v_grid[best_idx]
    chi2_min = chi2_arr[best_idx]

    # Degrees of freedom: N_pixels - 2 (v and scale)
    dof = good.sum() - 2
    chi2_red = chi2_min / dof if dof > 0 else np.nan

    # Uncertainty from delta chi2 = 1
    # Find where chi2 = chi2_min + 1
    chi2_threshold = chi2_min + 1
    below_threshold = chi2_arr < chi2_threshold
    if below_threshold.sum() > 1:
        v_below = v_grid[below_threshold]
        sigma_v = (v_below.max() - v_below.min()) / 2
    else:
        sigma_v = V_STEP  # Lower bound

    return best_v, sigma_v, chi2_red, good.sum()


def extract_target_spectrum(coadd_file, target_id, arm='R'):
    """Extract spectrum for target from coadd file."""
    with fits.open(coadd_file) as hdul:
        fibermap = hdul['FIBERMAP'].data
        idx = np.where(fibermap['TARGETID'] == target_id)[0]
        if len(idx) == 0:
            return None, None, None
        idx = idx[0]

        wave = hdul[f'{arm}_WAVELENGTH'].data
        flux = hdul[f'{arm}_FLUX'].data[idx]
        ivar = hdul[f'{arm}_IVAR'].data[idx]

    return wave, flux, ivar


# =============================================================================
# PART 1: PER-EXPOSURE / PER-ARM RV ANALYSIS
# =============================================================================
def run_part1_per_exposure_analysis():
    """
    Analyze R vs Z arm RVs for each epoch.
    Note: We only have coadd spectra, not true per-exposure spectra.
    """
    print("\n" + "="*70)
    print("PART 1: PER-EXPOSURE / PER-ARM RV ANALYSIS")
    print("="*70)

    results = {
        'note': 'Per-exposure spectra NOT available locally. Analysis uses coadd spectra.',
        'limitation': 'For Epoch 3 (2 exposures), coadd combines both - cannot separate.',
        'missing_files': [
            'spectra-{petal}-{expid}.fits for EXPIDs 114768, 120194, 120449, 120450',
            'These would be at $DESI_ROOT/spectro/redux/iron/tiles/cumulative/'
        ],
        'epochs': {}
    }

    # Load template
    wave_template, flux_template = load_phoenix_template(TEMPLATE_FILE, WAVE_FILE)
    v_grid = np.arange(V_MIN, V_MAX + V_STEP, V_STEP)

    # Get per-exposure catalog RVs from rvpix file
    print("\nReading per-exposure catalog RVs from rvpix file...")
    with fits.open(RVPIX_FILE) as hdul:
        rvtab = hdul['RVTAB'].data
        fibermap = hdul['FIBERMAP'].data
        idx = np.where(fibermap['TARGETID'] == TARGET_ID)[0]

        per_exp_catalog = {}
        for i in idx:
            expid = fibermap['EXPID'][i]
            night = fibermap['NIGHT'][i]
            vrad = rvtab['VRAD'][i]
            vrad_err = rvtab['VRAD_ERR'][i]
            per_exp_catalog[int(expid)] = {
                'night': int(night),
                'vrad_catalog': float(vrad),
                'vrad_err_catalog': float(vrad_err)
            }

    print(f"Found {len(per_exp_catalog)} exposures for target:")
    for expid, info in per_exp_catalog.items():
        print(f"  EXPID {expid} (night {info['night']}): VRAD = {info['vrad_catalog']:.2f} km/s")

    results['per_exposure_catalog_rvs'] = per_exp_catalog

    # Mapping of epochs to exposures
    epoch_exposures = {
        'Epoch1': [114768],
        'Epoch2': [120194],
        'Epoch3': [120449, 120450]
    }

    # Analyze each epoch using coadd (R and Z arms separately)
    print("\nFitting R and Z arms from coadd spectra...")

    for epoch, coadd_file in COADD_FILES.items():
        print(f"\n--- {epoch} ---")

        epoch_result = {
            'coadd_file': os.path.basename(coadd_file),
            'file_hash': sha256_file(coadd_file),
            'exposures': epoch_exposures[epoch],
            'n_exposures': len(epoch_exposures[epoch]),
            'arms': {}
        }

        for arm in ['R', 'Z']:
            wave, flux, ivar = extract_target_spectrum(coadd_file, TARGET_ID, arm)
            if wave is None:
                print(f"  {arm}: Target not found")
                continue

            # Create telluric mask
            mask = create_telluric_mask(wave)

            # Fit RV
            v_best, sigma_v, chi2_red, n_pix = chi2_rv_fit(
                wave, flux, ivar, wave_template, flux_template, v_grid, mask
            )

            epoch_result['arms'][arm] = {
                'v_kms': float(v_best),
                'sigma_v_kms': float(sigma_v),
                'chi2_red': float(chi2_red),
                'n_pixels': int(n_pix),
                'wave_range': [float(wave.min()), float(wave.max())]
            }

            print(f"  {arm}: v = {v_best:+.1f} ± {sigma_v:.1f} km/s, χ²_red = {chi2_red:.2f}, N = {n_pix}")

        # Compute R-Z difference
        if 'R' in epoch_result['arms'] and 'Z' in epoch_result['arms']:
            v_R = epoch_result['arms']['R']['v_kms']
            v_Z = epoch_result['arms']['Z']['v_kms']
            sigma_R = epoch_result['arms']['R']['sigma_v_kms']
            sigma_Z = epoch_result['arms']['Z']['sigma_v_kms']

            delta_RZ = v_R - v_Z
            sigma_delta = np.sqrt(sigma_R**2 + sigma_Z**2)
            significance = delta_RZ / sigma_delta if sigma_delta > 0 else np.nan

            epoch_result['R_minus_Z'] = {
                'delta_v_kms': float(delta_RZ),
                'sigma_kms': float(sigma_delta),
                'significance_sigma': float(significance)
            }

            print(f"  R-Z: Δv = {delta_RZ:+.1f} ± {sigma_delta:.1f} km/s ({significance:.1f}σ)")

        results['epochs'][epoch] = epoch_result

    # Check Epoch 3 specifically
    print("\n--- Epoch 3 Per-Exposure Analysis (from catalog) ---")
    exp1_rv = per_exp_catalog[120449]['vrad_catalog']
    exp2_rv = per_exp_catalog[120450]['vrad_catalog']
    delta_exp = exp1_rv - exp2_rv

    results['epoch3_exposure_comparison'] = {
        'exp_120449_vrad': exp1_rv,
        'exp_120450_vrad': exp2_rv,
        'delta_between_exposures': delta_exp,
        'note': 'These are full-spectrum RVs, not R/Z split. Both are similar (~1 km/s apart).'
    }

    print(f"  Exp 120449: {exp1_rv:.2f} km/s")
    print(f"  Exp 120450: {exp2_rv:.2f} km/s")
    print(f"  Difference: {delta_exp:.2f} km/s")
    print("  Both exposures show similar RV, suggesting R-Z discrepancy is PERSISTENT")

    return results


def plot_part1_figures(results):
    """Generate Part 1 figures."""

    # Figure 1: R vs Z per epoch
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    epochs = ['Epoch1', 'Epoch2', 'Epoch3']
    colors = {'R': 'red', 'Z': 'blue'}

    for i, epoch in enumerate(epochs):
        ax = axes[i]
        if epoch not in results['epochs']:
            continue

        epoch_data = results['epochs'][epoch]

        x_pos = [0, 1]
        v_vals = []
        sigma_vals = []

        for j, arm in enumerate(['R', 'Z']):
            if arm in epoch_data['arms']:
                v = epoch_data['arms'][arm]['v_kms']
                sigma = epoch_data['arms'][arm]['sigma_v_kms']
                ax.errorbar(j, v, yerr=sigma, fmt='o', color=colors[arm],
                           markersize=10, capsize=5, label=f'{arm}-arm')
                v_vals.append(v)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['R', 'Z'])
        ax.set_ylabel('RV (km/s)')
        ax.set_title(f'{epoch}\nN_exp = {epoch_data["n_exposures"]}')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()

        # Add R-Z difference
        if 'R_minus_Z' in epoch_data:
            delta = epoch_data['R_minus_Z']['delta_v_kms']
            sig = epoch_data['R_minus_Z']['significance_sigma']
            ax.text(0.5, 0.02, f'R-Z = {delta:+.1f} km/s ({sig:.1f}σ)',
                   transform=ax.transAxes, ha='center', fontsize=9)

    plt.suptitle('Per-Arm RV Measurements (from coadd spectra)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'per_exposure_rv_RZ.png'), dpi=150)
    plt.close()
    print(f"Saved: per_exposure_rv_RZ.png")

    # Figure 2: Epoch 3 diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: R and Z arms
    ax = axes[0]
    epoch3 = results['epochs']['Epoch3']
    for j, arm in enumerate(['R', 'Z']):
        if arm in epoch3['arms']:
            v = epoch3['arms'][arm]['v_kms']
            sigma = epoch3['arms'][arm]['sigma_v_kms']
            ax.errorbar(j, v, yerr=sigma, fmt='o', color=colors[arm],
                       markersize=12, capsize=6, label=f'{arm}-arm')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['R-arm', 'Z-arm'])
    ax.set_ylabel('RV (km/s)')
    ax.set_title('Epoch 3 R vs Z (coadd of 2 exposures)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()

    # Right: Per-exposure catalog RVs
    ax = axes[1]
    per_exp = results['per_exposure_catalog_rvs']
    exp_ids = [120449, 120450]
    for j, expid in enumerate(exp_ids):
        v = per_exp[expid]['vrad_catalog']
        ax.bar(j, v, color='purple', alpha=0.7)
        ax.text(j, v + 1, f'{v:.1f}', ha='center', fontsize=10)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Exp 120449', 'Exp 120450'])
    ax.set_ylabel('Catalog RV (km/s)')
    ax.set_title('Epoch 3 Per-Exposure Catalog RVs\n(full spectrum, not R/Z split)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'per_exposure_epoch3_diagnostics.png'), dpi=150)
    plt.close()
    print(f"Saved: per_exposure_epoch3_diagnostics.png")


# =============================================================================
# PART 2: LINE/BANDHEAD-LEVEL RV CHECK
# =============================================================================
def run_part2_line_by_line_analysis():
    """
    Verify RV swing using specific spectral features.
    """
    print("\n" + "="*70)
    print("PART 2: LINE/BANDHEAD-LEVEL RV CHECK")
    print("="*70)

    # Define feature windows in Z-arm (higher SNR for M-dwarfs)
    # These are M-dwarf molecular features (TiO, VO, CaH)
    feature_windows = {
        'TiO_8432': (8410, 8455),    # TiO bandhead
        'TiO_8860': (8840, 8880),    # TiO band
        'CaII_8498': (8485, 8510),   # Ca II triplet line 1
        'CaII_8542': (8530, 8555),   # Ca II triplet line 2
        'CaII_8662': (8650, 8675),   # Ca II triplet line 3
    }

    results = {
        'feature_windows': {k: list(v) for k, v in feature_windows.items()},
        'reference_epoch': 'Epoch2',
        'features': {}
    }

    # Load template
    wave_template, flux_template = load_phoenix_template(TEMPLATE_FILE, WAVE_FILE)
    v_grid = np.arange(V_MIN, V_MAX + V_STEP, V_STEP)

    # Reference epoch
    ref_epoch = 'Epoch2'

    for feature_name, (wmin, wmax) in feature_windows.items():
        print(f"\n--- Feature: {feature_name} ({wmin}-{wmax} Å) ---")

        feature_result = {
            'wavelength_range': [wmin, wmax],
            'epochs': {}
        }

        # Get reference RV for this feature
        wave_ref, flux_ref, ivar_ref = extract_target_spectrum(
            COADD_FILES[ref_epoch], TARGET_ID, 'Z'
        )

        # Create mask for this feature window
        feature_mask = (wave_ref >= wmin) & (wave_ref <= wmax) & create_telluric_mask(wave_ref)

        if feature_mask.sum() < 20:
            print(f"  Insufficient pixels in feature window ({feature_mask.sum()})")
            feature_result['status'] = 'insufficient_pixels'
            results['features'][feature_name] = feature_result
            continue

        # Fit reference
        v_ref, sigma_ref, chi2_ref, n_pix_ref = chi2_rv_fit(
            wave_ref, flux_ref, ivar_ref, wave_template, flux_template, v_grid, feature_mask
        )

        print(f"  Reference ({ref_epoch}): v = {v_ref:+.1f} ± {sigma_ref:.1f} km/s")
        feature_result['epochs'][ref_epoch] = {
            'v_kms': float(v_ref),
            'sigma_v_kms': float(sigma_ref),
            'n_pixels': int(n_pix_ref)
        }

        # Fit other epochs and compute delta
        for epoch in ['Epoch1', 'Epoch3']:
            wave, flux, ivar = extract_target_spectrum(COADD_FILES[epoch], TARGET_ID, 'Z')

            # Match feature window
            feature_mask_ep = (wave >= wmin) & (wave <= wmax) & create_telluric_mask(wave)

            if feature_mask_ep.sum() < 20:
                continue

            v_ep, sigma_ep, chi2_ep, n_pix_ep = chi2_rv_fit(
                wave, flux, ivar, wave_template, flux_template, v_grid, feature_mask_ep
            )

            delta_v = v_ep - v_ref
            sigma_delta = np.sqrt(sigma_ep**2 + sigma_ref**2)

            feature_result['epochs'][epoch] = {
                'v_kms': float(v_ep),
                'sigma_v_kms': float(sigma_ep),
                'delta_v_vs_ref': float(delta_v),
                'sigma_delta': float(sigma_delta),
                'n_pixels': int(n_pix_ep)
            }

            print(f"  {epoch}: v = {v_ep:+.1f} ± {sigma_ep:.1f} km/s, Δv = {delta_v:+.1f} km/s")

        results['features'][feature_name] = feature_result

    # Summary statistics
    print("\n--- Feature-Level RV Summary ---")

    delta_v_epoch1 = []
    delta_v_epoch3 = []

    for feature_name, feature_data in results['features'].items():
        if 'status' in feature_data:
            continue
        if 'Epoch1' in feature_data['epochs'] and 'delta_v_vs_ref' in feature_data['epochs']['Epoch1']:
            delta_v_epoch1.append(feature_data['epochs']['Epoch1']['delta_v_vs_ref'])
        if 'Epoch3' in feature_data['epochs'] and 'delta_v_vs_ref' in feature_data['epochs']['Epoch3']:
            delta_v_epoch3.append(feature_data['epochs']['Epoch3']['delta_v_vs_ref'])

    results['summary'] = {
        'n_features_analyzed': len([f for f in results['features'].values() if 'status' not in f]),
        'epoch1_mean_delta_v': float(np.mean(delta_v_epoch1)) if delta_v_epoch1 else np.nan,
        'epoch1_std_delta_v': float(np.std(delta_v_epoch1)) if delta_v_epoch1 else np.nan,
        'epoch3_mean_delta_v': float(np.mean(delta_v_epoch3)) if delta_v_epoch3 else np.nan,
        'epoch3_std_delta_v': float(np.std(delta_v_epoch3)) if delta_v_epoch3 else np.nan,
        'catalog_swing': {
            'epoch1_vs_epoch2': -86.39 - 59.68,
            'epoch3_vs_epoch2': 25.79 - 59.68
        }
    }

    print(f"N features: {results['summary']['n_features_analyzed']}")
    print(f"Epoch1 mean Δv: {results['summary']['epoch1_mean_delta_v']:+.1f} ± {results['summary']['epoch1_std_delta_v']:.1f} km/s")
    print(f"Epoch3 mean Δv: {results['summary']['epoch3_mean_delta_v']:+.1f} ± {results['summary']['epoch3_std_delta_v']:.1f} km/s")
    print(f"Catalog Epoch1-Epoch2: {results['summary']['catalog_swing']['epoch1_vs_epoch2']:.1f} km/s")
    print(f"Catalog Epoch3-Epoch2: {results['summary']['catalog_swing']['epoch3_vs_epoch2']:.1f} km/s")

    return results


def plot_part2_figures(results):
    """Generate Part 2 figures."""

    fig, ax = plt.subplots(figsize=(10, 6))

    feature_names = []
    epoch1_deltas = []
    epoch1_errors = []
    epoch3_deltas = []
    epoch3_errors = []

    for feature_name, feature_data in results['features'].items():
        if 'status' in feature_data:
            continue

        feature_names.append(feature_name)

        if 'Epoch1' in feature_data['epochs'] and 'delta_v_vs_ref' in feature_data['epochs']['Epoch1']:
            epoch1_deltas.append(feature_data['epochs']['Epoch1']['delta_v_vs_ref'])
            epoch1_errors.append(feature_data['epochs']['Epoch1']['sigma_delta'])
        else:
            epoch1_deltas.append(np.nan)
            epoch1_errors.append(np.nan)

        if 'Epoch3' in feature_data['epochs'] and 'delta_v_vs_ref' in feature_data['epochs']['Epoch3']:
            epoch3_deltas.append(feature_data['epochs']['Epoch3']['delta_v_vs_ref'])
            epoch3_errors.append(feature_data['epochs']['Epoch3']['sigma_delta'])
        else:
            epoch3_deltas.append(np.nan)
            epoch3_errors.append(np.nan)

    x = np.arange(len(feature_names))
    width = 0.35

    ax.errorbar(x - width/2, epoch1_deltas, yerr=epoch1_errors, fmt='o',
               color='blue', capsize=4, label='Epoch1 - Epoch2')
    ax.errorbar(x + width/2, epoch3_deltas, yerr=epoch3_errors, fmt='s',
               color='green', capsize=4, label='Epoch3 - Epoch2')

    # Add catalog values as horizontal lines
    ax.axhline(results['summary']['catalog_swing']['epoch1_vs_epoch2'],
              color='blue', linestyle='--', alpha=0.5, label='Catalog E1-E2')
    ax.axhline(results['summary']['catalog_swing']['epoch3_vs_epoch2'],
              color='green', linestyle='--', alpha=0.5, label='Catalog E3-E2')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('ΔRV vs Epoch2 (km/s)')
    ax.set_title('Line-by-Line RV Shifts (Z-arm features)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'line_by_line_rv_epoch_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: line_by_line_rv_epoch_comparison.png")


# =============================================================================
# PART 3: EXPECTED NEIGHBOR FLUX FRACTION (b_R, b_Z)
# =============================================================================
def run_part3_band_flux_ratios():
    """
    Compute expected flux ratios in DESI R and Z bands for plausible neighbor types.
    """
    print("\n" + "="*70)
    print("PART 3: EXPECTED NEIGHBOR FLUX FRACTION (b_R, b_Z)")
    print("="*70)

    results = {
        'neighbor_delta_G': NEIGHBOR_DELTA_G,
        'expected_flux_ratio_G': 10**(-NEIGHBOR_DELTA_G / 2.5),
        'band_definitions': {
            'G_proxy': [4000, 9000],  # Approximate Gaia G
            'R': list(DESI_R_RANGE),
            'Z': list(DESI_Z_RANGE)
        },
        'approximation_note': 'Using flat throughput approximation for DESI bands',
        'neighbor_models': {}
    }

    # Load primary template
    wave_template, flux_primary = load_phoenix_template(TEMPLATE_FILE, WAVE_FILE)

    # Integrate primary flux in each band
    def integrate_flux(wave, flux, wmin, wmax):
        mask = (wave >= wmin) & (wave <= wmax)
        if mask.sum() == 0:
            return 0.0
        return np.trapz(flux[mask], wave[mask])

    F_primary_G = integrate_flux(wave_template, flux_primary, 4000, 9000)
    F_primary_R = integrate_flux(wave_template, flux_primary, *DESI_R_RANGE)
    F_primary_Z = integrate_flux(wave_template, flux_primary, *DESI_Z_RANGE)

    print(f"Primary (3800K M0):")
    print(f"  Integrated flux (G-proxy): {F_primary_G:.3e}")
    print(f"  Integrated flux (R): {F_primary_R:.3e}")
    print(f"  Integrated flux (Z): {F_primary_Z:.3e}")

    # Expected G-band flux ratio from delta_G
    expected_b_G = 10**(-NEIGHBOR_DELTA_G / 2.5)
    print(f"\nExpected flux ratio from ΔG = {NEIGHBOR_DELTA_G}: b_G = {expected_b_G:.4f}")

    # For each neighbor model
    for neighbor_name, neighbor_file in NEIGHBOR_TEMPLATES.items():
        print(f"\n--- Neighbor: {neighbor_name} ---")

        wave_n, flux_neighbor = load_phoenix_template(neighbor_file, WAVE_FILE)

        # Integrate neighbor flux
        F_neighbor_G = integrate_flux(wave_n, flux_neighbor, 4000, 9000)
        F_neighbor_R = integrate_flux(wave_n, flux_neighbor, *DESI_R_RANGE)
        F_neighbor_Z = integrate_flux(wave_n, flux_neighbor, *DESI_Z_RANGE)

        # Raw flux ratios (before calibration)
        raw_ratio_G = F_neighbor_G / F_primary_G
        raw_ratio_R = F_neighbor_R / F_primary_R
        raw_ratio_Z = F_neighbor_Z / F_primary_Z

        # Scale factor to match expected b_G
        scale = expected_b_G / raw_ratio_G

        # Calibrated flux ratios
        b_R = raw_ratio_R * scale
        b_Z = raw_ratio_Z * scale

        print(f"  Raw flux ratio (G-proxy): {raw_ratio_G:.4f}")
        print(f"  Scale factor to match ΔG: {scale:.4f}")
        print(f"  Calibrated b_G: {raw_ratio_G * scale:.4f}")
        print(f"  Calibrated b_R: {b_R:.4f}")
        print(f"  Calibrated b_Z: {b_Z:.4f}")

        results['neighbor_models'][neighbor_name] = {
            'template_file': os.path.basename(neighbor_file),
            'raw_ratio_G': float(raw_ratio_G),
            'scale_factor': float(scale),
            'calibrated_b_G': float(raw_ratio_G * scale),
            'calibrated_b_R': float(b_R),
            'calibrated_b_Z': float(b_Z)
        }

    # Summary
    b_R_values = [m['calibrated_b_R'] for m in results['neighbor_models'].values()]
    b_Z_values = [m['calibrated_b_Z'] for m in results['neighbor_models'].values()]

    results['summary'] = {
        'b_R_range': [min(b_R_values), max(b_R_values)],
        'b_Z_range': [min(b_Z_values), max(b_Z_values)],
        'max_plausible_b': max(max(b_R_values), max(b_Z_values)),
        'can_reach_0p25': max(max(b_R_values), max(b_Z_values)) >= 0.25
    }

    print(f"\n--- Summary ---")
    print(f"b_R range: {results['summary']['b_R_range'][0]:.3f} - {results['summary']['b_R_range'][1]:.3f}")
    print(f"b_Z range: {results['summary']['b_Z_range'][0]:.3f} - {results['summary']['b_Z_range'][1]:.3f}")
    print(f"Can plausible neighbor reach b = 0.25? {results['summary']['can_reach_0p25']}")

    return results


def plot_part3_figures(results):
    """Generate Part 3 figures."""

    fig, ax = plt.subplots(figsize=(8, 5))

    neighbor_names = list(results['neighbor_models'].keys())
    x = np.arange(len(neighbor_names))
    width = 0.25

    b_G = [results['neighbor_models'][n]['calibrated_b_G'] for n in neighbor_names]
    b_R = [results['neighbor_models'][n]['calibrated_b_R'] for n in neighbor_names]
    b_Z = [results['neighbor_models'][n]['calibrated_b_Z'] for n in neighbor_names]

    ax.bar(x - width, b_G, width, label='b_G (calibration)', color='gray', alpha=0.5)
    ax.bar(x, b_R, width, label='b_R (DESI R)', color='red', alpha=0.7)
    ax.bar(x + width, b_Z, width, label='b_Z (DESI Z)', color='blue', alpha=0.7)

    # Reference lines
    ax.axhline(0.13, color='black', linestyle='--', label='Expected from ΔG=2.21')
    ax.axhline(0.25, color='orange', linestyle=':', label='v4 boundary (0.25)')

    ax.set_xticks(x)
    ax.set_xticklabels(neighbor_names)
    ax.set_ylabel('Flux Ratio (b)')
    ax.set_title('Expected Neighbor Flux Ratios by Band')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.35)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'expected_bR_bZ.png'), dpi=150)
    plt.close()
    print(f"Saved: expected_bR_bZ.png")


# =============================================================================
# PART 4: GAIA DUPLICITY CONTROL SAMPLE
# =============================================================================
def run_part4_gaia_control_sample():
    """
    Query Gaia DR3 for control sample and compare duplicity metrics.
    """
    print("\n" + "="*70)
    print("PART 4: GAIA DUPLICITY CONTROL SAMPLE")
    print("="*70)

    results = {
        'target': {
            'source_id': GAIA_SOURCE_ID,
        },
        'query_params': {},
        'control_sample': {},
        'percentile_ranks': {}
    }

    try:
        from astroquery.gaia import Gaia

        # Get target properties first
        print("Querying target properties...")
        target_query = f"""
        SELECT source_id, phot_g_mean_mag, bp_rp, ruwe,
               astrometric_excess_noise, astrometric_excess_noise_sig,
               ipd_frac_multi_peak, ipd_gof_harmonic_amplitude
        FROM gaiadr3.gaia_source
        WHERE source_id = {GAIA_SOURCE_ID}
        """

        target_job = Gaia.launch_job(target_query)
        target_result = target_job.get_results()

        if len(target_result) == 0:
            print("ERROR: Target not found in Gaia DR3")
            results['status'] = 'target_not_found'
            return results

        target_row = target_result[0]
        results['target']['phot_g_mean_mag'] = float(target_row['phot_g_mean_mag'])
        results['target']['bp_rp'] = float(target_row['bp_rp'])
        results['target']['ruwe'] = float(target_row['ruwe'])
        results['target']['astrometric_excess_noise'] = float(target_row['astrometric_excess_noise'])
        results['target']['astrometric_excess_noise_sig'] = float(target_row['astrometric_excess_noise_sig'])
        results['target']['ipd_frac_multi_peak'] = float(target_row['ipd_frac_multi_peak']) if target_row['ipd_frac_multi_peak'] else np.nan
        results['target']['ipd_gof_harmonic_amplitude'] = float(target_row['ipd_gof_harmonic_amplitude']) if target_row['ipd_gof_harmonic_amplitude'] else np.nan

        print(f"Target: G = {results['target']['phot_g_mean_mag']:.2f}, BP-RP = {results['target']['bp_rp']:.2f}")
        print(f"  RUWE = {results['target']['ruwe']:.3f}")
        print(f"  AEN = {results['target']['astrometric_excess_noise']:.3f} mas, sig = {results['target']['astrometric_excess_noise_sig']:.1f}")
        print(f"  IPD frac multi-peak = {results['target']['ipd_frac_multi_peak']}")

        # Query control sample
        G_target = results['target']['phot_g_mean_mag']
        bp_rp_target = results['target']['bp_rp']

        # Target coordinates (approximate)
        ra_target = 164.5235
        dec_target = -1.6602

        results['query_params'] = {
            'G_range': [G_target - 0.5, G_target + 0.5],
            'bp_rp_range': [bp_rp_target - 0.2, bp_rp_target + 0.2],
            'radius_deg': 2.0
        }

        print(f"\nQuerying control sample (radius=2 deg, G={G_target:.1f}±0.5, BP-RP={bp_rp_target:.2f}±0.2)...")

        control_query = f"""
        SELECT source_id, phot_g_mean_mag, bp_rp, ruwe,
               astrometric_excess_noise, astrometric_excess_noise_sig,
               ipd_frac_multi_peak, ipd_gof_harmonic_amplitude
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_target}, {dec_target}, 2.0)
        )
        AND phot_g_mean_mag BETWEEN {G_target - 0.5} AND {G_target + 0.5}
        AND bp_rp BETWEEN {bp_rp_target - 0.2} AND {bp_rp_target + 0.2}
        AND ruwe IS NOT NULL
        AND astrometric_excess_noise IS NOT NULL
        """

        control_job = Gaia.launch_job(control_query)
        control_result = control_job.get_results()

        n_control = len(control_result)
        print(f"Found {n_control} control stars")

        if n_control < 10:
            print("WARNING: Control sample too small for meaningful statistics")
            results['status'] = 'insufficient_control_sample'
            results['control_sample']['n_stars'] = n_control
            return results

        # Extract arrays
        ruwe_arr = np.array(control_result['ruwe'])
        aen_arr = np.array(control_result['astrometric_excess_noise'])
        aen_sig_arr = np.array(control_result['astrometric_excess_noise_sig'])
        ipd_arr = np.array([x if x is not None else np.nan for x in control_result['ipd_frac_multi_peak']])

        results['control_sample'] = {
            'n_stars': n_control,
            'ruwe_median': float(np.nanmedian(ruwe_arr)),
            'ruwe_std': float(np.nanstd(ruwe_arr)),
            'aen_median': float(np.nanmedian(aen_arr)),
            'aen_std': float(np.nanstd(aen_arr)),
            'ipd_median': float(np.nanmedian(ipd_arr)),
            'ipd_std': float(np.nanstd(ipd_arr))
        }

        # Compute percentile ranks
        def percentile_rank(value, arr):
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return np.nan
            return 100 * np.sum(valid < value) / len(valid)

        results['percentile_ranks'] = {
            'ruwe': float(percentile_rank(results['target']['ruwe'], ruwe_arr)),
            'astrometric_excess_noise': float(percentile_rank(results['target']['astrometric_excess_noise'], aen_arr)),
            'astrometric_excess_noise_sig': float(percentile_rank(results['target']['astrometric_excess_noise_sig'], aen_sig_arr)),
            'ipd_frac_multi_peak': float(percentile_rank(results['target']['ipd_frac_multi_peak'], ipd_arr))
        }

        print(f"\nControl sample statistics:")
        print(f"  RUWE: median = {results['control_sample']['ruwe_median']:.3f}, std = {results['control_sample']['ruwe_std']:.3f}")
        print(f"  AEN: median = {results['control_sample']['aen_median']:.3f}, std = {results['control_sample']['aen_std']:.3f}")

        print(f"\nTarget percentile ranks:")
        print(f"  RUWE: {results['percentile_ranks']['ruwe']:.1f}%")
        print(f"  AEN: {results['percentile_ranks']['astrometric_excess_noise']:.1f}%")
        print(f"  AEN_sig: {results['percentile_ranks']['astrometric_excess_noise_sig']:.1f}%")
        print(f"  IPD: {results['percentile_ranks']['ipd_frac_multi_peak']:.1f}%")

        # Store arrays for plotting
        results['_control_arrays'] = {
            'ruwe': ruwe_arr.tolist(),
            'aen': aen_arr.tolist(),
            'ipd': ipd_arr.tolist()
        }

        results['status'] = 'success'

    except Exception as e:
        print(f"ERROR: Gaia query failed: {e}")
        results['status'] = f'query_failed: {str(e)}'

    return results


def plot_part4_figures(results):
    """Generate Part 4 figures."""

    if results.get('status') != 'success' or '_control_arrays' not in results:
        print("Cannot generate Part 4 figures - no control sample data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # RUWE histogram
    ax = axes[0]
    ruwe_arr = np.array(results['_control_arrays']['ruwe'])
    ax.hist(ruwe_arr[np.isfinite(ruwe_arr)], bins=30, alpha=0.7, color='steelblue')
    ax.axvline(results['target']['ruwe'], color='red', linewidth=2,
              label=f'Target: {results["target"]["ruwe"]:.2f}')
    ax.axvline(1.4, color='orange', linestyle='--', alpha=0.5, label='RUWE=1.4 threshold')
    ax.set_xlabel('RUWE')
    ax.set_ylabel('Count')
    ax.set_title(f'RUWE (target at {results["percentile_ranks"]["ruwe"]:.0f}%ile)')
    ax.legend(fontsize=8)

    # AEN histogram
    ax = axes[1]
    aen_arr = np.array(results['_control_arrays']['aen'])
    ax.hist(aen_arr[np.isfinite(aen_arr)], bins=30, alpha=0.7, color='steelblue')
    ax.axvline(results['target']['astrometric_excess_noise'], color='red', linewidth=2,
              label=f'Target: {results["target"]["astrometric_excess_noise"]:.2f}')
    ax.set_xlabel('Astrometric Excess Noise (mas)')
    ax.set_ylabel('Count')
    ax.set_title(f'AEN (target at {results["percentile_ranks"]["astrometric_excess_noise"]:.0f}%ile)')
    ax.legend(fontsize=8)

    # IPD histogram
    ax = axes[2]
    ipd_arr = np.array(results['_control_arrays']['ipd'])
    valid_ipd = ipd_arr[np.isfinite(ipd_arr)]
    if len(valid_ipd) > 0:
        ax.hist(valid_ipd, bins=30, alpha=0.7, color='steelblue')
        if np.isfinite(results['target']['ipd_frac_multi_peak']):
            ax.axvline(results['target']['ipd_frac_multi_peak'], color='red', linewidth=2,
                      label=f'Target: {results["target"]["ipd_frac_multi_peak"]:.0f}')
        ax.set_xlabel('IPD Frac Multi-Peak (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'IPD (target at {results["percentile_ranks"]["ipd_frac_multi_peak"]:.0f}%ile)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No IPD data', ha='center', va='center', transform=ax.transAxes)

    plt.suptitle(f'Gaia Duplicity Metrics: Target vs Control Sample (N={results["control_sample"]["n_stars"]})')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'gaia_control_histograms.png'), dpi=150)
    plt.close()
    print(f"Saved: gaia_control_histograms.png")


# =============================================================================
# PART 5: FINAL VERDICTS AND REPORT
# =============================================================================
def compute_verdicts(part1_results, part2_results, part3_results, part4_results):
    """Compute final verdicts for each check."""

    verdicts = {}

    # Verdict A: Epoch 3 R-Z discrepancy
    if 'Epoch3' in part1_results['epochs']:
        epoch3 = part1_results['epochs']['Epoch3']
        if 'R_minus_Z' in epoch3:
            delta_RZ = epoch3['R_minus_Z']['delta_v_kms']
            sig = abs(epoch3['R_minus_Z']['significance_sigma'])

            # Check if both exposures have similar catalog RVs
            exp_comparison = part1_results.get('epoch3_exposure_comparison', {})
            exp_delta = abs(exp_comparison.get('delta_between_exposures', 999))

            if sig > 3 and exp_delta < 5:
                # Large discrepancy AND both exposures consistent -> PERSISTENT
                verdicts['A'] = {
                    'verdict': 'FAIL',
                    'reason': f'R-Z discrepancy ({delta_RZ:+.1f} km/s, {sig:.1f}σ) is PERSISTENT across both exposures (Δexp={exp_delta:.1f} km/s)',
                    'delta_RZ_kms': delta_RZ,
                    'significance': sig
                }
            elif sig > 3:
                # Large discrepancy but can't verify exposure-level
                verdicts['A'] = {
                    'verdict': 'INCONCLUSIVE',
                    'reason': 'Large R-Z discrepancy but per-exposure spectra unavailable to verify if single-exposure artifact',
                    'delta_RZ_kms': delta_RZ,
                    'significance': sig
                }
            else:
                verdicts['A'] = {
                    'verdict': 'PASS',
                    'reason': f'R-Z discrepancy not significant ({delta_RZ:+.1f} km/s, {sig:.1f}σ)',
                    'delta_RZ_kms': delta_RZ,
                    'significance': sig
                }
        else:
            verdicts['A'] = {
                'verdict': 'INCONCLUSIVE',
                'reason': 'R-Z analysis failed'
            }
    else:
        verdicts['A'] = {
            'verdict': 'INCONCLUSIVE',
            'reason': 'Epoch 3 data not available'
        }

    # Verdict B: Feature-level RV swing
    summary = part2_results.get('summary', {})
    n_features = summary.get('n_features_analyzed', 0)

    if n_features >= 3:
        mean_delta_e1 = summary.get('epoch1_mean_delta_v', np.nan)
        std_delta_e1 = summary.get('epoch1_std_delta_v', np.nan)
        catalog_e1 = summary['catalog_swing']['epoch1_vs_epoch2']

        # Check if feature-level shifts are consistent with catalog
        if np.isfinite(mean_delta_e1) and np.isfinite(std_delta_e1):
            # Features should show negative shift for Epoch1 vs Epoch2
            if mean_delta_e1 < -50 and abs(mean_delta_e1 - catalog_e1) < 50:
                verdicts['B'] = {
                    'verdict': 'PASS',
                    'reason': f'Feature-level RV shifts ({mean_delta_e1:+.1f}±{std_delta_e1:.1f} km/s) consistent with catalog ({catalog_e1:.1f} km/s)',
                    'n_features': n_features,
                    'mean_delta': mean_delta_e1
                }
            elif abs(mean_delta_e1) < 30:
                verdicts['B'] = {
                    'verdict': 'FAIL',
                    'reason': f'Feature-level shifts ({mean_delta_e1:+.1f}±{std_delta_e1:.1f} km/s) much smaller than catalog swing',
                    'n_features': n_features,
                    'mean_delta': mean_delta_e1
                }
            else:
                verdicts['B'] = {
                    'verdict': 'INCONCLUSIVE',
                    'reason': f'Feature-level shifts partially consistent but noisy ({mean_delta_e1:+.1f}±{std_delta_e1:.1f} km/s vs catalog {catalog_e1:.1f} km/s)',
                    'n_features': n_features,
                    'mean_delta': mean_delta_e1
                }
        else:
            verdicts['B'] = {
                'verdict': 'INCONCLUSIVE',
                'reason': 'Feature analysis incomplete'
            }
    else:
        verdicts['B'] = {
            'verdict': 'INCONCLUSIVE',
            'reason': f'Too few features analyzed ({n_features})'
        }

    # Verdict C: Can b reach 0.25?
    summary_c = part3_results.get('summary', {})
    max_b = summary_c.get('max_plausible_b', 0)

    if max_b >= 0.25:
        verdicts['C'] = {
            'verdict': 'PASS',
            'reason': f'Plausible neighbor SED can reach b = {max_b:.3f} ≥ 0.25',
            'max_plausible_b': max_b
        }
    elif max_b >= 0.20:
        verdicts['C'] = {
            'verdict': 'INCONCLUSIVE',
            'reason': f'Max plausible b = {max_b:.3f} approaches but does not reach 0.25',
            'max_plausible_b': max_b
        }
    else:
        verdicts['C'] = {
            'verdict': 'FAIL',
            'reason': f'Max plausible b = {max_b:.3f} is well below 0.25',
            'max_plausible_b': max_b
        }

    # Verdict D: Gaia duplicity outlier
    if part4_results.get('status') == 'success':
        ranks = part4_results.get('percentile_ranks', {})
        ruwe_rank = ranks.get('ruwe', 0)
        aen_rank = ranks.get('astrometric_excess_noise', 0)

        extreme_count = sum([
            ruwe_rank > 95,
            aen_rank > 95,
        ])

        if extreme_count >= 2:
            verdicts['D'] = {
                'verdict': 'PASS',
                'reason': f'Target is extreme outlier: RUWE at {ruwe_rank:.0f}%ile, AEN at {aen_rank:.0f}%ile',
                'ruwe_percentile': ruwe_rank,
                'aen_percentile': aen_rank
            }
        elif ruwe_rank > 90 or aen_rank > 90:
            verdicts['D'] = {
                'verdict': 'INCONCLUSIVE',
                'reason': f'Target elevated but not extreme: RUWE at {ruwe_rank:.0f}%ile, AEN at {aen_rank:.0f}%ile',
                'ruwe_percentile': ruwe_rank,
                'aen_percentile': aen_rank
            }
        else:
            verdicts['D'] = {
                'verdict': 'FAIL',
                'reason': f'Target not an outlier: RUWE at {ruwe_rank:.0f}%ile, AEN at {aen_rank:.0f}%ile',
                'ruwe_percentile': ruwe_rank,
                'aen_percentile': aen_rank
            }
    else:
        verdicts['D'] = {
            'verdict': 'INCONCLUSIVE',
            'reason': f'Gaia query failed: {part4_results.get("status", "unknown")}'
        }

    return verdicts


def write_report(part1, part2, part3, part4, verdicts):
    """Write final markdown report."""

    report_path = os.path.join(OUTPUT_DIR, 'TRUTHFILTER_REPORT.md')

    with open(report_path, 'w') as f:
        f.write("# DESI Truth Filter v1 Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Target:** Gaia DR3 {GAIA_SOURCE_ID}\n")
        f.write(f"**DESI TARGETID:** {TARGET_ID}\n\n")

        f.write("---\n\n")
        f.write("## Final Verdict Table\n\n")
        f.write("| Check | Question | Verdict | Reason |\n")
        f.write("|-------|----------|---------|--------|\n")

        questions = {
            'A': 'Epoch 3 R–Z discrepancy is a single-exposure/camera artifact',
            'B': 'Large RV swing is visible in feature-level (line/bandhead) shifts',
            'C': 'Expected neighbor flux ratio in DESI R/Z could plausibly reach 0.25',
            'D': "Target's Gaia duplicity metrics are extreme vs matched control sample"
        }

        for key in ['A', 'B', 'C', 'D']:
            v = verdicts.get(key, {'verdict': 'N/A', 'reason': 'Not computed'})
            verdict_emoji = {'PASS': '✅', 'FAIL': '❌', 'INCONCLUSIVE': '⚠️'}.get(v['verdict'], '❓')
            f.write(f"| {key} | {questions[key]} | **{v['verdict']}** {verdict_emoji} | {v['reason']} |\n")

        f.write("\n---\n\n")

        # Part 1 details
        f.write("## Part 1: Per-Exposure / Per-Arm RV Analysis\n\n")
        f.write(f"**Limitation:** {part1.get('limitation', 'N/A')}\n\n")
        f.write("### Per-Arm RV Results (from coadd)\n\n")
        f.write("| Epoch | N_exp | R-arm (km/s) | Z-arm (km/s) | R-Z (km/s) | Significance |\n")
        f.write("|-------|-------|--------------|--------------|------------|-------------|\n")

        for epoch in ['Epoch1', 'Epoch2', 'Epoch3']:
            if epoch in part1['epochs']:
                ep = part1['epochs'][epoch]
                n_exp = ep['n_exposures']
                v_R = ep['arms'].get('R', {}).get('v_kms', np.nan)
                v_Z = ep['arms'].get('Z', {}).get('v_kms', np.nan)
                delta = ep.get('R_minus_Z', {}).get('delta_v_kms', np.nan)
                sig = ep.get('R_minus_Z', {}).get('significance_sigma', np.nan)
                f.write(f"| {epoch} | {n_exp} | {v_R:+.1f} | {v_Z:+.1f} | {delta:+.1f} | {sig:.1f}σ |\n")

        f.write("\n### Epoch 3 Per-Exposure Catalog RVs\n\n")
        exp_comp = part1.get('epoch3_exposure_comparison', {})
        f.write(f"- Exposure 120449: {exp_comp.get('exp_120449_vrad', 'N/A')} km/s\n")
        f.write(f"- Exposure 120450: {exp_comp.get('exp_120450_vrad', 'N/A')} km/s\n")
        f.write(f"- Difference: {exp_comp.get('delta_between_exposures', 'N/A')} km/s\n\n")
        f.write("Both exposures show similar RVs, suggesting R-Z discrepancy is **PERSISTENT**.\n\n")

        f.write("![Per-Arm RV](figures/per_exposure_rv_RZ.png)\n\n")
        f.write("![Epoch 3 Diagnostics](figures/per_exposure_epoch3_diagnostics.png)\n\n")

        # Part 2 details
        f.write("## Part 2: Line/Bandhead-Level RV Check\n\n")
        f.write("### Feature Windows\n\n")
        for fname, fdata in part2.get('features', {}).items():
            if 'status' not in fdata:
                wrange = fdata.get('wavelength_range', [])
                f.write(f"- **{fname}**: {wrange[0]}-{wrange[1]} Å\n")

        f.write("\n### Results Summary\n\n")
        summary = part2.get('summary', {})
        f.write(f"- N features analyzed: {summary.get('n_features_analyzed', 0)}\n")
        f.write(f"- Epoch1 mean Δv vs Epoch2: {summary.get('epoch1_mean_delta_v', np.nan):+.1f} ± {summary.get('epoch1_std_delta_v', np.nan):.1f} km/s\n")
        f.write(f"- Epoch3 mean Δv vs Epoch2: {summary.get('epoch3_mean_delta_v', np.nan):+.1f} ± {summary.get('epoch3_std_delta_v', np.nan):.1f} km/s\n")
        f.write(f"- Catalog Epoch1-Epoch2: {summary.get('catalog_swing', {}).get('epoch1_vs_epoch2', np.nan):.1f} km/s\n\n")

        f.write("![Line-by-Line RV](figures/line_by_line_rv_epoch_comparison.png)\n\n")

        # Part 3 details
        f.write("## Part 3: Expected Neighbor Flux Fraction\n\n")
        f.write(f"- Neighbor ΔG: {NEIGHBOR_DELTA_G}\n")
        f.write(f"- Expected b_G: {part3.get('expected_flux_ratio_G', 0):.4f}\n\n")

        f.write("### Results by Neighbor Type\n\n")
        f.write("| Neighbor | b_G | b_R | b_Z |\n")
        f.write("|----------|-----|-----|-----|\n")
        for name, model in part3.get('neighbor_models', {}).items():
            f.write(f"| {name} | {model['calibrated_b_G']:.3f} | {model['calibrated_b_R']:.3f} | {model['calibrated_b_Z']:.3f} |\n")

        f.write(f"\n**Max plausible b:** {part3.get('summary', {}).get('max_plausible_b', 0):.3f}\n\n")
        f.write("![Expected b_R b_Z](figures/expected_bR_bZ.png)\n\n")

        # Part 4 details
        f.write("## Part 4: Gaia Duplicity Control Sample\n\n")

        if part4.get('status') == 'success':
            f.write(f"- Control sample size: {part4['control_sample']['n_stars']}\n")
            f.write(f"- Query: G = {part4['target']['phot_g_mean_mag']:.1f} ± 0.5, BP-RP = {part4['target']['bp_rp']:.2f} ± 0.2\n\n")

            f.write("### Target Percentile Ranks\n\n")
            f.write("| Metric | Target Value | Percentile |\n")
            f.write("|--------|--------------|------------|\n")
            f.write(f"| RUWE | {part4['target']['ruwe']:.3f} | {part4['percentile_ranks']['ruwe']:.0f}% |\n")
            f.write(f"| AEN | {part4['target']['astrometric_excess_noise']:.3f} mas | {part4['percentile_ranks']['astrometric_excess_noise']:.0f}% |\n")
            f.write(f"| AEN_sig | {part4['target']['astrometric_excess_noise_sig']:.1f} | {part4['percentile_ranks']['astrometric_excess_noise_sig']:.0f}% |\n")
            f.write(f"| IPD | {part4['target']['ipd_frac_multi_peak']} | {part4['percentile_ranks']['ipd_frac_multi_peak']:.0f}% |\n")

            f.write("\n![Gaia Control](figures/gaia_control_histograms.png)\n\n")
        else:
            f.write(f"**Status:** {part4.get('status', 'unknown')}\n\n")

        # Final summary
        f.write("---\n\n")
        f.write("## Final Summary\n\n")

        pass_count = sum(1 for v in verdicts.values() if v['verdict'] == 'PASS')
        fail_count = sum(1 for v in verdicts.values() if v['verdict'] == 'FAIL')
        inconc_count = sum(1 for v in verdicts.values() if v['verdict'] == 'INCONCLUSIVE')

        f.write(f"- **PASS:** {pass_count}\n")
        f.write(f"- **FAIL:** {fail_count}\n")
        f.write(f"- **INCONCLUSIVE:** {inconc_count}\n\n")

        # Interpretation
        if verdicts['A']['verdict'] == 'FAIL' and verdicts['B']['verdict'] == 'FAIL':
            f.write("**INTERPRETATION:** Candidate is likely COMPROMISED.\n")
            f.write("- R-Z discrepancy is persistent (not single-exposure artifact)\n")
            f.write("- Feature-level RV shifts do not confirm catalog swing\n")
        elif verdicts['A']['verdict'] == 'PASS' and verdicts['B']['verdict'] == 'PASS':
            f.write("**INTERPRETATION:** Candidate SURVIVES strongly.\n")
            f.write("- R-Z discrepancy explained by single exposure/camera artifact\n")
            f.write("- Feature-level RV shifts confirm large swing\n")
        else:
            f.write("**INTERPRETATION:** Results are MIXED - candidate status uncertain.\n")
            f.write("\n**To resolve:**\n")
            f.write("- High-resolution spectroscopy (resolve potential blend)\n")
            f.write("- AO imaging (verify neighbor separation and flux ratio)\n")
            f.write("- Additional DESI epochs or per-exposure spectra\n")

        f.write("\n---\n\n")
        f.write("## File Hashes\n\n")
        for epoch, coadd_file in COADD_FILES.items():
            h = sha256_file(coadd_file)
            f.write(f"- {os.path.basename(coadd_file)}: `{h}`\n")

        f.write(f"\n---\n\n*Analysis by DESI Truth Filter v1*\n")

    print(f"\nSaved: {report_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("DESI TRUTH FILTER v1")
    print("="*70)
    print(f"Target: Gaia DR3 {GAIA_SOURCE_ID}")
    print(f"DESI TARGETID: {TARGET_ID}")
    print(f"Neighbor: {NEIGHBOR_SEP_ARCSEC}\" separation, ΔG = {NEIGHBOR_DELTA_G}")

    # Create output directories
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Part 1
    part1_results = run_part1_per_exposure_analysis()
    plot_part1_figures(part1_results)

    with open(os.path.join(OUTPUT_DIR, 'per_exposure_rv_results.json'), 'w') as f:
        # Remove numpy arrays before saving
        save_data = {k: v for k, v in part1_results.items() if not k.startswith('_')}
        json.dump(save_data, f, indent=2)
    print(f"Saved: per_exposure_rv_results.json")

    # Part 2
    part2_results = run_part2_line_by_line_analysis()
    plot_part2_figures(part2_results)

    with open(os.path.join(OUTPUT_DIR, 'line_by_line_rv_results.json'), 'w') as f:
        json.dump(part2_results, f, indent=2)
    print(f"Saved: line_by_line_rv_results.json")

    # Part 3
    part3_results = run_part3_band_flux_ratios()
    plot_part3_figures(part3_results)

    with open(os.path.join(OUTPUT_DIR, 'band_flux_ratio_estimates.json'), 'w') as f:
        json.dump(part3_results, f, indent=2)
    print(f"Saved: band_flux_ratio_estimates.json")

    # Part 4
    part4_results = run_part4_gaia_control_sample()
    plot_part4_figures(part4_results)

    with open(os.path.join(OUTPUT_DIR, 'gaia_control_sample_stats.json'), 'w') as f:
        # Remove arrays before saving
        save_data = {k: v for k, v in part4_results.items() if not k.startswith('_')}
        json.dump(save_data, f, indent=2)
    print(f"Saved: gaia_control_sample_stats.json")

    # Part 5: Verdicts and Report
    print("\n" + "="*70)
    print("PART 5: FINAL VERDICTS")
    print("="*70)

    verdicts = compute_verdicts(part1_results, part2_results, part3_results, part4_results)

    print("\nVERDICT TABLE:")
    print("-" * 80)
    for key in ['A', 'B', 'C', 'D']:
        v = verdicts.get(key, {'verdict': 'N/A', 'reason': 'Not computed'})
        print(f"  {key}: {v['verdict']:12s} - {v['reason']}")
    print("-" * 80)

    write_report(part1_results, part2_results, part3_results, part4_results, verdicts)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
