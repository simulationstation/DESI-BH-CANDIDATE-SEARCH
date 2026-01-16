#!/usr/bin/env python3
"""
FORENSIC V6: LAMOST RV Re-measurement from Raw FITS
Independent verification of the ~20 km/s RV shift claim.

Method: Cross-correlation in velocity space using self-template approach.
- Cross-correlate epoch 1 vs epoch 2 to measure ΔRV directly
- Use CCF peak fitting for sub-pixel precision
- Multiple wavelength masks for consistency check
"""

import numpy as np
import json
import hashlib
from pathlib import Path
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Constants
C_KMS = 299792.458  # Speed of light in km/s
FITS_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH/data/lamost")
OUTPUT_DIR = Path("/home/primary/DESI-BH-CANDIDATE-SEARCH/outputs/forensic_v6")
FIG_DIR = OUTPUT_DIR / "figures"

# Target info
OBSID_1 = 437513049
OBSID_2 = 870813030

def sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            sha256.update(block)
    return sha256.hexdigest()

def load_lamost_spectrum(filepath):
    """
    Load LAMOST spectrum from FITS file.
    Returns wavelength (Angstroms), flux, and header info.

    LAMOST FITS format:
    - HDU 0: Primary header with metadata (no data)
    - HDU 1 (COADD): Binary table with FLUX, IVAR, WAVELENGTH columns
    """
    with fits.open(filepath) as hdul:
        header = hdul[0].header

        # Get spectrum data from COADD extension
        if len(hdul) > 1 and 'COADD' in [h.name for h in hdul]:
            coadd_data = hdul['COADD'].data
            flux = coadd_data['FLUX'][0]  # First (only) row
            wavelength = coadd_data['WAVELENGTH'][0]
            ivar = coadd_data['IVAR'][0] if 'IVAR' in coadd_data.names else None
        elif len(hdul) > 1:
            # Try HDU 1 as binary table
            coadd_data = hdul[1].data
            if hasattr(coadd_data, 'names') and 'FLUX' in coadd_data.names:
                flux = coadd_data['FLUX'][0]
                wavelength = coadd_data['WAVELENGTH'][0]
                ivar = coadd_data['IVAR'][0] if 'IVAR' in coadd_data.names else None
            else:
                # Fallback: try reading as image
                flux = hdul[0].data
                if flux is None:
                    flux = hdul[1].data
                # Build wavelength from header
                crval1 = header.get('CRVAL1', header.get('COEFF0', 3800))
                cdelt1 = header.get('CDELT1', header.get('COEFF1', 1.0))
                crpix1 = header.get('CRPIX1', 1)
                naxis1 = len(flux) if flux is not None else header.get('NAXIS1', 3906)
                if 'COEFF0' in header:
                    log_wave = crval1 + cdelt1 * (np.arange(naxis1) - (crpix1 - 1))
                    wavelength = 10**log_wave
                else:
                    wavelength = crval1 + cdelt1 * (np.arange(naxis1) - (crpix1 - 1))
                ivar = None
        else:
            raise ValueError(f"Cannot parse LAMOST FITS structure for {filepath}")

        # Extract key header info
        info = {
            'obsid': header.get('OBSID', 'unknown'),
            'date_obs': header.get('DATE-OBS', 'unknown'),
            'mjd': header.get('MJD', 0),
            'helio_rv': header.get('HELIO_RV', np.nan),
            'z': header.get('Z', np.nan),
            'snr_g': header.get('SNRG', np.nan),
            'snr_r': header.get('SNRR', np.nan),
            'snr_i': header.get('SNRI', np.nan),
            'ra': header.get('RA', np.nan),
            'dec': header.get('DEC', np.nan),
            'class': header.get('CLASS', 'unknown'),
            'subclass': header.get('SUBCLASS', 'unknown'),
        }

    return wavelength, flux, info

def continuum_normalize(wavelength, flux, poly_order=5, sigma_clip=3.0, iterations=3):
    """
    Continuum normalize spectrum using iterative polynomial fitting.
    """
    # Remove NaN/inf values
    good = np.isfinite(flux) & np.isfinite(wavelength) & (flux > 0)
    if np.sum(good) < 100:
        return flux / np.nanmedian(flux)

    wl = wavelength[good]
    fl = flux[good]

    # Iterative sigma clipping polynomial fit
    mask = np.ones(len(wl), dtype=bool)
    for _ in range(iterations):
        if np.sum(mask) < poly_order + 1:
            break
        coeffs = np.polyfit(wl[mask], fl[mask], poly_order)
        continuum = np.polyval(coeffs, wl)
        residuals = fl - continuum
        std = np.std(residuals[mask])
        mask = np.abs(residuals) < sigma_clip * std

    # Final continuum
    coeffs = np.polyfit(wl[mask], fl[mask], poly_order)
    full_continuum = np.polyval(coeffs, wavelength)
    full_continuum[full_continuum <= 0] = 1  # Avoid division by zero

    normalized = flux / full_continuum
    return normalized

def cross_correlate_spectra(wl1, fl1, wl2, fl2, v_min=-500, v_max=500, v_step=1.0):
    """
    Cross-correlate two spectra in velocity space.

    Returns:
    - velocities: velocity grid (km/s)
    - ccf: cross-correlation function
    """
    # Find common wavelength range
    wl_min = max(np.nanmin(wl1), np.nanmin(wl2))
    wl_max = min(np.nanmax(wl1), np.nanmax(wl2))

    # Create common wavelength grid (log-spaced for constant velocity sampling)
    n_pix = int(np.log(wl_max / wl_min) / np.log(1 + v_step / C_KMS))
    wl_common = np.geomspace(wl_min, wl_max, n_pix)

    # Interpolate both spectra onto common grid
    interp1 = interp1d(wl1, fl1, kind='linear', bounds_error=False, fill_value=np.nan)
    interp2 = interp1d(wl2, fl2, kind='linear', bounds_error=False, fill_value=np.nan)

    fl1_common = interp1(wl_common)
    fl2_common = interp2(wl_common)

    # Mask bad pixels
    good = np.isfinite(fl1_common) & np.isfinite(fl2_common)
    if np.sum(good) < 100:
        return np.array([0]), np.array([0])

    fl1_common[~good] = 0
    fl2_common[~good] = 0

    # Subtract mean and normalize
    fl1_common = (fl1_common - np.mean(fl1_common[good])) / np.std(fl1_common[good])
    fl2_common = (fl2_common - np.mean(fl2_common[good])) / np.std(fl2_common[good])

    # Velocity grid
    velocities = np.arange(v_min, v_max + v_step, v_step)
    ccf = np.zeros(len(velocities))

    # Compute CCF by shifting spectrum 2
    for i, v in enumerate(velocities):
        # Doppler shift factor
        doppler = np.sqrt((1 + v / C_KMS) / (1 - v / C_KMS))
        wl_shifted = wl_common * doppler

        # Interpolate shifted spectrum
        interp_shifted = interp1d(wl_shifted, fl2_common, kind='linear',
                                   bounds_error=False, fill_value=0)
        fl2_shifted = interp_shifted(wl_common)

        # Cross-correlation (normalized)
        ccf[i] = np.sum(fl1_common * fl2_shifted) / np.sqrt(np.sum(fl1_common**2) * np.sum(fl2_shifted**2))

    return velocities, ccf

def fit_ccf_peak(velocities, ccf, fit_width=50):
    """
    Fit CCF peak with Gaussian to get sub-pixel RV and uncertainty.

    Returns:
    - rv: peak velocity
    - rv_err: velocity uncertainty
    - peak_height: CCF peak value
    - fwhm: full width at half maximum
    - secondary_ratio: ratio of secondary peak to primary (SB2 indicator)
    """
    # Find peak
    peak_idx = np.argmax(ccf)
    peak_v = velocities[peak_idx]
    peak_height = ccf[peak_idx]

    # Select region around peak for fitting
    mask = np.abs(velocities - peak_v) < fit_width
    v_fit = velocities[mask]
    ccf_fit = ccf[mask]

    # Gaussian function
    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset

    try:
        # Initial guess
        p0 = [peak_height - np.min(ccf_fit), peak_v, 20, np.min(ccf_fit)]

        # Fit
        popt, pcov = curve_fit(gaussian, v_fit, ccf_fit, p0=p0, maxfev=5000)

        rv = popt[1]
        rv_err = np.sqrt(pcov[1, 1]) if np.isfinite(pcov[1, 1]) else np.abs(popt[2]) / np.sqrt(peak_height * len(v_fit))
        fwhm = 2.355 * np.abs(popt[2])

    except Exception as e:
        # Fallback to parabola fit
        if len(v_fit) >= 3:
            coeffs = np.polyfit(v_fit, ccf_fit, 2)
            rv = -coeffs[1] / (2 * coeffs[0])
            rv_err = 5.0  # Conservative estimate
            fwhm = 50.0
        else:
            rv = peak_v
            rv_err = 10.0
            fwhm = 50.0

    # Find secondary peaks (SB2 detection)
    # Smooth CCF and find local maxima
    ccf_smooth = gaussian_filter1d(ccf, 5)
    secondary_ratio = 0.0

    # Mask primary peak region
    primary_mask = np.abs(velocities - rv) < fwhm
    ccf_masked = ccf_smooth.copy()
    ccf_masked[primary_mask] = 0

    if np.max(ccf_masked) > 0:
        secondary_ratio = np.max(ccf_masked) / peak_height

    return {
        'rv': rv,
        'rv_err': rv_err,
        'peak_height': peak_height,
        'fwhm': fwhm,
        'secondary_ratio': secondary_ratio
    }

def analyze_wavelength_region(wl1, fl1_norm, wl2, fl2_norm, wl_min, wl_max, region_name):
    """Analyze a specific wavelength region."""
    # Mask to region
    mask1 = (wl1 >= wl_min) & (wl1 <= wl_max) & np.isfinite(fl1_norm)
    mask2 = (wl2 >= wl_min) & (wl2 <= wl_max) & np.isfinite(fl2_norm)

    if np.sum(mask1) < 100 or np.sum(mask2) < 100:
        return None

    # Cross-correlate
    velocities, ccf = cross_correlate_spectra(
        wl1[mask1], fl1_norm[mask1],
        wl2[mask2], fl2_norm[mask2],
        v_min=-300, v_max=300, v_step=0.5
    )

    if len(velocities) < 10:
        return None

    # Fit peak
    peak_info = fit_ccf_peak(velocities, ccf)
    peak_info['region'] = region_name
    peak_info['wl_min'] = wl_min
    peak_info['wl_max'] = wl_max
    peak_info['velocities'] = velocities.tolist()
    peak_info['ccf'] = ccf.tolist()

    return peak_info

def main():
    print("=" * 70)
    print("FORENSIC V6: LAMOST RV RE-MEASUREMENT")
    print("=" * 70)

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # File paths
    fits1 = FITS_DIR / f"lamost_{OBSID_1}.fits"
    fits2 = FITS_DIR / f"lamost_{OBSID_2}.fits"

    # Verify files exist
    if not fits1.exists() or not fits2.exists():
        print(f"ERROR: Missing FITS files")
        print(f"  Expected: {fits1}")
        print(f"  Expected: {fits2}")
        return

    # Compute file hashes
    hash1 = sha256_file(fits1)
    hash2 = sha256_file(fits2)
    print(f"\nFile hashes:")
    print(f"  {OBSID_1}: {hash1}")
    print(f"  {OBSID_2}: {hash2}")

    # Load spectra
    print("\nLoading spectra...")
    wl1, fl1, info1 = load_lamost_spectrum(fits1)
    wl2, fl2, info2 = load_lamost_spectrum(fits2)

    print(f"\nEpoch 1 (ObsID {OBSID_1}):")
    print(f"  Date: {info1['date_obs']}")
    print(f"  MJD: {info1['mjd']}")
    print(f"  Catalog HELIO_RV: {info1['helio_rv']:.2f} km/s")
    print(f"  SNR_i: {info1['snr_i']:.1f}")
    print(f"  Subclass: {info1['subclass']}")
    print(f"  Wavelength range: {np.nanmin(wl1):.1f} - {np.nanmax(wl1):.1f} A")

    print(f"\nEpoch 2 (ObsID {OBSID_2}):")
    print(f"  Date: {info2['date_obs']}")
    print(f"  MJD: {info2['mjd']}")
    print(f"  Catalog HELIO_RV: {info2['helio_rv']:.2f} km/s")
    print(f"  SNR_i: {info2['snr_i']:.1f}")
    print(f"  Subclass: {info2['subclass']}")
    print(f"  Wavelength range: {np.nanmin(wl2):.1f} - {np.nanmax(wl2):.1f} A")

    # Continuum normalize
    print("\nContinuum normalizing...")
    fl1_norm = continuum_normalize(wl1, fl1)
    fl2_norm = continuum_normalize(wl2, fl2)

    # Define wavelength masks for M-dwarf RV measurement
    # Focus on TiO bands and other M-dwarf features
    wavelength_regions = [
        (6200, 6800, "Red-TiO (6200-6800 A)"),      # TiO bands
        (7000, 7500, "Far-Red (7000-7500 A)"),       # TiO + telluric-free regions
        (7600, 8200, "I-band TiO (7600-8200 A)"),   # Strong TiO
        (8400, 8800, "Ca II triplet (8400-8800 A)"), # Ca II triplet
        (5800, 8800, "Full red (5800-8800 A)"),     # Full red coverage
    ]

    # Analyze each region
    print("\n" + "=" * 70)
    print("CROSS-CORRELATION ANALYSIS")
    print("=" * 70)
    print("\nNote: Positive ΔRV means epoch 2 is redshifted relative to epoch 1")
    print("      ΔRV = RV(epoch2) - RV(epoch1) in the self-template framework")

    region_results = []

    for wl_min, wl_max, region_name in wavelength_regions:
        result = analyze_wavelength_region(wl1, fl1_norm, wl2, fl2_norm,
                                          wl_min, wl_max, region_name)
        if result:
            region_results.append(result)
            print(f"\n{region_name}:")
            print(f"  ΔRV = {result['rv']:.2f} ± {result['rv_err']:.2f} km/s")
            print(f"  CCF peak height: {result['peak_height']:.3f}")
            print(f"  CCF FWHM: {result['fwhm']:.1f} km/s")
            print(f"  Secondary peak ratio: {result['secondary_ratio']:.3f}")

    # Compute weighted average ΔRV from all regions
    if len(region_results) > 0:
        rvs = np.array([r['rv'] for r in region_results])
        errs = np.array([r['rv_err'] for r in region_results])
        weights = 1 / errs**2

        delta_rv_weighted = np.sum(weights * rvs) / np.sum(weights)
        delta_rv_err = 1 / np.sqrt(np.sum(weights))

        # Also compute simple mean and std for consistency check
        delta_rv_mean = np.mean(rvs)
        delta_rv_scatter = np.std(rvs)

        print("\n" + "=" * 70)
        print("COMBINED RESULTS")
        print("=" * 70)
        print(f"\nWeighted average ΔRV: {delta_rv_weighted:.2f} ± {delta_rv_err:.2f} km/s")
        print(f"Simple mean ΔRV: {delta_rv_mean:.2f} km/s")
        print(f"Region-to-region scatter: {delta_rv_scatter:.2f} km/s")

        # The self-template CCF measures RV(epoch1) - RV(epoch2) when epoch1 is template
        # Positive ΔRV means epoch 2 is blueshifted relative to epoch 1
        # So actual RV change = -ΔRV
        # Wait, let me think about this more carefully:
        # When we cross-correlate fl1 (template) against fl2 (shifted),
        # and find peak at +V, it means fl2 is shifted by +V relative to fl1
        # So RV(fl2) - RV(fl1) = +V (the CCF peak)

        # Compare to catalog values
        catalog_delta_rv = info2['helio_rv'] - info1['helio_rv']
        print(f"\nCatalog HELIO_RV difference: {catalog_delta_rv:.2f} km/s")
        print(f"  Epoch 1: {info1['helio_rv']:.2f} km/s")
        print(f"  Epoch 2: {info2['helio_rv']:.2f} km/s")

        # Significance
        significance = np.abs(delta_rv_weighted) / delta_rv_err
        print(f"\nSignificance of ΔRV: {significance:.1f} sigma")

        # Check for consistency with v5 claim
        v5_claim_delta_rv = -49.36 - (-29.23)  # = -20.13 km/s (epoch 1 more negative)
        # But wait, v5 said epoch2 HELIO_RV = -29.23, epoch1 = -49.36
        # So ΔRV = epoch2 - epoch1 = -29.23 - (-49.36) = +20.13 km/s

        print(f"\nv5 claim: ΔRV = +20.1 km/s (epoch 2 less negative than epoch 1)")
        print(f"Our measurement: ΔRV = {delta_rv_weighted:.2f} ± {delta_rv_err:.2f} km/s")

        # Determine verdict
        if np.abs(delta_rv_weighted) > 10 and significance > 3:
            verdict = "PASS"
            verdict_reason = f"RV variability confirmed at {significance:.1f}σ"
        elif significance > 2:
            verdict = "PASS"
            verdict_reason = f"RV variability detected at {significance:.1f}σ"
        elif significance > 1:
            verdict = "INCONCLUSIVE"
            verdict_reason = f"Marginal detection at {significance:.1f}σ"
        else:
            verdict = "FAIL"
            verdict_reason = f"No significant variability ({significance:.1f}σ)"

        print(f"\n{'='*70}")
        print(f"VERDICT: {verdict}")
        print(f"Reason: {verdict_reason}")
        print(f"{'='*70}")

    else:
        delta_rv_weighted = np.nan
        delta_rv_err = np.nan
        significance = np.nan
        verdict = "FAIL"
        verdict_reason = "No valid wavelength regions for analysis"

    # Save results
    results = {
        'method': 'Self-template cross-correlation',
        'description': 'Cross-correlate epoch 1 vs epoch 2 to measure relative RV shift',
        'file_hashes': {
            f'lamost_{OBSID_1}.fits': hash1,
            f'lamost_{OBSID_2}.fits': hash2
        },
        'epoch1': {
            'obsid': int(OBSID_1),
            'date_obs': info1['date_obs'],
            'mjd': float(info1['mjd']),
            'catalog_helio_rv_kms': float(info1['helio_rv']),
            'snr_g': float(info1['snr_g']),
            'snr_r': float(info1['snr_r']),
            'snr_i': float(info1['snr_i']),
            'subclass': info1['subclass']
        },
        'epoch2': {
            'obsid': int(OBSID_2),
            'date_obs': info2['date_obs'],
            'mjd': float(info2['mjd']),
            'catalog_helio_rv_kms': float(info2['helio_rv']),
            'snr_g': float(info2['snr_g']),
            'snr_r': float(info2['snr_r']),
            'snr_i': float(info2['snr_i']),
            'subclass': info2['subclass']
        },
        'wavelength_regions': [
            {
                'region': r['region'],
                'wl_range_angstrom': [r['wl_min'], r['wl_max']],
                'delta_rv_kms': r['rv'],
                'delta_rv_err_kms': r['rv_err'],
                'ccf_peak_height': r['peak_height'],
                'ccf_fwhm_kms': r['fwhm'],
                'secondary_peak_ratio': r['secondary_ratio']
            }
            for r in region_results
        ],
        'combined_result': {
            'delta_rv_weighted_kms': float(delta_rv_weighted) if np.isfinite(delta_rv_weighted) else None,
            'delta_rv_err_kms': float(delta_rv_err) if np.isfinite(delta_rv_err) else None,
            'delta_rv_simple_mean_kms': float(delta_rv_mean) if 'delta_rv_mean' in dir() and np.isfinite(delta_rv_mean) else None,
            'region_scatter_kms': float(delta_rv_scatter) if 'delta_rv_scatter' in dir() and np.isfinite(delta_rv_scatter) else None,
            'significance_sigma': float(significance) if np.isfinite(significance) else None,
            'catalog_delta_rv_kms': float(info2['helio_rv'] - info1['helio_rv'])
        },
        'verdict': verdict,
        'verdict_reason': verdict_reason,
        'v5_comparison': {
            'v5_claimed_delta_rv_kms': 20.13,
            'v5_claimed_significance': 4.5,
            'our_delta_rv_kms': float(delta_rv_weighted) if np.isfinite(delta_rv_weighted) else None,
            'our_significance': float(significance) if np.isfinite(significance) else None
        }
    }

    # Save JSON
    output_json = OUTPUT_DIR / "lamost_rv_refit_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_json}")

    # Create figures
    print("\nGenerating figures...")

    # Figure 1: CCF diagnostics
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    axes1 = axes1.flatten()

    for i, r in enumerate(region_results[:6]):
        ax = axes1[i] if i < len(axes1) else None
        if ax is None:
            break
        velocities = np.array(r['velocities'])
        ccf = np.array(r['ccf'])
        ax.plot(velocities, ccf, 'b-', lw=1.5)
        ax.axvline(r['rv'], color='r', ls='--', label=f"ΔRV = {r['rv']:.1f} km/s")
        ax.axhline(r['peak_height'] * r['secondary_ratio'], color='orange', ls=':',
                   alpha=0.7, label=f"Secondary ratio: {r['secondary_ratio']:.2f}")
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('CCF')
        ax.set_title(r['region'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(region_results), len(axes1)):
        axes1[i].set_visible(False)

    fig1.suptitle(f'CCF Diagnostics: ObsID {OBSID_1} vs {OBSID_2}\n'
                  f'Weighted ΔRV = {delta_rv_weighted:.2f} ± {delta_rv_err:.2f} km/s ({significance:.1f}σ)',
                  fontsize=12)
    plt.tight_layout()
    fig1.savefig(FIG_DIR / 'lamost_ccf_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'lamost_ccf_diagnostics.png'}")
    plt.close(fig1)

    # Figure 2: Spectrum overlay
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: Full spectrum overlay
    ax = axes2[0]
    ax.plot(wl1, fl1_norm, 'b-', lw=0.5, alpha=0.7, label=f'Epoch 1 (ObsID {OBSID_1})')
    ax.plot(wl2, fl2_norm, 'r-', lw=0.5, alpha=0.7, label=f'Epoch 2 (ObsID {OBSID_2})')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Full Spectrum Overlay (Continuum Normalized)')
    ax.legend()
    ax.set_xlim(5500, 9000)
    ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.3)

    # Mark key M-dwarf features
    features = [
        (6563, 'Hα'),
        (6867, 'TiO'),
        (7054, 'TiO'),
        (7589, 'TiO'),
        (8183, 'Na I'),
        (8498, 'Ca II'),
        (8542, 'Ca II'),
        (8662, 'Ca II'),
    ]
    for wl, name in features:
        if 5500 < wl < 9000:
            ax.axvline(wl, color='green', ls=':', alpha=0.5, lw=0.5)
            ax.text(wl, 1.9, name, fontsize=7, ha='center', rotation=90)

    # Panel 2: Zoom on TiO region
    ax = axes2[1]
    mask1 = (wl1 > 7000) & (wl1 < 7700)
    mask2 = (wl2 > 7000) & (wl2 < 7700)
    ax.plot(wl1[mask1], fl1_norm[mask1], 'b-', lw=0.8, alpha=0.8, label=f'Epoch 1')
    ax.plot(wl2[mask2], fl2_norm[mask2], 'r-', lw=0.8, alpha=0.8, label=f'Epoch 2')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Zoom: TiO Band Region (7000-7700 Å)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Residual (epoch2 - epoch1 after shifting)
    ax = axes2[2]
    # Interpolate epoch 2 onto epoch 1 wavelength grid
    common_mask = (wl1 > 6000) & (wl1 < 8500)
    wl_common = wl1[common_mask]
    fl1_c = fl1_norm[common_mask]
    interp2 = interp1d(wl2, fl2_norm, bounds_error=False, fill_value=np.nan)
    fl2_c = interp2(wl_common)

    residual = fl2_c - fl1_c
    ax.plot(wl_common, residual, 'k-', lw=0.5, alpha=0.7)
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Residual (Epoch2 - Epoch1)')
    ax.set_title('Spectral Residual (not velocity-shifted)')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)

    fig2.suptitle(f'LAMOST Spectrum Comparison\n'
                  f'Epoch 1: {info1["date_obs"]} | Epoch 2: {info2["date_obs"]}\n'
                  f'Measured ΔRV = {delta_rv_weighted:.1f} km/s', fontsize=11)
    plt.tight_layout()
    fig2.savefig(FIG_DIR / 'lamost_epoch_overlay.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'lamost_epoch_overlay.png'}")
    plt.close(fig2)

    print("\n" + "=" * 70)
    print("LAMOST RV RE-MEASUREMENT COMPLETE")
    print("=" * 70)

    return results

if __name__ == '__main__':
    results = main()
