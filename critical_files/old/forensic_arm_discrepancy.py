#!/usr/bin/env python3
"""
Forensic analysis of DESI R-Z arm RV discrepancy.

Investigates why the 'r' and 'z' arms disagree by ~67 km/s through:
  Method A: Template mismatch check (CCF with M/K/G dwarf templates)
  Method B: Sky line & noise diagnostics

Target-specific: searches for TARGETID in FIBERMAP and analyzes that fiber only.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Target DESI ID for the BH candidate
TARGET_ID = 39627745210139276

# Placeholder - update with actual file path
DESI_SPECTRUM_FILE = "data/desi_spectra/coadd_XXXXX.fits"

# Speed of light
C_KMS = 299792.458

# Worst sky region in Z-arm (for masked SNR calculation)
SKY_WORST_REGION = (9000, 9800)  # Angstroms

# Known bright OH sky emission lines in the Z-arm range (nm -> Angstrom)
# These are the strongest OH Meinel bands
OH_SKY_LINES_ANGSTROM = [
    # Strong OH lines in Z-arm (7500-9800 A)
    7794.1, 7808.5, 7821.5, 7841.3, 7855.4,  # 7800 A region
    7913.7, 7932.3, 7948.1, 7964.7, 7993.3,  # 7950 A region
    8344.6, 8399.2, 8415.2, 8430.2, 8451.0,  # 8400 A region
    8504.6, 8542.1, 8634.6, 8648.5, 8667.4,  # 8600 A region
    8761.1, 8778.7, 8793.3, 8827.1, 8838.5,  # 8800 A region
    8885.8, 8903.1, 8919.6, 8943.4, 8958.1,  # 8900 A region
    9001.9, 9038.1, 9052.5, 9089.8,          # 9000 A region
    9275.5, 9310.3, 9339.0, 9375.9, 9419.7,  # 9300 A region
    9439.7, 9476.2, 9505.6, 9521.8, 9555.0,  # 9500 A region
]


# =============================================================================
# Synthetic Template Generation
# =============================================================================

def generate_synthetic_template(wave, teff, logg=4.5):
    """
    Generate a simple synthetic stellar template using a blackbody continuum
    plus absorption features characteristic of the spectral type.

    For real analysis, use PHOENIX models from data/templates/ if available.
    """
    # Blackbody continuum (Planck function in wavelength)
    h = 6.626e-34
    c = 3e8
    k = 1.38e-23
    wave_m = wave * 1e-10  # Angstrom to meters

    with np.errstate(over='ignore', divide='ignore'):
        bb = (2 * h * c**2 / wave_m**5) / (np.exp(h * c / (wave_m * k * teff)) - 1)
    bb = bb / np.nanmax(bb)  # Normalize

    # Add synthetic absorption features based on Teff
    template = bb.copy()

    # Common stellar absorption lines (approximate positions in Angstrom)
    absorption_lines = {
        # Ca II triplet (strong in K/M dwarfs)
        'CaII_8498': (8498, 15, 0.3 if teff < 5000 else 0.15),
        'CaII_8542': (8542, 20, 0.4 if teff < 5000 else 0.2),
        'CaII_8662': (8662, 18, 0.35 if teff < 5000 else 0.18),
        # Na I doublet
        'NaI_8183': (8183, 8, 0.2 if teff < 4500 else 0.1),
        'NaI_8195': (8195, 8, 0.2 if teff < 4500 else 0.1),
        # Fe I lines
        'FeI_8688': (8688, 5, 0.15),
        'FeI_8824': (8824, 5, 0.12),
        # TiO bands (strong in M dwarfs)
        'TiO_7050': (7050, 100, 0.4 if teff < 4000 else 0.05),
        'TiO_7600': (7600, 80, 0.3 if teff < 4000 else 0.03),
        'TiO_8450': (8450, 60, 0.25 if teff < 4000 else 0.02),
        # H-alpha (weak absorption in cool dwarfs)
        'Halpha': (6563, 10, 0.1 if teff > 5000 else 0.05),
    }

    for name, (center, width, depth) in absorption_lines.items():
        gaussian = depth * np.exp(-0.5 * ((wave - center) / width)**2)
        template = template * (1 - gaussian)

    # Smooth slightly
    template = gaussian_filter1d(template, 2)

    return template


def load_phoenix_template(teff, logg=4.5, template_dir="data/templates"):
    """
    Attempt to load a PHOENIX template if available.
    Returns (wave, flux) or (None, None) if not found.
    """
    import glob
    pattern = f"{template_dir}/phoenix_{teff}*.fits"
    matches = glob.glob(pattern)

    if not matches:
        # Try alternate naming
        pattern = f"{template_dir}/*{teff}*.fits"
        matches = glob.glob(pattern)

    if matches:
        try:
            with fits.open(matches[0]) as hdul:
                flux = hdul[0].data
                # Try to get wavelength
                wave_file = f"{template_dir}/WAVE_PHOENIX.fits"
                try:
                    with fits.open(wave_file) as whdul:
                        wave = whdul[0].data
                    return wave, flux
                except:
                    pass
        except:
            pass

    return None, None


# =============================================================================
# CCF Analysis
# =============================================================================

def compute_ccf(wave, flux, ivar, template_wave, template_flux, rv_range=(-500, 500), rv_step=1):
    """
    Compute the Cross-Correlation Function between observed spectrum and template.

    Returns:
        rv_grid: array of RV values in km/s
        ccf: cross-correlation values
        rv_peak: RV at CCF peak
        ccf_peak: CCF peak value
    """
    # Create RV grid
    rv_grid = np.arange(rv_range[0], rv_range[1] + rv_step, rv_step)
    ccf = np.zeros(len(rv_grid))

    # Interpolate template onto observed wavelength grid
    template_interp = interp1d(template_wave, template_flux, kind='linear',
                                bounds_error=False, fill_value=0)

    # Normalize observed flux
    good = (ivar > 0) & np.isfinite(flux)
    if good.sum() < 100:
        return rv_grid, ccf, np.nan, np.nan

    flux_norm = flux.copy()
    flux_norm[~good] = 0
    continuum = gaussian_filter1d(np.where(good, flux, np.nan), 50)
    continuum = np.nan_to_num(continuum, nan=np.nanmedian(flux[good]))
    continuum[continuum <= 0] = np.nanmedian(flux[good])
    flux_norm = flux_norm / continuum - 1
    flux_norm[~good] = 0

    # Compute CCF at each RV
    for i, rv in enumerate(rv_grid):
        # Doppler shift: wave_shifted = wave * (1 + rv/c)
        doppler = 1 + rv / C_KMS
        wave_shifted = wave / doppler

        # Get template at shifted wavelengths
        template_shifted = template_interp(wave_shifted)

        # Normalize template similarly
        tmpl_good = template_shifted > 0
        if tmpl_good.sum() < 100:
            continue
        tmpl_continuum = gaussian_filter1d(template_shifted, 50)
        tmpl_continuum[tmpl_continuum <= 0] = np.nanmedian(template_shifted[tmpl_good])
        template_norm = template_shifted / tmpl_continuum - 1

        # Weight by inverse variance
        weights = np.sqrt(ivar)
        weights[~good] = 0

        # Cross-correlation (weighted dot product)
        ccf[i] = np.sum(weights * flux_norm * template_norm) / np.sqrt(
            np.sum(weights * flux_norm**2) * np.sum(weights * template_norm**2) + 1e-10
        )

    # Find peak
    if np.all(ccf == 0):
        return rv_grid, ccf, np.nan, np.nan

    peak_idx = np.argmax(ccf)
    rv_peak = rv_grid[peak_idx]
    ccf_peak = ccf[peak_idx]

    # Refine peak with parabolic fit
    if 1 < peak_idx < len(rv_grid) - 2:
        x = rv_grid[peak_idx-1:peak_idx+2]
        y = ccf[peak_idx-1:peak_idx+2]
        if len(x) == 3 and np.all(np.isfinite(y)):
            # Parabolic interpolation
            denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
            if abs(denom) > 1e-10:
                a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
                b = (x[2]**2 * (y[0] - y[1]) + x[1]**2 * (y[2] - y[0]) + x[0]**2 * (y[1] - y[2])) / denom
                if abs(a) > 1e-10:
                    rv_peak = -b / (2 * a)

    return rv_grid, ccf, rv_peak, ccf_peak


def run_ccf_analysis(wave, flux, ivar, arm_name, templates):
    """
    Run CCF against multiple templates for a single arm.

    Returns dict of results.
    """
    results = {}

    for teff, (template_wave, template_flux) in templates.items():
        rv_grid, ccf, rv_peak, ccf_peak = compute_ccf(
            wave, flux, ivar, template_wave, template_flux
        )
        results[teff] = {
            'rv_grid': rv_grid,
            'ccf': ccf,
            'rv_peak': rv_peak,
            'ccf_peak': ccf_peak
        }
        print(f"  {arm_name} arm + {teff}K template: RV = {rv_peak:+7.1f} km/s (CCF peak = {ccf_peak:.3f})")

    return results


# =============================================================================
# SNR and Sky Line Analysis
# =============================================================================

def calculate_snr(flux, ivar, wave=None, mask_regions=None):
    """
    Calculate median Signal-to-Noise Ratio.

    Parameters:
        flux: flux array
        ivar: inverse variance array
        wave: wavelength array (optional, needed for masking)
        mask_regions: list of (min, max) wavelength tuples to exclude

    Returns:
        SNR value
    """
    good = (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)

    # Apply wavelength masks if provided
    if wave is not None and mask_regions is not None:
        for wmin, wmax in mask_regions:
            good = good & ~((wave >= wmin) & (wave <= wmax))

    if good.sum() < 10:
        return np.nan

    signal = np.median(flux[good])
    noise = np.median(1 / np.sqrt(ivar[good]))

    return signal / noise if noise > 0 else np.nan


def identify_sky_contamination(wave, flux, ivar, sky_lines, window=5):
    """
    Check for residual sky contamination near known OH lines.

    Returns fraction of pixels near sky lines that show excess variance.
    """
    contaminated = 0
    total = 0

    good = (ivar > 0) & np.isfinite(flux)
    median_flux = np.median(flux[good]) if good.sum() > 0 else 1

    for sky_wave in sky_lines:
        # Find pixels within window of sky line
        mask = np.abs(wave - sky_wave) < window
        if mask.sum() == 0:
            continue

        total += 1

        # Check for anomalous flux or variance
        local_flux = flux[mask]
        local_ivar = ivar[mask]

        # Flag if variance is much higher than expected or flux is anomalous
        if np.any(local_ivar < 0.1 * np.median(ivar[good])):
            contaminated += 1
        elif np.any(np.abs(local_flux - median_flux) > 5 * median_flux):
            contaminated += 1

    return contaminated / total if total > 0 else 0


# =============================================================================
# DESI Spectrum Loading
# =============================================================================

def load_desi_coadd(filename, target_id=None):
    """
    Load a DESI coadd spectrum for a specific target.

    Parameters:
        filename: path to DESI coadd FITS file
        target_id: DESI TARGETID to search for (if None, uses first row)

    Returns dict with keys 'b', 'r', 'z', each containing:
        wave, flux, ivar, mask
    Also returns 'row_index' and 'fibermap_row' if target found.
    """
    data = {}
    row_index = 0  # Default to first row

    with fits.open(filename) as hdul:
        # Print available extensions for debugging
        print(f"Available HDUs: {[h.name for h in hdul]}")

        # Search for target in FIBERMAP
        if target_id is not None:
            try:
                fibermap = hdul['FIBERMAP'].data
                targetids = fibermap['TARGETID']

                # Find matching row
                matches = np.where(targetids == target_id)[0]

                if len(matches) == 0:
                    # Print error in red and exit
                    print(f"\n\033[91mERROR: Target {target_id} not found in file!\033[0m")
                    print(f"File contains {len(targetids)} targets.")
                    print(f"TARGETID range: {targetids.min()} to {targetids.max()}")
                    sys.exit(1)

                row_index = matches[0]
                print(f"\n\033[92mFound TARGETID {target_id} at row {row_index}\033[0m")

                # Store fibermap info
                data['fibermap'] = {col: fibermap[col][row_index] for col in fibermap.dtype.names}
                data['row_index'] = row_index

            except KeyError:
                print("Warning: No FIBERMAP extension found, using row 0")

        for arm in ['B', 'R', 'Z']:
            try:
                wave = hdul[f'{arm}_WAVELENGTH'].data
                flux = hdul[f'{arm}_FLUX'].data
                ivar = hdul[f'{arm}_IVAR'].data

                # Handle 2D arrays - extract specific row
                if flux.ndim == 2:
                    if row_index < flux.shape[0]:
                        flux = flux[row_index]
                        ivar = ivar[row_index]
                    else:
                        print(f"Warning: row_index {row_index} out of range for {arm} arm")
                        flux = flux[0]
                        ivar = ivar[0]

                # Try to get mask
                try:
                    mask = hdul[f'{arm}_MASK'].data
                    if mask.ndim == 2:
                        mask = mask[row_index] if row_index < mask.shape[0] else mask[0]
                except:
                    mask = np.zeros_like(flux, dtype=int)

                data[arm.lower()] = {
                    'wave': wave,
                    'flux': flux,
                    'ivar': ivar,
                    'mask': mask
                }
                print(f"Loaded {arm} arm: {len(wave)} pixels, "
                      f"wave range {wave.min():.1f}-{wave.max():.1f} A")

            except KeyError as e:
                print(f"Warning: Could not load {arm} arm: {e}")

    return data


# =============================================================================
# Main Analysis & Plotting
# =============================================================================

def run_forensic_analysis(spectrum_file, target_id=TARGET_ID):
    """
    Run full forensic analysis on DESI spectrum for a specific target.
    """
    print("=" * 70)
    print("FORENSIC ANALYSIS: R-Z Arm RV Discrepancy")
    print("=" * 70)
    print(f"\nTarget: DESI TARGETID {target_id}")
    print(f"File: {spectrum_file}\n")

    # Load spectrum for specific target
    try:
        spec = load_desi_coadd(spectrum_file, target_id=target_id)
    except FileNotFoundError:
        print(f"\033[91mERROR: File not found: {spectrum_file}\033[0m")
        print("Please update DESI_SPECTRUM_FILE with the correct path.")
        return None
    except SystemExit:
        # Target not found - error already printed
        return None

    if 'r' not in spec or 'z' not in spec:
        print("\033[91mERROR: Missing R or Z arm data\033[0m")
        return None

    # Build wavelength grid for templates (cover full DESI range)
    full_wave = np.arange(3600, 9900, 0.5)

    # Generate/load templates
    print("\n" + "-" * 70)
    print("METHOD A: Template Mismatch Check")
    print("-" * 70)

    templates = {}
    template_configs = [
        (3500, "M-dwarf"),
        (4500, "K-dwarf"),
        (5500, "G-dwarf"),
    ]

    for teff, name in template_configs:
        # Try PHOENIX first
        phoenix_wave, phoenix_flux = load_phoenix_template(teff)
        if phoenix_wave is not None:
            print(f"Using PHOENIX template for {name} ({teff}K)")
            templates[teff] = (phoenix_wave, phoenix_flux)
        else:
            print(f"Using synthetic template for {name} ({teff}K)")
            template_flux = generate_synthetic_template(full_wave, teff)
            templates[teff] = (full_wave, template_flux)

    # Run CCF analysis for R and Z arms
    print("\nCCF Results:")
    print("-" * 40)

    r_results = run_ccf_analysis(
        spec['r']['wave'], spec['r']['flux'], spec['r']['ivar'],
        'R', templates
    )

    print()

    z_results = run_ccf_analysis(
        spec['z']['wave'], spec['z']['flux'], spec['z']['ivar'],
        'Z', templates
    )

    # Compute R-Z discrepancy for each template
    print("\nR-Z Discrepancy by Template:")
    print("-" * 40)
    for teff, name in template_configs:
        rv_r = r_results[teff]['rv_peak']
        rv_z = z_results[teff]['rv_peak']
        delta = rv_r - rv_z
        print(f"  {name:8s} ({teff}K): ΔRV(R-Z) = {delta:+7.1f} km/s")

    # Method B: Sky Line & Noise Diagnostics
    print("\n" + "-" * 70)
    print("METHOD B: Sky Line & Noise Diagnostics")
    print("-" * 70)

    # Calculate SNR for R arm (no masking needed)
    snr_r = calculate_snr(spec['r']['flux'], spec['r']['ivar'])

    # Calculate SNR for Z arm - both raw and with sky region masked
    snr_z_raw = calculate_snr(spec['z']['flux'], spec['z']['ivar'])
    snr_z_masked = calculate_snr(
        spec['z']['flux'], spec['z']['ivar'],
        wave=spec['z']['wave'],
        mask_regions=[SKY_WORST_REGION]
    )

    print(f"\nSignal-to-Noise Ratio:")
    print(f"  R arm:                    SNR = {snr_r:.1f}")
    print(f"  Z arm (full):             SNR = {snr_z_raw:.1f}")
    print(f"  Z arm (masked {SKY_WORST_REGION[0]}-{SKY_WORST_REGION[1]}A): SNR = {snr_z_masked:.1f}")
    if snr_z_raw > 0:
        print(f"  Ratio R/Z(full):          {snr_r/snr_z_raw:.2f}")
        print(f"  Ratio R/Z(masked):        {snr_r/snr_z_masked:.2f}" if snr_z_masked > 0 else "")

    # Check if the "high SNR" was dominated by sky
    if snr_z_raw > snr_z_masked * 1.5:
        print(f"\n  \033[93mWARNING: Z-arm SNR drops {(1 - snr_z_masked/snr_z_raw)*100:.0f}% when sky region masked!\033[0m")
        print(f"  \033[93mThe 'high SNR' may be dominated by sky noise, not stellar signal.\033[0m")

    # Use masked SNR for downstream analysis
    snr_z = snr_z_masked

    # Sky contamination check
    sky_contam = identify_sky_contamination(
        spec['z']['wave'], spec['z']['flux'], spec['z']['ivar'],
        OH_SKY_LINES_ANGSTROM
    )
    print(f"\nSky Line Contamination (Z arm):")
    print(f"  Fraction of OH lines with residuals: {sky_contam*100:.1f}%")

    # Generate diagnostic plot
    print("\n" + "-" * 70)
    print("Generating diagnostic plot...")
    print("-" * 70)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[1, 1])
    fig.suptitle(f'TARGETID {target_id}', fontsize=10, y=0.995)

    # Panel 1: CCF peaks for different templates
    ax1 = axes[0]
    colors = {'3500': 'red', '4500': 'orange', '5500': 'gold'}
    linestyles = {'r': '-', 'z': '--'}

    for teff, name in template_configs:
        # R arm CCF
        rv_grid = r_results[teff]['rv_grid']
        ccf_r = r_results[teff]['ccf']
        ccf_r_norm = ccf_r / np.max(ccf_r) if np.max(ccf_r) > 0 else ccf_r
        ax1.plot(rv_grid, ccf_r_norm, color=colors[str(teff)], linestyle='-',
                 label=f'{name} ({teff}K) R-arm', alpha=0.8)

        # Z arm CCF
        ccf_z = z_results[teff]['ccf']
        ccf_z_norm = ccf_z / np.max(ccf_z) if np.max(ccf_z) > 0 else ccf_z
        ax1.plot(rv_grid, ccf_z_norm, color=colors[str(teff)], linestyle='--',
                 label=f'{name} ({teff}K) Z-arm', alpha=0.8)

        # Mark peaks
        ax1.axvline(r_results[teff]['rv_peak'], color=colors[str(teff)],
                    linestyle='-', alpha=0.3, linewidth=2)
        ax1.axvline(z_results[teff]['rv_peak'], color=colors[str(teff)],
                    linestyle='--', alpha=0.3, linewidth=2)

    ax1.axvline(-86, color='black', linestyle=':', label='Pipeline RV (-86 km/s)')
    ax1.set_xlabel('Radial Velocity [km/s]')
    ax1.set_ylabel('Normalized CCF')
    ax1.set_title('Method A: CCF with Different Temperature Templates')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.set_xlim(-300, 200)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Z-arm spectrum with sky lines marked
    ax2 = axes[1]

    wave_z = spec['z']['wave']
    flux_z = spec['z']['flux']
    ivar_z = spec['z']['ivar']

    # Plot flux
    good = (ivar_z > 0) & np.isfinite(flux_z)
    ax2.plot(wave_z[good], flux_z[good], 'k-', linewidth=0.5, alpha=0.8,
             label='Z-arm flux')

    # Mark OH sky lines
    ymin, ymax = ax2.get_ylim()
    flux_median = np.median(flux_z[good])
    flux_std = np.std(flux_z[good])

    for i, sky_wave in enumerate(OH_SKY_LINES_ANGSTROM):
        if wave_z.min() <= sky_wave <= wave_z.max():
            ax2.axvline(sky_wave, color='red', alpha=0.3, linewidth=1,
                        label='OH sky line' if i == 0 else '')

    # Highlight regions with potential sky contamination
    ax2.fill_between(wave_z, flux_median - 3*flux_std, flux_median + 3*flux_std,
                     alpha=0.1, color='blue', label='±3σ region')

    ax2.set_xlabel('Wavelength [Å]')
    ax2.set_ylabel('Flux')
    ax2.set_title(f'Method B: Z-arm Spectrum with OH Sky Lines (SNR full={snr_z_raw:.1f}, masked={snr_z_masked:.1f})')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add text annotation with key results
    textstr = (f'R-arm SNR: {snr_r:.1f}\n'
               f'Z-arm SNR (full): {snr_z_raw:.1f}\n'
               f'Z-arm SNR (masked): {snr_z_masked:.1f}\n'
               f'Sky contamination: {sky_contam*100:.0f}%')
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Highlight the masked sky region
    ax2.axvspan(SKY_WORST_REGION[0], SKY_WORST_REGION[1], alpha=0.15, color='red',
                label=f'Masked region ({SKY_WORST_REGION[0]}-{SKY_WORST_REGION[1]}Å)')

    plt.tight_layout()

    output_file = 'forensic_arm_discrepancy.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_file}")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find best template (smallest R-Z discrepancy)
    best_teff = None
    best_delta = np.inf
    for teff, _ in template_configs:
        delta = abs(r_results[teff]['rv_peak'] - z_results[teff]['rv_peak'])
        if delta < best_delta:
            best_delta = delta
            best_teff = teff

    print(f"\n1. Template Mismatch:")
    print(f"   Best agreement with {best_teff}K template (ΔRV = {best_delta:.1f} km/s)")

    print(f"\n2. SNR Assessment:")
    if snr_z_masked < 10:
        print(f"   WARNING: Z-arm SNR ({snr_z_masked:.1f} masked) is LOW - RV may be unreliable")
    elif snr_z_masked < snr_r / 2:
        print(f"   CAUTION: Z-arm SNR significantly lower than R-arm")
    else:
        print(f"   SNR appears adequate in both arms")

    if snr_z_raw > snr_z_masked * 1.3:
        print(f"   NOTE: Z-arm 'high SNR' ({snr_z_raw:.1f}) drops to {snr_z_masked:.1f} when sky masked")
        print(f"         This suggests sky contamination is inflating the apparent SNR")

    print(f"\n3. Sky Contamination:")
    if sky_contam > 0.3:
        print(f"   WARNING: High sky line residuals ({sky_contam*100:.0f}%) - Z-arm may be compromised")
    else:
        print(f"   Sky residuals appear acceptable ({sky_contam*100:.0f}%)")

    return {
        'r_results': r_results,
        'z_results': z_results,
        'snr_r': snr_r,
        'snr_z_raw': snr_z_raw,
        'snr_z_masked': snr_z_masked,
        'sky_contamination': sky_contam,
        'target_id': target_id
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    # Allow command-line override of spectrum file
    if len(sys.argv) > 1:
        spectrum_file = sys.argv[1]
    else:
        spectrum_file = DESI_SPECTRUM_FILE

    results = run_forensic_analysis(spectrum_file)
