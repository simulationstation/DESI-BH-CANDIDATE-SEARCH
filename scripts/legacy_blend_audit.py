#!/usr/bin/env python3
"""
legacy_blend_audit.py - Legacy Survey Background Blend Audit

Checks for unresolved luminous blends that could explain the elevated RUWE
instead of a dark compact companion. Uses Legacy Survey imaging and
compares with Gaia IPD metrics.

Target: Gaia DR3 3802130935635096832
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# Target coordinates
TARGET_RA = 164.5235
TARGET_DEC = -1.6602
TARGET_NAME = "Gaia DR3 3802130935635096832"


def fetch_legacy_cutouts():
    """
    Fetch cutout images from Legacy Survey.

    Uses the Legacy Survey cutout service to get g, r, z band images.
    """
    print("=" * 70)
    print("LEGACY SURVEY CUTOUT RETRIEVAL")
    print("=" * 70)
    print()

    result = {
        'status': 'unknown',
        'bands': {},
        'method': None
    }

    # Try to fetch from Legacy Survey API
    try:
        import requests
        from io import BytesIO
        from PIL import Image

        print(f"Target: {TARGET_NAME}")
        print(f"Coordinates: RA={TARGET_RA}, Dec={TARGET_DEC}")
        print()

        # Legacy Survey cutout service
        # Using DECaLS/DR10 via the cutout service
        base_url = "https://www.legacysurvey.org/viewer/cutout.fits"

        # Parameters for cutout
        pixscale = 0.262  # arcsec/pixel
        size = 64  # pixels (~ 17 arcsec)

        bands_to_fetch = ['g', 'r', 'z']
        images = {}

        for band in bands_to_fetch:
            print(f"Fetching {band}-band cutout...")

            # Try FITS cutout
            url = f"{base_url}?ra={TARGET_RA}&dec={TARGET_DEC}&layer=ls-dr10&pixscale={pixscale}&size={size}&bands={band}"

            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    # Try to read FITS data
                    from astropy.io import fits
                    hdu = fits.open(BytesIO(response.content))
                    if len(hdu) > 0 and hdu[0].data is not None:
                        images[band] = {
                            'data': hdu[0].data,
                            'header': dict(hdu[0].header),
                            'pixscale': pixscale
                        }
                        print(f"  {band}: Success ({hdu[0].data.shape})")
                        result['bands'][band] = 'success'
                    else:
                        print(f"  {band}: Empty data")
                        result['bands'][band] = 'empty'
                else:
                    print(f"  {band}: HTTP {response.status_code}")
                    result['bands'][band] = f'http_{response.status_code}'
            except Exception as e:
                print(f"  {band}: Error - {str(e)[:50]}")
                result['bands'][band] = 'error'

        if images:
            result['status'] = 'partial_success' if len(images) < 3 else 'success'
            result['images'] = images
            result['method'] = 'legacy_api'
            result['pixscale_arcsec'] = pixscale
            result['cutout_size_pix'] = size
        else:
            print("  No images retrieved via API")

    except ImportError as e:
        print(f"Required library not available: {e}")
        result['status'] = 'import_error'
        result['error'] = str(e)

    except Exception as e:
        print(f"Cutout fetch failed: {str(e)[:100]}")
        result['status'] = 'fetch_error'
        result['error'] = str(e)[:200]

    print()
    return result


def analyze_psf_shape(image_data, pixscale=0.262):
    """
    Analyze PSF shape to check for elongation or secondary peaks.

    Returns metrics on ellipticity, asymmetry, and secondary peak detection.
    """
    if image_data is None:
        return {'status': 'no_data'}

    data = np.array(image_data, dtype=float)

    # Handle NaN/inf
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if data.max() <= 0:
        return {'status': 'no_signal'}

    # Find centroid
    y_idx, x_idx = np.indices(data.shape)
    total = data.sum()

    if total <= 0:
        return {'status': 'no_signal'}

    x_cen = (data * x_idx).sum() / total
    y_cen = (data * y_idx).sum() / total

    # Compute second moments
    dx = x_idx - x_cen
    dy = y_idx - y_cen

    Ixx = (data * dx**2).sum() / total
    Iyy = (data * dy**2).sum() / total
    Ixy = (data * dx * dy).sum() / total

    # Ellipticity from moments
    # e = (Ixx - Iyy + 2i*Ixy) / (Ixx + Iyy + 2*sqrt(Ixx*Iyy - Ixy^2))
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy**2

    if det < 0:
        det = 0

    denom = trace + 2 * np.sqrt(det)
    if denom > 0:
        e1 = (Ixx - Iyy) / denom
        e2 = 2 * Ixy / denom
        ellipticity = np.sqrt(e1**2 + e2**2)
    else:
        ellipticity = 0.0

    # Position angle
    pa_rad = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
    pa_deg = np.degrees(pa_rad)

    # FWHM estimate
    sigma = np.sqrt(0.5 * trace)
    fwhm_pix = 2.355 * sigma
    fwhm_arcsec = fwhm_pix * pixscale

    # Check for secondary peaks
    # Subtract smooth model and look for residuals
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(data, sigma=2)
    residual = data - smoothed

    # Find peaks in residual above 3-sigma
    residual_std = np.std(residual)
    peaks = residual > 3 * residual_std

    # Exclude central region (within FWHM)
    dist = np.sqrt(dx**2 + dy**2)
    central_mask = dist < fwhm_pix
    secondary_peaks = peaks & ~central_mask
    n_secondary = secondary_peaks.sum()

    # Compute asymmetry
    # Rotate 180 degrees and compare
    data_rot = np.rot90(np.rot90(data))
    asymmetry = np.abs(data - data_rot).sum() / (2 * total)

    return {
        'status': 'success',
        'centroid_x': float(x_cen),
        'centroid_y': float(y_cen),
        'Ixx': float(Ixx),
        'Iyy': float(Iyy),
        'Ixy': float(Ixy),
        'ellipticity': float(ellipticity),
        'position_angle_deg': float(pa_deg),
        'fwhm_pix': float(fwhm_pix),
        'fwhm_arcsec': float(fwhm_arcsec),
        'n_secondary_peaks': int(n_secondary),
        'asymmetry': float(asymmetry),
        'residual_std': float(residual_std)
    }


def load_gaia_ipd_metrics():
    """Load Gaia IPD metrics from existing analysis."""
    try:
        with open('gaia_astrometry_details.json', 'r') as f:
            gaia = json.load(f)
        return {
            'ipd_frac_multi_peak': gaia['gaia_data']['ipd_frac_multi_peak'],
            'ipd_gof_harmonic_amplitude': gaia['gaia_data']['ipd_gof_harmonic_amplitude'],
            'ipd_frac_odd_win': gaia['gaia_data']['ipd_frac_odd_win'],
            'duplicated_source': gaia['gaia_data']['duplicated_source']
        }
    except Exception as e:
        print(f"Could not load Gaia IPD metrics: {e}")
        return None


def interpret_blend_evidence(psf_metrics, gaia_ipd):
    """
    Interpret blend evidence from imaging and Gaia metrics.

    Returns a qualitative assessment and supporting evidence.
    """
    evidence = {
        'imaging': [],
        'gaia': [],
        'overall': None
    }

    # Imaging evidence
    if psf_metrics and psf_metrics.get('status') == 'success':
        ell = psf_metrics['ellipticity']
        asym = psf_metrics['asymmetry']
        n_sec = psf_metrics['n_secondary_peaks']

        if ell < 0.1:
            evidence['imaging'].append(('ellipticity', 'low', f'e={ell:.3f} < 0.1: consistent with single PSF'))
        elif ell < 0.2:
            evidence['imaging'].append(('ellipticity', 'moderate', f'e={ell:.3f}: slightly elongated'))
        else:
            evidence['imaging'].append(('ellipticity', 'high', f'e={ell:.3f} > 0.2: significantly elongated'))

        if asym < 0.05:
            evidence['imaging'].append(('asymmetry', 'low', f'A={asym:.3f} < 0.05: symmetric'))
        elif asym < 0.1:
            evidence['imaging'].append(('asymmetry', 'moderate', f'A={asym:.3f}: slightly asymmetric'))
        else:
            evidence['imaging'].append(('asymmetry', 'high', f'A={asym:.3f} > 0.1: asymmetric'))

        if n_sec == 0:
            evidence['imaging'].append(('secondary_peaks', 'none', 'No secondary peaks detected'))
        else:
            evidence['imaging'].append(('secondary_peaks', 'present', f'{n_sec} secondary peak(s) detected'))
    else:
        evidence['imaging'].append(('status', 'unavailable', 'PSF analysis not available'))

    # Gaia IPD evidence
    if gaia_ipd:
        ipd_multi = gaia_ipd['ipd_frac_multi_peak']
        ipd_harm = gaia_ipd['ipd_gof_harmonic_amplitude']
        dup = gaia_ipd['duplicated_source']

        if ipd_multi < 5:
            evidence['gaia'].append(('ipd_frac_multi_peak', 'low', f'{ipd_multi}% < 5%: single source'))
        elif ipd_multi < 20:
            evidence['gaia'].append(('ipd_frac_multi_peak', 'borderline', f'{ipd_multi}%: borderline'))
        else:
            evidence['gaia'].append(('ipd_frac_multi_peak', 'high', f'{ipd_multi}% > 20%: likely resolved'))

        if ipd_harm < 0.1:
            evidence['gaia'].append(('ipd_gof_harmonic', 'low', f'{ipd_harm:.3f} < 0.1: no periodic pattern'))
        else:
            evidence['gaia'].append(('ipd_gof_harmonic', 'elevated', f'{ipd_harm:.3f}: periodic pattern'))

        if dup:
            evidence['gaia'].append(('duplicated_source', 'yes', 'Flagged as duplicated'))
        else:
            evidence['gaia'].append(('duplicated_source', 'no', 'Not duplicated'))
    else:
        evidence['gaia'].append(('status', 'unavailable', 'Gaia IPD metrics not available'))

    # Overall assessment
    blend_score = 0

    # Check imaging
    for metric, level, _ in evidence['imaging']:
        if level == 'high':
            blend_score += 2
        elif level == 'moderate':
            blend_score += 1
        elif level == 'present':
            blend_score += 2

    # Check Gaia
    for metric, level, _ in evidence['gaia']:
        if metric == 'ipd_frac_multi_peak':
            if level == 'high':
                blend_score += 3
            elif level == 'borderline':
                blend_score += 1
        elif metric == 'duplicated_source' and level == 'yes':
            blend_score += 3

    if blend_score >= 5:
        evidence['overall'] = 'likely_blend'
        evidence['conclusion'] = 'Evidence suggests possible blend or resolved companion'
    elif blend_score >= 2:
        evidence['overall'] = 'possible_blend'
        evidence['conclusion'] = 'Some evidence for blend; further investigation recommended'
    else:
        evidence['overall'] = 'no_evidence_for_blend'
        evidence['conclusion'] = 'No significant evidence for luminous blend'

    evidence['blend_score'] = blend_score

    return evidence


def create_blend_plot(cutout_result, psf_metrics, evidence):
    """Create diagnostic plot for blend analysis."""
    print("Creating diagnostic plot...")

    # Determine layout based on available data
    has_images = (cutout_result.get('status') in ['success', 'partial_success'] and
                  'images' in cutout_result and len(cutout_result['images']) > 0)

    if has_images:
        n_bands = len(cutout_result['images'])
        fig, axes = plt.subplots(1, n_bands + 1, figsize=(4 * (n_bands + 1), 4))
        if n_bands == 1:
            axes = [axes[0], axes[1]]

        # Plot each band
        for i, (band, img_data) in enumerate(cutout_result['images'].items()):
            ax = axes[i]
            data = img_data['data']

            # Handle 3D data (multiple bands in one)
            if data.ndim == 3:
                data = data[0]

            # Plot with log scale
            vmin = np.percentile(data[data > 0], 1) if (data > 0).any() else 0.01
            vmax = np.percentile(data, 99)

            im = ax.imshow(data, origin='lower', cmap='viridis',
                          norm=LogNorm(vmin=max(vmin, 0.01), vmax=vmax))
            ax.set_title(f'{band}-band', fontsize=12)
            ax.set_xlabel('pixels')
            ax.set_ylabel('pixels')

            # Mark center
            center = data.shape[0] // 2
            ax.axhline(center, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
            ax.axvline(center, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

            plt.colorbar(im, ax=ax, shrink=0.8)

        # Summary panel
        ax_sum = axes[-1]
    else:
        fig, ax_sum = plt.subplots(1, 1, figsize=(6, 6))

    # Summary text
    ax_sum.axis('off')
    ax_sum.set_title('Blend Analysis Summary', fontsize=12)

    text_lines = [
        f"Target: {TARGET_NAME}",
        f"RA, Dec: {TARGET_RA:.4f}, {TARGET_DEC:.4f}",
        "",
        "=== Imaging Analysis ===" if has_images else "=== Imaging Analysis (no data) ===",
    ]

    if psf_metrics and psf_metrics.get('status') == 'success':
        text_lines.extend([
            f"Ellipticity: {psf_metrics['ellipticity']:.3f}",
            f"Asymmetry: {psf_metrics['asymmetry']:.3f}",
            f"FWHM: {psf_metrics['fwhm_arcsec']:.2f}\"",
            f"Secondary peaks: {psf_metrics['n_secondary_peaks']}",
        ])
    else:
        text_lines.append("PSF metrics unavailable")

    text_lines.extend([
        "",
        "=== Gaia IPD Metrics ===",
    ])

    for metric, level, desc in evidence.get('gaia', []):
        text_lines.append(f"  {desc}")

    text_lines.extend([
        "",
        "=== Verdict ===",
        f"Blend score: {evidence.get('blend_score', 'N/A')}",
        f"Assessment: {evidence.get('overall', 'N/A').upper()}",
        "",
        evidence.get('conclusion', ''),
    ])

    ax_sum.text(0.05, 0.95, '\n'.join(text_lines),
                transform=ax_sum.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('legacy_blend_cutout.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: legacy_blend_cutout.png")


def main():
    print("=" * 70)
    print("LEGACY SURVEY BACKGROUND BLEND AUDIT")
    print("=" * 70)
    print(f"Target: {TARGET_NAME}")
    print()

    results = {
        'target': TARGET_NAME,
        'ra': TARGET_RA,
        'dec': TARGET_DEC,
        'analysis': 'background_blend_audit'
    }

    # Fetch Legacy Survey cutouts
    cutout_result = fetch_legacy_cutouts()
    results['cutout'] = {
        'status': cutout_result['status'],
        'bands': cutout_result.get('bands', {}),
        'method': cutout_result.get('method'),
        'pixscale_arcsec': cutout_result.get('pixscale_arcsec'),
        'cutout_size_pix': cutout_result.get('cutout_size_pix')
    }

    # Analyze PSF shape if images available
    psf_metrics = None
    if 'images' in cutout_result and cutout_result['images']:
        # Use r-band preferentially, then g, then z
        for band in ['r', 'g', 'z']:
            if band in cutout_result['images']:
                data = cutout_result['images'][band]['data']
                if data.ndim == 3:
                    data = data[0]
                pixscale = cutout_result.get('pixscale_arcsec', 0.262)
                psf_metrics = analyze_psf_shape(data, pixscale)
                psf_metrics['band_used'] = band
                print(f"PSF analysis performed on {band}-band")
                break

    if psf_metrics:
        results['psf_analysis'] = psf_metrics
    else:
        results['psf_analysis'] = {'status': 'no_data'}

    # Load Gaia IPD metrics
    gaia_ipd = load_gaia_ipd_metrics()
    results['gaia_ipd'] = gaia_ipd

    # Interpret blend evidence
    evidence = interpret_blend_evidence(psf_metrics, gaia_ipd)
    results['evidence'] = evidence
    results['qualitative_blend_flag'] = evidence['overall']
    results['textual_notes'] = evidence['conclusion']

    print()
    print("=" * 70)
    print("BLEND ANALYSIS SUMMARY")
    print("=" * 70)
    print()

    print("Imaging evidence:")
    for metric, level, desc in evidence['imaging']:
        print(f"  {desc}")

    print()
    print("Gaia evidence:")
    for metric, level, desc in evidence['gaia']:
        print(f"  {desc}")

    print()
    print(f"Blend score: {evidence['blend_score']}")
    print(f"Assessment: {evidence['overall'].upper()}")
    print(f"Conclusion: {evidence['conclusion']}")
    print()

    # Create plot
    create_blend_plot(cutout_result, psf_metrics, evidence)

    # Save results
    # Remove image data from JSON (too large)
    with open('legacy_blend_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved: legacy_blend_results.json")

    return results


if __name__ == "__main__":
    main()
