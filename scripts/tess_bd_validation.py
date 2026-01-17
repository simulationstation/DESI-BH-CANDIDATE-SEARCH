#!/usr/bin/env python3
"""
TESS photometry validation for E4 Brown Dwarf candidates.
Checks for contact binary signatures, eclipses, and transits.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')
import os
import json
from datetime import datetime

OUTPUT_DIR = '/home/primary/DESI-BH-CANDIDATE-SEARCH/detailed_reports_e4'

# Target list with Gaia IDs and coordinates
TARGETS = [
    {
        'desi_targetid': 39627793289448405,
        'gaia_id': 3833554806157884928,
        'ra': 150.370905,
        'dec': 0.183262,
        'period': 0.5,
        'name': 'BD_Candidate_1',
        'priority': 'Highest'
    },
    {
        'desi_targetid': 39627785240582464,
        'gaia_id': 2507569101891941376,
        'ra': 30.691535,
        'dec': -0.000485,
        'period': 0.7,
        'name': 'BD_Candidate_2',
        'priority': 'High'
    },
    {
        'desi_targetid': 39627743385616469,
        'gaia_id': 3251124869651250304,
        'ra': 55.751012,
        'dec': -1.852596,
        'period': 1.1,
        'name': 'BD_Candidate_3',
        'priority': 'High'
    },
    {
        'desi_targetid': 39627842371195432,
        'gaia_id': 3691992718441759104,
        'ra': 195.961529,
        'dec': 2.206382,
        'period': 0.5,
        'name': 'BD_Candidate_4',
        'priority': 'High'
    },
]


def download_tess_lightcurve(ra, dec, target_name):
    """Download TESS light curve using lightkurve."""
    try:
        import lightkurve as lk

        # Search for TESS data
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))

        print(f"  Searching TESS for RA={ra:.4f}, Dec={dec:.4f}...")
        search_result = lk.search_lightcurve(
            coord,
            mission='TESS',
            author='SPOC'
        )

        if len(search_result) == 0:
            # Try QLP if SPOC not available
            search_result = lk.search_lightcurve(
                coord,
                mission='TESS',
                author='QLP'
            )

        if len(search_result) == 0:
            print(f"  No TESS data found")
            return None, None

        print(f"  Found {len(search_result)} TESS observations")

        # Download all available light curves and stitch
        lc_collection = search_result.download_all()

        if lc_collection is None or len(lc_collection) == 0:
            return None, None

        # Stitch together
        lc = lc_collection.stitch()

        # Use PDCSAP flux if available
        if hasattr(lc, 'flux'):
            time = lc.time.value
            flux = lc.flux.value
            flux_err = lc.flux_err.value if hasattr(lc, 'flux_err') else np.ones_like(flux) * 0.001
        else:
            return None, None

        return time, flux, flux_err, len(search_result)

    except Exception as e:
        print(f"  Error downloading TESS data: {e}")
        return None, None, None, 0


def remove_outliers(time, flux, flux_err, sigma=3.0):
    """Remove outliers using sigma clipping."""
    median = np.nanmedian(flux)
    std = np.nanstd(flux)

    mask = np.abs(flux - median) < sigma * std
    mask &= np.isfinite(flux)
    mask &= np.isfinite(time)

    return time[mask], flux[mask], flux_err[mask]


def fold_lightcurve(time, flux, period, t0=None):
    """Fold light curve at given period."""
    if t0 is None:
        t0 = time[0]

    phase = ((time - t0) / period) % 1.0

    # Sort by phase
    sort_idx = np.argsort(phase)

    return phase[sort_idx], flux[sort_idx]


def bin_lightcurve(phase, flux, n_bins=100):
    """Bin the folded light curve."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    binned_flux = np.zeros(n_bins)
    binned_err = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        if np.sum(mask) > 0:
            binned_flux[i] = np.nanmedian(flux[mask])
            binned_err[i] = np.nanstd(flux[mask]) / np.sqrt(np.sum(mask))
        else:
            binned_flux[i] = np.nan
            binned_err[i] = np.nan

    return bin_centers, binned_flux, binned_err


def analyze_lightcurve(phase, flux, binned_phase, binned_flux):
    """Analyze the folded light curve for signatures."""

    results = {
        'classification': 'Unknown',
        'confidence': 'Low',
        'notes': []
    }

    # Normalize flux
    median_flux = np.nanmedian(flux)
    norm_flux = flux / median_flux
    norm_binned = binned_flux / np.nanmedian(binned_flux)

    # Calculate amplitude
    amplitude = np.nanmax(norm_binned) - np.nanmin(norm_binned)
    amplitude_pct = amplitude * 100

    results['amplitude_pct'] = amplitude_pct
    results['scatter_ppm'] = np.nanstd(norm_flux) * 1e6

    # Check for sinusoidal modulation (contact binary)
    # Fit a simple sine wave
    try:
        from scipy.optimize import curve_fit

        def sine_model(x, a, phi, c):
            return a * np.sin(2 * np.pi * x + phi) + c

        valid = np.isfinite(binned_flux)
        if np.sum(valid) > 10:
            popt, _ = curve_fit(
                sine_model,
                binned_phase[valid],
                norm_binned[valid],
                p0=[0.01, 0, 1],
                maxfev=5000
            )
            sine_amplitude = np.abs(popt[0]) * 100  # percent

            # Also check for double-wave (ellipsoidal)
            def double_sine(x, a, phi, c):
                return a * np.sin(4 * np.pi * x + phi) + c

            popt2, _ = curve_fit(
                double_sine,
                binned_phase[valid],
                norm_binned[valid],
                p0=[0.01, 0, 1],
                maxfev=5000
            )
            ellipsoidal_amplitude = np.abs(popt2[0]) * 100

            results['sine_amplitude_pct'] = sine_amplitude
            results['ellipsoidal_amplitude_pct'] = ellipsoidal_amplitude

    except Exception as e:
        results['sine_amplitude_pct'] = 0
        results['ellipsoidal_amplitude_pct'] = 0

    # Classification logic
    if amplitude_pct > 20:
        results['classification'] = 'FALSE POSITIVE - Contact Binary (W UMa)'
        results['confidence'] = 'High'
        results['notes'].append(f'Large amplitude variation: {amplitude_pct:.1f}%')
        results['notes'].append('Continuous wavy modulation indicates two stars in contact')

    elif amplitude_pct > 10:
        results['classification'] = 'LIKELY FALSE POSITIVE - Eclipsing Binary'
        results['confidence'] = 'Medium'
        results['notes'].append(f'Significant variation: {amplitude_pct:.1f}%')
        results['notes'].append('Deep eclipses suggest M-dwarf companion, not BD')

    elif amplitude_pct > 1:
        # Check if it looks sinusoidal (ellipsoidal) vs transit-like
        if results.get('ellipsoidal_amplitude_pct', 0) > 0.5:
            results['classification'] = 'POSSIBLE - Ellipsoidal Variation'
            results['confidence'] = 'Medium'
            results['notes'].append(f'Amplitude: {amplitude_pct:.1f}%')
            results['notes'].append('Ellipsoidal modulation could indicate tidal distortion')
        else:
            results['classification'] = 'POSSIBLE - Low-amplitude Signal'
            results['confidence'] = 'Low'
            results['notes'].append(f'Amplitude: {amplitude_pct:.1f}%')

    elif amplitude_pct > 0.1:
        results['classification'] = 'CANDIDATE - Possible Transit'
        results['confidence'] = 'Low'
        results['notes'].append(f'Very small amplitude: {amplitude_pct:.2f}%')
        results['notes'].append('Could be shallow BD transit - needs closer inspection')

    else:
        results['classification'] = 'NON-TRANSITING / DARK'
        results['confidence'] = 'Medium'
        results['notes'].append(f'Flat light curve (amplitude: {amplitude_pct:.3f}%)')
        results['notes'].append('No eclipse/transit detected - system likely not edge-on')
        results['notes'].append('Still a valid BD candidate if RV signal is real')

    return results


def create_validation_plot(target, time, flux, phase, folded_flux,
                          binned_phase, binned_flux, analysis, output_path):
    """Create diagnostic plot."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw light curve
    ax1 = axes[0, 0]
    ax1.scatter(time, flux/np.nanmedian(flux), s=1, alpha=0.3, c='blue')
    ax1.set_xlabel('Time (BJD)')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title(f"Raw TESS Light Curve - {target['name']}")
    ax1.grid(True, alpha=0.3)

    # Phase-folded (scatter)
    ax2 = axes[0, 1]
    ax2.scatter(phase, folded_flux/np.nanmedian(folded_flux), s=1, alpha=0.2, c='gray')
    ax2.scatter(phase + 1, folded_flux/np.nanmedian(folded_flux), s=1, alpha=0.2, c='gray')
    ax2.set_xlabel('Orbital Phase')
    ax2.set_ylabel('Normalized Flux')
    ax2.set_title(f"Phase-Folded (P = {target['period']:.2f} days)")
    ax2.set_xlim(0, 2)
    ax2.axvline(1, color='gray', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Phase-folded (binned)
    ax3 = axes[1, 0]
    norm_binned = binned_flux / np.nanmedian(binned_flux)
    ax3.errorbar(binned_phase, norm_binned, fmt='o-', color='red', markersize=4)
    ax3.errorbar(binned_phase + 1, norm_binned, fmt='o-', color='red', markersize=4, alpha=0.5)
    ax3.set_xlabel('Orbital Phase')
    ax3.set_ylabel('Normalized Flux (binned)')
    ax3.set_title('Binned Phase-Folded Light Curve')
    ax3.set_xlim(0, 2)
    ax3.axvline(1, color='gray', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3)

    # Add amplitude annotation
    amp = analysis.get('amplitude_pct', 0)
    ax3.annotate(f'Amplitude: {amp:.2f}%', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top')

    # Classification summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    text = f"""
TARGET: {target['name']}
DESI TargetID: {target['desi_targetid']}
Gaia Source ID: {target['gaia_id']}
RA, Dec: {target['ra']:.4f}, {target['dec']:.4f}
Period: {target['period']} days

CLASSIFICATION: {analysis['classification']}
Confidence: {analysis['confidence']}

Amplitude: {analysis.get('amplitude_pct', 0):.3f}%
Scatter: {analysis.get('scatter_ppm', 0):.0f} ppm

NOTES:
"""
    for note in analysis.get('notes', []):
        text += f"  - {note}\n"

    ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("TESS PHOTOMETRY VALIDATION - E4 BROWN DWARF CANDIDATES")
    print("=" * 70)
    print(f"Generated: {datetime.now().isoformat()}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"Processing: {target['name']} (Priority: {target['priority']})")
        print(f"  DESI TargetID: {target['desi_targetid']}")
        print(f"  Period: {target['period']} days")
        print(f"{'='*60}")

        result = {
            'target': target,
            'tess_data_found': False,
            'analysis': None
        }

        # Download TESS data
        data = download_tess_lightcurve(target['ra'], target['dec'], target['name'])

        if data[0] is None:
            print(f"  No TESS data available for this target")
            result['analysis'] = {
                'classification': 'NO TESS DATA',
                'confidence': 'N/A',
                'notes': ['Target not observed by TESS or no data available']
            }
            all_results.append(result)
            continue

        time, flux, flux_err, n_sectors = data
        result['tess_data_found'] = True
        result['n_sectors'] = n_sectors
        result['n_points'] = len(time)

        print(f"  Downloaded {len(time)} data points from {n_sectors} sector(s)")

        # Remove outliers
        time_clean, flux_clean, flux_err_clean = remove_outliers(time, flux, flux_err)
        print(f"  After outlier removal: {len(time_clean)} points")

        # Fold at orbital period
        phase, folded_flux = fold_lightcurve(time_clean, flux_clean, target['period'])

        # Bin the folded light curve
        binned_phase, binned_flux, binned_err = bin_lightcurve(phase, folded_flux, n_bins=50)

        # Analyze
        analysis = analyze_lightcurve(phase, folded_flux, binned_phase, binned_flux)
        result['analysis'] = analysis

        print(f"\n  CLASSIFICATION: {analysis['classification']}")
        print(f"  Confidence: {analysis['confidence']}")
        print(f"  Amplitude: {analysis.get('amplitude_pct', 0):.3f}%")
        for note in analysis.get('notes', []):
            print(f"    - {note}")

        # Create plot
        plot_path = os.path.join(OUTPUT_DIR, f"{target['desi_targetid']}_tess_validation.png")
        create_validation_plot(
            target, time_clean, flux_clean, phase, folded_flux,
            binned_phase, binned_flux, analysis, plot_path
        )

        all_results.append(result)

    # Generate summary report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    report = []
    report.append("# E4 Brown Dwarf TESS Validation Report")
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append("")
    report.append("## Summary Table")
    report.append("")
    report.append("| Target | Period | TESS Data | Classification | Amplitude | Verdict |")
    report.append("|--------|--------|-----------|----------------|-----------|---------|")

    for r in all_results:
        target = r['target']
        analysis = r['analysis']

        if r['tess_data_found']:
            tess = f"{r['n_sectors']} sectors"
            amp = f"{analysis.get('amplitude_pct', 0):.2f}%"
        else:
            tess = "None"
            amp = "N/A"

        classification = analysis['classification']

        # Determine verdict
        if 'FALSE POSITIVE' in classification:
            verdict = "REJECT"
        elif 'NO TESS DATA' in classification:
            verdict = "INCONCLUSIVE"
        elif 'NON-TRANSITING' in classification:
            verdict = "VALID (non-edge-on)"
        elif 'CANDIDATE' in classification:
            verdict = "PROMISING"
        else:
            verdict = "NEEDS REVIEW"

        report.append(f"| {target['desi_targetid']} | {target['period']}d | {tess} | {classification} | {amp} | **{verdict}** |")

    report.append("")
    report.append("## Detailed Results")
    report.append("")

    for r in all_results:
        target = r['target']
        analysis = r['analysis']

        report.append(f"### {target['name']} ({target['desi_targetid']})")
        report.append("")
        report.append(f"- **Gaia ID**: {target['gaia_id']}")
        report.append(f"- **Coordinates**: RA={target['ra']:.4f}, Dec={target['dec']:.4f}")
        report.append(f"- **Orbital Period**: {target['period']} days")
        report.append(f"- **TESS Data**: {'Yes' if r['tess_data_found'] else 'No'}")
        if r['tess_data_found']:
            report.append(f"- **Data Points**: {r['n_points']}")
        report.append("")
        report.append(f"**Classification**: {analysis['classification']}")
        report.append(f"**Confidence**: {analysis['confidence']}")
        report.append("")
        if analysis.get('notes'):
            report.append("**Notes:**")
            for note in analysis['notes']:
                report.append(f"- {note}")
        report.append("")
        if r['tess_data_found']:
            report.append(f"![TESS Validation]({target['desi_targetid']}_tess_validation.png)")
        report.append("")
        report.append("---")
        report.append("")

    report.append("## Interpretation Guide")
    report.append("")
    report.append("- **FALSE POSITIVE - Contact Binary**: Large amplitude (>20%) sinusoidal variation indicates W UMa type contact binary. REJECT.")
    report.append("- **LIKELY FALSE POSITIVE - Eclipsing Binary**: Deep eclipses (>10%) suggest stellar companion, not brown dwarf. REJECT.")
    report.append("- **NON-TRANSITING / DARK**: Flat light curve means system is not edge-on. Still a valid BD candidate if RV is real.")
    report.append("- **CANDIDATE - Possible Transit**: Small dips (<1%) could indicate BD transit. Needs detailed analysis.")
    report.append("")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'TESS_VALIDATION_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to: {report_path}")

    # Save JSON results
    json_path = os.path.join(OUTPUT_DIR, 'tess_validation_results.json')
    with open(json_path, 'w') as f:
        # Convert to serializable format
        serializable = []
        for r in all_results:
            sr = {
                'desi_targetid': r['target']['desi_targetid'],
                'gaia_id': r['target']['gaia_id'],
                'period': r['target']['period'],
                'tess_data_found': r['tess_data_found'],
                'classification': r['analysis']['classification'],
                'confidence': r['analysis']['confidence'],
                'amplitude_pct': r['analysis'].get('amplitude_pct', None),
                'notes': r['analysis'].get('notes', [])
            }
            if r['tess_data_found']:
                sr['n_sectors'] = r['n_sectors']
                sr['n_points'] = r['n_points']
            serializable.append(sr)
        json.dump(serializable, f, indent=2)

    print(f"JSON results saved to: {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
