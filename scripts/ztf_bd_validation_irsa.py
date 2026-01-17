#!/usr/bin/env python3
"""
ZTF photometry validation for E4 Brown Dwarf candidates.
Uses IRSA direct query for ZTF light curves (not ALeRCE).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import warnings
warnings.filterwarnings('ignore')
import os
import json
from datetime import datetime
from io import StringIO
import pandas as pd

OUTPUT_DIR = '/home/primary/DESI-BH-CANDIDATE-SEARCH/detailed_reports_e4'

TARGETS = [
    {
        'desi_targetid': 39627793289448405,
        'gaia_id': 3833554806157884928,
        'ra': 150.370905,
        'dec': 0.183262,
        'period': 0.5,
        'gmag': 16.4,
        'name': 'BD_Candidate_1',
    },
    {
        'desi_targetid': 39627785240582464,
        'gaia_id': 2507569101891941376,
        'ra': 30.691535,
        'dec': -0.000485,
        'period': 0.7,
        'gmag': 17.2,
        'name': 'BD_Candidate_2',
    },
    {
        'desi_targetid': 39627743385616469,
        'gaia_id': 3251124869651250304,
        'ra': 55.751012,
        'dec': -1.852596,
        'period': 1.1,
        'gmag': 18.8,
        'name': 'BD_Candidate_3',
    },
    {
        'desi_targetid': 39627842371195432,
        'gaia_id': 3691992718441759104,
        'ra': 195.961529,
        'dec': 2.206382,
        'period': 0.5,
        'gmag': 16.4,
        'name': 'BD_Candidate_4',
    },
]


def query_ztf_irsa(ra, dec, radius_arcsec=5.0):
    """Query ZTF light curves directly from IRSA."""

    # IRSA ZTF Light Curve Service
    url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

    params = {
        'POS': f'CIRCLE {ra} {dec} {radius_arcsec/3600}',
        'FORMAT': 'CSV',
        'BAD_CATFLAGS_MASK': '32768',
    }

    print(f"  Querying IRSA ZTF at RA={ra:.4f}, Dec={dec:.4f}...")

    try:
        response = requests.get(url, params=params, timeout=120)

        if response.status_code != 200:
            print(f"  IRSA query failed: HTTP {response.status_code}")
            return None

        # Check if we got data
        if len(response.text) < 200:
            print(f"  No ZTF data in IRSA response")
            return None

        # Parse CSV
        try:
            data = pd.read_csv(StringIO(response.text))
        except Exception as e:
            print(f"  Failed to parse IRSA CSV: {e}")
            return None

        if len(data) == 0:
            print(f"  No ZTF observations found")
            return None

        print(f"  Found {len(data)} ZTF observations")

        # Check columns
        print(f"  Columns: {list(data.columns)}")

        return data

    except Exception as e:
        print(f"  IRSA query error: {e}")
        return None


def process_irsa_data(data):
    """Process IRSA ZTF data into time, mag, err arrays by filter."""

    if data is None or len(data) == 0:
        return {}, {}

    # IRSA ZTF columns typically include:
    # mjd, mag, magerr, filtercode (zg, zr, zi), catflags

    # Find the relevant columns
    time_col = None
    mag_col = None
    err_col = None
    filter_col = None

    for col in data.columns:
        col_lower = col.lower()
        if 'mjd' in col_lower or col_lower == 'hjd':
            time_col = col
        elif col_lower in ['mag', 'psfmag']:
            mag_col = col
        elif col_lower in ['magerr', 'psfmagerr', 'e_mag']:
            err_col = col
        elif 'filter' in col_lower or col_lower == 'filtercode':
            filter_col = col

    if time_col is None:
        # Try to find any column with time-like values
        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32]:
                vals = data[col].dropna()
                if len(vals) > 0 and 50000 < vals.mean() < 70000:
                    time_col = col
                    break

    print(f"  Using columns: time={time_col}, mag={mag_col}, err={err_col}, filter={filter_col}")

    if time_col is None or mag_col is None:
        print(f"  Could not identify required columns")
        return {}, {}

    results = {}
    filter_counts = {}

    # Get unique filters
    if filter_col:
        filters = data[filter_col].unique()
        print(f"  Filters found: {filters}")

        for filt in filters:
            mask = data[filter_col] == filt
            subset = data[mask]

            times = subset[time_col].values
            mags = subset[mag_col].values

            if err_col and err_col in subset.columns:
                errs = subset[err_col].values
            else:
                errs = np.full_like(mags, 0.05)

            # Clean data - remove NaN and bad values
            valid = np.isfinite(times) & np.isfinite(mags) & (mags > 0) & (mags < 30)

            results[filt] = {
                'time': times[valid],
                'mag': mags[valid],
                'err': errs[valid] if errs is not None else None
            }
            filter_counts[filt] = np.sum(valid)
    else:
        # No filter column - assume all one band
        times = data[time_col].values
        mags = data[mag_col].values
        errs = data[err_col].values if err_col else np.full_like(mags, 0.05)

        valid = np.isfinite(times) & np.isfinite(mags) & (mags > 0) & (mags < 30)

        results['all'] = {
            'time': times[valid],
            'mag': mags[valid],
            'err': errs[valid]
        }
        filter_counts['all'] = np.sum(valid)

    print(f"  Filter counts: {filter_counts}")
    return results, filter_counts


def fold_lightcurve(time, mag, period):
    """Fold light curve at period."""
    if len(time) == 0:
        return np.array([]), np.array([])

    phase = (time / period) % 1.0
    sort_idx = np.argsort(phase)
    return phase[sort_idx], mag[sort_idx]


def compute_amplitude(mags, percentile=5):
    """Compute robust amplitude using percentiles."""
    if len(mags) < 10:
        return np.nan

    low = np.nanpercentile(mags, percentile)
    high = np.nanpercentile(mags, 100 - percentile)
    return high - low


def fit_sinusoid(phase, mag):
    """Fit a simple sinusoid to phased light curve."""
    if len(phase) < 20:
        return None

    try:
        from scipy.optimize import curve_fit

        def sinusoid(x, A, phi, offset):
            return A * np.sin(2 * np.pi * x + phi) + offset

        p0 = [0.1, 0, np.median(mag)]
        bounds = ([0, -np.pi, np.min(mag)-1], [1, np.pi, np.max(mag)+1])

        popt, pcov = curve_fit(sinusoid, phase, mag, p0=p0, bounds=bounds, maxfev=5000)

        return {
            'amplitude': abs(popt[0]),
            'phase_offset': popt[1],
            'mean_mag': popt[2],
            'residual_std': np.std(mag - sinusoid(phase, *popt))
        }
    except:
        return None


def analyze_lightcurve(lc_data, period):
    """Analyze ZTF light curve for variability."""

    results = {
        'classification': 'Unknown',
        'confidence': 'Low',
        'notes': [],
        'per_filter': {}
    }

    total_points = 0
    max_amplitude = 0

    for filt, data in lc_data.items():
        time = data['time']
        mag = data['mag']
        n_pts = len(mag)
        total_points += n_pts

        if n_pts < 5:
            continue

        # Fold at period
        phase, folded = fold_lightcurve(time, mag, period)

        # Compute amplitude
        amp = compute_amplitude(mag)
        scatter = np.std(mag)

        # Fit sinusoid
        sin_fit = fit_sinusoid(phase, mag)

        results['per_filter'][filt] = {
            'n_points': n_pts,
            'amplitude': float(amp) if np.isfinite(amp) else None,
            'scatter': float(scatter),
            'sin_fit': sin_fit,
            'median_mag': float(np.median(mag)),
            'phase': phase.tolist(),
            'folded_mag': folded.tolist()
        }

        if np.isfinite(amp) and amp > max_amplitude:
            max_amplitude = amp

    results['total_points'] = total_points
    results['max_amplitude'] = max_amplitude

    # Classification logic
    if total_points < 20:
        results['classification'] = 'INSUFFICIENT DATA'
        results['confidence'] = 'N/A'
        results['notes'].append(f'Only {total_points} total ZTF points')
        return results

    # Check amplitude for contact binary signature
    if max_amplitude > 0.5:
        results['classification'] = 'FALSE POSITIVE - Contact Binary'
        results['confidence'] = 'High'
        results['notes'].append(f'Large amplitude variation: {max_amplitude:.2f} mag')
        results['notes'].append('W UMa-type contact binary signature')
        results['notes'].append('REJECT: This is a stellar binary, not a dark companion')

    elif max_amplitude > 0.2:
        results['classification'] = 'LIKELY FALSE POSITIVE - Eclipsing Binary'
        results['confidence'] = 'Medium-High'
        results['notes'].append(f'Significant amplitude: {max_amplitude:.2f} mag')
        results['notes'].append('Deep eclipses suggest stellar companion')
        results['notes'].append('Likely REJECT: Probably stellar binary')

    elif max_amplitude > 0.1:
        results['classification'] = 'POSSIBLE VARIABLE - Needs Review'
        results['confidence'] = 'Medium'
        results['notes'].append(f'Moderate amplitude: {max_amplitude:.2f} mag')
        results['notes'].append('Could be ellipsoidal variation, spots, or shallow eclipses')
        results['notes'].append('REVIEW: May still be valid if variation is ellipsoidal')

    elif max_amplitude > 0.03:
        results['classification'] = 'CANDIDATE - Possible Ellipsoidal'
        results['confidence'] = 'Medium'
        results['notes'].append(f'Small amplitude: {max_amplitude:.3f} mag')
        results['notes'].append('Could be ellipsoidal variation from tidal distortion')
        results['notes'].append('VALID with caution: Consistent with BD companion causing tidal distortion')

    else:
        results['classification'] = 'VALID CANDIDATE - Dark/Non-transiting'
        results['confidence'] = 'High'
        results['notes'].append(f'Flat light curve (amplitude: {max_amplitude:.3f} mag)')
        results['notes'].append('No eclipse or contact binary signal detected')
        results['notes'].append('VALID: Companion is dark or system not edge-on')

    # Warning for 1-day aliases
    if 0.45 < period < 0.55 or 0.95 < period < 1.05:
        results['notes'].append(f'CAUTION: Period ({period}d) near 1-day alias')

    return results


def create_ztf_plot(target, lc_data, analysis, output_path):
    """Create comprehensive ZTF validation plot."""

    n_filters = len(lc_data)
    if n_filters == 0:
        return

    fig = plt.figure(figsize=(16, 12))

    # Color map for filters
    filter_colors = {'zg': 'green', 'zr': 'red', 'zi': 'darkred',
                    'g': 'green', 'r': 'red', 'i': 'darkred', 'all': 'blue'}

    # 1. Raw light curve (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    for filt, data in lc_data.items():
        color = filter_colors.get(filt, 'blue')
        ax1.scatter(data['time'], data['mag'], s=8, alpha=0.5,
                   c=color, label=f'{filt} ({len(data["time"])} pts)')
    ax1.invert_yaxis()
    ax1.set_xlabel('MJD')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f"ZTF Light Curve - {target['name']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Phase-folded (top right) - all filters together
    ax2 = fig.add_subplot(2, 2, 2)
    for filt, data in lc_data.items():
        if len(data['time']) < 5:
            continue
        phase, folded = fold_lightcurve(data['time'], data['mag'], target['period'])
        color = filter_colors.get(filt, 'blue')
        ax2.scatter(phase, folded, s=10, alpha=0.4, c=color, label=filt)
        ax2.scatter(phase + 1, folded, s=10, alpha=0.2, c=color)
    ax2.invert_yaxis()
    ax2.set_xlabel('Orbital Phase')
    ax2.set_ylabel('Magnitude')
    ax2.set_title(f"Phase-Folded (P = {target['period']:.3f} days)")
    ax2.set_xlim(0, 2)
    ax2.axvline(1, color='gray', linestyle='--', alpha=0.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Binned phase curve (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    for filt, data in lc_data.items():
        if len(data['time']) < 20:
            continue
        phase, folded = fold_lightcurve(data['time'], data['mag'], target['period'])

        # Bin the data
        n_bins = 20
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_mags = []
        bin_errs = []

        for i in range(n_bins):
            mask = (phase >= bins[i]) & (phase < bins[i+1])
            if np.sum(mask) > 0:
                bin_mags.append(np.median(folded[mask]))
                bin_errs.append(np.std(folded[mask]) / np.sqrt(np.sum(mask)))
            else:
                bin_mags.append(np.nan)
                bin_errs.append(np.nan)

        color = filter_colors.get(filt, 'blue')
        ax3.errorbar(bin_centers, bin_mags, yerr=bin_errs, fmt='o-',
                    color=color, markersize=6, capsize=3, label=filt)
        ax3.errorbar(bin_centers + 1, bin_mags, yerr=bin_errs, fmt='o-',
                    color=color, markersize=6, capsize=3, alpha=0.5)

    ax3.invert_yaxis()
    ax3.set_xlabel('Orbital Phase')
    ax3.set_ylabel('Binned Magnitude')
    ax3.set_title(f"Binned Phase Curve (20 bins)")
    ax3.set_xlim(0, 2)
    ax3.axvline(1, color='gray', linestyle='--', alpha=0.3)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Classification summary (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Build summary text
    filter_summary = []
    for filt, info in analysis.get('per_filter', {}).items():
        amp_str = f"{info['amplitude']:.3f}" if info['amplitude'] else "N/A"
        filter_summary.append(f"  {filt}: {info['n_points']} pts, amp={amp_str} mag")

    text = f"""
TARGET: {target['name']}
DESI TargetID: {target['desi_targetid']}
Gaia Source ID: {target['gaia_id']}
RA, Dec: {target['ra']:.6f}, {target['dec']:.6f}
Orbital Period: {target['period']} days
Gaia G mag: {target['gmag']}

ZTF DATA SUMMARY:
  Total observations: {analysis.get('total_points', 0)}
{chr(10).join(filter_summary)}
  Max amplitude: {analysis.get('max_amplitude', 0):.3f} mag

CLASSIFICATION: {analysis['classification']}
Confidence: {analysis['confidence']}

NOTES:
"""
    for note in analysis.get('notes', []):
        text += f"  • {note}\n"

    ax4.text(0.02, 0.98, text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("ZTF PHOTOMETRY VALIDATION - E4 BROWN DWARF CANDIDATES")
    print("Using IRSA Direct Query (Real Data)")
    print("=" * 70)
    print(f"Generated: {datetime.now().isoformat()}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"Processing: {target['name']}")
        print(f"  DESI TargetID: {target['desi_targetid']}")
        print(f"  Period: {target['period']} days, G={target['gmag']}")
        print(f"  RA: {target['ra']:.6f}, Dec: {target['dec']:.6f}")
        print(f"{'='*60}")

        result = {
            'target': target,
            'ztf_data_found': False,
            'raw_data': None,
            'analysis': None
        }

        # Query IRSA directly
        raw_data = query_ztf_irsa(target['ra'], target['dec'], radius_arcsec=5.0)

        if raw_data is not None and len(raw_data) > 0:
            result['ztf_data_found'] = True

            # Process into per-filter arrays
            lc_data, filter_counts = process_irsa_data(raw_data)

            if len(lc_data) > 0:
                # Analyze
                analysis = analyze_lightcurve(lc_data, target['period'])
                result['analysis'] = analysis

                print(f"\n  CLASSIFICATION: {analysis['classification']}")
                print(f"  Confidence: {analysis['confidence']}")
                print(f"  Max amplitude: {analysis.get('max_amplitude', 0):.3f} mag")
                for note in analysis.get('notes', []):
                    print(f"    • {note}")

                # Create plot
                plot_path = os.path.join(OUTPUT_DIR, f"{target['desi_targetid']}_ztf_irsa_validation.png")
                create_ztf_plot(target, lc_data, analysis, plot_path)
            else:
                result['analysis'] = {
                    'classification': 'DATA PARSE ERROR',
                    'confidence': 'N/A',
                    'notes': ['Could not parse IRSA data into usable format']
                }
        else:
            print(f"  No ZTF data found in IRSA")
            result['analysis'] = {
                'classification': 'NO ZTF DATA',
                'confidence': 'N/A',
                'notes': ['Target not in ZTF footprint or too faint']
            }

        all_results.append(result)

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)

    report = []
    report.append("# E4 Brown Dwarf ZTF Validation Report (IRSA Direct Query)")
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append("\n## Data Source")
    report.append("- **Source**: IRSA ZTF Light Curve Service (direct query)")
    report.append("- **Data**: Real ZTF photometry, no simulations")
    report.append("")
    report.append("## Summary Table")
    report.append("")
    report.append("| Target | Period | G mag | ZTF Points | Max Amplitude | Classification | Verdict |")
    report.append("|--------|--------|-------|------------|---------------|----------------|---------|")

    for r in all_results:
        target = r['target']
        analysis = r['analysis']

        if r['ztf_data_found']:
            n_pts = analysis.get('total_points', 0)
            amp = analysis.get('max_amplitude', 0)
            amp_str = f"{amp:.3f} mag"
        else:
            n_pts = 0
            amp_str = "N/A"

        classification = analysis['classification']

        if 'FALSE POSITIVE' in classification:
            verdict = "**REJECT**"
        elif 'LIKELY FALSE' in classification:
            verdict = "**LIKELY REJECT**"
        elif 'NO ZTF DATA' in classification or 'INSUFFICIENT' in classification:
            verdict = "INCONCLUSIVE"
        elif 'VALID CANDIDATE' in classification:
            verdict = "**VALID**"
        elif 'Ellipsoidal' in classification:
            verdict = "VALID (caution)"
        elif 'REVIEW' in classification:
            verdict = "NEEDS REVIEW"
        else:
            verdict = "REVIEW"

        report.append(f"| {target['desi_targetid']} | {target['period']}d | {target['gmag']} | {n_pts} | {amp_str} | {classification} | {verdict} |")

    report.append("")
    report.append("## Classification Criteria")
    report.append("")
    report.append("| Amplitude | Classification | Interpretation |")
    report.append("|-----------|---------------|----------------|")
    report.append("| > 0.5 mag | FALSE POSITIVE - Contact Binary | W UMa type, stellar binary |")
    report.append("| 0.2 - 0.5 mag | LIKELY FALSE POSITIVE | Deep eclipses, stellar companion |")
    report.append("| 0.1 - 0.2 mag | POSSIBLE VARIABLE | Could be eclipsing or ellipsoidal |")
    report.append("| 0.03 - 0.1 mag | CANDIDATE - Ellipsoidal | Tidal distortion consistent with BD |")
    report.append("| < 0.03 mag | VALID CANDIDATE - Dark | No variability, dark companion |")
    report.append("")
    report.append("## Detailed Results")
    report.append("")

    for r in all_results:
        target = r['target']
        analysis = r['analysis']

        report.append(f"### {target['name']} ({target['desi_targetid']})")
        report.append("")
        report.append(f"- **Gaia Source ID**: {target['gaia_id']}")
        report.append(f"- **Coordinates**: RA={target['ra']:.6f}, Dec={target['dec']:.6f}")
        report.append(f"- **Orbital Period**: {target['period']} days")
        report.append(f"- **Gaia G magnitude**: {target['gmag']}")
        report.append(f"- **ZTF Data Found**: {'Yes' if r['ztf_data_found'] else 'No'}")

        if r['ztf_data_found']:
            report.append(f"- **Total ZTF Points**: {analysis.get('total_points', 0)}")
            report.append(f"- **Maximum Amplitude**: {analysis.get('max_amplitude', 0):.3f} mag")

            if 'per_filter' in analysis:
                report.append("\n**Per-Filter Summary:**")
                for filt, info in analysis['per_filter'].items():
                    amp_str = f"{info['amplitude']:.3f}" if info['amplitude'] else "N/A"
                    report.append(f"- {filt}: {info['n_points']} points, amplitude={amp_str} mag, median={info['median_mag']:.2f} mag")

        report.append("")
        report.append(f"**Classification**: {analysis['classification']}")
        report.append(f"**Confidence**: {analysis['confidence']}")
        report.append("")

        if analysis.get('notes'):
            report.append("**Assessment Notes:**")
            for note in analysis['notes']:
                report.append(f"- {note}")
        report.append("")

        if r['ztf_data_found']:
            report.append(f"![ZTF Validation]({target['desi_targetid']}_ztf_irsa_validation.png)")
        report.append("")
        report.append("---")
        report.append("")

    # Overall summary
    report.append("## Overall Assessment")
    report.append("")

    valid = sum(1 for r in all_results if 'VALID' in r['analysis']['classification'])
    reject = sum(1 for r in all_results if 'FALSE POSITIVE' in r['analysis']['classification'] or 'LIKELY FALSE' in r['analysis']['classification'])
    inconclusive = sum(1 for r in all_results if 'INSUFFICIENT' in r['analysis']['classification'] or 'NO ZTF' in r['analysis']['classification'])
    review = len(all_results) - valid - reject - inconclusive

    report.append(f"- **Valid Candidates**: {valid}")
    report.append(f"- **Rejected (False Positives)**: {reject}")
    report.append(f"- **Inconclusive (insufficient data)**: {inconclusive}")
    report.append(f"- **Needs Review**: {review}")
    report.append("")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'ZTF_VALIDATION_REPORT_IRSA.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Report saved: {report_path}")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, 'ztf_validation_results_irsa.json')
    with open(json_path, 'w') as f:
        serializable = []
        for r in all_results:
            sr = {
                'desi_targetid': r['target']['desi_targetid'],
                'gaia_id': r['target']['gaia_id'],
                'ra': r['target']['ra'],
                'dec': r['target']['dec'],
                'period': r['target']['period'],
                'ztf_data_found': r['ztf_data_found'],
                'total_points': r['analysis'].get('total_points', 0) if r['analysis'] else 0,
                'max_amplitude': r['analysis'].get('max_amplitude', 0) if r['analysis'] else 0,
                'classification': r['analysis']['classification'],
                'confidence': r['analysis']['confidence'],
                'notes': r['analysis'].get('notes', [])
            }
            serializable.append(sr)
        json.dump(serializable, f, indent=2)
    print(f"JSON saved: {json_path}")

    print("\n" + "=" * 70)
    print("ZTF VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
