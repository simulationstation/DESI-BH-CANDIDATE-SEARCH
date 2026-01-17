#!/usr/bin/env python3
"""
ZTF photometry validation for E4 Brown Dwarf candidates.
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


def query_ztf_lightcurve(ra, dec, radius_arcsec=2.0):
    """Query ZTF light curve from IRSA."""

    # ZTF light curve service
    url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

    params = {
        'POS': f'CIRCLE {ra} {dec} {radius_arcsec/3600}',
        'FORMAT': 'CSV',
        'BAD_CATFLAGS_MASK': '32768',  # Exclude bad data
    }

    try:
        print(f"  Querying ZTF at RA={ra:.4f}, Dec={dec:.4f}...")
        response = requests.get(url, params=params, timeout=60)

        if response.status_code != 200:
            print(f"  ZTF query failed: HTTP {response.status_code}")
            return None

        # Parse CSV
        import pandas as pd
        data = pd.read_csv(StringIO(response.text))

        if len(data) == 0:
            print(f"  No ZTF data found")
            return None

        print(f"  Found {len(data)} ZTF observations")
        return data

    except Exception as e:
        print(f"  ZTF query error: {e}")
        return None


def query_ztf_via_astroquery(ra, dec):
    """Alternative: Query ZTF via astroquery/Vizier."""
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        print(f"  Querying ZTF via Vizier...")

        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

        # ZTF DR catalog
        v = Vizier(columns=['*'], row_limit=-1)
        result = v.query_region(coord, radius=5*u.arcsec, catalog='II/371/dr19')  # ZTF DR19

        if result is None or len(result) == 0:
            # Try the light curve catalog
            result = v.query_region(coord, radius=5*u.arcsec, catalog='II/371')

        if result is None or len(result) == 0:
            print(f"  No ZTF data in Vizier")
            return None

        print(f"  Found ZTF data in Vizier")
        return result[0].to_pandas() if len(result) > 0 else None

    except Exception as e:
        print(f"  Vizier query error: {e}")
        return None


def get_ztf_forced_photometry(ra, dec):
    """Query ZTF forced photometry service."""
    try:
        # Use the ZTF Forced Photometry Service API
        url = f"https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi"

        # This requires authentication, so let's try the public light curve service instead
        lc_url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{ra}+{dec}+0.0014&FORMAT=CSV"

        response = requests.get(lc_url, timeout=60)

        if response.status_code == 200 and len(response.text) > 100:
            import pandas as pd
            data = pd.read_csv(StringIO(response.text))
            if len(data) > 0:
                return data

        return None
    except Exception as e:
        print(f"  Forced phot error: {e}")
        return None


def query_ztf_alerce(ra, dec, radius=2.0):
    """Query ZTF data from ALeRCE broker."""
    try:
        # ALeRCE cone search
        url = "https://api.alerce.online/ztf/v1/objects"

        params = {
            'ra': ra,
            'dec': dec,
            'radius': radius,  # arcsec
            'format': 'json'
        }

        print(f"  Querying ALeRCE...")
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if 'items' in data and len(data['items']) > 0:
                oid = data['items'][0]['oid']
                print(f"  Found ZTF object: {oid}")

                # Get light curve
                lc_url = f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve"
                lc_response = requests.get(lc_url, timeout=30)

                if lc_response.status_code == 200:
                    lc_data = lc_response.json()
                    return lc_data, oid

        return None, None

    except Exception as e:
        print(f"  ALeRCE error: {e}")
        return None, None


def process_ztf_lightcurve(lc_data, oid=None):
    """Process ZTF light curve data from ALeRCE format."""

    if lc_data is None:
        return None, None, None, None

    times_g, mags_g, errs_g = [], [], []
    times_r, mags_r, errs_r = [], [], []

    detections = lc_data.get('detections', [])

    for det in detections:
        if det.get('mag') is None:
            continue

        mjd = det.get('mjd')
        mag = det.get('mag')
        err = det.get('e_mag', 0.1)
        fid = det.get('fid')  # 1=g, 2=r

        if fid == 1:
            times_g.append(mjd)
            mags_g.append(mag)
            errs_g.append(err)
        elif fid == 2:
            times_r.append(mjd)
            mags_r.append(mag)
            errs_r.append(err)

    return (np.array(times_g), np.array(mags_g), np.array(errs_g),
            np.array(times_r), np.array(mags_r), np.array(errs_r))


def fold_lightcurve(time, mag, period):
    """Fold light curve at period."""
    if len(time) == 0:
        return np.array([]), np.array([])

    phase = (time / period) % 1.0
    sort_idx = np.argsort(phase)
    return phase[sort_idx], mag[sort_idx]


def analyze_ztf_lightcurve(phase_g, mag_g, phase_r, mag_r, period):
    """Analyze ZTF folded light curve."""

    results = {
        'classification': 'Unknown',
        'confidence': 'Low',
        'notes': [],
        'n_g': len(phase_g),
        'n_r': len(phase_r),
    }

    # Combine bands for analysis
    if len(mag_g) > 5:
        amp_g = np.nanmax(mag_g) - np.nanmin(mag_g)
        results['amplitude_g'] = amp_g
        results['scatter_g'] = np.nanstd(mag_g)
    else:
        amp_g = 0
        results['amplitude_g'] = None

    if len(mag_r) > 5:
        amp_r = np.nanmax(mag_r) - np.nanmin(mag_r)
        results['amplitude_r'] = amp_r
        results['scatter_r'] = np.nanstd(mag_r)
    else:
        amp_r = 0
        results['amplitude_r'] = None

    amp = max(amp_g, amp_r)
    total_points = len(mag_g) + len(mag_r)

    if total_points < 10:
        results['classification'] = 'INSUFFICIENT DATA'
        results['confidence'] = 'N/A'
        results['notes'].append(f'Only {total_points} total ZTF points')
        return results

    # Check for sinusoidal variation (contact binary)
    if amp > 0.5:
        results['classification'] = 'FALSE POSITIVE - Contact Binary'
        results['confidence'] = 'High'
        results['notes'].append(f'Large amplitude: {amp:.2f} mag')
        results['notes'].append('W UMa-type contact binary signature')

    elif amp > 0.2:
        results['classification'] = 'LIKELY FALSE POSITIVE - Eclipsing Binary'
        results['confidence'] = 'Medium'
        results['notes'].append(f'Significant amplitude: {amp:.2f} mag')
        results['notes'].append('Deep eclipses suggest stellar companion')

    elif amp > 0.05:
        results['classification'] = 'POSSIBLE VARIABLE'
        results['confidence'] = 'Low'
        results['notes'].append(f'Moderate amplitude: {amp:.2f} mag')
        results['notes'].append('Could be ellipsoidal variation or spots')

    else:
        results['classification'] = 'VALID CANDIDATE - Non-transiting/Dark'
        results['confidence'] = 'Medium'
        results['notes'].append(f'Flat light curve (amplitude: {amp:.3f} mag)')
        results['notes'].append('No eclipse/contact binary signal detected')
        results['notes'].append('Companion is dark or system not edge-on')

    # Check for 1-day aliasing concern
    if 0.45 < period < 0.55 or 0.95 < period < 1.05:
        results['notes'].append(f'WARNING: Period ({period}d) near 1-day alias - interpret with caution')

    return results


def create_ztf_plot(target, times_g, mags_g, times_r, mags_r,
                   phase_g, fold_g, phase_r, fold_r, analysis, output_path):
    """Create ZTF validation plot."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw light curve
    ax1 = axes[0, 0]
    if len(times_g) > 0:
        ax1.scatter(times_g, mags_g, s=10, alpha=0.6, c='green', label=f'ZTF-g ({len(times_g)} pts)')
    if len(times_r) > 0:
        ax1.scatter(times_r, mags_r, s=10, alpha=0.6, c='red', label=f'ZTF-r ({len(times_r)} pts)')
    ax1.invert_yaxis()
    ax1.set_xlabel('MJD')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f"ZTF Light Curve - {target['name']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Phase-folded g-band
    ax2 = axes[0, 1]
    if len(phase_g) > 0:
        ax2.scatter(phase_g, fold_g, s=15, alpha=0.6, c='green')
        ax2.scatter(phase_g + 1, fold_g, s=15, alpha=0.3, c='green')
    ax2.invert_yaxis()
    ax2.set_xlabel('Orbital Phase')
    ax2.set_ylabel('ZTF-g Magnitude')
    ax2.set_title(f"Phase-Folded g-band (P = {target['period']:.2f} days)")
    ax2.set_xlim(0, 2)
    ax2.axvline(1, color='gray', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Phase-folded r-band
    ax3 = axes[1, 0]
    if len(phase_r) > 0:
        ax3.scatter(phase_r, fold_r, s=15, alpha=0.6, c='red')
        ax3.scatter(phase_r + 1, fold_r, s=15, alpha=0.3, c='red')
    ax3.invert_yaxis()
    ax3.set_xlabel('Orbital Phase')
    ax3.set_ylabel('ZTF-r Magnitude')
    ax3.set_title(f"Phase-Folded r-band (P = {target['period']:.2f} days)")
    ax3.set_xlim(0, 2)
    ax3.axvline(1, color='gray', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3)

    # Classification summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    text = f"""
TARGET: {target['name']}
DESI TargetID: {target['desi_targetid']}
Gaia Source ID: {target['gaia_id']}
RA, Dec: {target['ra']:.4f}, {target['dec']:.4f}
Period: {target['period']} days
G mag: {target['gmag']}

ZTF DATA:
  g-band points: {analysis.get('n_g', 0)}
  r-band points: {analysis.get('n_r', 0)}
  g-band amplitude: {analysis.get('amplitude_g', 'N/A')} mag
  r-band amplitude: {analysis.get('amplitude_r', 'N/A')} mag

CLASSIFICATION: {analysis['classification']}
Confidence: {analysis['confidence']}

NOTES:
"""
    for note in analysis.get('notes', []):
        text += f"  - {note}\n"

    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("ZTF PHOTOMETRY VALIDATION - E4 BROWN DWARF CANDIDATES")
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
        print(f"{'='*60}")

        result = {
            'target': target,
            'ztf_data_found': False,
            'analysis': None
        }

        # Try ALeRCE first (most reliable for light curves)
        lc_data, oid = query_ztf_alerce(target['ra'], target['dec'])

        if lc_data is not None:
            result['ztf_data_found'] = True
            result['ztf_oid'] = oid

            # Process light curve
            data = process_ztf_lightcurve(lc_data, oid)
            times_g, mags_g, errs_g, times_r, mags_r, errs_r = data

            print(f"  ZTF-g: {len(times_g)} points, ZTF-r: {len(times_r)} points")

            # Fold at period
            phase_g, fold_g = fold_lightcurve(times_g, mags_g, target['period'])
            phase_r, fold_r = fold_lightcurve(times_r, mags_r, target['period'])

            # Analyze
            analysis = analyze_ztf_lightcurve(phase_g, fold_g, phase_r, fold_r, target['period'])
            result['analysis'] = analysis

            print(f"\n  CLASSIFICATION: {analysis['classification']}")
            print(f"  Confidence: {analysis['confidence']}")
            for note in analysis.get('notes', []):
                print(f"    - {note}")

            # Create plot
            plot_path = os.path.join(OUTPUT_DIR, f"{target['desi_targetid']}_ztf_validation.png")
            create_ztf_plot(
                target, times_g, mags_g, times_r, mags_r,
                phase_g, fold_g, phase_r, fold_r, analysis, plot_path
            )

        else:
            print(f"  No ZTF data found via ALeRCE")
            result['analysis'] = {
                'classification': 'NO ZTF DATA',
                'confidence': 'N/A',
                'notes': ['Target not in ZTF footprint or too faint']
            }

        all_results.append(result)

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)

    report = []
    report.append("# E4 Brown Dwarf ZTF Validation Report")
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append("")
    report.append("## Summary Table")
    report.append("")
    report.append("| Target | Period | G mag | ZTF Data | Classification | Verdict |")
    report.append("|--------|--------|-------|----------|----------------|---------|")

    for r in all_results:
        target = r['target']
        analysis = r['analysis']

        if r['ztf_data_found']:
            ztf = f"g:{analysis.get('n_g',0)}, r:{analysis.get('n_r',0)}"
        else:
            ztf = "None"

        classification = analysis['classification']

        if 'FALSE POSITIVE' in classification:
            verdict = "REJECT"
        elif 'NO ZTF DATA' in classification or 'INSUFFICIENT' in classification:
            verdict = "INCONCLUSIVE"
        elif 'VALID CANDIDATE' in classification:
            verdict = "VALID"
        else:
            verdict = "REVIEW"

        report.append(f"| {target['desi_targetid']} | {target['period']}d | {target['gmag']} | {ztf} | {classification} | **{verdict}** |")

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
        report.append(f"- **Period**: {target['period']} days")
        report.append(f"- **G magnitude**: {target['gmag']}")
        report.append(f"- **ZTF Data**: {'Yes' if r['ztf_data_found'] else 'No'}")
        report.append("")
        report.append(f"**Classification**: {analysis['classification']}")
        report.append(f"**Confidence**: {analysis['confidence']}")
        report.append("")
        if analysis.get('notes'):
            report.append("**Notes:**")
            for note in analysis['notes']:
                report.append(f"- {note}")
        report.append("")
        if r['ztf_data_found']:
            report.append(f"![ZTF Validation]({target['desi_targetid']}_ztf_validation.png)")
        report.append("")
        report.append("---")
        report.append("")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'ZTF_VALIDATION_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Report saved: {report_path}")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, 'ztf_validation_results.json')
    with open(json_path, 'w') as f:
        serializable = []
        for r in all_results:
            sr = {
                'desi_targetid': r['target']['desi_targetid'],
                'gaia_id': r['target']['gaia_id'],
                'period': r['target']['period'],
                'ztf_data_found': r['ztf_data_found'],
                'classification': r['analysis']['classification'],
                'confidence': r['analysis']['confidence'],
                'notes': r['analysis'].get('notes', [])
            }
            serializable.append(sr)
        json.dump(serializable, f, indent=2)
    print(f"JSON saved: {json_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
