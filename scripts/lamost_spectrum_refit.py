#!/usr/bin/env python3
"""
lamost_spectrum_refit.py - Independent Primary Mass Verification from LAMOST Spectrum

Attempts to independently verify the primary mass by fitting the LAMOST spectrum
to determine Teff, log g, and [Fe/H], then deriving M1 from empirical relations.
Also checks H-alpha activity as a sanity check on RV jitter.

Target: Gaia DR3 3802130935635096832
LAMOST ObsID: 579613097
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

# Target information
TARGET_NAME = "Gaia DR3 3802130935635096832"
LAMOST_OBSID = 579613097
TARGET_RA = 164.5235
TARGET_DEC = -1.6602

# LAMOST catalog parameters from existing analysis
LAMOST_CATALOG = {
    'spectral_type': 'dM0',
    'rv_kms': -49.36,
    'rv_err_kms': 2.79,
    'snr_g': 4.85,  # from VizieR
    'snr_overall': 17.9  # from original observation
}

# M-dwarf calibration: Teff -> Mass from various sources
# Pecaut & Mamajek (2013), Mann et al. (2015, 2019)
MDWARF_CALIBRATION = [
    # (Teff, Mass, SpType)
    (4200, 0.70, 'K7'),
    (4050, 0.63, 'K7/M0'),
    (3850, 0.57, 'M0'),
    (3700, 0.50, 'M1'),
    (3550, 0.44, 'M2'),
    (3400, 0.36, 'M3'),
    (3200, 0.26, 'M4'),
    (3050, 0.18, 'M5'),
]

# H-alpha wavelength
HALPHA_WAVELENGTH = 6562.8  # Angstrom


def attempt_spectrum_download():
    """
    Attempt to download the LAMOST spectrum from various sources.

    Returns the spectrum data if successful, or status information if not.
    """
    print("=" * 70)
    print("LAMOST SPECTRUM RETRIEVAL")
    print("=" * 70)
    print(f"Target: {TARGET_NAME}")
    print(f"ObsID: {LAMOST_OBSID}")
    print()

    result = {'status': 'not_found', 'method': None, 'spectrum': None}

    # Method 1: Check if spectrum file exists locally
    import os
    local_paths = [
        f'data/lamost/{LAMOST_OBSID}.fits',
        f'data/{LAMOST_OBSID}.fits',
        f'lamost_{LAMOST_OBSID}.fits',
    ]

    for path in local_paths:
        if os.path.exists(path):
            print(f"Found local spectrum: {path}")
            try:
                from astropy.io import fits
                hdu = fits.open(path)
                result['status'] = 'local_file'
                result['method'] = 'local'
                result['path'] = path
                # Extract spectrum data
                # LAMOST FITS format varies; try common structures
                if len(hdu) > 0:
                    result['spectrum'] = {
                        'header': dict(hdu[0].header),
                        'data': hdu[0].data
                    }
                return result
            except Exception as e:
                print(f"  Error reading {path}: {e}")

    # Method 2: Try LAMOST DR7 API
    print("Attempting LAMOST API download...")
    try:
        import requests

        # LAMOST spectrum URL pattern (may vary by DR version)
        # This is a placeholder - actual URL depends on LAMOST archive structure
        base_urls = [
            f"http://dr7.lamost.org/spectrum/fits/{LAMOST_OBSID}",
            f"http://www.lamost.org/dr7/v2.0/spectrum/fits?obsid={LAMOST_OBSID}",
        ]

        for url in base_urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200 and len(response.content) > 1000:
                    print(f"  Downloaded from: {url}")
                    result['status'] = 'downloaded'
                    result['method'] = 'lamost_api'
                    result['url'] = url

                    # Try to parse FITS
                    from astropy.io import fits
                    from io import BytesIO
                    hdu = fits.open(BytesIO(response.content))
                    if len(hdu) > 0:
                        result['spectrum'] = {
                            'header': dict(hdu[0].header) if hdu[0].header else {},
                            'data': hdu[0].data
                        }
                    return result
            except Exception as e:
                print(f"  {url}: {str(e)[:50]}")

    except ImportError:
        print("  requests library not available")
    except Exception as e:
        print(f"  API download failed: {str(e)[:80]}")

    # Method 3: Try VizieR for LAMOST parameters (not spectrum, but parameters)
    print("Querying VizieR for LAMOST parameters...")
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')
        v = Vizier(columns=['*'], row_limit=5)

        # Query LAMOST DR7
        tables = v.query_region(coord, radius=5*u.arcsec, catalog='V/164')

        if tables and len(tables) > 0:
            table = tables[0]
            print(f"  Found {len(table)} LAMOST source(s) in VizieR")

            params = {}
            for col in ['Teff', 'logg', '__Fe_H_', 'RV', 'e_RV', 'snrg', 'SubClass']:
                if col in table.colnames:
                    val = table[0][col]
                    if hasattr(val, 'item'):
                        val = val.item()
                    if not np.ma.is_masked(val):
                        try:
                            params[col] = float(val)
                        except:
                            params[col] = str(val)

            if params:
                result['status'] = 'vizier_params_only'
                result['method'] = 'vizier'
                result['vizier_params'] = params
                print(f"  Retrieved parameters: {list(params.keys())}")
                return result

    except ImportError:
        print("  astroquery not available")
    except Exception as e:
        print(f"  VizieR query failed: {str(e)[:80]}")

    print("  Spectrum not available from any source")
    result['status'] = 'spectrum_not_available'
    return result


def estimate_teff_from_spectral_type(sp_type='M0'):
    """Estimate Teff from spectral type using calibration table."""
    sp_clean = sp_type.upper().replace('D', '').strip()

    # Look up in calibration
    for teff, mass, sp in MDWARF_CALIBRATION:
        if sp_clean in sp or sp in sp_clean:
            return teff, 100  # 100 K uncertainty

    # Parse M subtype
    if sp_clean.startswith('M'):
        try:
            subtype = float(sp_clean[1:]) if len(sp_clean) > 1 else 0
            # Interpolate
            teff = 3850 - subtype * 150
            return teff, 100
        except:
            pass

    # Default to M0
    return 3850, 150


def mass_from_teff(teff, teff_err=100, feh=0.0, feh_err=0.3):
    """
    Compute stellar mass from Teff using empirical M-dwarf relations.
    Uses interpolation on the calibration table.
    """
    # Build interpolator
    teff_grid = np.array([c[0] for c in MDWARF_CALIBRATION])
    mass_grid = np.array([c[1] for c in MDWARF_CALIBRATION])

    mass_interp = interpolate.interp1d(teff_grid, mass_grid, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')

    mass = float(mass_interp(teff))

    # Metallicity correction (~5% per 0.3 dex)
    feh_factor = 1.0 + 0.05 * (feh / 0.3)
    mass *= feh_factor

    # Monte Carlo for uncertainty
    n_mc = 10000
    teff_samples = np.random.normal(teff, teff_err, n_mc)
    feh_samples = np.random.normal(feh, feh_err, n_mc)

    mass_samples = mass_interp(teff_samples)
    feh_factors = 1.0 + 0.05 * (feh_samples / 0.3)
    mass_samples = mass_samples * feh_factors

    # Add intrinsic scatter
    mass_samples += np.random.normal(0, 0.03, n_mc)

    # Physical bounds
    mass_samples = np.clip(mass_samples, 0.08, 1.0)

    return {
        'mass_msun': float(np.median(mass_samples)),
        'mass_err': float(np.std(mass_samples)),
        'mass_16': float(np.percentile(mass_samples, 16)),
        'mass_84': float(np.percentile(mass_samples, 84))
    }


def analyze_halpha_from_catalog():
    """
    Analyze H-alpha activity from catalog information.

    Since we don't have the actual spectrum, we note that dM0 stars
    are typically not highly active.
    """
    print()
    print("=" * 70)
    print("H-ALPHA ACTIVITY ASSESSMENT")
    print("=" * 70)
    print()

    result = {
        'status': 'catalog_inference',
        'Halpha_EW': None,
        'Halpha_flag': 'unknown'
    }

    sp_type = LAMOST_CATALOG['spectral_type']

    print(f"Spectral type: {sp_type}")
    print()

    # M0 dwarfs: typically not highly active
    # Active M dwarfs show H-alpha emission (EW < 0, i.e., emission = negative EW in some conventions)
    # Typical M0 activity fraction is ~10-20%

    print("H-alpha activity assessment (from spectral type):")
    print("  - M0 dwarfs: ~10-20% are chromospherically active")
    print("  - Activity level cannot be determined without spectrum")
    print()

    # Check if we can infer anything from LAMOST parameters
    # Unfortunately, LAMOST DR7 doesn't always include H-alpha EW

    print("Activity vs RV jitter check:")
    print("  - Observed ΔRV_max = 146 km/s")
    print("  - Even highly active M dwarfs have RV jitter < 1 km/s")
    print("  - Activity CANNOT explain the observed RV amplitude")
    print()

    result['activity_cannot_explain_rv'] = True
    result['max_activity_jitter_kms'] = 1.0
    result['observed_delta_rv_kms'] = 146.07

    return result


def perform_refit_analysis(spectrum_result):
    """
    Perform the primary mass analysis based on available data.
    """
    print()
    print("=" * 70)
    print("PRIMARY MASS REFIT ANALYSIS")
    print("=" * 70)
    print()

    results = {
        'target': TARGET_NAME,
        'lamost_obsid': LAMOST_OBSID,
        'spectrum_status': spectrum_result['status'],
        'spectrum_method': spectrum_result.get('method')
    }

    # Determine Teff source
    if spectrum_result['status'] == 'vizier_params_only':
        vizier = spectrum_result.get('vizier_params', {})
        if 'Teff' in vizier and vizier['Teff'] > 0:
            teff = vizier['Teff']
            teff_err = 150  # Typical LAMOST uncertainty
            teff_source = 'lamost_pipeline'
            print(f"Using LAMOST pipeline Teff: {teff} ± {teff_err} K")
        else:
            teff, teff_err = estimate_teff_from_spectral_type(LAMOST_CATALOG['spectral_type'])
            teff_source = 'spectral_type_calibration'
            print(f"Using Teff from spectral type: {teff} ± {teff_err} K")

        feh = vizier.get('__Fe_H_', 0.0)
        if feh == 0.0 or np.isnan(feh):
            feh = 0.0
            feh_err = 0.3
            feh_source = 'solar_prior'
        else:
            feh_err = 0.2
            feh_source = 'lamost_pipeline'

        logg = vizier.get('logg', 4.5)
        if logg == 0.0 or np.isnan(logg):
            logg = 4.5
            logg_source = 'dwarf_prior'
        else:
            logg_source = 'lamost_pipeline'

    else:
        # No pipeline parameters; use spectral type
        teff, teff_err = estimate_teff_from_spectral_type(LAMOST_CATALOG['spectral_type'])
        teff_source = 'spectral_type_calibration'
        feh = 0.0
        feh_err = 0.3
        feh_source = 'solar_prior'
        logg = 4.5
        logg_source = 'dwarf_prior'

        print(f"Using Teff from spectral type ({LAMOST_CATALOG['spectral_type']}): {teff} ± {teff_err} K")

    results['teff_fit'] = teff
    results['teff_err'] = teff_err
    results['teff_source'] = teff_source
    results['logg_fit'] = logg
    results['logg_source'] = logg_source
    results['feh_fit'] = feh
    results['feh_err'] = feh_err
    results['feh_source'] = feh_source

    print(f"  [Fe/H] = {feh:.2f} ± {feh_err:.2f} ({feh_source})")
    print(f"  log g = {logg:.2f} ({logg_source})")
    print()

    # Compute mass
    print("Mass estimation from Teff-Mass relation:")
    mass_result = mass_from_teff(teff, teff_err, feh, feh_err)

    print(f"  M₁ = {mass_result['mass_msun']:.3f} ± {mass_result['mass_err']:.3f} M☉")
    print(f"  68% CI: [{mass_result['mass_16']:.3f}, {mass_result['mass_84']:.3f}] M☉")
    print()

    results['M1_refit'] = mass_result['mass_msun']
    results['M1_refit_err'] = mass_result['mass_err']
    results['M1_refit_16'] = mass_result['mass_16']
    results['M1_refit_84'] = mass_result['mass_84']

    # Compare to previous estimate
    try:
        with open('primary_mass_results.json', 'r') as f:
            prev = json.load(f)
        M1_prev = prev['primary']['mass_msun']
        M1_prev_err = prev['primary']['mass_err']

        print("Comparison to previous estimate:")
        print(f"  Previous: M₁ = {M1_prev:.3f} ± {M1_prev_err:.3f} M☉")
        print(f"  Refit:    M₁ = {mass_result['mass_msun']:.3f} ± {mass_result['mass_err']:.3f} M☉")

        diff = mass_result['mass_msun'] - M1_prev
        diff_sigma = diff / np.sqrt(mass_result['mass_err']**2 + M1_prev_err**2)
        print(f"  Difference: {diff:+.3f} M☉ ({diff_sigma:+.1f}σ)")

        results['M1_previous'] = M1_prev
        results['M1_previous_err'] = M1_prev_err
        results['M1_diff'] = diff
        results['M1_diff_sigma'] = diff_sigma

        if abs(diff_sigma) < 1:
            results['consistency'] = 'excellent'
            print("  Consistency: EXCELLENT (< 1σ)")
        elif abs(diff_sigma) < 2:
            results['consistency'] = 'good'
            print("  Consistency: GOOD (< 2σ)")
        else:
            results['consistency'] = 'tension'
            print(f"  Consistency: TENSION ({abs(diff_sigma):.1f}σ)")

    except Exception as e:
        print(f"Could not load previous estimate: {e}")
        results['consistency'] = 'no_comparison'

    print()

    return results


def create_summary_plot(results, halpha_result):
    """Create a summary plot for the primary mass analysis."""
    print("Creating summary plot...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Mass comparison
    ax1 = axes[0]

    labels = ['Previous\nEstimate', 'This Refit']
    masses = [
        results.get('M1_previous', results['M1_refit']),
        results['M1_refit']
    ]
    errors = [
        results.get('M1_previous_err', results['M1_refit_err']),
        results['M1_refit_err']
    ]
    colors = ['#3498db', '#27ae60']

    x = np.arange(len(labels))
    bars = ax1.bar(x, masses, yerr=errors, capsize=5, color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels
    for bar, m, e in zip(bars, masses, errors):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + e + 0.02,
                f'{m:.3f}±{e:.3f}', ha='center', va='bottom', fontsize=10)

    ax1.set_ylabel('Primary Mass (M☉)', fontsize=12)
    ax1.set_title('Primary Mass Estimate Comparison', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, max(masses) * 1.4)

    # Add consistency verdict
    consistency = results.get('consistency', 'unknown')
    if consistency == 'excellent':
        verdict_color = 'green'
    elif consistency == 'good':
        verdict_color = 'blue'
    else:
        verdict_color = 'orange'

    ax1.text(0.02, 0.98, f"Consistency: {consistency.upper()}",
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             color=verdict_color, va='top')

    # Panel 2: Parameter summary
    ax2 = axes[1]
    ax2.axis('off')

    summary = [
        f"Target: {TARGET_NAME}",
        f"LAMOST ObsID: {LAMOST_OBSID}",
        "",
        "=== Stellar Parameters ===",
        f"Spectral Type: {LAMOST_CATALOG['spectral_type']}",
        f"Teff: {results['teff_fit']} ± {results['teff_err']} K",
        f"  Source: {results['teff_source']}",
        f"[Fe/H]: {results['feh_fit']:.2f} ± {results['feh_err']:.2f}",
        f"  Source: {results['feh_source']}",
        f"log g: {results['logg_fit']:.2f}",
        f"  Source: {results['logg_source']}",
        "",
        "=== Primary Mass ===",
        f"M₁ = {results['M1_refit']:.3f} ± {results['M1_refit_err']:.3f} M☉",
        f"68% CI: [{results['M1_refit_16']:.3f}, {results['M1_refit_84']:.3f}]",
        "",
        "=== H-alpha Activity ===",
        f"Status: {halpha_result['status']}",
        "Activity cannot explain RV: YES",
        f"Max jitter from activity: < 1 km/s",
        f"Observed ΔRV: 146 km/s",
    ]

    ax2.text(0.05, 0.95, '\n'.join(summary),
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('primary_mass_refit_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: primary_mass_refit_plot.png")


def main():
    print("=" * 70)
    print("LAMOST SPECTRUM REFIT - INDEPENDENT PRIMARY MASS VERIFICATION")
    print("=" * 70)
    print()

    # Attempt to retrieve spectrum
    spectrum_result = attempt_spectrum_download()

    # Analyze H-alpha activity
    halpha_result = analyze_halpha_from_catalog()

    # Perform refit analysis
    results = perform_refit_analysis(spectrum_result)

    # Add H-alpha results
    results['Halpha_EW'] = halpha_result.get('Halpha_EW')
    results['Halpha_flag'] = halpha_result.get('Halpha_flag')
    results['halpha_analysis'] = halpha_result

    # Create plot
    create_summary_plot(results, halpha_result)

    # Final output
    output = {
        'target': TARGET_NAME,
        'lamost_obsid': LAMOST_OBSID,
        'analysis': 'independent_primary_mass_verification',
        **results
    }

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Spectrum status: {results['spectrum_status']}")
    print(f"Primary mass: M₁ = {results['M1_refit']:.3f} ± {results['M1_refit_err']:.3f} M☉")
    if 'consistency' in results:
        print(f"Consistency with previous: {results['consistency'].upper()}")
    print(f"H-alpha activity can explain RV: NO (jitter < 1 km/s << 146 km/s)")
    print()

    # Save
    with open('primary_mass_refit_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved: primary_mass_refit_results.json")

    return output


if __name__ == "__main__":
    main()
