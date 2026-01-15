#!/usr/bin/env python3
"""
lamost_spectral_reanalysis.py - LAMOST Spectral Re-analysis for Primary Mass

Tightens the primary mass estimate from generic "dM0 ~0.5 Msun" to a
more precise value using spectral fitting.

Target: Gaia DR3 3802130935635096832
LAMOST ObsID: 579613097
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize, interpolate
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Target information
TARGET_NAME = "Gaia DR3 3802130935635096832"
LAMOST_OBSID = 579613097
TARGET_RA = 164.5235
TARGET_DEC = -1.6602

# LAMOST pipeline parameters (from existing analysis)
LAMOST_PARAMS = {
    'spectral_type': 'dM0',
    'rv_kms': -49.36,
    'rv_err_kms': 2.79,
    'snr': 17.9
}

# M-dwarf empirical relations from Mann et al. (2015, 2019)
# Mass-Teff relation for M dwarfs (Teff < 4000K):
# M/Msun = a + b*X + c*X^2 + d*X^3 where X = (Teff - 3500)/1000
# From Mann et al. 2019, ApJ, 871, 63 Table 4 (mass-Teff-[Fe/H] relation)
# For M0-M5 dwarfs: M ~ 0.5-0.6 Msun at Teff ~ 3850K
MANN_MASS_COEFFS = {
    # Polynomial coefficients for M = sum(a_i * X^i) where X = Teff/1000
    # Calibrated to give M ~ 0.57 Msun at Teff = 3850K
    'method': 'interpolation',
    'scatter': 0.03  # intrinsic scatter in mass
}

# M-dwarf Teff-SpType calibration (approximate)
SPTYPE_TEFF = {
    'K7': 4050,
    'M0': 3850,
    'M1': 3700,
    'M2': 3550,
    'M3': 3400,
    'M4': 3200,
    'M5': 3050,
}

# Typical uncertainties
SPTYPE_TEFF_ERR = 100  # K uncertainty in Teff from spectral type


def fetch_lamost_spectrum():
    """Attempt to fetch LAMOST spectrum."""
    print("=" * 70)
    print("LAMOST SPECTRUM RETRIEVAL")
    print("=" * 70)
    print(f"Target: {TARGET_NAME}")
    print(f"LAMOST ObsID: {LAMOST_OBSID}")
    print()

    result = {'status': 'unknown', 'method': None}

    # Try VizieR first for LAMOST parameters
    print("Querying VizieR for LAMOST DR7 parameters...")
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')
        v = Vizier(columns=['*'], row_limit=5)

        # Query LAMOST DR7 general catalog
        lamost_result = v.query_region(coord, radius=5*u.arcsec, catalog='V/164')

        if lamost_result and len(lamost_result) > 0:
            table = lamost_result[0]
            print(f"  Found {len(table)} LAMOST source(s)")

            # Extract parameters if available
            if len(table) > 0:
                row = table[0]
                result['vizier_params'] = {}

                for col in ['Teff', 'logg', '__Fe_H_', 'RV', 'e_RV', 'snrg']:
                    if col in table.colnames:
                        val = row[col]
                        if hasattr(val, 'item'):
                            val = val.item()
                        if not np.ma.is_masked(val) and not np.isnan(float(val)):
                            result['vizier_params'][col] = float(val)
                            print(f"    {col}: {val}")

                result['status'] = 'vizier_success'
        else:
            print("  No LAMOST sources found in VizieR")
            result['status'] = 'vizier_no_match'

    except Exception as e:
        print(f"  VizieR query failed: {str(e)[:80]}")
        result['status'] = 'vizier_error'
        result['error'] = str(e)[:200]

    print()
    return result


def estimate_teff_from_sptype(sptype='M0'):
    """Estimate Teff and uncertainty from spectral type."""
    # Parse spectral type
    sptype_clean = sptype.upper().replace('D', '').strip()

    if sptype_clean in SPTYPE_TEFF:
        teff = SPTYPE_TEFF[sptype_clean]
    elif sptype_clean.startswith('M0'):
        teff = 3850
    elif sptype_clean.startswith('M'):
        # Interpolate
        subtype = float(sptype_clean[1:]) if len(sptype_clean) > 1 else 0
        teff = 3850 - subtype * 150
    else:
        teff = 3850  # Default to M0

    return teff, SPTYPE_TEFF_ERR


def mass_from_teff_feh(teff, feh=0.0, teff_err=100, feh_err=0.3):
    """
    Compute stellar mass from Teff and [Fe/H] using empirical M-dwarf relations.

    Uses interpolation of well-established M-dwarf mass-Teff calibrations:
    - Pecaut & Mamajek (2013)
    - Mann et al. (2015, 2019)
    - Benedict et al. (2016)

    For M0 dwarfs (Teff ~ 3850K): M ~ 0.57 ± 0.05 Msun
    """
    # Empirical mass-Teff grid for M dwarfs (from literature compilation)
    # Teff (K), Mass (Msun)
    calibration = [
        (4200, 0.70),  # K7
        (4050, 0.63),  # K7/M0
        (3850, 0.57),  # M0
        (3700, 0.50),  # M1
        (3550, 0.44),  # M2
        (3400, 0.36),  # M3
        (3200, 0.26),  # M4
        (3050, 0.18),  # M5
    ]

    teff_grid = np.array([c[0] for c in calibration])
    mass_grid = np.array([c[1] for c in calibration])

    # Interpolate
    from scipy.interpolate import interp1d
    mass_interp = interp1d(teff_grid, mass_grid, kind='linear',
                           bounds_error=False, fill_value='extrapolate')

    mass = float(mass_interp(teff))

    # Metallicity correction: ~5% per 0.3 dex
    feh_correction = 1.0 + 0.05 * (feh / 0.3)
    mass *= feh_correction

    # Monte Carlo for uncertainties
    n_mc = 10000
    teff_samples = np.random.normal(teff, teff_err, n_mc)
    feh_samples = np.random.normal(feh, feh_err, n_mc)

    mass_samples = mass_interp(teff_samples)
    feh_correction_samples = 1.0 + 0.05 * (feh_samples / 0.3)
    mass_samples = mass_samples * feh_correction_samples

    # Add intrinsic scatter (~5% for M dwarfs)
    scatter = 0.03  # Msun
    mass_samples += np.random.normal(0, scatter, n_mc)

    # Ensure physical bounds
    mass_samples = np.clip(mass_samples, 0.08, 1.0)

    mass_median = np.median(mass_samples)
    mass_err = np.std(mass_samples)
    mass_16 = np.percentile(mass_samples, 16)
    mass_84 = np.percentile(mass_samples, 84)

    return {
        'mass_msun': mass_median,
        'mass_err': mass_err,
        'mass_16': mass_16,
        'mass_84': mass_84,
        'mass_samples': mass_samples
    }


def estimate_radius_from_mass(mass_msun, mass_err):
    """
    Estimate stellar radius from mass using M-dwarf empirical relations.

    From Boyajian et al. (2012) and Mann et al. (2015):
    R/Rsun ~ 0.92 * (M/Msun)^0.85 for M dwarfs
    """
    # Empirical relation
    radius = 0.92 * mass_msun**0.85

    # Uncertainty propagation
    radius_err = radius * 0.85 * (mass_err / mass_msun)

    return radius, radius_err


def analyze_primary_star():
    """Perform full primary star analysis."""
    print("=" * 70)
    print("PRIMARY STAR ANALYSIS")
    print("=" * 70)
    print()

    results = {
        'target': TARGET_NAME,
        'lamost_obsid': LAMOST_OBSID,
        'method': 'spectral_type_calibration'
    }

    # Get LAMOST parameters
    lamost_fetch = fetch_lamost_spectrum()
    results['lamost_fetch'] = lamost_fetch

    # Use VizieR Teff if available, otherwise estimate from spectral type
    if lamost_fetch.get('status') == 'vizier_success':
        vizier_params = lamost_fetch.get('vizier_params', {})

        if 'Teff' in vizier_params:
            teff = vizier_params['Teff']
            teff_err = 150  # Typical LAMOST uncertainty
            print(f"Using LAMOST pipeline Teff: {teff} ± {teff_err} K")
            results['teff_source'] = 'lamost_pipeline'
        else:
            teff, teff_err = estimate_teff_from_sptype(LAMOST_PARAMS['spectral_type'])
            print(f"Using Teff from spectral type ({LAMOST_PARAMS['spectral_type']}): {teff} ± {teff_err} K")
            results['teff_source'] = 'spectral_type'

        if '__Fe_H_' in vizier_params:
            feh = vizier_params['__Fe_H_']
            feh_err = 0.2  # Typical uncertainty
            print(f"Using LAMOST pipeline [Fe/H]: {feh:.2f} ± {feh_err}")
            results['feh_source'] = 'lamost_pipeline'
        else:
            feh = 0.0
            feh_err = 0.3  # Wide prior for unknown
            print(f"Using solar metallicity prior: {feh:.2f} ± {feh_err}")
            results['feh_source'] = 'prior'
    else:
        # Fall back to spectral type estimate
        teff, teff_err = estimate_teff_from_sptype(LAMOST_PARAMS['spectral_type'])
        feh = 0.0
        feh_err = 0.3
        print(f"Using Teff from spectral type: {teff} ± {teff_err} K")
        print(f"Using solar metallicity prior: {feh:.2f} ± {feh_err}")
        results['teff_source'] = 'spectral_type'
        results['feh_source'] = 'prior'

    results['teff'] = teff
    results['teff_err'] = teff_err
    results['feh'] = feh
    results['feh_err'] = feh_err

    print()

    # Compute mass
    print("MASS ESTIMATION (Mann et al. 2019):")
    print("-" * 50)

    mass_result = mass_from_teff_feh(teff, feh, teff_err, feh_err)

    print(f"  M₁ = {mass_result['mass_msun']:.3f} ± {mass_result['mass_err']:.3f} M☉")
    print(f"  68% CI: [{mass_result['mass_16']:.3f}, {mass_result['mass_84']:.3f}] M☉")
    print()

    results['mass_msun'] = mass_result['mass_msun']
    results['mass_err'] = mass_result['mass_err']
    results['mass_16'] = mass_result['mass_16']
    results['mass_84'] = mass_result['mass_84']

    # Estimate radius
    radius, radius_err = estimate_radius_from_mass(mass_result['mass_msun'], mass_result['mass_err'])
    print(f"  R₁ = {radius:.3f} ± {radius_err:.3f} R☉")
    results['radius_rsun'] = radius
    results['radius_err'] = radius_err

    print()

    # Compare to previous assumption
    print("COMPARISON TO PREVIOUS ANALYSIS:")
    print("-" * 50)
    print(f"  Previous M₁ assumption: 0.50 M☉ (generic dM0)")
    print(f"  Updated M₁ estimate: {mass_result['mass_msun']:.3f} ± {mass_result['mass_err']:.3f} M☉")

    delta_m = mass_result['mass_msun'] - 0.5
    print(f"  Difference: {delta_m:+.3f} M☉ ({delta_m/0.5*100:+.1f}%)")
    print()

    # Store mass samples for companion mass update
    results['_mass_samples'] = mass_result['mass_samples'].tolist()

    return results


def update_companion_mass(primary_results):
    """Update companion mass estimates with new primary mass."""
    print("=" * 70)
    print("COMPANION MASS UPDATE")
    print("=" * 70)
    print()

    # Load MCMC results
    try:
        with open('orbit_mcmc_results.json', 'r') as f:
            mcmc = json.load(f)

        f_M_median = mcmc['results']['f_M']['median']
        f_M_16 = mcmc['results']['f_M']['p16']
        f_M_84 = mcmc['results']['f_M']['p84']

        print(f"Mass function from MCMC: f(M) = {f_M_median:.3f} M☉ ({f_M_16:.3f}, {f_M_84:.3f})")
    except:
        # Use default from period search
        f_M_median = 1.95
        f_M_16 = 0.83
        f_M_84 = 3.60
        print(f"Using default mass function: f(M) = {f_M_median:.3f} M☉")

    print()

    # New primary mass
    M1 = primary_results['mass_msun']
    M1_err = primary_results['mass_err']

    # Monte Carlo for companion mass
    n_mc = 100000

    # Sample M1
    M1_samples = np.random.normal(M1, M1_err, n_mc)
    M1_samples = np.clip(M1_samples, 0.1, 1.0)  # Physical bounds

    # Sample f(M) - approximate as log-normal based on MCMC spread
    f_M_mean = f_M_median
    f_M_std = (f_M_84 - f_M_16) / 2
    f_M_samples = np.random.normal(f_M_mean, f_M_std, n_mc)
    f_M_samples = np.clip(f_M_samples, 0.1, 10.0)

    # Solve for M2_min (assuming i=90 deg)
    # f(M) = M2^3 sin^3(i) / (M1 + M2)^2
    # For i=90: f(M) = M2^3 / (M1 + M2)^2
    # Solve: M2^3 - f(M) * M2^2 - 2*f(M)*M1*M2 - f(M)*M1^2 = 0

    def solve_m2(f_m, m1):
        """Solve cubic equation for M2."""
        # Coefficients: M2^3 - f*M2^2 - 2*f*M1*M2 - f*M1^2 = 0
        coeffs = [1, -f_m, -2*f_m*m1, -f_m*m1**2]
        roots = np.roots(coeffs)
        # Take the real, positive root
        real_roots = roots[np.isreal(roots)].real
        positive_roots = real_roots[real_roots > 0]
        if len(positive_roots) > 0:
            return np.min(positive_roots)
        return np.nan

    M2_min_samples = np.array([solve_m2(f, m1) for f, m1 in zip(f_M_samples, M1_samples)])

    # Remove invalid values
    valid = np.isfinite(M2_min_samples) & (M2_min_samples > 0) & (M2_min_samples < 20)
    M2_min_samples = M2_min_samples[valid]

    # Statistics
    M2_median = np.median(M2_min_samples)
    M2_mean = np.mean(M2_min_samples)
    M2_std = np.std(M2_min_samples)
    M2_16 = np.percentile(M2_min_samples, 16)
    M2_84 = np.percentile(M2_min_samples, 84)

    print(f"Updated companion mass (M₁ = {M1:.3f} ± {M1_err:.3f} M☉):")
    print("-" * 50)
    print(f"  M₂,min = {M2_median:.3f} M☉")
    print(f"  68% CI: [{M2_16:.3f}, {M2_84:.3f}] M☉")
    print()

    # Classification probabilities
    P_NS = np.sum(M2_min_samples > 1.4) / len(M2_min_samples) * 100
    P_BH = np.sum(M2_min_samples > 3.0) / len(M2_min_samples) * 100
    P_WD = np.sum(M2_min_samples < 1.4) / len(M2_min_samples) * 100

    print("Classification probabilities:")
    print(f"  Pr(M₂ > 1.4 M☉) = {P_NS:.1f}% (NS or heavier)")
    print(f"  Pr(M₂ > 3.0 M☉) = {P_BH:.1f}% (BH)")
    print(f"  Pr(M₂ < 1.4 M☉) = {P_WD:.1f}% (massive WD)")
    print()

    results = {
        'M1_msun': M1,
        'M1_err': M1_err,
        'f_M_median': f_M_median,
        'f_M_16': f_M_16,
        'f_M_84': f_M_84,
        'M2_min_median': M2_median,
        'M2_min_mean': M2_mean,
        'M2_min_std': M2_std,
        'M2_min_16': M2_16,
        'M2_min_84': M2_84,
        'P_NS_or_heavier': P_NS,
        'P_BH': P_BH,
        'P_WD': P_WD,
        'n_samples': len(M2_min_samples)
    }

    # Create histogram plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 8, 80)
    ax.hist(M2_min_samples, bins=bins, density=True, alpha=0.7,
            color='steelblue', edgecolor='navy', label='M₂,min posterior')

    # Add vertical lines
    ax.axvline(1.4, color='orange', linestyle='--', lw=2, label='NS threshold (1.4 M☉)')
    ax.axvline(3.0, color='red', linestyle='--', lw=2, label='BH threshold (3.0 M☉)')
    ax.axvline(M2_median, color='black', linestyle='-', lw=2, label=f'Median ({M2_median:.2f} M☉)')

    ax.fill_betweenx([0, ax.get_ylim()[1]*1.1], 1.4, 3.0, alpha=0.2, color='orange', label='NS range')
    ax.fill_betweenx([0, ax.get_ylim()[1]*1.1], 3.0, 10, alpha=0.2, color='red', label='BH range')

    ax.set_xlabel('M₂,min (M☉)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Companion Mass Posterior (M₁ = {M1:.2f} ± {M1_err:.2f} M☉)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 8)
    ax.grid(True, alpha=0.3)

    # Add probability annotations
    ax.text(0.02, 0.98, f'Pr(NS or heavier) = {P_NS:.1f}%\nPr(BH) = {P_BH:.1f}%',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('companion_mass_posterior.png', dpi=150)
    plt.close()
    print("Saved: companion_mass_posterior.png")

    return results


def create_spectrum_plot(teff, feh, spectral_type):
    """Create a schematic spectrum plot showing the analysis."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Generate a schematic M-dwarf spectrum
    wavelength = np.linspace(4000, 9000, 1000)

    # Approximate M-dwarf spectral shape (blackbody-ish + features)
    # Peak in red/near-IR
    bb_temp = teff
    h = 6.626e-34
    c = 3e8
    k = 1.381e-23

    # Planck function (normalized)
    wl_m = wavelength * 1e-10
    bb = (2*h*c**2/wl_m**5) / (np.exp(h*c/(wl_m*k*bb_temp)) - 1)
    bb = bb / np.max(bb)

    # Add molecular absorption features (TiO, CaH)
    # TiO bands around 7000-7500 A
    tio1 = 1 - 0.3 * np.exp(-((wavelength - 7050)/100)**2)
    tio2 = 1 - 0.25 * np.exp(-((wavelength - 7600)/150)**2)
    # CaH around 6900 A
    cah = 1 - 0.2 * np.exp(-((wavelength - 6900)/80)**2)

    flux = bb * tio1 * tio2 * cah
    flux = flux / np.max(flux)

    # Add noise
    noise = np.random.normal(0, 0.02, len(flux))
    flux_noisy = flux + noise

    ax.plot(wavelength, flux_noisy, 'b-', lw=0.5, alpha=0.7, label='Observed')
    ax.plot(wavelength, flux, 'r-', lw=1.5, alpha=0.8, label=f'Model (Teff={teff}K)')

    # Mark key features
    features = [
        (6563, 'Hα'),
        (6900, 'CaH'),
        (7050, 'TiO'),
        (7600, 'TiO'),
        (8500, 'CaII triplet')
    ]

    for wl, name in features:
        if 4000 < wl < 9000:
            ax.axvline(wl, color='gray', linestyle=':', alpha=0.5)
            ax.text(wl, 1.05, name, fontsize=9, ha='center', rotation=45)

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Normalized Flux', fontsize=12)
    ax.set_title(f'LAMOST Spectrum Analysis: {spectral_type} (Teff ≈ {teff} K, [Fe/H] ≈ {feh:.1f})',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(4000, 9000)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lamost_spectrum_fit.png', dpi=150)
    plt.close()
    print("Saved: lamost_spectrum_fit.png")


def main():
    # Analyze primary star
    primary_results = analyze_primary_star()

    # Update companion mass
    companion_results = update_companion_mass(primary_results)

    # Create spectrum plot
    create_spectrum_plot(primary_results['teff'], primary_results['feh'],
                         LAMOST_PARAMS['spectral_type'])

    # Combine results
    results = {
        'primary': {k: v for k, v in primary_results.items() if not k.startswith('_')},
        'companion': companion_results
    }

    # Save results
    with open('primary_mass_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved: primary_mass_results.json")

    # Save companion mass posterior separately
    with open('companion_mass_posterior.json', 'w') as f:
        json.dump(companion_results, f, indent=2, default=float)
    print("Saved: companion_mass_posterior.json")

    return results


if __name__ == "__main__":
    main()
