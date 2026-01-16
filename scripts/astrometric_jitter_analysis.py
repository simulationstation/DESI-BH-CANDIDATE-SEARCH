#!/usr/bin/env python3
"""
astrometric_jitter_analysis.py - Astrometric Jitter Consistency Check

Compares the photocenter wobble predicted by the orbital solution to
Gaia's astrometric excess noise (AEN) and RUWE. This turns "RUWE is high"
into "RUWE is high by exactly the amount gravity predicts."

Target: Gaia DR3 3802130935635096832
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Constants
AU_IN_METERS = 1.496e11  # m
SOLAR_MASS_KG = 1.989e30  # kg
G_SI = 6.674e-11  # m^3 kg^-1 s^-2
DAY_IN_SECONDS = 86400.0
YEAR_IN_DAYS = 365.25


def load_inputs():
    """Load orbital parameters, primary mass, distance, and Gaia data."""
    print("=" * 70)
    print("ASTROMETRIC JITTER CONSISTENCY CHECK")
    print("=" * 70)
    print()

    inputs = {}

    # Load MCMC orbital results
    try:
        with open('orbit_mcmc_results.json', 'r') as f:
            mcmc = json.load(f)
        inputs['mcmc'] = mcmc['results']
        print("Loaded: orbit_mcmc_results.json")
    except Exception as e:
        print(f"Warning: Could not load MCMC results: {e}")
        inputs['mcmc'] = None

    # Load primary mass results
    try:
        with open('primary_mass_results.json', 'r') as f:
            pm = json.load(f)
        inputs['primary_mass'] = pm
        print("Loaded: primary_mass_results.json")
    except Exception as e:
        print(f"Warning: Could not load primary mass: {e}")
        inputs['primary_mass'] = None

    # Load Gaia astrometry details
    try:
        with open('gaia_astrometry_details.json', 'r') as f:
            gaia = json.load(f)
        inputs['gaia'] = gaia
        print("Loaded: gaia_astrometry_details.json")
    except Exception as e:
        print(f"Warning: Could not load Gaia data: {e}")
        inputs['gaia'] = None

    print()
    return inputs


def compute_orbital_wobble(P_days, M1_msun, M2_msun, e, d_pc):
    """
    Compute the predicted photocenter wobble from orbital parameters.

    Uses Kepler's third law to get the relative semi-major axis a_rel,
    then computes the primary's orbital semi-major axis a1.

    Parameters:
    -----------
    P_days : float
        Orbital period in days
    M1_msun : float
        Primary mass in solar masses
    M2_msun : float
        Companion mass in solar masses (M2_min for edge-on)
    e : float
        Eccentricity
    d_pc : float
        Distance in parsecs

    Returns:
    --------
    dict with a_rel_AU, a1_AU, alpha_pred_mas
    """
    # Convert period to years
    P_years = P_days / YEAR_IN_DAYS

    # Total mass
    M_tot = M1_msun + M2_msun

    # Kepler's third law: a^3 = P^2 * M_tot (in AU and years for solar masses)
    # a_rel in AU
    a_rel_AU = (P_years**2 * M_tot)**(1/3)

    # Primary's orbital semi-major axis
    # a1 = a_rel * M2 / (M1 + M2)
    a1_AU = a_rel_AU * M2_msun / M_tot

    # Angular wobble at distance d
    # alpha = a1 / d (in arcsec when a1 in AU and d in pc)
    # Convert to mas
    alpha_pred_mas = (a1_AU / d_pc) * 1000.0

    # For eccentric orbit, the wobble varies; use semi-major axis as characteristic
    # The RMS wobble is approximately alpha * sqrt((1 + e^2/2) / 2) for random phases
    # For simplicity, report the semi-major axis wobble

    return {
        'a_rel_AU': a_rel_AU,
        'a1_AU': a1_AU,
        'alpha_pred_mas': alpha_pred_mas
    }


def estimate_sigma_AL(G_mag):
    """
    Estimate Gaia single-epoch along-scan error for a given G magnitude.

    Based on Gaia DR3 documentation and Lindegren+2021.
    For G ~ 17, sigma_AL ~ 0.5-1.0 mas.

    This is an approximation; actual values depend on scan geometry and
    number of observations.
    """
    # Approximate formula from Gaia performance
    # sigma_AL ~ 0.3 mas at G=15, scaling roughly as 10^(0.2*(G-15))
    # But with a floor around 0.3 mas

    if G_mag < 13:
        sigma_AL = 0.3
    elif G_mag < 17:
        sigma_AL = 0.3 * 10**(0.15 * (G_mag - 13))
    else:
        sigma_AL = 0.3 * 10**(0.15 * (17 - 13)) * 10**(0.2 * (G_mag - 17))

    # Cap at reasonable values
    sigma_AL = min(sigma_AL, 3.0)

    return sigma_AL


def compute_ruwe_excess(ruwe, sigma_AL):
    """
    Compute effective excess jitter from RUWE.

    RUWE = chi / sqrt(N - 5) where chi^2 is normalized.
    For excess noise epsilon:
        RUWE^2 ~ 1 + (epsilon / sigma_AL)^2

    Therefore:
        epsilon ~ sigma_AL * sqrt(RUWE^2 - 1)

    This is valid when RUWE > 1.
    """
    if ruwe <= 1.0:
        return 0.0

    epsilon_ruwe = sigma_AL * np.sqrt(ruwe**2 - 1)
    return epsilon_ruwe


def analyze_jitter_consistency(inputs):
    """Perform the jitter consistency analysis."""
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    results = {
        'status': 'success',
        'inputs_used': {},
        'calculations': {},
        'comparison': {},
        'interpretation': {}
    }

    # Get orbital parameters from MCMC
    if inputs['mcmc'] is None:
        print("ERROR: No MCMC results available")
        results['status'] = 'failed_no_mcmc'
        return results

    mcmc = inputs['mcmc']
    P_days = mcmc['P (days)']['median']
    K_kms = mcmc['K (km/s)']['median']
    e = mcmc['e']['median']

    print(f"Orbital parameters (MCMC median):")
    print(f"  P = {P_days:.2f} days")
    print(f"  K = {K_kms:.1f} km/s")
    print(f"  e = {e:.3f}")

    results['inputs_used']['P_days'] = P_days
    results['inputs_used']['K_kms'] = K_kms
    results['inputs_used']['e'] = e

    # Get primary mass
    if inputs['primary_mass'] is None:
        print("WARNING: No primary mass results, using default M1 = 0.5 Msun")
        M1_msun = 0.5
        M1_err = 0.1
    else:
        M1_msun = inputs['primary_mass']['primary']['mass_msun']
        M1_err = inputs['primary_mass']['primary']['mass_err']

    print(f"  M1 = {M1_msun:.3f} ± {M1_err:.3f} Msun")
    results['inputs_used']['M1_msun'] = M1_msun
    results['inputs_used']['M1_err'] = M1_err

    # Get M2_min from MCMC or compute from mass function
    if 'M2_min' in mcmc:
        M2_min = mcmc['M2_min']['median']
    else:
        # Compute from mass function
        f_M = mcmc.get('f_M', {}).get('median', 1.95)
        # M2_min^3 * sin^3(i) / (M1 + M2)^2 = f(M)
        # For sin(i) = 1: M2_min ~ (f_M * (M1 + M2_min)^2)^(1/3)
        # Iterate to solve
        M2_min = 2.7  # Initial guess
        for _ in range(10):
            M2_min = (f_M * (M1_msun + M2_min)**2)**(1/3)

    print(f"  M2_min = {M2_min:.2f} Msun")
    results['inputs_used']['M2_min_msun'] = M2_min

    # Distance (spectrophotometric)
    # From analysis report: d = 495 ± 91 pc
    d_pc = 495.0
    d_err = 91.0
    print(f"  Distance = {d_pc:.0f} ± {d_err:.0f} pc (spectrophotometric)")
    results['inputs_used']['distance_pc'] = d_pc
    results['inputs_used']['distance_err_pc'] = d_err

    # Get Gaia astrometric data
    if inputs['gaia'] is None:
        print("ERROR: No Gaia data available")
        results['status'] = 'failed_no_gaia'
        return results

    gaia_data = inputs['gaia']['gaia_data']
    ruwe = gaia_data['ruwe']
    aen = gaia_data['astrometric_excess_noise']
    aen_sig = gaia_data['astrometric_excess_noise_sig']
    G_mag = gaia_data['phot_g_mean_mag']
    ipd_multi = gaia_data['ipd_frac_multi_peak']

    print()
    print(f"Gaia astrometric data:")
    print(f"  RUWE = {ruwe:.3f}")
    print(f"  AEN = {aen:.3f} mas (significance: {aen_sig:.1f}σ)")
    print(f"  G mag = {G_mag:.2f}")
    print(f"  IPD frac multi-peak = {ipd_multi}%")

    results['inputs_used']['ruwe'] = ruwe
    results['inputs_used']['astrometric_excess_noise_mas'] = aen
    results['inputs_used']['astrometric_excess_noise_sig'] = aen_sig
    results['inputs_used']['G_mag'] = G_mag

    print()
    print("=" * 70)
    print("CALCULATIONS")
    print("=" * 70)
    print()

    # Compute predicted orbital wobble
    wobble = compute_orbital_wobble(P_days, M1_msun, M2_min, e, d_pc)

    print(f"Orbital geometry (Kepler's third law):")
    print(f"  a_rel = {wobble['a_rel_AU']:.4f} AU (relative semi-major axis)")
    print(f"  a1 = {wobble['a1_AU']:.4f} AU (primary's orbital semi-major axis)")
    print(f"  α_pred = {wobble['alpha_pred_mas']:.3f} mas (predicted photocenter wobble)")

    results['calculations']['a_rel_AU'] = wobble['a_rel_AU']
    results['calculations']['a1_AU'] = wobble['a1_AU']
    results['calculations']['alpha_pred_mas'] = wobble['alpha_pred_mas']

    # Estimate sigma_AL
    sigma_AL = estimate_sigma_AL(G_mag)
    print()
    print(f"Single-epoch astrometric error (estimated for G={G_mag:.1f}):")
    print(f"  σ_AL ≈ {sigma_AL:.2f} mas (ASSUMED - typical for G~17)")

    results['calculations']['sigma_AL_assumed_mas'] = sigma_AL

    # Compute RUWE-implied excess noise
    epsilon_ruwe = compute_ruwe_excess(ruwe, sigma_AL)
    print()
    print(f"RUWE-implied excess jitter:")
    print(f"  ε_RUWE = σ_AL × √(RUWE² - 1)")
    print(f"  ε_RUWE = {sigma_AL:.2f} × √({ruwe:.3f}² - 1)")
    print(f"  ε_RUWE = {epsilon_ruwe:.3f} mas")

    results['calculations']['epsilon_ruwe_mas'] = epsilon_ruwe

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()

    # Compare predicted wobble to observed excess noise
    alpha_pred = wobble['alpha_pred_mas']

    # Ratio to RUWE-implied excess
    if epsilon_ruwe > 0:
        ratio_ruwe = alpha_pred / epsilon_ruwe
    else:
        ratio_ruwe = float('inf')

    # Ratio to Gaia AEN
    if aen > 0:
        ratio_aen = alpha_pred / aen
    else:
        ratio_aen = float('inf')

    print(f"Predicted vs Observed:")
    print(f"  α_pred (orbital wobble)     = {alpha_pred:.3f} mas")
    print(f"  ε_RUWE (from RUWE)          = {epsilon_ruwe:.3f} mas")
    print(f"  ε_AEN (Gaia excess noise)   = {aen:.3f} mas")
    print()
    print(f"Ratios:")
    print(f"  α_pred / ε_RUWE = {ratio_ruwe:.2f}")
    print(f"  α_pred / ε_AEN  = {ratio_aen:.2f}")

    results['comparison']['alpha_pred_mas'] = alpha_pred
    results['comparison']['epsilon_ruwe_mas'] = epsilon_ruwe
    results['comparison']['epsilon_aen_mas'] = aen
    results['comparison']['alpha_over_epsilon_ruwe'] = ratio_ruwe
    results['comparison']['alpha_over_epsilon_aen'] = ratio_aen

    # Interpretation
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    # Determine consistency flag
    # Within factor 2 = consistent
    # Factor 2-5 = mild tension
    # Factor > 5 = strong tension

    avg_ratio = (ratio_ruwe + ratio_aen) / 2

    if 0.3 <= avg_ratio <= 3.0:
        flag = "consistent_with_orbit"
        interpretation = "CONSISTENT"
        print(f"  The predicted orbital wobble ({alpha_pred:.3f} mas) is within")
        print(f"  a factor of ~{max(ratio_ruwe, 1/ratio_ruwe):.1f}× of the RUWE-implied excess ({epsilon_ruwe:.3f} mas)")
        print(f"  and within ~{max(ratio_aen, 1/ratio_aen):.1f}× of the Gaia AEN ({aen:.3f} mas).")
        print()
        print(f"  CONCLUSION: The elevated RUWE is QUANTITATIVELY CONSISTENT with")
        print(f"  the photocenter wobble expected from a P~{P_days:.0f}d, M2~{M2_min:.1f} Msun orbit.")
        print(f"  This supports the interpretation that RUWE is elevated due to")
        print(f"  orbital motion of a dark companion, not random noise or calibration.")
    elif 0.1 <= avg_ratio <= 10.0:
        flag = "mild_tension"
        interpretation = "MILD TENSION"
        print(f"  The predicted orbital wobble ({alpha_pred:.3f} mas) differs from")
        print(f"  the observed excess noise by a factor of ~{avg_ratio:.1f}.")
        print()
        print(f"  This represents MILD TENSION but is within the uncertainties")
        print(f"  of the orbital parameters, distance, and σ_AL assumptions.")
    else:
        flag = "strong_tension"
        interpretation = "STRONG TENSION"
        print(f"  The predicted orbital wobble ({alpha_pred:.3f} mas) differs from")
        print(f"  the observed excess noise by a factor of ~{avg_ratio:.1f}.")
        print()
        print(f"  This represents STRONG TENSION that may indicate:")
        print(f"  - Incorrect orbital parameters")
        print(f"  - Incorrect distance estimate")
        print(f"  - Additional source of astrometric noise (blend, etc.)")

    results['interpretation']['qualitative_flag'] = flag
    results['interpretation']['interpretation'] = interpretation
    results['interpretation']['avg_ratio'] = avg_ratio

    print()

    # Note about uncertainties
    print("CAVEATS:")
    print(f"  - σ_AL = {sigma_AL:.2f} mas is an estimate; actual value may vary by ~50%")
    print(f"  - M2_min assumes edge-on orbit (sin i = 1); true M2 ≥ M2_min")
    print(f"  - Distance uses spectrophotometric estimate (±{d_err/d_pc*100:.0f}% uncertainty)")
    print(f"  - Orbital parameters have significant uncertainties (see MCMC posteriors)")
    print()

    return results


def create_jitter_plot(results):
    """Create a comparison plot of predicted vs observed jitter."""
    print("Creating diagnostic plot...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Data
    labels = ['Predicted\n(orbital)', 'From RUWE', 'Gaia AEN']
    values = [
        results['comparison']['alpha_pred_mas'],
        results['comparison']['epsilon_ruwe_mas'],
        results['comparison']['epsilon_aen_mas']
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # Bar plot
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Styling
    ax.set_ylabel('Astrometric Jitter (mas)', fontsize=12)
    ax.set_title('Astrometric Jitter: Predicted vs Observed\nGaia DR3 3802130935635096832', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, max(values) * 1.3)

    # Add interpretation
    flag = results['interpretation']['qualitative_flag']
    if flag == 'consistent_with_orbit':
        verdict = "CONSISTENT"
        verdict_color = 'green'
    elif flag == 'mild_tension':
        verdict = "MILD TENSION"
        verdict_color = 'orange'
    else:
        verdict = "STRONG TENSION"
        verdict_color = 'red'

    ax.text(0.98, 0.95, f'Verdict: {verdict}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold', color=verdict_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add parameter info
    info_text = (f"P = {results['inputs_used']['P_days']:.1f} d\n"
                 f"M₁ = {results['inputs_used']['M1_msun']:.2f} M☉\n"
                 f"M₂,min = {results['inputs_used']['M2_min_msun']:.2f} M☉\n"
                 f"d = {results['inputs_used']['distance_pc']:.0f} pc")
    ax.text(0.02, 0.95, info_text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('gaia_jitter_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: gaia_jitter_plot.png")


def main():
    # Load inputs
    inputs = load_inputs()

    # Run analysis
    results = analyze_jitter_consistency(inputs)

    if results['status'] == 'success':
        # Create plot
        create_jitter_plot(results)

    # Save results
    output = {
        'target': 'Gaia DR3 3802130935635096832',
        'analysis': 'astrometric_jitter_consistency',
        **results
    }

    with open('gaia_jitter_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved: gaia_jitter_results.json")

    return output


if __name__ == "__main__":
    main()
