#!/usr/bin/env python3
"""
period_search.py - Period search using combined LAMOST + DESI RV epochs

Fits circular orbits to the 5 RV epochs to constrain orbital period.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json

# Combined RV epochs (LAMOST + DESI)
EPOCHS = [
    {"source": "LAMOST", "mjd": 57457.0, "rv_kms": -49.36, "rv_err_kms": 2.79},
    {"source": "DESI", "mjd": 59568.488, "rv_kms": -86.39, "rv_err_kms": 0.55},
    {"source": "DESI", "mjd": 59605.380, "rv_kms": 59.68, "rv_err_kms": 0.83},
    {"source": "DESI", "mjd": 59607.374, "rv_kms": 26.43, "rv_err_kms": 1.06},
    {"source": "DESI", "mjd": 59607.389, "rv_kms": 25.16, "rv_err_kms": 1.11},
]

# Physical constants
G_SI = 6.674e-11  # m^3 kg^-1 s^-2
M_SUN = 1.989e30  # kg
DAY_S = 86400     # seconds per day


def circular_rv_model(t, P, K, gamma, phi0):
    """
    Circular orbit RV model.

    RV(t) = gamma + K * sin(2*pi*(t - t0)/P + phi0)

    Parameters
    ----------
    t : array-like
        Time in MJD
    P : float
        Orbital period in days
    K : float
        RV semi-amplitude in km/s
    gamma : float
        Systemic velocity in km/s
    phi0 : float
        Phase offset in radians

    Returns
    -------
    rv : array-like
        Model RV in km/s
    """
    phase = 2 * np.pi * t / P + phi0
    return gamma + K * np.sin(phase)


def chi_squared(params, t, rv, rv_err):
    """Compute chi-squared for circular orbit fit."""
    P, K, gamma, phi0 = params
    model = circular_rv_model(t, P, K, gamma, phi0)
    return np.sum(((rv - model) / rv_err)**2)


def mass_function(P_days, K_kms):
    """
    Compute mass function f(M) = (M2 * sin(i))^3 / (M1 + M2)^2

    f(M) = P * K^3 / (2 * pi * G)

    Parameters
    ----------
    P_days : float
        Orbital period in days
    K_kms : float
        RV semi-amplitude in km/s

    Returns
    -------
    f_M : float
        Mass function in solar masses
    """
    P_s = P_days * DAY_S
    K_ms = K_kms * 1000
    f_M = (P_s * K_ms**3) / (2 * np.pi * G_SI)
    return f_M / M_SUN


def minimum_companion_mass(f_M, M1):
    """
    Compute minimum companion mass (edge-on, i=90 deg).

    Solves: M2^3 / (M1 + M2)^2 = f(M)

    Parameters
    ----------
    f_M : float
        Mass function in solar masses
    M1 : float
        Primary mass in solar masses

    Returns
    -------
    M2_min : float
        Minimum companion mass in solar masses
    """
    from scipy.optimize import brentq

    def equation(M2):
        return M2**3 / (M1 + M2)**2 - f_M

    try:
        M2_min = brentq(equation, 0.01, 100)
        return M2_min
    except:
        return np.nan


def period_search(t, rv, rv_err, P_range=(5, 100), n_periods=1000):
    """
    Search for best-fit period over a range.

    Parameters
    ----------
    t : array-like
        Times in MJD
    rv : array-like
        RV values in km/s
    rv_err : array-like
        RV uncertainties in km/s
    P_range : tuple
        (P_min, P_max) in days
    n_periods : int
        Number of periods to test

    Returns
    -------
    results : dict
        Dictionary with best-fit parameters and chi-squared grid
    """
    periods = np.linspace(P_range[0], P_range[1], n_periods)
    chi2_grid = np.zeros(n_periods)

    # Initial guesses
    K_init = (np.max(rv) - np.min(rv)) / 2
    gamma_init = np.mean(rv)

    best_chi2 = np.inf
    best_params = None

    for i, P in enumerate(periods):
        # Fit for K, gamma, phi0 at fixed P
        def chi2_at_P(params):
            K, gamma, phi0 = params
            return chi_squared([P, K, gamma, phi0], t, rv, rv_err)

        # Try multiple initial phases
        best_chi2_at_P = np.inf
        for phi0_init in np.linspace(0, 2*np.pi, 8):
            try:
                res = minimize(chi2_at_P, [K_init, gamma_init, phi0_init],
                              method='Nelder-Mead',
                              options={'maxiter': 1000})
                if res.fun < best_chi2_at_P:
                    best_chi2_at_P = res.fun
                    best_params_at_P = [P] + list(res.x)
            except:
                pass

        chi2_grid[i] = best_chi2_at_P

        if best_chi2_at_P < best_chi2:
            best_chi2 = best_chi2_at_P
            best_params = best_params_at_P

    return {
        'periods': periods,
        'chi2_grid': chi2_grid,
        'best_period': best_params[0],
        'best_K': best_params[1],
        'best_gamma': best_params[2],
        'best_phi0': best_params[3],
        'best_chi2': best_chi2,
    }


def classify_companion(M2_min):
    """Classify companion based on minimum mass."""
    if M2_min < 1.0:
        return "WHITE DWARF (possible)"
    elif M2_min < 1.4:
        return "WHITE DWARF / NEUTRON STAR boundary"
    elif M2_min < 3.0:
        return "NEUTRON STAR RANGE"
    else:
        return "BLACK HOLE RANGE"


def main():
    print("=" * 70)
    print("PERIOD SEARCH: Combined LAMOST + DESI RV Epochs")
    print("Target: Gaia DR3 3802130935635096832")
    print("=" * 70)
    print()

    # Extract data
    t = np.array([e['mjd'] for e in EPOCHS])
    rv = np.array([e['rv_kms'] for e in EPOCHS])
    rv_err = np.array([e['rv_err_kms'] for e in EPOCHS])

    print("INPUT DATA:")
    print("-" * 50)
    print(f"{'Source':<10} {'MJD':<12} {'RV (km/s)':<12} {'σRV (km/s)':<10}")
    print("-" * 50)
    for e in EPOCHS:
        print(f"{e['source']:<10} {e['mjd']:<12.3f} {e['rv_kms']:<12.2f} {e['rv_err_kms']:<10.2f}")
    print()
    print(f"Total epochs: {len(EPOCHS)}")
    print(f"Baseline: {t.max() - t.min():.1f} days ({(t.max() - t.min())/365.25:.2f} years)")
    print()

    # Run period search
    print("PERIOD SEARCH:")
    print("-" * 50)
    print("Searching P = 5-100 days...")

    results = period_search(t, rv, rv_err, P_range=(5, 100), n_periods=2000)

    P_best = results['best_period']
    K_best = results['best_K']
    gamma_best = results['best_gamma']
    chi2_best = results['best_chi2']

    print()
    print("BEST-FIT PARAMETERS:")
    print("-" * 50)
    print(f"  Period:      P = {P_best:.2f} days")
    print(f"  Semi-amp:    K = {K_best:.2f} km/s")
    print(f"  Systemic:    γ = {gamma_best:.2f} km/s")
    print(f"  Chi-squared: χ² = {chi2_best:.2f}")
    print(f"  Reduced χ²:  χ²_red = {chi2_best/(len(EPOCHS)-4):.2f}")
    print()

    # Mass function and minimum companion mass
    f_M = mass_function(P_best, K_best)
    M1_assumed = 0.5  # dM0 dwarf from LAMOST
    M2_min = minimum_companion_mass(f_M, M1_assumed)

    print("MASS ANALYSIS (assuming M₁ = 0.5 M☉ from LAMOST dM0):")
    print("-" * 50)
    print(f"  Mass function:  f(M) = {f_M:.3f} M☉")
    print(f"  Min companion:  M₂_min = {M2_min:.2f} M☉ (edge-on)")
    print()
    print(f"  Classification: {classify_companion(M2_min)}")
    print()

    # Verification tests
    print("VERIFICATION TESTS:")
    print("-" * 50)

    # Test 1: Same-night stability
    rv_model_e3 = circular_rv_model(t[3], P_best, K_best, gamma_best, results['best_phi0'])
    rv_model_e4 = circular_rv_model(t[4], P_best, K_best, gamma_best, results['best_phi0'])
    expected_change = abs(rv_model_e4 - rv_model_e3)
    observed_change = abs(rv[4] - rv[3])
    print(f"  1. Same-night (21 min): Expected ΔRV = {expected_change:.2f} km/s, Observed = {observed_change:.2f} km/s")
    if observed_change < expected_change + 3:
        print("     → CONSISTENT")
    else:
        print("     → INCONSISTENT")

    # Test 2: TESS ellipsoidal
    R1 = 0.5  # R_sun for M0 dwarf
    M_tot = M1_assumed + M2_min
    a_rsun = ((P_best * DAY_S)**2 * G_SI * M_tot * M_SUN / (4 * np.pi**2))**(1/3) / 6.957e8
    q = M2_min / M1_assumed
    A_ellip = 3.0 * q * (R1/a_rsun)**3 * 1e6  # ppm
    print(f"  2. TESS ellipsoidal: Expected A = {A_ellip:.1f} ppm (detection limit ~460 ppm)")
    if A_ellip < 460:
        print("     → BELOW DETECTION (consistent with non-detection)")
    else:
        print("     → SHOULD BE DETECTED")

    # Test 3: Fit quality
    print(f"  3. Fit quality: χ²_red = {chi2_best/(len(EPOCHS)-4):.2f}")
    if chi2_best/(len(EPOCHS)-4) < 2.0:
        print("     → EXCELLENT FIT")
    else:
        print("     → POOR FIT")

    print()

    # Create visualization
    create_plot(t, rv, rv_err, results, f_M, M2_min)

    # Save results
    output = {
        'target': 'Gaia DR3 3802130935635096832',
        'n_epochs': len(EPOCHS),
        'baseline_days': float(t.max() - t.min()),
        'best_fit': {
            'period_days': float(P_best),
            'K_kms': float(K_best),
            'gamma_kms': float(gamma_best),
            'chi2': float(chi2_best),
            'chi2_reduced': float(chi2_best/(len(EPOCHS)-4)),
        },
        'mass_analysis': {
            'M1_assumed_msun': M1_assumed,
            'mass_function_msun': float(f_M),
            'M2_min_msun': float(M2_min),
            'classification': classify_companion(M2_min),
        },
    }

    with open('period_search_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved: period_search_results.json")

    return results


def create_plot(t, rv, rv_err, results, f_M, M2_min):
    """Create period search visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Chi-squared vs Period
    ax1 = axes[0, 0]
    ax1.plot(results['periods'], results['chi2_grid'], 'b-', lw=0.5)
    ax1.axvline(results['best_period'], color='red', linestyle='--',
                label=f"Best P = {results['best_period']:.2f} d")
    ax1.axhline(results['best_chi2'], color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Period (days)', fontsize=12)
    ax1.set_ylabel('χ²', fontsize=12)
    ax1.set_title('Period Search: χ² vs Period', fontsize=12)
    ax1.legend()
    ax1.set_ylim(0, min(100, np.percentile(results['chi2_grid'], 99)))
    ax1.grid(True, alpha=0.3)

    # Panel 2: RV vs Time with best-fit model
    ax2 = axes[0, 1]
    sources = [e['source'] for e in EPOCHS]
    colors = ['blue' if s == 'LAMOST' else 'red' for s in sources]

    for i in range(len(t)):
        ax2.errorbar(t[i], rv[i], yerr=rv_err[i], fmt='o', color=colors[i],
                    markersize=8, capsize=3,
                    label=sources[i] if sources[i] not in [sources[j] for j in range(i)] else '')

    # Model curve
    t_model = np.linspace(t.min() - 50, t.max() + 50, 1000)
    rv_model = circular_rv_model(t_model, results['best_period'],
                                  results['best_K'], results['best_gamma'],
                                  results['best_phi0'])
    ax2.plot(t_model, rv_model, 'k-', alpha=0.5, lw=1, label='Best-fit model')

    ax2.set_xlabel('MJD', fontsize=12)
    ax2.set_ylabel('RV (km/s)', fontsize=12)
    ax2.set_title(f'RV Time Series (P = {results["best_period"]:.2f} d, K = {results["best_K"]:.1f} km/s)',
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Phase-folded RV
    ax3 = axes[1, 0]
    phase = ((t - t[0]) / results['best_period']) % 1

    for i in range(len(t)):
        ax3.errorbar(phase[i], rv[i], yerr=rv_err[i], fmt='o', color=colors[i],
                    markersize=10, capsize=3)
        # Wrap to show full orbit
        ax3.errorbar(phase[i] + 1, rv[i], yerr=rv_err[i], fmt='o', color=colors[i],
                    markersize=10, capsize=3, alpha=0.3)

    phase_model = np.linspace(0, 2, 200)
    t_model_phase = phase_model * results['best_period'] + t[0]
    rv_model_phase = circular_rv_model(t_model_phase, results['best_period'],
                                        results['best_K'], results['best_gamma'],
                                        results['best_phi0'])
    ax3.plot(phase_model, rv_model_phase, 'k-', alpha=0.7, lw=2)

    ax3.set_xlabel('Orbital Phase', fontsize=12)
    ax3.set_ylabel('RV (km/s)', fontsize=12)
    ax3.set_title('Phase-Folded RV Curve', fontsize=12)
    ax3.set_xlim(0, 2)
    ax3.axvline(1, color='gray', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Mass constraints
    ax4 = axes[1, 1]

    # Plot M2_min vs period for different M1
    P_range = np.linspace(10, 80, 100)
    K = results['best_K']

    for M1, color, label in [(0.4, 'blue', 'M₁=0.4 M☉'),
                              (0.5, 'green', 'M₁=0.5 M☉ (LAMOST dM0)'),
                              (0.7, 'orange', 'M₁=0.7 M☉')]:
        M2_vals = []
        for P in P_range:
            f = mass_function(P, K)
            M2_vals.append(minimum_companion_mass(f, M1))
        ax4.plot(P_range, M2_vals, color=color, label=label, lw=2)

    # Mark best-fit period
    ax4.axvline(results['best_period'], color='red', linestyle='--', lw=2)
    ax4.axhline(M2_min, color='red', linestyle=':', lw=2)
    ax4.plot(results['best_period'], M2_min, 'r*', markersize=20,
             label=f'Best: P={results["best_period"]:.1f}d, M₂={M2_min:.2f}M☉')

    # Companion type regions
    ax4.axhspan(0, 1.4, alpha=0.1, color='blue', label='WD region')
    ax4.axhspan(1.4, 3.0, alpha=0.1, color='orange', label='NS region')
    ax4.axhspan(3.0, 10, alpha=0.1, color='black', label='BH region')

    ax4.set_xlabel('Period (days)', fontsize=12)
    ax4.set_ylabel('M₂_min (M☉)', fontsize=12)
    ax4.set_title(f'Minimum Companion Mass (K = {K:.1f} km/s)', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.set_ylim(0, 6)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('period_search_results.png', dpi=150)
    plt.close()
    print("Saved: period_search_results.png")


if __name__ == "__main__":
    main()
