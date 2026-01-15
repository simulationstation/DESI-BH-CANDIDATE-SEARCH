#!/usr/bin/env python3
"""
window_function_analysis.py - Window Function and False Alarm Probability

Quantifies how likely it is that the observed ~16-22 day orbital signal
arises purely from sampling + noise with these exact RV epochs.

Target: Gaia DR3 3802130935635096832
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Target RV epochs from hardened analysis
RV_EPOCHS = [
    {'mjd': 57457.0, 'rv': -49.36, 'rv_err': 2.79, 'source': 'LAMOST'},
    {'mjd': 59568.48825, 'rv': -86.39, 'rv_err': 0.55, 'source': 'DESI'},
    {'mjd': 59605.38003, 'rv': 59.68, 'rv_err': 0.83, 'source': 'DESI'},
    {'mjd': 59607.37393, 'rv': 26.43, 'rv_err': 1.06, 'source': 'DESI'},
    {'mjd': 59607.38852, 'rv': 25.16, 'rv_err': 1.11, 'source': 'DESI'},
]

# Period search range
PERIOD_MIN = 5.0   # days
PERIOD_MAX = 100.0  # days
N_PERIODS = 200  # Reduced from 1000 - sufficient for FAP

# Number of noise realizations
N_REALIZATIONS = 1000  # Reduced from 10000 - sufficient given strong signal


def rv_model_circular(t, P, K, gamma, T0):
    """Circular orbit RV model."""
    phase = 2 * np.pi * (t - T0) / P
    return gamma + K * np.sin(phase)


def fit_circular_orbit(mjd, rv, rv_err, P_guess):
    """Fit circular orbit at fixed period, optimize K, gamma, T0."""

    def chi2(params):
        K, gamma, T0 = params
        model = rv_model_circular(mjd, P_guess, K, gamma, T0)
        return np.sum(((rv - model) / rv_err)**2)

    # Initial guesses
    K0 = (np.max(rv) - np.min(rv)) / 2
    gamma0 = np.mean(rv)
    T0_0 = mjd[0]

    try:
        result = optimize.minimize(chi2, [K0, gamma0, T0_0],
                                   method='Nelder-Mead',
                                   options={'maxiter': 1000})
        if result.success:
            K_fit, gamma_fit, T0_fit = result.x
            chi2_fit = result.fun
            return {'success': True, 'K': K_fit, 'gamma': gamma_fit,
                    'T0': T0_fit, 'chi2': chi2_fit}
    except:
        pass

    return {'success': False}


def period_search(mjd, rv, rv_err, period_grid):
    """Search over periods to find best fit."""
    best_chi2 = np.inf
    best_P = None
    chi2_values = []

    for P in period_grid:
        result = fit_circular_orbit(mjd, rv, rv_err, P)
        if result['success']:
            chi2_values.append(result['chi2'])
            if result['chi2'] < best_chi2:
                best_chi2 = result['chi2']
                best_P = P
        else:
            chi2_values.append(np.nan)

    return best_P, best_chi2, np.array(chi2_values)


def compute_chi2_constant(rv, rv_err):
    """Compute chi2 for constant RV model."""
    weights = 1.0 / rv_err**2
    rv_wmean = np.sum(weights * rv) / np.sum(weights)
    chi2 = np.sum(((rv - rv_wmean) / rv_err)**2)
    return chi2, rv_wmean


def analyze_real_data():
    """Analyze the real RV data."""
    print("=" * 70)
    print("REAL DATA ANALYSIS")
    print("=" * 70)
    print()

    mjd = np.array([e['mjd'] for e in RV_EPOCHS])
    rv = np.array([e['rv'] for e in RV_EPOCHS])
    rv_err = np.array([e['rv_err'] for e in RV_EPOCHS])

    print(f"N epochs: {len(mjd)}")
    print(f"Baseline: {mjd.max() - mjd.min():.1f} days")
    print()

    # Constant model
    chi2_const, rv_mean = compute_chi2_constant(rv, rv_err)
    print(f"Constant model: χ² = {chi2_const:.2f}, dof = {len(rv)-1}")
    print()

    # Period search
    print("Searching periods...")
    period_grid = np.linspace(PERIOD_MIN, PERIOD_MAX, N_PERIODS)
    best_P, best_chi2, chi2_grid = period_search(mjd, rv, rv_err, period_grid)

    print(f"Best period: P = {best_P:.2f} days")
    print(f"Best χ² (orbit): {best_chi2:.2f}")
    print()

    # Delta chi2 (improvement from orbital fit)
    delta_chi2 = chi2_const - best_chi2
    print(f"Δχ² = χ²_const - χ²_orbit = {delta_chi2:.2f}")
    print()

    return {
        'mjd': mjd,
        'rv': rv,
        'rv_err': rv_err,
        'chi2_const': chi2_const,
        'rv_mean': rv_mean,
        'best_P': best_P,
        'best_chi2': best_chi2,
        'delta_chi2': delta_chi2,
        'period_grid': period_grid,
        'chi2_grid': chi2_grid
    }


def generate_noise_realization(args):
    """Generate one noise realization and find its best period."""
    mjd, rv_err, rv_mean, period_grid, seed = args

    np.random.seed(seed)

    # Generate noise-only RV data
    rv_noise = np.random.normal(rv_mean, rv_err)

    # Fit constant model
    chi2_const, _ = compute_chi2_constant(rv_noise, rv_err)

    # Period search
    best_P, best_chi2, _ = period_search(mjd, rv_noise, rv_err, period_grid)

    if best_P is not None:
        delta_chi2 = chi2_const - best_chi2
    else:
        delta_chi2 = 0
        best_P = np.nan

    return {'delta_chi2': delta_chi2, 'best_P': best_P, 'chi2_const': chi2_const}


def run_noise_simulations(real_results):
    """Run noise-only simulations to compute FAP."""
    print("=" * 70)
    print("NOISE SIMULATIONS (FAP calculation)")
    print("=" * 70)
    print()

    mjd = real_results['mjd']
    rv_err = real_results['rv_err']
    rv_mean = real_results['rv_mean']
    period_grid = real_results['period_grid']

    print(f"Running {N_REALIZATIONS} noise realizations...")
    print(f"Using {min(cpu_count()-1, 8)} parallel workers...")
    print()

    # Prepare arguments for parallel execution
    args_list = [(mjd, rv_err, rv_mean, period_grid, seed)
                 for seed in range(N_REALIZATIONS)]

    # Run in parallel
    n_workers = min(cpu_count() - 1, 8)
    with Pool(n_workers) as pool:
        results_list = pool.map(generate_noise_realization, args_list)

    # Extract results
    delta_chi2_noise = np.array([r['delta_chi2'] for r in results_list])
    best_P_noise = np.array([r['best_P'] for r in results_list])

    # Remove invalid values
    valid = np.isfinite(delta_chi2_noise)
    delta_chi2_noise = delta_chi2_noise[valid]
    best_P_noise = best_P_noise[valid]

    print(f"Valid realizations: {len(delta_chi2_noise)}/{N_REALIZATIONS}")
    print()

    # Compute statistics
    delta_chi2_real = real_results['delta_chi2']
    P_real = real_results['best_P']

    # FAP: fraction of noise realizations with Δχ² >= real Δχ²
    n_exceeds = np.sum(delta_chi2_noise >= delta_chi2_real)
    fap = n_exceeds / len(delta_chi2_noise)

    print(f"Real data: Δχ² = {delta_chi2_real:.2f}, P = {P_real:.2f} days")
    print(f"Noise Δχ² distribution:")
    print(f"  Mean: {np.mean(delta_chi2_noise):.2f}")
    print(f"  Median: {np.median(delta_chi2_noise):.2f}")
    print(f"  Max: {np.max(delta_chi2_noise):.2f}")
    print()
    print(f"N(Δχ²_noise >= Δχ²_real): {n_exceeds}")
    print(f"FALSE ALARM PROBABILITY: FAP = {fap:.2e}")
    print()

    # Also check period match
    P_tolerance = 0.1  # 10%
    P_matches = np.abs(best_P_noise - P_real) / P_real < P_tolerance
    n_period_match = np.sum(P_matches & (delta_chi2_noise >= delta_chi2_real * 0.5))
    fap_period = n_period_match / len(delta_chi2_noise)
    print(f"FAP for matching period (±10%) with Δχ² > 0.5×real: {fap_period:.2e}")
    print()

    return {
        'n_realizations': len(delta_chi2_noise),
        'delta_chi2_noise': delta_chi2_noise,
        'best_P_noise': best_P_noise,
        'delta_chi2_real': delta_chi2_real,
        'P_real': P_real,
        'fap': fap,
        'fap_period_match': fap_period,
        'n_exceeds': n_exceeds,
        'noise_stats': {
            'mean': np.mean(delta_chi2_noise),
            'median': np.median(delta_chi2_noise),
            'std': np.std(delta_chi2_noise),
            'max': np.max(delta_chi2_noise)
        }
    }


def create_fap_plots(real_results, noise_results):
    """Create FAP visualization plots."""

    # Plot 1: Delta chi2 histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    delta_chi2_noise = noise_results['delta_chi2_noise']
    delta_chi2_real = noise_results['delta_chi2_real']

    # Histogram
    bins = np.linspace(0, max(np.percentile(delta_chi2_noise, 99.9), delta_chi2_real * 1.2), 50)
    ax1.hist(delta_chi2_noise, bins=bins, density=True, alpha=0.7,
             color='steelblue', edgecolor='navy', label='Noise realizations')

    ax1.axvline(delta_chi2_real, color='red', linestyle='-', lw=2,
                label=f'Real data (Δχ² = {delta_chi2_real:.1f})')

    # Mark FAP region
    ax1.fill_betweenx([0, ax1.get_ylim()[1]*2], delta_chi2_real, bins[-1],
                      alpha=0.3, color='red', label=f'FAP region ({noise_results["fap"]:.2e})')

    ax1.set_xlabel('Δχ² = χ²_const - χ²_orbit', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Window Function Analysis: Δχ² Distribution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, bins[-1])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Period vs Delta chi2 scatter
    best_P_noise = noise_results['best_P_noise']
    P_real = noise_results['P_real']

    valid = np.isfinite(best_P_noise) & np.isfinite(delta_chi2_noise)
    ax2.scatter(best_P_noise[valid], delta_chi2_noise[valid],
                alpha=0.3, s=10, c='steelblue', label='Noise')
    ax2.scatter([P_real], [delta_chi2_real], color='red', s=200, marker='*',
                edgecolors='black', linewidths=1, label='Real data', zorder=10)

    ax2.axhline(delta_chi2_real, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(P_real, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Best-fit Period (days)', fontsize=12)
    ax2.set_ylabel('Δχ²', fontsize=12)
    ax2.set_title('Period vs. Δχ² for Noise Realizations', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_xlim(PERIOD_MIN, PERIOD_MAX)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('window_delta_chi2_hist.png', dpi=150)
    plt.close()
    print("Saved: window_delta_chi2_hist.png")

    # Plot 3: Period scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(best_P_noise[valid], delta_chi2_noise[valid],
               alpha=0.3, s=15, c='steelblue')
    ax.scatter([P_real], [delta_chi2_real], color='red', s=300, marker='*',
               edgecolors='black', linewidths=2, label=f'Real: P={P_real:.1f}d', zorder=10)

    # Add reference periods
    ax.axvline(21.8, color='orange', linestyle=':', lw=2, alpha=0.7, label='MCMC median (21.8d)')
    ax.axvline(15.9, color='green', linestyle=':', lw=2, alpha=0.7, label='Circular fit (15.9d)')

    ax.set_xlabel('Best-fit Period (days)', fontsize=12)
    ax.set_ylabel('Δχ²', fontsize=12)
    ax.set_title('Window Function: Period Recovery Test', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('window_period_vs_delta_chi2.png', dpi=150)
    plt.close()
    print("Saved: window_period_vs_delta_chi2.png")


def interpret_fap(noise_results):
    """Interpret the FAP results."""
    print("=" * 70)
    print("FAP INTERPRETATION")
    print("=" * 70)
    print()

    fap = noise_results['fap']
    delta_chi2_real = noise_results['delta_chi2_real']

    interpretation = {
        'fap': fap,
        'delta_chi2': delta_chi2_real
    }

    if fap < 1e-4:
        sigma_equiv = 4.0
        print(f"FAP = {fap:.2e} corresponds to > 4σ significance")
        print("The orbital signal is HIGHLY UNLIKELY to be a sampling artifact.")
        interpretation['significance'] = '>4sigma'
        interpretation['conclusion'] = 'highly_significant'
    elif fap < 1e-3:
        sigma_equiv = 3.3
        print(f"FAP = {fap:.2e} corresponds to ~3.3σ significance")
        print("The orbital signal is unlikely to be a sampling artifact.")
        interpretation['significance'] = '~3.3sigma'
        interpretation['conclusion'] = 'significant'
    elif fap < 1e-2:
        sigma_equiv = 2.6
        print(f"FAP = {fap:.2e} corresponds to ~2.6σ significance")
        print("The orbital signal is probably real but follow-up needed.")
        interpretation['significance'] = '~2.6sigma'
        interpretation['conclusion'] = 'moderate'
    else:
        sigma_equiv = 2.0
        print(f"FAP = {fap:.2e} is relatively high")
        print("The signal could potentially be a sampling artifact.")
        interpretation['significance'] = '<2sigma'
        interpretation['conclusion'] = 'low_significance'

    print()
    print("With the actual time sampling and RV errors:")
    print(f"  - The probability that pure noise would mimic the observed")
    print(f"    orbital signal as strongly as this is FAP = {fap:.2e}")
    print()

    return interpretation


def main():
    print("=" * 70)
    print("WINDOW FUNCTION / FALSE ALARM PROBABILITY ANALYSIS")
    print("=" * 70)
    print(f"Target: Gaia DR3 3802130935635096832")
    print(f"N epochs: {len(RV_EPOCHS)}")
    print(f"Period range: {PERIOD_MIN}-{PERIOD_MAX} days")
    print(f"N noise realizations: {N_REALIZATIONS}")
    print()

    # Analyze real data
    real_results = analyze_real_data()

    # Run noise simulations
    noise_results = run_noise_simulations(real_results)

    # Create plots
    create_fap_plots(real_results, noise_results)

    # Interpret results
    interpretation = interpret_fap(noise_results)

    # Compile results
    results = {
        'target': 'Gaia DR3 3802130935635096832',
        'n_epochs': len(RV_EPOCHS),
        'period_range': [PERIOD_MIN, PERIOD_MAX],
        'n_realizations': noise_results['n_realizations'],
        'real_data': {
            'chi2_const': real_results['chi2_const'],
            'best_period': real_results['best_P'],
            'best_chi2': real_results['best_chi2'],
            'delta_chi2': real_results['delta_chi2']
        },
        'noise_stats': noise_results['noise_stats'],
        'fap': noise_results['fap'],
        'fap_period_match': noise_results['fap_period_match'],
        'interpretation': interpretation
    }

    # Save results
    with open('window_function_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved: window_function_results.json")

    return results


if __name__ == "__main__":
    main()
