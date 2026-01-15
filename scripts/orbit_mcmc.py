#!/usr/bin/env python3
"""
orbit_mcmc.py - Bayesian orbit fitting using MCMC

Samples the posterior distribution over orbital parameters:
- P (period)
- K (semi-amplitude)
- e (eccentricity)
- omega (argument of periastron)
- T0 (time of periastron)
- gamma (systemic velocity)

Uses emcee for MCMC sampling.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from multiprocessing import Pool, cpu_count

# Physical constants
G_SI = 6.674e-11  # m^3 kg^-1 s^-2
M_SUN = 1.989e30  # kg
DAY_S = 86400     # seconds per day

# RV epochs (LAMOST + DESI)
EPOCHS = [
    {"source": "LAMOST", "mjd": 57457.0, "rv_kms": -49.36, "rv_err_kms": 2.79},
    {"source": "DESI", "mjd": 59568.488, "rv_kms": -86.39, "rv_err_kms": 0.55},
    {"source": "DESI", "mjd": 59605.380, "rv_kms": 59.68, "rv_err_kms": 0.83},
    {"source": "DESI", "mjd": 59607.374, "rv_kms": 26.43, "rv_err_kms": 1.06},
    {"source": "DESI", "mjd": 59607.389, "rv_kms": 25.16, "rv_err_kms": 1.11},
]

# Extract arrays
t_obs = np.array([e['mjd'] for e in EPOCHS])
rv_obs = np.array([e['rv_kms'] for e in EPOCHS])
rv_err = np.array([e['rv_err_kms'] for e in EPOCHS])


def rv_model_keplerian(t, P, K, e, omega, T0, gamma):
    """
    Keplerian radial velocity model.

    RV(t) = gamma + K * (cos(nu + omega) + e * cos(omega))

    where nu is the true anomaly computed from the eccentric anomaly.

    Parameters
    ----------
    t : array-like
        Times (MJD)
    P : float
        Orbital period (days)
    K : float
        RV semi-amplitude (km/s)
    e : float
        Eccentricity [0, 1)
    omega : float
        Argument of periastron (radians)
    T0 : float
        Time of periastron passage (MJD)
    gamma : float
        Systemic velocity (km/s)

    Returns
    -------
    rv : array-like
        Model RV (km/s)
    """
    # Mean anomaly
    M = 2 * np.pi * (t - T0) / P
    M = M % (2 * np.pi)

    # Solve Kepler's equation for eccentric anomaly E
    # M = E - e*sin(E)
    E = solve_kepler(M, e)

    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))

    # Radial velocity
    rv = gamma + K * (np.cos(nu + omega) + e * np.cos(omega))

    return rv


def solve_kepler(M, e, tol=1e-10, max_iter=100):
    """
    Solve Kepler's equation M = E - e*sin(E) for E.

    Uses Newton-Raphson iteration.
    """
    M = np.atleast_1d(M)
    E = M.copy()  # Initial guess

    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E = E + dE
        if np.all(np.abs(dE) < tol):
            break

    return E


def log_likelihood(theta):
    """
    Gaussian log-likelihood for RV data.

    L = -0.5 * sum((rv_obs - rv_model)^2 / sigma^2 + log(2*pi*sigma^2))
    """
    P, K, e, omega, T0, gamma = theta

    # Compute model
    rv_model = rv_model_keplerian(t_obs, P, K, e, omega, T0, gamma)

    # Chi-squared
    chi2 = np.sum(((rv_obs - rv_model) / rv_err)**2)

    # Log-likelihood (ignoring constant term)
    log_L = -0.5 * chi2

    return log_L


def log_prior(theta):
    """
    Log-prior for orbital parameters.

    Priors:
    - P: log-uniform on [5, 200] days
    - K: uniform on [10, 200] km/s
    - e: uniform on [0, 0.8]
    - omega: uniform on [0, 2*pi]
    - T0: uniform within one period of first observation
    - gamma: uniform on [-200, 100] km/s
    """
    P, K, e, omega, T0, gamma = theta

    # Bounds check
    if not (5 < P < 200):
        return -np.inf
    if not (10 < K < 200):
        return -np.inf
    if not (0 <= e < 0.8):
        return -np.inf
    if not (0 <= omega < 2 * np.pi):
        return -np.inf
    if not (t_obs.min() - 200 < T0 < t_obs.max() + 200):
        return -np.inf
    if not (-200 < gamma < 100):
        return -np.inf

    # Log-uniform prior on P
    log_prior_P = -np.log(P)

    # Flat priors on other parameters (contribute constant)
    return log_prior_P


def log_probability(theta):
    """Log-posterior = log-prior + log-likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def mass_function(P_days, K_kms):
    """Compute mass function f(M) in solar masses."""
    P_s = P_days * DAY_S
    K_ms = K_kms * 1000
    f_M = (P_s * K_ms**3) / (2 * np.pi * G_SI)
    return f_M / M_SUN


def min_companion_mass(f_M, M1):
    """Solve for minimum companion mass (sin i = 1)."""
    from scipy.optimize import brentq

    def equation(M2):
        return M2**3 / (M1 + M2)**2 - f_M

    try:
        return brentq(equation, 0.01, 100)
    except:
        return np.nan


def run_mcmc(n_walkers=32, n_steps=5000, n_burn=1000, n_processes=None):
    """
    Run MCMC sampling of orbital posterior.
    """
    import emcee

    print("=" * 70)
    print("BAYESIAN ORBIT MCMC")
    print("=" * 70)
    print()

    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    print(f"MCMC Configuration:")
    print(f"  Walkers: {n_walkers}")
    print(f"  Steps: {n_steps}")
    print(f"  Burn-in: {n_burn}")
    print(f"  Processes: {n_processes}")
    print()

    # Initialize walkers
    ndim = 6  # P, K, e, omega, T0, gamma

    # Starting position based on rough estimates
    P_init = 20.0
    K_init = 80.0
    e_init = 0.1
    omega_init = np.pi
    T0_init = t_obs.mean()
    gamma_init = -20.0

    # Initialize with small scatter around starting point
    pos = np.zeros((n_walkers, ndim))
    pos[:, 0] = P_init + np.random.uniform(-10, 10, n_walkers)      # P
    pos[:, 1] = K_init + np.random.uniform(-20, 20, n_walkers)      # K
    pos[:, 2] = e_init + np.random.uniform(0, 0.3, n_walkers)       # e
    pos[:, 3] = omega_init + np.random.uniform(-1, 1, n_walkers)    # omega
    pos[:, 4] = T0_init + np.random.uniform(-50, 50, n_walkers)     # T0
    pos[:, 5] = gamma_init + np.random.uniform(-20, 20, n_walkers)  # gamma

    # Ensure all starting positions are valid
    for i in range(n_walkers):
        while not np.isfinite(log_probability(pos[i])):
            pos[i, 0] = np.random.uniform(10, 100)
            pos[i, 1] = np.random.uniform(50, 150)
            pos[i, 2] = np.random.uniform(0, 0.5)
            pos[i, 3] = np.random.uniform(0, 2*np.pi)
            pos[i, 4] = t_obs.mean() + np.random.uniform(-100, 100)
            pos[i, 5] = np.random.uniform(-100, 50)

    print("Running MCMC...")

    # Run sampler
    with Pool(n_processes) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(pos, n_steps, progress=True)

    print()
    print("MCMC complete.")
    print()

    # Get samples (discard burn-in)
    samples = sampler.get_chain(discard=n_burn, flat=True)
    log_probs = sampler.get_log_prob(discard=n_burn, flat=True)

    print(f"Total samples: {len(samples)}")
    print()

    # Parameter names
    labels = ['P (days)', 'K (km/s)', 'e', 'ω (rad)', 'T0 (MJD)', 'γ (km/s)']

    # Summary statistics
    print("POSTERIOR SUMMARY:")
    print("-" * 70)
    print(f"{'Parameter':<15} {'Median':<12} {'16%':<12} {'84%':<12} {'Mean':<12}")
    print("-" * 70)

    results = {}
    for i, label in enumerate(labels):
        med = np.median(samples[:, i])
        lo = np.percentile(samples[:, i], 16)
        hi = np.percentile(samples[:, i], 84)
        mean = np.mean(samples[:, i])
        print(f"{label:<15} {med:<12.3f} {lo:<12.3f} {hi:<12.3f} {mean:<12.3f}")
        results[label] = {'median': med, 'p16': lo, 'p84': hi, 'mean': mean}

    print()

    # Derived quantities
    print("DERIVED QUANTITIES:")
    print("-" * 70)

    # Mass function for each sample
    f_M_samples = np.array([mass_function(s[0], s[1]) for s in samples])
    M2_min_samples = np.array([min_companion_mass(f, 0.5) for f in f_M_samples])

    f_M_med = np.median(f_M_samples)
    f_M_lo = np.percentile(f_M_samples, 16)
    f_M_hi = np.percentile(f_M_samples, 84)

    M2_med = np.nanmedian(M2_min_samples)
    M2_lo = np.nanpercentile(M2_min_samples, 16)
    M2_hi = np.nanpercentile(M2_min_samples, 84)

    print(f"Mass function f(M): {f_M_med:.3f} [{f_M_lo:.3f}, {f_M_hi:.3f}] M_sun")
    print(f"M2_min (M1=0.5): {M2_med:.2f} [{M2_lo:.2f}, {M2_hi:.2f}] M_sun")
    print()

    results['f_M'] = {'median': f_M_med, 'p16': f_M_lo, 'p84': f_M_hi}
    results['M2_min'] = {'median': M2_med, 'p16': M2_lo, 'p84': M2_hi}

    # Probabilities
    print("COMPANION TYPE PROBABILITIES (M1 = 0.5 M_sun):")
    print("-" * 70)

    P_NS = np.sum(M2_min_samples > 1.4) / len(M2_min_samples) * 100
    P_BH = np.sum(M2_min_samples > 3.0) / len(M2_min_samples) * 100

    print(f"Pr(M2_min > 1.4 M_sun): {P_NS:.1f}% (neutron star or heavier)")
    print(f"Pr(M2_min > 3.0 M_sun): {P_BH:.1f}% (black hole)")
    print()

    results['probabilities'] = {
        'P_M2_gt_1.4': P_NS,
        'P_M2_gt_3.0': P_BH,
    }

    # Short period rejection
    print("SHORT PERIOD REJECTION:")
    print("-" * 70)

    P_samples = samples[:, 0]
    P_lt_2d = np.sum(P_samples < 2) / len(P_samples) * 100
    P_lt_5d = np.sum(P_samples < 5) / len(P_samples) * 100
    P_lt_10d = np.sum(P_samples < 10) / len(P_samples) * 100

    print(f"Pr(P < 2 days): {P_lt_2d:.2f}%")
    print(f"Pr(P < 5 days): {P_lt_5d:.2f}%")
    print(f"Pr(P < 10 days): {P_lt_10d:.2f}%")
    print()

    results['short_period_rejection'] = {
        'P_lt_2d': P_lt_2d,
        'P_lt_5d': P_lt_5d,
        'P_lt_10d': P_lt_10d,
    }

    # Check for multimodality
    print("PERIOD DISTRIBUTION:")
    print("-" * 70)

    # Find peaks in period histogram
    P_hist, P_edges = np.histogram(P_samples, bins=50)
    P_centers = (P_edges[:-1] + P_edges[1:]) / 2
    peak_idx = np.argmax(P_hist)
    P_mode = P_centers[peak_idx]

    print(f"Period mode: {P_mode:.1f} days")
    print(f"Period range (2.5-97.5%): [{np.percentile(P_samples, 2.5):.1f}, {np.percentile(P_samples, 97.5):.1f}] days")
    print()

    results['period_distribution'] = {
        'mode': P_mode,
        'p2.5': np.percentile(P_samples, 2.5),
        'p97.5': np.percentile(P_samples, 97.5),
    }

    # Best-fit parameters (maximum likelihood)
    best_idx = np.argmax(log_probs)
    best_params = samples[best_idx]
    print("BEST-FIT (Maximum Likelihood):")
    print("-" * 70)
    for i, label in enumerate(labels):
        print(f"  {label}: {best_params[i]:.4f}")
    print(f"  log(L): {log_probs[best_idx]:.2f}")
    print()

    results['best_fit'] = {labels[i]: best_params[i] for i in range(len(labels))}
    results['best_fit']['log_L'] = log_probs[best_idx]

    # Save results
    output = {
        'mcmc_config': {
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'n_burn': n_burn,
            'n_samples': len(samples),
        },
        'results': results,
    }

    with open('orbit_mcmc_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print("Saved: orbit_mcmc_results.json")

    # Create plots
    create_corner_plot(samples, labels)
    create_period_histogram(P_samples, M2_min_samples)
    create_rv_posterior_plot(samples, best_params)

    return samples, results


def create_corner_plot(samples, labels):
    """Create corner plot of posterior."""
    try:
        import corner

        # Use only P, K, e for clarity
        fig = corner.corner(
            samples[:, :3],
            labels=labels[:3],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
        )
        fig.savefig('orbit_mcmc_corner.png', dpi=150)
        plt.close()
        print("Saved: orbit_mcmc_corner.png")
    except ImportError:
        print("WARNING: corner package not installed, skipping corner plot")


def create_period_histogram(P_samples, M2_samples):
    """Create period and mass histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Period histogram
    ax1.hist(P_samples, bins=50, density=True, alpha=0.7, color='blue')
    ax1.axvline(np.median(P_samples), color='red', linestyle='--',
                label=f'Median = {np.median(P_samples):.1f} d')
    ax1.axvline(np.percentile(P_samples, 16), color='red', linestyle=':', alpha=0.5)
    ax1.axvline(np.percentile(P_samples, 84), color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Period (days)', fontsize=12)
    ax1.set_ylabel('Probability density', fontsize=12)
    ax1.set_title('Orbital Period Posterior', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # M2_min histogram
    valid_M2 = M2_samples[np.isfinite(M2_samples)]
    ax2.hist(valid_M2, bins=50, density=True, alpha=0.7, color='green')
    ax2.axvline(np.median(valid_M2), color='red', linestyle='--',
                label=f'Median = {np.median(valid_M2):.2f} M☉')
    ax2.axvline(1.4, color='purple', linestyle='--', linewidth=2, label='NS threshold')
    ax2.axvline(3.0, color='black', linestyle='--', linewidth=2, label='BH threshold')
    ax2.set_xlabel('M2_min (M☉)', fontsize=12)
    ax2.set_ylabel('Probability density', fontsize=12)
    ax2.set_title('Minimum Companion Mass Posterior (M1=0.5 M☉)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('orbit_mcmc_posteriors.png', dpi=150)
    plt.close()
    print("Saved: orbit_mcmc_posteriors.png")


def create_rv_posterior_plot(samples, best_params):
    """Create RV curve with posterior samples."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Draw random samples from posterior
    n_draw = 100
    idx = np.random.choice(len(samples), n_draw, replace=False)

    # Time grid
    t_grid = np.linspace(t_obs.min() - 50, t_obs.max() + 50, 500)

    # Plot posterior samples
    for i in idx:
        P, K, e, omega, T0, gamma = samples[i]
        rv_model = rv_model_keplerian(t_grid, P, K, e, omega, T0, gamma)
        ax1.plot(t_grid, rv_model, 'b-', alpha=0.05, lw=0.5)

    # Plot best-fit
    P, K, e, omega, T0, gamma = best_params
    rv_best = rv_model_keplerian(t_grid, P, K, e, omega, T0, gamma)
    ax1.plot(t_grid, rv_best, 'r-', lw=2, label='Best fit')

    # Plot data
    colors = ['blue' if e['source'] == 'LAMOST' else 'red' for e in EPOCHS]
    for i, ep in enumerate(EPOCHS):
        ax1.errorbar(ep['mjd'], ep['rv_kms'], yerr=ep['rv_err_kms'],
                    fmt='o', color=colors[i], markersize=10, capsize=5,
                    label=ep['source'] if i == 0 or (i == 1 and ep['source'] != EPOCHS[0]['source']) else '')

    ax1.set_xlabel('MJD', fontsize=12)
    ax1.set_ylabel('RV (km/s)', fontsize=12)
    ax1.set_title('RV Time Series with Posterior Samples', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Phase-folded plot using best-fit period
    phase = ((t_obs - best_params[4]) / best_params[0]) % 1

    # Plot data
    for i, ep in enumerate(EPOCHS):
        ph = ((ep['mjd'] - best_params[4]) / best_params[0]) % 1
        ax2.errorbar(ph, ep['rv_kms'], yerr=ep['rv_err_kms'],
                    fmt='o', color=colors[i], markersize=10, capsize=5)
        ax2.errorbar(ph + 1, ep['rv_kms'], yerr=ep['rv_err_kms'],
                    fmt='o', color=colors[i], markersize=10, capsize=5, alpha=0.3)

    # Plot best-fit model
    phase_grid = np.linspace(0, 2, 200)
    t_phase = phase_grid * best_params[0] + best_params[4]
    rv_phase = rv_model_keplerian(t_phase, *best_params)
    ax2.plot(phase_grid, rv_phase, 'r-', lw=2)

    ax2.set_xlabel('Orbital Phase', fontsize=12)
    ax2.set_ylabel('RV (km/s)', fontsize=12)
    ax2.set_title(f'Phase-Folded RV (P = {best_params[0]:.1f} d)', fontsize=12)
    ax2.set_xlim(0, 2)
    ax2.axvline(1, color='gray', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('orbit_mcmc_rv.png', dpi=150)
    plt.close()
    print("Saved: orbit_mcmc_rv.png")


if __name__ == "__main__":
    # Check for emcee
    try:
        import emcee
        print(f"Using emcee version {emcee.__version__}")
    except ImportError:
        print("ERROR: emcee not installed. Install with: pip install emcee")
        exit(1)

    # Run MCMC
    samples, results = run_mcmc(n_walkers=32, n_steps=5000, n_burn=1000)
