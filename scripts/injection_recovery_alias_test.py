#!/usr/bin/env python3
"""
injection_recovery_alias_test.py - Injection-Recovery Alias Analysis

Tests whether the ~20-25 day period could be an alias of a different true period
given the specific DESI+LAMOST sampling. Goes beyond noise-only FAP to test
if short/long periods can masquerade as intermediate periods.

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

# RV epochs from the real data
RV_EPOCHS = [
    {'mjd': 57457.0, 'rv_err': 2.79},      # LAMOST
    {'mjd': 59568.48825, 'rv_err': 0.55},  # DESI
    {'mjd': 59605.38003, 'rv_err': 0.83},  # DESI
    {'mjd': 59607.37393, 'rv_err': 1.06},  # DESI
    {'mjd': 59607.38852, 'rv_err': 1.11},  # DESI
]

# Period classes
PERIOD_CLASSES = {
    'short': (1.0, 5.0),
    'intermediate': (5.0, 15.0),
    'target': (15.0, 30.0),
    'long': (30.0, 100.0)
}

# Simulation parameters
N_REALIZATIONS_PER_CLASS = 500  # Reduced from 2000 - sufficient for statistics
K_RANGE = (80, 110)  # km/s
GAMMA_RANGE = (-60, -20)  # km/s

# Period search grid
PERIOD_GRID = np.linspace(1.0, 100.0, 150)  # Reduced from 500 - sufficient resolution


def rv_model_circular(t, P, K, gamma, phi):
    """Circular orbit RV model."""
    phase = 2 * np.pi * t / P + phi
    return gamma + K * np.sin(phase)


def fit_circular_orbit(mjd, rv, rv_err, P_guess):
    """Fit circular orbit at fixed period."""
    def chi2(params):
        K, gamma, phi = params
        model = rv_model_circular(mjd, P_guess, K, gamma, phi)
        return np.sum(((rv - model) / rv_err)**2)

    K0 = (np.max(rv) - np.min(rv)) / 2
    gamma0 = np.mean(rv)
    phi0 = 0.0

    try:
        result = optimize.minimize(chi2, [K0, gamma0, phi0],
                                   method='Nelder-Mead',
                                   options={'maxiter': 500})
        if result.success:
            return {'success': True, 'chi2': result.fun, 'params': result.x}
    except:
        pass

    return {'success': False}


def period_search(mjd, rv, rv_err, period_grid):
    """Search over periods to find best fit."""
    best_chi2 = np.inf
    best_P = None

    for P in period_grid:
        result = fit_circular_orbit(mjd, rv, rv_err, P)
        if result['success'] and result['chi2'] < best_chi2:
            best_chi2 = result['chi2']
            best_P = P

    return best_P, best_chi2


def classify_period(P):
    """Classify a period into one of the defined classes."""
    if P is None:
        return 'failed'

    for class_name, (P_min, P_max) in PERIOD_CLASSES.items():
        if P_min <= P < P_max:
            return class_name

    if P >= 100:
        return 'very_long'
    return 'unknown'


def simulate_one_realization(args):
    """Simulate one injection-recovery realization."""
    mjd, rv_err, P_true, K_true, gamma_true, phi_true, period_grid, seed = args

    np.random.seed(seed)

    # Generate synthetic RVs
    rv_true = rv_model_circular(mjd, P_true, K_true, gamma_true, phi_true)
    rv_obs = rv_true + np.random.normal(0, rv_err)

    # Period search
    P_best, chi2_best = period_search(mjd, rv_obs, rv_err, period_grid)

    # Classify recovered period
    class_best = classify_period(P_best)

    return {
        'P_true': P_true,
        'P_best': P_best,
        'chi2_best': chi2_best,
        'class_true': classify_period(P_true),
        'class_best': class_best
    }


def run_simulations():
    """Run injection-recovery simulations for all period classes."""
    print("=" * 70)
    print("INJECTION-RECOVERY ALIAS TEST")
    print("=" * 70)
    print()

    mjd = np.array([e['mjd'] for e in RV_EPOCHS])
    rv_err = np.array([e['rv_err'] for e in RV_EPOCHS])

    print(f"Epochs: {len(mjd)}")
    print(f"Baseline: {mjd.max() - mjd.min():.1f} days")
    print()

    results_by_class = {}
    all_results = []

    n_workers = min(cpu_count() - 1, 8)
    print(f"Using {n_workers} parallel workers")
    print()

    for class_name, (P_min, P_max) in PERIOD_CLASSES.items():
        print(f"Simulating {class_name} periods ({P_min}-{P_max} days)...")

        # Generate random parameters
        np.random.seed(42 + hash(class_name) % 1000)

        args_list = []
        for i in range(N_REALIZATIONS_PER_CLASS):
            P_true = np.random.uniform(P_min, P_max)
            K_true = np.random.uniform(*K_RANGE)
            gamma_true = np.random.uniform(*GAMMA_RANGE)
            phi_true = np.random.uniform(0, 2 * np.pi)
            seed = i + hash(class_name) % 10000

            args_list.append((mjd, rv_err, P_true, K_true, gamma_true, phi_true, PERIOD_GRID, seed))

        # Run in parallel
        with Pool(n_workers) as pool:
            class_results = pool.map(simulate_one_realization, args_list)

        # Analyze results
        n_total = len(class_results)
        recovery_counts = {c: 0 for c in list(PERIOD_CLASSES.keys()) + ['very_long', 'failed', 'unknown']}

        for r in class_results:
            recovery_counts[r['class_best']] = recovery_counts.get(r['class_best'], 0) + 1
            all_results.append(r)

        results_by_class[class_name] = {
            'n_total': n_total,
            'recovery_counts': recovery_counts,
            'recovery_fractions': {k: v / n_total for k, v in recovery_counts.items()}
        }

        print(f"  N = {n_total}")
        for cls, count in recovery_counts.items():
            if count > 0:
                print(f"    Recovered as {cls}: {count} ({count/n_total*100:.1f}%)")

        print()

    return results_by_class, all_results


def compute_alias_metrics(results_by_class):
    """Compute summary metrics for aliasing."""
    metrics = {}

    # Key metric: what fraction of non-target periods get recovered as target?
    alias_to_target = []
    for class_name in ['short', 'intermediate', 'long']:
        frac = results_by_class[class_name]['recovery_fractions'].get('target', 0)
        alias_to_target.append(frac)
        metrics[f'frac_{class_name}_recovered_as_target'] = frac

    metrics['avg_alias_fap_for_target'] = np.mean(alias_to_target)

    # What fraction of target periods are correctly recovered?
    frac_target_correct = results_by_class['target']['recovery_fractions'].get('target', 0)
    metrics['frac_target_recovered_correctly'] = frac_target_correct

    # What fraction of target periods are aliased to other classes?
    for other_class in ['short', 'intermediate', 'long']:
        frac = results_by_class['target']['recovery_fractions'].get(other_class, 0)
        metrics[f'frac_target_aliased_to_{other_class}'] = frac

    return metrics


def create_alias_plot(results_by_class, all_results):
    """Create diagnostic plot for alias analysis."""
    print("Creating diagnostic plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Histogram of recovered periods by true period class
    ax1 = axes[0]

    colors = {
        'short': '#e74c3c',
        'intermediate': '#f39c12',
        'target': '#27ae60',
        'long': '#3498db'
    }

    P_best_by_class = {cls: [] for cls in PERIOD_CLASSES.keys()}
    for r in all_results:
        cls = r['class_true']
        if r['P_best'] is not None:
            P_best_by_class[cls].append(r['P_best'])

    bins = np.linspace(0, 100, 50)
    bottom = np.zeros(len(bins) - 1)

    for cls in ['short', 'intermediate', 'target', 'long']:
        P_best = P_best_by_class[cls]
        if P_best:
            counts, _ = np.histogram(P_best, bins=bins)
            ax1.bar(bins[:-1], counts, width=np.diff(bins), bottom=bottom,
                   color=colors[cls], alpha=0.7, label=f'True: {cls}', edgecolor='black', linewidth=0.5)
            bottom += counts

    ax1.axvspan(15, 30, color='green', alpha=0.1, label='Target range')
    ax1.set_xlabel('Recovered Period (days)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of Recovered Periods by True Period Class', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 100)

    # Panel 2: Recovery matrix
    ax2 = axes[1]

    classes = ['short', 'intermediate', 'target', 'long']
    matrix = np.zeros((4, 4))

    for i, true_cls in enumerate(classes):
        for j, rec_cls in enumerate(classes):
            frac = results_by_class[true_cls]['recovery_fractions'].get(rec_cls, 0)
            matrix[i, j] = frac * 100

    im = ax2.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')

    # Add text annotations
    for i in range(4):
        for j in range(4):
            color = 'white' if matrix[i, j] > 50 else 'black'
            ax2.text(j, i, f'{matrix[i, j]:.1f}%', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(classes, fontsize=10)
    ax2.set_yticklabels(classes, fontsize=10)
    ax2.set_xlabel('Recovered Period Class', fontsize=11)
    ax2.set_ylabel('True Period Class', fontsize=11)
    ax2.set_title('Period Recovery Matrix\n(% of true class recovered as given class)', fontsize=12)

    plt.colorbar(im, ax=ax2, shrink=0.8, label='Recovery %')

    plt.tight_layout()
    plt.savefig('injection_recovery_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: injection_recovery_plot.png")


def main():
    print("=" * 70)
    print("INJECTION-RECOVERY ALIAS ANALYSIS")
    print("=" * 70)
    print(f"Target: Gaia DR3 3802130935635096832")
    print()
    print(f"Period classes:")
    for cls, (P_min, P_max) in PERIOD_CLASSES.items():
        print(f"  {cls}: {P_min}-{P_max} days")
    print()
    print(f"Realizations per class: {N_REALIZATIONS_PER_CLASS}")
    print(f"K range: {K_RANGE[0]}-{K_RANGE[1]} km/s")
    print()

    # Run simulations
    results_by_class, all_results = run_simulations()

    # Compute metrics
    metrics = compute_alias_metrics(results_by_class)

    print("=" * 70)
    print("ALIAS METRICS")
    print("=" * 70)
    print()

    print(f"Fraction of TARGET periods recovered correctly: {metrics['frac_target_recovered_correctly']*100:.1f}%")
    print()
    print("Alias FAP (fraction of non-target periods recovered as target):")
    for cls in ['short', 'intermediate', 'long']:
        frac = metrics[f'frac_{cls}_recovered_as_target']
        print(f"  {cls} → target: {frac*100:.1f}%")
    print()
    print(f"Average alias FAP for target period: {metrics['avg_alias_fap_for_target']*100:.1f}%")
    print()

    # Create plot
    create_alias_plot(results_by_class, all_results)

    # Save results
    output = {
        'target': 'Gaia DR3 3802130935635096832',
        'analysis': 'injection_recovery_alias_test',
        'config': {
            'period_classes': {k: list(v) for k, v in PERIOD_CLASSES.items()},
            'n_realizations_per_class': N_REALIZATIONS_PER_CLASS,
            'K_range_kms': list(K_RANGE),
            'gamma_range_kms': list(GAMMA_RANGE),
            'n_period_grid': len(PERIOD_GRID)
        },
        'results_by_class': results_by_class,
        'metrics': metrics,
        'interpretation': {
            'frac_target_recovered_correctly': metrics['frac_target_recovered_correctly'],
            'avg_alias_fap': metrics['avg_alias_fap_for_target'],
            'short_to_target_alias': metrics['frac_short_recovered_as_target'],
            'conclusion': None
        }
    }

    # Interpretation
    if metrics['frac_target_recovered_correctly'] > 0.5:
        if metrics['avg_alias_fap_for_target'] < 0.1:
            conclusion = "Target period range (15-30d) is strongly favored; low alias probability"
        else:
            conclusion = "Target period recoverable but some aliasing present"
    else:
        if metrics['avg_alias_fap_for_target'] > 0.3:
            conclusion = "Significant aliasing; period is not well constrained by sampling"
        else:
            conclusion = "Poor recovery; period uncertain"

    output['interpretation']['conclusion'] = conclusion
    print(f"Conclusion: {conclusion}")
    print()

    with open('injection_recovery_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved: injection_recovery_results.json")

    return output


if __name__ == "__main__":
    main()
