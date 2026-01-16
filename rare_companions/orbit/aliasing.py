"""
Period aliasing and reliability tests.
"""

import numpy as np
from scipy import optimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import logging

logger = logging.getLogger(__name__)


@dataclass
class InjectionRecoveryResult:
    """Result from injection-recovery test."""
    n_realizations: int
    period_classes: Dict[str, Tuple[float, float]]
    recovery_matrix: np.ndarray  # [true_class, recovered_class]
    recovery_fractions: Dict[str, Dict[str, float]]
    frac_correct_target: float
    alias_fap: float  # fraction of non-target aliased to target
    period_reliability_score: float  # 0-1


def rv_model_circular(t: np.ndarray, P: float, K: float, gamma: float, phi: float) -> np.ndarray:
    """Circular orbit RV model."""
    phase = 2 * np.pi * t / P + phi
    return gamma + K * np.sin(phase)


def _fit_period(mjd: np.ndarray, rv: np.ndarray, rv_err: np.ndarray,
                P_grid: np.ndarray) -> Tuple[float, float]:
    """Find best period from grid search."""
    best_chi2 = np.inf
    best_P = P_grid[0]

    for P in P_grid:
        # Fit K, gamma, phi at fixed P
        def chi2_func(params):
            K, gamma, phi = params
            model = rv_model_circular(mjd, P, K, gamma, phi)
            return np.sum(((rv - model) / rv_err)**2)

        K0 = (np.max(rv) - np.min(rv)) / 2
        gamma0 = np.mean(rv)

        try:
            result = optimize.minimize(
                chi2_func, [K0, gamma0, 0.0],
                method='Nelder-Mead',
                options={'maxiter': 200}
            )
            if result.success and result.fun < best_chi2:
                best_chi2 = result.fun
                best_P = P
        except:
            pass

    return best_P, best_chi2


def _injection_worker(args):
    """Worker for parallel injection-recovery."""
    mjd, rv_err, P_true, K, gamma, phi, P_grid, seed = args

    np.random.seed(seed)

    # Generate synthetic RVs
    rv_true = rv_model_circular(mjd, P_true, K, gamma, phi)
    rv_obs = rv_true + np.random.normal(0, rv_err)

    # Recover period
    P_recovered, _ = _fit_period(mjd, rv_obs, rv_err, P_grid)

    return P_true, P_recovered


def classify_period(P: float, period_classes: Dict[str, Tuple[float, float]]) -> str:
    """Classify period into a period class."""
    for name, (P_min, P_max) in period_classes.items():
        if P_min <= P < P_max:
            return name
    return 'unknown'


def injection_recovery_test(mjd: np.ndarray, rv_err: np.ndarray,
                            target_period_range: Tuple[float, float],
                            K_range: Tuple[float, float] = (50, 150),
                            n_realizations: int = 200,
                            n_workers: int = None) -> InjectionRecoveryResult:
    """
    Run injection-recovery test for period aliasing.

    Parameters
    ----------
    mjd : array
        Observation timestamps (MJD)
    rv_err : array
        RV errors (km/s)
    target_period_range : tuple
        Target period range (P_min, P_max) in days
    K_range : tuple
        Range of semi-amplitudes to inject (km/s)
    n_realizations : int
        Number of realizations per period class
    n_workers : int
        Number of parallel workers

    Returns
    -------
    InjectionRecoveryResult
        Test results
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # Period classes
    period_classes = {
        'short': (0.5, 5.0),
        'intermediate': (5.0, 15.0),
        'target': target_period_range,
        'long': (max(target_period_range[1], 30.0), 200.0)
    }

    # Period grid for recovery
    P_grid = np.logspace(np.log10(0.5), np.log10(200), 150)

    # Generate injection parameters
    all_args = []
    class_names = list(period_classes.keys())

    for class_name, (P_min, P_max) in period_classes.items():
        for i in range(n_realizations):
            P_true = np.random.uniform(P_min, P_max)
            K = np.random.uniform(*K_range)
            gamma = np.random.uniform(-50, 50)
            phi = np.random.uniform(0, 2 * np.pi)
            seed = hash((class_name, i)) % (2**31)

            all_args.append((mjd, rv_err, P_true, K, gamma, phi, P_grid, seed))

    # Run in parallel
    logger.info(f"Running {len(all_args)} injection-recovery realizations...")
    with Pool(n_workers) as pool:
        results = pool.map(_injection_worker, all_args)

    # Analyze results
    n_classes = len(class_names)
    recovery_matrix = np.zeros((n_classes, n_classes))
    recovery_fractions = {c: {c2: 0.0 for c2 in class_names} for c in class_names}

    results_by_class = {c: [] for c in class_names}

    for P_true, P_recovered in results:
        true_class = classify_period(P_true, period_classes)
        rec_class = classify_period(P_recovered, period_classes)

        if true_class in class_names:
            results_by_class[true_class].append((P_true, P_recovered, rec_class))

    # Build matrix
    for i, true_class in enumerate(class_names):
        n_true = len(results_by_class[true_class])
        if n_true == 0:
            continue

        for _, _, rec_class in results_by_class[true_class]:
            if rec_class in class_names:
                j = class_names.index(rec_class)
                recovery_matrix[i, j] += 1

        # Normalize
        recovery_matrix[i, :] /= n_true
        for j, rec_class in enumerate(class_names):
            recovery_fractions[true_class][rec_class] = recovery_matrix[i, j]

    # Compute metrics
    target_idx = class_names.index('target')
    frac_correct_target = recovery_matrix[target_idx, target_idx]

    # Alias FAP: fraction of non-target periods recovered as target
    alias_counts = 0
    alias_total = 0
    for i, class_name in enumerate(class_names):
        if class_name != 'target':
            alias_counts += recovery_matrix[i, target_idx] * len(results_by_class[class_name])
            alias_total += len(results_by_class[class_name])

    alias_fap = alias_counts / alias_total if alias_total > 0 else 0.0

    # Reliability score
    # High if target correctly recovered, low if high aliasing
    reliability = frac_correct_target * (1 - alias_fap)

    return InjectionRecoveryResult(
        n_realizations=n_realizations,
        period_classes=period_classes,
        recovery_matrix=recovery_matrix,
        recovery_fractions=recovery_fractions,
        frac_correct_target=frac_correct_target,
        alias_fap=alias_fap,
        period_reliability_score=reliability
    )


def period_reliability_score(mjd: np.ndarray, rv_err: np.ndarray,
                             best_period: float,
                             n_realizations: int = 100) -> float:
    """
    Quick period reliability score based on injection-recovery.

    Parameters
    ----------
    mjd : array
        Observation timestamps
    rv_err : array
        RV errors (km/s)
    best_period : float
        Best-fit period (days)
    n_realizations : int
        Number of test realizations

    Returns
    -------
    float
        Reliability score (0-1)
    """
    # Define target range around best period
    target_range = (best_period * 0.8, best_period * 1.2)

    result = injection_recovery_test(
        mjd, rv_err, target_range,
        n_realizations=n_realizations // 4  # Per class
    )

    return result.period_reliability_score


def window_function_fap(mjd: np.ndarray, rv: np.ndarray, rv_err: np.ndarray,
                        n_trials: int = 500) -> Tuple[float, float]:
    """
    Compute false alarm probability from window function.

    Parameters
    ----------
    mjd : array
        Observation timestamps
    rv : array
        Observed RVs (km/s)
    rv_err : array
        RV errors (km/s)
    n_trials : int
        Number of noise trials

    Returns
    -------
    fap : float
        False alarm probability
    real_delta_chi2 : float
        Real signal improvement over constant
    """
    # Period grid
    P_grid = np.logspace(np.log10(0.5), np.log10(500), 200)

    # Real data
    real_P, real_chi2 = _fit_period(mjd, rv, rv_err, P_grid)

    # Chi2 for constant model
    weights = 1.0 / rv_err**2
    rv_mean = np.sum(rv * weights) / np.sum(weights)
    chi2_constant = np.sum(((rv - rv_mean) / rv_err)**2)

    real_delta_chi2 = chi2_constant - real_chi2

    # Noise trials
    noise_delta_chi2 = []

    for _ in range(n_trials):
        # Shuffle RVs (preserves errors and timestamps)
        rv_noise = np.random.normal(rv_mean, rv_err)
        _, noise_chi2 = _fit_period(mjd, rv_noise, rv_err, P_grid)

        chi2_const_noise = np.sum(((rv_noise - np.mean(rv_noise)) / rv_err)**2)
        noise_delta_chi2.append(chi2_const_noise - noise_chi2)

    noise_delta_chi2 = np.array(noise_delta_chi2)

    # FAP: fraction of noise trials with delta_chi2 >= real
    fap = np.mean(noise_delta_chi2 >= real_delta_chi2)

    return fap, real_delta_chi2
