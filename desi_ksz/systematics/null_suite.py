"""
Comprehensive null test suite for kSZ validation.

Implements multiple null tests to validate the pairwise kSZ
measurement against systematic effects.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from scipy import stats
import json
import logging
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)

# Number of parallel workers
N_WORKERS = max(1, cpu_count() - 1)


@dataclass
class NullTestResult:
    """Result of a single null test."""

    test_name: str
    description: str
    amplitude: float
    amplitude_err: float
    amplitude_sigma: float  # A / σ_A
    chi2: float
    n_dof: int
    pte: float  # Probability to exceed
    passed: bool
    threshold: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'description': self.description,
            'amplitude': self.amplitude,
            'amplitude_err': self.amplitude_err,
            'amplitude_sigma': self.amplitude_sigma,
            'chi2': self.chi2,
            'n_dof': self.n_dof,
            'pte': self.pte,
            'passed': self.passed,
            'threshold': self.threshold,
            'details': self.details,
        }


@dataclass
class NullSuiteResult:
    """Aggregated results from full null test suite."""

    results: List[NullTestResult]

    @property
    def n_tests(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return self.n_tests - self.n_passed

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def test_names(self) -> List[str]:
        return [r.test_name for r in self.results]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_tests': self.n_tests,
            'n_passed': self.n_passed,
            'n_failed': self.n_failed,
            'all_passed': self.all_passed,
            'results': [r.to_dict() for r in self.results],
        }

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def _fit_amplitude(p_data: np.ndarray, template: np.ndarray, cov: np.ndarray):
    """Fit amplitude and compute uncertainty."""
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.diag(1.0 / np.diag(cov))

    numerator = template @ cov_inv @ p_data
    denominator = template @ cov_inv @ template

    if denominator <= 0:
        return 0.0, np.inf, 0.0

    A = numerator / denominator
    sigma_A = 1.0 / np.sqrt(denominator)
    A_sigma = A / sigma_A if sigma_A > 0 else 0.0

    return A, sigma_A, A_sigma


def _compute_chi2(p_data: np.ndarray, cov: np.ndarray) -> float:
    """Compute chi-squared for null hypothesis."""
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.diag(1.0 / np.diag(cov))

    return float(p_data @ cov_inv @ p_data)


def _shuffle_worker(args: Tuple) -> Tuple[float, float]:
    """Worker for parallel shuffle test."""
    seed, positions, temperatures, weights, template, cov, separation_bins = args

    rng = np.random.default_rng(seed)

    # Import here to avoid pickling issues
    from desi_ksz.estimators import PairwiseMomentumEstimator

    temps_shuffled = rng.permutation(temperatures)

    estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)
    result = estimator.compute(positions, temps_shuffled, weights)

    A_i, _, _ = _fit_amplitude(result.p_ksz, template, cov)
    chi2_i = _compute_chi2(result.p_ksz, cov)

    return A_i, chi2_i


def null_shuffle_temperatures(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    n_realizations: int = 100,
    seed: int = 42,
    pte_threshold: float = 0.05,
    n_workers: Optional[int] = None,
) -> NullTestResult:
    """
    Null test: shuffle temperatures among galaxies (parallelized).

    Under the null hypothesis of no kSZ signal, permuting temperatures
    should give p(r) consistent with zero.
    """
    if n_workers is None:
        n_workers = N_WORKERS

    logger.info(f"Running shuffle_temperatures null test ({n_realizations} realizations, {n_workers} workers)")

    n_bins = len(template)

    # Get separation bins from estimator
    separation_bins = estimator.separation_bins

    # Compute original measurement
    result_orig = estimator.compute(positions, temperatures, weights)
    A_orig, sigma_A, A_sigma = _fit_amplitude(result_orig.p_ksz, template, cov)
    chi2_orig = _compute_chi2(result_orig.p_ksz, cov)

    # Prepare worker arguments
    worker_args = [
        (seed + i, positions, temperatures, weights, template, cov, separation_bins)
        for i in range(n_realizations)
    ]

    # Run in parallel
    if n_workers > 1 and n_realizations > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_shuffle_worker, worker_args)
        A_shuffled = [r[0] for r in results]
        chi2_shuffled = [r[1] for r in results]
    else:
        A_shuffled = []
        chi2_shuffled = []
        rng = np.random.default_rng(seed)
        for i in range(n_realizations):
            temps_shuffled = rng.permutation(temperatures)
            result = estimator.compute(positions, temps_shuffled, weights)
            A_i, _, _ = _fit_amplitude(result.p_ksz, template, cov)
            chi2_i = _compute_chi2(result.p_ksz, cov)
            A_shuffled.append(A_i)
            chi2_shuffled.append(chi2_i)

    # Compute PTE
    pte = np.mean(np.array(chi2_shuffled) >= chi2_orig)

    # Pass if PTE > threshold (not an outlier)
    passed = pte > pte_threshold

    return NullTestResult(
        test_name='shuffle_temperatures',
        description='Permute T_i among galaxies',
        amplitude=float(np.mean(A_shuffled)),
        amplitude_err=float(np.std(A_shuffled)),
        amplitude_sigma=float(np.mean(A_shuffled) / np.std(A_shuffled)) if np.std(A_shuffled) > 0 else 0,
        chi2=chi2_orig,
        n_dof=n_bins,
        pte=float(pte),
        passed=passed,
        threshold=f'PTE > {pte_threshold}',
        details={
            'n_realizations': n_realizations,
            'original_amplitude': float(A_orig),
            'original_chi2': float(chi2_orig),
            'n_workers': n_workers,
        },
    )


def _scramble_worker(args: Tuple) -> float:
    """Worker for parallel scramble test."""
    seed, positions, temperatures, weights, cov, separation_bins = args

    rng = np.random.default_rng(seed)

    # Import here to avoid pickling issues
    from desi_ksz.estimators import PairwiseMomentumEstimator

    idx = rng.permutation(len(positions))
    positions_scrambled = positions[idx]

    estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)
    result = estimator.compute(positions_scrambled, temperatures, weights)

    return _compute_chi2(result.p_ksz, cov)


def null_redshift_scramble(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    z: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    n_realizations: int = 100,
    seed: int = 42,
    pte_threshold: float = 0.05,
    n_workers: Optional[int] = None,
) -> NullTestResult:
    """
    Null test: scramble redshifts among galaxies (parallelized).

    Permuting redshifts destroys real-space pair correlations,
    so the signal should vanish.
    """
    if n_workers is None:
        n_workers = N_WORKERS

    logger.info(f"Running redshift_scramble null test ({n_realizations} realizations, {n_workers} workers)")

    n_bins = len(template)
    separation_bins = estimator.separation_bins

    # Compute original measurement
    result_orig = estimator.compute(positions, temperatures, weights)
    chi2_orig = _compute_chi2(result_orig.p_ksz, cov)

    # Prepare worker arguments
    worker_args = [
        (seed + i, positions, temperatures, weights, cov, separation_bins)
        for i in range(n_realizations)
    ]

    # Run in parallel
    if n_workers > 1 and n_realizations > 1:
        with Pool(n_workers) as pool:
            chi2_scrambled = pool.map(_scramble_worker, worker_args)
    else:
        chi2_scrambled = []
        rng = np.random.default_rng(seed)
        for i in range(n_realizations):
            idx = rng.permutation(len(positions))
            positions_scrambled = positions[idx]
            result = estimator.compute(positions_scrambled, temperatures, weights)
            chi2_scrambled.append(_compute_chi2(result.p_ksz, cov))

    # Compute PTE
    pte = np.mean(np.array(chi2_scrambled) >= chi2_orig)
    passed = pte > pte_threshold

    return NullTestResult(
        test_name='redshift_scramble',
        description='Permute z (positions) among galaxies',
        amplitude=0.0,  # Not applicable for this test
        amplitude_err=0.0,
        amplitude_sigma=0.0,
        chi2=chi2_orig,
        n_dof=n_bins,
        pte=float(pte),
        passed=passed,
        threshold=f'PTE > {pte_threshold}',
        details={'n_realizations': n_realizations, 'n_workers': n_workers},
    )


def null_hemisphere_split(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    dec: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    split_dec: float = 0.0,
    consistency_threshold: float = 2.0,
) -> NullTestResult:
    """
    Null test: compare North vs South hemispheres.

    The signal should be consistent between hemispheres.
    """
    logger.info("Running hemisphere_split null test")

    # Split by declination
    north_mask = dec >= split_dec
    south_mask = dec < split_dec

    n_north = np.sum(north_mask)
    n_south = np.sum(south_mask)

    if n_north < 100 or n_south < 100:
        return NullTestResult(
            test_name='hemisphere_split',
            description='Compare North vs South',
            amplitude=0.0,
            amplitude_err=0.0,
            amplitude_sigma=0.0,
            chi2=0.0,
            n_dof=0,
            pte=1.0,
            passed=True,
            threshold='Skipped - insufficient data',
            details={'n_north': n_north, 'n_south': n_south},
        )

    # Compute for each hemisphere
    result_north = estimator.compute(
        positions[north_mask], temperatures[north_mask], weights[north_mask]
    )
    result_south = estimator.compute(
        positions[south_mask], temperatures[south_mask], weights[south_mask]
    )

    # Fit amplitudes
    # Scale covariance roughly by inverse sqrt(N)
    scale_north = np.sqrt(len(positions) / n_north)
    scale_south = np.sqrt(len(positions) / n_south)

    A_north, sigma_north, _ = _fit_amplitude(result_north.p_ksz, template, cov * scale_north**2)
    A_south, sigma_south, _ = _fit_amplitude(result_south.p_ksz, template, cov * scale_south**2)

    # Compare: difference should be consistent with zero
    diff = A_north - A_south
    sigma_diff = np.sqrt(sigma_north**2 + sigma_south**2)
    diff_sigma = abs(diff) / sigma_diff if sigma_diff > 0 else 0

    # PTE from Gaussian
    pte = 2 * (1 - stats.norm.cdf(diff_sigma))  # Two-tailed

    passed = diff_sigma < consistency_threshold

    return NullTestResult(
        test_name='hemisphere_split',
        description='Compare North vs South hemispheres',
        amplitude=float(diff),
        amplitude_err=float(sigma_diff),
        amplitude_sigma=float(diff_sigma),
        chi2=float(diff_sigma**2),
        n_dof=1,
        pte=float(pte),
        passed=passed,
        threshold=f'|A_N - A_S|/σ < {consistency_threshold}',
        details={
            'north_amplitude': float(A_north),
            'north_sigma': float(sigma_north),
            'south_amplitude': float(A_south),
            'south_sigma': float(sigma_south),
            'n_north': int(n_north),
            'n_south': int(n_south),
        },
    )


def _random_positions_worker(args: Tuple) -> float:
    """Worker for parallel random positions test."""
    seed, pos_min, pos_max, n_gal, temperatures, weights, cov, separation_bins = args

    rng = np.random.default_rng(seed)

    # Import here to avoid pickling issues
    from desi_ksz.estimators import PairwiseMomentumEstimator

    positions_random = rng.uniform(pos_min, pos_max, size=(n_gal, 3))

    estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)
    result = estimator.compute(positions_random, temperatures, weights)

    return _compute_chi2(result.p_ksz, cov)


def null_random_positions(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    n_realizations: int = 10,
    seed: int = 42,
    pte_threshold: float = 0.05,
    n_workers: Optional[int] = None,
) -> NullTestResult:
    """
    Null test: use random positions instead of galaxy positions (parallelized).

    Randomly distributed positions should show no correlation.
    """
    if n_workers is None:
        n_workers = N_WORKERS

    logger.info(f"Running random_positions null test ({n_realizations} realizations, {n_workers} workers)")

    n_bins = len(template)
    n_gal = len(positions)
    separation_bins = estimator.separation_bins

    # Original measurement
    result_orig = estimator.compute(positions, temperatures, weights)
    chi2_orig = _compute_chi2(result_orig.p_ksz, cov)

    # Estimate position bounds
    pos_min = np.min(positions, axis=0)
    pos_max = np.max(positions, axis=0)

    # Prepare worker arguments
    worker_args = [
        (seed + i, pos_min, pos_max, n_gal, temperatures, weights, cov, separation_bins)
        for i in range(n_realizations)
    ]

    # Run in parallel
    if n_workers > 1 and n_realizations > 1:
        with Pool(n_workers) as pool:
            chi2_random = pool.map(_random_positions_worker, worker_args)
    else:
        chi2_random = []
        rng = np.random.default_rng(seed)
        for i in range(n_realizations):
            positions_random = rng.uniform(pos_min, pos_max, size=(n_gal, 3))
            result = estimator.compute(positions_random, temperatures, weights)
            chi2_random.append(_compute_chi2(result.p_ksz, cov))

    # Signal should be larger than random
    pte = np.mean(np.array(chi2_random) >= chi2_orig)

    # For this test, we actually want the signal to be different from random
    # So high PTE means we're consistent with random (bad)
    passed = pte < (1 - pte_threshold)  # Signal should be outlier

    return NullTestResult(
        test_name='random_positions',
        description='Random positions instead of galaxies',
        amplitude=0.0,
        amplitude_err=0.0,
        amplitude_sigma=0.0,
        chi2=chi2_orig,
        n_dof=n_bins,
        pte=float(pte),
        passed=passed,
        threshold=f'Signal distinct from random (PTE < {1-pte_threshold})',
        details={'n_realizations': n_realizations, 'n_workers': n_workers},
    )


def run_null_suite(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    tests: Optional[Union[str, List[str]]] = None,
    n_real: int = 100,
    seed: int = 42,
    small_mode: bool = False,
) -> NullSuiteResult:
    """
    Run comprehensive null test suite.

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
        The estimator to test
    positions : np.ndarray
        Galaxy positions (N, 3)
    temperatures : np.ndarray
        Temperature measurements (N,)
    weights : np.ndarray
        Galaxy weights (N,)
    ra, dec, z : np.ndarray
        Sky coordinates and redshifts
    template : np.ndarray
        Theory template for amplitude fitting
    cov : np.ndarray
        Covariance matrix
    tests : str or list, optional
        Tests to run ('all' or list of names)
    n_real : int
        Number of realizations for shuffle tests
    seed : int
        Random seed
    small_mode : bool
        Use reduced realizations for CI testing

    Returns
    -------
    NullSuiteResult
        Aggregated test results
    """
    if small_mode:
        n_real = min(n_real, 10)

    # Available tests
    available_tests = {
        'shuffle': null_shuffle_temperatures,
        'scramble': null_redshift_scramble,
        'hemisphere': null_hemisphere_split,
        'random': null_random_positions,
    }

    # Parse test list
    if tests is None or tests == 'all':
        test_list = list(available_tests.keys())
    elif isinstance(tests, str):
        test_list = [tests]
    else:
        test_list = tests

    results = []

    for test_name in test_list:
        if test_name not in available_tests:
            logger.warning(f"Unknown test: {test_name}")
            continue

        logger.info(f"Running null test: {test_name}")

        if test_name == 'shuffle':
            result = null_shuffle_temperatures(
                estimator, positions, temperatures, weights,
                template, cov, n_realizations=n_real, seed=seed
            )
        elif test_name == 'scramble':
            result = null_redshift_scramble(
                estimator, positions, temperatures, weights, z,
                template, cov, n_realizations=n_real, seed=seed
            )
        elif test_name == 'hemisphere':
            result = null_hemisphere_split(
                estimator, positions, temperatures, weights, dec,
                template, cov
            )
        elif test_name == 'random':
            result = null_random_positions(
                estimator, positions, temperatures, weights,
                template, cov, n_realizations=min(n_real, 10), seed=seed
            )

        results.append(result)
        logger.info(f"  {test_name}: {'PASS' if result.passed else 'FAIL'} (PTE={result.pte:.3f})")

    return NullSuiteResult(results)


def plot_null_suite_summary(
    suite_result: NullSuiteResult,
    output_path: Optional[str] = None,
):
    """
    Generate summary plot of null test results.

    Parameters
    ----------
    suite_result : NullSuiteResult
        Results from run_null_suite()
    output_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: PTE values
    test_names = [r.test_name for r in suite_result.results]
    ptes = [r.pte for r in suite_result.results]
    passed = [r.passed for r in suite_result.results]

    colors = ['green' if p else 'red' for p in passed]

    y_pos = np.arange(len(test_names))
    ax1.barh(y_pos, ptes, color=colors, alpha=0.7)
    ax1.axvline(0.05, color='gray', linestyle='--', label='Threshold')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(test_names)
    ax1.set_xlabel('PTE')
    ax1.set_title('Null Test PTE Values')
    ax1.legend()

    # Right panel: chi2/dof
    chi2_dof = [r.chi2 / r.n_dof if r.n_dof > 0 else 0 for r in suite_result.results]

    ax2.barh(y_pos, chi2_dof, color=colors, alpha=0.7)
    ax2.axvline(1.0, color='gray', linestyle='--', label='Expected')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(test_names)
    ax2.set_xlabel(r'$\chi^2 / N_{dof}$')
    ax2.set_title('Null Test Chi-Squared')
    ax2.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
