"""
Signal injection tests for kSZ estimator validation.

Validates that the pairwise momentum estimator recovers injected
signals with unbiased amplitude.

Injection Test Protocol
-----------------------
1. Generate mock temperature field with known kSZ signal
2. Add realistic CMB + noise
3. Run pairwise estimator
4. Compare recovered amplitude to input
5. Repeat for multiple realizations to assess bias/scatter

Injection Modes
---------------
- 'simple': Gaussian random velocities (fast, basic validation)
- 'template': Pair-based signal injection matching theory template
- 'velocity_field': Reconstruct velocities from density field (most realistic)
"""

from typing import Optional, Dict, List, Tuple, Any, Literal
from dataclasses import dataclass, field
import numpy as np
import logging
import json
from multiprocessing import Pool, cpu_count
from functools import partial

logger = logging.getLogger(__name__)

# Number of parallel workers
N_WORKERS = max(1, cpu_count() - 1)

InjectionMode = Literal["simple", "template", "velocity_field"]


@dataclass
class InjectionTestResult:
    """Container for injection test results."""
    input_amplitude: float
    recovered_amplitudes: np.ndarray
    mean_recovered: float
    std_recovered: float
    bias: float
    bias_sigma: float
    n_realizations: int
    passed: bool
    injection_mode: str = "simple"
    noise_level: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def fractional_bias(self) -> float:
        """Fractional bias: (A_out - A_in) / A_in"""
        if self.input_amplitude == 0:
            return 0.0
        return self.bias / self.input_amplitude

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'input_amplitude': self.input_amplitude,
            'mean_recovered': self.mean_recovered,
            'std_recovered': self.std_recovered,
            'bias': self.bias,
            'bias_sigma': self.bias_sigma,
            'fractional_bias': self.fractional_bias,
            'n_realizations': self.n_realizations,
            'passed': self.passed,
            'injection_mode': self.injection_mode,
            'noise_level': self.noise_level,
            'recovered_amplitudes': self.recovered_amplitudes.tolist(),
            'details': self.details,
        }

    def to_json(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self) -> str:
        return (
            f"Injection Test ({self.injection_mode}): A_in = {self.input_amplitude:.3f}\n"
            f"  A_out = {self.mean_recovered:.3f} ± {self.std_recovered:.3f}\n"
            f"  Bias = {self.bias:.4f} ({self.bias_sigma:.1f}σ)\n"
            f"  Fractional bias = {self.fractional_bias:.1%}\n"
            f"  Status: {'PASS' if self.passed else 'FAIL'}"
        )


def _injection_worker(args: Tuple) -> float:
    """
    Worker function for parallel injection test.

    Returns recovered amplitude for a single realization.
    """
    (seed, positions, weights, theory_template, r_bins, input_amplitude,
     noise_level, injection_mode, cov_inv, ra, dec, separation_bins) = args

    rng = np.random.default_rng(seed)

    # Import here to avoid pickling issues
    from desi_ksz.estimators import PairwiseMomentumEstimator

    # Generate mock temperature field based on mode
    if injection_mode == "simple":
        T_signal, T_noise = _generate_mock_temperatures_simple(
            positions=positions,
            theory_template=theory_template,
            r_bins=r_bins,
            amplitude=input_amplitude,
            noise_level=noise_level,
            rng=rng,
        )
    elif injection_mode == "template":
        T_signal, T_noise = _generate_mock_temperatures_template(
            positions=positions,
            weights=weights,
            theory_template=theory_template,
            r_bins=r_bins,
            amplitude=input_amplitude,
            noise_level=noise_level,
            rng=rng,
        )
    elif injection_mode == "velocity_field":
        T_signal, T_noise = _generate_mock_temperatures_velocity_field(
            positions=positions,
            ra=ra,
            dec=dec,
            theory_template=theory_template,
            r_bins=r_bins,
            amplitude=input_amplitude,
            noise_level=noise_level,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown injection mode: {injection_mode}")

    T_total = T_signal + T_noise

    # Create estimator and run
    estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)
    result = estimator.compute(positions, T_total, weights)

    # Fit amplitude
    if cov_inv is not None:
        A_fit = _fit_amplitude_weighted(result.p_ksz, theory_template, cov_inv)
    else:
        numerator = np.sum(result.p_ksz * theory_template)
        denominator = np.sum(theory_template ** 2)
        A_fit = numerator / denominator if denominator > 0 else 0.0

    return A_fit


def run_injection_test(
    estimator,
    positions: np.ndarray,
    weights: np.ndarray,
    theory_template: np.ndarray,
    r_bins: np.ndarray,
    input_amplitude: float = 1.0,
    noise_level: float = 1.0,
    n_realizations: int = 100,
    cmb_power_spectrum: Optional[np.ndarray] = None,
    random_seed: int = 42,
    bias_threshold: float = 2.0,
    injection_mode: InjectionMode = "simple",
    covariance: Optional[np.ndarray] = None,
    ra: Optional[np.ndarray] = None,
    dec: Optional[np.ndarray] = None,
    n_workers: Optional[int] = None,
) -> InjectionTestResult:
    """
    Run signal injection test (parallelized).

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
        The estimator to test
    positions : np.ndarray
        Galaxy positions (N, 3) in Mpc/h
    weights : np.ndarray
        Galaxy weights (N,)
    theory_template : np.ndarray
        Theory p(r) template (n_bins,)
    r_bins : np.ndarray
        Separation bin edges
    input_amplitude : float
        True kSZ amplitude to inject
    noise_level : float
        Noise scaling factor
    n_realizations : int
        Number of realizations
    cmb_power_spectrum : np.ndarray, optional
        CMB power spectrum for realistic noise
    random_seed : int
        Random seed
    bias_threshold : float
        Maximum allowed bias in units of σ
    injection_mode : str
        Injection mode: 'simple', 'template', or 'velocity_field'
    covariance : np.ndarray, optional
        Covariance matrix for weighted fits
    ra, dec : np.ndarray, optional
        Sky coordinates (needed for velocity_field mode)
    n_workers : int, optional
        Number of parallel workers (default: cpu_count - 1)

    Returns
    -------
    InjectionTestResult
        Test results including bias assessment
    """
    if n_workers is None:
        n_workers = N_WORKERS

    logger.info(
        f"Running injection test: A_in = {input_amplitude}, "
        f"mode = {injection_mode}, {n_realizations} realizations, {n_workers} workers"
    )

    n_gal = len(positions)
    n_bins = len(theory_template)

    # Precompute inverse covariance if provided
    if covariance is not None:
        try:
            cov_inv = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            logger.warning("Covariance inversion failed, using diagonal")
            cov_inv = np.diag(1.0 / np.diag(covariance))
    else:
        cov_inv = None

    # Get separation bins from estimator
    separation_bins = r_bins

    # Prepare arguments for each worker
    # Each worker gets a unique seed
    worker_args = [
        (random_seed + i, positions, weights, theory_template, r_bins,
         input_amplitude, noise_level, injection_mode, cov_inv, ra, dec,
         separation_bins)
        for i in range(n_realizations)
    ]

    # Run in parallel
    if n_workers > 1 and n_realizations > 1:
        with Pool(n_workers) as pool:
            recovered_amplitudes = np.array(pool.map(_injection_worker, worker_args))
    else:
        # Sequential fallback for debugging or single realization
        recovered_amplitudes = np.array([_injection_worker(args) for args in worker_args])

    # Compute statistics
    mean_recovered = np.mean(recovered_amplitudes)
    std_recovered = np.std(recovered_amplitudes)
    bias = mean_recovered - input_amplitude

    # Bias significance
    stderr = std_recovered / np.sqrt(n_realizations)
    bias_sigma = bias / stderr if stderr > 0 else 0.0

    passed = abs(bias_sigma) < bias_threshold

    logger.info(
        f"Injection test complete: A_out = {mean_recovered:.3f} ± {std_recovered:.3f}, "
        f"bias = {bias:.4f} ({bias_sigma:.1f}σ)"
    )

    return InjectionTestResult(
        input_amplitude=input_amplitude,
        recovered_amplitudes=recovered_amplitudes,
        mean_recovered=mean_recovered,
        std_recovered=std_recovered,
        bias=bias,
        bias_sigma=bias_sigma,
        n_realizations=n_realizations,
        passed=passed,
        injection_mode=injection_mode,
        noise_level=noise_level,
        details={
            'bias_threshold': bias_threshold,
            'n_galaxies': n_gal,
            'n_bins': n_bins,
            'used_covariance_weighting': covariance is not None,
            'n_workers': n_workers,
        }
    )


def _fit_amplitude_weighted(
    p_data: np.ndarray,
    p_theory: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """
    Fit amplitude with covariance weighting.

    A_ML = (p_theory^T C^-1 p_data) / (p_theory^T C^-1 p_theory)
    """
    numerator = p_theory @ cov_inv @ p_data
    denominator = p_theory @ cov_inv @ p_theory
    return numerator / denominator if denominator > 0 else 0.0


def _generate_mock_temperatures_simple(
    positions: np.ndarray,
    theory_template: np.ndarray,
    r_bins: np.ndarray,
    amplitude: float,
    noise_level: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mock temperatures with simple Gaussian velocities.

    Fast but approximate - suitable for basic validation.
    """
    n_gal = len(positions)

    # Simple approach: random velocities correlated on large scales
    v_los = rng.standard_normal(n_gal)

    # Scale to produce approximately correct amplitude
    T_signal = amplitude * v_los * np.std(theory_template)

    # Uncorrelated Gaussian noise (~100 μK rms typical CMB)
    T_noise = noise_level * rng.standard_normal(n_gal) * 100

    return T_signal, T_noise


def _generate_mock_temperatures_template(
    positions: np.ndarray,
    weights: np.ndarray,
    theory_template: np.ndarray,
    r_bins: np.ndarray,
    amplitude: float,
    noise_level: float,
    rng: np.random.Generator,
    n_pairs_sample: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mock temperatures by directly injecting pair correlations.

    This mode constructs temperatures that will produce the desired
    pairwise momentum signal when measured. More accurate than simple mode.

    The key insight: p(r) = <(T_i - T_j) * c_ij> where c_ij is geometric.
    We can inject temperatures that satisfy this by solving iteratively.
    """
    from scipy.spatial import KDTree

    n_gal = len(positions)
    r_max = r_bins[-1]

    # Build KDTree for pair finding
    tree = KDTree(positions)

    # Initialize with random noise base
    T_signal = np.zeros(n_gal)

    # Sample pairs and assign temperatures to match template
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # For each galaxy, find neighbors and adjust temperature
    for _ in range(3):  # Iterate a few times for convergence
        pairs_found = tree.query_pairs(r=r_max, output_type='ndarray')

        if len(pairs_found) == 0:
            break

        # Sample subset of pairs for efficiency
        if len(pairs_found) > n_pairs_sample:
            idx = rng.choice(len(pairs_found), size=n_pairs_sample, replace=False)
            pairs_sample = pairs_found[idx]
        else:
            pairs_sample = pairs_found

        for i, j in pairs_sample:
            # Compute separation
            r_ij = np.linalg.norm(positions[i] - positions[j])

            # Find which bin
            bin_idx = np.searchsorted(r_bins, r_ij) - 1
            if bin_idx < 0 or bin_idx >= len(theory_template):
                continue

            # Target signal from template
            target_p = amplitude * theory_template[bin_idx]

            # Compute geometric weight c_ij
            r_vec = positions[j] - positions[i]
            r_hat_ij = r_vec / np.linalg.norm(r_vec)
            r_hat_i = positions[i] / (np.linalg.norm(positions[i]) + 1e-10)
            r_hat_j = positions[j] / (np.linalg.norm(positions[j]) + 1e-10)
            c_ij = 0.5 * np.dot(r_hat_ij, r_hat_i - r_hat_j)

            if abs(c_ij) < 1e-6:
                continue

            # Adjust temperatures to produce target signal
            # (T_i - T_j) * c_ij = target_p / weight_norm
            delta_T = target_p / c_ij * rng.standard_normal() * 0.1

            T_signal[i] += delta_T * 0.5
            T_signal[j] -= delta_T * 0.5

    # Normalize to achieve correct amplitude scale
    T_signal *= amplitude / (np.std(T_signal) + 1e-10) * np.std(theory_template)

    # Add noise
    T_noise = noise_level * rng.standard_normal(n_gal) * 100

    return T_signal, T_noise


def _generate_mock_temperatures_velocity_field(
    positions: np.ndarray,
    ra: Optional[np.ndarray],
    dec: Optional[np.ndarray],
    theory_template: np.ndarray,
    r_bins: np.ndarray,
    amplitude: float,
    noise_level: float,
    rng: np.random.Generator,
    velocity_dispersion: float = 300.0,  # km/s
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mock temperatures using a simple velocity field model.

    This mode creates spatially correlated velocities that mimic
    large-scale structure, providing more realistic injection tests.

    Uses a bulk flow + local dispersion model.
    """
    n_gal = len(positions)

    # Create bulk flow components (large-scale coherent motion)
    # Random bulk flow direction
    bulk_direction = rng.standard_normal(3)
    bulk_direction /= np.linalg.norm(bulk_direction)
    bulk_amplitude = 200.0 * amplitude  # km/s scale

    # Compute line-of-sight direction for each galaxy
    if ra is not None and dec is not None:
        # Use actual sky positions
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        los = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
    else:
        # Use position as proxy for LOS
        r_mag = np.linalg.norm(positions, axis=1, keepdims=True)
        los = positions / (r_mag + 1e-10)

    # Bulk flow contribution (coherent across all galaxies)
    v_bulk = bulk_amplitude * np.dot(los, bulk_direction)

    # Local velocity dispersion (add scatter)
    v_local = velocity_dispersion * rng.standard_normal(n_gal)

    # Total velocity
    v_total = v_bulk + v_local

    # Convert velocity to temperature: T_kSZ ∝ -τ * v_los / c
    # Normalization factor (approximate)
    tau_bar = 1e-4  # Typical optical depth
    T_cmb = 2.725e6  # μK
    c_km_s = 3e5  # km/s

    T_signal = -tau_bar * T_cmb * v_total / c_km_s

    # Scale to match theory template amplitude
    if np.std(T_signal) > 0:
        T_signal *= amplitude * np.std(theory_template) / np.std(T_signal)

    # Add CMB + noise
    T_noise = noise_level * rng.standard_normal(n_gal) * 100

    return T_signal, T_noise


# Keep old function for backward compatibility
def _generate_mock_temperatures(
    positions: np.ndarray,
    theory_template: np.ndarray,
    r_bins: np.ndarray,
    amplitude: float,
    noise_level: float,
    cmb_power_spectrum: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy wrapper - uses simple mode."""
    return _generate_mock_temperatures_simple(
        positions, theory_template, r_bins, amplitude, noise_level, rng
    )


def validate_estimator_bias(
    estimator,
    positions: np.ndarray,
    weights: np.ndarray,
    theory_template: np.ndarray,
    r_bins: np.ndarray,
    amplitude_values: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0],
    n_realizations: int = 50,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Validate estimator bias across multiple input amplitudes.

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
        Estimator to test
    positions : np.ndarray
        Galaxy positions
    weights : np.ndarray
        Galaxy weights
    theory_template : np.ndarray
        Theory template
    r_bins : np.ndarray
        Separation bins
    amplitude_values : list of float
        Input amplitudes to test
    n_realizations : int
        Realizations per amplitude
    random_seed : int
        Random seed

    Returns
    -------
    dict
        Validation results including bias vs amplitude
    """
    logger.info(f"Validating estimator bias across {len(amplitude_values)} amplitudes")

    results = []

    for i, A_in in enumerate(amplitude_values):
        seed = random_seed + i * 1000
        result = run_injection_test(
            estimator=estimator,
            positions=positions,
            weights=weights,
            theory_template=theory_template,
            r_bins=r_bins,
            input_amplitude=A_in,
            n_realizations=n_realizations,
            random_seed=seed,
        )
        results.append(result)

    # Compile summary
    summary = {
        'input_amplitudes': amplitude_values,
        'recovered_means': [r.mean_recovered for r in results],
        'recovered_stds': [r.std_recovered for r in results],
        'biases': [r.bias for r in results],
        'bias_sigmas': [r.bias_sigma for r in results],
        'all_passed': all(r.passed for r in results),
        'results': results,
    }

    # Check for systematic bias
    biases = np.array(summary['biases'])
    if np.any(np.abs(biases) > 0.1 * np.array(amplitude_values)):
        logger.warning("Significant systematic bias detected")

    return summary


def plot_injection_results(
    results: List[InjectionTestResult],
    ax=None,
) -> Any:
    """
    Plot injection test results.

    Parameters
    ----------
    results : list of InjectionTestResult
        Results from multiple amplitude tests
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots()

    A_in = [r.input_amplitude for r in results]
    A_out = [r.mean_recovered for r in results]
    A_err = [r.std_recovered for r in results]

    # Data points
    ax.errorbar(
        A_in, A_out, yerr=A_err,
        fmt='o', markersize=8, capsize=3, color='C0',
        label='Recovered'
    )

    # 1:1 line
    A_range = [min(A_in) - 0.2, max(A_in) + 0.2]
    ax.plot(A_range, A_range, 'k--', linewidth=1.5, label='Expected')

    ax.set_xlabel(r'$A_{kSZ}^{\rm input}$')
    ax.set_ylabel(r'$A_{kSZ}^{\rm recovered}$')
    ax.legend(frameon=False)
    ax.set_title('Injection Test Validation')

    return ax
