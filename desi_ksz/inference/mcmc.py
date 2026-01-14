"""
MCMC sampling for kSZ parameter inference.

Uses emcee for affine-invariant MCMC sampling.
"""

from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    emcee = None
    EMCEE_AVAILABLE = False


@dataclass
class MCMCResult:
    """Container for MCMC sampling results."""
    samples: np.ndarray  # Shape (n_samples, n_params)
    log_prob: np.ndarray  # Shape (n_samples,)
    param_names: List[str]
    n_walkers: int
    n_steps: int
    n_burnin: int
    acceptance_fraction: float
    autocorr_time: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]

    @property
    def n_params(self) -> int:
        return self.samples.shape[1]

    def get_chain(self, param: str, flat: bool = True) -> np.ndarray:
        """Get samples for a specific parameter."""
        if param not in self.param_names:
            raise ValueError(f"Unknown parameter: {param}")
        idx = self.param_names.index(param)
        return self.samples[:, idx]

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each parameter."""
        summary = {}
        for i, name in enumerate(self.param_names):
            samples = self.samples[:, i]
            summary[name] = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "median": float(np.median(samples)),
                "lower_68": float(np.percentile(samples, 16)),
                "upper_68": float(np.percentile(samples, 84)),
                "lower_95": float(np.percentile(samples, 2.5)),
                "upper_95": float(np.percentile(samples, 97.5)),
            }
        return summary

    def get_map_estimate(self) -> Dict[str, float]:
        """Get MAP (maximum a posteriori) estimate."""
        idx_max = np.argmax(self.log_prob)
        return {name: float(self.samples[idx_max, i])
                for i, name in enumerate(self.param_names)}


def run_mcmc(
    log_prob_fn: Callable,
    initial_params: np.ndarray,
    param_names: List[str],
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burnin: int = 1000,
    progress: bool = True,
    random_seed: int = 42,
) -> MCMCResult:
    """
    Run MCMC sampling using emcee.

    Parameters
    ----------
    log_prob_fn : callable
        Function returning log-probability for parameter vector
    initial_params : np.ndarray
        Initial parameter values (n_params,)
    param_names : list of str
        Names of parameters
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_burnin : int
        Number of burn-in steps to discard
    progress : bool
        Show progress bar
    random_seed : int
        Random seed

    Returns
    -------
    MCMCResult
        MCMC sampling results
    """
    if not EMCEE_AVAILABLE:
        raise ImportError("emcee is required for MCMC sampling")

    n_params = len(initial_params)
    logger.info(
        f"Running MCMC: {n_params} parameters, {n_walkers} walkers, "
        f"{n_steps} steps ({n_burnin} burn-in)"
    )

    # Initialize walkers around initial position
    rng = np.random.default_rng(random_seed)
    initial_spread = 0.01 * np.abs(initial_params) + 1e-4
    pos = initial_params + initial_spread * rng.standard_normal((n_walkers, n_params))

    # Set up sampler
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_prob_fn)

    # Run MCMC
    sampler.run_mcmc(pos, n_steps, progress=progress)

    # Get results
    samples_full = sampler.get_chain(flat=False)  # (n_steps, n_walkers, n_params)
    log_prob_full = sampler.get_log_prob(flat=False)  # (n_steps, n_walkers)

    # Discard burn-in and flatten
    samples = samples_full[n_burnin:].reshape(-1, n_params)
    log_prob = log_prob_full[n_burnin:].flatten()

    # Compute diagnostics
    acceptance_fraction = np.mean(sampler.acceptance_fraction)

    try:
        autocorr_time = sampler.get_autocorr_time(quiet=True)
    except Exception:
        autocorr_time = np.array([])
        logger.warning("Could not compute autocorrelation time")

    logger.info(f"MCMC complete. Acceptance fraction: {acceptance_fraction:.2%}")

    return MCMCResult(
        samples=samples,
        log_prob=log_prob,
        param_names=param_names,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burnin=n_burnin,
        acceptance_fraction=acceptance_fraction,
        autocorr_time=autocorr_time,
    )


def compute_map_estimate(
    samples: np.ndarray,
    log_prob: np.ndarray,
) -> np.ndarray:
    """
    Compute MAP (maximum a posteriori) estimate from samples.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples (n_samples, n_params)
    log_prob : np.ndarray
        Log-probability for each sample

    Returns
    -------
    np.ndarray
        MAP parameter values
    """
    idx_max = np.argmax(log_prob)
    return samples[idx_max]


def create_ksz_log_prob(
    data: np.ndarray,
    covariance: np.ndarray,
    theory: np.ndarray,
    prior_bounds: Dict[str, Tuple[float, float]],
) -> Callable:
    """
    Create log-probability function for kSZ amplitude inference.

    Parameters
    ----------
    data : np.ndarray
        Observed p(r)
    covariance : np.ndarray
        Covariance matrix
    theory : np.ndarray
        Theory template
    prior_bounds : dict
        Prior bounds for each parameter

    Returns
    -------
    callable
        Log-probability function
    """
    try:
        precision = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(covariance)

    def log_prob(theta):
        A_ksz = theta[0]

        # Prior
        if not (prior_bounds["A_ksz"][0] <= A_ksz <= prior_bounds["A_ksz"][1]):
            return -np.inf

        # Likelihood
        model = A_ksz * theory
        residual = data - model
        chi2 = residual @ precision @ residual

        return -0.5 * chi2

    return log_prob
