"""
Likelihood computation for kSZ parameter inference.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| L           | Likelihood                                      | dimensionless|
| χ²          | Chi-squared statistic                           | dimensionless|
| A_kSZ       | kSZ amplitude parameter                         | dimensionless|
| Ψ           | Precision matrix (inverse covariance)           | varies      |
| d           | Data vector (observed p(r))                     | muK         |
| m           | Model prediction                                | muK         |

Gaussian Likelihood
-------------------
The likelihood is:

    L(θ|d) ∝ exp(-χ²/2)

where:

    χ² = (d - m(θ))ᵀ Ψ (d - m(θ))

For the simple amplitude model m = A_kSZ × p_theory, the ML estimate is:

    Â_kSZ = (pᵀ Ψ d) / (pᵀ Ψ p)
    σ_A = 1 / √(pᵀ Ψ p)
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LikelihoodResult:
    """Container for likelihood evaluation results."""
    A_ksz: float
    A_ksz_err: float
    chi2: float
    chi2_null: float
    n_dof: int
    pte: float
    detection_sigma: float


class KSZLikelihood:
    """
    Gaussian likelihood for kSZ amplitude inference.

    Parameters
    ----------
    data : np.ndarray
        Observed pairwise momentum p(r)
    covariance : np.ndarray
        Covariance matrix
    theory_template : np.ndarray
        Theory prediction p_theory(r) for A_kSZ = 1
    hartlap_n_sims : int, optional
        Number of simulations for Hartlap correction

    Examples
    --------
    >>> likelihood = KSZLikelihood(p_obs, covariance, p_theory)
    >>> A_ml, sigma_A = likelihood.fit_amplitude()
    >>> log_L = likelihood.log_likelihood(A_ksz=1.0)
    """

    def __init__(
        self,
        data: np.ndarray,
        covariance: np.ndarray,
        theory_template: np.ndarray,
        hartlap_n_sims: Optional[int] = None,
    ):
        self.data = np.asarray(data)
        self.covariance = np.asarray(covariance)
        self.theory = np.asarray(theory_template)
        self.n_bins = len(data)

        if covariance.shape != (self.n_bins, self.n_bins):
            raise ValueError(
                f"Covariance shape {covariance.shape} doesn't match "
                f"data length {self.n_bins}"
            )

        if len(theory_template) != self.n_bins:
            raise ValueError(
                f"Theory template length {len(theory_template)} doesn't match "
                f"data length {self.n_bins}"
            )

        # Compute precision matrix with optional Hartlap correction
        self._compute_precision(hartlap_n_sims)

    def _compute_precision(self, hartlap_n_sims: Optional[int]) -> None:
        """Compute precision matrix with optional Hartlap correction."""
        try:
            self.precision = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, using pseudoinverse")
            self.precision = np.linalg.pinv(self.covariance)

        # Apply Hartlap correction
        if hartlap_n_sims is not None and hartlap_n_sims > self.n_bins + 2:
            alpha = (hartlap_n_sims - self.n_bins - 2) / (hartlap_n_sims - 1)
            self.precision *= alpha
            self._hartlap_factor = alpha
            logger.info(f"Applied Hartlap correction: α = {alpha:.4f}")
        else:
            self._hartlap_factor = 1.0

    def log_likelihood(self, A_ksz: float) -> float:
        """
        Compute log-likelihood for amplitude A_kSZ.

        log L = -χ²/2 (up to normalization constant)

        Parameters
        ----------
        A_ksz : float
            kSZ amplitude parameter

        Returns
        -------
        float
            Log-likelihood value
        """
        model = A_ksz * self.theory
        residual = self.data - model
        chi2 = residual @ self.precision @ residual
        return -0.5 * chi2

    def chi2(self, A_ksz: float) -> float:
        """Compute chi-squared for amplitude A_kSZ."""
        model = A_ksz * self.theory
        residual = self.data - model
        return residual @ self.precision @ residual

    def fit_amplitude(self) -> Tuple[float, float]:
        """
        Analytic ML estimate for amplitude.

        For the linear model m = A × p_theory:

            Â = (pᵀ Ψ d) / (pᵀ Ψ p)
            σ_A = 1 / √(pᵀ Ψ p)

        Returns
        -------
        A_ml : float
            Maximum likelihood amplitude
        sigma_A : float
            1-sigma uncertainty on amplitude
        """
        # Numerator: p^T Psi d
        numerator = self.theory @ self.precision @ self.data

        # Denominator: p^T Psi p
        denominator = self.theory @ self.precision @ self.theory

        if denominator <= 0:
            logger.warning("Denominator <= 0 in amplitude fit")
            return 0.0, np.inf

        A_ml = numerator / denominator
        sigma_A = 1.0 / np.sqrt(denominator)

        return A_ml, sigma_A

    def compute_detection_significance(self) -> LikelihoodResult:
        """
        Compute detection significance and related statistics.

        Returns
        -------
        LikelihoodResult
            Detection significance and fit results
        """
        # Fit amplitude
        A_ml, sigma_A = self.fit_amplitude()

        # Chi-squared at best fit
        chi2_best = self.chi2(A_ml)

        # Chi-squared for null (A=0)
        chi2_null = self.chi2(0.0)

        # Degrees of freedom
        n_dof = self.n_bins - 1  # One fitted parameter

        # PTE for goodness of fit
        from scipy.stats import chi2 as chi2_dist
        pte = 1 - chi2_dist.cdf(chi2_best, n_dof)

        # Detection significance: sqrt(delta chi2) with sign
        delta_chi2 = chi2_null - chi2_best
        detection_sigma = np.sign(A_ml) * np.sqrt(max(delta_chi2, 0))

        return LikelihoodResult(
            A_ksz=A_ml,
            A_ksz_err=sigma_A,
            chi2=chi2_best,
            chi2_null=chi2_null,
            n_dof=n_dof,
            pte=pte,
            detection_sigma=detection_sigma,
        )

    def sample_posterior(
        self,
        n_samples: int = 10000,
        prior_bounds: Tuple[float, float] = (-5, 5),
    ) -> np.ndarray:
        """
        Sample posterior distribution using simple rejection sampling.

        For single-parameter case, this is efficient. For multi-parameter
        inference, use the MCMC module.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        prior_bounds : tuple
            (min, max) for flat prior on A_kSZ

        Returns
        -------
        np.ndarray
            Posterior samples for A_kSZ
        """
        A_ml, sigma_A = self.fit_amplitude()

        # Use Gaussian proposal centered on ML estimate
        proposal_sigma = 3 * sigma_A

        samples = []
        n_accepted = 0
        n_total = 0

        rng = np.random.default_rng(42)

        while n_accepted < n_samples:
            # Propose from Gaussian
            A_prop = A_ml + proposal_sigma * rng.standard_normal()
            n_total += 1

            # Check prior bounds
            if not (prior_bounds[0] <= A_prop <= prior_bounds[1]):
                continue

            # Accept/reject based on likelihood ratio
            log_L = self.log_likelihood(A_prop)
            log_L_ml = self.log_likelihood(A_ml)

            if log_L - log_L_ml > -10:  # Avoid underflow
                accept_prob = np.exp(log_L - log_L_ml)
                if rng.random() < accept_prob:
                    samples.append(A_prop)
                    n_accepted += 1

        logger.info(f"Acceptance rate: {n_samples / n_total:.2%}")
        return np.array(samples)


def compute_chi2(
    data: np.ndarray,
    model: np.ndarray,
    precision: np.ndarray,
) -> float:
    """
    Compute chi-squared statistic.

    χ² = (d - m)ᵀ Ψ (d - m)
    """
    residual = data - model
    return float(residual @ precision @ residual)


def fit_amplitude_analytic(
    data: np.ndarray,
    theory: np.ndarray,
    precision: np.ndarray,
) -> Tuple[float, float]:
    """
    Analytic ML amplitude fit.

    Â = (pᵀ Ψ d) / (pᵀ Ψ p)
    σ_A = 1 / √(pᵀ Ψ p)

    Parameters
    ----------
    data : np.ndarray
        Observed data vector
    theory : np.ndarray
        Theory template
    precision : np.ndarray
        Precision matrix

    Returns
    -------
    A_ml : float
        ML amplitude
    sigma_A : float
        Uncertainty
    """
    numerator = theory @ precision @ data
    denominator = theory @ precision @ theory

    if denominator <= 0:
        return 0.0, np.inf

    A_ml = numerator / denominator
    sigma_A = 1.0 / np.sqrt(denominator)

    return A_ml, sigma_A
