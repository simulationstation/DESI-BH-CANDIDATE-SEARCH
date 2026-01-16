"""
MCMC Keplerian orbit fitting.
"""

import numpy as np
from scipy import optimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import emcee
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    logger.warning("emcee not installed - MCMC fitting unavailable")


@dataclass
class OrbitPosterior:
    """Posterior summary from MCMC orbit fit."""
    # Period
    P_median: float
    P_16: float
    P_84: float

    # Semi-amplitude
    K_median: float
    K_16: float
    K_84: float

    # Eccentricity
    e_median: float
    e_16: float
    e_84: float

    # Argument of periastron
    omega_median: float
    omega_16: float
    omega_84: float

    # Time of periastron
    T0_median: float
    T0_16: float
    T0_84: float

    # Systemic velocity
    gamma_median: float
    gamma_16: float
    gamma_84: float

    # Mass function
    f_M_median: float
    f_M_16: float
    f_M_84: float

    # Derived minimum companion mass (for given M1)
    M2_min_median: float
    M2_min_16: float
    M2_min_84: float

    # Probabilities
    prob_M2_gt_1p4: float  # Pr(M2 > 1.4 Msun)
    prob_M2_gt_3p0: float  # Pr(M2 > 3.0 Msun)

    # Quality metrics
    n_samples: int
    acceptance_fraction: float
    converged: bool

    # Full chains (optional)
    chains: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'P': {'median': self.P_median, 'p16': self.P_16, 'p84': self.P_84},
            'K': {'median': self.K_median, 'p16': self.K_16, 'p84': self.K_84},
            'e': {'median': self.e_median, 'p16': self.e_16, 'p84': self.e_84},
            'omega': {'median': self.omega_median, 'p16': self.omega_16, 'p84': self.omega_84},
            'T0': {'median': self.T0_median, 'p16': self.T0_16, 'p84': self.T0_84},
            'gamma': {'median': self.gamma_median, 'p16': self.gamma_16, 'p84': self.gamma_84},
            'f_M': {'median': self.f_M_median, 'p16': self.f_M_16, 'p84': self.f_M_84},
            'M2_min': {'median': self.M2_min_median, 'p16': self.M2_min_16, 'p84': self.M2_min_84},
            'prob_M2_gt_1p4': self.prob_M2_gt_1p4,
            'prob_M2_gt_3p0': self.prob_M2_gt_3p0,
            'n_samples': self.n_samples,
            'acceptance_fraction': self.acceptance_fraction,
            'converged': self.converged
        }


def rv_keplerian(t: np.ndarray, P: float, K: float, e: float,
                 omega: float, T0: float, gamma: float) -> np.ndarray:
    """
    Compute Keplerian RV model.

    Parameters
    ----------
    t : array
        Times (MJD)
    P : float
        Period (days)
    K : float
        Semi-amplitude (km/s)
    e : float
        Eccentricity
    omega : float
        Argument of periastron (radians)
    T0 : float
        Time of periastron (MJD)
    gamma : float
        Systemic velocity (km/s)

    Returns
    -------
    rv : array
        Model RV (km/s)
    """
    # Mean anomaly
    M = 2 * np.pi * (t - T0) / P

    # Solve Kepler's equation for eccentric anomaly
    E = _solve_kepler(M, e)

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    # Radial velocity
    rv = gamma + K * (np.cos(nu + omega) + e * np.cos(omega))

    return rv


def _solve_kepler(M: np.ndarray, e: float, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """Solve Kepler's equation M = E - e*sin(E) for E."""
    M = np.asarray(M)
    E = M.copy()

    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E += dE
        if np.all(np.abs(dE) < tol):
            break

    return E


class MCMCOrbitFitter:
    """
    MCMC Keplerian orbit fitter using emcee.
    """

    def __init__(self, n_walkers: int = 32, n_steps: int = 3000,
                 n_burnin: int = 500, e_max: float = 0.95):
        """
        Parameters
        ----------
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of MCMC steps
        n_burnin : int
            Number of burn-in steps to discard
        e_max : float
            Maximum eccentricity allowed
        """
        if not HAS_EMCEE:
            raise ImportError("emcee required for MCMC fitting")

        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_burnin = n_burnin
        self.e_max = e_max

    def fit(self, mjd: np.ndarray, rv: np.ndarray, rv_err: np.ndarray,
            M1: float = 0.5, M1_err: float = 0.1,
            period_hint: float = None) -> OrbitPosterior:
        """
        Fit Keplerian orbit with MCMC.

        Parameters
        ----------
        mjd : array
            MJD timestamps
        rv : array
            RV values (km/s)
        rv_err : array
            RV errors (km/s)
        M1 : float
            Primary mass (Msun)
        M1_err : float
            Primary mass uncertainty (Msun)
        period_hint : float, optional
            Initial period guess (days)

        Returns
        -------
        OrbitPosterior
            Posterior summary
        """
        n_data = len(mjd)
        baseline = mjd.max() - mjd.min()

        # Initial guesses
        if period_hint is None:
            period_hint = baseline / 4

        K_hint = (np.max(rv) - np.min(rv)) / 2
        gamma_hint = np.mean(rv)

        # Log-likelihood
        def log_likelihood(theta):
            P, K, e, omega, T0, gamma = theta

            if P <= 0 or K <= 0 or e < 0 or e >= self.e_max:
                return -np.inf

            try:
                model = rv_keplerian(mjd, P, K, e, omega, T0, gamma)
                chi2 = np.sum(((rv - model) / rv_err)**2)
                return -0.5 * chi2
            except:
                return -np.inf

        # Log-prior
        def log_prior(theta):
            P, K, e, omega, T0, gamma = theta

            # Period: log-uniform from 0.5 to 2*baseline
            if P < 0.5 or P > 2 * baseline:
                return -np.inf

            # K: positive
            if K < 0 or K > 500:
                return -np.inf

            # Eccentricity: uniform [0, e_max)
            if e < 0 or e >= self.e_max:
                return -np.inf

            # omega: uniform [0, 2pi)
            if omega < 0 or omega >= 2 * np.pi:
                return -np.inf

            # T0: within baseline
            if T0 < mjd.min() - P or T0 > mjd.max() + P:
                return -np.inf

            # gamma: broad uniform
            if gamma < -500 or gamma > 500:
                return -np.inf

            # Log-uniform prior on P
            return -np.log(P)

        # Log-posterior
        def log_posterior(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta)

        # Initialize walkers
        ndim = 6
        p0 = np.zeros((self.n_walkers, ndim))

        for i in range(self.n_walkers):
            p0[i, 0] = period_hint * np.exp(np.random.normal(0, 0.3))  # P
            p0[i, 1] = K_hint * np.exp(np.random.normal(0, 0.3))  # K
            p0[i, 2] = np.random.uniform(0, 0.5)  # e
            p0[i, 3] = np.random.uniform(0, 2*np.pi)  # omega
            p0[i, 4] = mjd.min() + np.random.uniform(0, period_hint)  # T0
            p0[i, 5] = gamma_hint + np.random.normal(0, 10)  # gamma

        # Run MCMC
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler = emcee.EnsembleSampler(self.n_walkers, ndim, log_posterior)
            sampler.run_mcmc(p0, self.n_steps, progress=False)

        # Get chains
        chains = sampler.get_chain(discard=self.n_burnin, flat=True)
        acceptance = np.mean(sampler.acceptance_fraction)

        # Check convergence
        converged = acceptance > 0.1 and len(chains) > 100

        if len(chains) < 100:
            logger.warning("MCMC did not converge - insufficient samples")
            return self._empty_posterior()

        # Extract parameters
        P_samples = chains[:, 0]
        K_samples = chains[:, 1]
        e_samples = chains[:, 2]
        omega_samples = chains[:, 3]
        T0_samples = chains[:, 4]
        gamma_samples = chains[:, 5]

        # Compute mass function
        # f(M) = P * K^3 * (1 - e^2)^1.5 / (2*pi*G)
        # In solar units: f(M) [Msun] = 1.0361e-7 * P[days] * K[km/s]^3 * (1-e^2)^1.5
        f_M_samples = 1.0361e-7 * P_samples * K_samples**3 * (1 - e_samples**2)**1.5

        # Compute M2_min with M1 uncertainty
        M1_samples = np.random.normal(M1, M1_err, len(f_M_samples))
        M1_samples = np.maximum(M1_samples, 0.1)  # Floor

        M2_min_samples = self._solve_m2_min(f_M_samples, M1_samples)

        # Compute probabilities
        prob_M2_gt_1p4 = np.mean(M2_min_samples > 1.4)
        prob_M2_gt_3p0 = np.mean(M2_min_samples > 3.0)

        # Percentiles
        def pct(arr):
            return np.percentile(arr, [50, 16, 84])

        P_pct = pct(P_samples)
        K_pct = pct(K_samples)
        e_pct = pct(e_samples)
        omega_pct = pct(omega_samples)
        T0_pct = pct(T0_samples)
        gamma_pct = pct(gamma_samples)
        f_M_pct = pct(f_M_samples)
        M2_min_pct = pct(M2_min_samples)

        return OrbitPosterior(
            P_median=P_pct[0], P_16=P_pct[1], P_84=P_pct[2],
            K_median=K_pct[0], K_16=K_pct[1], K_84=K_pct[2],
            e_median=e_pct[0], e_16=e_pct[1], e_84=e_pct[2],
            omega_median=omega_pct[0], omega_16=omega_pct[1], omega_84=omega_pct[2],
            T0_median=T0_pct[0], T0_16=T0_pct[1], T0_84=T0_pct[2],
            gamma_median=gamma_pct[0], gamma_16=gamma_pct[1], gamma_84=gamma_pct[2],
            f_M_median=f_M_pct[0], f_M_16=f_M_pct[1], f_M_84=f_M_pct[2],
            M2_min_median=M2_min_pct[0], M2_min_16=M2_min_pct[1], M2_min_84=M2_min_pct[2],
            prob_M2_gt_1p4=prob_M2_gt_1p4,
            prob_M2_gt_3p0=prob_M2_gt_3p0,
            n_samples=len(chains),
            acceptance_fraction=acceptance,
            converged=converged,
            chains=chains
        )

    def _solve_m2_min(self, f_M: np.ndarray, M1: np.ndarray) -> np.ndarray:
        """Solve for minimum companion mass from mass function."""
        # f(M) = M2^3 * sin^3(i) / (M1 + M2)^2
        # For sin(i) = 1 (edge-on): f(M) = M2^3 / (M1 + M2)^2
        # This is a cubic equation in M2

        M2_min = np.zeros_like(f_M)

        for i in range(len(f_M)):
            # Solve: M2^3 - f_M * (M1 + M2)^2 = 0
            # Numerically solve
            def func(m2):
                return m2**3 - f_M[i] * (M1[i] + m2)**2

            try:
                from scipy.optimize import brentq
                M2_min[i] = brentq(func, 0.01, 100)
            except:
                M2_min[i] = np.nan

        return M2_min

    def _empty_posterior(self) -> OrbitPosterior:
        """Return empty posterior for failed fit."""
        return OrbitPosterior(
            P_median=0, P_16=0, P_84=0,
            K_median=0, K_16=0, K_84=0,
            e_median=0, e_16=0, e_84=0,
            omega_median=0, omega_16=0, omega_84=0,
            T0_median=0, T0_16=0, T0_84=0,
            gamma_median=0, gamma_16=0, gamma_84=0,
            f_M_median=0, f_M_16=0, f_M_84=0,
            M2_min_median=0, M2_min_16=0, M2_min_84=0,
            prob_M2_gt_1p4=0, prob_M2_gt_3p0=0,
            n_samples=0, acceptance_fraction=0, converged=False
        )
