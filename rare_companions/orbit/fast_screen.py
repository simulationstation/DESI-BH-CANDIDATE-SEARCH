"""
Fast orbital period screening.
"""

import numpy as np
from scipy import optimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)


@dataclass
class FastOrbitResult:
    """Result from fast period search."""
    best_period: float  # days
    best_chi2: float
    best_K: float  # km/s
    best_gamma: float  # km/s
    delta_chi2: float  # improvement over constant
    periods_searched: np.ndarray
    chi2_grid: np.ndarray
    n_local_minima: int
    is_multimodal: bool


def rv_model_circular(t: np.ndarray, P: float, K: float, gamma: float, phi: float) -> np.ndarray:
    """Circular orbit RV model."""
    phase = 2 * np.pi * t / P + phi
    return gamma + K * np.sin(phase)


def fit_circular_at_period(mjd: np.ndarray, rv: np.ndarray, rv_err: np.ndarray,
                           P: float) -> Tuple[float, float, float, float]:
    """
    Fit circular orbit at fixed period.

    Returns (chi2, K, gamma, phi).
    """
    def chi2_func(params):
        K, gamma, phi = params
        model = rv_model_circular(mjd, P, K, gamma, phi)
        return np.sum(((rv - model) / rv_err)**2)

    # Initial guesses
    K0 = (np.max(rv) - np.min(rv)) / 2
    gamma0 = np.mean(rv)
    phi0 = 0.0

    try:
        result = optimize.minimize(
            chi2_func,
            [K0, gamma0, phi0],
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 0.1, 'fatol': 0.1}
        )
        if result.success:
            K, gamma, phi = result.x
            return result.fun, abs(K), gamma, phi % (2 * np.pi)
    except:
        pass

    return np.inf, 0.0, 0.0, 0.0


def _fit_period_worker(args):
    """Worker function for parallel period fitting."""
    mjd, rv, rv_err, P = args
    chi2, K, gamma, phi = fit_circular_at_period(mjd, rv, rv_err, P)
    return P, chi2, K, gamma, phi


class FastOrbitScreen:
    """
    Fast orbital period screening using circular orbit grid search.

    This provides a quick first-pass estimate of the orbital period
    and semi-amplitude for prioritizing candidates.
    """

    def __init__(self, period_min: float = 0.5, period_max: float = 1000.0,
                 n_periods: int = 500, n_workers: int = None):
        """
        Parameters
        ----------
        period_min : float
            Minimum period to search (days)
        period_max : float
            Maximum period to search (days)
        n_periods : int
            Number of period grid points
        n_workers : int
            Number of parallel workers (default: cpu_count - 1)
        """
        self.period_min = period_min
        self.period_max = period_max
        self.n_periods = n_periods
        self.n_workers = n_workers or max(1, cpu_count() - 1)

        # Log-uniform period grid
        self.period_grid = np.logspace(
            np.log10(period_min),
            np.log10(period_max),
            n_periods
        )

    def search(self, mjd: np.ndarray, rv: np.ndarray, rv_err: np.ndarray,
               parallel: bool = True) -> FastOrbitResult:
        """
        Search for best orbital period.

        Parameters
        ----------
        mjd : array
            MJD timestamps
        rv : array
            RV values (km/s)
        rv_err : array
            RV errors (km/s)
        parallel : bool
            Use parallel processing

        Returns
        -------
        FastOrbitResult
            Search results
        """
        if len(mjd) < 3:
            return self._empty_result()

        # Chi-squared for constant RV
        weights = 1.0 / rv_err**2
        rv_mean = np.sum(rv * weights) / np.sum(weights)
        chi2_constant = np.sum(((rv - rv_mean) / rv_err)**2)

        # Search over periods
        if parallel and self.n_workers > 1:
            args = [(mjd, rv, rv_err, P) for P in self.period_grid]
            with Pool(self.n_workers) as pool:
                results = pool.map(_fit_period_worker, args)
        else:
            results = [_fit_period_worker((mjd, rv, rv_err, P)) for P in self.period_grid]

        # Unpack results
        chi2_grid = np.array([r[1] for r in results])
        K_grid = np.array([r[2] for r in results])
        gamma_grid = np.array([r[3] for r in results])

        # Find best period
        i_best = np.argmin(chi2_grid)
        best_period = self.period_grid[i_best]
        best_chi2 = chi2_grid[i_best]
        best_K = K_grid[i_best]
        best_gamma = gamma_grid[i_best]

        # Improvement over constant
        delta_chi2 = chi2_constant - best_chi2

        # Count local minima (multimodality check)
        n_local_minima = self._count_local_minima(chi2_grid)
        is_multimodal = n_local_minima > 3

        return FastOrbitResult(
            best_period=best_period,
            best_chi2=best_chi2,
            best_K=best_K,
            best_gamma=best_gamma,
            delta_chi2=delta_chi2,
            periods_searched=self.period_grid,
            chi2_grid=chi2_grid,
            n_local_minima=n_local_minima,
            is_multimodal=is_multimodal
        )

    def _count_local_minima(self, chi2_grid: np.ndarray, threshold: float = 0.9) -> int:
        """Count significant local minima in chi2 grid."""
        chi2_min = np.min(chi2_grid)
        chi2_max = np.max(chi2_grid)
        chi2_range = chi2_max - chi2_min

        if chi2_range == 0:
            return 1

        # Normalize
        chi2_norm = (chi2_grid - chi2_min) / chi2_range

        # Find local minima
        n_minima = 0
        for i in range(1, len(chi2_norm) - 1):
            if chi2_norm[i] < chi2_norm[i-1] and chi2_norm[i] < chi2_norm[i+1]:
                if chi2_norm[i] < threshold:
                    n_minima += 1

        return max(1, n_minima)

    def _empty_result(self) -> FastOrbitResult:
        """Return empty result for invalid input."""
        return FastOrbitResult(
            best_period=0.0,
            best_chi2=np.inf,
            best_K=0.0,
            best_gamma=0.0,
            delta_chi2=0.0,
            periods_searched=self.period_grid,
            chi2_grid=np.full(len(self.period_grid), np.inf),
            n_local_minima=0,
            is_multimodal=False
        )


def fast_period_search(mjd: np.ndarray, rv: np.ndarray, rv_err: np.ndarray,
                       period_min: float = 0.5, period_max: float = 1000.0,
                       n_periods: int = 500) -> FastOrbitResult:
    """
    Convenience function for fast period search.

    Parameters
    ----------
    mjd : array
        MJD timestamps
    rv : array
        RV values (km/s)
    rv_err : array
        RV errors (km/s)
    period_min : float
        Minimum period (days)
    period_max : float
        Maximum period (days)
    n_periods : int
        Number of grid points

    Returns
    -------
    FastOrbitResult
        Search results
    """
    screener = FastOrbitScreen(period_min, period_max, n_periods)
    return screener.search(mjd, rv, rv_err)
