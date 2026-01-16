"""
RV variability metrics and significance calculations.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RVMetrics:
    """Container for RV variability metrics."""
    # Basic stats
    n_epochs: int
    delta_rv: float  # max - min RV (km/s)
    median_rv_err: float

    # Significance metrics
    S: float  # Global significance
    S_min_loo: float  # Minimum LOO significance
    S_robust: float  # Conservative significance (= S_min_loo)

    # Chi-squared test
    chi2_constant: float
    chi2_dof: int
    chi2_pvalue: float
    chi2_reduced: float

    # Leverage metrics
    d_max: float  # Maximum leverage distance
    high_leverage_epoch: int  # Index of high leverage epoch
    leverage_fraction: float  # Fraction of signal from high leverage

    # Consistency metrics
    same_night_consistent: bool
    cross_survey_offset: Optional[float]  # DESI - LAMOST offset if both present

    # Quality flags
    has_high_leverage: bool
    is_robust: bool  # S_robust > threshold
    passed_chi2: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'n_epochs': self.n_epochs,
            'delta_rv': self.delta_rv,
            'median_rv_err': self.median_rv_err,
            'S': self.S,
            'S_min_loo': self.S_min_loo,
            'S_robust': self.S_robust,
            'chi2_constant': self.chi2_constant,
            'chi2_dof': self.chi2_dof,
            'chi2_pvalue': self.chi2_pvalue,
            'chi2_reduced': self.chi2_reduced,
            'd_max': self.d_max,
            'high_leverage_epoch': self.high_leverage_epoch,
            'leverage_fraction': self.leverage_fraction,
            'same_night_consistent': self.same_night_consistent,
            'cross_survey_offset': self.cross_survey_offset,
            'has_high_leverage': self.has_high_leverage,
            'is_robust': self.is_robust,
            'passed_chi2': self.passed_chi2
        }


def compute_rv_significance(rv: np.ndarray, rv_err: np.ndarray) -> float:
    """
    Compute RV variability significance S.

    S = sqrt(sum((rv_i - rv_mean)^2 / err_i^2) - N)

    This is approximately the SNR of variability detection.

    Parameters
    ----------
    rv : array
        RV values (km/s)
    rv_err : array
        RV errors (km/s)

    Returns
    -------
    float
        Significance S (sigma-equivalent)
    """
    if len(rv) < 2:
        return 0.0

    # Weighted mean
    weights = 1.0 / rv_err**2
    rv_mean = np.sum(rv * weights) / np.sum(weights)

    # Chi-squared relative to mean
    chi2 = np.sum(((rv - rv_mean) / rv_err)**2)

    # Significance (excess over expectation)
    N = len(rv)
    S = np.sqrt(max(0, chi2 - N))

    return float(S)


def compute_chi2_constant(rv: np.ndarray, rv_err: np.ndarray) -> Tuple[float, int, float]:
    """
    Compute chi-squared test for constant RV hypothesis.

    Parameters
    ----------
    rv : array
        RV values (km/s)
    rv_err : array
        RV errors (km/s)

    Returns
    -------
    chi2 : float
        Chi-squared statistic
    dof : int
        Degrees of freedom (N - 1)
    pvalue : float
        p-value for chi2 test
    """
    if len(rv) < 2:
        return 0.0, 0, 1.0

    # Weighted mean
    weights = 1.0 / rv_err**2
    rv_mean = np.sum(rv * weights) / np.sum(weights)

    # Chi-squared
    chi2 = np.sum(((rv - rv_mean) / rv_err)**2)
    dof = len(rv) - 1

    # p-value
    pvalue = 1.0 - stats.chi2.cdf(chi2, dof)

    return float(chi2), dof, float(pvalue)


def compute_leverage(rv: np.ndarray, rv_err: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """
    Compute leverage (influence) of each epoch on the mean.

    d_i = |rv_i - rv_mean_(-i)| / err_i

    where rv_mean_(-i) is the weighted mean excluding epoch i.

    Parameters
    ----------
    rv : array
        RV values (km/s)
    rv_err : array
        RV errors (km/s)

    Returns
    -------
    d : array
        Leverage distance for each epoch
    i_max : int
        Index of maximum leverage epoch
    d_max : float
        Maximum leverage distance
    """
    N = len(rv)
    if N < 2:
        return np.array([0.0]), 0, 0.0

    weights = 1.0 / rv_err**2
    d = np.zeros(N)

    for i in range(N):
        # Mean excluding epoch i
        mask = np.ones(N, dtype=bool)
        mask[i] = False

        w_other = weights[mask]
        rv_other = rv[mask]

        rv_mean_loo = np.sum(rv_other * w_other) / np.sum(w_other)

        # Leverage distance
        d[i] = np.abs(rv[i] - rv_mean_loo) / rv_err[i]

    i_max = int(np.argmax(d))
    d_max = float(d[i_max])

    return d, i_max, d_max


def compute_loo_significance(rv: np.ndarray, rv_err: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute leave-one-out significance for each epoch.

    Parameters
    ----------
    rv : array
        RV values (km/s)
    rv_err : array
        RV errors (km/s)

    Returns
    -------
    S_loo : array
        Significance with each epoch removed
    S_min : float
        Minimum LOO significance (= S_robust)
    """
    N = len(rv)
    if N < 3:
        return np.array([0.0]), 0.0

    S_loo = np.zeros(N)

    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False

        S_loo[i] = compute_rv_significance(rv[mask], rv_err[mask])

    return S_loo, float(np.min(S_loo))


def check_same_night_consistency(mjd: np.ndarray, rv: np.ndarray, rv_err: np.ndarray,
                                 max_dt_hours: float = 6.0,
                                 max_sigma: float = 3.0) -> Tuple[bool, List[Dict]]:
    """
    Check consistency of same-night observations.

    Parameters
    ----------
    mjd : array
        MJD timestamps
    rv : array
        RV values (km/s)
    rv_err : array
        RV errors (km/s)
    max_dt_hours : float
        Maximum time separation to consider "same night"
    max_sigma : float
        Maximum sigma deviation allowed

    Returns
    -------
    consistent : bool
        True if all same-night pairs are consistent
    pairs : list
        List of same-night pair comparisons
    """
    N = len(mjd)
    pairs = []
    consistent = True

    for i in range(N):
        for j in range(i + 1, N):
            dt_days = np.abs(mjd[j] - mjd[i])
            dt_hours = dt_days * 24

            if dt_hours < max_dt_hours:
                delta_rv = np.abs(rv[j] - rv[i])
                combined_err = np.sqrt(rv_err[i]**2 + rv_err[j]**2)
                sigma = delta_rv / combined_err

                pair_consistent = sigma < max_sigma
                if not pair_consistent:
                    consistent = False

                pairs.append({
                    'i': i,
                    'j': j,
                    'dt_hours': dt_hours,
                    'delta_rv': delta_rv,
                    'sigma': sigma,
                    'consistent': pair_consistent
                })

    return consistent, pairs


def compute_cross_survey_offset(rv: np.ndarray, rv_err: np.ndarray,
                                instruments: List[str]) -> Optional[float]:
    """
    Compute systematic offset between surveys (e.g., DESI vs LAMOST).

    Parameters
    ----------
    rv : array
        RV values (km/s)
    rv_err : array
        RV errors (km/s)
    instruments : list
        Instrument name for each epoch

    Returns
    -------
    offset : float or None
        Weighted mean offset (DESI - LAMOST) if both present, else None
    """
    instruments = np.array(instruments)

    desi_mask = instruments == 'DESI'
    lamost_mask = instruments == 'LAMOST'

    if not np.any(desi_mask) or not np.any(lamost_mask):
        return None

    # Weighted means
    w_desi = 1.0 / rv_err[desi_mask]**2
    w_lamost = 1.0 / rv_err[lamost_mask]**2

    rv_desi_mean = np.sum(rv[desi_mask] * w_desi) / np.sum(w_desi)
    rv_lamost_mean = np.sum(rv[lamost_mask] * w_lamost) / np.sum(w_lamost)

    return float(rv_desi_mean - rv_lamost_mean)
