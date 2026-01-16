"""
Mass function and companion mass calculations.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
MSUN = 1.989e30  # kg
DAY = 86400  # seconds
KM = 1000  # meters


def compute_mass_function(P: float, K: float, e: float = 0.0) -> float:
    """
    Compute spectroscopic mass function.

    f(M) = P * K^3 * (1 - e^2)^(3/2) / (2*pi*G)

    Parameters
    ----------
    P : float
        Period (days)
    K : float
        Semi-amplitude (km/s)
    e : float
        Eccentricity

    Returns
    -------
    f_M : float
        Mass function (solar masses)
    """
    # Convert to SI
    P_s = P * DAY
    K_ms = K * KM

    # Mass function in kg
    f_M_kg = P_s * K_ms**3 * (1 - e**2)**1.5 / (2 * np.pi * G)

    # Convert to solar masses
    return f_M_kg / MSUN


def compute_m2_min(f_M: float, M1: float, max_m2: float = 100.0) -> float:
    """
    Compute minimum companion mass from mass function.

    For edge-on orbit (sin i = 1):
    f(M) = M2^3 / (M1 + M2)^2

    Parameters
    ----------
    f_M : float
        Mass function (solar masses)
    M1 : float
        Primary mass (solar masses)
    max_m2 : float
        Maximum allowed M2 (for capping pathological fits)

    Returns
    -------
    M2_min : float
        Minimum companion mass (solar masses)
    """
    if f_M <= 0 or M1 <= 0:
        return 0.0

    # Cap f_M to avoid pathological results
    # For reasonable systems, f_M < 100 Msun (even massive BH binaries)
    if f_M > 1000:
        logger.debug(f"f_M={f_M:.2f} very large, capping M2_min at {max_m2}")
        return max_m2

    # Solve: M2^3 - f_M * (M1 + M2)^2 = 0
    def func(m2):
        return m2**3 - f_M * (M1 + m2)**2

    try:
        # For small f_M, M2_min ~ f_M^(1/3)
        # For large f_M, M2_min approaches sqrt(f_M * M1^2)^(1/3) ~ f_M^(1/3)
        m2_low = 0.001
        m2_high = min(max_m2 * 10, max(100, f_M + M1))

        # Check if there's a root in the interval
        f_low = func(m2_low)
        f_high = func(m2_high)

        if f_low * f_high > 0:
            # No sign change, use approximation
            m2_approx = f_M**(1/3) * (1 + M1 / (3 * max(f_M**(1/3), 0.1)))**(2/3)
            return min(m2_approx, max_m2)

        result = brentq(func, m2_low, m2_high)
        return min(result, max_m2)
    except Exception as e:
        logger.debug(f"Failed to solve for M2_min (f_M={f_M:.2f}, M1={M1:.2f}): {e}")
        # Approximation for large f_M
        m2_approx = f_M**(1/3) * (1 + M1 / (3 * max(f_M**(1/3), 0.1)))**(2/3)
        return min(m2_approx, max_m2)


def compute_m2_at_inclination(f_M: float, M1: float, sin_i: float) -> float:
    """
    Compute companion mass at given inclination.

    f(M) = M2^3 * sin^3(i) / (M1 + M2)^2
    => M2^3 = f(M) * (M1 + M2)^2 / sin^3(i)

    Parameters
    ----------
    f_M : float
        Mass function (solar masses)
    M1 : float
        Primary mass (solar masses)
    sin_i : float
        sin(inclination)

    Returns
    -------
    M2 : float
        Companion mass (solar masses)
    """
    if sin_i <= 0 or sin_i > 1:
        return np.inf

    # Effective mass function for this inclination
    f_M_eff = f_M / sin_i**3

    return compute_m2_min(f_M_eff, M1)


def companion_mass_probabilities(f_M: float, M1: float, M1_err: float = 0.0,
                                 n_samples: int = 10000) -> Dict[str, float]:
    """
    Compute companion mass probabilities assuming random inclination.

    Assumes isotropic inclination distribution: P(cos i) = uniform.

    Parameters
    ----------
    f_M : float
        Mass function (solar masses)
    M1 : float
        Primary mass (solar masses)
    M1_err : float
        Primary mass uncertainty (solar masses)
    n_samples : int
        Number of Monte Carlo samples

    Returns
    -------
    dict
        Dictionary with:
        - M2_min: minimum mass (sin i = 1)
        - M2_median: median mass
        - M2_16, M2_84: 68% credible interval
        - prob_wd: Pr(M2 < 1.4 Msun)
        - prob_ns: Pr(1.4 < M2 < 3.0 Msun)
        - prob_bh: Pr(M2 > 3.0 Msun)
        - prob_gt_1p4: Pr(M2 > 1.4 Msun)
        - prob_gt_3p0: Pr(M2 > 3.0 Msun)
    """
    # Sample inclinations (isotropic: uniform in cos i)
    cos_i = np.random.uniform(0, 1, n_samples)
    sin_i = np.sqrt(1 - cos_i**2)

    # Sample M1 if uncertainty given
    if M1_err > 0:
        M1_samples = np.random.normal(M1, M1_err, n_samples)
        M1_samples = np.maximum(M1_samples, 0.1)
    else:
        M1_samples = np.full(n_samples, M1)

    # Compute M2 for each sample
    M2_samples = np.zeros(n_samples)
    for i in range(n_samples):
        M2_samples[i] = compute_m2_at_inclination(f_M, M1_samples[i], sin_i[i])

    # Remove infinities/nans
    valid = np.isfinite(M2_samples) & (M2_samples > 0) & (M2_samples < 1000)
    M2_valid = M2_samples[valid]

    if len(M2_valid) < 100:
        return {
            'M2_min': compute_m2_min(f_M, M1),
            'M2_median': np.nan,
            'M2_16': np.nan,
            'M2_84': np.nan,
            'prob_wd': 0.0,
            'prob_ns': 0.0,
            'prob_bh': 0.0,
            'prob_gt_1p4': 0.0,
            'prob_gt_3p0': 0.0
        }

    M2_min = compute_m2_min(f_M, M1)
    M2_median = np.median(M2_valid)
    M2_16, M2_84 = np.percentile(M2_valid, [16, 84])

    prob_wd = np.mean(M2_valid < 1.4)
    prob_ns = np.mean((M2_valid >= 1.4) & (M2_valid < 3.0))
    prob_bh = np.mean(M2_valid >= 3.0)
    prob_gt_1p4 = np.mean(M2_valid >= 1.4)
    prob_gt_3p0 = np.mean(M2_valid >= 3.0)

    return {
        'M2_min': M2_min,
        'M2_median': M2_median,
        'M2_16': M2_16,
        'M2_84': M2_84,
        'prob_wd': prob_wd,
        'prob_ns': prob_ns,
        'prob_bh': prob_bh,
        'prob_gt_1p4': prob_gt_1p4,
        'prob_gt_3p0': prob_gt_3p0
    }


def estimate_primary_mass_from_color(bp_rp: float, abs_g: float = None) -> Tuple[float, float]:
    """
    Estimate primary mass from Gaia BP-RP color.

    Simple empirical relation for main sequence stars.

    Parameters
    ----------
    bp_rp : float
        Gaia BP - RP color
    abs_g : float, optional
        Absolute G magnitude (for additional constraint)

    Returns
    -------
    M1 : float
        Estimated mass (solar masses)
    M1_err : float
        Uncertainty (solar masses)
    """
    # Simple polynomial fit to main sequence
    # Valid for BP-RP ~ 0.5 to 3.5 (F to M stars)

    if bp_rp < 0.3:
        # Hot stars
        M1 = 2.0
        M1_err = 0.5
    elif bp_rp < 0.8:
        # F/G stars
        M1 = 1.4 - 0.5 * bp_rp
        M1_err = 0.2
    elif bp_rp < 1.5:
        # K stars
        M1 = 1.1 - 0.35 * bp_rp
        M1_err = 0.15
    elif bp_rp < 2.5:
        # Early M stars
        M1 = 0.9 - 0.25 * bp_rp
        M1_err = 0.1
    else:
        # Late M stars
        M1 = 0.5 - 0.1 * (bp_rp - 2.5)
        M1_err = 0.1

    M1 = max(0.08, min(M1, 3.0))

    return M1, M1_err
