"""
Weight computation for DESI galaxy samples.

This module provides functions for computing and manipulating
galaxy weights for optimal kSZ estimation.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| w_tot       | Total weight = w_sys * w_comp * w_zfail * w_fkp | dimensionless|
| w_sys       | Imaging systematics weight                      | dimensionless|
| w_comp      | Completeness weight                             | dimensionless|
| w_zfail     | Redshift failure weight                         | dimensionless|
| w_fkp       | FKP optimal weight                              | dimensionless|
| n(z)        | Galaxy number density at redshift z             | (Mpc/h)^-3  |
| P_0         | FKP power normalization (~10000)                | (Mpc/h)^3   |
| N_eff       | Effective number = (sum w)^2 / sum w^2          | dimensionless|

References
----------
- Feldman, Kaiser, Peacock 1994, ApJ, 426, 23 (FKP weighting)
- DESI Collaboration 2024, AJ, 168, 58 (DESI weighting)
"""

from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_total_weight(
    weight_sys: np.ndarray,
    weight_comp: np.ndarray,
    weight_zfail: np.ndarray,
    weight_fkp: np.ndarray,
) -> np.ndarray:
    """
    Compute total galaxy weight from component weights.

    The total weight is the product of all component weights:
        w_tot = w_sys * w_comp * w_zfail * w_fkp

    Parameters
    ----------
    weight_sys : np.ndarray
        Imaging systematics weight (corrects for depth variations)
    weight_comp : np.ndarray
        Completeness weight (corrects for fiber assignment)
    weight_zfail : np.ndarray
        Redshift failure weight (corrects for failed redshifts)
    weight_fkp : np.ndarray
        FKP optimal weight (inverse variance weighting)

    Returns
    -------
    np.ndarray
        Total combined weight
    """
    return weight_sys * weight_comp * weight_zfail * weight_fkp


def compute_fkp_weights(
    z: np.ndarray,
    nz: Optional[np.ndarray] = None,
    P0: float = 10000.0,
    nbar_func: Optional[callable] = None,
) -> np.ndarray:
    """
    Compute FKP (Feldman-Kaiser-Peacock) optimal weights.

    The FKP weight minimizes variance for power spectrum estimation:
        w_fkp = 1 / (1 + n(z) * P_0)

    Parameters
    ----------
    z : np.ndarray
        Galaxy redshifts
    nz : np.ndarray, optional
        Galaxy number density at each redshift (pre-computed)
    P0 : float
        Power spectrum normalization (fiducial P(k) at k ~ 0.1 h/Mpc)
    nbar_func : callable, optional
        Function nbar(z) returning number density at redshift z

    Returns
    -------
    np.ndarray
        FKP weights

    Notes
    -----
    For kSZ analysis, the optimal weighting may differ from FKP
    since we're measuring a velocity-weighted signal. However,
    FKP weights are a reasonable starting point.
    """
    if nz is not None:
        # Use provided n(z) values
        pass
    elif nbar_func is not None:
        # Compute n(z) from function
        nz = nbar_func(z)
    else:
        # Estimate n(z) from data (simple histogram approach)
        nz = _estimate_nz_from_data(z)

    # FKP weight formula
    w_fkp = 1.0 / (1.0 + nz * P0)

    return w_fkp


def compute_effective_number(weights: np.ndarray) -> float:
    """
    Compute effective number of galaxies.

    The effective number accounts for non-uniform weighting:
        N_eff = (sum w_i)^2 / sum w_i^2

    For uniform weights, N_eff = N.
    For highly non-uniform weights, N_eff < N.

    Parameters
    ----------
    weights : np.ndarray
        Galaxy weights

    Returns
    -------
    float
        Effective number of galaxies
    """
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)

    if sum_w2 == 0:
        return 0.0

    return sum_w**2 / sum_w2


def normalize_weights(
    weights: np.ndarray,
    method: str = "mean",
) -> np.ndarray:
    """
    Normalize weights.

    Parameters
    ----------
    weights : np.ndarray
        Original weights
    method : str
        Normalization method:
        - 'mean': Normalize to mean of 1
        - 'sum': Normalize to sum of N
        - 'max': Normalize to max of 1

    Returns
    -------
    np.ndarray
        Normalized weights
    """
    if method == "mean":
        return weights / np.mean(weights)
    elif method == "sum":
        return weights * len(weights) / np.sum(weights)
    elif method == "max":
        return weights / np.max(weights)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_weight_statistics(weights: np.ndarray) -> dict:
    """
    Compute statistics of weight distribution.

    Parameters
    ----------
    weights : np.ndarray
        Galaxy weights

    Returns
    -------
    dict
        Dictionary of weight statistics
    """
    return {
        "n_total": len(weights),
        "n_eff": compute_effective_number(weights),
        "efficiency": compute_effective_number(weights) / len(weights),
        "sum": float(np.sum(weights)),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "median": float(np.median(weights)),
        "percentiles": {
            "5": float(np.percentile(weights, 5)),
            "25": float(np.percentile(weights, 25)),
            "75": float(np.percentile(weights, 75)),
            "95": float(np.percentile(weights, 95)),
        },
    }


def clip_extreme_weights(
    weights: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip extreme weights to reduce outlier influence.

    Parameters
    ----------
    weights : np.ndarray
        Original weights
    percentile_low : float
        Lower percentile for clipping
    percentile_high : float
        Upper percentile for clipping

    Returns
    -------
    clipped_weights : np.ndarray
        Clipped weights
    clip_mask : np.ndarray
        Boolean mask indicating clipped values
    """
    w_low = np.percentile(weights, percentile_low)
    w_high = np.percentile(weights, percentile_high)

    clipped = np.clip(weights, w_low, w_high)
    clip_mask = (weights < w_low) | (weights > w_high)

    n_clipped = np.sum(clip_mask)
    if n_clipped > 0:
        logger.info(f"Clipped {n_clipped} weights to [{w_low:.3f}, {w_high:.3f}]")

    return clipped, clip_mask


def _estimate_nz_from_data(
    z: np.ndarray,
    n_bins: int = 50,
    smooth: bool = True,
) -> np.ndarray:
    """
    Estimate n(z) from redshift distribution.

    This is a simple histogram-based estimate. For production use,
    the n(z) should be computed from the full catalog with proper
    volume normalization.
    """
    # Compute histogram
    z_min, z_max = np.min(z), np.max(z)
    z_edges = np.linspace(z_min, z_max, n_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    counts, _ = np.histogram(z, bins=z_edges)
    counts = counts.astype(float)

    # Normalize (simple approximation - not proper n(z) calculation)
    counts /= np.sum(counts)
    counts /= (z_edges[1] - z_edges[0])

    # Smooth if requested
    if smooth:
        from scipy.ndimage import gaussian_filter1d
        counts = gaussian_filter1d(counts, sigma=2)

    # Interpolate to each galaxy's redshift
    nz = np.interp(z, z_centers, counts)

    # Scale to reasonable units (this is approximate)
    nz *= 1e-4  # Rough scaling to (Mpc/h)^-3

    return nz


def combine_weights_for_pairs(
    weights_i: np.ndarray,
    weights_j: np.ndarray,
    method: str = "product",
) -> np.ndarray:
    """
    Combine weights for galaxy pairs.

    Parameters
    ----------
    weights_i : np.ndarray
        Weights for first galaxy in each pair
    weights_j : np.ndarray
        Weights for second galaxy in each pair
    method : str
        Combination method:
        - 'product': w_ij = w_i * w_j
        - 'geometric': w_ij = sqrt(w_i * w_j)
        - 'harmonic': w_ij = 2 * w_i * w_j / (w_i + w_j)

    Returns
    -------
    np.ndarray
        Combined pair weights
    """
    if method == "product":
        return weights_i * weights_j
    elif method == "geometric":
        return np.sqrt(weights_i * weights_j)
    elif method == "harmonic":
        return 2 * weights_i * weights_j / (weights_i + weights_j + 1e-10)
    else:
        raise ValueError(f"Unknown combination method: {method}")
