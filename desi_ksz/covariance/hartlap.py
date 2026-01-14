"""
Hartlap correction and covariance matrix utilities.

The Hartlap correction accounts for bias in the inverse of a covariance
matrix estimated from a finite number of samples.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| Ψ           | Precision matrix (inverse covariance)           | varies      |
| C           | Covariance matrix                               | varies      |
| N_s         | Number of samples/simulations                   | dimensionless|
| N_d         | Number of data points (bins)                    | dimensionless|
| α           | Hartlap correction factor                       | dimensionless|

Hartlap Correction
------------------
When estimating the precision matrix from N_s samples:

    Ψ_est = α × C_est^{-1}

where the correction factor is:

    α = (N_s - N_d - 2) / (N_s - 1)

This debiases the precision matrix for Gaussian data.

References
----------
- Hartlap, J., Simon, P., Schneider, P. 2007, A&A, 464, 399
"""

from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_hartlap_correction(
    covariance: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, float]:
    """
    Apply Hartlap correction to precision matrix.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix (shape n_bins, n_bins)
    n_samples : int
        Number of samples used to estimate covariance

    Returns
    -------
    precision : np.ndarray
        Corrected precision matrix
    alpha : float
        Correction factor applied

    Raises
    ------
    ValueError
        If n_samples is too small relative to n_bins
    """
    n_bins = covariance.shape[0]

    if n_samples <= n_bins + 2:
        raise ValueError(
            f"n_samples ({n_samples}) must be > n_bins + 2 ({n_bins + 2}) "
            f"for Hartlap correction"
        )

    # Hartlap factor
    alpha = (n_samples - n_bins - 2) / (n_samples - 1)

    if alpha < 0.5:
        logger.warning(
            f"Hartlap factor α = {alpha:.3f} is small. "
            f"Consider using more samples."
        )

    # Compute precision matrix
    precision_raw = np.linalg.inv(covariance)
    precision = alpha * precision_raw

    logger.info(f"Applied Hartlap correction: α = {alpha:.4f}")

    return precision, alpha


def compute_precision_matrix(
    covariance: np.ndarray,
    n_samples: Optional[int] = None,
    regularization: str = "none",
    reg_param: float = 0.01,
) -> np.ndarray:
    """
    Compute precision matrix with optional regularization.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix
    n_samples : int, optional
        Number of samples (for Hartlap correction)
    regularization : str
        Regularization method: 'none', 'shrinkage', 'eigenvalue_floor'
    reg_param : float
        Regularization parameter

    Returns
    -------
    np.ndarray
        Precision matrix
    """
    cov = covariance.copy()
    n_bins = cov.shape[0]

    # Apply regularization first
    if regularization == "shrinkage":
        cov = regularize_covariance(cov, method="shrinkage", alpha=reg_param)
    elif regularization == "eigenvalue_floor":
        cov = regularize_covariance(cov, method="eigenvalue_floor", floor=reg_param)

    # Invert
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix singular, using pseudoinverse")
        precision = np.linalg.pinv(cov)

    # Apply Hartlap correction if n_samples provided
    if n_samples is not None and n_samples > n_bins + 2:
        alpha = (n_samples - n_bins - 2) / (n_samples - 1)
        precision *= alpha

    return precision


def regularize_covariance(
    covariance: np.ndarray,
    method: str = "shrinkage",
    alpha: float = 0.1,
    floor: float = 1e-10,
) -> np.ndarray:
    """
    Regularize covariance matrix for numerical stability.

    Parameters
    ----------
    covariance : np.ndarray
        Input covariance matrix
    method : str
        Regularization method:
        - 'shrinkage': Shrink towards diagonal
        - 'eigenvalue_floor': Set minimum eigenvalue
        - 'ridge': Add scaled identity matrix
    alpha : float
        Shrinkage parameter (0 = no shrinkage, 1 = full diagonal)
    floor : float
        Minimum eigenvalue for eigenvalue_floor method

    Returns
    -------
    np.ndarray
        Regularized covariance matrix
    """
    n = covariance.shape[0]

    if method == "shrinkage":
        # Ledoit-Wolf style shrinkage towards diagonal
        diag = np.diag(np.diag(covariance))
        return (1 - alpha) * covariance + alpha * diag

    elif method == "eigenvalue_floor":
        # Ensure positive definiteness by flooring eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, floor)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    elif method == "ridge":
        # Add scaled identity matrix
        return covariance + alpha * np.eye(n) * np.mean(np.diag(covariance))

    else:
        raise ValueError(f"Unknown regularization method: {method}")


def check_covariance_condition(
    covariance: np.ndarray,
    warn_threshold: float = 1e6,
) -> dict:
    """
    Check numerical condition of covariance matrix.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix
    warn_threshold : float
        Threshold for warning on condition number

    Returns
    -------
    dict
        Diagnostic information
    """
    eigenvalues = np.linalg.eigvalsh(covariance)

    diagnostics = {
        "shape": covariance.shape,
        "is_symmetric": np.allclose(covariance, covariance.T),
        "eigenvalue_min": float(np.min(eigenvalues)),
        "eigenvalue_max": float(np.max(eigenvalues)),
        "condition_number": float(np.max(eigenvalues) / np.max(np.min(eigenvalues), 1e-20)),
        "is_positive_definite": np.all(eigenvalues > 0),
        "n_negative_eigenvalues": int(np.sum(eigenvalues < 0)),
    }

    if diagnostics["condition_number"] > warn_threshold:
        logger.warning(
            f"Covariance condition number {diagnostics['condition_number']:.2e} "
            f"is large - consider regularization"
        )

    if not diagnostics["is_positive_definite"]:
        logger.warning(
            f"Covariance has {diagnostics['n_negative_eigenvalues']} "
            f"negative eigenvalues"
        )

    return diagnostics


def compute_chi2(
    data: np.ndarray,
    model: np.ndarray,
    precision: np.ndarray,
) -> float:
    """
    Compute chi-squared statistic.

    χ² = (d - m)ᵀ Ψ (d - m)

    Parameters
    ----------
    data : np.ndarray
        Data vector
    model : np.ndarray
        Model prediction
    precision : np.ndarray
        Precision matrix

    Returns
    -------
    float
        Chi-squared value
    """
    residual = data - model
    return float(residual @ precision @ residual)


def compute_pte(
    chi2: float,
    n_dof: int,
) -> float:
    """
    Compute probability to exceed (p-value) for chi-squared.

    Parameters
    ----------
    chi2 : float
        Chi-squared value
    n_dof : int
        Number of degrees of freedom

    Returns
    -------
    float
        Probability to exceed (p-value)
    """
    from scipy.stats import chi2 as chi2_dist
    return 1 - chi2_dist.cdf(chi2, n_dof)
