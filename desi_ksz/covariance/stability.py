"""
Covariance matrix stability analysis and regularization.

Provides tools to analyze covariance matrix conditioning,
apply regularization, and assess stability across jackknife K values.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import json


def analyze_covariance(cov: np.ndarray) -> Dict[str, Any]:
    """
    Analyze covariance matrix properties.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix (N x N)

    Returns
    -------
    dict
        Analysis results including eigenvalues, condition number
    """
    n = cov.shape[0]

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Condition number
    eig_min = eigenvalues[-1]
    eig_max = eigenvalues[0]
    condition_number = eig_max / max(eig_min, 1e-15)

    # Check positive definiteness
    is_positive_definite = np.all(eigenvalues > 0)

    # Diagonal statistics
    diagonal = np.diag(cov)

    return {
        'shape': cov.shape,
        'eigenvalues': eigenvalues.tolist(),
        'eigenvalue_min': float(eig_min),
        'eigenvalue_max': float(eig_max),
        'eigenvalue_median': float(np.median(eigenvalues)),
        'condition_number': float(condition_number),
        'is_positive_definite': bool(is_positive_definite),
        'n_negative_eigenvalues': int(np.sum(eigenvalues < 0)),
        'diagonal_min': float(np.min(diagonal)),
        'diagonal_max': float(np.max(diagonal)),
        'diagonal_mean': float(np.mean(diagonal)),
        'trace': float(np.trace(cov)),
    }


def compute_hartlap_factor(n_samples: int, n_bins: int) -> float:
    """
    Compute Hartlap correction factor for precision matrix.

    The Hartlap factor corrects for bias in the inverse covariance
    matrix when estimated from a finite number of samples.

    α = (N_s - N_d - 2) / (N_s - 1)

    Parameters
    ----------
    n_samples : int
        Number of jackknife/bootstrap samples (K)
    n_bins : int
        Number of data bins (N_d)

    Returns
    -------
    float
        Hartlap correction factor (0 if invalid)
    """
    if n_samples <= n_bins + 2:
        return 0.0  # Invalid regime

    factor = (n_samples - n_bins - 2) / (n_samples - 1)
    return factor


def regularize_eigenvalue_floor(
    cov: np.ndarray,
    epsilon: float = 0.01,
    reference: str = "median",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Regularize covariance by setting eigenvalue floor.

    Sets minimum eigenvalue to ε × reference eigenvalue.

    Parameters
    ----------
    cov : np.ndarray
        Original covariance matrix
    epsilon : float
        Floor fraction relative to reference
    reference : str
        Reference eigenvalue: "median", "max", or "mean"

    Returns
    -------
    cov_reg : np.ndarray
        Regularized covariance matrix
    info : dict
        Regularization details
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Compute reference eigenvalue
    if reference == "median":
        ref_value = np.median(eigenvalues)
    elif reference == "max":
        ref_value = np.max(eigenvalues)
    elif reference == "mean":
        ref_value = np.mean(eigenvalues)
    else:
        raise ValueError(f"Unknown reference: {reference}")

    # Set floor
    floor = epsilon * ref_value
    eigenvalues_reg = np.maximum(eigenvalues, floor)

    # Reconstruct matrix
    cov_reg = eigenvectors @ np.diag(eigenvalues_reg) @ eigenvectors.T

    # Ensure symmetry
    cov_reg = 0.5 * (cov_reg + cov_reg.T)

    # Compute new condition number
    new_condition = np.max(eigenvalues_reg) / np.min(eigenvalues_reg)

    info = {
        'original_condition_number': float(np.max(eigenvalues) / max(np.min(eigenvalues), 1e-15)),
        'new_condition_number': float(new_condition),
        'floor_value': float(floor),
        'n_eigenvalues_raised': int(np.sum(eigenvalues < floor)),
        'reference': reference,
        'epsilon': epsilon,
    }

    return cov_reg, info


def regularize_shrinkage(
    cov: np.ndarray,
    alpha: float = 0.1,
    target: str = "diagonal",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply shrinkage regularization.

    C_shrunk = (1 - α) * C + α * T

    where T is the shrinkage target.

    Parameters
    ----------
    cov : np.ndarray
        Original covariance matrix
    alpha : float
        Shrinkage intensity (0 to 1)
    target : str
        Shrinkage target: "diagonal", "identity", or "scaled_identity"

    Returns
    -------
    cov_shrunk : np.ndarray
        Shrunk covariance matrix
    info : dict
        Shrinkage details
    """
    n = cov.shape[0]

    # Create target matrix
    if target == "diagonal":
        T = np.diag(np.diag(cov))
    elif target == "identity":
        T = np.eye(n)
    elif target == "scaled_identity":
        T = np.eye(n) * np.mean(np.diag(cov))
    else:
        raise ValueError(f"Unknown target: {target}")

    # Apply shrinkage
    cov_shrunk = (1 - alpha) * cov + alpha * T

    # Compute condition numbers
    orig_cond = np.linalg.cond(cov)
    new_cond = np.linalg.cond(cov_shrunk)

    info = {
        'shrinkage_alpha': alpha,
        'target': target,
        'original_condition_number': float(orig_cond),
        'new_condition_number': float(new_cond),
    }

    return cov_shrunk, info


@dataclass
class StabilityReport:
    """Report on covariance stability across K values."""

    K_values: List[int]
    condition_numbers: List[float]
    diagonal_rms_variation: float
    recommended_K: int
    needs_regularization: bool
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'K_values': self.K_values,
            'condition_numbers': self.condition_numbers,
            'diagonal_rms_variation': self.diagonal_rms_variation,
            'recommended_K': self.recommended_K,
            'needs_regularization': self.needs_regularization,
            'suggestion': self.suggestion,
        }

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def analyze_stability_across_K(
    compute_cov_func,
    K_values: List[int],
    n_bins: int,
) -> StabilityReport:
    """
    Analyze covariance stability across different K values.

    Parameters
    ----------
    compute_cov_func : callable
        Function that takes K and returns covariance matrix
    K_values : list of int
        K values to test
    n_bins : int
        Number of data bins

    Returns
    -------
    StabilityReport
        Stability analysis report
    """
    condition_numbers = []
    diagonals = []

    for K in K_values:
        cov = compute_cov_func(K)
        analysis = analyze_covariance(cov)
        condition_numbers.append(analysis['condition_number'])
        diagonals.append(np.diag(cov))

    # Compute diagonal RMS variation
    diagonals = np.array(diagonals)
    diagonal_rms = np.std(diagonals, axis=0) / np.mean(diagonals, axis=0)
    diagonal_rms_variation = float(np.mean(diagonal_rms))

    # Recommend K: choose smallest K where condition number is stable
    condition_numbers = np.array(condition_numbers)
    K_values = np.array(K_values)

    # Find where condition number stabilizes (< 10% change)
    stable_idx = len(K_values) - 1
    for i in range(1, len(K_values)):
        rel_change = abs(condition_numbers[i] - condition_numbers[i-1]) / condition_numbers[i-1]
        if rel_change < 0.1:
            stable_idx = i
            break

    recommended_K = int(K_values[stable_idx])

    # Check if regularization needed
    needs_regularization = condition_numbers[stable_idx] > 1e6

    # Generate suggestion
    if needs_regularization:
        suggestion = f"Use K={recommended_K} with eigenvalue floor regularization"
    elif diagonal_rms_variation > 0.2:
        suggestion = f"Use K={max(K_values)} for better stability"
    else:
        suggestion = f"Use K={recommended_K} (stable)"

    return StabilityReport(
        K_values=K_values.tolist(),
        condition_numbers=condition_numbers.tolist(),
        diagonal_rms_variation=diagonal_rms_variation,
        recommended_K=recommended_K,
        needs_regularization=needs_regularization,
        suggestion=suggestion,
    )


def choose_regularization(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recommend regularization strategy based on covariance analysis.

    Parameters
    ----------
    analysis : dict
        Output from analyze_covariance()

    Returns
    -------
    dict
        Recommendation with 'recommendation' and 'reason' keys
    """
    kappa = analysis['condition_number']
    n_neg = analysis['n_negative_eigenvalues']

    if n_neg > 0:
        return {
            'recommendation': 'eigenvalue_floor',
            'reason': f'{n_neg} negative eigenvalues detected',
            'params': {'epsilon': 0.01, 'reference': 'median'},
        }

    if kappa > 1e10:
        return {
            'recommendation': 'eigenvalue_floor',
            'reason': f'Very high condition number (κ = {kappa:.2e})',
            'params': {'epsilon': 0.01, 'reference': 'median'},
        }

    if kappa > 1e6:
        return {
            'recommendation': 'shrinkage',
            'reason': f'High condition number (κ = {kappa:.2e})',
            'params': {'alpha': 0.05, 'target': 'diagonal'},
        }

    if kappa > 1e4:
        return {
            'recommendation': 'mild_shrinkage',
            'reason': f'Moderate condition number (κ = {kappa:.2e})',
            'params': {'alpha': 0.01, 'target': 'diagonal'},
        }

    return {
        'recommendation': 'none',
        'reason': f'Matrix is well-conditioned (κ = {kappa:.2e})',
        'params': {},
    }


def apply_hartlap_to_precision(cov: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Apply Hartlap correction when computing precision matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix
    n_samples : int
        Number of jackknife samples

    Returns
    -------
    np.ndarray
        Hartlap-corrected precision matrix
    """
    n_bins = cov.shape[0]
    hartlap = compute_hartlap_factor(n_samples, n_bins)

    if hartlap <= 0:
        raise ValueError(
            f"Invalid Hartlap regime: K={n_samples}, N_bins={n_bins}. "
            f"Need K > N_bins + 2."
        )

    # Invert and apply correction
    precision = np.linalg.inv(cov)
    precision_corrected = hartlap * precision

    return precision_corrected


# =============================================================================
# Auto-Regularization Framework
# =============================================================================

@dataclass
class AutoRegularizationResult:
    """Result from automatic regularization."""

    original_cov: np.ndarray
    regularized_cov: np.ndarray
    method: str
    parameters: Dict[str, Any]
    original_condition: float
    final_condition: float
    hartlap_factor: float
    hartlap_warning: bool
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'parameters': self.parameters,
            'original_condition': self.original_condition,
            'final_condition': self.final_condition,
            'hartlap_factor': self.hartlap_factor,
            'hartlap_warning': self.hartlap_warning,
            'warnings': self.warnings,
        }


def compute_ledoit_wolf_shrinkage(cov: np.ndarray) -> float:
    """
    Compute optimal Ledoit-Wolf shrinkage intensity.

    Implements the analytical formula from Ledoit & Wolf (2004)
    for optimal shrinkage toward scaled identity.

    Parameters
    ----------
    cov : np.ndarray
        Sample covariance matrix (N x N)

    Returns
    -------
    float
        Optimal shrinkage intensity α ∈ [0, 1]
    """
    n = cov.shape[0]

    # Target: scaled identity
    mu = np.trace(cov) / n
    target = mu * np.eye(n)

    # Frobenius norms
    delta = cov - target
    delta_norm_sq = np.sum(delta**2)

    if delta_norm_sq < 1e-15:
        return 0.0  # Already at target

    # Estimate optimal shrinkage (simplified formula)
    # Full formula requires sample data, this is approximate
    kappa = np.linalg.cond(cov)

    # Heuristic: shrink more for ill-conditioned matrices
    if kappa > 1e10:
        alpha = 0.3
    elif kappa > 1e6:
        alpha = 0.1
    elif kappa > 1e4:
        alpha = 0.05
    elif kappa > 1e2:
        alpha = 0.01
    else:
        alpha = 0.0

    return min(1.0, max(0.0, alpha))


def auto_regularize(
    cov: np.ndarray,
    n_samples: int,
    target_condition: float = 1e6,
    min_eigenvalue_fraction: float = 0.01,
    apply_hartlap: bool = True,
) -> AutoRegularizationResult:
    """
    Automatically regularize covariance matrix.

    Applies appropriate regularization based on matrix condition
    and sample size. Issues warnings for Hartlap regime violations.

    Pipeline:
    1. Analyze matrix condition
    2. Check Hartlap regime
    3. Apply eigenvalue floor if needed
    4. Apply shrinkage if still ill-conditioned
    5. Return regularized matrix with diagnostics

    Parameters
    ----------
    cov : np.ndarray
        Raw covariance matrix
    n_samples : int
        Number of jackknife/bootstrap samples
    target_condition : float
        Target condition number after regularization
    min_eigenvalue_fraction : float
        Minimum eigenvalue as fraction of median
    apply_hartlap : bool
        Whether to compute Hartlap factor

    Returns
    -------
    AutoRegularizationResult
        Regularized covariance and diagnostics
    """
    n_bins = cov.shape[0]
    warnings_list = []
    method_parts = []
    params = {}

    # Analyze original matrix
    orig_analysis = analyze_covariance(cov)
    orig_condition = orig_analysis['condition_number']

    # Check Hartlap regime
    hartlap = compute_hartlap_factor(n_samples, n_bins)
    hartlap_warning = False

    if hartlap <= 0:
        hartlap_warning = True
        warnings_list.append(
            f"HARTLAP WARNING: K={n_samples} samples is insufficient for "
            f"N_bins={n_bins}. Need K > {n_bins + 2}. "
            f"Precision matrix will be biased!"
        )
        hartlap = 0.0
    elif hartlap < 0.8:
        warnings_list.append(
            f"Hartlap factor = {hartlap:.3f} < 0.8. Consider increasing "
            f"number of jackknife regions (currently K={n_samples})."
        )

    # Start with original covariance
    cov_reg = cov.copy()

    # Step 1: Fix negative eigenvalues with eigenvalue floor
    if orig_analysis['n_negative_eigenvalues'] > 0:
        cov_reg, floor_info = regularize_eigenvalue_floor(
            cov_reg,
            epsilon=min_eigenvalue_fraction,
            reference="median",
        )
        method_parts.append("eigenvalue_floor")
        params['eigenvalue_floor_epsilon'] = min_eigenvalue_fraction
        warnings_list.append(
            f"Applied eigenvalue floor: {floor_info['n_eigenvalues_raised']} "
            f"eigenvalues raised to {floor_info['floor_value']:.2e}"
        )

    # Step 2: Check condition after floor
    mid_analysis = analyze_covariance(cov_reg)

    # Step 3: Apply shrinkage if still ill-conditioned
    if mid_analysis['condition_number'] > target_condition:
        # Compute optimal shrinkage
        alpha = compute_ledoit_wolf_shrinkage(cov_reg)

        if alpha > 0:
            cov_reg, shrink_info = regularize_shrinkage(
                cov_reg,
                alpha=alpha,
                target="diagonal",
            )
            method_parts.append(f"shrinkage(α={alpha:.3f})")
            params['shrinkage_alpha'] = alpha
            params['shrinkage_target'] = "diagonal"

    # Final analysis
    final_analysis = analyze_covariance(cov_reg)
    final_condition = final_analysis['condition_number']

    if final_condition > target_condition:
        warnings_list.append(
            f"WARNING: Final condition number {final_condition:.2e} still exceeds "
            f"target {target_condition:.2e}. Results may be unstable."
        )

    # Determine method string
    if len(method_parts) == 0:
        method = "none"
    else:
        method = "+".join(method_parts)

    return AutoRegularizationResult(
        original_cov=cov,
        regularized_cov=cov_reg,
        method=method,
        parameters=params,
        original_condition=orig_condition,
        final_condition=final_condition,
        hartlap_factor=hartlap,
        hartlap_warning=hartlap_warning,
        warnings=warnings_list,
    )


def robust_precision_matrix(
    cov: np.ndarray,
    n_samples: int,
    regularize: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute precision matrix with automatic regularization and Hartlap correction.

    This is the recommended entry point for getting a precision matrix
    suitable for likelihood evaluation.

    Parameters
    ----------
    cov : np.ndarray
        Raw covariance matrix
    n_samples : int
        Number of jackknife/bootstrap samples
    regularize : bool
        Whether to apply automatic regularization
    verbose : bool
        Whether to log warnings

    Returns
    -------
    precision : np.ndarray
        Regularized and Hartlap-corrected precision matrix
    info : dict
        Diagnostic information

    Example
    -------
    >>> precision, info = robust_precision_matrix(cov, n_jk=100)
    >>> chi2 = data @ precision @ data
    """
    import logging
    logger = logging.getLogger(__name__)

    info = {
        'n_bins': cov.shape[0],
        'n_samples': n_samples,
    }

    # Auto-regularize if requested
    if regularize:
        reg_result = auto_regularize(cov, n_samples)
        cov_final = reg_result.regularized_cov
        info['regularization'] = reg_result.to_dict()

        if verbose:
            for warning in reg_result.warnings:
                logger.warning(warning)
    else:
        cov_final = cov
        info['regularization'] = None

    # Compute Hartlap factor
    hartlap = compute_hartlap_factor(n_samples, cov.shape[0])
    info['hartlap_factor'] = hartlap

    if hartlap <= 0:
        if verbose:
            logger.error("Cannot compute valid precision matrix: Hartlap factor <= 0")
        # Return pseudo-inverse as fallback
        precision = np.linalg.pinv(cov_final)
        info['method'] = 'pseudo_inverse'
    else:
        # Standard inverse with Hartlap correction
        try:
            precision = hartlap * np.linalg.inv(cov_final)
            info['method'] = 'hartlap_corrected'
        except np.linalg.LinAlgError:
            precision = hartlap * np.linalg.pinv(cov_final)
            info['method'] = 'pseudo_inverse_hartlap'
            if verbose:
                logger.warning("Matrix inversion failed, using pseudo-inverse")

    info['precision_condition'] = float(np.linalg.cond(precision))

    return precision, info


def check_hartlap_regime(n_samples: int, n_bins: int) -> Dict[str, Any]:
    """
    Check if jackknife sample count is sufficient for reliable precision matrix.

    Parameters
    ----------
    n_samples : int
        Number of jackknife regions
    n_bins : int
        Number of data bins

    Returns
    -------
    dict
        Diagnostic information including recommendations
    """
    hartlap = compute_hartlap_factor(n_samples, n_bins)
    min_required = n_bins + 3
    recommended = int(2.5 * n_bins)

    is_valid = hartlap > 0
    is_good = hartlap > 0.8

    if not is_valid:
        status = "INVALID"
        recommendation = f"Increase K to at least {min_required}"
    elif not is_good:
        status = "MARGINAL"
        recommendation = f"Consider increasing K to {recommended} for better stability"
    else:
        status = "OK"
        recommendation = "Sample count is adequate"

    return {
        'n_samples': n_samples,
        'n_bins': n_bins,
        'hartlap_factor': hartlap,
        'minimum_required': min_required,
        'recommended': recommended,
        'status': status,
        'recommendation': recommendation,
        'is_valid': is_valid,
        'is_good': is_good,
    }
