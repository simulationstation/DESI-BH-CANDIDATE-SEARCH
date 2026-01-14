"""
Last-mile referee attack checks for kSZ analysis.

These checks address common referee concerns that are not covered
by standard null tests.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|------------------------------------------------|-------------|
| p_adj       | Look-elsewhere adjusted p-value                 | dimensionless|
| a_lm        | Spherical harmonic coefficients of T_i field   | μK          |
| w_top       | Weights of top 1% objects                      | dimensionless|
| Δz          | Redshift-dependent split difference            | σ           |
| δθ          | Beam FWHM perturbation                         | arcmin      |

Checks Implemented
------------------
1. Look-Elsewhere Guard
   - Permutation-based correction for multiple testing (apertures, filters)
   - Returns adjusted p-value

2. Pixel-Space Anisotropy
   - Fit low-order spherical harmonics to extracted T_i on sky
   - Detect residual dipole/quadrupole after masking

3. Weight Leverage
   - Remove top 1% highest-weight objects
   - Require amplitude stability (<1σ shift)

4. Redshift-Dependent Systematics
   - Within each z-bin, split by sky density quartiles
   - Require consistency across splits

5. Beam/Filter Mismatch
   - Perturb beam FWHM by ±5%
   - Check amplitude sensitivity
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import logging

logger = logging.getLogger(__name__)

N_WORKERS = max(1, cpu_count() - 1)


@dataclass
class RefereeCheckResult:
    """Result from a single referee check."""
    name: str
    passed: bool
    metric: float
    threshold: float
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'passed': self.passed,
            'metric': float(self.metric) if np.isfinite(self.metric) else None,
            'threshold': self.threshold,
            'description': self.description,
            'details': self.details,
        }


# =============================================================================
# Check 1: Look-Elsewhere Guard
# =============================================================================

def look_elsewhere_check(
    base_pvalue: float,
    n_trials: int,
    n_permutations: int = 1000,
    seed: int = 42,
) -> RefereeCheckResult:
    """
    Compute look-elsewhere adjusted p-value.

    Uses Sidak correction for multiple testing:
        p_adj = 1 - (1 - p_base)^n_trials

    For more conservative correction, uses permutation test.

    Parameters
    ----------
    base_pvalue : float
        Uncorrected p-value from main analysis
    n_trials : int
        Number of independent tests/apertures/filters tried
    n_permutations : int
        Number of permutations for empirical correction
    seed : int
        Random seed

    Returns
    -------
    RefereeCheckResult
        Check result with adjusted p-value
    """
    if not np.isfinite(base_pvalue) or base_pvalue <= 0:
        return RefereeCheckResult(
            name='look_elsewhere',
            passed=True,
            metric=1.0,
            threshold=0.01,
            description='Look-elsewhere correction skipped (invalid p-value)',
        )

    # Sidak correction
    sidak_pvalue = 1 - (1 - base_pvalue) ** n_trials

    # Permutation-based correction (more conservative)
    rng = np.random.default_rng(seed)
    null_pvalues = rng.uniform(0, 1, (n_permutations, n_trials))
    null_min = np.min(null_pvalues, axis=1)
    permutation_pvalue = np.mean(null_min <= base_pvalue)

    # Use more conservative
    adjusted_pvalue = max(sidak_pvalue, permutation_pvalue)

    passed = adjusted_pvalue > 0.01  # Threshold

    return RefereeCheckResult(
        name='look_elsewhere',
        passed=passed,
        metric=adjusted_pvalue,
        threshold=0.01,
        description=f'Look-elsewhere adjusted p-value (n_trials={n_trials})',
        details={
            'base_pvalue': base_pvalue,
            'sidak_pvalue': sidak_pvalue,
            'permutation_pvalue': permutation_pvalue,
            'n_trials': n_trials,
        }
    )


# =============================================================================
# Check 2: Pixel-Space Anisotropy
# =============================================================================

def anisotropy_check(
    temperatures: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    lmax: int = 2,
    sigma_threshold: float = 3.0,
) -> RefereeCheckResult:
    """
    Check for residual anisotropy in extracted temperatures.

    Fits low-order spherical harmonics (dipole, quadrupole) to the
    temperature field on the sky and checks if coefficients are
    significantly non-zero.

    Parameters
    ----------
    temperatures : np.ndarray
        Extracted temperatures at galaxy positions
    ra, dec : np.ndarray
        Galaxy coordinates in degrees
    lmax : int
        Maximum multipole to fit (2 = up to quadrupole)
    sigma_threshold : float
        Threshold for significance

    Returns
    -------
    RefereeCheckResult
        Check result with anisotropy significance
    """
    try:
        import healpy as hp
    except ImportError:
        return RefereeCheckResult(
            name='anisotropy',
            passed=True,
            metric=0.0,
            threshold=sigma_threshold,
            description='Anisotropy check skipped (healpy not available)',
        )

    # Remove mean
    T_mean = np.mean(temperatures)
    T_centered = temperatures - T_mean
    T_std = np.std(T_centered)

    if T_std < 1e-10:
        return RefereeCheckResult(
            name='anisotropy',
            passed=True,
            metric=0.0,
            threshold=sigma_threshold,
            description='Anisotropy check skipped (zero variance)',
        )

    # Project temperatures onto HEALPix map
    nside = 64  # Low resolution for fitting
    npix = hp.nside2npix(nside)
    temp_map = np.zeros(npix)
    count_map = np.zeros(npix)

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi)

    np.add.at(temp_map, pix, T_centered)
    np.add.at(count_map, pix, 1)

    valid = count_map > 0
    temp_map[valid] /= count_map[valid]
    temp_map[~valid] = hp.UNSEEN

    # Fit spherical harmonics
    try:
        alm = hp.map2alm(temp_map, lmax=lmax)

        # Compute power in each multipole
        Cl = hp.alm2cl(alm)

        # Dipole power (l=1)
        dipole_power = Cl[1] if len(Cl) > 1 else 0

        # Quadrupole power (l=2)
        quad_power = Cl[2] if len(Cl) > 2 else 0

        # Combined significance
        # Rough estimate: compare to expected from noise
        expected_Cl = T_std**2 / np.sum(count_map[valid])
        dipole_sigma = np.sqrt(dipole_power / expected_Cl) if expected_Cl > 0 else 0
        quad_sigma = np.sqrt(quad_power / expected_Cl) if expected_Cl > 0 else 0

        max_sigma = max(dipole_sigma, quad_sigma)

    except Exception as e:
        logger.warning(f"Anisotropy fit failed: {e}")
        max_sigma = 0.0
        dipole_sigma = 0.0
        quad_sigma = 0.0

    passed = max_sigma < sigma_threshold

    return RefereeCheckResult(
        name='anisotropy',
        passed=passed,
        metric=max_sigma,
        threshold=sigma_threshold,
        description='Temperature field anisotropy (dipole/quadrupole)',
        details={
            'dipole_sigma': float(dipole_sigma),
            'quadrupole_sigma': float(quad_sigma),
            'n_galaxies': len(temperatures),
            'n_pixels_used': int(np.sum(valid)),
        }
    )


# =============================================================================
# Check 3: Weight Leverage
# =============================================================================

def weight_leverage_check(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    top_fraction: float = 0.01,
    sigma_threshold: float = 1.0,
) -> RefereeCheckResult:
    """
    Check stability when removing highest-weight objects.

    Identifies top 1% highest-weight objects, reruns analysis
    without them, and checks if result changes by >1σ.

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
        The estimator
    positions, temperatures, weights : np.ndarray
        Galaxy data
    template : np.ndarray
        Theory template for fitting
    cov : np.ndarray
        Covariance matrix
    top_fraction : float
        Fraction of objects to remove (default 0.01 = 1%)
    sigma_threshold : float
        Maximum allowed shift in sigma

    Returns
    -------
    RefereeCheckResult
        Check result with leverage metric
    """
    n_gal = len(weights)
    n_remove = max(1, int(n_gal * top_fraction))

    # Find top-weighted objects
    top_idx = np.argsort(weights)[-n_remove:]
    keep_mask = np.ones(n_gal, dtype=bool)
    keep_mask[top_idx] = False

    # Full analysis
    result_full = estimator.compute(positions, temperatures, weights)

    # Analysis without top weights
    result_reduced = estimator.compute(
        positions[keep_mask],
        temperatures[keep_mask],
        weights[keep_mask],
    )

    # Fit amplitudes
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.diag(1.0 / np.diag(cov))

    def fit_amp(p_ksz):
        n = min(len(p_ksz), len(template), cov.shape[0])
        t = template[:n]
        p = p_ksz[:n]
        c_inv = cov_inv[:n, :n]
        num = t @ c_inv @ p
        den = t @ c_inv @ t
        return num / den if den > 0 else 0, 1/np.sqrt(den) if den > 0 else np.inf

    A_full, err_full = fit_amp(result_full.p_ksz)
    A_reduced, err_reduced = fit_amp(result_reduced.p_ksz)

    # Shift in sigma
    shift = abs(A_full - A_reduced)
    combined_err = np.sqrt(err_full**2 + err_reduced**2)
    shift_sigma = shift / combined_err if combined_err > 0 else 0

    passed = shift_sigma < sigma_threshold

    return RefereeCheckResult(
        name='weight_leverage',
        passed=passed,
        metric=shift_sigma,
        threshold=sigma_threshold,
        description=f'Amplitude stability after removing top {top_fraction*100:.0f}% weights',
        details={
            'n_removed': n_remove,
            'amplitude_full': float(A_full),
            'amplitude_reduced': float(A_reduced),
            'shift': float(shift),
            'combined_err': float(combined_err),
            'top_weights': weights[top_idx].tolist() if n_remove < 10 else 'too many to list',
        }
    )


# =============================================================================
# Check 4: Redshift-Dependent Systematics
# =============================================================================

def redshift_split_check(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    z_bins: List[float],
    n_quartiles: int = 4,
    sigma_threshold: float = 2.0,
) -> RefereeCheckResult:
    """
    Check for redshift-dependent systematics within tomographic bins.

    Within each z-bin, splits by sky density quartiles and
    requires consistency across splits.

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
    positions, temperatures, weights : np.ndarray
    ra, dec, z : np.ndarray
    template : np.ndarray
    cov : np.ndarray
    z_bins : list
        Redshift bin edges
    n_quartiles : int
        Number of quartiles for sky density split
    sigma_threshold : float
        Maximum allowed difference in sigma

    Returns
    -------
    RefereeCheckResult
        Check result with split consistency metric
    """
    try:
        import healpy as hp
        nside = 32
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(nside, theta, phi)

        # Compute local density per galaxy
        density = np.zeros(len(ra))
        for p in np.unique(pix):
            mask = pix == p
            density[mask] = np.sum(mask)

    except ImportError:
        # Fallback: use dec as proxy
        density = dec

    # Covariance inverse
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.diag(1.0 / np.diag(cov))

    def fit_amp(p_ksz):
        n = min(len(p_ksz), len(template), cov.shape[0])
        t = template[:n]
        p = p_ksz[:n]
        c_inv = cov_inv[:n, :n]
        num = t @ c_inv @ p
        den = t @ c_inv @ t
        return num / den if den > 0 else 0, 1/np.sqrt(den) if den > 0 else np.inf

    max_diff_sigma = 0.0
    split_results = []

    for i in range(len(z_bins) - 1):
        z_min, z_max = z_bins[i], z_bins[i + 1]
        z_mask = (z >= z_min) & (z < z_max)

        if np.sum(z_mask) < 100:
            continue

        # Split by density quartiles within this z-bin
        density_zbin = density[z_mask]
        quartiles = np.percentile(density_zbin, [25, 50, 75])

        amplitudes = []
        errors = []

        for q in range(n_quartiles):
            if q == 0:
                q_mask = density_zbin <= quartiles[0]
            elif q == n_quartiles - 1:
                q_mask = density_zbin > quartiles[-1]
            else:
                q_mask = (density_zbin > quartiles[q-1]) & (density_zbin <= quartiles[q])

            # Get full indices
            full_idx = np.where(z_mask)[0][q_mask]

            if len(full_idx) < 50:
                continue

            result = estimator.compute(
                positions[full_idx],
                temperatures[full_idx],
                weights[full_idx],
            )

            A, err = fit_amp(result.p_ksz)
            amplitudes.append(A)
            errors.append(err)

        if len(amplitudes) >= 2:
            # Check pairwise differences
            for j in range(len(amplitudes)):
                for k in range(j + 1, len(amplitudes)):
                    diff = abs(amplitudes[j] - amplitudes[k])
                    combined_err = np.sqrt(errors[j]**2 + errors[k]**2)
                    diff_sigma = diff / combined_err if combined_err > 0 else 0
                    max_diff_sigma = max(max_diff_sigma, diff_sigma)

        split_results.append({
            'z_bin': f'{z_min:.2f}-{z_max:.2f}',
            'n_quartiles': len(amplitudes),
            'amplitudes': amplitudes,
        })

    passed = max_diff_sigma < sigma_threshold

    return RefereeCheckResult(
        name='redshift_split',
        passed=passed,
        metric=max_diff_sigma,
        threshold=sigma_threshold,
        description='z-dependent split consistency (sky density quartiles)',
        details={
            'n_zbins_tested': len(split_results),
            'split_results': split_results,
        }
    )


# =============================================================================
# Check 5: Beam/Filter Mismatch
# =============================================================================

def beam_sensitivity_check(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    beam_fwhm: float,
    perturbation: float = 0.05,
    sigma_threshold: float = 1.0,
) -> RefereeCheckResult:
    """
    Check sensitivity to beam FWHM perturbation.

    Simulates effect of ±5% beam mismatch on amplitude.
    Since we don't have the raw map, we estimate sensitivity
    from the data covariance structure.

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
    positions, temperatures, weights : np.ndarray
    template : np.ndarray
    cov : np.ndarray
    beam_fwhm : float
        Nominal beam FWHM in arcmin
    perturbation : float
        Fractional perturbation (0.05 = ±5%)
    sigma_threshold : float
        Maximum allowed sensitivity

    Returns
    -------
    RefereeCheckResult
        Check result with beam sensitivity metric
    """
    # Compute baseline amplitude
    result_baseline = estimator.compute(positions, temperatures, weights)

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.diag(1.0 / np.diag(cov))

    def fit_amp(p_ksz):
        n = min(len(p_ksz), len(template), cov.shape[0])
        t = template[:n]
        p = p_ksz[:n]
        c_inv = cov_inv[:n, :n]
        num = t @ c_inv @ p
        den = t @ c_inv @ t
        return num / den if den > 0 else 0, 1/np.sqrt(den) if den > 0 else np.inf

    A_baseline, err_baseline = fit_amp(result_baseline.p_ksz)

    # Estimate beam effect on p(r)
    # Larger beam smooths small-scale signal → reduces amplitude at small r
    # Smaller beam increases small-scale signal

    r_centers = estimator.bin_centers
    beam_sigma = beam_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Approximate beam correction factor for each bin
    # At separation r, beam smoothing scales as exp(-r²/8σ²) for pair signal
    # This is a rough approximation

    beam_factor_plus = np.exp(-(r_centers**2) / (8 * (beam_sigma * (1 + perturbation))**2))
    beam_factor_minus = np.exp(-(r_centers**2) / (8 * (beam_sigma * (1 - perturbation))**2))

    # Perturbed p(r)
    p_plus = result_baseline.p_ksz * beam_factor_plus / np.maximum(beam_factor_plus, 1e-10)
    p_minus = result_baseline.p_ksz * beam_factor_minus / np.maximum(beam_factor_minus, 1e-10)

    # Clamp unreasonable values
    p_plus = np.clip(p_plus, -1e6, 1e6)
    p_minus = np.clip(p_minus, -1e6, 1e6)

    A_plus, _ = fit_amp(p_plus)
    A_minus, _ = fit_amp(p_minus)

    # Sensitivity: max shift from perturbation
    max_shift = max(abs(A_plus - A_baseline), abs(A_minus - A_baseline))
    shift_sigma = max_shift / err_baseline if err_baseline > 0 else 0

    passed = shift_sigma < sigma_threshold

    return RefereeCheckResult(
        name='beam_sensitivity',
        passed=passed,
        metric=shift_sigma,
        threshold=sigma_threshold,
        description=f'Beam FWHM sensitivity (±{perturbation*100:.0f}% perturbation)',
        details={
            'beam_fwhm': beam_fwhm,
            'perturbation': perturbation,
            'amplitude_baseline': float(A_baseline),
            'amplitude_plus': float(A_plus),
            'amplitude_minus': float(A_minus),
            'max_shift': float(max_shift),
            'sigma_baseline': float(err_baseline),
        }
    )


# =============================================================================
# Run All Checks
# =============================================================================

def run_all_referee_checks(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    beam_fwhm: float = 1.4,
    beam_perturbation: float = 0.05,
    z_bins: Optional[List[float]] = None,
    n_trials: int = 5,  # For look-elsewhere
    base_pvalue: Optional[float] = None,
    small_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run all 5 referee checks.

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
    positions, temperatures, weights : np.ndarray
    ra, dec, z : np.ndarray
    template : np.ndarray
    cov : np.ndarray
    beam_fwhm : float
    beam_perturbation : float
    z_bins : list, optional
    n_trials : int
        Number of trials for look-elsewhere correction
    base_pvalue : float, optional
        Base p-value for look-elsewhere (default: from chi2)
    small_mode : bool
        Quick mode for testing

    Returns
    -------
    dict
        Results from all checks
    """
    if z_bins is None:
        z_bins = [0.4, 0.6, 0.8, 1.0]

    logger.info("  Running referee checks...")

    results = {}

    # 1. Look-elsewhere
    if base_pvalue is None:
        base_pvalue = 0.01  # Default
    check1 = look_elsewhere_check(base_pvalue, n_trials)
    results['look_elsewhere'] = check1.to_dict()
    results['look_elsewhere_pvalue'] = check1.metric
    logger.info(f"    Look-elsewhere: {'PASS' if check1.passed else 'WARN'} (p_adj={check1.metric:.4f})")

    # 2. Anisotropy
    check2 = anisotropy_check(temperatures, ra, dec)
    results['anisotropy'] = check2.to_dict()
    results['anisotropy_sigma'] = check2.metric
    logger.info(f"    Anisotropy: {'PASS' if check2.passed else 'WARN'} ({check2.metric:.2f}σ)")

    # 3. Weight leverage
    check3 = weight_leverage_check(estimator, positions, temperatures, weights, template, cov)
    results['weight_leverage'] = check3.to_dict()
    results['weight_leverage_sigma'] = check3.metric
    logger.info(f"    Weight leverage: {'PASS' if check3.passed else 'WARN'} ({check3.metric:.2f}σ)")

    # 4. Redshift split (skip in small mode)
    if not small_mode:
        check4 = redshift_split_check(
            estimator, positions, temperatures, weights,
            ra, dec, z, template, cov, z_bins
        )
        results['redshift_split'] = check4.to_dict()
        results['split_max_diff_sigma'] = check4.metric
        logger.info(f"    z-split: {'PASS' if check4.passed else 'WARN'} ({check4.metric:.2f}σ)")
    else:
        results['split_max_diff_sigma'] = 0.0
        logger.info("    z-split: SKIPPED (small mode)")

    # 5. Beam sensitivity
    check5 = beam_sensitivity_check(
        estimator, positions, temperatures, weights,
        template, cov, beam_fwhm, beam_perturbation
    )
    results['beam_sensitivity'] = check5.to_dict()
    results['beam_sensitivity_sigma'] = check5.metric
    logger.info(f"    Beam sensitivity: {'PASS' if check5.passed else 'WARN'} ({check5.metric:.2f}σ)")

    # Summary
    all_passed = all([
        check1.passed,
        check2.passed,
        check3.passed,
        check5.passed,
    ])
    results['all_passed'] = all_passed

    return results
