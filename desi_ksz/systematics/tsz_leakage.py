"""
tSZ leakage control and stress testing for kSZ analysis.

This module implements diagnostics for thermal Sunyaev-Zel'dovich (tSZ)
contamination in the pairwise kSZ measurement.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| y           | Compton-y parameter                             | dimensionless|
| T_tSZ       | tSZ temperature decrement                       | μK          |
| θ_mask      | Cluster mask radius                             | arcmin      |
| Δp(r)       | Change in pairwise momentum from masking        | μK          |
| A_kSZ       | kSZ amplitude                                   | dimensionless|

tSZ Contamination
-----------------
The tSZ effect produces a temperature decrement at CMB frequencies below
~217 GHz:

    T_tSZ = f(ν) × y × T_CMB

where f(ν) ≈ -2 for ν < 217 GHz. Clusters hosting LRGs produce negative
tSZ signal that can bias kSZ measurements.

References
----------
- Schaan, E., et al. 2021, PRD, 103, 063513
- Planck Collaboration 2016, A&A, 594, A27 (cluster catalog)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
from multiprocessing import Pool, cpu_count
import logging
import json

logger = logging.getLogger(__name__)

N_WORKERS = max(1, cpu_count() - 1)

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    hp = None
    HEALPY_AVAILABLE = False


@dataclass
class ClusterMaskSweepResult:
    """Result from cluster mask radius sweep test."""

    mask_radii_arcmin: List[float]
    amplitudes: List[float]
    amplitude_errors: List[float]
    delta_amplitudes: List[float]  # Relative to smallest mask
    p_ksz_by_radius: Dict[float, np.ndarray]
    n_clusters_masked: List[int]
    fraction_masked: List[float]
    baseline_amplitude: float
    converged: bool
    convergence_radius: Optional[float]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mask_radii_arcmin': self.mask_radii_arcmin,
            'amplitudes': self.amplitudes,
            'amplitude_errors': self.amplitude_errors,
            'delta_amplitudes': self.delta_amplitudes,
            'n_clusters_masked': self.n_clusters_masked,
            'fraction_masked': self.fraction_masked,
            'baseline_amplitude': self.baseline_amplitude,
            'converged': self.converged,
            'convergence_radius': self.convergence_radius,
            'recommendation': self.recommendation,
        }

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class YMapRegressionResult:
    """Result from y-map template regression."""

    regression_coefficient: float
    regression_coefficient_err: float
    residual_correlation: float
    original_amplitude: float
    cleaned_amplitude: float
    amplitude_shift: float
    amplitude_shift_sigma: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regression_coefficient': self.regression_coefficient,
            'regression_coefficient_err': self.regression_coefficient_err,
            'residual_correlation': self.residual_correlation,
            'original_amplitude': self.original_amplitude,
            'cleaned_amplitude': self.cleaned_amplitude,
            'amplitude_shift': self.amplitude_shift,
            'amplitude_shift_sigma': self.amplitude_shift_sigma,
            'passed': self.passed,
            'details': self.details,
        }


def load_planck_cluster_catalog(
    catalog_path: Optional[str] = None,
    min_snr: float = 4.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Planck cluster catalog (PSZ2 or similar).

    Parameters
    ----------
    catalog_path : str, optional
        Path to cluster catalog FITS file. If None, returns mock catalog.
    min_snr : float
        Minimum SNR for cluster selection

    Returns
    -------
    ra, dec : np.ndarray
        Cluster positions in degrees
    theta_500 : np.ndarray
        θ_500 angular sizes in arcmin
    """
    if catalog_path is not None:
        try:
            from astropy.io import fits
            with fits.open(catalog_path) as hdul:
                data = hdul[1].data
                # PSZ2 format
                ra = data['RA']
                dec = data['DEC']
                theta_500 = data.get('THETA', np.full(len(ra), 5.0))
                snr = data.get('SNR', np.full(len(ra), 10.0))

                mask = snr >= min_snr
                return ra[mask], dec[mask], theta_500[mask]
        except Exception as e:
            logger.warning(f"Failed to load cluster catalog: {e}")

    # Return empty arrays if no catalog
    logger.info("No cluster catalog provided - using empty catalog")
    return np.array([]), np.array([]), np.array([])


def create_cluster_mask(
    cluster_ra: np.ndarray,
    cluster_dec: np.ndarray,
    mask_radius_arcmin: float,
    nside: int = 2048,
    theta_500: Optional[np.ndarray] = None,
    use_theta_500_scaling: bool = False,
) -> np.ndarray:
    """
    Create HEALPix mask around clusters.

    Parameters
    ----------
    cluster_ra, cluster_dec : np.ndarray
        Cluster positions in degrees
    mask_radius_arcmin : float
        Base mask radius in arcmin
    nside : int
        HEALPix nside
    theta_500 : np.ndarray, optional
        θ_500 for each cluster (scales mask if use_theta_500_scaling)
    use_theta_500_scaling : bool
        If True, mask radius = mask_radius_arcmin × θ_500 / 5'

    Returns
    -------
    np.ndarray
        HEALPix mask (1 = valid, 0 = masked)
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required for cluster masking")

    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float32)

    if len(cluster_ra) == 0:
        return mask

    for i, (ra, dec) in enumerate(zip(cluster_ra, cluster_dec)):
        # Compute mask radius for this cluster
        if use_theta_500_scaling and theta_500 is not None:
            radius = mask_radius_arcmin * theta_500[i] / 5.0
        else:
            radius = mask_radius_arcmin

        # Convert to radians
        radius_rad = np.radians(radius / 60.0)

        # Get vector
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        vec = hp.ang2vec(theta, phi)

        # Query disc
        pixels = hp.query_disc(nside, vec, radius_rad)
        mask[pixels] = 0

    n_masked = np.sum(mask == 0)
    f_masked = n_masked / npix
    logger.debug(f"Masked {n_masked} pixels ({f_masked:.2%}) around {len(cluster_ra)} clusters")

    return mask


def cluster_mask_sweep(
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    cluster_ra: np.ndarray,
    cluster_dec: np.ndarray,
    mask_radii_arcmin: List[float] = [0, 5, 10, 15, 20],
    theta_500: Optional[np.ndarray] = None,
    nside: int = 2048,
    convergence_threshold: float = 0.5,  # σ
) -> ClusterMaskSweepResult:
    """
    Run cluster mask radius sweep to assess tSZ contamination.

    For each mask radius, recomputes the pairwise measurement excluding
    galaxies near clusters. If the amplitude stabilizes as mask radius
    increases, tSZ contamination is under control.

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
        The estimator to use
    positions, temperatures, weights : np.ndarray
        Galaxy data
    ra, dec : np.ndarray
        Galaxy sky coordinates in degrees
    template : np.ndarray
        Theory template for amplitude fitting
    cov : np.ndarray
        Covariance matrix
    cluster_ra, cluster_dec : np.ndarray
        Cluster positions in degrees
    mask_radii_arcmin : list of float
        Mask radii to test (arcmin)
    theta_500 : np.ndarray, optional
        Cluster angular sizes
    nside : int
        HEALPix nside for masking
    convergence_threshold : float
        Maximum Δ amplitude in σ for convergence

    Returns
    -------
    ClusterMaskSweepResult
        Sweep results including amplitude vs radius
    """
    logger.info(f"Running cluster mask sweep: radii = {mask_radii_arcmin} arcmin")

    if len(cluster_ra) == 0:
        logger.warning("No clusters provided - sweep will be trivial")

    # Convert galaxy positions to HEALPix pixels for masking
    if HEALPY_AVAILABLE:
        theta_gal = np.radians(90.0 - dec)
        phi_gal = np.radians(ra)
        gal_pixels = hp.ang2pix(nside, theta_gal, phi_gal)
    else:
        raise ImportError("healpy required for cluster mask sweep")

    amplitudes = []
    amplitude_errors = []
    p_ksz_by_radius = {}
    n_clusters_masked = []
    fraction_masked = []

    # Precompute covariance inverse
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.diag(1.0 / np.diag(cov))

    for radius in mask_radii_arcmin:
        if radius == 0:
            # No masking - use all galaxies
            mask_gal = np.ones(len(ra), dtype=bool)
        else:
            # Create cluster mask
            cluster_mask = create_cluster_mask(
                cluster_ra, cluster_dec, radius, nside, theta_500
            )
            # Apply to galaxies
            mask_gal = cluster_mask[gal_pixels] > 0.5

        n_valid = np.sum(mask_gal)
        frac = 1 - n_valid / len(ra)
        n_clusters_masked.append(len(cluster_ra) if radius > 0 else 0)
        fraction_masked.append(frac)

        if n_valid < 100:
            logger.warning(f"Radius {radius}': only {n_valid} galaxies left")
            amplitudes.append(np.nan)
            amplitude_errors.append(np.nan)
            continue

        # Compute p(r) with masked galaxies
        result = estimator.compute(
            positions[mask_gal],
            temperatures[mask_gal],
            weights[mask_gal],
        )

        p_ksz_by_radius[radius] = result.p_ksz.copy()

        # Fit amplitude
        numerator = template @ cov_inv @ result.p_ksz
        denominator = template @ cov_inv @ template
        A = numerator / denominator if denominator > 0 else 0
        sigma_A = 1.0 / np.sqrt(denominator) if denominator > 0 else np.inf

        amplitudes.append(float(A))
        amplitude_errors.append(float(sigma_A))

        logger.info(f"  Radius {radius}': A = {A:.4f} ± {sigma_A:.4f}, N_gal = {n_valid}")

    # Compute delta amplitudes relative to first valid measurement
    baseline_idx = 0
    while baseline_idx < len(amplitudes) and not np.isfinite(amplitudes[baseline_idx]):
        baseline_idx += 1

    if baseline_idx >= len(amplitudes):
        baseline_amplitude = 0.0
        delta_amplitudes = [0.0] * len(amplitudes)
    else:
        baseline_amplitude = amplitudes[baseline_idx]
        delta_amplitudes = [
            (A - baseline_amplitude) if np.isfinite(A) else np.nan
            for A in amplitudes
        ]

    # Check convergence: amplitude should stabilize at large radii
    converged = False
    convergence_radius = None

    for i in range(len(mask_radii_arcmin) - 1):
        if not np.isfinite(amplitudes[i]) or not np.isfinite(amplitudes[i+1]):
            continue

        diff = abs(amplitudes[i+1] - amplitudes[i])
        sigma = np.sqrt(amplitude_errors[i]**2 + amplitude_errors[i+1]**2)

        if sigma > 0 and diff / sigma < convergence_threshold:
            converged = True
            convergence_radius = mask_radii_arcmin[i]
            break

    # Generate recommendation
    if len(cluster_ra) == 0:
        recommendation = "No cluster catalog provided - cannot assess tSZ contamination"
    elif converged:
        recommendation = f"Amplitude stable at radius >= {convergence_radius}' - use this mask"
    else:
        recommendation = "Amplitude not converged - increase mask radius or add clusters"

    return ClusterMaskSweepResult(
        mask_radii_arcmin=mask_radii_arcmin,
        amplitudes=amplitudes,
        amplitude_errors=amplitude_errors,
        delta_amplitudes=delta_amplitudes,
        p_ksz_by_radius=p_ksz_by_radius,
        n_clusters_masked=n_clusters_masked,
        fraction_masked=fraction_masked,
        baseline_amplitude=baseline_amplitude,
        converged=converged,
        convergence_radius=convergence_radius,
        recommendation=recommendation,
    )


def regress_ymap_template(
    cmb_map: np.ndarray,
    y_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    nside: Optional[int] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Regress out y-map template from CMB map.

    Performs linear regression: T_cmb = α × y + residual

    Parameters
    ----------
    cmb_map : np.ndarray
        CMB temperature map (HEALPix)
    y_map : np.ndarray
        Compton-y map (HEALPix, same nside)
    mask : np.ndarray, optional
        Binary mask (1 = valid)
    nside : int, optional
        Override nside (for downgrading)

    Returns
    -------
    cleaned_map : np.ndarray
        CMB map with y-correlated component removed
    alpha : float
        Regression coefficient (μK per unit y)
    alpha_err : float
        Uncertainty on alpha
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required for y-map regression")

    # Ensure same resolution
    if len(cmb_map) != len(y_map):
        raise ValueError(f"Map sizes don't match: {len(cmb_map)} vs {len(y_map)}")

    # Apply mask
    if mask is not None:
        valid = mask > 0.5
    else:
        valid = np.isfinite(cmb_map) & np.isfinite(y_map)

    T = cmb_map[valid]
    y = y_map[valid]

    # Remove means
    T_mean = np.mean(T)
    y_mean = np.mean(y)
    T_centered = T - T_mean
    y_centered = y - y_mean

    # Linear regression: T = α × y + const
    # α = Σ(T × y) / Σ(y²)
    numerator = np.sum(T_centered * y_centered)
    denominator = np.sum(y_centered**2)

    alpha = numerator / denominator if denominator > 0 else 0.0

    # Uncertainty from residual variance
    residual = T_centered - alpha * y_centered
    var_residual = np.var(residual)
    alpha_err = np.sqrt(var_residual / denominator) if denominator > 0 else np.inf

    # Create cleaned map
    cleaned_map = cmb_map.copy()
    cleaned_map[valid] = cmb_map[valid] - alpha * y_map[valid]

    logger.info(f"Y-map regression: α = {alpha:.2f} ± {alpha_err:.2f} μK/y")

    return cleaned_map, alpha, alpha_err


def ymap_regression_test(
    estimator,
    positions: np.ndarray,
    temperatures_original: np.ndarray,
    temperatures_cleaned: np.ndarray,
    weights: np.ndarray,
    template: np.ndarray,
    cov: np.ndarray,
    alpha: float,
    alpha_err: float,
    shift_threshold: float = 2.0,
) -> YMapRegressionResult:
    """
    Test effect of y-map template regression on kSZ amplitude.

    Compares amplitude before and after removing y-correlated signal.
    If shift > threshold × σ, tSZ contamination may be significant.

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
        The estimator
    positions : np.ndarray
        Galaxy positions
    temperatures_original : np.ndarray
        Temperatures before cleaning
    temperatures_cleaned : np.ndarray
        Temperatures after y-regression
    weights : np.ndarray
        Galaxy weights
    template : np.ndarray
        Theory template
    cov : np.ndarray
        Covariance matrix
    alpha, alpha_err : float
        Regression coefficient and error
    shift_threshold : float
        Maximum allowed shift in σ

    Returns
    -------
    YMapRegressionResult
        Comparison of original vs cleaned amplitudes
    """
    # Compute with original temperatures
    result_orig = estimator.compute(positions, temperatures_original, weights)

    # Compute with cleaned temperatures
    result_clean = estimator.compute(positions, temperatures_cleaned, weights)

    # Fit amplitudes
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.diag(1.0 / np.diag(cov))

    def fit_amp(p_ksz):
        num = template @ cov_inv @ p_ksz
        den = template @ cov_inv @ template
        return num / den if den > 0 else 0, 1/np.sqrt(den) if den > 0 else np.inf

    A_orig, sigma_orig = fit_amp(result_orig.p_ksz)
    A_clean, sigma_clean = fit_amp(result_clean.p_ksz)

    # Amplitude shift
    shift = A_clean - A_orig
    sigma_shift = np.sqrt(sigma_orig**2 + sigma_clean**2)
    shift_sigma = abs(shift) / sigma_shift if sigma_shift > 0 else 0

    # Residual correlation (cleaned T with y-correlated part)
    y_contribution = temperatures_original - temperatures_cleaned
    residual_corr = np.corrcoef(temperatures_cleaned, y_contribution)[0, 1]

    passed = shift_sigma < shift_threshold

    return YMapRegressionResult(
        regression_coefficient=alpha,
        regression_coefficient_err=alpha_err,
        residual_correlation=float(residual_corr) if np.isfinite(residual_corr) else 0,
        original_amplitude=float(A_orig),
        cleaned_amplitude=float(A_clean),
        amplitude_shift=float(shift),
        amplitude_shift_sigma=float(shift_sigma),
        passed=passed,
        details={
            'shift_threshold': shift_threshold,
            'p_ksz_original': result_orig.p_ksz.tolist(),
            'p_ksz_cleaned': result_clean.p_ksz.tolist(),
        }
    )


def plot_cluster_mask_sweep(
    result: ClusterMaskSweepResult,
    output_path: Optional[str] = None,
):
    """Generate cluster mask sweep diagnostic plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Amplitude vs mask radius
    radii = result.mask_radii_arcmin
    amps = result.amplitudes
    errs = result.amplitude_errors

    valid = [i for i in range(len(amps)) if np.isfinite(amps[i])]

    ax1.errorbar(
        [radii[i] for i in valid],
        [amps[i] for i in valid],
        yerr=[errs[i] for i in valid],
        fmt='o-', capsize=3, color='C0'
    )

    if result.convergence_radius is not None:
        ax1.axvline(result.convergence_radius, color='green', linestyle='--',
                   label=f'Converged at {result.convergence_radius}\'')

    ax1.set_xlabel('Mask radius (arcmin)')
    ax1.set_ylabel(r'$A_{kSZ}$')
    ax1.set_title('Amplitude vs Cluster Mask Radius')
    ax1.legend()

    # Right: p(r) for different radii
    for radius, p_ksz in result.p_ksz_by_radius.items():
        ax2.plot(range(len(p_ksz)), p_ksz, label=f'{radius}\'')

    ax2.set_xlabel('Separation bin')
    ax2.set_ylabel(r'$\hat{p}(r)$ [$\mu$K]')
    ax2.set_title('Pairwise Momentum vs Mask Radius')
    ax2.legend()
    ax2.axhline(0, color='gray', linestyle=':')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
