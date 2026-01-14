"""
Aperture photometry for kSZ temperature extraction.

This module implements aperture photometry methods for extracting
CMB temperatures at galaxy positions.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| T_AP        | Aperture photometry temperature                 | muK         |
| T_inner     | Mean temperature in inner aperture              | muK         |
| T_annulus   | Mean temperature in annulus                     | muK         |
| theta_in    | Inner aperture radius                           | arcmin      |
| theta_out   | Outer aperture radius                           | arcmin      |
| Omega       | Solid angle                                     | sr          |

Compensated Aperture Filter
---------------------------
The compensated aperture filter subtracts the local background:

    T_AP = T_inner - T_annulus

where:
    T_inner = (1/Omega_in) ∫_{θ<θ_in} T(θ) dΩ
    T_annulus = (1/Omega_ann) ∫_{θ_in<θ<θ_out} T(θ) dΩ

This removes large-scale modes (primary CMB) and is less sensitive
to calibration errors.

References
----------
- Schaan, E., et al. 2016, PRD, 93, 082002
- Hand, N., et al. 2012, PRL, 109, 041101
"""

from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ApertureConfig:
    """Configuration for aperture photometry."""
    inner_radius_arcmin: float = 1.8
    outer_radius_arcmin: float = 5.0
    filter_type: str = "compensated"  # 'compensated', 'tophat', 'gaussian'
    use_variance_weighting: bool = False


class AperturePhotometryStacker:
    """
    Aperture photometry stacker for kSZ analysis.

    This class extracts CMB temperatures at galaxy positions using
    various aperture filters and can stack them in bins of galaxy
    properties (mass proxy, redshift, etc.).

    Parameters
    ----------
    inner_radius_arcmin : float
        Inner aperture radius in arcmin
    outer_radius_arcmin : float
        Outer aperture radius in arcmin (for compensated filter)
    filter_type : str
        Filter type: 'compensated', 'tophat', 'gaussian'

    Examples
    --------
    >>> stacker = AperturePhotometryStacker(inner_radius_arcmin=2.0)
    >>> T = stacker.extract_temperatures(cmb_map, ra, dec)
    >>> stacked_T = stacker.stack_by_mass(T, mass_proxy, n_bins=5)
    """

    def __init__(
        self,
        inner_radius_arcmin: float = 1.8,
        outer_radius_arcmin: float = 5.0,
        filter_type: str = "compensated",
    ):
        self.inner_radius_arcmin = inner_radius_arcmin
        self.outer_radius_arcmin = outer_radius_arcmin
        self.filter_type = filter_type

        if outer_radius_arcmin <= inner_radius_arcmin and filter_type == "compensated":
            raise ValueError(
                f"outer_radius ({outer_radius_arcmin}) must be > "
                f"inner_radius ({inner_radius_arcmin}) for compensated filter"
            )

    def extract_temperatures(
        self,
        cmb_map,
        ra: np.ndarray,
        dec: np.ndarray,
        ivar_map=None,
    ) -> np.ndarray:
        """
        Extract aperture temperatures at galaxy positions.

        Parameters
        ----------
        cmb_map : CMBTemperatureMap
            CMB temperature map object
        ra : np.ndarray
            Right Ascension in degrees
        dec : np.ndarray
            Declination in degrees
        ivar_map : optional
            Inverse variance map for weighted extraction

        Returns
        -------
        np.ndarray
            Aperture photometry temperatures in muK
        """
        n_galaxies = len(ra)
        logger.info(f"Extracting temperatures for {n_galaxies} galaxies")

        # Use CMB map's built-in extraction
        if self.filter_type == "compensated":
            temperatures = cmb_map.get_temperature_at_positions(
                ra, dec,
                aperture_radius_arcmin=self.inner_radius_arcmin,
                method="compensated",
                outer_radius_arcmin=self.outer_radius_arcmin,
            )
        elif self.filter_type == "tophat":
            temperatures = cmb_map.get_temperature_at_positions(
                ra, dec,
                aperture_radius_arcmin=self.inner_radius_arcmin,
                method="aperture",
            )
        elif self.filter_type == "interpolate":
            temperatures = cmb_map.get_pixel_values(ra, dec)
        else:
            raise ValueError(f"Unknown filter_type: {self.filter_type}")

        # Log statistics
        valid = np.isfinite(temperatures)
        n_valid = np.sum(valid)
        if n_valid < n_galaxies:
            logger.warning(f"Only {n_valid}/{n_galaxies} valid temperature extractions")

        if n_valid > 0:
            logger.info(
                f"Temperature statistics: "
                f"mean={np.nanmean(temperatures):.2f} muK, "
                f"std={np.nanstd(temperatures):.2f} muK"
            )

        return temperatures

    def stack_by_property(
        self,
        temperatures: np.ndarray,
        property_values: np.ndarray,
        n_bins: int = 5,
        property_bins: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stack temperatures in bins of galaxy property.

        Parameters
        ----------
        temperatures : np.ndarray
            Aperture temperatures
        property_values : np.ndarray
            Galaxy property values (e.g., stellar mass, luminosity)
        n_bins : int
            Number of bins (if property_bins not provided)
        property_bins : np.ndarray, optional
            Bin edges for property
        weights : np.ndarray, optional
            Galaxy weights

        Returns
        -------
        bin_centers : np.ndarray
            Property bin centers
        stacked_T : np.ndarray
            Mean stacked temperature per bin
        stacked_T_err : np.ndarray
            Error on stacked temperature (from bootstrap or scatter)
        counts : np.ndarray
            Number of galaxies per bin
        """
        valid = np.isfinite(temperatures) & np.isfinite(property_values)

        T = temperatures[valid]
        prop = property_values[valid]
        w = weights[valid] if weights is not None else np.ones(len(T))

        if property_bins is None:
            property_bins = np.percentile(prop, np.linspace(0, 100, n_bins + 1))

        n_bins = len(property_bins) - 1
        bin_centers = np.zeros(n_bins)
        stacked_T = np.zeros(n_bins)
        stacked_T_err = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = (prop >= property_bins[i]) & (prop < property_bins[i + 1])
            counts[i] = np.sum(mask)

            if counts[i] == 0:
                bin_centers[i] = 0.5 * (property_bins[i] + property_bins[i + 1])
                stacked_T[i] = np.nan
                stacked_T_err[i] = np.nan
                continue

            bin_centers[i] = np.average(prop[mask], weights=w[mask])
            stacked_T[i] = np.average(T[mask], weights=w[mask])

            # Error from weighted standard error of the mean
            var = np.average((T[mask] - stacked_T[i])**2, weights=w[mask])
            n_eff = np.sum(w[mask])**2 / np.sum(w[mask]**2)
            stacked_T_err[i] = np.sqrt(var / n_eff)

        return bin_centers, stacked_T, stacked_T_err, counts

    def compute_detection_significance(
        self,
        temperatures: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float, float]:
        """
        Compute detection significance of mean temperature signal.

        Parameters
        ----------
        temperatures : np.ndarray
            Aperture temperatures
        weights : np.ndarray, optional
            Galaxy weights
        n_bootstrap : int
            Number of bootstrap samples

        Returns
        -------
        mean_T : float
            Mean temperature
        T_err : float
            Bootstrap error on mean
        snr : float
            Signal-to-noise ratio
        """
        valid = np.isfinite(temperatures)
        T = temperatures[valid]
        w = weights[valid] if weights is not None else np.ones(len(T))

        # Weighted mean
        mean_T = np.average(T, weights=w)

        # Bootstrap error
        rng = np.random.default_rng(42)
        bootstrap_means = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            idx = rng.choice(len(T), size=len(T), replace=True)
            bootstrap_means[i] = np.average(T[idx], weights=w[idx])

        T_err = np.std(bootstrap_means)
        snr = np.abs(mean_T) / T_err if T_err > 0 else 0.0

        return mean_T, T_err, snr


def extract_temperatures_aperture(
    cmb_map,
    ra: np.ndarray,
    dec: np.ndarray,
    inner_radius_arcmin: float = 1.8,
    outer_radius_arcmin: float = 5.0,
    filter_type: str = "compensated",
) -> np.ndarray:
    """
    Convenience function to extract aperture temperatures.

    Parameters
    ----------
    cmb_map : CMBTemperatureMap
        CMB temperature map
    ra, dec : np.ndarray
        Galaxy coordinates in degrees
    inner_radius_arcmin : float
        Inner aperture radius
    outer_radius_arcmin : float
        Outer aperture radius
    filter_type : str
        Filter type

    Returns
    -------
    np.ndarray
        Temperatures in muK
    """
    stacker = AperturePhotometryStacker(
        inner_radius_arcmin=inner_radius_arcmin,
        outer_radius_arcmin=outer_radius_arcmin,
        filter_type=filter_type,
    )
    return stacker.extract_temperatures(cmb_map, ra, dec)


def compensated_aperture_filter(
    T_map: np.ndarray,
    theta_inner: float,
    theta_outer: float,
) -> callable:
    """
    Create a compensated aperture filter function.

    Parameters
    ----------
    T_map : np.ndarray
        Temperature map data
    theta_inner : float
        Inner radius in radians
    theta_outer : float
        Outer radius in radians

    Returns
    -------
    callable
        Filter function that takes (theta, phi) position and returns T_AP
    """
    def filter_at_position(theta_c, phi_c):
        """Apply compensated aperture at position (theta_c, phi_c)."""
        # This would require map-specific implementation
        # Placeholder for now
        raise NotImplementedError("Compensated filter requires map-specific implementation")

    return filter_at_position


def compute_optimal_aperture(
    cmb_map,
    ra: np.ndarray,
    dec: np.ndarray,
    aperture_grid: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    signal_proxy: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Find optimal aperture size via signal-to-noise maximization.

    Parameters
    ----------
    cmb_map : CMBTemperatureMap
        CMB temperature map
    ra, dec : np.ndarray
        Galaxy coordinates
    aperture_grid : list
        Aperture sizes to test (arcmin)
    signal_proxy : np.ndarray, optional
        Expected signal sign/amplitude for S/N calculation

    Returns
    -------
    optimal_aperture : float
        Optimal aperture size in arcmin
    snr_values : np.ndarray
        S/N at each aperture size
    """
    snr_values = np.zeros(len(aperture_grid))

    for i, aperture in enumerate(aperture_grid):
        stacker = AperturePhotometryStacker(
            inner_radius_arcmin=aperture,
            outer_radius_arcmin=2.5 * aperture,
        )
        T = stacker.extract_temperatures(cmb_map, ra, dec)
        _, _, snr = stacker.compute_detection_significance(T)
        snr_values[i] = snr

    optimal_idx = np.argmax(snr_values)
    optimal_aperture = aperture_grid[optimal_idx]

    logger.info(f"Optimal aperture: {optimal_aperture} arcmin (S/N = {snr_values[optimal_idx]:.2f})")

    return optimal_aperture, snr_values


def physical_to_angular_aperture(
    physical_radius_mpc: float,
    z: float,
    cosmology: Optional[dict] = None,
) -> float:
    """
    Convert physical aperture (Mpc/h) to angular aperture (arcmin).

    Parameters
    ----------
    physical_radius_mpc : float
        Physical radius in Mpc/h
    z : float
        Redshift
    cosmology : dict, optional
        Cosmology parameters

    Returns
    -------
    float
        Angular radius in arcmin
    """
    if cosmology is None:
        cosmology = {"h": 0.6736, "Omega_m": 0.3153}

    # Angular diameter distance
    try:
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u

        h = cosmology["h"]
        cosmo = FlatLambdaCDM(H0=100 * h, Om0=cosmology["Omega_m"])
        d_A = cosmo.angular_diameter_distance(z).to(u.Mpc).value * h  # Mpc/h

    except ImportError:
        # Simple approximation
        from ..config.defaults import C_LIGHT_KM_S
        h = cosmology.get("h", 0.7)
        Omega_m = cosmology.get("Omega_m", 0.3)

        # Comoving distance
        chi = _compute_chi_simple(z, h, Omega_m)
        d_A = chi / (1 + z)

    # Angular size: theta = r / d_A [radians]
    theta_rad = physical_radius_mpc / d_A

    # Convert to arcmin
    theta_arcmin = np.rad2deg(theta_rad) * 60.0

    return theta_arcmin


def _compute_chi_simple(z: float, h: float, Omega_m: float) -> float:
    """Simple comoving distance computation."""
    from scipy.integrate import quad

    c_over_H0 = 2997.92458 / h  # Mpc/h

    def integrand(zp):
        Omega_Lambda = 1.0 - Omega_m
        Ez = np.sqrt(Omega_m * (1 + zp)**3 + Omega_Lambda)
        return 1.0 / Ez

    chi, _ = quad(integrand, 0, z)
    return chi * c_over_H0
