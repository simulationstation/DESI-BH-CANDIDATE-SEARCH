"""
Temperature extraction at galaxy positions.

This module provides high-level functions for extracting CMB temperatures
at galaxy positions with various methods and diagnostics.
"""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemperatureExtractionResult:
    """
    Container for temperature extraction results.

    Attributes
    ----------
    temperatures : np.ndarray
        Extracted temperatures in muK
    valid_mask : np.ndarray
        Boolean mask indicating valid extractions
    method : str
        Extraction method used
    aperture_inner : float
        Inner aperture radius (arcmin)
    aperture_outer : float
        Outer aperture radius (arcmin)
    statistics : dict
        Summary statistics
    """
    temperatures: np.ndarray
    valid_mask: np.ndarray
    method: str
    aperture_inner: float
    aperture_outer: float
    statistics: Dict[str, float] = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        return int(np.sum(self.valid_mask))

    @property
    def n_total(self) -> int:
        return len(self.temperatures)

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_total if self.n_total > 0 else 0.0

    def compute_statistics(self) -> Dict[str, float]:
        """Compute summary statistics."""
        valid_T = self.temperatures[self.valid_mask]

        if len(valid_T) == 0:
            return {
                "n_total": self.n_total,
                "n_valid": 0,
                "valid_fraction": 0.0,
            }

        self.statistics = {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "valid_fraction": self.valid_fraction,
            "mean": float(np.mean(valid_T)),
            "std": float(np.std(valid_T)),
            "median": float(np.median(valid_T)),
            "min": float(np.min(valid_T)),
            "max": float(np.max(valid_T)),
            "percentile_5": float(np.percentile(valid_T, 5)),
            "percentile_95": float(np.percentile(valid_T, 95)),
        }

        return self.statistics


def extract_object_temperatures(
    cmb_map,
    ra: np.ndarray,
    dec: np.ndarray,
    z: Optional[np.ndarray] = None,
    method: str = "compensated",
    aperture_inner: float = 1.8,
    aperture_outer: float = 5.0,
    use_physical_aperture: bool = False,
    physical_radius_mpc: float = 0.5,
    cosmology: Optional[dict] = None,
    batch_size: int = 10000,
    cache_path: Optional[Union[str, Path]] = None,
) -> TemperatureExtractionResult:
    """
    Extract CMB temperatures at galaxy positions.

    This is the main entry point for temperature extraction, supporting
    multiple methods and optional caching.

    Parameters
    ----------
    cmb_map : CMBTemperatureMap
        CMB temperature map object (PlanckMap or ACTMap)
    ra : np.ndarray
        Right Ascension in degrees
    dec : np.ndarray
        Declination in degrees
    z : np.ndarray, optional
        Redshifts (required if use_physical_aperture=True)
    method : str
        Extraction method: 'compensated', 'aperture', 'interpolate', 'matched'
    aperture_inner : float
        Inner aperture radius in arcmin
    aperture_outer : float
        Outer aperture radius in arcmin
    use_physical_aperture : bool
        If True, use fixed physical aperture converted to angular at each z
    physical_radius_mpc : float
        Physical aperture radius in Mpc/h
    cosmology : dict, optional
        Cosmology parameters for physical aperture conversion
    batch_size : int
        Process in batches for memory efficiency
    cache_path : str or Path, optional
        Path to cache temperatures for later use

    Returns
    -------
    TemperatureExtractionResult
        Extracted temperatures with metadata

    Examples
    --------
    >>> result = extract_object_temperatures(
    ...     cmb_map, catalog.ra, catalog.dec,
    ...     method='compensated',
    ...     aperture_inner=2.0,
    ... )
    >>> T = result.temperatures
    """
    n_objects = len(ra)
    logger.info(f"Extracting temperatures for {n_objects} objects using '{method}' method")

    # Initialize output
    temperatures = np.full(n_objects, np.nan)

    # Handle physical aperture conversion
    if use_physical_aperture:
        if z is None:
            raise ValueError("Redshifts required for physical aperture")

        from .aperture_photometry import physical_to_angular_aperture

        # Convert physical to angular at mean redshift for logging
        z_mean = np.mean(z)
        theta_mean = physical_to_angular_aperture(physical_radius_mpc, z_mean, cosmology)
        logger.info(
            f"Physical aperture {physical_radius_mpc} Mpc/h = "
            f"{theta_mean:.2f} arcmin at z={z_mean:.2f}"
        )

    # Process in batches
    n_batches = (n_objects + batch_size - 1) // batch_size

    for b in range(n_batches):
        i_start = b * batch_size
        i_end = min((b + 1) * batch_size, n_objects)

        ra_batch = ra[i_start:i_end]
        dec_batch = dec[i_start:i_end]

        if use_physical_aperture:
            z_batch = z[i_start:i_end]
            # Extract with varying aperture per object
            T_batch = _extract_with_varying_aperture(
                cmb_map, ra_batch, dec_batch, z_batch,
                physical_radius_mpc, method, cosmology,
            )
        else:
            # Fixed angular aperture
            if method == "compensated":
                T_batch = cmb_map.get_temperature_at_positions(
                    ra_batch, dec_batch,
                    aperture_radius_arcmin=aperture_inner,
                    method="compensated",
                    outer_radius_arcmin=aperture_outer,
                )
            elif method == "aperture":
                T_batch = cmb_map.get_temperature_at_positions(
                    ra_batch, dec_batch,
                    aperture_radius_arcmin=aperture_inner,
                    method="aperture",
                )
            elif method == "interpolate":
                T_batch = cmb_map.get_pixel_values(ra_batch, dec_batch)
            else:
                raise ValueError(f"Unknown method: {method}")

        temperatures[i_start:i_end] = T_batch

        if (b + 1) % 10 == 0 or b == n_batches - 1:
            logger.debug(f"Processed batch {b + 1}/{n_batches}")

    # Create result
    valid_mask = np.isfinite(temperatures)

    result = TemperatureExtractionResult(
        temperatures=temperatures,
        valid_mask=valid_mask,
        method=method,
        aperture_inner=aperture_inner,
        aperture_outer=aperture_outer,
    )
    result.compute_statistics()

    logger.info(
        f"Extracted {result.n_valid}/{result.n_total} valid temperatures "
        f"({100*result.valid_fraction:.1f}%)"
    )

    # Cache if requested
    if cache_path is not None:
        _save_temperatures_to_cache(result, cache_path)

    return result


def _extract_with_varying_aperture(
    cmb_map,
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    physical_radius_mpc: float,
    method: str,
    cosmology: Optional[dict],
) -> np.ndarray:
    """
    Extract temperatures with aperture varying by redshift.

    This accounts for the fact that a fixed physical aperture corresponds
    to different angular sizes at different redshifts.
    """
    from .aperture_photometry import physical_to_angular_aperture

    n_objects = len(ra)
    temperatures = np.full(n_objects, np.nan)

    # Group by similar redshifts for efficiency
    z_unique = np.unique(np.round(z, 2))

    for zi in z_unique:
        # Find objects at this redshift
        mask = np.abs(z - zi) < 0.01

        if not np.any(mask):
            continue

        # Convert physical to angular at this redshift
        theta_inner = physical_to_angular_aperture(physical_radius_mpc, zi, cosmology)
        theta_outer = 2.5 * theta_inner

        # Extract
        if method == "compensated":
            T = cmb_map.get_temperature_at_positions(
                ra[mask], dec[mask],
                aperture_radius_arcmin=theta_inner,
                method="compensated",
                outer_radius_arcmin=theta_outer,
            )
        elif method == "aperture":
            T = cmb_map.get_temperature_at_positions(
                ra[mask], dec[mask],
                aperture_radius_arcmin=theta_inner,
                method="aperture",
            )
        else:
            T = cmb_map.get_pixel_values(ra[mask], dec[mask])

        temperatures[mask] = T

    return temperatures


def _save_temperatures_to_cache(
    result: TemperatureExtractionResult,
    cache_path: Union[str, Path],
) -> None:
    """Save temperature extraction to cache file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "temperatures": result.temperatures,
        "valid_mask": result.valid_mask,
    }

    try:
        import h5py
        with h5py.File(cache_path, "w") as f:
            for key, val in data.items():
                f.create_dataset(key, data=val, compression="gzip")
            f.attrs["method"] = result.method
            f.attrs["aperture_inner"] = result.aperture_inner
            f.attrs["aperture_outer"] = result.aperture_outer
    except ImportError:
        np.savez_compressed(cache_path, **data)

    logger.info(f"Cached temperatures to {cache_path}")


def load_temperatures_from_cache(
    cache_path: Union[str, Path],
) -> TemperatureExtractionResult:
    """Load temperature extraction from cache file."""
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    try:
        import h5py
        with h5py.File(cache_path, "r") as f:
            temperatures = f["temperatures"][:]
            valid_mask = f["valid_mask"][:]
            method = f.attrs.get("method", "unknown")
            aperture_inner = f.attrs.get("aperture_inner", 0.0)
            aperture_outer = f.attrs.get("aperture_outer", 0.0)
    except ImportError:
        data = np.load(cache_path)
        temperatures = data["temperatures"]
        valid_mask = data["valid_mask"]
        method = "unknown"
        aperture_inner = 0.0
        aperture_outer = 0.0

    result = TemperatureExtractionResult(
        temperatures=temperatures,
        valid_mask=valid_mask,
        method=method,
        aperture_inner=aperture_inner,
        aperture_outer=aperture_outer,
    )
    result.compute_statistics()

    logger.info(f"Loaded {result.n_valid} temperatures from cache")
    return result
