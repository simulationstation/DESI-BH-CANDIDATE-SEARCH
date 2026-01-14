"""
Masking utilities for CMB maps.

Provides functions for creating and manipulating masks for
point sources, Galactic emission, and clusters.
"""

from typing import Optional, List, Union, Tuple
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    hp = None
    HEALPY_AVAILABLE = False


def create_point_source_mask(
    ra_sources: np.ndarray,
    dec_sources: np.ndarray,
    nside: int,
    mask_radius_arcmin: float = 5.0,
) -> np.ndarray:
    """
    Create point source mask.

    Parameters
    ----------
    ra_sources, dec_sources : np.ndarray
        Source coordinates in degrees
    nside : int
        HEALPix nside
    mask_radius_arcmin : float
        Mask radius around each source

    Returns
    -------
    np.ndarray
        Boolean mask (True = unmasked/good pixels)
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=bool)

    radius_rad = np.deg2rad(mask_radius_arcmin / 60.0)

    for ra, dec in zip(ra_sources, dec_sources):
        theta = np.deg2rad(90.0 - dec)
        phi = np.deg2rad(ra)
        vec = hp.ang2vec(theta, phi)

        # Find pixels within radius
        pix_masked = hp.query_disc(nside, vec, radius_rad)
        mask[pix_masked] = False

    n_masked = np.sum(~mask)
    fsky = np.mean(mask)
    logger.info(f"Point source mask: {n_masked} pixels masked, f_sky = {fsky:.3f}")

    return mask


def create_galactic_mask(
    nside: int,
    b_cut_deg: float = 20.0,
) -> np.ndarray:
    """
    Create Galactic plane mask.

    Parameters
    ----------
    nside : int
        HEALPix nside
    b_cut_deg : float
        Galactic latitude cut in degrees

    Returns
    -------
    np.ndarray
        Boolean mask (True = unmasked)
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    npix = hp.nside2npix(nside)

    # Get Galactic coordinates for each pixel
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Convert to Galactic coordinates
    # Note: healpy default is Galactic coordinates for most CMB maps
    # If map is in equatorial, need coordinate rotation

    # Simple approach: assume input is in Galactic
    b_rad = np.pi / 2 - theta  # Galactic latitude in radians
    b_deg = np.rad2deg(b_rad)

    mask = np.abs(b_deg) > b_cut_deg

    fsky = np.mean(mask)
    logger.info(f"Galactic mask (|b| > {b_cut_deg}°): f_sky = {fsky:.3f}")

    return mask


def create_cluster_mask(
    ra_clusters: np.ndarray,
    dec_clusters: np.ndarray,
    nside: int,
    r500_arcmin: Optional[np.ndarray] = None,
    mask_factor: float = 3.0,
    default_radius_arcmin: float = 5.0,
) -> np.ndarray:
    """
    Create mask around galaxy clusters to mitigate tSZ contamination.

    Parameters
    ----------
    ra_clusters, dec_clusters : np.ndarray
        Cluster coordinates in degrees
    nside : int
        HEALPix nside
    r500_arcmin : np.ndarray, optional
        R500 angular size for each cluster
    mask_factor : float
        Multiply R500 by this factor for mask radius
    default_radius_arcmin : float
        Default radius if R500 not provided

    Returns
    -------
    np.ndarray
        Boolean mask (True = unmasked)
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=bool)

    if r500_arcmin is None:
        r500_arcmin = np.full(len(ra_clusters), default_radius_arcmin)

    for i, (ra, dec) in enumerate(zip(ra_clusters, dec_clusters)):
        theta = np.deg2rad(90.0 - dec)
        phi = np.deg2rad(ra)
        vec = hp.ang2vec(theta, phi)

        radius_rad = np.deg2rad(mask_factor * r500_arcmin[i] / 60.0)
        pix_masked = hp.query_disc(nside, vec, radius_rad)
        mask[pix_masked] = False

    n_masked = np.sum(~mask)
    fsky = np.mean(mask)
    logger.info(f"Cluster mask: {len(ra_clusters)} clusters, {n_masked} pixels masked, f_sky = {fsky:.3f}")

    return mask


def combine_masks(
    masks: List[np.ndarray],
    operation: str = "and",
) -> np.ndarray:
    """
    Combine multiple masks.

    Parameters
    ----------
    masks : list of np.ndarray
        List of boolean masks
    operation : str
        Combination operation: 'and', 'or'

    Returns
    -------
    np.ndarray
        Combined mask
    """
    if len(masks) == 0:
        raise ValueError("No masks provided")

    if len(masks) == 1:
        return masks[0]

    combined = masks[0].copy()

    for m in masks[1:]:
        if operation == "and":
            combined = combined & m
        elif operation == "or":
            combined = combined | m
        else:
            raise ValueError(f"Unknown operation: {operation}")

    fsky = np.mean(combined)
    logger.info(f"Combined mask: f_sky = {fsky:.3f}")

    return combined


def apodize_mask(
    mask: np.ndarray,
    apodization_deg: float = 2.0,
    nside: Optional[int] = None,
) -> np.ndarray:
    """
    Apodize mask edges (smooth transition).

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask
    apodization_deg : float
        Apodization scale in degrees
    nside : int, optional
        HEALPix nside (inferred if not provided)

    Returns
    -------
    np.ndarray
        Apodized mask (values between 0 and 1)
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    if nside is None:
        nside = hp.npix2nside(len(mask))

    # Convert binary mask to float
    mask_float = mask.astype(float)

    # Smooth the mask
    sigma_rad = np.deg2rad(apodization_deg)
    apodized = hp.smoothing(mask_float, sigma=sigma_rad)

    # Ensure values are in [0, 1]
    apodized = np.clip(apodized, 0, 1)

    return apodized


def compute_mask_statistics(mask: np.ndarray) -> dict:
    """
    Compute statistics for a mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or float mask

    Returns
    -------
    dict
        Mask statistics
    """
    if mask.dtype == bool:
        fsky = np.mean(mask)
        n_good = np.sum(mask)
        n_masked = np.sum(~mask)
    else:
        fsky = np.mean(mask)
        n_good = np.sum(mask > 0.5)
        n_masked = np.sum(mask <= 0.5)

    return {
        "f_sky": float(fsky),
        "n_pixels_good": int(n_good),
        "n_pixels_masked": int(n_masked),
        "n_pixels_total": len(mask),
    }


def load_mask_from_file(
    mask_path: Union[str, Path],
    nside: Optional[int] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Load mask from FITS file.

    Parameters
    ----------
    mask_path : str or Path
        Path to mask file
    nside : int, optional
        Target nside (resample if different)
    threshold : float
        Threshold for binary mask

    Returns
    -------
    np.ndarray
        Boolean mask
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    mask_data = hp.read_map(mask_path, dtype=np.float64)

    # Resample if needed
    if nside is not None:
        actual_nside = hp.npix2nside(len(mask_data))
        if actual_nside != nside:
            mask_data = hp.ud_grade(mask_data, nside)

    # Convert to boolean
    mask = mask_data > threshold

    return mask
