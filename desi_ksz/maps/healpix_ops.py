"""
HEALPix map operations for kSZ analysis.

Provides wrapper functions for common HEALPix operations using healpy.
"""

from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    hp = None
    HEALPY_AVAILABLE = False


def upgrade_downgrade_map(
    map_data: np.ndarray,
    nside_out: int,
    order_in: str = "RING",
    order_out: str = "RING",
) -> np.ndarray:
    """
    Change resolution of HEALPix map.

    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix map
    nside_out : int
        Output nside
    order_in : str
        Input pixel ordering ('RING' or 'NESTED')
    order_out : str
        Output pixel ordering

    Returns
    -------
    np.ndarray
        Resampled map
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required for HEALPix operations")

    return hp.ud_grade(map_data, nside_out, order_in=order_in, order_out=order_out)


def get_alm(
    map_data: np.ndarray,
    lmax: Optional[int] = None,
    mmax: Optional[int] = None,
) -> np.ndarray:
    """
    Compute spherical harmonic coefficients.

    Parameters
    ----------
    map_data : np.ndarray
        HEALPix map
    lmax : int, optional
        Maximum multipole (default: 3*nside - 1)
    mmax : int, optional
        Maximum m value (default: lmax)

    Returns
    -------
    np.ndarray
        Complex alm array
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    nside = hp.npix2nside(len(map_data))
    if lmax is None:
        lmax = 3 * nside - 1

    return hp.map2alm(map_data, lmax=lmax, mmax=mmax)


def alm2map(
    alm: np.ndarray,
    nside: int,
    lmax: Optional[int] = None,
) -> np.ndarray:
    """
    Convert alm to HEALPix map.

    Parameters
    ----------
    alm : np.ndarray
        Spherical harmonic coefficients
    nside : int
        Output nside
    lmax : int, optional
        Maximum multipole

    Returns
    -------
    np.ndarray
        HEALPix map
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    return hp.alm2map(alm, nside, lmax=lmax)


def apply_beam(
    map_data: np.ndarray,
    beam_fwhm_arcmin: float,
    lmax: Optional[int] = None,
) -> np.ndarray:
    """
    Apply Gaussian beam smoothing.

    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix map
    beam_fwhm_arcmin : float
        Beam FWHM in arcminutes
    lmax : int, optional
        Maximum multipole for transform

    Returns
    -------
    np.ndarray
        Beam-smoothed map
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    beam_fwhm_rad = np.deg2rad(beam_fwhm_arcmin / 60.0)
    return hp.smoothing(map_data, fwhm=beam_fwhm_rad, lmax=lmax)


def smooth_map(
    map_data: np.ndarray,
    sigma_arcmin: float,
) -> np.ndarray:
    """
    Smooth map with Gaussian kernel.

    Parameters
    ----------
    map_data : np.ndarray
        Input map
    sigma_arcmin : float
        Smoothing scale in arcmin

    Returns
    -------
    np.ndarray
        Smoothed map
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    sigma_rad = np.deg2rad(sigma_arcmin / 60.0)
    return hp.smoothing(map_data, sigma=sigma_rad)


def compute_map_power_spectrum(
    map_data: np.ndarray,
    lmax: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute angular power spectrum of map.

    Parameters
    ----------
    map_data : np.ndarray
        HEALPix map
    lmax : int, optional
        Maximum multipole
    mask : np.ndarray, optional
        Boolean mask

    Returns
    -------
    ell : np.ndarray
        Multipole values
    cl : np.ndarray
        Power spectrum
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    if mask is not None:
        map_masked = map_data * mask
        fsky = np.mean(mask)
    else:
        map_masked = map_data
        fsky = 1.0

    cl = hp.anafast(map_masked, lmax=lmax)
    cl /= fsky  # Simple f_sky correction

    ell = np.arange(len(cl))
    return ell, cl


def rotate_map(
    map_data: np.ndarray,
    rotation_angle_deg: float,
    axis: str = "z",
) -> np.ndarray:
    """
    Rotate HEALPix map around axis.

    Parameters
    ----------
    map_data : np.ndarray
        Input map
    rotation_angle_deg : float
        Rotation angle in degrees
    axis : str
        Rotation axis ('x', 'y', 'z')

    Returns
    -------
    np.ndarray
        Rotated map
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    nside = hp.npix2nside(len(map_data))
    rot_angle_rad = np.deg2rad(rotation_angle_deg)

    # Define rotation matrix
    if axis == "z":
        rot = hp.Rotator(rot=[rot_angle_rad, 0, 0], eulertype="ZYX")
    elif axis == "y":
        rot = hp.Rotator(rot=[0, rot_angle_rad, 0], eulertype="ZYX")
    elif axis == "x":
        rot = hp.Rotator(rot=[0, 0, rot_angle_rad], eulertype="ZYX")
    else:
        raise ValueError(f"Unknown axis: {axis}")

    # Apply rotation
    return rot.rotate_map_alms(map_data)
