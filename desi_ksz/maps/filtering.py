"""
Optimal filtering for kSZ signal extraction.

Implements matched filter and Wiener filter for CMB maps.
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


def apply_matched_filter(
    map_data: np.ndarray,
    signal_profile_cl: np.ndarray,
    noise_cl: np.ndarray,
    beam_bl: Optional[np.ndarray] = None,
    lmax: Optional[int] = None,
) -> np.ndarray:
    """
    Apply matched filter to CMB map.

    The matched filter maximizes S/N for a known signal profile:
        F(ℓ) = B(ℓ) × S(ℓ) / [C_tot(ℓ)]

    where B is the beam, S is the signal power, and C_tot is total power.

    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix map in muK
    signal_profile_cl : np.ndarray
        Expected signal power spectrum
    noise_cl : np.ndarray
        Total noise power spectrum (CMB + instrumental + foregrounds)
    beam_bl : np.ndarray, optional
        Beam transfer function
    lmax : int, optional
        Maximum multipole

    Returns
    -------
    np.ndarray
        Filtered map
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    nside = hp.npix2nside(len(map_data))
    if lmax is None:
        lmax = min(3 * nside - 1, len(noise_cl) - 1)

    # Get alm
    alm = hp.map2alm(map_data, lmax=lmax)

    # Compute filter
    ell = np.arange(lmax + 1)

    if beam_bl is None:
        beam_bl = np.ones(lmax + 1)
    else:
        beam_bl = beam_bl[:lmax + 1]

    # Extend signal and noise if needed
    signal_cl = np.zeros(lmax + 1)
    signal_cl[:min(len(signal_profile_cl), lmax + 1)] = signal_profile_cl[:min(len(signal_profile_cl), lmax + 1)]

    noise = np.zeros(lmax + 1)
    noise[:min(len(noise_cl), lmax + 1)] = noise_cl[:min(len(noise_cl), lmax + 1)]

    # Avoid division by zero
    noise = np.maximum(noise, 1e-20)

    # Matched filter
    filter_ell = beam_bl * signal_cl / noise

    # Normalize
    norm = np.sum((2 * ell + 1) * filter_ell * signal_cl)
    if norm > 0:
        filter_ell /= np.sqrt(norm)

    # Apply filter to alm
    alm_filtered = hp.almxfl(alm, filter_ell)

    # Transform back
    return hp.alm2map(alm_filtered, nside, lmax=lmax)


def apply_wiener_filter(
    map_data: np.ndarray,
    signal_cl: np.ndarray,
    noise_cl: np.ndarray,
    lmax: Optional[int] = None,
) -> np.ndarray:
    """
    Apply Wiener filter to CMB map.

    The Wiener filter minimizes mean squared error:
        W(ℓ) = C_signal(ℓ) / [C_signal(ℓ) + C_noise(ℓ)]

    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix map
    signal_cl : np.ndarray
        Signal power spectrum
    noise_cl : np.ndarray
        Noise power spectrum
    lmax : int, optional
        Maximum multipole

    Returns
    -------
    np.ndarray
        Wiener-filtered map
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    nside = hp.npix2nside(len(map_data))
    if lmax is None:
        lmax = min(3 * nside - 1, len(noise_cl) - 1)

    alm = hp.map2alm(map_data, lmax=lmax)

    # Wiener filter
    signal = np.zeros(lmax + 1)
    signal[:min(len(signal_cl), lmax + 1)] = signal_cl[:min(len(signal_cl), lmax + 1)]

    noise = np.zeros(lmax + 1)
    noise[:min(len(noise_cl), lmax + 1)] = noise_cl[:min(len(noise_cl), lmax + 1)]

    total = signal + noise
    total = np.maximum(total, 1e-20)

    wiener_ell = signal / total

    alm_filtered = hp.almxfl(alm, wiener_ell)

    return hp.alm2map(alm_filtered, nside, lmax=lmax)


def apply_highpass_filter(
    map_data: np.ndarray,
    ell_min: int = 100,
    lmax: Optional[int] = None,
) -> np.ndarray:
    """
    Apply high-pass filter to remove large-scale modes.

    Parameters
    ----------
    map_data : np.ndarray
        Input map
    ell_min : int
        Minimum multipole to keep
    lmax : int, optional
        Maximum multipole

    Returns
    -------
    np.ndarray
        High-pass filtered map
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    nside = hp.npix2nside(len(map_data))
    if lmax is None:
        lmax = 3 * nside - 1

    alm = hp.map2alm(map_data, lmax=lmax)

    # High-pass filter: smooth transition
    ell = np.arange(lmax + 1)
    filter_ell = np.zeros(lmax + 1)
    filter_ell[ell >= ell_min] = 1.0

    # Smooth transition
    transition_width = 20
    transition = np.where(
        ell < ell_min,
        np.exp(-0.5 * ((ell - ell_min) / transition_width)**2),
        1.0
    )
    filter_ell = transition

    alm_filtered = hp.almxfl(alm, filter_ell)

    return hp.alm2map(alm_filtered, nside, lmax=lmax)


def compute_filter_transfer_function(
    filter_type: str,
    lmax: int,
    signal_cl: Optional[np.ndarray] = None,
    noise_cl: Optional[np.ndarray] = None,
    beam_fwhm_arcmin: Optional[float] = None,
) -> np.ndarray:
    """
    Compute transfer function for a given filter.

    Parameters
    ----------
    filter_type : str
        Filter type: 'matched', 'wiener', 'highpass', 'beam'
    lmax : int
        Maximum multipole
    signal_cl : np.ndarray, optional
        Signal power spectrum
    noise_cl : np.ndarray, optional
        Noise power spectrum
    beam_fwhm_arcmin : float, optional
        Beam FWHM for beam filter

    Returns
    -------
    np.ndarray
        Filter transfer function F(ℓ)
    """
    ell = np.arange(lmax + 1)

    if filter_type == "matched" and signal_cl is not None and noise_cl is not None:
        noise = np.maximum(noise_cl[:lmax + 1], 1e-20)
        return signal_cl[:lmax + 1] / noise

    elif filter_type == "wiener" and signal_cl is not None and noise_cl is not None:
        total = signal_cl[:lmax + 1] + noise_cl[:lmax + 1]
        total = np.maximum(total, 1e-20)
        return signal_cl[:lmax + 1] / total

    elif filter_type == "highpass":
        ell_min = 100
        return np.where(ell >= ell_min, 1.0, 0.0)

    elif filter_type == "beam" and beam_fwhm_arcmin is not None:
        sigma_rad = np.deg2rad(beam_fwhm_arcmin / 60.0) / np.sqrt(8 * np.log(2))
        return np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)

    else:
        raise ValueError(f"Cannot compute transfer function for {filter_type}")


def deconvolve_beam(
    map_data: np.ndarray,
    beam_fwhm_arcmin: float,
    lmax: Optional[int] = None,
    ell_max_deconv: int = 2000,
) -> np.ndarray:
    """
    Deconvolve beam from map (with regularization).

    Parameters
    ----------
    map_data : np.ndarray
        Input map (beam-convolved)
    beam_fwhm_arcmin : float
        Beam FWHM in arcmin
    lmax : int, optional
        Maximum multipole
    ell_max_deconv : int
        Maximum ell for deconvolution (regularization)

    Returns
    -------
    np.ndarray
        Deconvolved map
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")

    nside = hp.npix2nside(len(map_data))
    if lmax is None:
        lmax = 3 * nside - 1

    # Get beam transfer function
    sigma_rad = np.deg2rad(beam_fwhm_arcmin / 60.0) / np.sqrt(8 * np.log(2))
    ell = np.arange(lmax + 1)
    bl = np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)

    # Regularize: don't deconvolve beyond ell_max_deconv
    bl_inv = np.zeros(lmax + 1)
    bl_inv[ell <= ell_max_deconv] = 1.0 / np.maximum(bl[ell <= ell_max_deconv], 0.01)

    # Apply deconvolution
    alm = hp.map2alm(map_data, lmax=lmax)
    alm_deconv = hp.almxfl(alm, bl_inv)

    return hp.alm2map(alm_deconv, nside, lmax=lmax)
