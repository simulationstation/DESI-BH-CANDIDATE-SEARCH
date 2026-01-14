"""
Quality cuts for DESI galaxy catalogs.

This module implements standard quality cuts for DESI LSS catalogs,
including redshift quality, stellar contamination removal, and
fiber collision corrections.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| ZWARN       | Redshift warning flag (0 = good)                | bitmask     |
| DELTACHI2   | Delta chi^2 between best and second-best z     | dimensionless|
| COADD_FIBERSTATUS | Fiber status flag                        | bitmask     |

References
----------
- DESI Collaboration 2024, AJ, 168, 58 (DESI DR1 LSS paper)
"""

from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_quality_cuts(
    z: np.ndarray,
    zwarn: Optional[np.ndarray] = None,
    deltachi2: Optional[np.ndarray] = None,
    fiberstatus: Optional[np.ndarray] = None,
    z_min: float = 0.0,
    z_max: float = 3.0,
    deltachi2_min: float = 25.0,
    tracer: str = "LRG",
) -> np.ndarray:
    """
    Apply standard quality cuts for DESI galaxy samples.

    Parameters
    ----------
    z : np.ndarray
        Spectroscopic redshifts
    zwarn : np.ndarray, optional
        Redshift warning flags (ZWARN column)
    deltachi2 : np.ndarray, optional
        Delta chi^2 values for redshift confidence
    fiberstatus : np.ndarray, optional
        Fiber status flags
    z_min : float
        Minimum redshift cut
    z_max : float
        Maximum redshift cut
    deltachi2_min : float
        Minimum delta chi^2 for redshift reliability
    tracer : str
        Galaxy tracer type for tracer-specific cuts

    Returns
    -------
    np.ndarray
        Boolean mask (True = passes all cuts)

    Notes
    -----
    The standard DESI quality cuts include:
    - ZWARN == 0 (no redshift warnings)
    - DELTACHI2 > 25 (confident redshift, tracer-dependent)
    - Valid fiber status
    - Redshift within specified range

    For LRGs, the DELTACHI2 threshold is typically 40.
    For BGS, the threshold can be lower (~15).
    """
    n_galaxies = len(z)
    mask = np.ones(n_galaxies, dtype=bool)

    # Redshift range cut
    mask &= (z >= z_min) & (z < z_max)
    n_pass_z = np.sum(mask)
    logger.debug(f"After z cut [{z_min}, {z_max}): {n_pass_z}/{n_galaxies}")

    # ZWARN cut (good redshifts have ZWARN=0)
    if zwarn is not None:
        mask &= zwarn == 0
        n_pass_zwarn = np.sum(mask)
        logger.debug(f"After ZWARN cut: {n_pass_zwarn}/{n_galaxies}")

    # DELTACHI2 cut (redshift confidence)
    if deltachi2 is not None:
        # Tracer-specific thresholds
        if tracer in ["LRG"]:
            thresh = max(deltachi2_min, 40.0)
        elif tracer in ["ELG_LOP", "ELG"]:
            thresh = deltachi2_min
        elif tracer in ["BGS_BRIGHT", "BGS_FAINT", "BGS"]:
            thresh = max(deltachi2_min, 15.0)
        else:
            thresh = deltachi2_min

        mask &= deltachi2 > thresh
        n_pass_chi2 = np.sum(mask)
        logger.debug(f"After DELTACHI2 > {thresh} cut: {n_pass_chi2}/{n_galaxies}")

    # Fiber status cut
    if fiberstatus is not None:
        # Valid fibers have status 0
        mask &= fiberstatus == 0
        n_pass_fiber = np.sum(mask)
        logger.debug(f"After fiberstatus cut: {n_pass_fiber}/{n_galaxies}")

    n_final = np.sum(mask)
    logger.info(f"Quality cuts: {n_final}/{n_galaxies} galaxies pass ({100*n_final/n_galaxies:.1f}%)")

    return mask


def apply_fiber_collision_correction(
    weights: np.ndarray,
    close_pair_weights: Optional[np.ndarray] = None,
    correction_type: str = "weight",
) -> np.ndarray:
    """
    Apply fiber collision correction to galaxy weights.

    DESI fibers cannot be placed closer than ~75 arcsec, leading to
    under-sampling of close pairs. This is corrected via weighting.

    Parameters
    ----------
    weights : np.ndarray
        Current galaxy weights
    close_pair_weights : np.ndarray, optional
        Pre-computed close-pair correction weights (WEIGHT_CP or similar)
    correction_type : str
        Correction method: 'weight' (upweight), 'remove' (remove affected)

    Returns
    -------
    np.ndarray
        Corrected weights

    Notes
    -----
    The fiber collision correction in DESI is typically pre-computed
    and included in the WEIGHT_COMP column. This function allows for
    additional corrections if needed.
    """
    corrected_weights = weights.copy()

    if close_pair_weights is not None:
        if correction_type == "weight":
            # Apply close-pair weighting
            corrected_weights *= close_pair_weights
        elif correction_type == "remove":
            # Remove galaxies with significant close-pair corrections
            valid = close_pair_weights < 2.0  # Threshold for removal
            corrected_weights[~valid] = 0.0
        else:
            raise ValueError(f"Unknown correction_type: {correction_type}")

    return corrected_weights


def remove_stellar_contamination(
    mask: np.ndarray,
    stellar_prob: Optional[np.ndarray] = None,
    morphology_type: Optional[np.ndarray] = None,
    stellar_prob_max: float = 0.01,
) -> np.ndarray:
    """
    Remove stellar contamination from galaxy sample.

    Parameters
    ----------
    mask : np.ndarray
        Current selection mask
    stellar_prob : np.ndarray, optional
        Probability of object being a star (from photometry)
    morphology_type : np.ndarray, optional
        Morphology classification (PSF vs extended)
    stellar_prob_max : float
        Maximum stellar probability to keep

    Returns
    -------
    np.ndarray
        Updated mask with stellar contamination removed

    Notes
    -----
    DESI spectroscopic targets have already been selected to be
    galaxies, but some stellar contamination may remain. This
    function provides additional cuts if needed.
    """
    updated_mask = mask.copy()

    if stellar_prob is not None:
        stellar_cut = stellar_prob < stellar_prob_max
        updated_mask &= stellar_cut
        n_removed = np.sum(mask & ~stellar_cut)
        logger.info(f"Removed {n_removed} objects with stellar_prob > {stellar_prob_max}")

    if morphology_type is not None:
        # Remove point sources (TYPE = 'PSF' or similar)
        # This depends on the specific column format
        if isinstance(morphology_type[0], str):
            extended = np.array([t.upper() != 'PSF' for t in morphology_type])
        else:
            # Assume PSF = 0 or similar encoding
            extended = morphology_type != 0

        updated_mask &= extended
        n_removed_morph = np.sum(mask & ~extended)
        logger.info(f"Removed {n_removed_morph} point sources by morphology")

    return updated_mask


def flag_bright_star_proximity(
    ra: np.ndarray,
    dec: np.ndarray,
    bright_star_ra: np.ndarray,
    bright_star_dec: np.ndarray,
    exclusion_radius_arcmin: float = 2.0,
) -> np.ndarray:
    """
    Flag galaxies near bright stars.

    Parameters
    ----------
    ra, dec : np.ndarray
        Galaxy coordinates in degrees
    bright_star_ra, bright_star_dec : np.ndarray
        Bright star coordinates in degrees
    exclusion_radius_arcmin : float
        Exclusion radius around bright stars

    Returns
    -------
    np.ndarray
        Boolean mask (True = far from bright stars, safe to use)
    """
    from scipy.spatial import KDTree

    # Convert to radians
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    star_ra_rad = np.deg2rad(bright_star_ra)
    star_dec_rad = np.deg2rad(bright_star_dec)

    # Convert to Cartesian on unit sphere
    def to_cartesian(ra, dec):
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.column_stack([x, y, z])

    gal_xyz = to_cartesian(ra_rad, dec_rad)
    star_xyz = to_cartesian(star_ra_rad, star_dec_rad)

    # Build KDTree for bright stars
    star_tree = KDTree(star_xyz)

    # Convert exclusion radius to Euclidean distance on unit sphere
    # For small angles: chord_length ≈ 2 * sin(theta/2) ≈ theta
    exclusion_rad = np.deg2rad(exclusion_radius_arcmin / 60.0)
    chord_length = 2 * np.sin(exclusion_rad / 2)

    # Query galaxies near any bright star
    near_star = star_tree.query_ball_point(gal_xyz, chord_length)

    # Create mask (True = not near any bright star)
    mask = np.array([len(neighbors) == 0 for neighbors in near_star])

    n_flagged = np.sum(~mask)
    logger.info(f"Flagged {n_flagged} galaxies near bright stars")

    return mask
