"""
Map processing module for kSZ analysis.

Provides utilities for HEALPix and CAR map operations,
filtering, and masking.
"""

from .healpix_ops import (
    upgrade_downgrade_map,
    get_alm,
    alm2map,
    apply_beam,
    smooth_map,
)
from .filtering import (
    apply_matched_filter,
    apply_wiener_filter,
    compute_filter_transfer_function,
)
from .masking import (
    create_point_source_mask,
    create_galactic_mask,
    create_cluster_mask,
    combine_masks,
    apodize_mask,
)

__all__ = [
    "upgrade_downgrade_map",
    "get_alm",
    "alm2map",
    "apply_beam",
    "smooth_map",
    "apply_matched_filter",
    "apply_wiener_filter",
    "compute_filter_transfer_function",
    "create_point_source_mask",
    "create_galactic_mask",
    "create_cluster_mask",
    "combine_masks",
    "apodize_mask",
]
