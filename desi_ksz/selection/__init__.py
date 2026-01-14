"""
Selection module for kSZ analysis pipeline.

Provides functions for galaxy selection, quality cuts, redshift binning,
and weight computation.
"""

from .quality_cuts import (
    apply_quality_cuts,
    apply_fiber_collision_correction,
    remove_stellar_contamination,
)
from .redshift_bins import (
    RedshiftBinning,
    create_equal_number_bins,
    create_fixed_bins,
    get_default_bins,
)
from .weights import (
    compute_total_weight,
    compute_fkp_weights,
    compute_effective_number,
    normalize_weights,
)

__all__ = [
    # Quality cuts
    "apply_quality_cuts",
    "apply_fiber_collision_correction",
    "remove_stellar_contamination",
    # Redshift bins
    "RedshiftBinning",
    "create_equal_number_bins",
    "create_fixed_bins",
    "get_default_bins",
    # Weights
    "compute_total_weight",
    "compute_fkp_weights",
    "compute_effective_number",
    "normalize_weights",
]
