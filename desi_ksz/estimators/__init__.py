"""
kSZ estimator module.

This module provides the core estimators for kSZ signal extraction,
including aperture photometry, pairwise momentum estimation, and
theory template generation.
"""

from .pair_counting import (
    EfficientPairCounter,
    count_pairs_in_bins,
    get_pair_indices,
)
from .aperture_photometry import (
    AperturePhotometryStacker,
    extract_temperatures_aperture,
    compensated_aperture_filter,
)
from .pairwise_momentum import (
    PairwiseMomentumEstimator,
    PairwiseMomentumResult,
    compute_geometric_weight,
)
from .theory_template import (
    compute_theory_template,
    compute_mean_pairwise_velocity,
    compute_correlation_function,
)
from .temperature_extraction import (
    extract_object_temperatures,
    TemperatureExtractionResult,
)

__all__ = [
    # Pair counting
    "EfficientPairCounter",
    "count_pairs_in_bins",
    "get_pair_indices",
    # Aperture photometry
    "AperturePhotometryStacker",
    "extract_temperatures_aperture",
    "compensated_aperture_filter",
    # Pairwise momentum
    "PairwiseMomentumEstimator",
    "PairwiseMomentumResult",
    "compute_geometric_weight",
    # Theory
    "compute_theory_template",
    "compute_mean_pairwise_velocity",
    "compute_correlation_function",
    # Temperature extraction
    "extract_object_temperatures",
    "TemperatureExtractionResult",
]
