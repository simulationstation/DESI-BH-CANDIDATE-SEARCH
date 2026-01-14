"""
Covariance estimation module for kSZ analysis.

Provides jackknife resampling, Hartlap correction, and mock interface.
"""

from .jackknife import (
    SpatialJackknife,
    compute_jackknife_covariance,
    define_jackknife_regions,
)
from .hartlap import (
    apply_hartlap_correction,
    compute_precision_matrix,
    regularize_covariance,
)
from .mock_interface import (
    MockCatalogInterface,
    load_mock_velocities,
)

__all__ = [
    "SpatialJackknife",
    "compute_jackknife_covariance",
    "define_jackknife_regions",
    "apply_hartlap_correction",
    "compute_precision_matrix",
    "regularize_covariance",
    "MockCatalogInterface",
    "load_mock_velocities",
]
