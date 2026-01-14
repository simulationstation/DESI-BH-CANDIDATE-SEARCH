"""
Configuration module for kSZ analysis pipeline.

Provides configuration schemas, defaults, and preset configurations
for different CMB maps and galaxy tracers.
"""

from .schemas import (
    PipelineConfig,
    DESIConfig,
    CMBConfig,
    EstimatorConfig,
    CovarianceConfig,
    InferenceConfig,
    load_config,
    save_config,
)
from .defaults import (
    DEFAULT_SEPARATION_BINS,
    DEFAULT_REDSHIFT_BINS,
    DEFAULT_APERTURE_INNER,
    DEFAULT_APERTURE_OUTER,
    DEFAULT_JACKKNIFE_REGIONS,
    COSMOLOGY_PARAMS,
)

__all__ = [
    "PipelineConfig",
    "DESIConfig",
    "CMBConfig",
    "EstimatorConfig",
    "CovarianceConfig",
    "InferenceConfig",
    "load_config",
    "save_config",
    "DEFAULT_SEPARATION_BINS",
    "DEFAULT_REDSHIFT_BINS",
    "DEFAULT_APERTURE_INNER",
    "DEFAULT_APERTURE_OUTER",
    "DEFAULT_JACKKNIFE_REGIONS",
    "COSMOLOGY_PARAMS",
]
