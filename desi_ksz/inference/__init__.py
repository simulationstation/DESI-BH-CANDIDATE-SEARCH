"""
Parameter inference module for kSZ analysis.

Provides likelihood computation and MCMC sampling for
kSZ amplitude and cosmological parameter inference.
"""

from .likelihood import (
    KSZLikelihood,
    compute_chi2,
    fit_amplitude_analytic,
)
from .mcmc import (
    run_mcmc,
    MCMCResult,
    compute_map_estimate,
)

__all__ = [
    "KSZLikelihood",
    "compute_chi2",
    "fit_amplitude_analytic",
    "run_mcmc",
    "MCMCResult",
    "compute_map_estimate",
]
