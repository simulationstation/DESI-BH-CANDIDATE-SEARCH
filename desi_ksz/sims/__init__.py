"""
Simulation module for kSZ analysis validation.

Provides signal injection tests and CMB simulation interfaces.
"""

from .injection_tests import (
    run_injection_test,
    InjectionTestResult,
    validate_estimator_bias,
)
from .cmb_realizations import (
    generate_cmb_realization,
    CMBSimulator,
)

__all__ = [
    "run_injection_test",
    "InjectionTestResult",
    "validate_estimator_bias",
    "generate_cmb_realization",
    "CMBSimulator",
]
