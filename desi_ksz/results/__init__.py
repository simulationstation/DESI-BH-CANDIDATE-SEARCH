"""
Results packaging module for kSZ analysis.

Provides:
- ResultsPackager: Bundle builder for complete analysis outputs
"""

from .packager import ResultsPackager, create_results_bundle

__all__ = [
    'ResultsPackager',
    'create_results_bundle',
]
