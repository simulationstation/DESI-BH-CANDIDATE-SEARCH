"""
Plotting module for kSZ analysis.

Provides publication-quality figures for kSZ results.
"""

from .pairwise_momentum import (
    plot_pairwise_momentum,
    plot_pairwise_momentum_tomography,
)
from .null_test_summary import (
    plot_null_test_summary,
    plot_null_chi2_distribution,
)
from .corner_plots import (
    plot_corner,
    plot_amplitude_posterior,
)
from .decision_report import (
    plot_gate_dashboard,
    plot_detection_summary,
    plot_quality_metrics,
    generate_decision_report_figures,
)

__all__ = [
    "plot_pairwise_momentum",
    "plot_pairwise_momentum_tomography",
    "plot_null_test_summary",
    "plot_null_chi2_distribution",
    "plot_corner",
    "plot_amplitude_posterior",
    "plot_gate_dashboard",
    "plot_detection_summary",
    "plot_quality_metrics",
    "generate_decision_report_figures",
]
