"""
Decision report visualization for kSZ analysis.

Generates publication-quality summary figures showing:
- Gate evaluation status (dashboard)
- Detection significance across z-bins
- Quality metric comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Color scheme
PASS_COLOR = '#2ecc71'  # Green
FAIL_COLOR = '#e74c3c'  # Red
WARN_COLOR = '#f39c12'  # Orange
SKIP_COLOR = '#95a5a6'  # Gray
INCONCLUSIVE_COLOR = '#9b59b6'  # Purple


def plot_gate_dashboard(
    gate_result: Any,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Create visual dashboard of gate evaluation results.

    Parameters
    ----------
    gate_result : GateEvaluationResult
        Gate evaluation results
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top-left: Overall status banner
    ax_status = axes[0, 0]
    _plot_status_banner(ax_status, gate_result)

    # Top-right: Gate status grid
    ax_gates = axes[0, 1]
    _plot_gate_grid(ax_gates, gate_result)

    # Bottom-left: Critical vs warning breakdown
    ax_breakdown = axes[1, 0]
    _plot_breakdown_pie(ax_breakdown, gate_result)

    # Bottom-right: Metric values vs thresholds
    ax_metrics = axes[1, 1]
    _plot_metric_bars(ax_metrics, gate_result)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        # Also save PNG
        fig.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        logger.info(f"Saved gate dashboard to {output_path}")

    return fig


def _plot_status_banner(ax: plt.Axes, gate_result: Any):
    """Plot overall status banner."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    status = gate_result.overall_status
    if status == "PASS":
        color = PASS_COLOR
        text = "PASS"
    elif status == "FAIL":
        color = FAIL_COLOR
        text = "FAIL"
    else:
        color = INCONCLUSIVE_COLOR
        text = "INCONCLUSIVE"

    # Draw banner
    rect = mpatches.FancyBboxPatch(
        (0.05, 0.2), 0.9, 0.6,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(rect)

    # Status text
    ax.text(0.5, 0.5, text, ha='center', va='center',
            fontsize=32, fontweight='bold', color='white')

    # Subtitle
    ax.text(0.5, 0.1, gate_result.recommendation,
            ha='center', va='center', fontsize=10, style='italic')


def _plot_gate_grid(ax: plt.Axes, gate_result: Any):
    """Plot grid of gate statuses."""
    ax.set_title("Gate Status Overview", fontsize=12, fontweight='bold')

    gate_results = gate_result.gate_results
    n_gates = len(gate_results)

    # Calculate grid dimensions
    n_cols = 4
    n_rows = (n_gates + n_cols - 1) // n_cols

    # Create grid
    for i, gr in enumerate(gate_results):
        row = i // n_cols
        col = i % n_cols

        # Position
        x = col / n_cols + 0.5 / n_cols
        y = 1 - (row + 1) / (n_rows + 1)

        # Color based on status
        if gr.status.value == "PASS":
            color = PASS_COLOR
        elif gr.status.value == "FAIL":
            color = FAIL_COLOR
        elif gr.status.value == "WARN":
            color = WARN_COLOR
        else:
            color = SKIP_COLOR

        # Draw circle
        circle = mpatches.Circle((x, y), 0.08, facecolor=color,
                                  edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)

        # Gate name (shortened)
        name = gr.name.replace('_', '\n')
        ax.text(x, y - 0.15, name, ha='center', va='top',
                fontsize=7, fontweight='bold' if gr.is_critical else 'normal')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def _plot_breakdown_pie(ax: plt.Axes, gate_result: Any):
    """Plot pie chart of gate outcomes."""
    ax.set_title("Gate Breakdown", fontsize=12, fontweight='bold')

    sizes = []
    labels = []
    colors = []

    if gate_result.critical_passed > 0:
        sizes.append(gate_result.critical_passed)
        labels.append(f'Critical Pass ({gate_result.critical_passed})')
        colors.append(PASS_COLOR)

    if gate_result.critical_failed > 0:
        sizes.append(gate_result.critical_failed)
        labels.append(f'Critical Fail ({gate_result.critical_failed})')
        colors.append(FAIL_COLOR)

    if gate_result.warnings > 0:
        sizes.append(gate_result.warnings)
        labels.append(f'Warnings ({gate_result.warnings})')
        colors.append(WARN_COLOR)

    if gate_result.skipped > 0:
        sizes.append(gate_result.skipped)
        labels.append(f'Skipped ({gate_result.skipped})')
        colors.append(SKIP_COLOR)

    if not sizes:
        sizes = [1]
        labels = ['No gates']
        colors = [SKIP_COLOR]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=90,
        explode=[0.05] * len(sizes)
    )

    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')


def _plot_metric_bars(ax: plt.Axes, gate_result: Any):
    """Plot bar chart of metric values vs thresholds."""
    ax.set_title("Metric Values vs Thresholds", fontsize=12, fontweight='bold')

    # Filter gates with finite metrics
    valid_gates = [gr for gr in gate_result.gate_results
                   if np.isfinite(gr.metric) and gr.status.value != "SKIP"]

    if not valid_gates:
        ax.text(0.5, 0.5, "No metrics to display", ha='center', va='center')
        ax.axis('off')
        return

    names = [gr.name.replace('_', '\n') for gr in valid_gates]
    metrics = [gr.metric for gr in valid_gates]
    thresholds = [gr.threshold for gr in valid_gates]
    colors = []

    for gr in valid_gates:
        if gr.status.value == "PASS":
            colors.append(PASS_COLOR)
        elif gr.status.value == "FAIL":
            colors.append(FAIL_COLOR)
        else:
            colors.append(WARN_COLOR)

    x = np.arange(len(names))
    width = 0.35

    # Normalize metrics relative to thresholds for visualization
    # (so we can see both on same scale)
    norm_metrics = []
    norm_thresholds = []
    for m, t in zip(metrics, thresholds):
        if t != 0:
            norm_metrics.append(m / t)
            norm_thresholds.append(1.0)  # Threshold normalized to 1
        else:
            norm_metrics.append(m)
            norm_thresholds.append(t)

    bars = ax.bar(x, norm_metrics, width, color=colors, alpha=0.8, label='Metric')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Threshold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Value / Threshold')
    ax.legend(loc='upper right')

    # Add value labels on bars
    for bar, m, t in zip(bars, metrics, thresholds):
        height = bar.get_height()
        ax.annotate(f'{m:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)


def plot_detection_summary(
    z_bin_results: List[Any],
    joint_snr: Optional[float] = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot detection significance across z-bins.

    Parameters
    ----------
    z_bin_results : list
        List of TomographicResult objects
    joint_snr : float, optional
        Joint detection significance
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: SNR per z-bin
    z_means = [zr.z_mean for zr in z_bin_results]
    snrs = [zr.snr for zr in z_bin_results]
    amplitudes = [zr.amplitude for zr in z_bin_results]
    amp_errs = [zr.amplitude_err for zr in z_bin_results]

    colors = [PASS_COLOR if s >= 3 else WARN_COLOR if s >= 2 else FAIL_COLOR for s in snrs]

    ax1.bar(range(len(z_bin_results)), snrs, color=colors, alpha=0.8)
    ax1.axhline(y=3, color='green', linestyle='--', linewidth=2, label='3-sigma')
    ax1.axhline(y=2, color='orange', linestyle=':', linewidth=2, label='2-sigma')

    ax1.set_xticks(range(len(z_bin_results)))
    ax1.set_xticklabels([zr.z_bin_label for zr in z_bin_results], rotation=45, ha='right')
    ax1.set_ylabel('Detection Significance (sigma)')
    ax1.set_title('S/N per Redshift Bin')
    ax1.legend()

    # Add joint SNR marker if available
    if joint_snr is not None:
        ax1.axhline(y=joint_snr, color='purple', linestyle='-', linewidth=2,
                    label=f'Joint: {joint_snr:.1f}sigma')
        ax1.legend()

    # Right: Amplitude vs z
    ax2.errorbar(z_means, amplitudes, yerr=amp_errs, fmt='o', capsize=5,
                 color='navy', markersize=8, linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mean Redshift')
    ax2.set_ylabel('kSZ Amplitude (A)')
    ax2.set_title('Amplitude Evolution with Redshift')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        fig.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        logger.info(f"Saved detection summary to {output_path}")

    return fig


def plot_quality_metrics(
    metrics: Dict[str, float],
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot quality metrics comparison.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric name -> value
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    # Define thresholds for key metrics
    thresholds = {
        'null_pass_rate': (0.8, 'ge', 'Null Test Pass Rate'),
        'injection_bias_sigma': (2.0, 'abs_lt', 'Injection Bias (sigma)'),
        'transfer_bias': (0.05, 'abs_lt', 'Transfer Bias'),
        'tsz_delta_sigma': (1.0, 'lt', 'tSZ Stability (sigma)'),
        'hartlap_factor': (0.5, 'gt', 'Hartlap Factor'),
        'condition_number': (1e6, 'lt', 'Condition Number'),
        'map_diff_sigma': (2.0, 'lt', 'Map Consistency (sigma)'),
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Filter to available metrics
    available = [(k, v, thresholds[k]) for k, v in metrics.items()
                 if k in thresholds and np.isfinite(v)]

    if not available:
        ax.text(0.5, 0.5, "No quality metrics available", ha='center', va='center')
        ax.axis('off')
        return fig

    names = [t[2] for _, _, t in available]
    values = [v for _, v, _ in available]
    thresh_vals = [t[0] for _, _, t in available]
    comparators = [t[1] for _, _, t in available]

    # Determine pass/fail
    colors = []
    for v, t, c in zip(values, thresh_vals, comparators):
        if c == 'ge':
            passed = v >= t
        elif c == 'gt':
            passed = v > t
        elif c == 'lt':
            passed = v < t
        elif c == 'abs_lt':
            passed = abs(v) < t
        else:
            passed = True
        colors.append(PASS_COLOR if passed else FAIL_COLOR)

    x = np.arange(len(names))

    # Normalize values for display
    norm_values = []
    for v, t, c in zip(values, thresh_vals, comparators):
        if c in ['ge', 'gt']:
            # Higher is better
            norm_values.append(v / t if t != 0 else v)
        else:
            # Lower is better - invert
            norm_values.append(t / v if v != 0 else t)

    bars = ax.bar(x, norm_values, color=colors, alpha=0.8)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Threshold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Normalized Value (1.0 = threshold)')
    ax.set_title('Quality Metrics Summary')
    ax.legend()

    # Add actual values as labels
    for bar, v in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{v:.3g}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        fig.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        logger.info(f"Saved quality metrics to {output_path}")

    return fig


def generate_decision_report_figures(
    phase34_result: Any,
    gate_result: Any,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate all decision report figures.

    Parameters
    ----------
    phase34_result : Phase34Result
        Complete Phase 3-4 results
    gate_result : GateEvaluationResult
        Gate evaluation results
    output_dir : Path
        Directory to save figures

    Returns
    -------
    dict
        Dictionary of figure name -> path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Gate dashboard
    dashboard_path = output_dir / "gate_dashboard.pdf"
    plot_gate_dashboard(gate_result, dashboard_path)
    figures['gate_dashboard'] = dashboard_path

    # Detection summary
    if hasattr(phase34_result, 'z_bin_results') and phase34_result.z_bin_results:
        detection_path = output_dir / "detection_summary.pdf"
        joint_snr = getattr(phase34_result, 'joint_snr', None)
        plot_detection_summary(phase34_result.z_bin_results, joint_snr, detection_path)
        figures['detection_summary'] = detection_path

    # Quality metrics
    if hasattr(phase34_result, 'metrics') and phase34_result.metrics:
        metrics_path = output_dir / "quality_metrics.pdf"
        plot_quality_metrics(phase34_result.metrics, metrics_path)
        figures['quality_metrics'] = metrics_path

    logger.info(f"Generated {len(figures)} decision report figures")
    return figures
