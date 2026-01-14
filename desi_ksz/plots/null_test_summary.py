"""
Null test summary plotting functions.

Creates visualization of null test results including chi2 distributions
and PTE summaries.
"""

from typing import Optional, Dict, List, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mpatches = None
    MATPLOTLIB_AVAILABLE = False


def plot_null_test_summary(
    results: Dict[str, Any],
    ax: Optional[Any] = None,
    pte_threshold: float = 0.05,
    show_pte: bool = True,
) -> Any:
    """
    Plot summary of null test results as horizontal bar chart.

    Parameters
    ----------
    results : dict
        Dictionary of test_name -> NullTestResult
    ax : matplotlib axis, optional
    pte_threshold : float
        Threshold for pass/fail coloring
    show_pte : bool
        Show PTE values as text

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    test_names = list(results.keys())
    n_tests = len(test_names)

    # Extract PTE values
    pte_values = []
    for name in test_names:
        result = results[name]
        if hasattr(result, 'pte'):
            pte_values.append(result.pte)
        else:
            pte_values.append(result.get('pte', 0.5))

    pte_values = np.array(pte_values)

    # Colors based on pass/fail
    colors = ['#2ecc71' if p > pte_threshold else '#e74c3c' for p in pte_values]

    # Create horizontal bar chart
    y_pos = np.arange(n_tests)
    bars = ax.barh(y_pos, pte_values, color=colors, edgecolor='black', linewidth=0.5)

    # Threshold line
    ax.axvline(pte_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ').title() for name in test_names])
    ax.set_xlabel('PTE')
    ax.set_xlim(0, 1)

    # Add PTE values as text
    if show_pte:
        for i, (pte, bar) in enumerate(zip(pte_values, bars)):
            width = bar.get_width()
            if width > 0.15:
                ax.text(width - 0.02, bar.get_y() + bar.get_height() / 2,
                        f'{pte:.3f}', ha='right', va='center', fontsize=8,
                        color='white', fontweight='bold')
            else:
                ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                        f'{pte:.3f}', ha='left', va='center', fontsize=8)

    # Legend
    pass_patch = mpatches.Patch(color='#2ecc71', label='Pass')
    fail_patch = mpatches.Patch(color='#e74c3c', label='Fail')
    ax.legend(handles=[pass_patch, fail_patch], loc='lower right', frameon=False)

    # Title with summary
    n_passed = sum(p > pte_threshold for p in pte_values)
    ax.set_title(f'Null Test Summary: {n_passed}/{n_tests} passed')

    ax.invert_yaxis()  # Top to bottom

    return ax


def plot_null_chi2_distribution(
    null_chi2: np.ndarray,
    observed_chi2: float,
    expected_chi2: float,
    test_name: str = 'Null Test',
    ax: Optional[Any] = None,
    n_bins: int = 50,
) -> Any:
    """
    Plot chi2 distribution from null realizations.

    Parameters
    ----------
    null_chi2 : np.ndarray
        Chi2 values from null realizations
    observed_chi2 : float
        Observed chi2 from data
    expected_chi2 : float
        Expected chi2 under null (typically n_dof)
    test_name : str
        Name of the test for title
    ax : matplotlib axis, optional
    n_bins : int
        Number of histogram bins

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots()

    # Histogram of null chi2
    ax.hist(
        null_chi2, bins=n_bins, density=True,
        alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5,
        label='Null realizations'
    )

    # Expected chi2 distribution
    from scipy.stats import chi2 as chi2_dist
    x = np.linspace(0, np.percentile(null_chi2, 99.9), 200)
    ax.plot(
        x, chi2_dist.pdf(x, expected_chi2),
        'k--', linewidth=1.5, alpha=0.7,
        label=f'$\\chi^2_{{{int(expected_chi2)}}}$'
    )

    # Observed value
    ax.axvline(
        observed_chi2, color='red', linestyle='-', linewidth=2,
        label=f'Observed ($\\chi^2 = {observed_chi2:.1f}$)'
    )

    # PTE annotation
    pte = np.mean(null_chi2 >= observed_chi2)
    ax.text(
        0.95, 0.95, f'PTE = {pte:.3f}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('Probability density')
    ax.set_title(test_name.replace('_', ' ').title())
    ax.legend(frameon=False, fontsize=8)

    return ax


def plot_null_test_grid(
    results: Dict[str, Any],
    figsize: tuple = (10, 8),
) -> Any:
    """
    Create grid of chi2 distributions for all null tests.

    Parameters
    ----------
    results : dict
        Dictionary of test_name -> NullTestResult
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    n_tests = len(results)
    n_cols = min(3, n_tests)
    n_rows = (n_tests + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i, (name, result) in enumerate(results.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Check if we have chi2 distribution data
        if hasattr(result, 'null_chi2_distribution'):
            null_chi2 = result.null_chi2_distribution
            if len(null_chi2) > 0:
                plot_null_chi2_distribution(
                    null_chi2,
                    result.observed_chi2,
                    result.expected_chi2,
                    test_name=name,
                    ax=ax,
                )
            else:
                # Analytic PTE case
                ax.text(
                    0.5, 0.5, f'PTE = {result.pte:.3f}\n(analytic)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12
                )
                ax.set_title(name.replace('_', ' ').title())
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')

    # Hide empty subplots
    for i in range(n_tests, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    fig.tight_layout()
    return fig


def create_null_test_table(
    results: Dict[str, Any],
    output_format: str = 'latex',
) -> str:
    """
    Create formatted table of null test results.

    Parameters
    ----------
    results : dict
        Dictionary of test_name -> NullTestResult
    output_format : str
        'latex', 'markdown', or 'ascii'

    Returns
    -------
    str
        Formatted table string
    """
    lines = []

    if output_format == 'latex':
        lines.append(r'\begin{tabular}{lccc}')
        lines.append(r'\hline')
        lines.append(r'Test & $\chi^2$ & PTE & Status \\')
        lines.append(r'\hline')

        for name, result in results.items():
            if hasattr(result, 'pte'):
                pte = result.pte
                chi2 = result.observed_chi2
                passed = result.passed
            else:
                pte = result.get('pte', 0)
                chi2 = result.get('chi2', 0)
                passed = pte > 0.05

            status = r'\checkmark' if passed else r'\times'
            test_label = name.replace('_', r'\_')
            lines.append(f'{test_label} & {chi2:.1f} & {pte:.3f} & {status} \\\\')

        lines.append(r'\hline')
        lines.append(r'\end{tabular}')

    elif output_format == 'markdown':
        lines.append('| Test | χ² | PTE | Status |')
        lines.append('|------|---:|----:|:------:|')

        for name, result in results.items():
            if hasattr(result, 'pte'):
                pte = result.pte
                chi2 = result.observed_chi2
                passed = result.passed
            else:
                pte = result.get('pte', 0)
                chi2 = result.get('chi2', 0)
                passed = pte > 0.05

            status = '✓' if passed else '✗'
            test_label = name.replace('_', ' ')
            lines.append(f'| {test_label} | {chi2:.1f} | {pte:.3f} | {status} |')

    else:  # ascii
        lines.append(f"{'Test':<25} {'Chi2':>8} {'PTE':>8} {'Status':>8}")
        lines.append('-' * 52)

        for name, result in results.items():
            if hasattr(result, 'pte'):
                pte = result.pte
                chi2 = result.observed_chi2
                passed = result.passed
            else:
                pte = result.get('pte', 0)
                chi2 = result.get('chi2', 0)
                passed = pte > 0.05

            status = 'PASS' if passed else 'FAIL'
            test_label = name.replace('_', ' ')
            lines.append(f'{test_label:<25} {chi2:>8.1f} {pte:>8.3f} {status:>8}')

    return '\n'.join(lines)
