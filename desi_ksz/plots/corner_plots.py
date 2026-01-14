"""
Corner plot and posterior visualization functions.

Creates parameter posterior plots for MCMC chains.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


def plot_corner(
    samples: np.ndarray,
    param_names: List[str],
    truths: Optional[List[float]] = None,
    quantiles: List[float] = [0.16, 0.5, 0.84],
    color: str = 'steelblue',
    truth_color: str = 'red',
    figsize: Optional[Tuple[float, float]] = None,
) -> Any:
    """
    Create corner plot for MCMC samples.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples, shape (n_samples, n_params)
    param_names : list of str
        Parameter names for labels
    truths : list of float, optional
        True parameter values to mark
    quantiles : list of float
        Quantiles to mark on 1D histograms
    color : str
        Color for histograms and contours
    truth_color : str
        Color for truth markers
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    n_params = samples.shape[1]

    if figsize is None:
        figsize = (2 * n_params, 2 * n_params)

    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

    # Make axes 2D even for single parameter
    if n_params == 1:
        axes = np.array([[axes]])

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if j > i:
                # Upper triangle - hide
                ax.set_visible(False)

            elif i == j:
                # Diagonal - 1D histogram
                _plot_1d_hist(
                    ax, samples[:, i], param_names[i],
                    quantiles=quantiles, color=color
                )
                if truths is not None and truths[i] is not None:
                    ax.axvline(truths[i], color=truth_color, linestyle='-', linewidth=1.5)

            else:
                # Lower triangle - 2D contour
                _plot_2d_contour(
                    ax, samples[:, j], samples[:, i],
                    color=color
                )
                if truths is not None:
                    if truths[j] is not None:
                        ax.axvline(truths[j], color=truth_color, linestyle='-', linewidth=1, alpha=0.7)
                    if truths[i] is not None:
                        ax.axhline(truths[i], color=truth_color, linestyle='-', linewidth=1, alpha=0.7)

            # Labels
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            else:
                ax.set_xticklabels([])

            if j == 0 and i != j:
                ax.set_ylabel(param_names[i])
            elif i == j:
                ax.set_ylabel('')
            else:
                ax.set_yticklabels([])

    fig.tight_layout()
    return fig


def _plot_1d_hist(
    ax: Any,
    samples: np.ndarray,
    param_name: str,
    quantiles: List[float],
    color: str,
    n_bins: int = 40,
) -> None:
    """Plot 1D histogram with quantile markers."""
    ax.hist(
        samples, bins=n_bins, density=True,
        color=color, alpha=0.7, edgecolor='black', linewidth=0.5
    )

    # Mark quantiles
    for q in quantiles:
        val = np.percentile(samples, 100 * q)
        ax.axvline(val, color='black', linestyle='--', linewidth=1, alpha=0.7)

    # Title with median and errors
    med = np.percentile(samples, 50)
    lo = med - np.percentile(samples, 16)
    hi = np.percentile(samples, 84) - med
    ax.set_title(f'${med:.3f}_{{-{lo:.3f}}}^{{+{hi:.3f}}}$', fontsize=9)


def _plot_2d_contour(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    levels: List[float] = [0.393, 0.865],  # 1-sigma, 2-sigma for 2D Gaussian
    n_bins: int = 50,
) -> None:
    """Plot 2D histogram with contours."""
    # 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=n_bins)
    H = H.T  # Transpose to match x-y orientation

    # Compute levels for contours
    H_sorted = np.sort(H.flatten())[::-1]
    H_cumsum = np.cumsum(H_sorted)
    H_cumsum /= H_cumsum[-1]

    contour_levels = []
    for level in levels:
        idx = np.searchsorted(H_cumsum, level)
        if idx < len(H_sorted):
            contour_levels.append(H_sorted[idx])

    # Plot filled contours
    X, Y = np.meshgrid(
        0.5 * (xedges[:-1] + xedges[1:]),
        0.5 * (yedges[:-1] + yedges[1:])
    )

    ax.contourf(
        X, Y, H, levels=[0] + contour_levels + [H.max()],
        colors=[color], alpha=[0.2, 0.5, 0.8]
    )

    # Plot contour lines
    ax.contour(
        X, Y, H, levels=contour_levels,
        colors=[color], linewidths=1, alpha=0.8
    )


def plot_amplitude_posterior(
    samples: np.ndarray,
    ax: Optional[Any] = None,
    truth: Optional[float] = None,
    prior_bounds: Optional[Tuple[float, float]] = None,
    label: str = r'$A_{kSZ}$',
    color: str = 'steelblue',
    n_bins: int = 50,
) -> Any:
    """
    Plot 1D posterior for kSZ amplitude.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples for A_kSZ
    ax : matplotlib axis, optional
    truth : float, optional
        True/expected value
    prior_bounds : tuple, optional
        Prior bounds (min, max)
    label : str
        Parameter label
    color : str
    n_bins : int

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots()

    # Histogram
    ax.hist(
        samples, bins=n_bins, density=True,
        color=color, alpha=0.7, edgecolor='black', linewidth=0.5,
        label='Posterior'
    )

    # Statistics
    med = np.median(samples)
    lo = med - np.percentile(samples, 16)
    hi = np.percentile(samples, 84) - med

    # Mark median and 68% CI
    ax.axvline(med, color='black', linestyle='-', linewidth=2,
               label=f'Median: {med:.3f}')
    ax.axvline(med - lo, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(med + hi, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # Mark truth if provided
    if truth is not None:
        ax.axvline(truth, color='red', linestyle='-', linewidth=2,
                   label=f'Expected: {truth:.3f}')

    # Mark prior bounds
    if prior_bounds is not None:
        ax.axvline(prior_bounds[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(prior_bounds[1], color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Zero line
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel(label)
    ax.set_ylabel('Probability density')

    # Title with constraint
    ax.set_title(f'{label} = ${med:.3f}_{{-{lo:.3f}}}^{{+{hi:.3f}}}$')

    ax.legend(frameon=False, fontsize=9)

    return ax


def plot_chain_diagnostics(
    chain: np.ndarray,
    param_names: List[str],
    figsize: Tuple[float, float] = (10, 6),
) -> Any:
    """
    Plot MCMC chain diagnostics (trace plots).

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain, shape (n_steps, n_walkers, n_params) or (n_steps, n_params)
    param_names : list of str
        Parameter names
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    if chain.ndim == 2:
        # Flat chain
        n_steps, n_params = chain.shape
        chain = chain.reshape(n_steps, 1, n_params)

    n_steps, n_walkers, n_params = chain.shape

    fig, axes = plt.subplots(n_params, 2, figsize=figsize)
    if n_params == 1:
        axes = axes.reshape(1, 2)

    for i, name in enumerate(param_names):
        # Trace plot (left)
        ax_trace = axes[i, 0]
        for w in range(n_walkers):
            ax_trace.plot(chain[:, w, i], alpha=0.3, linewidth=0.5)
        ax_trace.set_ylabel(name)
        ax_trace.set_xlabel('Step')
        ax_trace.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Histogram (right)
        ax_hist = axes[i, 1]
        flat_samples = chain[:, :, i].flatten()
        ax_hist.hist(flat_samples, bins=50, density=True,
                     color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_hist.set_xlabel(name)
        ax_hist.set_ylabel('Density')

    axes[0, 0].set_title('Trace')
    axes[0, 1].set_title('Posterior')

    fig.tight_layout()
    return fig


def summarize_mcmc(
    samples: np.ndarray,
    param_names: List[str],
    output_format: str = 'dict',
) -> Any:
    """
    Summarize MCMC results.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples, shape (n_samples, n_params)
    param_names : list of str
        Parameter names
    output_format : str
        'dict', 'latex', or 'markdown'

    Returns
    -------
    Summary in requested format
    """
    summary = {}
    for i, name in enumerate(param_names):
        s = samples[:, i]
        summary[name] = {
            'mean': float(np.mean(s)),
            'std': float(np.std(s)),
            'median': float(np.median(s)),
            'lower_68': float(np.percentile(s, 16)),
            'upper_68': float(np.percentile(s, 84)),
            'lower_95': float(np.percentile(s, 2.5)),
            'upper_95': float(np.percentile(s, 97.5)),
        }

    if output_format == 'dict':
        return summary

    elif output_format == 'latex':
        lines = [r'\begin{tabular}{lcc}']
        lines.append(r'\hline')
        lines.append(r'Parameter & Median & 68\% CI \\')
        lines.append(r'\hline')

        for name, stats in summary.items():
            med = stats['median']
            lo = med - stats['lower_68']
            hi = stats['upper_68'] - med
            lines.append(f'${name}$ & ${med:.3f}$ & $^{{+{hi:.3f}}}_{{-{lo:.3f}}}$ \\\\')

        lines.append(r'\hline')
        lines.append(r'\end{tabular}')
        return '\n'.join(lines)

    elif output_format == 'markdown':
        lines = ['| Parameter | Median | 68% CI |']
        lines.append('|-----------|-------:|:------:|')

        for name, stats in summary.items():
            med = stats['median']
            lo = med - stats['lower_68']
            hi = stats['upper_68'] - med
            lines.append(f'| {name} | {med:.3f} | +{hi:.3f}/-{lo:.3f} |')

        return '\n'.join(lines)

    return summary
