"""
Pairwise momentum plotting functions.

Creates publication-quality plots of p(r) measurements.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mpl = None
    MATPLOTLIB_AVAILABLE = False


def setup_plot_style(style: str = 'paper') -> None:
    """
    Set up matplotlib style for publication plots.

    Parameters
    ----------
    style : str
        'paper' for journal figures, 'presentation' for slides
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    if style == 'paper':
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.figsize': (3.4, 2.8),  # Single column
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
        })
    elif style == 'presentation':
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (8, 6),
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'axes.linewidth': 1.2,
        })


def plot_pairwise_momentum(
    r_centers: np.ndarray,
    p_ksz: np.ndarray,
    p_ksz_err: np.ndarray,
    theory: Optional[np.ndarray] = None,
    theory_amplitude: float = 1.0,
    ax: Optional[Any] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    show_zero: bool = True,
    xlabel: str = r'$r$ [Mpc/$h$]',
    ylabel: str = r'$\hat{p}(r)$ [$\mu$K]',
    title: Optional[str] = None,
) -> Any:
    """
    Plot pairwise momentum p(r) with error bars.

    Parameters
    ----------
    r_centers : np.ndarray
        Separation bin centers (Mpc/h)
    p_ksz : np.ndarray
        Measured pairwise momentum (μK)
    p_ksz_err : np.ndarray
        1-sigma errors (μK)
    theory : np.ndarray, optional
        Theory template
    theory_amplitude : float
        Amplitude to scale theory by
    ax : matplotlib axis, optional
        Axis to plot on
    label : str, optional
        Label for legend
    color : str, optional
        Color for data points
    show_zero : bool
        Show horizontal line at zero
    xlabel, ylabel : str
        Axis labels
    title : str, optional
        Plot title

    Returns
    -------
    ax : matplotlib axis
        The plot axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots()

    # Default color
    if color is None:
        color = 'C0'

    # Plot data with error bars
    ax.errorbar(
        r_centers, p_ksz, yerr=p_ksz_err,
        fmt='o', color=color, markersize=5,
        capsize=2, capthick=1, elinewidth=1,
        label=label,
    )

    # Plot theory if provided
    if theory is not None:
        # Smooth curve for theory
        r_fine = np.linspace(r_centers.min(), r_centers.max(), 100)
        theory_interp = np.interp(r_fine, r_centers, theory)
        ax.plot(
            r_fine, theory_amplitude * theory_interp,
            'k--', linewidth=1.5, alpha=0.7,
            label=f'Theory ($A_{{kSZ}} = {theory_amplitude:.2f}$)'
        )

    # Zero line
    if show_zero:
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if label or theory is not None:
        ax.legend(frameon=False)

    return ax


def plot_pairwise_momentum_tomography(
    results: List[Dict[str, Any]],
    ax: Optional[Any] = None,
    colors: Optional[List[str]] = None,
    show_theory: bool = True,
) -> Any:
    """
    Plot pairwise momentum for multiple redshift bins.

    Parameters
    ----------
    results : list of dict
        Each dict should have keys: 'r_centers', 'p_ksz', 'p_ksz_err',
        'z_min', 'z_max', and optionally 'theory'
    ax : matplotlib axis, optional
        Axis to plot on
    colors : list of str, optional
        Colors for each redshift bin
    show_theory : bool
        Show theory curves

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    n_bins = len(results)

    # Default colors
    if colors is None:
        cmap = plt.cm.viridis
        colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]

    for i, result in enumerate(results):
        z_min = result.get('z_min', 0)
        z_max = result.get('z_max', 1)
        label = f'${z_min:.1f} < z < {z_max:.1f}$'

        ax.errorbar(
            result['r_centers'],
            result['p_ksz'],
            yerr=result['p_ksz_err'],
            fmt='o', color=colors[i], markersize=4,
            capsize=2, capthick=0.8, elinewidth=0.8,
            label=label,
        )

        if show_theory and 'theory' in result:
            ax.plot(
                result['r_centers'],
                result['theory'],
                '--', color=colors[i], linewidth=1, alpha=0.7,
            )

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel(r'$r$ [Mpc/$h$]')
    ax.set_ylabel(r'$\hat{p}(r)$ [$\mu$K]')
    ax.legend(frameon=False, fontsize=8)

    return ax


def plot_amplitude_vs_redshift(
    z_centers: np.ndarray,
    A_ksz: np.ndarray,
    A_ksz_err: np.ndarray,
    ax: Optional[Any] = None,
    theory_A: Optional[float] = None,
    color: str = 'C0',
    label: Optional[str] = None,
) -> Any:
    """
    Plot kSZ amplitude as function of redshift.

    Parameters
    ----------
    z_centers : np.ndarray
        Redshift bin centers
    A_ksz : np.ndarray
        Measured amplitudes
    A_ksz_err : np.ndarray
        Amplitude errors
    ax : matplotlib axis, optional
    theory_A : float, optional
        Expected theory amplitude (usually 1)
    color : str
    label : str, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(
        z_centers, A_ksz, yerr=A_ksz_err,
        fmt='s', color=color, markersize=6,
        capsize=3, capthick=1, elinewidth=1,
        label=label,
    )

    if theory_A is not None:
        ax.axhline(theory_A, color='k', linestyle='--', linewidth=1, alpha=0.7,
                   label=f'Expected ($A_{{kSZ}} = {theory_A}$)')

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$A_{kSZ}$')

    if label or theory_A is not None:
        ax.legend(frameon=False)

    return ax


def plot_comparison(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    ax: Optional[Any] = None,
    offset_factor: float = 0.02,
) -> Any:
    """
    Plot comparison of multiple measurements (e.g., ACT vs Planck).

    Parameters
    ----------
    results_dict : dict
        Keys are labels (e.g., 'ACT DR6', 'Planck PR4')
        Values are dicts with 'r_centers', 'p_ksz', 'p_ksz_err'
    ax : matplotlib axis, optional
    offset_factor : float
        Horizontal offset between datasets for visibility

    Returns
    -------
    ax : matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots()

    colors = ['C0', 'C1', 'C2', 'C3']
    markers = ['o', 's', '^', 'D']

    for i, (label, data) in enumerate(results_dict.items()):
        r = data['r_centers']
        # Apply small horizontal offset
        r_offset = r * (1 + (i - len(results_dict) / 2) * offset_factor)

        ax.errorbar(
            r_offset, data['p_ksz'], yerr=data['p_ksz_err'],
            fmt=markers[i % len(markers)],
            color=colors[i % len(colors)],
            markersize=5, capsize=2, capthick=1, elinewidth=1,
            label=label,
        )

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r'$r$ [Mpc/$h$]')
    ax.set_ylabel(r'$\hat{p}(r)$ [$\mu$K]')
    ax.legend(frameon=False)

    return ax


def save_figure(
    fig: Any,
    filename: str,
    formats: List[str] = ['pdf'],
    dpi: int = 300,
) -> None:
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib figure
    filename : str
        Base filename (without extension)
    formats : list of str
        Output formats ('pdf', 'png', 'svg')
    dpi : int
        Resolution for raster formats
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")

    for fmt in formats:
        outfile = f"{filename}.{fmt}"
        fig.savefig(outfile, format=fmt, dpi=dpi, bbox_inches='tight')
