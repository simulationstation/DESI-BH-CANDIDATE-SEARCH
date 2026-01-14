"""
Redshift binning for kSZ tomography.

This module provides functions for defining and managing tomographic
redshift bins for kSZ analysis.
"""

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class RedshiftBin:
    """Single redshift bin definition."""
    z_min: float
    z_max: float
    label: Optional[str] = None

    def __post_init__(self):
        if self.z_min >= self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be < z_max ({self.z_max})")
        if self.label is None:
            self.label = f"z{self.z_min:.2f}-{self.z_max:.2f}"

    @property
    def z_center(self) -> float:
        """Central redshift of bin."""
        return 0.5 * (self.z_min + self.z_max)

    @property
    def z_width(self) -> float:
        """Width of redshift bin."""
        return self.z_max - self.z_min

    def contains(self, z: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if redshift(s) fall within bin."""
        return (z >= self.z_min) & (z < self.z_max)


@dataclass
class RedshiftBinning:
    """
    Container for multiple redshift bins.

    Manages a set of tomographic redshift bins for kSZ analysis,
    supporting both equal-number and fixed-edge binning schemes.

    Parameters
    ----------
    bins : List[RedshiftBin]
        List of redshift bins
    tracer : str
        Galaxy tracer type

    Examples
    --------
    >>> binning = create_equal_number_bins(z_array, n_bins=4)
    >>> for i, zbin in enumerate(binning):
    ...     mask = zbin.contains(z)
    ...     print(f"Bin {i}: {np.sum(mask)} galaxies")
    """
    bins: List[RedshiftBin]
    tracer: str = "unknown"
    _z_edges: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        # Sort bins by z_min
        self.bins = sorted(self.bins, key=lambda b: b.z_min)
        # Compute edges
        edges = [b.z_min for b in self.bins] + [self.bins[-1].z_max]
        self._z_edges = np.array(edges)

    def __len__(self) -> int:
        return len(self.bins)

    def __iter__(self):
        return iter(self.bins)

    def __getitem__(self, idx: int) -> RedshiftBin:
        return self.bins[idx]

    @property
    def n_bins(self) -> int:
        """Number of redshift bins."""
        return len(self.bins)

    @property
    def z_edges(self) -> np.ndarray:
        """Bin edges array."""
        return self._z_edges

    @property
    def z_centers(self) -> np.ndarray:
        """Bin center array."""
        return np.array([b.z_center for b in self.bins])

    @property
    def z_min(self) -> float:
        """Minimum redshift across all bins."""
        return self.bins[0].z_min

    @property
    def z_max(self) -> float:
        """Maximum redshift across all bins."""
        return self.bins[-1].z_max

    def get_bin_index(self, z: np.ndarray) -> np.ndarray:
        """
        Get bin index for each redshift.

        Parameters
        ----------
        z : np.ndarray
            Redshift values

        Returns
        -------
        np.ndarray
            Bin indices (-1 for out of range)
        """
        indices = np.digitize(z, self._z_edges) - 1
        # Mark out-of-range as -1
        indices[indices < 0] = -1
        indices[indices >= self.n_bins] = -1
        return indices

    def get_bin_masks(self, z: np.ndarray) -> List[np.ndarray]:
        """
        Get boolean masks for each bin.

        Parameters
        ----------
        z : np.ndarray
            Redshift values

        Returns
        -------
        List[np.ndarray]
            List of boolean masks, one per bin
        """
        return [zbin.contains(z) for zbin in self.bins]

    def get_summary(self) -> dict:
        """Get summary of binning scheme."""
        return {
            "tracer": self.tracer,
            "n_bins": self.n_bins,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "z_edges": self._z_edges.tolist(),
            "z_centers": self.z_centers.tolist(),
        }


def create_equal_number_bins(
    z: np.ndarray,
    n_bins: int = 4,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    tracer: str = "unknown",
) -> RedshiftBinning:
    """
    Create redshift bins with approximately equal numbers of galaxies.

    Parameters
    ----------
    z : np.ndarray
        Galaxy redshifts
    n_bins : int
        Number of bins to create
    z_min : float, optional
        Minimum redshift (default: min of z)
    z_max : float, optional
        Maximum redshift (default: max of z)
    weights : np.ndarray, optional
        Galaxy weights for weighted bin edges
    tracer : str
        Galaxy tracer type

    Returns
    -------
    RedshiftBinning
        Binning scheme with equal-number bins
    """
    # Apply redshift cuts
    if z_min is None:
        z_min = np.min(z)
    if z_max is None:
        z_max = np.max(z)

    mask = (z >= z_min) & (z < z_max)
    z_cut = z[mask]

    if weights is not None:
        w_cut = weights[mask]
    else:
        w_cut = None

    if len(z_cut) == 0:
        raise ValueError("No galaxies in specified redshift range")

    # Compute quantiles for equal-number bins
    if w_cut is not None:
        # Weighted quantiles
        edges = _weighted_quantiles(z_cut, w_cut, n_bins + 1)
    else:
        # Unweighted quantiles
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(z_cut, percentiles)

    # Ensure exact endpoints
    edges[0] = z_min
    edges[-1] = z_max

    # Create bins
    bins = []
    for i in range(n_bins):
        bins.append(RedshiftBin(
            z_min=edges[i],
            z_max=edges[i + 1],
            label=f"bin{i}",
        ))

    binning = RedshiftBinning(bins=bins, tracer=tracer)

    # Log bin statistics
    logger.info(f"Created {n_bins} equal-number bins for {tracer}:")
    for i, zbin in enumerate(binning):
        n_in_bin = np.sum(zbin.contains(z))
        logger.info(f"  Bin {i}: z=[{zbin.z_min:.3f}, {zbin.z_max:.3f}), N={n_in_bin}")

    return binning


def create_fixed_bins(
    z_edges: Union[List[float], np.ndarray],
    tracer: str = "unknown",
) -> RedshiftBinning:
    """
    Create redshift bins with fixed edges.

    Parameters
    ----------
    z_edges : array-like
        Bin edges (n_bins + 1 values)
    tracer : str
        Galaxy tracer type

    Returns
    -------
    RedshiftBinning
        Binning scheme with fixed edges
    """
    z_edges = np.asarray(z_edges)
    n_bins = len(z_edges) - 1

    bins = []
    for i in range(n_bins):
        bins.append(RedshiftBin(
            z_min=z_edges[i],
            z_max=z_edges[i + 1],
            label=f"bin{i}",
        ))

    return RedshiftBinning(bins=bins, tracer=tracer)


def get_default_bins(tracer: str) -> RedshiftBinning:
    """
    Get default redshift binning for a tracer.

    Parameters
    ----------
    tracer : str
        Galaxy tracer type

    Returns
    -------
    RedshiftBinning
        Default binning scheme for tracer
    """
    default_edges = {
        "BGS_BRIGHT": [0.1, 0.2, 0.3, 0.4],
        "BGS_FAINT": [0.1, 0.2, 0.3, 0.4],
        "LRG": [0.4, 0.5, 0.6, 0.7, 0.8],
        "ELG_LOP": [0.8, 1.0, 1.2, 1.4, 1.6],
        "QSO": [0.8, 1.2, 1.6, 2.1],
    }

    if tracer not in default_edges:
        raise ValueError(f"No default binning for tracer: {tracer}")

    return create_fixed_bins(default_edges[tracer], tracer=tracer)


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    n_quantiles: int,
) -> np.ndarray:
    """Compute weighted quantiles."""
    # Sort by value
    sort_idx = np.argsort(values)
    sorted_values = values[sort_idx]
    sorted_weights = weights[sort_idx]

    # Cumulative sum of weights
    cum_weights = np.cumsum(sorted_weights)
    cum_weights /= cum_weights[-1]  # Normalize

    # Quantile positions
    quantile_positions = np.linspace(0, 1, n_quantiles)

    # Interpolate
    edges = np.interp(quantile_positions, cum_weights, sorted_values)

    return edges


def combine_binnings(
    binnings: List[RedshiftBinning],
) -> RedshiftBinning:
    """
    Combine multiple binning schemes.

    Parameters
    ----------
    binnings : List[RedshiftBinning]
        List of binning schemes to combine

    Returns
    -------
    RedshiftBinning
        Combined binning scheme
    """
    all_bins = []
    tracers = []

    for binning in binnings:
        all_bins.extend(binning.bins)
        tracers.append(binning.tracer)

    # Sort by z_min
    all_bins = sorted(all_bins, key=lambda b: b.z_min)

    combined_tracer = "+".join(set(tracers))
    return RedshiftBinning(bins=all_bins, tracer=combined_tracer)
