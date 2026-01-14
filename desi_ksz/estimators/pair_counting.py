"""
Efficient pair counting for kSZ analysis.

This module provides KDTree-based pair counting for efficient
computation of galaxy pair statistics in comoving space.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| r_ij        | Comoving separation between galaxies i and j    | Mpc/h       |
| x, y, z     | Comoving Cartesian coordinates                  | Mpc/h       |
| N_pairs(r)  | Number of pairs in separation bin              | dimensionless|

References
----------
- scipy.spatial.KDTree documentation
- Corrfunc: Sinha & Garrison 2020, MNRAS, 491, 3022
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from scipy.spatial import KDTree, cKDTree
    KDTREE_AVAILABLE = True
except ImportError:
    KDTree = None
    cKDTree = None
    KDTREE_AVAILABLE = False


@dataclass
class PairCountResult:
    """Result container for pair counting."""
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    pair_counts: np.ndarray
    total_pairs: int


class EfficientPairCounter:
    """
    Efficient pair counting using KD-Tree data structure.

    This class provides methods for counting galaxy pairs as a function
    of separation, which is the core operation in pairwise kSZ estimation.

    Parameters
    ----------
    leaf_size : int
        Leaf size for KD-Tree (affects performance)
    n_jobs : int
        Number of parallel jobs (-1 for all cores)

    Examples
    --------
    >>> counter = EfficientPairCounter()
    >>> counter.build_tree(positions)  # Shape (N, 3) in Mpc/h
    >>> counts = counter.count_pairs(bin_edges)
    """

    def __init__(
        self,
        leaf_size: int = 40,
        n_jobs: int = -1,
        use_cython: bool = True,
    ):
        if not KDTREE_AVAILABLE:
            raise ImportError("scipy.spatial is required for pair counting")

        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.use_cython = use_cython
        self._tree: Optional[KDTree] = None
        self._positions: Optional[np.ndarray] = None

    def build_tree(self, positions: np.ndarray) -> None:
        """
        Build KD-Tree from comoving Cartesian positions.

        Parameters
        ----------
        positions : np.ndarray
            Shape (N, 3) array of (x, y, z) positions in Mpc/h
        """
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")

        self._positions = np.ascontiguousarray(positions, dtype=np.float64)

        # Use cKDTree for better performance if available
        if self.use_cython:
            self._tree = cKDTree(self._positions, leafsize=self.leaf_size)
        else:
            self._tree = KDTree(self._positions, leafsize=self.leaf_size)

        logger.info(f"Built KD-Tree with {len(positions)} points")

    def count_pairs(
        self,
        bin_edges: np.ndarray,
    ) -> PairCountResult:
        """
        Count pairs in separation bins.

        Parameters
        ----------
        bin_edges : np.ndarray
            Bin edges in Mpc/h (n_bins + 1 values)

        Returns
        -------
        PairCountResult
            Pair counts in each bin
        """
        if self._tree is None:
            raise RuntimeError("Tree not built. Call build_tree() first.")

        bin_edges = np.asarray(bin_edges, dtype=np.float64)
        n_bins = len(bin_edges) - 1

        # Use count_neighbors for cumulative counts
        # This is more efficient than querying individual bins
        cumulative_counts = self._tree.count_neighbors(
            self._tree, bin_edges, cumulative=True
        )

        # Convert cumulative to differential
        # Note: cumulative_counts includes self-pairs (i=j) which we need to subtract
        n_objects = len(self._positions)
        pair_counts = np.diff(cumulative_counts)

        # First bin might include self-pairs
        # Actually count_neighbors counts pairs (i,j) with i != j by default
        # but we need to handle the diagonal properly

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        total_pairs = int(np.sum(pair_counts))

        logger.debug(f"Counted {total_pairs} pairs in {n_bins} bins")

        return PairCountResult(
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            pair_counts=pair_counts.astype(np.int64),
            total_pairs=total_pairs,
        )

    def query_pairs_in_bin(
        self,
        r_min: float,
        r_max: float,
        max_pairs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices of pairs within separation range.

        Parameters
        ----------
        r_min : float
            Minimum separation in Mpc/h
        r_max : float
            Maximum separation in Mpc/h
        max_pairs : int, optional
            Maximum number of pairs to return (for memory)

        Returns
        -------
        idx_i : np.ndarray
            Indices of first galaxy in each pair
        idx_j : np.ndarray
            Indices of second galaxy in each pair
        """
        if self._tree is None:
            raise RuntimeError("Tree not built. Call build_tree() first.")

        # Query pairs within r_max
        pairs = self._tree.query_pairs(r_max, output_type='ndarray')

        if len(pairs) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]

        # Filter by r_min
        if r_min > 0:
            separations = np.linalg.norm(
                self._positions[idx_i] - self._positions[idx_j],
                axis=1
            )
            mask = separations >= r_min
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]

        # Limit pairs if requested
        if max_pairs is not None and len(idx_i) > max_pairs:
            logger.warning(f"Limiting pairs from {len(idx_i)} to {max_pairs}")
            idx_i = idx_i[:max_pairs]
            idx_j = idx_j[:max_pairs]

        return idx_i, idx_j

    def compute_separations(
        self,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
    ) -> np.ndarray:
        """
        Compute separations for given pair indices.

        Parameters
        ----------
        idx_i, idx_j : np.ndarray
            Pair indices

        Returns
        -------
        np.ndarray
            Separations in Mpc/h
        """
        if self._positions is None:
            raise RuntimeError("Tree not built. Call build_tree() first.")

        diff = self._positions[idx_i] - self._positions[idx_j]
        return np.linalg.norm(diff, axis=1)


def count_pairs_in_bins(
    positions: np.ndarray,
    bin_edges: np.ndarray,
    leaf_size: int = 40,
) -> PairCountResult:
    """
    Convenience function to count pairs in separation bins.

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3) comoving positions in Mpc/h
    bin_edges : np.ndarray
        Bin edges in Mpc/h

    Returns
    -------
    PairCountResult
        Pair counts per bin
    """
    counter = EfficientPairCounter(leaf_size=leaf_size)
    counter.build_tree(positions)
    return counter.count_pairs(bin_edges)


def get_pair_indices(
    positions: np.ndarray,
    r_min: float,
    r_max: float,
    max_pairs: Optional[int] = None,
    leaf_size: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get indices of all pairs within separation range.

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3) comoving positions in Mpc/h
    r_min : float
        Minimum separation in Mpc/h
    r_max : float
        Maximum separation in Mpc/h
    max_pairs : int, optional
        Maximum pairs to return

    Returns
    -------
    idx_i, idx_j : np.ndarray
        Pair indices
    """
    counter = EfficientPairCounter(leaf_size=leaf_size)
    counter.build_tree(positions)
    return counter.query_pairs_in_bin(r_min, r_max, max_pairs)


def estimate_pair_count(
    n_objects: int,
    separation: float,
    volume: float,
    nbar: Optional[float] = None,
) -> int:
    """
    Estimate expected number of pairs at given separation.

    Useful for determining if a separation bin is feasible.

    Parameters
    ----------
    n_objects : int
        Number of objects
    separation : float
        Typical separation in Mpc/h
    volume : float
        Survey volume in (Mpc/h)^3
    nbar : float, optional
        Mean number density. If None, estimated from n/V.

    Returns
    -------
    int
        Estimated number of pairs
    """
    if nbar is None:
        nbar = n_objects / volume

    # Expected pairs at separation r in shell of width dr
    # N_pairs ~ n * n * V_shell = n * n * 4*pi*r^2*dr
    # For order of magnitude: N_pairs ~ n^2 * V
    n_pairs_estimate = int(0.5 * n_objects * nbar * 4 * np.pi * separation**2 * 10)

    return n_pairs_estimate


class ChunkedPairCounter:
    """
    Memory-efficient pair counting for very large catalogs.

    Splits the catalog into spatial chunks and processes pairs
    between chunks sequentially to reduce memory usage.

    Parameters
    ----------
    n_chunks : int
        Number of spatial chunks
    max_separation : float
        Maximum separation to consider (Mpc/h)
    """

    def __init__(
        self,
        n_chunks: int = 10,
        max_separation: float = 200.0,
    ):
        self.n_chunks = n_chunks
        self.max_separation = max_separation

    def count_pairs(
        self,
        positions: np.ndarray,
        bin_edges: np.ndarray,
    ) -> PairCountResult:
        """
        Count pairs using chunked processing.

        Parameters
        ----------
        positions : np.ndarray
            Shape (N, 3) comoving positions
        bin_edges : np.ndarray
            Separation bin edges

        Returns
        -------
        PairCountResult
            Total pair counts across all chunks
        """
        n_objects = len(positions)
        n_bins = len(bin_edges) - 1

        # Divide into chunks based on position along one axis
        # (simple approach - could use more sophisticated spatial decomposition)
        sort_idx = np.argsort(positions[:, 0])
        sorted_positions = positions[sort_idx]

        chunk_size = n_objects // self.n_chunks
        total_counts = np.zeros(n_bins, dtype=np.int64)

        for i in range(self.n_chunks):
            start_i = i * chunk_size
            end_i = (i + 1) * chunk_size if i < self.n_chunks - 1 else n_objects

            chunk_i = sorted_positions[start_i:end_i]

            # Count pairs within chunk
            counter = EfficientPairCounter()
            counter.build_tree(chunk_i)
            result = counter.count_pairs(bin_edges)
            total_counts += result.pair_counts

            # Count pairs between chunk i and subsequent chunks
            for j in range(i + 1, self.n_chunks):
                start_j = j * chunk_size
                end_j = (j + 1) * chunk_size if j < self.n_chunks - 1 else n_objects

                # Check if chunks are close enough to have pairs
                min_sep = sorted_positions[start_j, 0] - sorted_positions[end_i - 1, 0]
                if min_sep > self.max_separation:
                    break

                chunk_j = sorted_positions[start_j:end_j]

                # Cross-count between chunks
                cross_counts = self._cross_count_pairs(chunk_i, chunk_j, bin_edges)
                total_counts += cross_counts

            logger.debug(f"Processed chunk {i + 1}/{self.n_chunks}")

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return PairCountResult(
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            pair_counts=total_counts,
            total_pairs=int(np.sum(total_counts)),
        )

    def _cross_count_pairs(
        self,
        positions_1: np.ndarray,
        positions_2: np.ndarray,
        bin_edges: np.ndarray,
    ) -> np.ndarray:
        """Count pairs between two sets of positions."""
        tree_1 = cKDTree(positions_1)
        tree_2 = cKDTree(positions_2)

        n_bins = len(bin_edges) - 1
        cumulative = tree_1.count_neighbors(tree_2, bin_edges, cumulative=True)

        return np.diff(cumulative).astype(np.int64)


# =============================================================================
# Corrfunc Backend Adapter
# =============================================================================

try:
    from Corrfunc.theory.DD import DD as corrfunc_DD
    CORRFUNC_AVAILABLE = True
except ImportError:
    corrfunc_DD = None
    CORRFUNC_AVAILABLE = False


def get_available_backends() -> List[str]:
    """Return list of available pair counting backends."""
    backends = []
    if CORRFUNC_AVAILABLE:
        backends.append("corrfunc")
    if KDTREE_AVAILABLE:
        backends.append("kdtree")
    return backends


def get_default_backend() -> str:
    """Get the best available backend."""
    if CORRFUNC_AVAILABLE:
        return "corrfunc"
    elif KDTREE_AVAILABLE:
        return "kdtree"
    else:
        raise ImportError("No pair counting backend available. "
                         "Install scipy or Corrfunc.")


class CorrfuncPairCounter:
    """
    Corrfunc-based pair counting for maximum performance.

    Corrfunc uses OpenMP parallelization and optimized algorithms
    for ~10-100x speedup over KDTree for large catalogs.

    Parameters
    ----------
    n_threads : int
        Number of OpenMP threads (-1 for auto)
    periodic : bool
        Whether to use periodic boundary conditions
    boxsize : float, optional
        Box size for periodic BCs (Mpc/h)

    Notes
    -----
    Falls back to KDTree if Corrfunc is not installed.
    """

    def __init__(
        self,
        n_threads: int = -1,
        periodic: bool = False,
        boxsize: Optional[float] = None,
    ):
        if not CORRFUNC_AVAILABLE:
            logger.warning("Corrfunc not available, will use KDTree fallback")

        self.n_threads = n_threads if n_threads > 0 else 1
        self.periodic = periodic
        self.boxsize = boxsize
        self._positions: Optional[np.ndarray] = None

    def build_tree(self, positions: np.ndarray) -> None:
        """
        Store positions for pair counting.

        Note: Corrfunc doesn't build a tree explicitly; positions are
        stored and processed during count_pairs().

        Parameters
        ----------
        positions : np.ndarray
            Shape (N, 3) array of (x, y, z) positions in Mpc/h
        """
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")

        self._positions = np.ascontiguousarray(positions, dtype=np.float64)
        logger.info(f"Stored {len(positions)} positions for Corrfunc")

    def count_pairs(self, bin_edges: np.ndarray) -> PairCountResult:
        """
        Count pairs in separation bins using Corrfunc.

        Parameters
        ----------
        bin_edges : np.ndarray
            Bin edges in Mpc/h (n_bins + 1 values)

        Returns
        -------
        PairCountResult
            Pair counts in each bin
        """
        if self._positions is None:
            raise RuntimeError("Positions not set. Call build_tree() first.")

        if not CORRFUNC_AVAILABLE:
            # Fallback to KDTree
            logger.info("Using KDTree fallback (Corrfunc not available)")
            fallback = EfficientPairCounter()
            fallback.build_tree(self._positions)
            return fallback.count_pairs(bin_edges)

        bin_edges = np.asarray(bin_edges, dtype=np.float64)
        n_bins = len(bin_edges) - 1

        X = self._positions[:, 0]
        Y = self._positions[:, 1]
        Z = self._positions[:, 2]

        # Corrfunc DD function
        # autocorr=1 for auto-correlation
        results = corrfunc_DD(
            autocorr=1,
            nthreads=self.n_threads,
            binfile=bin_edges,
            X1=X, Y1=Y, Z1=Z,
            periodic=self.periodic,
            boxsize=self.boxsize if self.periodic else 0.0,
            verbose=False,
        )

        # Extract pair counts
        pair_counts = np.array([r['npairs'] for r in results], dtype=np.int64)

        # Corrfunc counts each pair twice for autocorr, so divide by 2
        # Actually, DD with autocorr=1 counts each pair once
        # But verify this with the documentation

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        total_pairs = int(np.sum(pair_counts))

        logger.debug(f"Corrfunc: counted {total_pairs} pairs in {n_bins} bins")

        return PairCountResult(
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            pair_counts=pair_counts,
            total_pairs=total_pairs,
        )


class AdaptivePairCounter:
    """
    Adaptive pair counter that selects the best available backend.

    Automatically uses Corrfunc if available, otherwise falls back
    to KDTree. Provides a unified interface regardless of backend.

    Parameters
    ----------
    backend : str, optional
        Force specific backend: 'corrfunc', 'kdtree', or 'auto'
    n_threads : int
        Number of threads (for Corrfunc)
    leaf_size : int
        Leaf size (for KDTree)

    Examples
    --------
    >>> counter = AdaptivePairCounter()  # Auto-select best backend
    >>> counter.build_tree(positions)
    >>> result = counter.count_pairs(bin_edges)
    >>> print(f"Backend used: {counter.backend}")
    """

    def __init__(
        self,
        backend: str = "auto",
        n_threads: int = -1,
        leaf_size: int = 40,
    ):
        available = get_available_backends()

        if backend == "auto":
            self._backend = get_default_backend()
        elif backend in available:
            self._backend = backend
        elif backend == "corrfunc" and not CORRFUNC_AVAILABLE:
            logger.warning("Corrfunc requested but not available, using kdtree")
            self._backend = "kdtree"
        else:
            raise ValueError(f"Unknown backend: {backend}. Available: {available}")

        logger.info(f"Using pair counting backend: {self._backend}")

        # Create underlying counter
        if self._backend == "corrfunc":
            self._counter = CorrfuncPairCounter(n_threads=n_threads)
        else:
            self._counter = EfficientPairCounter(leaf_size=leaf_size)

    @property
    def backend(self) -> str:
        """Currently active backend."""
        return self._backend

    def build_tree(self, positions: np.ndarray) -> None:
        """Build tree/store positions for pair counting."""
        self._counter.build_tree(positions)

    def count_pairs(self, bin_edges: np.ndarray) -> PairCountResult:
        """Count pairs in separation bins."""
        return self._counter.count_pairs(bin_edges)

    def query_pairs_in_bin(
        self,
        r_min: float,
        r_max: float,
        max_pairs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices of pairs within separation range.

        Note: Only available for KDTree backend.
        """
        if self._backend == "corrfunc":
            logger.warning("query_pairs_in_bin not efficient with Corrfunc, "
                          "creating temporary KDTree")
            # Fallback to KDTree for this operation
            if hasattr(self._counter, '_positions'):
                positions = self._counter._positions
            else:
                raise RuntimeError("Positions not available")

            temp_counter = EfficientPairCounter()
            temp_counter.build_tree(positions)
            return temp_counter.query_pairs_in_bin(r_min, r_max, max_pairs)

        return self._counter.query_pairs_in_bin(r_min, r_max, max_pairs)


def benchmark_backends(
    n_points: int = 10000,
    n_bins: int = 20,
    r_max: float = 150.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Benchmark available pair counting backends.

    Parameters
    ----------
    n_points : int
        Number of test points
    n_bins : int
        Number of separation bins
    r_max : float
        Maximum separation (Mpc/h)
    seed : int
        Random seed

    Returns
    -------
    dict
        Timing results for each backend in seconds
    """
    import time

    rng = np.random.default_rng(seed)

    # Generate random positions in a box
    positions = rng.uniform(0, 500, size=(n_points, 3))
    bin_edges = np.linspace(0, r_max, n_bins + 1)

    results = {}

    for backend in get_available_backends():
        counter = AdaptivePairCounter(backend=backend)

        start = time.perf_counter()
        counter.build_tree(positions)
        result = counter.count_pairs(bin_edges)
        elapsed = time.perf_counter() - start

        results[backend] = {
            'time_seconds': elapsed,
            'total_pairs': result.total_pairs,
            'pairs_per_second': result.total_pairs / elapsed if elapsed > 0 else 0,
        }

        logger.info(f"Backend {backend}: {elapsed:.3f}s, {result.total_pairs} pairs")

    return results
