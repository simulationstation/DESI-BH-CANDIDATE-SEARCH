"""
Pairwise kSZ momentum estimator.

This module implements the core pairwise kSZ momentum estimator
following Ferreira et al. (1999) and Hand et al. (2012).

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| p(r)        | Pairwise kSZ momentum                           | muK         |
| T_i         | CMB temperature at galaxy i position            | muK         |
| c_ij        | Geometric weight for pair (i,j)                 | dimensionless|
| r̂_i         | Unit vector from observer to galaxy i           | dimensionless|
| r̂_ij        | Unit vector along pair separation               | dimensionless|
| w_i         | Galaxy weight                                   | dimensionless|
| r_ij        | Comoving pair separation                        | Mpc/h       |

Pairwise Momentum Estimator
---------------------------
The estimator computes the mean temperature difference weighted by
the geometric projection factor:

    p̂(r) = Σ_{ij ∈ bin(r)} w_ij (T_i - T_j) c_ij
           ----------------------------------------
           Σ_{ij ∈ bin(r)} w_ij c_ij²

where:
    - w_ij = w_i × w_j is the pair weight
    - c_ij = ½ r̂_ij · (r̂_i - r̂_j) is the geometric weight
    - r̂_i = x_i / |x_i| is the unit vector to galaxy i
    - r̂_ij = (x_j - x_i) / |x_j - x_i| is the pair separation unit vector

The geometric weight c_ij projects the line-of-sight velocity
difference onto the pair separation direction.

References
----------
- Ferreira, P. G., et al. 1999, ApJ, 515, L1
- Hand, N., et al. 2012, PRL, 109, 041101
- Schaan, E., et al. 2021, PRD, 103, 063513
"""

from typing import Optional, Tuple, List, Union
from dataclasses import dataclass, field
import numpy as np
import logging

from .pair_counting import EfficientPairCounter

logger = logging.getLogger(__name__)


@dataclass
class PairwiseMomentumResult:
    """
    Container for pairwise momentum measurement results.

    Attributes
    ----------
    bin_centers : np.ndarray
        Separation bin centers in Mpc/h
    bin_edges : np.ndarray
        Separation bin edges in Mpc/h
    p_ksz : np.ndarray
        Pairwise kSZ momentum in muK
    p_ksz_err : np.ndarray
        Statistical uncertainty on p_ksz (if computed)
    pair_counts : np.ndarray
        Number of pairs in each bin
    sum_weights : np.ndarray
        Sum of w_ij × c_ij² in each bin
    sum_weights_c : np.ndarray
        Sum of w_ij × c_ij in each bin
    z_bin : Tuple[float, float]
        Redshift range (z_min, z_max)
    tracer : str
        Galaxy tracer type
    n_galaxies : int
        Number of galaxies used
    """
    bin_centers: np.ndarray
    bin_edges: np.ndarray
    p_ksz: np.ndarray
    p_ksz_err: np.ndarray = field(default_factory=lambda: np.array([]))
    pair_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    sum_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    sum_weights_c: np.ndarray = field(default_factory=lambda: np.array([]))
    z_bin: Tuple[float, float] = (0.0, 10.0)
    tracer: str = "unknown"
    n_galaxies: int = 0

    @property
    def n_bins(self) -> int:
        return len(self.p_ksz)

    @property
    def total_pairs(self) -> int:
        return int(np.sum(self.pair_counts))

    @property
    def detection_snr(self) -> float:
        """Signal-to-noise ratio (requires p_ksz_err)."""
        if len(self.p_ksz_err) == 0 or np.all(self.p_ksz_err == 0):
            return np.nan
        # Simple SNR: sum of signal / sqrt(sum of variance)
        signal = np.sum(np.abs(self.p_ksz))
        noise = np.sqrt(np.sum(self.p_ksz_err**2))
        return signal / noise if noise > 0 else np.nan

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bin_centers": self.bin_centers.tolist(),
            "bin_edges": self.bin_edges.tolist(),
            "p_ksz": self.p_ksz.tolist(),
            "p_ksz_err": self.p_ksz_err.tolist() if len(self.p_ksz_err) else [],
            "pair_counts": self.pair_counts.tolist(),
            "z_bin": list(self.z_bin),
            "tracer": self.tracer,
            "n_galaxies": self.n_galaxies,
            "total_pairs": self.total_pairs,
        }


class PairwiseMomentumEstimator:
    """
    Pairwise kSZ momentum estimator.

    This class implements the pairwise momentum estimator for measuring
    the kinetic Sunyaev-Zel'dovich effect from galaxy-CMB cross-correlations.

    Parameters
    ----------
    separation_bins : np.ndarray
        Bin edges for comoving separation in Mpc/h
    max_pairs_per_bin : int, optional
        Maximum pairs per bin (for memory management)
    min_pairs_per_bin : int
        Minimum pairs required for valid estimate

    Examples
    --------
    >>> estimator = PairwiseMomentumEstimator(
    ...     separation_bins=np.linspace(5, 150, 15)
    ... )
    >>> result = estimator.compute(
    ...     positions=catalog.positions,
    ...     temperatures=T_cmb,
    ...     weights=catalog.weights,
    ... )
    """

    def __init__(
        self,
        separation_bins: np.ndarray,
        max_pairs_per_bin: Optional[int] = None,
        min_pairs_per_bin: int = 100,
    ):
        self.separation_bins = np.asarray(separation_bins, dtype=np.float64)
        self.max_pairs_per_bin = max_pairs_per_bin
        self.min_pairs_per_bin = min_pairs_per_bin

        self.n_bins = len(self.separation_bins) - 1
        self.bin_centers = 0.5 * (
            self.separation_bins[:-1] + self.separation_bins[1:]
        )

    def compute(
        self,
        positions: np.ndarray,
        temperatures: np.ndarray,
        weights: Optional[np.ndarray] = None,
        z_bin: Tuple[float, float] = (0.0, 10.0),
        tracer: str = "unknown",
    ) -> PairwiseMomentumResult:
        """
        Compute pairwise kSZ momentum.

        Parameters
        ----------
        positions : np.ndarray
            Shape (N, 3) comoving Cartesian positions in Mpc/h
        temperatures : np.ndarray
            CMB temperatures at galaxy positions in muK
        weights : np.ndarray, optional
            Galaxy weights (default: uniform)
        z_bin : tuple
            Redshift range for labeling
        tracer : str
            Galaxy tracer type for labeling

        Returns
        -------
        PairwiseMomentumResult
            Pairwise momentum measurement
        """
        n_galaxies = len(positions)

        if len(temperatures) != n_galaxies:
            raise ValueError(
                f"temperatures length ({len(temperatures)}) != "
                f"positions length ({n_galaxies})"
            )

        if weights is None:
            weights = np.ones(n_galaxies)

        # Remove invalid data
        valid = np.isfinite(temperatures) & np.isfinite(weights)
        valid &= weights > 0

        if np.sum(valid) < n_galaxies:
            logger.warning(
                f"Removing {n_galaxies - np.sum(valid)} galaxies with invalid data"
            )
            positions = positions[valid]
            temperatures = temperatures[valid]
            weights = weights[valid]
            n_galaxies = len(positions)

        logger.info(f"Computing pairwise momentum for {n_galaxies} galaxies")

        # Compute unit vectors from observer (assumes observer at origin)
        r_hat = positions / np.linalg.norm(positions, axis=1, keepdims=True)

        # Build KD-tree for pair finding
        pair_counter = EfficientPairCounter()
        pair_counter.build_tree(positions)

        # Initialize accumulators
        numerator = np.zeros(self.n_bins)
        denominator = np.zeros(self.n_bins)
        pair_counts = np.zeros(self.n_bins, dtype=np.int64)

        # Process each separation bin
        for b in range(self.n_bins):
            r_min = self.separation_bins[b]
            r_max = self.separation_bins[b + 1]

            # Get pairs in this bin
            idx_i, idx_j = pair_counter.query_pairs_in_bin(
                r_min, r_max, max_pairs=self.max_pairs_per_bin
            )

            n_pairs = len(idx_i)
            pair_counts[b] = n_pairs

            if n_pairs < self.min_pairs_per_bin:
                logger.warning(
                    f"Bin {b} (r={self.bin_centers[b]:.1f} Mpc/h): "
                    f"only {n_pairs} pairs < {self.min_pairs_per_bin} minimum"
                )
                continue

            # Compute geometric weights for these pairs
            c_ij = compute_geometric_weight(
                positions[idx_i],
                positions[idx_j],
                r_hat[idx_i],
                r_hat[idx_j],
            )

            # Pair weights
            w_ij = weights[idx_i] * weights[idx_j]

            # Temperature differences
            dT = temperatures[idx_i] - temperatures[idx_j]

            # Accumulate estimator
            # p(r) = sum(w_ij * dT * c_ij) / sum(w_ij * c_ij^2)
            numerator[b] = np.sum(w_ij * dT * c_ij)
            denominator[b] = np.sum(w_ij * c_ij**2)

            logger.debug(
                f"Bin {b}: r={self.bin_centers[b]:.1f} Mpc/h, "
                f"N_pairs={n_pairs}, num={numerator[b]:.2e}, den={denominator[b]:.2e}"
            )

        # Compute p(r)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_ksz = numerator / denominator
            p_ksz[~np.isfinite(p_ksz)] = 0.0

        total_pairs = np.sum(pair_counts)
        logger.info(f"Total pairs: {total_pairs:,}")

        return PairwiseMomentumResult(
            bin_centers=self.bin_centers,
            bin_edges=self.separation_bins,
            p_ksz=p_ksz,
            pair_counts=pair_counts,
            sum_weights=denominator,
            sum_weights_c=numerator / np.where(p_ksz != 0, p_ksz, 1),
            z_bin=z_bin,
            tracer=tracer,
            n_galaxies=n_galaxies,
        )

    def compute_with_cross_pairs(
        self,
        positions_1: np.ndarray,
        temperatures_1: np.ndarray,
        weights_1: np.ndarray,
        positions_2: np.ndarray,
        temperatures_2: np.ndarray,
        weights_2: np.ndarray,
    ) -> PairwiseMomentumResult:
        """
        Compute pairwise momentum using cross-pairs between two samples.

        Useful for cross-bin measurements in tomography.

        Parameters
        ----------
        positions_1, temperatures_1, weights_1 : np.ndarray
            First galaxy sample
        positions_2, temperatures_2, weights_2 : np.ndarray
            Second galaxy sample

        Returns
        -------
        PairwiseMomentumResult
            Cross-pair momentum measurement
        """
        # Concatenate samples with labels
        n1 = len(positions_1)
        n2 = len(positions_2)

        positions = np.vstack([positions_1, positions_2])
        temperatures = np.concatenate([temperatures_1, temperatures_2])
        weights = np.concatenate([weights_1, weights_2])
        sample_id = np.concatenate([np.zeros(n1), np.ones(n2)])

        # Compute unit vectors
        r_hat = positions / np.linalg.norm(positions, axis=1, keepdims=True)

        # Build tree
        pair_counter = EfficientPairCounter()
        pair_counter.build_tree(positions)

        numerator = np.zeros(self.n_bins)
        denominator = np.zeros(self.n_bins)
        pair_counts = np.zeros(self.n_bins, dtype=np.int64)

        for b in range(self.n_bins):
            r_min = self.separation_bins[b]
            r_max = self.separation_bins[b + 1]

            idx_i, idx_j = pair_counter.query_pairs_in_bin(r_min, r_max)

            # Keep only cross-pairs (one from each sample)
            cross_mask = sample_id[idx_i] != sample_id[idx_j]
            idx_i = idx_i[cross_mask]
            idx_j = idx_j[cross_mask]

            n_pairs = len(idx_i)
            pair_counts[b] = n_pairs

            if n_pairs < self.min_pairs_per_bin:
                continue

            c_ij = compute_geometric_weight(
                positions[idx_i], positions[idx_j],
                r_hat[idx_i], r_hat[idx_j],
            )
            w_ij = weights[idx_i] * weights[idx_j]
            dT = temperatures[idx_i] - temperatures[idx_j]

            numerator[b] = np.sum(w_ij * dT * c_ij)
            denominator[b] = np.sum(w_ij * c_ij**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            p_ksz = numerator / denominator
            p_ksz[~np.isfinite(p_ksz)] = 0.0

        return PairwiseMomentumResult(
            bin_centers=self.bin_centers,
            bin_edges=self.separation_bins,
            p_ksz=p_ksz,
            pair_counts=pair_counts,
            sum_weights=denominator,
            n_galaxies=n1 + n2,
        )


def compute_geometric_weight(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    rhat_i: np.ndarray,
    rhat_j: np.ndarray,
) -> np.ndarray:
    """
    Compute geometric weight c_ij for galaxy pairs.

    The geometric weight projects the velocity difference onto the
    pair separation direction:

        c_ij = ½ r̂_ij · (r̂_i - r̂_j)

    where:
        r̂_ij = (x_j - x_i) / |x_j - x_i| is the pair separation unit vector
        r̂_i = x_i / |x_i| is the unit vector to galaxy i

    Parameters
    ----------
    pos_i : np.ndarray
        Shape (N_pairs, 3) positions of first galaxy in each pair
    pos_j : np.ndarray
        Shape (N_pairs, 3) positions of second galaxy in each pair
    rhat_i : np.ndarray
        Shape (N_pairs, 3) unit vectors to first galaxy
    rhat_j : np.ndarray
        Shape (N_pairs, 3) unit vectors to second galaxy

    Returns
    -------
    np.ndarray
        Shape (N_pairs,) geometric weights

    Notes
    -----
    The geometric weight is related to the cosine of the angle between
    the pair separation and the line-of-sight:
    - c_ij ~ 0 for pairs perpendicular to line of sight
    - c_ij ~ ±1 for pairs along line of sight
    """
    # Pair separation vector
    sep = pos_j - pos_i
    sep_norm = np.linalg.norm(sep, axis=1, keepdims=True)

    # Handle zero separation (shouldn't happen but be safe)
    sep_norm = np.maximum(sep_norm, 1e-10)

    # Unit vector along separation
    rhat_ij = sep / sep_norm

    # Geometric weight: c_ij = 0.5 * r̂_ij · (r̂_i - r̂_j)
    los_diff = rhat_i - rhat_j
    c_ij = 0.5 * np.sum(rhat_ij * los_diff, axis=1)

    return c_ij


def combine_results(
    results: List[PairwiseMomentumResult],
    method: str = "inverse_variance",
) -> PairwiseMomentumResult:
    """
    Combine multiple pairwise momentum measurements.

    Parameters
    ----------
    results : List[PairwiseMomentumResult]
        List of results to combine
    method : str
        Combination method: 'inverse_variance', 'simple_average', 'weighted_pairs'

    Returns
    -------
    PairwiseMomentumResult
        Combined result
    """
    if len(results) == 0:
        raise ValueError("No results to combine")

    if len(results) == 1:
        return results[0]

    # Check consistent binning
    ref_bins = results[0].bin_edges
    for r in results[1:]:
        if not np.allclose(r.bin_edges, ref_bins):
            raise ValueError("Results have inconsistent binning")

    n_bins = len(ref_bins) - 1
    bin_centers = results[0].bin_centers

    if method == "simple_average":
        p_ksz = np.mean([r.p_ksz for r in results], axis=0)
        pair_counts = np.sum([r.pair_counts for r in results], axis=0)

    elif method == "weighted_pairs":
        # Weight by number of pairs
        total_pairs = np.sum([r.pair_counts for r in results], axis=0)
        p_ksz = np.zeros(n_bins)
        for r in results:
            weight = r.pair_counts / np.maximum(total_pairs, 1)
            p_ksz += weight * r.p_ksz
        pair_counts = total_pairs

    elif method == "inverse_variance":
        # Weight by inverse variance (requires errors)
        # Fall back to weighted_pairs if no errors
        has_errors = all(len(r.p_ksz_err) > 0 for r in results)

        if has_errors:
            var = np.array([r.p_ksz_err**2 for r in results])
            weights = 1.0 / np.maximum(var, 1e-20)
            p_ksz = np.sum(weights * np.array([r.p_ksz for r in results]), axis=0)
            p_ksz /= np.sum(weights, axis=0)
        else:
            return combine_results(results, method="weighted_pairs")

        pair_counts = np.sum([r.pair_counts for r in results], axis=0)

    else:
        raise ValueError(f"Unknown combination method: {method}")

    total_galaxies = sum(r.n_galaxies for r in results)

    return PairwiseMomentumResult(
        bin_centers=bin_centers,
        bin_edges=ref_bins,
        p_ksz=p_ksz,
        pair_counts=pair_counts,
        n_galaxies=total_galaxies,
        tracer="combined",
    )
