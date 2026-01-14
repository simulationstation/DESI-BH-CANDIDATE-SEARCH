"""
Spatial jackknife resampling for covariance estimation.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| C_jk        | Jackknife covariance matrix                     | varies      |
| N_jk        | Number of jackknife regions                     | dimensionless|
| x_k         | Estimator with region k removed                 | varies      |
| x̄           | Mean of jackknife estimates                     | varies      |

Jackknife Covariance
--------------------
The jackknife covariance estimate is:

    Ĉ_ab = (N_jk - 1) / N_jk × Σ_k (x_a^{(k)} - x̄_a)(x_b^{(k)} - x̄_b)

where x^{(k)} is the estimator computed with jackknife region k removed.
"""

from typing import Optional, Callable, List, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)

# Number of parallel workers
N_WORKERS = max(1, cpu_count() - 1)

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    hp = None
    HEALPY_AVAILABLE = False


@dataclass
class JackknifeResult:
    """Container for jackknife covariance results."""
    covariance: np.ndarray
    correlation: np.ndarray
    estimates: np.ndarray  # Shape (n_jk, n_bins)
    mean_estimate: np.ndarray
    std_estimate: np.ndarray
    region_counts: np.ndarray
    n_regions: int


class SpatialJackknife:
    """
    Spatial jackknife resampling for covariance estimation.

    Divides the survey footprint into N spatial regions and recomputes
    the estimator leaving each region out in turn.

    Parameters
    ----------
    n_regions : int
        Number of jackknife regions
    method : str
        Region definition method: 'healpix', 'kmeans'
    random_seed : int
        Random seed for reproducibility

    Examples
    --------
    >>> jk = SpatialJackknife(n_regions=100)
    >>> jk.define_regions(ra, dec)
    >>> result = jk.compute_covariance(estimator_func, catalog, temperatures)
    """

    def __init__(
        self,
        n_regions: int = 100,
        method: str = "healpix",
        random_seed: int = 42,
    ):
        self.n_regions = n_regions
        self.method = method
        self.random_seed = random_seed
        self.region_assignments: Optional[np.ndarray] = None
        self.region_centers: Optional[np.ndarray] = None

    def define_regions(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        nside: int = 16,
    ) -> np.ndarray:
        """
        Assign galaxies to jackknife regions.

        Parameters
        ----------
        ra : np.ndarray
            Right Ascension in degrees
        dec : np.ndarray
            Declination in degrees
        nside : int
            HEALPix nside for healpix method

        Returns
        -------
        np.ndarray
            Region assignment for each galaxy (0 to n_regions-1)
        """
        if self.method == "healpix":
            self.region_assignments = self._define_regions_healpix(ra, dec, nside)
        elif self.method == "kmeans":
            self.region_assignments = self._define_regions_kmeans(ra, dec)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Count galaxies per region
        unique, counts = np.unique(self.region_assignments, return_counts=True)
        self.region_counts = np.zeros(self.n_regions, dtype=int)
        for u, c in zip(unique, counts):
            if 0 <= u < self.n_regions:
                self.region_counts[u] = c

        logger.info(
            f"Defined {self.n_regions} jackknife regions "
            f"(median {np.median(self.region_counts):.0f} galaxies per region)"
        )

        return self.region_assignments

    def _define_regions_healpix(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        nside: int,
    ) -> np.ndarray:
        """Define regions using HEALPix pixels."""
        if not HEALPY_AVAILABLE:
            raise ImportError("healpy required for healpix method")

        # Convert to theta, phi
        theta = np.deg2rad(90.0 - dec)
        phi = np.deg2rad(ra)

        # Get HEALPix pixel indices
        pix = hp.ang2pix(nside, theta, phi)

        # Map HEALPix pixels to jackknife regions
        unique_pix = np.unique(pix)
        n_healpix = len(unique_pix)

        # Distribute HEALPix pixels among jackknife regions
        regions_per_jk = max(1, n_healpix // self.n_regions)

        pix_to_region = {}
        for i, p in enumerate(unique_pix):
            pix_to_region[p] = min(i // regions_per_jk, self.n_regions - 1)

        regions = np.array([pix_to_region.get(p, 0) for p in pix])

        return regions

    def _define_regions_kmeans(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
    ) -> np.ndarray:
        """Define regions using k-means clustering on the sphere."""
        from sklearn.cluster import KMeans

        # Convert to Cartesian on unit sphere
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)

        coords = np.column_stack([x, y, z])

        # K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_regions,
            random_state=self.random_seed,
            n_init=10,
        )
        regions = kmeans.fit_predict(coords)

        self.region_centers = kmeans.cluster_centers_

        return regions

    def compute_covariance(
        self,
        estimator_func: Callable,
        *args,
        n_workers: Optional[int] = None,
        **kwargs,
    ) -> JackknifeResult:
        """
        Compute jackknife covariance matrix (parallelized).

        Parameters
        ----------
        estimator_func : callable
            Function that takes (mask, *args, **kwargs) and returns 1D array
            The mask indicates which galaxies to include
        *args
            Additional positional arguments passed to estimator_func
        n_workers : int, optional
            Number of parallel workers (default: cpu_count - 1)
        **kwargs
            Additional keyword arguments passed to estimator_func

        Returns
        -------
        JackknifeResult
            Jackknife covariance and related quantities
        """
        if self.region_assignments is None:
            raise RuntimeError("Regions not defined. Call define_regions() first.")

        if n_workers is None:
            n_workers = N_WORKERS

        n_galaxies = len(self.region_assignments)

        # Create masks for all regions upfront
        masks = [self.region_assignments != k for k in range(self.n_regions)]

        # Define worker function for this specific estimator
        def _jackknife_worker(k: int) -> np.ndarray:
            return estimator_func(masks[k], *args, **kwargs)

        # Run in parallel or sequential
        if n_workers > 1 and self.n_regions > 1:
            logger.info(f"Running jackknife with {n_workers} workers")
            with Pool(n_workers) as pool:
                estimates = pool.map(_jackknife_worker, range(self.n_regions))
        else:
            estimates = []
            for k in range(self.n_regions):
                estimate_k = _jackknife_worker(k)
                estimates.append(estimate_k)
                if (k + 1) % 20 == 0:
                    logger.debug(f"Jackknife: completed {k + 1}/{self.n_regions}")

        estimates = np.array(estimates)
        n_bins = estimates.shape[1]

        # Jackknife mean and covariance
        mean_estimate = np.mean(estimates, axis=0)

        # Jackknife covariance formula
        delta = estimates - mean_estimate
        covariance = (self.n_regions - 1) / self.n_regions * (delta.T @ delta)

        # Correlation matrix
        std = np.sqrt(np.diag(covariance))
        std_outer = np.outer(std, std)
        std_outer = np.maximum(std_outer, 1e-20)
        correlation = covariance / std_outer

        return JackknifeResult(
            covariance=covariance,
            correlation=correlation,
            estimates=estimates,
            mean_estimate=mean_estimate,
            std_estimate=std,
            region_counts=self.region_counts,
            n_regions=self.n_regions,
        )


def compute_jackknife_covariance(
    data_vector: np.ndarray,
    jackknife_estimates: np.ndarray,
) -> np.ndarray:
    """
    Compute jackknife covariance from pre-computed estimates.

    Parameters
    ----------
    data_vector : np.ndarray
        Full data vector (shape n_bins)
    jackknife_estimates : np.ndarray
        Jackknife estimates (shape n_jk, n_bins)

    Returns
    -------
    np.ndarray
        Covariance matrix (shape n_bins, n_bins)
    """
    n_jk = jackknife_estimates.shape[0]
    mean = np.mean(jackknife_estimates, axis=0)
    delta = jackknife_estimates - mean

    covariance = (n_jk - 1) / n_jk * (delta.T @ delta)

    return covariance


def define_jackknife_regions(
    ra: np.ndarray,
    dec: np.ndarray,
    n_regions: int = 100,
    method: str = "healpix",
) -> np.ndarray:
    """
    Convenience function to define jackknife regions.

    Parameters
    ----------
    ra, dec : np.ndarray
        Galaxy coordinates in degrees
    n_regions : int
        Number of regions
    method : str
        Method: 'healpix' or 'kmeans'

    Returns
    -------
    np.ndarray
        Region assignments
    """
    jk = SpatialJackknife(n_regions=n_regions, method=method)
    return jk.define_regions(ra, dec)
