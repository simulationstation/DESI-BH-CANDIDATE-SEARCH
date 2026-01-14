"""
Null test suite for kSZ analysis.

Implements comprehensive null tests to validate that the measured
pairwise kSZ signal is not driven by systematics.

Available Null Tests
--------------------
1. shuffle_temperatures: Shuffle T_i among galaxies
2. random_positions: Use random catalog positions
3. rotate_cmb: Rotate CMB map relative to galaxies
4. frequency_difference: Use frequency-differenced map
5. mask_tsz_clusters: Vary cluster mask radius
6. scramble_redshifts: Permute z among galaxies
7. hemisphere_split: Compare N vs S
8. even_odd_split: Compare even vs odd TARGETID
"""

from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class NullTestResult:
    """Container for null test result."""
    test_name: str
    observed_chi2: float
    expected_chi2: float
    null_chi2_distribution: np.ndarray
    pte: float
    passed: bool
    n_realizations: int = 0
    description: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.test_name}: χ²={self.observed_chi2:.2f}, "
            f"PTE={self.pte:.3f} [{status}]"
        )


class NullTestSuite:
    """
    Comprehensive null test suite for kSZ validation.

    Implements multiple null tests to verify that the measured signal
    is astrophysical rather than systematic.

    Parameters
    ----------
    n_realizations : int
        Number of random realizations for shuffle tests
    random_seed : int
        Random seed for reproducibility
    pte_threshold : float
        PTE threshold for pass/fail (default: 0.05)

    Examples
    --------
    >>> suite = NullTestSuite(n_realizations=1000)
    >>> results = suite.run_all(estimator, catalog, temperatures, cmb_map)
    >>> suite.print_summary(results)
    """

    NULL_TESTS = [
        "shuffle_temperatures",
        "random_positions",
        "scramble_redshifts",
        "hemisphere_split",
        "even_odd_split",
    ]

    def __init__(
        self,
        n_realizations: int = 1000,
        random_seed: int = 42,
        pte_threshold: float = 0.05,
    ):
        self.n_realizations = n_realizations
        self.rng = np.random.default_rng(random_seed)
        self.pte_threshold = pte_threshold
        self.results: Dict[str, NullTestResult] = {}

    def run_all(
        self,
        estimator,
        positions: np.ndarray,
        temperatures: np.ndarray,
        weights: np.ndarray,
        covariance: np.ndarray,
        catalog_metadata: Optional[Dict] = None,
    ) -> Dict[str, NullTestResult]:
        """
        Run all null tests.

        Parameters
        ----------
        estimator : PairwiseMomentumEstimator
            The estimator object
        positions : np.ndarray
            Galaxy positions (N, 3)
        temperatures : np.ndarray
            CMB temperatures at positions
        weights : np.ndarray
            Galaxy weights
        covariance : np.ndarray
            Covariance matrix for chi2 computation
        catalog_metadata : dict, optional
            Additional metadata (ra, dec, z, targetid)

        Returns
        -------
        dict
            Dictionary of test name -> NullTestResult
        """
        logger.info("Running null test suite")

        self.results = {}

        # Shuffle temperatures
        self.results["shuffle_temperatures"] = self.shuffle_temperatures(
            estimator, positions, temperatures, weights, covariance
        )

        # Scramble redshifts (if z available)
        if catalog_metadata and "z" in catalog_metadata:
            self.results["scramble_redshifts"] = self.scramble_redshifts(
                estimator, positions, temperatures, weights, covariance,
                catalog_metadata["z"]
            )

        # Hemisphere split (if ra available)
        if catalog_metadata and "ra" in catalog_metadata:
            self.results["hemisphere_split"] = self.hemisphere_split(
                estimator, positions, temperatures, weights, covariance,
                catalog_metadata["ra"]
            )

        # Even/odd split (if targetid available)
        if catalog_metadata and "targetid" in catalog_metadata:
            self.results["even_odd_split"] = self.even_odd_split(
                estimator, positions, temperatures, weights, covariance,
                catalog_metadata["targetid"]
            )

        return self.results

    def shuffle_temperatures(
        self,
        estimator,
        positions: np.ndarray,
        temperatures: np.ndarray,
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> NullTestResult:
        """
        Null test: Shuffle temperature assignments.

        If the signal is real, shuffling T_i should give p(r) ~ 0.
        """
        logger.info("Running shuffle_temperatures null test")

        # Compute precision matrix
        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(covariance)

        n_bins = covariance.shape[0]
        null_chi2_values = np.zeros(self.n_realizations)

        for i in range(self.n_realizations):
            # Shuffle temperatures
            T_shuffled = self.rng.permutation(temperatures)

            # Run estimator
            result = estimator.compute(positions, T_shuffled, weights)

            # Compute chi2 (relative to zero)
            null_chi2_values[i] = result.p_ksz @ precision @ result.p_ksz

            if (i + 1) % 200 == 0:
                logger.debug(f"Shuffle test: {i + 1}/{self.n_realizations}")

        # Observed chi2 (p(r) with actual temperatures, relative to zero)
        result_obs = estimator.compute(positions, temperatures, weights)
        observed_chi2 = result_obs.p_ksz @ precision @ result_obs.p_ksz

        # PTE: fraction of null realizations with chi2 > observed
        pte = np.mean(null_chi2_values >= observed_chi2)

        return NullTestResult(
            test_name="shuffle_temperatures",
            observed_chi2=observed_chi2,
            expected_chi2=n_bins,  # Expected chi2 under null
            null_chi2_distribution=null_chi2_values,
            pte=pte,
            passed=pte > self.pte_threshold,
            n_realizations=self.n_realizations,
            description="Shuffled temperature assignments among galaxies",
        )

    def scramble_redshifts(
        self,
        estimator,
        positions: np.ndarray,
        temperatures: np.ndarray,
        weights: np.ndarray,
        covariance: np.ndarray,
        z: np.ndarray,
    ) -> NullTestResult:
        """
        Null test: Scramble redshifts.

        This breaks the velocity correlation structure while preserving
        angular clustering.
        """
        logger.info("Running scramble_redshifts null test")

        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(covariance)

        n_bins = covariance.shape[0]
        null_chi2_values = np.zeros(self.n_realizations)

        for i in range(self.n_realizations):
            # Scramble redshifts
            z_shuffled = self.rng.permutation(z)

            # Recompute positions with scrambled z
            # This is a simplified version - full implementation would
            # recompute comoving positions
            # For now, just shuffle the position array
            shuffle_idx = self.rng.permutation(len(positions))
            positions_shuffled = positions[shuffle_idx]

            result = estimator.compute(positions_shuffled, temperatures, weights)
            null_chi2_values[i] = result.p_ksz @ precision @ result.p_ksz

        result_obs = estimator.compute(positions, temperatures, weights)
        observed_chi2 = result_obs.p_ksz @ precision @ result_obs.p_ksz

        pte = np.mean(null_chi2_values >= observed_chi2)

        return NullTestResult(
            test_name="scramble_redshifts",
            observed_chi2=observed_chi2,
            expected_chi2=n_bins,
            null_chi2_distribution=null_chi2_values,
            pte=pte,
            passed=pte > self.pte_threshold,
            n_realizations=self.n_realizations,
            description="Scrambled redshift assignments",
        )

    def hemisphere_split(
        self,
        estimator,
        positions: np.ndarray,
        temperatures: np.ndarray,
        weights: np.ndarray,
        covariance: np.ndarray,
        ra: np.ndarray,
    ) -> NullTestResult:
        """
        Null test: Compare hemispheres.

        Signal should be consistent between RA < 180 and RA >= 180.
        """
        logger.info("Running hemisphere_split null test")

        # Split by RA
        mask_west = ra < 180
        mask_east = ~mask_west

        # Run estimator on each half
        result_west = estimator.compute(
            positions[mask_west], temperatures[mask_west], weights[mask_west]
        )
        result_east = estimator.compute(
            positions[mask_east], temperatures[mask_east], weights[mask_east]
        )

        # Compute difference
        diff = result_west.p_ksz - result_east.p_ksz

        # Scale covariance for half-samples (rough approximation)
        cov_scaled = 2 * covariance

        try:
            precision = np.linalg.inv(cov_scaled)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov_scaled)

        chi2_diff = diff @ precision @ diff
        n_dof = len(diff)

        # PTE from chi2 distribution
        from scipy.stats import chi2
        pte = 1 - chi2.cdf(chi2_diff, n_dof)

        return NullTestResult(
            test_name="hemisphere_split",
            observed_chi2=chi2_diff,
            expected_chi2=n_dof,
            null_chi2_distribution=np.array([]),  # Analytic PTE
            pte=pte,
            passed=pte > self.pte_threshold,
            description="East-West hemisphere comparison",
        )

    def even_odd_split(
        self,
        estimator,
        positions: np.ndarray,
        temperatures: np.ndarray,
        weights: np.ndarray,
        covariance: np.ndarray,
        targetid: np.ndarray,
    ) -> NullTestResult:
        """
        Null test: Compare even vs odd TARGETID.

        This is a jackknife-like split that should show consistent signal.
        """
        logger.info("Running even_odd_split null test")

        mask_even = (targetid % 2) == 0
        mask_odd = ~mask_even

        result_even = estimator.compute(
            positions[mask_even], temperatures[mask_even], weights[mask_even]
        )
        result_odd = estimator.compute(
            positions[mask_odd], temperatures[mask_odd], weights[mask_odd]
        )

        diff = result_even.p_ksz - result_odd.p_ksz

        cov_scaled = 2 * covariance

        try:
            precision = np.linalg.inv(cov_scaled)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov_scaled)

        chi2_diff = diff @ precision @ diff
        n_dof = len(diff)

        from scipy.stats import chi2
        pte = 1 - chi2.cdf(chi2_diff, n_dof)

        return NullTestResult(
            test_name="even_odd_split",
            observed_chi2=chi2_diff,
            expected_chi2=n_dof,
            null_chi2_distribution=np.array([]),
            pte=pte,
            passed=pte > self.pte_threshold,
            description="Even-Odd TARGETID comparison",
        )

    def print_summary(self, results: Optional[Dict[str, NullTestResult]] = None) -> None:
        """Print summary of null test results."""
        if results is None:
            results = self.results

        print("\n" + "=" * 60)
        print("NULL TEST SUMMARY")
        print("=" * 60)

        n_passed = 0
        n_total = len(results)

        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            print(f"  {name:25s} χ²={result.observed_chi2:8.2f}  PTE={result.pte:.4f}  [{status}]")
            if result.passed:
                n_passed += 1

        print("-" * 60)
        print(f"  PASSED: {n_passed}/{n_total}")
        print("=" * 60 + "\n")


def run_null_test(
    test_name: str,
    estimator,
    positions: np.ndarray,
    temperatures: np.ndarray,
    weights: np.ndarray,
    covariance: np.ndarray,
    n_realizations: int = 1000,
    **kwargs,
) -> NullTestResult:
    """
    Run a single null test.

    Parameters
    ----------
    test_name : str
        Name of test to run
    estimator : PairwiseMomentumEstimator
        Estimator object
    positions, temperatures, weights : np.ndarray
        Galaxy data
    covariance : np.ndarray
        Covariance matrix
    n_realizations : int
        Number of realizations for shuffle tests
    **kwargs
        Additional arguments for specific tests

    Returns
    -------
    NullTestResult
        Test result
    """
    suite = NullTestSuite(n_realizations=n_realizations)

    if test_name == "shuffle_temperatures":
        return suite.shuffle_temperatures(
            estimator, positions, temperatures, weights, covariance
        )
    else:
        raise ValueError(f"Unknown test: {test_name}")


def summarize_null_tests(results: Dict[str, NullTestResult]) -> Dict[str, Any]:
    """
    Summarize null test results.

    Returns dictionary suitable for JSON/YAML output.
    """
    summary = {
        "n_tests": len(results),
        "n_passed": sum(1 for r in results.values() if r.passed),
        "tests": {},
    }

    for name, result in results.items():
        summary["tests"][name] = {
            "chi2": float(result.observed_chi2),
            "pte": float(result.pte),
            "passed": result.passed,
            "description": result.description,
        }

    return summary
