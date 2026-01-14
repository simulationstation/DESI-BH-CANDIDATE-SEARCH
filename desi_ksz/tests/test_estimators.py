"""
Tests for estimator modules.

Tests pair counting, pairwise momentum estimator, and aperture photometry.
"""

import pytest
import numpy as np


class TestPairCounting:
    """Tests for pair counting functionality."""

    def test_pair_counter_creation(self, mock_positions, separation_bins):
        """Test EfficientPairCounter initialization."""
        from desi_ksz.estimators import EfficientPairCounter

        counter = EfficientPairCounter(separation_bins=separation_bins)
        assert counter is not None
        assert len(counter.separation_bins) == len(separation_bins)

    def test_pair_counting_returns_pairs(self, mock_positions, separation_bins):
        """Test that pair counting returns valid pairs."""
        from desi_ksz.estimators import EfficientPairCounter

        counter = EfficientPairCounter(separation_bins=separation_bins)
        pairs, separations, bin_indices = counter.find_pairs(mock_positions)

        # Should find some pairs
        assert len(pairs) > 0

        # Pairs should be valid indices
        assert pairs[:, 0].max() < len(mock_positions)
        assert pairs[:, 1].max() < len(mock_positions)

        # Pairs should have i < j
        assert np.all(pairs[:, 0] < pairs[:, 1])

        # Separations should be within range
        assert np.all(separations >= separation_bins[0])
        assert np.all(separations <= separation_bins[-1])

    def test_pair_counting_symmetry(self, mock_positions, separation_bins):
        """Test that pair counting is symmetric."""
        from desi_ksz.estimators import EfficientPairCounter

        counter = EfficientPairCounter(separation_bins=separation_bins)

        # Run twice
        pairs1, _, _ = counter.find_pairs(mock_positions)
        pairs2, _, _ = counter.find_pairs(mock_positions)

        # Should get same pairs
        assert np.array_equal(pairs1, pairs2)


class TestPairwiseMomentum:
    """Tests for pairwise momentum estimator."""

    def test_estimator_creation(self, separation_bins):
        """Test PairwiseMomentumEstimator initialization."""
        from desi_ksz.estimators import PairwiseMomentumEstimator

        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)
        assert estimator is not None

    def test_estimator_compute(
        self, mock_positions, mock_temperatures, mock_weights, separation_bins
    ):
        """Test pairwise momentum computation."""
        from desi_ksz.estimators import PairwiseMomentumEstimator

        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)
        result = estimator.compute(mock_positions, mock_temperatures, mock_weights)

        # Check result structure
        assert hasattr(result, 'p_ksz')
        assert hasattr(result, 'p_ksz_err')
        assert hasattr(result, 'r_centers')
        assert hasattr(result, 'n_pairs')

        # Check dimensions
        n_bins = len(separation_bins) - 1
        assert len(result.p_ksz) == n_bins
        assert len(result.r_centers) == n_bins

        # p(r) should not be all zeros (unless no pairs found)
        if np.any(result.n_pairs > 0):
            assert not np.allclose(result.p_ksz, 0)

    def test_estimator_zero_signal(self, mock_positions, mock_weights, separation_bins, rng):
        """Test that shuffled temperatures give p(r) ~ 0."""
        from desi_ksz.estimators import PairwiseMomentumEstimator

        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)

        # Run multiple shuffled realizations
        n_realizations = 10
        p_ksz_sum = np.zeros(len(separation_bins) - 1)

        for _ in range(n_realizations):
            T_shuffled = rng.standard_normal(len(mock_positions)) * 100
            result = estimator.compute(mock_positions, T_shuffled, mock_weights)
            p_ksz_sum += result.p_ksz

        p_ksz_mean = p_ksz_sum / n_realizations

        # Mean should be close to zero (within ~few sigma)
        # This is a statistical test, so use relaxed threshold
        assert np.mean(np.abs(p_ksz_mean)) < 1.0  # μK

    def test_geometric_weight(self):
        """Test geometric weight computation."""
        from desi_ksz.estimators.pairwise_momentum import compute_geometric_weight

        # Two galaxies on x-axis
        pos_i = np.array([[1.0, 0, 0]])
        pos_j = np.array([[2.0, 0, 0]])

        # Unit vectors (observer at origin)
        rhat_i = pos_i / np.linalg.norm(pos_i, axis=1, keepdims=True)
        rhat_j = pos_j / np.linalg.norm(pos_j, axis=1, keepdims=True)

        c_ij = compute_geometric_weight(pos_i, pos_j, rhat_i, rhat_j)

        # For collinear case, c_ij should be non-zero
        assert np.abs(c_ij[0]) > 0


class TestTheoryTemplate:
    """Tests for theory template computation."""

    def test_theory_template_shape(self, separation_bins):
        """Test theory template output shape."""
        from desi_ksz.estimators import compute_theory_template

        z_mean = 0.5
        r_centers = 0.5 * (separation_bins[:-1] + separation_bins[1:])

        p_theory = compute_theory_template(r_centers, z_mean)

        assert len(p_theory) == len(r_centers)

    def test_theory_template_sign(self, separation_bins):
        """Test theory template has correct sign (negative p(r))."""
        from desi_ksz.estimators import compute_theory_template

        z_mean = 0.5
        r_centers = 0.5 * (separation_bins[:-1] + separation_bins[1:])

        p_theory = compute_theory_template(r_centers, z_mean)

        # kSZ pairwise momentum should be negative
        # (galaxies moving toward each other have blueshift-redshift signal)
        assert np.mean(p_theory) < 0


class TestAperturePhotometry:
    """Tests for aperture photometry."""

    def test_aperture_photometry_creation(self):
        """Test AperturePhotometry initialization."""
        from desi_ksz.estimators import AperturePhotometry

        ap = AperturePhotometry(theta_inner=1.8, theta_outer=5.0)
        assert ap is not None
        assert ap.theta_inner == 1.8
        assert ap.theta_outer == 5.0

    def test_compensated_filter_sum_to_zero(self):
        """Test that compensated filter integrates to zero."""
        from desi_ksz.estimators import AperturePhotometry

        ap = AperturePhotometry(theta_inner=1.8, theta_outer=5.0)

        # Get filter weights
        theta_test = np.linspace(0, 10, 1000)  # arcmin
        weights = ap.get_filter_weights(theta_test)

        # Weighted sum should be close to zero for compensated filter
        # (area-weighted: multiply by theta)
        weighted_sum = np.trapz(weights * theta_test, theta_test)
        assert abs(weighted_sum) < 0.1 * np.max(np.abs(weights))
