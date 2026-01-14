"""
Tests for covariance estimation modules.

Tests jackknife, Hartlap correction, and mock interface.
"""

import pytest
import numpy as np


class TestJackknife:
    """Tests for jackknife covariance estimation."""

    def test_jackknife_creation(self, mock_positions):
        """Test SpatialJackknife initialization."""
        from desi_ksz.covariance import SpatialJackknife

        jk = SpatialJackknife(n_regions=10, method='healpix')
        assert jk is not None
        assert jk.n_regions == 10

    def test_jackknife_region_assignment(self, mock_catalog):
        """Test that jackknife assigns all galaxies to regions."""
        from desi_ksz.covariance import SpatialJackknife

        jk = SpatialJackknife(n_regions=10, method='healpix')
        regions = jk.assign_regions(mock_catalog['ra'], mock_catalog['dec'])

        # All galaxies should have a region
        assert len(regions) == len(mock_catalog['ra'])

        # Region indices should be valid
        assert regions.min() >= 0
        assert regions.max() < jk.n_regions

    def test_jackknife_covariance_shape(self, mock_pksz_data, mock_catalog):
        """Test that jackknife covariance has correct shape."""
        from desi_ksz.covariance import SpatialJackknife

        n_bins = len(mock_pksz_data['r_centers'])
        jk = SpatialJackknife(n_regions=10)

        # Generate mock jackknife samples
        jk_samples = np.random.randn(10, n_bins)

        cov = jk.compute_covariance(jk_samples)

        assert cov.shape == (n_bins, n_bins)

    def test_jackknife_covariance_positive_definite(self):
        """Test that jackknife covariance is positive semi-definite."""
        from desi_ksz.covariance import SpatialJackknife

        n_bins = 10
        n_regions = 20

        jk = SpatialJackknife(n_regions=n_regions)

        # Generate mock samples with some correlation
        rng = np.random.default_rng(42)
        jk_samples = rng.standard_normal((n_regions, n_bins))

        # Add correlation between neighboring bins
        for i in range(1, n_bins):
            jk_samples[:, i] += 0.3 * jk_samples[:, i - 1]

        cov = jk.compute_covariance(jk_samples)

        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    def test_jackknife_formula(self):
        """Test jackknife covariance formula: C = (N-1)/N * sum[(x-xbar)(x-xbar)^T]"""
        from desi_ksz.covariance import SpatialJackknife

        n_regions = 5
        n_bins = 3

        jk = SpatialJackknife(n_regions=n_regions)

        # Simple test data
        samples = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 2.9],
            [0.9, 1.9, 3.1],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ])

        cov = jk.compute_covariance(samples)

        # Manual calculation
        mean = samples.mean(axis=0)
        diff = samples - mean
        cov_manual = (n_regions - 1) / n_regions * (diff.T @ diff)

        np.testing.assert_allclose(cov, cov_manual, rtol=1e-10)


class TestHartlap:
    """Tests for Hartlap correction."""

    def test_hartlap_factor(self):
        """Test Hartlap correction factor computation."""
        from desi_ksz.covariance import compute_hartlap_factor

        n_sims = 100
        n_data = 10

        alpha = compute_hartlap_factor(n_sims, n_data)

        # Should be < 1
        assert 0 < alpha < 1

        # Check formula: (N_s - N_d - 2) / (N_s - 1)
        expected = (n_sims - n_data - 2) / (n_sims - 1)
        assert np.isclose(alpha, expected)

    def test_hartlap_correction_reduces_precision(self):
        """Test that Hartlap correction reduces precision matrix."""
        from desi_ksz.covariance import apply_hartlap_correction

        # Create test covariance
        n_bins = 5
        n_sims = 50
        cov = np.eye(n_bins)

        # Apply correction
        precision_corrected = apply_hartlap_correction(cov, n_sims)

        # Corrected precision should be smaller (diagonal elements)
        precision_uncorrected = np.linalg.inv(cov)

        assert np.all(np.diag(precision_corrected) < np.diag(precision_uncorrected))

    def test_hartlap_requires_enough_simulations(self):
        """Test that Hartlap correction requires N_s > N_d + 2."""
        from desi_ksz.covariance import compute_hartlap_factor

        # Too few simulations
        n_sims = 10
        n_data = 10

        alpha = compute_hartlap_factor(n_sims, n_data)

        # Should return 1.0 (no correction) or raise warning
        assert alpha == 1.0 or alpha <= 0  # Invalid case


class TestCovarianceRegularization:
    """Tests for covariance regularization."""

    def test_regularize_singular_covariance(self, mock_covariance):
        """Test regularization of nearly singular covariance."""
        from desi_ksz.covariance import regularize_covariance

        # Make covariance nearly singular
        cov_singular = mock_covariance.copy()
        cov_singular[0, :] = cov_singular[1, :]
        cov_singular[:, 0] = cov_singular[:, 1]

        # Regularize
        cov_reg = regularize_covariance(cov_singular)

        # Should be invertible
        try:
            precision = np.linalg.inv(cov_reg)
            assert True
        except np.linalg.LinAlgError:
            pytest.fail("Regularized covariance should be invertible")

    def test_regularization_preserves_structure(self, mock_covariance):
        """Test that regularization preserves covariance structure."""
        from desi_ksz.covariance import regularize_covariance

        cov_reg = regularize_covariance(mock_covariance, epsilon=1e-6)

        # Diagonal should be similar
        np.testing.assert_allclose(
            np.diag(cov_reg), np.diag(mock_covariance),
            rtol=1e-4
        )

        # Should still be symmetric
        np.testing.assert_allclose(cov_reg, cov_reg.T)
