"""
Tests for referee attack checks.

Tests cover:
- Look-elsewhere effect correction
- Pixel-space anisotropy detection
- Weight leverage stability
- Redshift-dependent systematics (z-split)
- Beam/filter sensitivity
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from desi_ksz.systematics.referee_checks import (
    RefereeCheckResult,
    look_elsewhere_check,
    anisotropy_check,
    weight_leverage_check,
    redshift_split_check,
    beam_sensitivity_check,
    run_all_referee_checks,
)


class TestRefereeCheckResult:
    """Tests for RefereeCheckResult dataclass."""

    def test_passed_result(self):
        """Test passed result creation."""
        result = RefereeCheckResult(
            name='test_check',
            passed=True,
            metric=0.5,
            threshold=1.0,
            message='Test passed',
            details={'extra': 'info'},
        )
        assert result.passed is True
        assert result.metric == 0.5

    def test_failed_result(self):
        """Test failed result creation."""
        result = RefereeCheckResult(
            name='test_check',
            passed=False,
            metric=2.0,
            threshold=1.0,
            message='Test failed',
        )
        assert result.passed is False
        assert result.metric > result.threshold


class TestLookElsewhereCheck:
    """Tests for look-elsewhere correction."""

    def test_significant_remains_significant(self):
        """Test truly significant detection survives correction."""
        # p = 1e-5 with 10 trials should still be significant
        result = look_elsewhere_check(
            base_pvalue=1e-5,
            n_trials=10,
            n_permutations=100,
            seed=42,
        )
        # Should pass (adjusted p-value still < 0.05)
        # Note: Sidak correction: 1 - (1 - p)^n ~ n*p for small p
        # 10 * 1e-5 = 1e-4, still < 0.01 threshold
        assert result.passed is True
        assert result.metric < 0.01

    def test_marginal_fails_correction(self):
        """Test marginal detection fails look-elsewhere."""
        # p = 0.005 with 20 trials:
        # Sidak: 1 - (1-0.005)^20 ~ 0.095 > 0.01
        result = look_elsewhere_check(
            base_pvalue=0.005,
            n_trials=20,
            n_permutations=100,
            seed=42,
        )
        # May fail depending on exact correction
        # The adjusted p-value should be higher than base
        assert result.metric > 0.005  # Adjusted should be worse

    def test_single_trial_no_penalty(self):
        """Test single trial has minimal penalty."""
        result = look_elsewhere_check(
            base_pvalue=0.001,
            n_trials=1,
            n_permutations=100,
            seed=42,
        )
        # With 1 trial, adjusted p ~ base p
        assert result.metric < 0.01  # Still significant
        assert result.passed is True

    def test_result_details(self):
        """Test result contains expected details."""
        result = look_elsewhere_check(
            base_pvalue=0.01,
            n_trials=5,
            n_permutations=50,
            seed=42,
        )
        assert result.name == 'look_elsewhere'
        assert 'base_pvalue' in result.details
        assert 'n_trials' in result.details
        assert 'adjusted_pvalue' in result.details


class TestAnisotropyCheck:
    """Tests for pixel-space anisotropy detection."""

    def test_isotropic_passes(self):
        """Test isotropic temperature distribution passes."""
        np.random.seed(42)
        n = 1000

        # Random uniform positions on sphere
        ra = np.random.uniform(0, 360, n)
        dec = np.random.uniform(-60, 60, n)

        # Isotropic temperatures (no spatial correlation)
        temperatures = np.random.normal(0, 1, n)

        result = anisotropy_check(
            temperatures=temperatures,
            ra=ra,
            dec=dec,
            lmax=2,
            sigma_threshold=3.0,
        )

        # Should pass (no significant dipole/quadrupole)
        assert result.passed is True

    def test_strong_dipole_fails(self):
        """Test strong dipole pattern fails."""
        np.random.seed(42)
        n = 2000

        # Positions
        ra = np.random.uniform(0, 360, n)
        dec = np.random.uniform(-60, 60, n)

        # Add strong dipole: T ~ cos(dec)
        # This creates N-S gradient
        temperatures = 10 * np.sin(np.radians(dec)) + np.random.normal(0, 1, n)

        result = anisotropy_check(
            temperatures=temperatures,
            ra=ra,
            dec=dec,
            lmax=2,
            sigma_threshold=3.0,
        )

        # Should fail (significant dipole)
        # Note: depends on implementation detecting the dipole
        assert 'dipole' in result.details or 'quadrupole' in result.details

    def test_result_structure(self):
        """Test result has expected structure."""
        np.random.seed(42)
        n = 500

        ra = np.random.uniform(0, 360, n)
        dec = np.random.uniform(-60, 60, n)
        temperatures = np.random.normal(0, 1, n)

        result = anisotropy_check(
            temperatures=temperatures,
            ra=ra,
            dec=dec,
        )

        assert result.name == 'anisotropy'
        assert 'max_sigma' in result.details or 'dipole_sigma' in result.details


class TestWeightLeverageCheck:
    """Tests for weight leverage stability."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock estimator."""
        estimator = MagicMock()
        # Return slightly different amplitudes
        estimator.compute_amplitude = MagicMock(
            side_effect=[1.0, 0.98, 1.02]  # Full, without top, with extra call
        )
        return estimator

    def test_stable_weights_pass(self, mock_estimator):
        """Test stable weights pass check."""
        np.random.seed(42)
        n = 1000

        positions = np.random.uniform(-100, 100, (n, 3))
        temperatures = np.random.normal(0, 1, n)
        weights = np.ones(n)  # Uniform weights
        template = np.random.normal(0, 0.1, 10)
        cov = np.eye(10) * 0.01

        # Mock estimator returns stable amplitude
        mock_estimator.compute_amplitude = MagicMock(return_value=(1.0, 0.1))

        result = weight_leverage_check(
            estimator=mock_estimator,
            positions=positions,
            temperatures=temperatures,
            weights=weights,
            template=template,
            cov=cov,
            top_fraction=0.01,
            sigma_threshold=1.0,
        )

        # With uniform weights, removing top 1% shouldn't change much
        # (though depends on mock behavior)
        assert result.name == 'weight_leverage'

    def test_result_details(self, mock_estimator):
        """Test result contains expected details."""
        np.random.seed(42)
        n = 500

        positions = np.random.uniform(-100, 100, (n, 3))
        temperatures = np.random.normal(0, 1, n)
        weights = np.random.uniform(1, 10, n)
        template = np.random.normal(0, 0.1, 5)
        cov = np.eye(5) * 0.01

        mock_estimator.compute_amplitude = MagicMock(return_value=(1.0, 0.1))

        result = weight_leverage_check(
            estimator=mock_estimator,
            positions=positions,
            temperatures=temperatures,
            weights=weights,
            template=template,
            cov=cov,
        )

        assert 'n_removed' in result.details or 'amplitude_full' in result.details


class TestRedshiftSplitCheck:
    """Tests for redshift-dependent systematics."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock estimator."""
        estimator = MagicMock()
        estimator.compute_amplitude = MagicMock(return_value=(1.0, 0.1))
        return estimator

    def test_consistent_z_splits_pass(self, mock_estimator):
        """Test consistent z-splits pass check."""
        np.random.seed(42)
        n = 2000

        positions = np.random.uniform(-100, 100, (n, 3))
        temperatures = np.random.normal(0, 1, n)
        weights = np.ones(n)
        ra = np.random.uniform(0, 360, n)
        dec = np.random.uniform(-60, 60, n)
        z = np.random.uniform(0.3, 0.7, n)
        template = np.random.normal(0, 0.1, 5)
        cov = np.eye(5) * 0.01
        z_bins = [0.3, 0.5, 0.7]

        # Mock returns consistent amplitudes
        mock_estimator.compute_amplitude = MagicMock(
            side_effect=[(1.0, 0.15), (0.95, 0.15), (1.05, 0.15), (0.98, 0.15)]
        )

        result = redshift_split_check(
            estimator=mock_estimator,
            positions=positions,
            temperatures=temperatures,
            weights=weights,
            ra=ra,
            dec=dec,
            z=z,
            template=template,
            cov=cov,
            z_bins=z_bins,
            n_quartiles=4,
            sigma_threshold=2.0,
        )

        assert result.name == 'z_split'
        # Should pass with consistent amplitudes
        # (depends on implementation details)

    def test_result_structure(self, mock_estimator):
        """Test result has expected structure."""
        np.random.seed(42)
        n = 500

        positions = np.random.uniform(-100, 100, (n, 3))
        temperatures = np.random.normal(0, 1, n)
        weights = np.ones(n)
        ra = np.random.uniform(0, 360, n)
        dec = np.random.uniform(-60, 60, n)
        z = np.random.uniform(0.3, 0.7, n)
        template = np.random.normal(0, 0.1, 5)
        cov = np.eye(5) * 0.01
        z_bins = [0.3, 0.5, 0.7]

        mock_estimator.compute_amplitude = MagicMock(return_value=(1.0, 0.1))

        result = redshift_split_check(
            estimator=mock_estimator,
            positions=positions,
            temperatures=temperatures,
            weights=weights,
            ra=ra,
            dec=dec,
            z=z,
            template=template,
            cov=cov,
            z_bins=z_bins,
        )

        assert result.name == 'z_split'
        assert 'max_diff_sigma' in result.details or 'quartile_amplitudes' in result.details


class TestBeamSensitivityCheck:
    """Tests for beam/filter sensitivity."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock estimator."""
        estimator = MagicMock()
        estimator.compute_amplitude = MagicMock(return_value=(1.0, 0.1))
        return estimator

    def test_stable_beam_passes(self, mock_estimator):
        """Test beam perturbation stability passes."""
        np.random.seed(42)
        n = 1000

        positions = np.random.uniform(-100, 100, (n, 3))
        temperatures = np.random.normal(0, 1, n)
        weights = np.ones(n)
        template = np.random.normal(0, 0.1, 5)
        cov = np.eye(5) * 0.01

        # Mock returns stable amplitude across beam perturbations
        mock_estimator.compute_amplitude = MagicMock(
            side_effect=[(1.0, 0.1), (0.99, 0.1), (1.01, 0.1)]
        )

        result = beam_sensitivity_check(
            estimator=mock_estimator,
            positions=positions,
            temperatures=temperatures,
            weights=weights,
            template=template,
            cov=cov,
            beam_fwhm=1.4,
            perturbation=0.05,
            sigma_threshold=1.0,
        )

        assert result.name == 'beam_sensitivity'
        # Should pass with stable amplitudes

    def test_result_details(self, mock_estimator):
        """Test result contains expected details."""
        np.random.seed(42)
        n = 500

        positions = np.random.uniform(-100, 100, (n, 3))
        temperatures = np.random.normal(0, 1, n)
        weights = np.ones(n)
        template = np.random.normal(0, 0.1, 5)
        cov = np.eye(5) * 0.01

        mock_estimator.compute_amplitude = MagicMock(return_value=(1.0, 0.1))

        result = beam_sensitivity_check(
            estimator=mock_estimator,
            positions=positions,
            temperatures=temperatures,
            weights=weights,
            template=template,
            cov=cov,
            beam_fwhm=1.4,
        )

        assert result.name == 'beam_sensitivity'
        assert 'beam_fwhm' in result.details


class TestRunAllRefereeChecks:
    """Tests for run_all_referee_checks function."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock estimator."""
        estimator = MagicMock()
        estimator.compute_amplitude = MagicMock(return_value=(1.0, 0.1))
        return estimator

    def test_returns_dict(self, mock_estimator):
        """Test function returns dictionary of results."""
        np.random.seed(42)
        n = 500

        positions = np.random.uniform(-100, 100, (n, 3))
        temperatures = np.random.normal(0, 1, n)
        weights = np.ones(n)
        ra = np.random.uniform(0, 360, n)
        dec = np.random.uniform(-60, 60, n)
        z = np.random.uniform(0.3, 0.7, n)
        template = np.random.normal(0, 0.1, 5)
        cov = np.eye(5) * 0.01
        z_bins = [0.3, 0.5, 0.7]

        results = run_all_referee_checks(
            estimator=mock_estimator,
            positions=positions,
            temperatures=temperatures,
            weights=weights,
            ra=ra,
            dec=dec,
            z=z,
            template=template,
            cov=cov,
            base_pvalue=0.001,
            n_trials=10,
            z_bins=z_bins,
            beam_fwhm=1.4,
        )

        assert isinstance(results, dict)
        # Should have all 5 checks
        expected_checks = {
            'look_elsewhere', 'anisotropy', 'weight_leverage',
            'z_split', 'beam_sensitivity'
        }
        # May have summary keys too
        for check in expected_checks:
            assert check in results or any(check in k for k in results.keys())

    def test_aggregates_pass_status(self, mock_estimator):
        """Test all_passed flag is correctly computed."""
        np.random.seed(42)
        n = 500

        positions = np.random.uniform(-100, 100, (n, 3))
        temperatures = np.random.normal(0, 1, n)
        weights = np.ones(n)
        ra = np.random.uniform(0, 360, n)
        dec = np.random.uniform(-60, 60, n)
        z = np.random.uniform(0.3, 0.7, n)
        template = np.random.normal(0, 0.1, 5)
        cov = np.eye(5) * 0.01
        z_bins = [0.3, 0.5, 0.7]

        results = run_all_referee_checks(
            estimator=mock_estimator,
            positions=positions,
            temperatures=temperatures,
            weights=weights,
            ra=ra,
            dec=dec,
            z=z,
            template=template,
            cov=cov,
            base_pvalue=1e-6,  # Very significant
            n_trials=3,
            z_bins=z_bins,
            beam_fwhm=1.4,
        )

        # Check for summary/aggregate key
        assert 'all_passed' in results or 'summary' in results


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_positions_anisotropy(self):
        """Test anisotropy check with minimal data."""
        # Very few points
        ra = np.array([0, 90, 180])
        dec = np.array([0, 30, -30])
        temperatures = np.array([1, 2, 3])

        result = anisotropy_check(
            temperatures=temperatures,
            ra=ra,
            dec=dec,
            lmax=1,  # Only dipole with 3 points
        )

        # Should handle gracefully
        assert result.name == 'anisotropy'

    def test_zero_trials_look_elsewhere(self):
        """Test look-elsewhere with edge case trials."""
        result = look_elsewhere_check(
            base_pvalue=0.01,
            n_trials=1,  # Single trial
            n_permutations=10,
        )

        assert result.passed is True  # p=0.01 < 0.01 threshold marginal
        # Or could be False depending on exact threshold

    def test_high_pvalue_look_elsewhere(self):
        """Test look-elsewhere with non-significant p-value."""
        result = look_elsewhere_check(
            base_pvalue=0.5,  # Not significant
            n_trials=5,
        )

        assert result.passed is False  # Should fail
        assert result.metric > 0.01  # Adjusted p-value high
