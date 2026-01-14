"""
Integration tests for kSZ analysis pipeline.

Tests full pipeline workflows with mock data.
"""

import pytest
import numpy as np


class TestFullPipeline:
    """Integration tests for complete pipeline."""

    def test_mock_pipeline(
        self,
        mock_positions,
        mock_temperatures,
        mock_weights,
        separation_bins,
        mock_theory_template,
    ):
        """Test complete analysis pipeline with mock data."""
        from desi_ksz.estimators import PairwiseMomentumEstimator
        from desi_ksz.covariance import SpatialJackknife
        from desi_ksz.inference import KSZLikelihood

        # Step 1: Compute pairwise momentum
        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)
        result = estimator.compute(mock_positions, mock_temperatures, mock_weights)

        assert len(result.p_ksz) == len(separation_bins) - 1

        # Step 2: Estimate covariance (simplified - just diagonal)
        n_bins = len(result.p_ksz)
        mock_cov = np.diag(result.p_ksz_err ** 2)

        # Step 3: Fit amplitude
        likelihood = KSZLikelihood(
            data=result.p_ksz,
            covariance=mock_cov,
            theory_template=mock_theory_template,
        )

        A_ml, sigma_A = likelihood.fit_amplitude()

        assert np.isfinite(A_ml)
        assert sigma_A > 0

        # Step 4: Compute detection significance
        detection = likelihood.compute_detection_significance()

        assert hasattr(detection, 'detection_sigma')
        assert hasattr(detection, 'pte')

    def test_redshift_tomography(
        self,
        mock_catalog,
        separation_bins,
        mock_theory_template,
    ):
        """Test analysis in multiple redshift bins."""
        from desi_ksz.io import DESIGalaxyCatalog
        from desi_ksz.estimators import PairwiseMomentumEstimator

        # Create catalog object
        catalog = DESIGalaxyCatalog(
            ra=mock_catalog['ra'],
            dec=mock_catalog['dec'],
            z=mock_catalog['z'],
            weights=mock_catalog['weights'],
        )
        catalog.compute_comoving_distances()
        catalog.compute_positions()

        # Define redshift bins
        z_bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.8)]

        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)

        results = []
        for z_min, z_max in z_bins:
            # Select redshift bin
            cat_bin = catalog.select_redshift_bin(z_min, z_max)

            if len(cat_bin.ra) < 10:
                continue

            cat_bin.compute_comoving_distances()
            cat_bin.compute_positions()

            # Generate mock temperatures for this bin
            rng = np.random.default_rng(42)
            temps = rng.standard_normal(len(cat_bin.ra)) * 100

            # Compute pairwise momentum
            result = estimator.compute(cat_bin.positions, temps, cat_bin.weights)
            results.append({
                'z_min': z_min,
                'z_max': z_max,
                'p_ksz': result.p_ksz,
                'n_gal': len(cat_bin.ra),
            })

        # Should have results for at least one bin
        assert len(results) >= 1


class TestNullTestIntegration:
    """Integration tests for null test suite."""

    def test_shuffle_null_test(
        self,
        mock_positions,
        mock_temperatures,
        mock_weights,
        separation_bins,
        mock_covariance,
    ):
        """Test shuffle temperatures null test."""
        from desi_ksz.estimators import PairwiseMomentumEstimator
        from desi_ksz.systematics import NullTestSuite

        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)

        # Run with very few realizations for speed
        suite = NullTestSuite(n_realizations=10, pte_threshold=0.05)

        result = suite.shuffle_temperatures(
            estimator=estimator,
            positions=mock_positions,
            temperatures=mock_temperatures,
            weights=mock_weights,
            covariance=mock_covariance,
        )

        assert hasattr(result, 'pte')
        assert hasattr(result, 'observed_chi2')
        assert result.n_realizations == 10

        # PTE should be in valid range
        assert 0 <= result.pte <= 1


class TestInjectionIntegration:
    """Integration tests for injection tests."""

    @pytest.mark.slow
    def test_injection_recovery(
        self,
        mock_positions,
        mock_weights,
        separation_bins,
        mock_theory_template,
    ):
        """Test that injection test recovers input amplitude."""
        from desi_ksz.estimators import PairwiseMomentumEstimator
        from desi_ksz.sims import run_injection_test

        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)

        result = run_injection_test(
            estimator=estimator,
            positions=mock_positions,
            weights=mock_weights,
            theory_template=mock_theory_template,
            r_bins=separation_bins,
            input_amplitude=1.0,
            n_realizations=5,  # Very few for speed
            random_seed=42,
        )

        assert hasattr(result, 'mean_recovered')
        assert hasattr(result, 'bias')
        assert result.n_realizations == 5


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_info_command(self):
        """Test CLI info command runs without error."""
        from click.testing import CliRunner
        from desi_ksz.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['info'])

        assert result.exit_code == 0
        assert 'Dependencies' in result.output

    def test_cli_help(self):
        """Test CLI help works."""
        from click.testing import CliRunner
        from desi_ksz.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'kSZ Analysis Pipeline' in result.output


class TestConfigIntegration:
    """Integration tests for configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from desi_ksz.config import DEFAULT_SEPARATION_BINS, DEFAULT_REDSHIFT_BINS

        assert len(DEFAULT_SEPARATION_BINS) > 1
        assert len(DEFAULT_REDSHIFT_BINS) > 0

        # Bins should be increasing
        assert all(
            DEFAULT_SEPARATION_BINS[i] < DEFAULT_SEPARATION_BINS[i + 1]
            for i in range(len(DEFAULT_SEPARATION_BINS) - 1)
        )

    def test_config_schema(self):
        """Test configuration schema validation."""
        try:
            from desi_ksz.config import PipelineConfig
        except ImportError:
            pytest.skip("pydantic not available")

        config = PipelineConfig(
            tracer='LRG',
            cmb_source='act_dr6',
        )

        assert config.tracer == 'LRG'
        assert config.cmb_source == 'act_dr6'
