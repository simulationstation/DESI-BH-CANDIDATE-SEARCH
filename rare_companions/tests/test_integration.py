"""
Integration tests for full pipeline.
"""

import unittest
import tempfile
import os
import numpy as np

from ..config import Config
from ..ingest.unified import RVTimeSeries, RVEpoch
from ..rv.hardening import RVHardener, harden_rv_series
from ..experiments.e1_mass_gap import MassGapExperiment
from ..experiments.e2_dark_companions import DarkCompanionExperiment
from ..scoring.unified import GlobalLeaderboard


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)

        self.config = Config()

        # Create synthetic time series
        self.timeseries_list = []

        for i in range(20):
            n_epochs = np.random.randint(4, 8)

            # Random orbital parameters
            P = np.random.uniform(10, 50)
            K = np.random.uniform(30, 150)
            gamma = np.random.uniform(-30, 30)

            mjd = np.sort(np.random.uniform(57000, 60000, n_epochs))
            phase = 2 * np.pi * mjd / P
            rv_true = gamma + K * np.sin(phase)
            rv_err = np.random.uniform(1, 5, n_epochs)
            rv = rv_true + np.random.normal(0, rv_err)

            epochs = [
                RVEpoch(
                    mjd=mjd[j],
                    rv=rv[j],
                    rv_err=rv_err[j],
                    instrument='TEST',
                    survey='test',
                    quality=1.0
                )
                for j in range(n_epochs)
            ]

            ts = RVTimeSeries(
                targetid=1000 + i,
                source_id=3800000000000000000 + i,
                ra=np.random.uniform(0, 360),
                dec=np.random.uniform(-90, 90),
                epochs=epochs,
                metadata={'M1': 0.5}
            )

            self.timeseries_list.append(ts)

    def test_rv_hardening_pipeline(self):
        """Test RV hardening on synthetic data."""
        hardener = RVHardener()

        for ts in self.timeseries_list:
            metrics = hardener.compute_metrics(ts)

            # Basic sanity checks
            self.assertEqual(metrics.n_epochs, ts.n_epochs)
            self.assertGreaterEqual(metrics.S, 0.0)
            self.assertGreaterEqual(metrics.chi2_constant, 0.0)

    def test_experiment_runs(self):
        """Test that experiments run without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test E1 experiment
            exp1 = MassGapExperiment(self.config)
            result1 = exp1.run(self.timeseries_list, output_dir=tmpdir)

            self.assertEqual(result1.n_input, len(self.timeseries_list))
            self.assertGreaterEqual(result1.n_final, 0)

            # Test E2 experiment
            exp2 = DarkCompanionExperiment(self.config)
            result2 = exp2.run(self.timeseries_list, output_dir=tmpdir)

            self.assertEqual(result2.n_input, len(self.timeseries_list))

    def test_leaderboard_merging(self):
        """Test global leaderboard merging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard = GlobalLeaderboard()

            # Run multiple experiments
            exp1 = MassGapExperiment(self.config)
            result1 = exp1.run(self.timeseries_list, output_dir=tmpdir)
            leaderboard.add_experiment_results(result1)

            exp2 = DarkCompanionExperiment(self.config)
            result2 = exp2.run(self.timeseries_list, output_dir=tmpdir)
            leaderboard.add_experiment_results(result2)

            # Rank candidates
            ranked = leaderboard.rank()

            # Check ordering
            for i in range(1, len(ranked)):
                self.assertGreaterEqual(
                    ranked[i-1].unified_score,
                    ranked[i].unified_score
                )

    def test_output_files_created(self):
        """Test that output files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = MassGapExperiment(self.config)
            result = exp.run(self.timeseries_list, output_dir=tmpdir)

            # Check result file exists
            result_file = os.path.join(tmpdir, f'{exp.name}_results.json')
            self.assertTrue(os.path.exists(result_file))


class TestConfigValidation(unittest.TestCase):
    """Test configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        self.assertEqual(config.selection.min_epochs, 3)
        self.assertGreater(config.selection.min_delta_rv, 0)
        self.assertGreater(config.budget.stage2_deep_fit_top_k, 0)

    def test_config_to_dict(self):
        """Test config serialization."""
        config = Config()
        d = config.to_dict()

        self.assertIn('run_name', d)
        self.assertIn('selection', d)
        self.assertIn('experiments', d)

    def test_config_from_dict(self):
        """Test config deserialization."""
        d = {
            'run_name': 'test_run',
            'selection': {'min_epochs': 5}
        }

        config = Config.from_dict(d)
        self.assertEqual(config.run_name, 'test_run')
        self.assertEqual(config.selection.min_epochs, 5)


if __name__ == '__main__':
    unittest.main()
