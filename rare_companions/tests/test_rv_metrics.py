"""
Unit tests for RV metrics calculations.
"""

import unittest
import numpy as np

from ..rv.metrics import (
    compute_rv_significance,
    compute_chi2_constant,
    compute_leverage,
    compute_loo_significance,
    check_same_night_consistency
)


class TestRVSignificance(unittest.TestCase):
    """Tests for RV significance calculation."""

    def test_constant_rv(self):
        """Constant RV should have S ~ 0."""
        rv = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        rv_err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        S = compute_rv_significance(rv, rv_err)
        self.assertLess(S, 1.0)

    def test_high_amplitude(self):
        """High amplitude RV should have high S."""
        rv = np.array([-100.0, 100.0, -100.0, 100.0, -100.0])
        rv_err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        S = compute_rv_significance(rv, rv_err)
        self.assertGreater(S, 10.0)

    def test_error_scaling(self):
        """Larger errors should reduce S."""
        rv = np.array([-50.0, 50.0, -50.0, 50.0])
        rv_err_small = np.array([1.0, 1.0, 1.0, 1.0])
        rv_err_large = np.array([10.0, 10.0, 10.0, 10.0])

        S_small = compute_rv_significance(rv, rv_err_small)
        S_large = compute_rv_significance(rv, rv_err_large)

        self.assertGreater(S_small, S_large)

    def test_minimum_epochs(self):
        """Should return 0 for < 2 epochs."""
        rv = np.array([10.0])
        rv_err = np.array([1.0])

        S = compute_rv_significance(rv, rv_err)
        self.assertEqual(S, 0.0)


class TestChi2Constant(unittest.TestCase):
    """Tests for chi-squared test against constant RV."""

    def test_constant_rv(self):
        """Constant RV should have low chi2."""
        rv = np.array([0.0, 0.0, 0.0, 0.0])
        rv_err = np.array([1.0, 1.0, 1.0, 1.0])

        chi2, dof, pvalue = compute_chi2_constant(rv, rv_err)

        self.assertEqual(dof, 3)
        self.assertGreater(pvalue, 0.1)

    def test_variable_rv(self):
        """Variable RV should have high chi2 and low p-value."""
        rv = np.array([-100.0, 100.0, -100.0, 100.0])
        rv_err = np.array([1.0, 1.0, 1.0, 1.0])

        chi2, dof, pvalue = compute_chi2_constant(rv, rv_err)

        self.assertGreater(chi2, 100)
        self.assertLess(pvalue, 0.001)


class TestLeverage(unittest.TestCase):
    """Tests for leverage calculation."""

    def test_outlier_detection(self):
        """Outlier should have highest leverage."""
        rv = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        rv_err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        d, i_max, d_max = compute_leverage(rv, rv_err)

        self.assertEqual(i_max, 4)  # Last epoch is outlier
        self.assertGreater(d_max, d[0])

    def test_uniform_no_leverage(self):
        """Uniform data should have similar leverage."""
        rv = np.array([10.0, 20.0, 30.0, 40.0])
        rv_err = np.array([1.0, 1.0, 1.0, 1.0])

        d, i_max, d_max = compute_leverage(rv, rv_err)

        # Edge values should have slightly higher leverage
        self.assertTrue(i_max in [0, 3])


class TestLOOSignificance(unittest.TestCase):
    """Tests for leave-one-out significance."""

    def test_robust_signal(self):
        """Signal robust to single epoch removal."""
        rv = np.array([-50.0, 50.0, -50.0, 50.0, -50.0])
        rv_err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        S_loo, S_min = compute_loo_significance(rv, rv_err)

        # Should still be significant after removing any epoch
        self.assertGreater(S_min, 5.0)

    def test_fragile_signal(self):
        """Signal dependent on single epoch."""
        rv = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        rv_err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        S_loo, S_min = compute_loo_significance(rv, rv_err)

        # S drops to near zero when outlier removed
        self.assertLess(S_min, 1.0)


class TestSameNightConsistency(unittest.TestCase):
    """Tests for same-night consistency check."""

    def test_consistent_pair(self):
        """Same-night epochs with similar RV should be consistent."""
        mjd = np.array([59000.0, 59000.1])  # ~2.4 hours apart
        rv = np.array([50.0, 51.0])
        rv_err = np.array([2.0, 2.0])

        consistent, pairs = check_same_night_consistency(mjd, rv, rv_err)

        self.assertTrue(consistent)
        self.assertEqual(len(pairs), 1)

    def test_inconsistent_pair(self):
        """Same-night epochs with very different RV should be flagged."""
        mjd = np.array([59000.0, 59000.1])  # ~2.4 hours apart
        rv = np.array([0.0, 100.0])
        rv_err = np.array([2.0, 2.0])

        consistent, pairs = check_same_night_consistency(mjd, rv, rv_err)

        self.assertFalse(consistent)

    def test_no_same_night(self):
        """Epochs on different nights should have no pairs."""
        mjd = np.array([59000.0, 59001.0, 59002.0])
        rv = np.array([0.0, 50.0, 100.0])
        rv_err = np.array([1.0, 1.0, 1.0])

        consistent, pairs = check_same_night_consistency(mjd, rv, rv_err)

        self.assertTrue(consistent)
        self.assertEqual(len(pairs), 0)


if __name__ == '__main__':
    unittest.main()
