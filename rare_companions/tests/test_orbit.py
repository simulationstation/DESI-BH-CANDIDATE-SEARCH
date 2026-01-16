"""
Unit tests for orbital fitting.
"""

import unittest
import numpy as np

from ..orbit.fast_screen import FastOrbitScreen, rv_model_circular
from ..orbit.mass_function import compute_mass_function, compute_m2_min


class TestRVModel(unittest.TestCase):
    """Tests for RV model."""

    def test_circular_amplitude(self):
        """Circular model should have correct amplitude."""
        t = np.linspace(0, 10, 100)
        P = 10.0
        K = 50.0
        gamma = 0.0
        phi = 0.0

        rv = rv_model_circular(t, P, K, gamma, phi)

        self.assertAlmostEqual(np.max(rv), K, places=1)
        self.assertAlmostEqual(np.min(rv), -K, places=1)

    def test_circular_period(self):
        """Circular model should have correct period."""
        t = np.array([0, 5, 10, 15, 20])  # Two full periods for P=10
        P = 10.0
        K = 50.0
        gamma = 0.0
        phi = 0.0

        rv = rv_model_circular(t, P, K, gamma, phi)

        # RV at t=0 and t=10 should be the same
        self.assertAlmostEqual(rv[0], rv[2], places=5)
        self.assertAlmostEqual(rv[0], rv[4], places=5)

    def test_systemic_velocity(self):
        """Systemic velocity should shift all RVs."""
        t = np.linspace(0, 10, 100)
        P = 10.0
        K = 50.0
        gamma = 25.0
        phi = 0.0

        rv = rv_model_circular(t, P, K, gamma, phi)

        self.assertAlmostEqual(np.mean(rv), gamma, places=1)


class TestFastOrbitScreen(unittest.TestCase):
    """Tests for fast orbit screening."""

    def test_recover_injected_period(self):
        """Should recover injected period from clean signal."""
        P_true = 15.0
        K_true = 80.0
        gamma_true = -20.0

        # Generate synthetic data
        np.random.seed(42)
        mjd = np.array([57000, 57010, 57025, 57040, 57060])
        rv_true = rv_model_circular(mjd, P_true, K_true, gamma_true, 0.5)
        rv_err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        rv = rv_true + np.random.normal(0, rv_err)

        # Screen
        screener = FastOrbitScreen(period_min=5.0, period_max=50.0, n_periods=100)
        result = screener.search(mjd, rv, rv_err, parallel=False)

        # Should recover period within 20%
        self.assertGreater(result.best_period, P_true * 0.8)
        self.assertLess(result.best_period, P_true * 1.2)

    def test_delta_chi2_improvement(self):
        """Orbital fit should improve chi2 over constant."""
        P_true = 20.0
        K_true = 100.0
        gamma_true = 0.0

        np.random.seed(123)
        mjd = np.array([57000, 57015, 57030, 57045, 57060])
        rv_true = rv_model_circular(mjd, P_true, K_true, gamma_true, 0.0)
        rv_err = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        rv = rv_true + np.random.normal(0, rv_err)

        screener = FastOrbitScreen(period_min=5.0, period_max=50.0, n_periods=100)
        result = screener.search(mjd, rv, rv_err, parallel=False)

        self.assertGreater(result.delta_chi2, 10.0)


class TestMassFunction(unittest.TestCase):
    """Tests for mass function calculations."""

    def test_mass_function_formula(self):
        """Test mass function formula with known values."""
        # For P=1 day, K=100 km/s, e=0:
        # f(M) = 1.0361e-7 * 1 * 100^3 = 0.1036 Msun
        f_M = compute_mass_function(P=1.0, K=100.0, e=0.0)
        self.assertAlmostEqual(f_M, 0.1036, places=3)

    def test_eccentricity_effect(self):
        """Higher eccentricity should reduce mass function."""
        f_M_circ = compute_mass_function(P=10.0, K=50.0, e=0.0)
        f_M_ecc = compute_mass_function(P=10.0, K=50.0, e=0.5)

        self.assertGreater(f_M_circ, f_M_ecc)

    def test_m2_min_ordering(self):
        """Higher f(M) should give higher M2_min."""
        f_M_low = 0.5
        f_M_high = 2.0
        M1 = 0.5

        m2_min_low = compute_m2_min(f_M_low, M1)
        m2_min_high = compute_m2_min(f_M_high, M1)

        self.assertGreater(m2_min_high, m2_min_low)

    def test_m2_min_positive(self):
        """M2_min should always be positive."""
        for f_M in [0.1, 0.5, 1.0, 5.0]:
            for M1 in [0.3, 0.5, 1.0, 2.0]:
                m2_min = compute_m2_min(f_M, M1)
                self.assertGreater(m2_min, 0.0)


if __name__ == '__main__':
    unittest.main()
