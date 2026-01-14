"""
Tests for inference modules.

Tests likelihood computation and MCMC sampling.
"""

import pytest
import numpy as np


class TestLikelihood:
    """Tests for likelihood computation."""

    def test_likelihood_creation(
        self, mock_pksz_data, mock_covariance, mock_theory_template
    ):
        """Test KSZLikelihood initialization."""
        from desi_ksz.inference import KSZLikelihood

        likelihood = KSZLikelihood(
            data=mock_pksz_data['p_ksz'],
            covariance=mock_covariance,
            theory_template=mock_theory_template,
        )

        assert likelihood is not None
        assert likelihood.n_bins == len(mock_pksz_data['p_ksz'])

    def test_log_likelihood_value(
        self, mock_pksz_data, mock_covariance, mock_theory_template
    ):
        """Test log-likelihood returns finite value."""
        from desi_ksz.inference import KSZLikelihood

        likelihood = KSZLikelihood(
            data=mock_pksz_data['p_ksz'],
            covariance=mock_covariance,
            theory_template=mock_theory_template,
        )

        log_L = likelihood.log_likelihood(A_ksz=1.0)

        assert np.isfinite(log_L)
        assert log_L < 0  # Log-likelihood should be negative

    def test_log_likelihood_maximum_at_ml(
        self, mock_pksz_data, mock_covariance, mock_theory_template
    ):
        """Test that log-likelihood is maximized at ML estimate."""
        from desi_ksz.inference import KSZLikelihood

        likelihood = KSZLikelihood(
            data=mock_pksz_data['p_ksz'],
            covariance=mock_covariance,
            theory_template=mock_theory_template,
        )

        # Get ML estimate
        A_ml, sigma_A = likelihood.fit_amplitude()

        # Evaluate log-likelihood at ML and nearby points
        log_L_ml = likelihood.log_likelihood(A_ml)
        log_L_above = likelihood.log_likelihood(A_ml + 2 * sigma_A)
        log_L_below = likelihood.log_likelihood(A_ml - 2 * sigma_A)

        # ML should have highest log-likelihood
        assert log_L_ml >= log_L_above
        assert log_L_ml >= log_L_below

    def test_chi2_at_ml(
        self, mock_pksz_data, mock_covariance, mock_theory_template
    ):
        """Test chi2 computation at ML estimate."""
        from desi_ksz.inference import KSZLikelihood

        likelihood = KSZLikelihood(
            data=mock_pksz_data['p_ksz'],
            covariance=mock_covariance,
            theory_template=mock_theory_template,
        )

        A_ml, _ = likelihood.fit_amplitude()
        chi2_ml = likelihood.chi2(A_ml)

        # Chi2 should be positive
        assert chi2_ml >= 0

        # Chi2 should be reasonable (order of n_dof)
        n_dof = likelihood.n_bins - 1
        assert chi2_ml < 5 * n_dof  # Very loose bound

    def test_amplitude_fit_analytic(
        self, mock_pksz_data, mock_covariance, mock_theory_template
    ):
        """Test analytic amplitude fitting."""
        from desi_ksz.inference import KSZLikelihood

        likelihood = KSZLikelihood(
            data=mock_pksz_data['p_ksz'],
            covariance=mock_covariance,
            theory_template=mock_theory_template,
        )

        A_ml, sigma_A = likelihood.fit_amplitude()

        # Amplitude should be finite
        assert np.isfinite(A_ml)
        assert np.isfinite(sigma_A)

        # Error should be positive
        assert sigma_A > 0

    def test_detection_significance(
        self, mock_pksz_data, mock_covariance, mock_theory_template
    ):
        """Test detection significance computation."""
        from desi_ksz.inference import KSZLikelihood

        likelihood = KSZLikelihood(
            data=mock_pksz_data['p_ksz'],
            covariance=mock_covariance,
            theory_template=mock_theory_template,
        )

        result = likelihood.compute_detection_significance()

        # Check result fields
        assert hasattr(result, 'A_ksz')
        assert hasattr(result, 'A_ksz_err')
        assert hasattr(result, 'detection_sigma')
        assert hasattr(result, 'pte')

        # PTE should be in [0, 1]
        assert 0 <= result.pte <= 1

    def test_hartlap_correction_effect(
        self, mock_pksz_data, mock_covariance, mock_theory_template
    ):
        """Test that Hartlap correction affects error bars."""
        from desi_ksz.inference import KSZLikelihood

        # Without Hartlap
        likelihood_no_hartlap = KSZLikelihood(
            data=mock_pksz_data['p_ksz'],
            covariance=mock_covariance,
            theory_template=mock_theory_template,
            hartlap_n_sims=None,
        )

        # With Hartlap
        likelihood_hartlap = KSZLikelihood(
            data=mock_pksz_data['p_ksz'],
            covariance=mock_covariance,
            theory_template=mock_theory_template,
            hartlap_n_sims=50,
        )

        _, sigma_no = likelihood_no_hartlap.fit_amplitude()
        _, sigma_yes = likelihood_hartlap.fit_amplitude()

        # Hartlap should increase error bars
        assert sigma_yes > sigma_no


class TestFitAmplitudeAnalytic:
    """Tests for standalone amplitude fitting function."""

    def test_fit_amplitude_analytic(self, mock_pksz_data, mock_theory_template):
        """Test fit_amplitude_analytic function."""
        from desi_ksz.inference import fit_amplitude_analytic

        # Create precision matrix
        n_bins = len(mock_pksz_data['p_ksz'])
        precision = np.eye(n_bins) / 0.04  # σ = 0.2

        A_ml, sigma_A = fit_amplitude_analytic(
            data=mock_pksz_data['p_ksz'],
            theory=mock_theory_template,
            precision=precision,
        )

        assert np.isfinite(A_ml)
        assert sigma_A > 0

    def test_fit_amplitude_formula(self):
        """Test amplitude fit formula: A = (p^T Psi d) / (p^T Psi p)."""
        from desi_ksz.inference import fit_amplitude_analytic

        # Simple test case
        data = np.array([2.0, 4.0, 6.0])
        theory = np.array([1.0, 2.0, 3.0])  # data = 2 * theory
        precision = np.eye(3)

        A_ml, sigma_A = fit_amplitude_analytic(data, theory, precision)

        # Should recover A = 2
        assert np.isclose(A_ml, 2.0)


class TestMCMC:
    """Tests for MCMC sampling."""

    @pytest.mark.requires_emcee
    @pytest.mark.slow
    def test_mcmc_basic(self, mock_pksz_data, mock_covariance, mock_theory_template):
        """Test basic MCMC sampling."""
        try:
            from desi_ksz.inference import run_mcmc
        except ImportError:
            pytest.skip("emcee not available")

        # Simple log-prob function
        def log_prob(theta):
            A_ksz = theta[0]
            if not (-5 < A_ksz < 5):
                return -np.inf
            chi2 = (A_ksz - 1.0) ** 2 / 0.1 ** 2
            return -0.5 * chi2

        result = run_mcmc(
            log_prob_fn=log_prob,
            initial_params=np.array([1.0]),
            param_names=['A_ksz'],
            n_walkers=8,
            n_steps=100,
            n_burnin=20,
            progress=False,
        )

        # Check result structure
        assert hasattr(result, 'samples')
        assert hasattr(result, 'param_names')
        assert hasattr(result, 'acceptance_fraction')

        # Check samples shape
        expected_samples = 8 * (100 - 20)  # walkers * (steps - burnin)
        assert result.samples.shape[0] == expected_samples
        assert result.samples.shape[1] == 1

    @pytest.mark.requires_emcee
    def test_mcmc_result_methods(self):
        """Test MCMCResult methods."""
        from desi_ksz.inference import MCMCResult

        # Create mock result
        rng = np.random.default_rng(42)
        samples = rng.normal(1.0, 0.1, (1000, 1))
        log_prob = -0.5 * ((samples[:, 0] - 1.0) / 0.1) ** 2

        result = MCMCResult(
            samples=samples,
            log_prob=log_prob,
            param_names=['A_ksz'],
            n_walkers=8,
            n_steps=200,
            n_burnin=50,
            acceptance_fraction=0.3,
        )

        # Test get_summary
        summary = result.get_summary()
        assert 'A_ksz' in summary
        assert 'mean' in summary['A_ksz']
        assert np.isclose(summary['A_ksz']['mean'], 1.0, atol=0.1)

        # Test get_map_estimate
        map_est = result.get_map_estimate()
        assert 'A_ksz' in map_est

        # Test get_chain
        chain = result.get_chain('A_ksz')
        assert len(chain) == 1000


class TestComputeChi2:
    """Tests for chi2 computation."""

    def test_compute_chi2_zero_residual(self):
        """Test chi2 = 0 when model matches data."""
        from desi_ksz.inference import compute_chi2

        data = np.array([1.0, 2.0, 3.0])
        model = np.array([1.0, 2.0, 3.0])
        precision = np.eye(3)

        chi2 = compute_chi2(data, model, precision)

        assert np.isclose(chi2, 0.0)

    def test_compute_chi2_formula(self):
        """Test chi2 formula: (d-m)^T Psi (d-m)."""
        from desi_ksz.inference import compute_chi2

        data = np.array([1.0, 2.0])
        model = np.array([0.0, 0.0])
        precision = np.array([[2.0, 0.0], [0.0, 2.0]])

        chi2 = compute_chi2(data, model, precision)

        # Manual: chi2 = [1,2] @ [[2,0],[0,2]] @ [1,2] = 2*1 + 2*4 = 10
        expected = 10.0
        assert np.isclose(chi2, expected)
