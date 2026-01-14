"""
Tests for P0 credibility-critical modules.

Tests:
  - tSZ leakage control (cluster mask sweep, y-regression)
  - Multi-frequency map set operations
  - Pair counting backend adapter (Corrfunc/KDTree)
  - Covariance auto-regularization
  - Map transfer function test
"""

import pytest
import numpy as np
from dataclasses import dataclass


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def random_positions():
    """Generate random galaxy positions."""
    rng = np.random.default_rng(42)
    n_gal = 1000

    ra = rng.uniform(0, 360, n_gal)
    dec = rng.uniform(-60, 60, n_gal)

    # Convert to comoving positions (simplified)
    z = rng.uniform(0.1, 0.8, n_gal)
    from astropy.cosmology import Planck18
    chi = Planck18.comoving_distance(z).value  # Mpc

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    x = chi * np.cos(dec_rad) * np.cos(ra_rad)
    y = chi * np.cos(dec_rad) * np.sin(ra_rad)
    z_pos = chi * np.sin(dec_rad)

    positions = np.column_stack([x, y, z_pos])
    weights = np.ones(n_gal)

    return {
        'positions': positions,
        'weights': weights,
        'ra': ra,
        'dec': dec,
        'z': z,
    }


@pytest.fixture
def mock_temperatures():
    """Generate mock temperature measurements."""
    rng = np.random.default_rng(123)
    n_gal = 1000
    return rng.standard_normal(n_gal) * 100  # μK


@pytest.fixture
def mock_covariance():
    """Generate mock covariance matrix."""
    n_bins = 15
    rng = np.random.default_rng(456)

    # Random positive definite matrix
    A = rng.standard_normal((n_bins, n_bins))
    cov = A @ A.T + np.eye(n_bins) * 10
    return cov


@pytest.fixture
def mock_template():
    """Generate mock theory template."""
    r_bins = np.linspace(20, 150, 16)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    return 1.0 / (1 + r_centers / 50.0)


# =============================================================================
# tSZ Leakage Tests
# =============================================================================

class TestTSZLeakage:
    """Tests for tSZ leakage control module."""

    def test_import(self):
        """Test module imports successfully."""
        from desi_ksz.systematics.tsz_leakage import (
            ClusterMaskSweepResult,
            YMapRegressionResult,
            load_planck_cluster_catalog,
            create_cluster_mask,
        )
        assert ClusterMaskSweepResult is not None
        assert YMapRegressionResult is not None

    def test_cluster_mask_sweep_result_dataclass(self):
        """Test ClusterMaskSweepResult dataclass."""
        from desi_ksz.systematics.tsz_leakage import ClusterMaskSweepResult

        result = ClusterMaskSweepResult(
            mask_radii_arcmin=[0, 5, 10],
            amplitudes=[1.0, 0.95, 0.93],
            amplitude_errors=[0.1, 0.12, 0.15],
            delta_amplitudes=[0.0, -0.05, -0.07],
            p_ksz_by_radius={0: np.zeros(10), 5: np.zeros(10)},
            n_clusters_masked=[0, 100, 100],
            fraction_masked=[0.0, 0.01, 0.02],
            baseline_amplitude=1.0,
            converged=True,
            convergence_radius=5.0,
            recommendation="Amplitude stable at radius >= 5' - use this mask",
        )

        assert result.converged is True
        assert result.convergence_radius == 5.0

        # Test to_dict
        d = result.to_dict()
        assert 'mask_radii_arcmin' in d
        assert 'converged' in d

    def test_ymap_regression_result_dataclass(self):
        """Test YMapRegressionResult dataclass."""
        from desi_ksz.systematics.tsz_leakage import YMapRegressionResult

        result = YMapRegressionResult(
            regression_coefficient=-2.3,
            regression_coefficient_err=0.5,
            residual_correlation=0.01,
            original_amplitude=1.0,
            cleaned_amplitude=0.98,
            amplitude_shift=-0.02,
            amplitude_shift_sigma=0.5,
            passed=True,
        )

        assert result.passed is True
        assert result.amplitude_shift_sigma < 2.0

    def test_load_empty_cluster_catalog(self):
        """Test loading cluster catalog with no file."""
        from desi_ksz.systematics.tsz_leakage import load_planck_cluster_catalog

        ra, dec, theta = load_planck_cluster_catalog(None)

        assert len(ra) == 0
        assert len(dec) == 0
        assert len(theta) == 0

    @pytest.mark.skipif(
        not pytest.importorskip("healpy", reason="healpy not available"),
        reason="healpy not available"
    )
    def test_create_cluster_mask_empty(self):
        """Test creating cluster mask with no clusters."""
        from desi_ksz.systematics.tsz_leakage import create_cluster_mask
        import healpy as hp

        nside = 64
        mask = create_cluster_mask(
            cluster_ra=np.array([]),
            cluster_dec=np.array([]),
            mask_radius_arcmin=10.0,
            nside=nside,
        )

        # All pixels should be valid
        assert mask.shape == (hp.nside2npix(nside),)
        assert np.all(mask == 1)


# =============================================================================
# Map Set Tests
# =============================================================================

class TestMapSet:
    """Tests for multi-frequency map set operations."""

    def test_import(self):
        """Test module imports successfully."""
        from desi_ksz.io.map_set import (
            FrequencyMap,
            MapSetResult,
            MapSet,
        )
        assert FrequencyMap is not None
        assert MapSetResult is not None
        assert MapSet is not None

    def test_frequency_map_creation(self):
        """Test FrequencyMap dataclass creation."""
        from desi_ksz.io.map_set import FrequencyMap

        data = np.random.randn(12 * 64**2)  # nside=64
        fmap = FrequencyMap(
            data=data,
            frequency_ghz=150.0,
            beam_fwhm_arcmin=1.4,
        )

        assert fmap.frequency_ghz == 150.0
        assert fmap.beam_fwhm_arcmin == 1.4
        assert fmap.valid_fraction > 0

    def test_map_set_add_and_get(self):
        """Test adding and retrieving maps from MapSet."""
        from desi_ksz.io.map_set import MapSet

        mapset = MapSet(name="test")

        data_150 = np.random.randn(1000)
        data_090 = np.random.randn(1000)

        mapset.add_map(data_150, frequency_ghz=150.0, beam_fwhm_arcmin=1.4)
        mapset.add_map(data_090, frequency_ghz=90.0, beam_fwhm_arcmin=2.1)

        assert mapset.n_frequencies == 2
        assert 150.0 in mapset.frequencies
        assert 90.0 in mapset.frequencies

        # Retrieve
        fmap = mapset.get_map(150.0)
        assert fmap.frequency_ghz == 150.0

    def test_map_set_null_map(self):
        """Test creating null (difference) map."""
        from desi_ksz.io.map_set import MapSet

        mapset = MapSet(name="test")

        # Create two maps with known difference
        n_pix = 1000
        data_150 = np.ones(n_pix) * 100
        data_090 = np.ones(n_pix) * 50

        mapset.add_map(data_150, frequency_ghz=150.0, beam_fwhm_arcmin=1.4)
        mapset.add_map(data_090, frequency_ghz=90.0, beam_fwhm_arcmin=2.1)

        result = mapset.create_null_map(150.0, 90.0)

        assert result.operation == 'difference'
        assert np.allclose(result.data, 50.0)  # 100 - 50
        assert result.effective_beam_fwhm_arcmin == 2.1  # max of the two

    def test_map_set_coadd(self):
        """Test creating inverse-variance coadd."""
        from desi_ksz.io.map_set import MapSet

        mapset = MapSet(name="test")

        n_pix = 1000
        data_1 = np.ones(n_pix) * 100
        data_2 = np.ones(n_pix) * 200

        # Equal weights (no ivar)
        mapset.add_map(data_1, frequency_ghz=150.0, beam_fwhm_arcmin=1.4)
        mapset.add_map(data_2, frequency_ghz=90.0, beam_fwhm_arcmin=2.1)

        result = mapset.create_coadd()

        assert result.operation == 'coadd'
        # With equal weights, should be average
        assert np.allclose(result.data, 150.0)  # (100 + 200) / 2

    def test_map_set_validation(self):
        """Test map set validation."""
        from desi_ksz.io.map_set import MapSet

        mapset = MapSet(name="test")
        data = np.random.randn(1000)
        mapset.add_map(data, frequency_ghz=150.0)

        validation = mapset.validate()

        assert validation['valid'] is True
        assert validation['n_frequencies'] == 1


# =============================================================================
# Pair Counting Backend Tests
# =============================================================================

class TestPairCountingBackend:
    """Tests for pair counting backend adapter."""

    def test_import(self):
        """Test module imports successfully."""
        from desi_ksz.estimators.pair_counting import (
            EfficientPairCounter,
            CorrfuncPairCounter,
            AdaptivePairCounter,
            get_available_backends,
            get_default_backend,
        )
        assert EfficientPairCounter is not None
        assert AdaptivePairCounter is not None

    def test_get_available_backends(self):
        """Test getting available backends."""
        from desi_ksz.estimators.pair_counting import get_available_backends

        backends = get_available_backends()

        assert isinstance(backends, list)
        assert 'kdtree' in backends  # Should always be available

    def test_get_default_backend(self):
        """Test getting default backend."""
        from desi_ksz.estimators.pair_counting import get_default_backend

        backend = get_default_backend()

        assert backend in ['corrfunc', 'kdtree']

    def test_adaptive_counter_kdtree(self):
        """Test AdaptivePairCounter with KDTree backend."""
        from desi_ksz.estimators.pair_counting import AdaptivePairCounter

        counter = AdaptivePairCounter(backend='kdtree')

        assert counter.backend == 'kdtree'

        # Test with random positions
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (500, 3))

        counter.build_tree(positions)
        bin_edges = np.linspace(0, 50, 11)
        result = counter.count_pairs(bin_edges)

        assert result.total_pairs > 0
        assert len(result.pair_counts) == 10

    def test_pair_count_consistency(self):
        """Test that pair counts are consistent."""
        from desi_ksz.estimators.pair_counting import count_pairs_in_bins

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (200, 3))
        bin_edges = np.linspace(0, 50, 6)

        result = count_pairs_in_bins(positions, bin_edges)

        # Total pairs should be less than N*(N-1)/2
        max_pairs = len(positions) * (len(positions) - 1) // 2
        assert result.total_pairs <= max_pairs

    def test_benchmark_backends(self):
        """Test backend benchmarking function."""
        from desi_ksz.estimators.pair_counting import benchmark_backends

        results = benchmark_backends(n_points=500, n_bins=5)

        assert isinstance(results, dict)
        assert 'kdtree' in results
        assert 'time_seconds' in results['kdtree']
        assert results['kdtree']['time_seconds'] > 0


# =============================================================================
# Covariance Auto-Regularization Tests
# =============================================================================

class TestCovarianceAutoRegularization:
    """Tests for covariance auto-regularization."""

    def test_import(self):
        """Test module imports successfully."""
        from desi_ksz.covariance.stability import (
            AutoRegularizationResult,
            auto_regularize,
            robust_precision_matrix,
            check_hartlap_regime,
            compute_ledoit_wolf_shrinkage,
        )
        assert AutoRegularizationResult is not None
        assert auto_regularize is not None

    def test_check_hartlap_regime_valid(self):
        """Test Hartlap regime check with valid parameters."""
        from desi_ksz.covariance.stability import check_hartlap_regime

        result = check_hartlap_regime(n_samples=100, n_bins=15)

        assert result['status'] == 'OK'
        assert result['is_valid'] is True
        assert result['hartlap_factor'] > 0.8

    def test_check_hartlap_regime_invalid(self):
        """Test Hartlap regime check with invalid parameters."""
        from desi_ksz.covariance.stability import check_hartlap_regime

        result = check_hartlap_regime(n_samples=10, n_bins=15)

        assert result['status'] == 'INVALID'
        assert result['is_valid'] is False

    def test_ledoit_wolf_shrinkage(self):
        """Test Ledoit-Wolf shrinkage computation."""
        from desi_ksz.covariance.stability import compute_ledoit_wolf_shrinkage

        # Well-conditioned matrix
        cov = np.eye(10) + 0.1 * np.random.randn(10, 10)
        cov = cov @ cov.T
        alpha = compute_ledoit_wolf_shrinkage(cov)

        assert 0 <= alpha <= 1

    def test_auto_regularize_well_conditioned(self, mock_covariance):
        """Test auto-regularization on well-conditioned matrix."""
        from desi_ksz.covariance.stability import auto_regularize

        result = auto_regularize(mock_covariance, n_samples=100)

        assert result.original_condition < 1e10
        assert result.final_condition <= result.original_condition
        assert result.hartlap_warning is False

    def test_auto_regularize_ill_conditioned(self):
        """Test auto-regularization on ill-conditioned matrix."""
        from desi_ksz.covariance.stability import auto_regularize

        # Create ill-conditioned matrix
        n = 15
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        cov = A @ A.T
        cov += np.outer(np.ones(n), np.ones(n)) * 1e6  # Make ill-conditioned

        result = auto_regularize(cov, n_samples=100)

        # Should have applied some regularization
        assert result.final_condition < result.original_condition

    def test_robust_precision_matrix(self, mock_covariance):
        """Test robust precision matrix computation."""
        from desi_ksz.covariance.stability import robust_precision_matrix

        precision, info = robust_precision_matrix(mock_covariance, n_samples=100)

        assert precision.shape == mock_covariance.shape
        assert info['hartlap_factor'] > 0
        assert 'method' in info

    def test_auto_regularization_result_to_dict(self):
        """Test AutoRegularizationResult serialization."""
        from desi_ksz.covariance.stability import AutoRegularizationResult

        result = AutoRegularizationResult(
            original_cov=np.eye(5),
            regularized_cov=np.eye(5),
            method='none',
            parameters={},
            original_condition=1.0,
            final_condition=1.0,
            hartlap_factor=0.95,
            hartlap_warning=False,
            warnings=[],
        )

        d = result.to_dict()
        assert 'method' in d
        assert 'hartlap_factor' in d


# =============================================================================
# Map Transfer Function Tests
# =============================================================================

class TestMapTransferFunction:
    """Tests for map transfer function test module."""

    def test_import(self):
        """Test module imports successfully."""
        from desi_ksz.maps.validation import (
            TransferFunctionTestResult,
            map_transfer_function_test,
            pairwise_signal_propagation_test,
        )
        assert TransferFunctionTestResult is not None
        assert map_transfer_function_test is not None

    def test_transfer_function_result_dataclass(self):
        """Test TransferFunctionTestResult dataclass."""
        from desi_ksz.maps.validation import TransferFunctionTestResult

        result = TransferFunctionTestResult(
            passed=True,
            input_power_spectrum=np.ones(10),
            output_power_spectrum=np.ones(10),
            transfer_function=np.ones(10),
            ell_bins=np.arange(10),
            mean_transfer=1.0,
            std_transfer=0.1,
            bias=0.01,
            details={'n_realizations': 5},
        )

        assert result.passed is True
        assert result.mean_transfer == 1.0

        d = result.to_dict()
        assert 'passed' in d
        assert 'bias' in d

    @pytest.mark.skipif(
        not pytest.importorskip("healpy", reason="healpy not available"),
        reason="healpy not available"
    )
    def test_map_transfer_function_test(self, random_positions):
        """Test map transfer function test execution."""
        from desi_ksz.maps.validation import map_transfer_function_test
        import healpy as hp

        ra = random_positions['ra'][:100]  # Use subset for speed
        dec = random_positions['dec'][:100]

        def extract_temps(map_data, ra, dec):
            theta = np.radians(90.0 - dec)
            phi = np.radians(ra)
            nside = hp.npix2nside(len(map_data))
            pix = hp.ang2pix(nside, theta, phi)
            return map_data[pix]

        result = map_transfer_function_test(
            temperature_extraction_func=extract_temps,
            positions_ra=ra,
            positions_dec=dec,
            nside=64,
            lmax=100,
            n_realizations=2,
        )

        assert hasattr(result, 'passed')
        assert hasattr(result, 'mean_transfer')


# =============================================================================
# Integration Tests
# =============================================================================

class TestP0Integration:
    """Integration tests for P0 modules."""

    def test_full_pipeline_mock(self, random_positions, mock_temperatures, mock_covariance, mock_template):
        """Test P0 modules work together."""
        # 1. Pair counting
        from desi_ksz.estimators.pair_counting import AdaptivePairCounter

        counter = AdaptivePairCounter(backend='kdtree')
        counter.build_tree(random_positions['positions'])
        bin_edges = np.linspace(20, 150, 16)
        pair_result = counter.count_pairs(bin_edges)

        assert pair_result.total_pairs > 0

        # 2. Covariance regularization
        from desi_ksz.covariance.stability import robust_precision_matrix

        precision, info = robust_precision_matrix(mock_covariance, n_samples=100)
        assert precision.shape == mock_covariance.shape

        # 3. Map set operations
        from desi_ksz.io.map_set import MapSet

        mapset = MapSet(name='test')
        mapset.add_map(np.random.randn(1000), frequency_ghz=150.0)
        mapset.add_map(np.random.randn(1000), frequency_ghz=90.0)

        validation = mapset.validate()
        assert validation['valid'] is True

    def test_all_p0_modules_importable(self):
        """Test that all P0 modules can be imported."""
        modules_to_import = [
            'desi_ksz.systematics.tsz_leakage',
            'desi_ksz.io.map_set',
            'desi_ksz.estimators.pair_counting',
            'desi_ksz.covariance.stability',
            'desi_ksz.maps.validation',
        ]

        for module_name in modules_to_import:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
