"""
Tests for P1 (Priority 1) credibility-critical modules.

Tests provenance, validation, covariance stability, injection tests,
and null suite functionality.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_covariance():
    """Create a mock positive-definite covariance matrix."""
    n_bins = 10
    # Create positive definite matrix
    A = np.random.randn(n_bins, n_bins)
    cov = A @ A.T + np.eye(n_bins) * 0.1
    return cov


@pytest.fixture
def mock_ill_conditioned_cov():
    """Create an ill-conditioned covariance matrix."""
    n_bins = 10
    # Create matrix with high condition number
    eigenvalues = np.logspace(-6, 2, n_bins)
    Q, _ = np.linalg.qr(np.random.randn(n_bins, n_bins))
    cov = Q @ np.diag(eigenvalues) @ Q.T
    return cov


@pytest.fixture
def mock_catalog():
    """Create mock galaxy catalog."""
    n_gal = 500
    rng = np.random.default_rng(42)

    ra = rng.uniform(0, 360, n_gal)
    dec = rng.uniform(-60, 60, n_gal)
    z = rng.uniform(0.1, 0.8, n_gal)
    weights = rng.uniform(0.5, 1.5, n_gal)

    # Compute positions (simplified)
    from astropy.cosmology import Planck18
    d_c = Planck18.comoving_distance(z).value  # Mpc

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    positions = np.column_stack([
        d_c * np.sin(theta) * np.cos(phi),
        d_c * np.sin(theta) * np.sin(phi),
        d_c * np.cos(theta),
    ])

    return {
        'ra': ra,
        'dec': dec,
        'z': z,
        'weights': weights,
        'positions': positions,
    }


@pytest.fixture
def mock_temperatures(mock_catalog):
    """Create mock temperature measurements."""
    n_gal = len(mock_catalog['ra'])
    rng = np.random.default_rng(42)
    return rng.standard_normal(n_gal) * 100  # ~100 uK rms


@pytest.fixture
def separation_bins():
    """Standard separation bins."""
    return np.linspace(20, 150, 11)  # 10 bins


@pytest.fixture
def theory_template(separation_bins):
    """Simple theory template."""
    r_centers = 0.5 * (separation_bins[:-1] + separation_bins[1:])
    return 1.0 / (1 + r_centers / 50.0)


# ============================================================================
# Test Provenance Module
# ============================================================================

class TestProvenance:
    """Tests for desi_ksz/io/provenance.py."""

    def test_compute_config_hash(self):
        """Test config hash computation."""
        from desi_ksz.io.provenance import compute_config_hash

        config = {'tracer': 'LRG', 'z_min': 0.4, 'z_max': 0.6}
        hash1 = compute_config_hash(config)

        assert isinstance(hash1, str)
        assert len(hash1) == 16  # Short hash

        # Same config should give same hash
        hash2 = compute_config_hash(config)
        assert hash1 == hash2

        # Different config should give different hash
        config2 = {'tracer': 'BGS', 'z_min': 0.1, 'z_max': 0.3}
        hash3 = compute_config_hash(config2)
        assert hash1 != hash3

    def test_sha256_file(self):
        """Test file checksum computation."""
        from desi_ksz.io.provenance import sha256_file

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            checksum = sha256_file(temp_path)
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # Full SHA-256

            # Same file should give same checksum
            checksum2 = sha256_file(temp_path)
            assert checksum == checksum2
        finally:
            Path(temp_path).unlink()

    def test_run_manifest_creation(self):
        """Test RunManifest creation."""
        from desi_ksz.io.provenance import RunManifest

        config = {'tracer': 'LRG'}
        manifest = RunManifest.create(config, command='test')

        assert manifest.run_id is not None
        assert manifest.config_hash is not None
        assert manifest.command == 'test'
        assert manifest.status == 'running'

    def test_run_manifest_serialization(self):
        """Test RunManifest JSON serialization."""
        from desi_ksz.io.provenance import RunManifest

        config = {'tracer': 'LRG'}
        manifest = RunManifest.create(config)

        # To dict
        d = manifest.to_dict()
        assert isinstance(d, dict)
        assert 'run_id' in d

        # To/from JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            manifest.to_json(temp_path)
            loaded = RunManifest.from_json(temp_path)
            assert loaded.run_id == manifest.run_id
        finally:
            Path(temp_path).unlink()

    def test_stage_timer(self):
        """Test stage timing context manager."""
        from desi_ksz.io.provenance import RunManifest, stage_timer
        import time

        config = {'tracer': 'LRG'}
        manifest = RunManifest.create(config)

        with stage_timer(manifest, 'test_stage'):
            time.sleep(0.01)  # 10ms

        assert 'test_stage' in manifest.stage_timings
        assert manifest.stage_timings['test_stage'] > 0


# ============================================================================
# Test Covariance Stability Module
# ============================================================================

class TestCovarianceStability:
    """Tests for desi_ksz/covariance/stability.py."""

    def test_analyze_covariance(self, mock_covariance):
        """Test covariance analysis."""
        from desi_ksz.covariance.stability import analyze_covariance

        analysis = analyze_covariance(mock_covariance)

        assert 'shape' in analysis
        assert 'condition_number' in analysis
        assert 'is_positive_definite' in analysis
        assert 'eigenvalue_min' in analysis
        assert 'eigenvalue_max' in analysis

        assert analysis['is_positive_definite'] is True
        assert analysis['condition_number'] > 0

    def test_hartlap_factor(self):
        """Test Hartlap correction factor computation."""
        from desi_ksz.covariance.stability import compute_hartlap_factor

        # N_samples = 100, N_bins = 10
        factor = compute_hartlap_factor(100, 10)

        # Should be (100 - 10 - 2) / (100 - 1) = 88/99 ≈ 0.889
        expected = (100 - 10 - 2) / (100 - 1)
        assert abs(factor - expected) < 1e-6

        # Factor should be < 1
        assert factor < 1.0

        # Edge case: N_samples too small
        factor_edge = compute_hartlap_factor(12, 10)
        assert factor_edge == 0.0  # Invalid case

    def test_regularize_eigenvalue_floor(self, mock_ill_conditioned_cov):
        """Test eigenvalue floor regularization."""
        from desi_ksz.covariance.stability import (
            analyze_covariance, regularize_eigenvalue_floor
        )

        # Original condition number
        analysis = analyze_covariance(mock_ill_conditioned_cov)
        original_kappa = analysis['condition_number']

        # Regularize
        cov_reg, info = regularize_eigenvalue_floor(mock_ill_conditioned_cov)

        # New condition number should be smaller
        assert info['new_condition_number'] < original_kappa

        # Regularized matrix should still be positive definite
        analysis_reg = analyze_covariance(cov_reg)
        assert analysis_reg['is_positive_definite']

    def test_regularize_shrinkage(self, mock_covariance):
        """Test shrinkage regularization."""
        from desi_ksz.covariance.stability import regularize_shrinkage

        cov_shrunk, info = regularize_shrinkage(mock_covariance, alpha=0.1)

        # Shape should be preserved
        assert cov_shrunk.shape == mock_covariance.shape

        # Diagonal should change
        assert not np.allclose(np.diag(cov_shrunk), np.diag(mock_covariance))

        # Info should contain shrinkage factor
        assert 'shrinkage_alpha' in info

    def test_choose_regularization(self, mock_covariance, mock_ill_conditioned_cov):
        """Test regularization recommendation."""
        from desi_ksz.covariance.stability import (
            analyze_covariance, choose_regularization
        )

        # Well-conditioned matrix
        analysis_good = analyze_covariance(mock_covariance)
        choice_good = choose_regularization(analysis_good)
        # Should recommend none or mild regularization

        # Ill-conditioned matrix
        analysis_bad = analyze_covariance(mock_ill_conditioned_cov)
        choice_bad = choose_regularization(analysis_bad)
        # Should recommend regularization
        assert choice_bad['recommendation'] != 'none'


# ============================================================================
# Test Map Validation Module
# ============================================================================

class TestMapValidation:
    """Tests for desi_ksz/maps/validation.py."""

    def test_check_nan_inf(self):
        """Test NaN/Inf detection."""
        from desi_ksz.maps.validation import check_nan_inf

        # Clean array
        clean = np.random.randn(100)
        result = check_nan_inf(clean)
        assert result['passed'] is True
        assert result['n_nan'] == 0

        # Array with NaN
        with_nan = clean.copy()
        with_nan[0] = np.nan
        result = check_nan_inf(with_nan)
        assert result['passed'] is False
        assert result['n_nan'] == 1

    def test_check_map_statistics(self):
        """Test map statistics checks."""
        from desi_ksz.maps.validation import check_map_statistics

        # Good CMB-like map
        rng = np.random.default_rng(42)
        good_map = rng.standard_normal(1000) * 150  # ~150 uK rms

        result = check_map_statistics(good_map, expected_unit="uK_CMB")
        assert 'mean' in result
        assert 'std' in result
        assert result['n_valid'] == 1000

    def test_validate_catalog(self, mock_catalog):
        """Test catalog validation."""
        from desi_ksz.maps.validation import validate_catalog

        result = validate_catalog(
            mock_catalog['ra'],
            mock_catalog['dec'],
            mock_catalog['z'],
            mock_catalog['weights'],
        )

        assert result['passed'] is True
        assert 'coordinates' in result['checks']
        assert 'redshift' in result['checks']

    def test_validate_catalog_invalid_coords(self):
        """Test catalog validation with invalid coordinates."""
        from desi_ksz.maps.validation import validate_catalog

        # Invalid RA
        result = validate_catalog(
            ra=np.array([400, 100, 200]),  # 400 > 360
            dec=np.array([0, 10, 20]),
            z=np.array([0.1, 0.2, 0.3]),
        )

        assert result['passed'] is False
        assert result['checks']['coordinates']['passed'] is False


# ============================================================================
# Test Injection Tests Module
# ============================================================================

class TestInjectionTests:
    """Tests for desi_ksz/sims/injection_tests.py."""

    def test_injection_result_dataclass(self):
        """Test InjectionTestResult dataclass."""
        from desi_ksz.sims.injection_tests import InjectionTestResult

        result = InjectionTestResult(
            input_amplitude=1.0,
            recovered_amplitudes=np.array([0.9, 1.0, 1.1]),
            mean_recovered=1.0,
            std_recovered=0.1,
            bias=0.0,
            bias_sigma=0.0,
            n_realizations=3,
            passed=True,
        )

        assert result.fractional_bias == 0.0
        assert 'PASS' in str(result)

    def test_injection_result_serialization(self):
        """Test InjectionTestResult JSON serialization."""
        from desi_ksz.sims.injection_tests import InjectionTestResult

        result = InjectionTestResult(
            input_amplitude=1.0,
            recovered_amplitudes=np.array([0.9, 1.0, 1.1]),
            mean_recovered=1.0,
            std_recovered=0.1,
            bias=0.0,
            bias_sigma=0.0,
            n_realizations=3,
            passed=True,
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'input_amplitude' in d

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            result.to_json(temp_path)
            with open(temp_path) as f:
                loaded = json.load(f)
            assert loaded['input_amplitude'] == 1.0
        finally:
            Path(temp_path).unlink()

    def test_fit_amplitude_weighted(self):
        """Test weighted amplitude fitting."""
        from desi_ksz.sims.injection_tests import _fit_amplitude_weighted

        # Perfect recovery case
        p_theory = np.array([1.0, 0.5, 0.25])
        p_data = p_theory * 2.0  # A = 2
        cov_inv = np.eye(3)

        A_fit = _fit_amplitude_weighted(p_data, p_theory, cov_inv)
        assert abs(A_fit - 2.0) < 1e-10

    @pytest.mark.slow
    def test_run_injection_simple(self, mock_catalog, separation_bins, theory_template):
        """Test simple injection mode (slow)."""
        from desi_ksz.sims.injection_tests import run_injection_test
        from desi_ksz.estimators import PairwiseMomentumEstimator

        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)

        result = run_injection_test(
            estimator=estimator,
            positions=mock_catalog['positions'],
            weights=mock_catalog['weights'],
            theory_template=theory_template,
            r_bins=separation_bins,
            input_amplitude=1.0,
            n_realizations=5,  # Very few for speed
            injection_mode='simple',
        )

        assert result.n_realizations == 5
        assert np.isfinite(result.mean_recovered)


# ============================================================================
# Test Null Suite Module
# ============================================================================

class TestNullSuite:
    """Tests for desi_ksz/systematics/null_suite.py."""

    def test_null_result_dataclass(self):
        """Test NullTestResult dataclass."""
        from desi_ksz.systematics.null_suite import NullTestResult

        result = NullTestResult(
            test_name='test',
            description='Test description',
            amplitude=0.1,
            amplitude_err=0.5,
            amplitude_sigma=0.2,
            chi2=10.0,
            n_dof=9,
            pte=0.35,
            passed=True,
            threshold='A/σ < 2',
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d['test_name'] == 'test'

    def test_null_suite_result(self):
        """Test NullSuiteResult aggregation."""
        from desi_ksz.systematics.null_suite import NullTestResult, NullSuiteResult

        results = [
            NullTestResult('test1', '', 0.1, 0.5, 0.2, 10, 9, 0.35, True, ''),
            NullTestResult('test2', '', 0.1, 0.5, 0.2, 10, 9, 0.35, True, ''),
            NullTestResult('test3', '', 1.0, 0.5, 2.0, 30, 9, 0.01, False, ''),
        ]

        suite_result = NullSuiteResult(results)

        assert suite_result.n_tests == 3
        assert suite_result.n_passed == 2
        assert suite_result.n_failed == 1
        assert suite_result.all_passed is False

    def test_null_shuffle_basic(
        self, mock_catalog, mock_temperatures, separation_bins, theory_template
    ):
        """Test shuffle temperatures null test."""
        from desi_ksz.systematics.null_suite import null_shuffle_temperatures
        from desi_ksz.estimators import PairwiseMomentumEstimator

        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)
        n_bins = len(separation_bins) - 1
        cov = np.eye(n_bins) * 100

        result = null_shuffle_temperatures(
            estimator=estimator,
            positions=mock_catalog['positions'],
            temperatures=mock_temperatures,
            weights=mock_catalog['weights'],
            template=theory_template,
            cov=cov,
            n_realizations=5,  # Very few for speed
        )

        assert result.test_name == 'shuffle_temperatures'
        assert 0 <= result.pte <= 1

    def test_null_hemisphere_split(
        self, mock_catalog, mock_temperatures, separation_bins, theory_template
    ):
        """Test hemisphere split null test."""
        from desi_ksz.systematics.null_suite import null_hemisphere_split
        from desi_ksz.estimators import PairwiseMomentumEstimator

        estimator = PairwiseMomentumEstimator(separation_bins=separation_bins)
        n_bins = len(separation_bins) - 1
        cov = np.eye(n_bins) * 100

        result = null_hemisphere_split(
            estimator=estimator,
            positions=mock_catalog['positions'],
            temperatures=mock_temperatures,
            weights=mock_catalog['weights'],
            dec=mock_catalog['dec'],
            template=theory_template,
            cov=cov,
        )

        assert result.test_name == 'hemisphere_split'
        assert 'north_amplitude' in result.details
        assert 'south_amplitude' in result.details


# ============================================================================
# Integration Tests
# ============================================================================

class TestP1Integration:
    """Integration tests for P1 modules."""

    def test_full_validation_workflow(self, mock_catalog):
        """Test validation workflow end-to-end."""
        from desi_ksz.maps.validation import validate_catalog

        # Validate catalog
        result = validate_catalog(
            mock_catalog['ra'],
            mock_catalog['dec'],
            mock_catalog['z'],
            mock_catalog['weights'],
        )

        assert result['passed'] is True

    def test_provenance_with_stages(self):
        """Test provenance tracking with multiple stages."""
        from desi_ksz.io.provenance import RunManifest, stage_timer
        import time

        config = {'tracer': 'LRG', 'z_min': 0.4}
        manifest = RunManifest.create(config, command='test_pipeline')

        stages = ['load_data', 'compute_pairwise', 'run_nulls']

        for stage in stages:
            with stage_timer(manifest, stage):
                time.sleep(0.001)

        assert len(manifest.stage_timings) == 3
        assert all(s in manifest.stage_timings for s in stages)

        manifest.mark_completed()
        assert manifest.status == 'completed'
