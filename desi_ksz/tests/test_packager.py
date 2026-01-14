"""
Tests for results packager.

Tests cover:
- Bundle directory creation
- File registration and checksums
- Summary generation
- Manifest creation
- Decision report writing
"""

import pytest
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from desi_ksz.results.packager import (
    ResultsPackager,
    ResultsSummary,
    BundleManifest,
    create_results_bundle,
)
from desi_ksz.runner.gates import GateStatus, GateResult, GateEvaluationResult


# Mock classes for testing
@dataclass
class MockTomographicResult:
    """Mock tomographic result."""
    z_bin_label: str = "0.3 < z < 0.5"
    z_mean: float = 0.4
    n_galaxies: int = 10000
    n_pairs: int = 500000
    r_centers: np.ndarray = field(default_factory=lambda: np.array([10, 20, 30, 40, 50]))
    pairwise_momentum: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.2, 0.15, 0.1, 0.05]))
    pairwise_momentum_err: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.02, 0.02]))
    theory_template: np.ndarray = field(default_factory=lambda: np.array([0.12, 0.18, 0.14, 0.09, 0.04]))
    amplitude: float = 1.05
    amplitude_err: float = 0.15
    snr: float = 7.0


@dataclass
class MockPhase34Result:
    """Mock Phase 3-4 result."""
    z_bin_results: List[MockTomographicResult] = field(default_factory=list)
    covariance: np.ndarray = None
    joint_amplitude: float = 1.0
    joint_amplitude_err: float = 0.1
    joint_snr: float = 10.0
    plots: Dict[str, Path] = field(default_factory=dict)
    null_test_result: Any = None
    gate_result: GateEvaluationResult = None
    metrics: Dict[str, float] = field(default_factory=dict)
    referee_results: Dict[str, Any] = field(default_factory=dict)


def create_mock_gate_result(
    overall_status: str = "PASS",
    critical_passed: int = 6,
    critical_failed: int = 0,
    warnings: int = 0,
) -> GateEvaluationResult:
    """Create mock gate evaluation result."""
    gate_results = [
        GateResult(
            name='injection_bias',
            status=GateStatus.PASS,
            metric=0.5,
            threshold=2.0,
            message='Injection bias < 2 sigma',
            is_critical=True,
        ),
        GateResult(
            name='null_suite_pass_rate',
            status=GateStatus.PASS,
            metric=0.9,
            threshold=0.8,
            message='Null test pass rate >= 80%',
            is_critical=True,
        ),
    ]

    return GateEvaluationResult(
        overall_status=overall_status,
        critical_passed=critical_passed,
        critical_failed=critical_failed,
        warnings=warnings,
        skipped=0,
        gate_results=gate_results,
        failed_gates=[],
        warning_gates=[],
        recommendation="All gates passed.",
        rerun_commands=[],
    )


class TestResultsSummary:
    """Tests for ResultsSummary dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = ResultsSummary(
            overall_status="PASS",
            recommendation="Results are publication-ready.",
            detection_significance=5.0,
            amplitude=1.0,
            amplitude_error=0.2,
            amplitude_snr=5.0,
            catalog="LRG",
            n_galaxies=100000,
            z_bins=["0.3-0.5", "0.5-0.7"],
            map_sources=["Planck PR4"],
            n_critical_gates_passed=6,
            n_critical_gates_total=6,
            n_warnings=0,
            null_test_pass_rate=0.9,
            injection_bias_sigma=0.5,
            transfer_bias_percent=2.0,
            tsz_stability_sigma=0.3,
            hartlap_factor=0.85,
            referee_check_results={"look_elsewhere": "PASS"},
            failed_gates=[],
            warning_gates=[],
        )

        d = summary.to_dict()
        assert d['overall_status'] == "PASS"
        assert d['detection_significance'] == 5.0
        assert d['n_galaxies'] == 100000
        assert len(d['z_bins']) == 2


class TestBundleManifest:
    """Tests for BundleManifest dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        manifest = BundleManifest(
            bundle_name="test_bundle",
            created_at="2024-01-01T00:00:00",
            pipeline_version="1.0.0",
            git_commit="abc123",
            config_hash="def456",
            files=[{"path": "test.csv", "size_bytes": 100}],
            total_size_bytes=100,
            n_files=1,
        )

        d = manifest.to_dict()
        assert d['bundle_name'] == "test_bundle"
        assert d['pipeline_version'] == "1.0.0"
        assert d['n_files'] == 1


class TestResultsPackager:
    """Tests for ResultsPackager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_result(self):
        """Create mock Phase34 result."""
        result = MockPhase34Result(
            z_bin_results=[
                MockTomographicResult(z_bin_label="0.3 < z < 0.5", z_mean=0.4),
                MockTomographicResult(z_bin_label="0.5 < z < 0.7", z_mean=0.6),
            ],
            covariance=np.eye(5) * 0.01,
            joint_amplitude=1.0,
            joint_amplitude_err=0.1,
            joint_snr=10.0,
            metrics={
                'null_pass_rate': 0.9,
                'injection_bias_sigma': 0.5,
                'transfer_bias': 0.02,
                'tsz_delta_sigma': 0.3,
                'hartlap_factor': 0.85,
            },
        )
        return result

    @pytest.fixture
    def mock_gate_result(self):
        """Create mock gate result."""
        return create_mock_gate_result()

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        return {
            'catalog': 'LRG',
            'map_sources': ['Planck PR4', 'ACT DR6'],
            'z_bins': [0.3, 0.5, 0.7],
        }

    def test_packager_init(self, temp_dir):
        """Test packager initialization."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test_bundle")
        assert packager.bundle_name == "test_bundle"
        assert packager.output_dir == temp_dir

    def test_packager_auto_name(self, temp_dir):
        """Test auto-generated bundle name."""
        packager = ResultsPackager(str(temp_dir))
        assert packager.bundle_name.startswith("ksz_results_")

    def test_create_directories(self, temp_dir):
        """Test directory structure creation."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test")
        packager._create_directories()

        assert packager.bundle_path.exists()
        assert packager.plots_dir.exists()
        assert packager.tables_dir.exists()
        assert packager.configs_dir.exists()
        assert packager.data_dir.exists()

    def test_create_bundle(self, temp_dir, mock_result, mock_gate_result, mock_config):
        """Test complete bundle creation."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test_bundle")
        bundle_path = packager.create_bundle(
            phase34_result=mock_result,
            gate_result=mock_gate_result,
            config=mock_config,
        )

        # Check bundle exists
        assert bundle_path.exists()
        assert bundle_path.name == "test_bundle"

        # Check key files
        assert (bundle_path / "summary.json").exists()
        assert (bundle_path / "results.md").exists()
        assert (bundle_path / "manifest.json").exists()
        assert (bundle_path / "checksums.sha256").exists()

        # Check subdirectories have content
        assert (bundle_path / "tables").exists()
        assert (bundle_path / "configs").exists()
        assert (bundle_path / "data").exists()

    def test_summary_json_valid(self, temp_dir, mock_result, mock_gate_result, mock_config):
        """Test summary.json is valid JSON."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test")
        bundle_path = packager.create_bundle(
            phase34_result=mock_result,
            gate_result=mock_gate_result,
            config=mock_config,
        )

        summary_path = bundle_path / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary['overall_status'] == "PASS"
        assert 'detection_significance' in summary
        assert summary['catalog'] == "LRG"

    def test_manifest_json_valid(self, temp_dir, mock_result, mock_gate_result, mock_config):
        """Test manifest.json is valid JSON."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test")
        bundle_path = packager.create_bundle(
            phase34_result=mock_result,
            gate_result=mock_gate_result,
            config=mock_config,
        )

        manifest_path = bundle_path / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest['bundle_name'] == "test"
        assert 'files' in manifest
        assert manifest['n_files'] > 0

    def test_results_md_content(self, temp_dir, mock_result, mock_gate_result, mock_config):
        """Test results.md has expected content."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test")
        bundle_path = packager.create_bundle(
            phase34_result=mock_result,
            gate_result=mock_gate_result,
            config=mock_config,
        )

        results_path = bundle_path / "results.md"
        content = results_path.read_text()

        assert "# DESI DR1 Pairwise kSZ Analysis Results" in content
        assert "## Decision: PASS" in content
        assert "Detection Summary" in content
        assert "Gate Evaluation" in content

    def test_checksums_valid(self, temp_dir, mock_result, mock_gate_result, mock_config):
        """Test checksums file has valid format."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test")
        bundle_path = packager.create_bundle(
            phase34_result=mock_result,
            gate_result=mock_gate_result,
            config=mock_config,
        )

        checksums_path = bundle_path / "checksums.sha256"
        content = checksums_path.read_text()

        # Each line should be: sha256_hash  filepath
        for line in content.strip().split('\n'):
            parts = line.split('  ')
            assert len(parts) == 2
            hash_value, filepath = parts
            assert len(hash_value) == 64  # SHA256 is 64 hex chars

    def test_tables_created(self, temp_dir, mock_result, mock_gate_result, mock_config):
        """Test CSV tables are created."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test")
        bundle_path = packager.create_bundle(
            phase34_result=mock_result,
            gate_result=mock_gate_result,
            config=mock_config,
        )

        tables_dir = bundle_path / "tables"

        # Check pairwise momentum tables
        momentum_files = list(tables_dir.glob("pairwise_momentum_*.csv"))
        assert len(momentum_files) >= 2  # Two z-bins

        # Check amplitude summary
        assert (tables_dir / "amplitude_summary.csv").exists()

        # Check covariance
        assert (tables_dir / "covariance_matrix.csv").exists()

    def test_data_files_created(self, temp_dir, mock_result, mock_gate_result, mock_config):
        """Test data files are created."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test")
        bundle_path = packager.create_bundle(
            phase34_result=mock_result,
            gate_result=mock_gate_result,
            config=mock_config,
        )

        data_dir = bundle_path / "data"

        # Check covariance numpy file
        assert (data_dir / "covariance.npy").exists()

        # Check pairwise data
        pairwise_files = list(data_dir.glob("pairwise_*.npz"))
        assert len(pairwise_files) >= 2

    def test_config_saved(self, temp_dir, mock_result, mock_gate_result, mock_config):
        """Test configuration is saved."""
        packager = ResultsPackager(str(temp_dir), bundle_name="test")
        bundle_path = packager.create_bundle(
            phase34_result=mock_result,
            gate_result=mock_gate_result,
            config=mock_config,
        )

        config_path = bundle_path / "configs" / "run_config.yaml"
        assert config_path.exists()


class TestCreateResultsBundle:
    """Tests for create_results_bundle convenience function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_convenience_function(self, temp_dir):
        """Test convenience function creates bundle."""
        mock_result = MockPhase34Result(
            z_bin_results=[MockTomographicResult()],
            covariance=np.eye(5) * 0.01,
        )
        mock_gate = create_mock_gate_result()
        mock_config = {'catalog': 'LRG'}

        bundle_path = create_results_bundle(
            output_dir=str(temp_dir),
            phase34_result=mock_result,
            gate_result=mock_gate,
            config=mock_config,
            bundle_name="convenience_test",
        )

        assert bundle_path.exists()
        assert bundle_path.name == "convenience_test"


class TestFailedBundleReport:
    """Tests for FAIL/INCONCLUSIVE status bundles."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_fail_status_report(self, temp_dir):
        """Test FAIL status is properly reported."""
        mock_result = MockPhase34Result(
            z_bin_results=[MockTomographicResult()],
            covariance=np.eye(5) * 0.01,
        )

        failed_gate = GateEvaluationResult(
            overall_status="FAIL",
            critical_passed=5,
            critical_failed=1,
            warnings=0,
            skipped=0,
            gate_results=[
                GateResult(
                    name='injection_bias',
                    status=GateStatus.FAIL,
                    metric=3.0,
                    threshold=2.0,
                    message='Injection bias > 2 sigma',
                    is_critical=True,
                )
            ],
            failed_gates=['injection_bias'],
            warning_gates=[],
            recommendation="Address failed critical gates.",
            rerun_commands=['python -m desi_ksz.cli injection-test -n 200'],
        )

        bundle_path = create_results_bundle(
            output_dir=str(temp_dir),
            phase34_result=mock_result,
            gate_result=failed_gate,
            config={'catalog': 'LRG'},
            bundle_name="fail_test",
        )

        # Check report shows FAIL
        results_md = (bundle_path / "results.md").read_text()
        assert "## Decision: FAIL" in results_md
        assert "injection_bias" in results_md

        # Check summary shows FAIL
        with open(bundle_path / "summary.json") as f:
            summary = json.load(f)
        assert summary['overall_status'] == "FAIL"
        assert 'injection_bias' in summary['failed_gates']

    def test_inconclusive_status_report(self, temp_dir):
        """Test INCONCLUSIVE status is properly reported."""
        mock_result = MockPhase34Result(
            z_bin_results=[MockTomographicResult()],
            covariance=np.eye(5) * 0.01,
        )

        inconclusive_gate = GateEvaluationResult(
            overall_status="INCONCLUSIVE",
            critical_passed=6,
            critical_failed=0,
            warnings=3,
            skipped=0,
            gate_results=[],
            failed_gates=[],
            warning_gates=['covariance_condition', 'look_elsewhere', 'anisotropy_residual'],
            recommendation="Multiple warnings. Review before publication.",
            rerun_commands=[],
        )

        bundle_path = create_results_bundle(
            output_dir=str(temp_dir),
            phase34_result=mock_result,
            gate_result=inconclusive_gate,
            config={'catalog': 'LRG'},
            bundle_name="inconclusive_test",
        )

        results_md = (bundle_path / "results.md").read_text()
        assert "## Decision: INCONCLUSIVE" in results_md
