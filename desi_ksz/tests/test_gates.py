"""
Tests for gate definitions and evaluation.

Tests cover:
- Gate evaluation logic for all comparator types
- Critical vs warning gate distinction
- Overall status determination
- Rerun command generation
- Metrics extraction from result objects
"""

import pytest
import numpy as np

from desi_ksz.runner.gates import (
    Gate,
    GateStatus,
    GateResult,
    GateEvaluationResult,
    CRITICAL_GATES,
    WARNING_GATES,
    evaluate_gates,
    generate_rerun_commands,
    create_metrics_from_results,
)


class TestGate:
    """Tests for Gate class."""

    def test_gate_lt_pass(self):
        """Test less-than comparator passing."""
        gate = Gate(
            name='test_lt',
            description='Test < comparator',
            threshold=1.0,
            comparator='lt',
            is_critical=True,
        )
        assert gate.evaluate(0.5) == GateStatus.PASS

    def test_gate_lt_fail(self):
        """Test less-than comparator failing."""
        gate = Gate(
            name='test_lt',
            description='Test < comparator',
            threshold=1.0,
            comparator='lt',
            is_critical=True,
        )
        assert gate.evaluate(1.5) == GateStatus.FAIL

    def test_gate_gt_pass(self):
        """Test greater-than comparator passing."""
        gate = Gate(
            name='test_gt',
            description='Test > comparator',
            threshold=0.5,
            comparator='gt',
            is_critical=True,
        )
        assert gate.evaluate(0.8) == GateStatus.PASS

    def test_gate_gt_fail(self):
        """Test greater-than comparator failing."""
        gate = Gate(
            name='test_gt',
            description='Test > comparator',
            threshold=0.5,
            comparator='gt',
            is_critical=True,
        )
        assert gate.evaluate(0.3) == GateStatus.FAIL

    def test_gate_abs_lt_pass(self):
        """Test absolute less-than comparator passing."""
        gate = Gate(
            name='test_abs',
            description='Test abs_lt comparator',
            threshold=2.0,
            comparator='abs_lt',
            is_critical=True,
        )
        assert gate.evaluate(-1.5) == GateStatus.PASS
        assert gate.evaluate(1.5) == GateStatus.PASS

    def test_gate_abs_lt_fail(self):
        """Test absolute less-than comparator failing."""
        gate = Gate(
            name='test_abs',
            description='Test abs_lt comparator',
            threshold=2.0,
            comparator='abs_lt',
            is_critical=True,
        )
        assert gate.evaluate(-2.5) == GateStatus.FAIL
        assert gate.evaluate(2.5) == GateStatus.FAIL

    def test_gate_nan_skip(self):
        """Test NaN metric results in SKIP."""
        gate = Gate(
            name='test_nan',
            description='Test NaN handling',
            threshold=1.0,
            comparator='lt',
            is_critical=True,
        )
        assert gate.evaluate(np.nan) == GateStatus.SKIP
        assert gate.evaluate(np.inf) == GateStatus.SKIP

    def test_gate_noncritical_warn(self):
        """Test non-critical gate returns WARN instead of FAIL."""
        gate = Gate(
            name='test_warn',
            description='Test warning',
            threshold=1.0,
            comparator='lt',
            is_critical=False,
        )
        assert gate.evaluate(1.5) == GateStatus.WARN

    def test_gate_ge_comparator(self):
        """Test greater-equal comparator."""
        gate = Gate(
            name='test_ge',
            description='Test >= comparator',
            threshold=0.8,
            comparator='ge',
            is_critical=True,
        )
        assert gate.evaluate(0.8) == GateStatus.PASS
        assert gate.evaluate(0.9) == GateStatus.PASS
        assert gate.evaluate(0.7) == GateStatus.FAIL

    def test_gate_le_comparator(self):
        """Test less-equal comparator."""
        gate = Gate(
            name='test_le',
            description='Test <= comparator',
            threshold=1.0,
            comparator='le',
            is_critical=True,
        )
        assert gate.evaluate(1.0) == GateStatus.PASS
        assert gate.evaluate(0.5) == GateStatus.PASS
        assert gate.evaluate(1.1) == GateStatus.FAIL


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = GateResult(
            name='test_gate',
            status=GateStatus.PASS,
            metric=0.5,
            threshold=1.0,
            message='Test passed',
            is_critical=True,
            details={'extra': 'info'},
        )
        d = result.to_dict()
        assert d['name'] == 'test_gate'
        assert d['status'] == 'PASS'
        assert d['metric'] == 0.5
        assert d['threshold'] == 1.0
        assert d['is_critical'] is True

    def test_to_dict_nan_metric(self):
        """Test NaN metric is converted to None."""
        result = GateResult(
            name='test_gate',
            status=GateStatus.SKIP,
            metric=np.nan,
            threshold=1.0,
            message='Test skipped',
            is_critical=True,
        )
        d = result.to_dict()
        assert d['metric'] is None


class TestCriticalGates:
    """Tests for critical gate definitions."""

    def test_critical_gates_defined(self):
        """Test all expected critical gates are defined."""
        expected = {
            'injection_bias',
            'null_suite_pass_rate',
            'transfer_test',
            'tsz_sweep_stability',
            'covariance_hartlap',
            'map_consistency',
        }
        assert set(CRITICAL_GATES.keys()) == expected

    def test_all_critical_gates_are_critical(self):
        """Test all gates in CRITICAL_GATES are marked critical."""
        for name, gate in CRITICAL_GATES.items():
            assert gate.is_critical is True, f"{name} should be critical"


class TestWarningGates:
    """Tests for warning gate definitions."""

    def test_warning_gates_defined(self):
        """Test all expected warning gates are defined."""
        expected = {
            'covariance_condition',
            'look_elsewhere',
            'anisotropy_residual',
            'weight_leverage',
            'split_consistency',
            'beam_sensitivity',
            'ymap_regression',
        }
        assert set(WARNING_GATES.keys()) == expected

    def test_all_warning_gates_noncritical(self):
        """Test all gates in WARNING_GATES are non-critical."""
        for name, gate in WARNING_GATES.items():
            assert gate.is_critical is False, f"{name} should be non-critical"


class TestEvaluateGates:
    """Tests for evaluate_gates function."""

    def test_all_pass(self):
        """Test all gates passing returns PASS status."""
        metrics = {
            'injection_bias_sigma': 0.5,  # < 2
            'null_pass_rate': 0.9,        # >= 0.8
            'transfer_bias': 0.02,        # < 0.05
            'tsz_delta_sigma': 0.5,       # < 1
            'hartlap_factor': 0.8,        # > 0.5
            'map_diff_sigma': 1.0,        # < 2
            # Warning gates
            'condition_number': 1e5,      # < 1e6
            'adjusted_pvalue': 0.1,       # > 0.01
            'anisotropy_sigma': 1.0,      # < 3
            'weight_leverage_sigma': 0.5, # < 1
            'split_max_diff_sigma': 1.0,  # < 2
            'beam_sensitivity_sigma': 0.3,# < 1
            'ymap_shift_sigma': 0.5,      # < 2
        }
        result = evaluate_gates(metrics)
        assert result.overall_status == "PASS"
        assert result.critical_failed == 0
        assert result.warnings == 0

    def test_critical_fail(self):
        """Test critical gate failure returns FAIL status."""
        metrics = {
            'injection_bias_sigma': 5.0,  # > 2, FAIL
            'null_pass_rate': 0.9,
            'transfer_bias': 0.02,
            'tsz_delta_sigma': 0.5,
            'hartlap_factor': 0.8,
            'map_diff_sigma': 1.0,
        }
        result = evaluate_gates(metrics, include_warnings=False)
        assert result.overall_status == "FAIL"
        assert result.critical_failed == 1
        assert 'injection_bias' in result.failed_gates

    def test_single_warning_pass(self):
        """Test single warning still results in PASS."""
        metrics = {
            'injection_bias_sigma': 0.5,
            'null_pass_rate': 0.9,
            'transfer_bias': 0.02,
            'tsz_delta_sigma': 0.5,
            'hartlap_factor': 0.8,
            'map_diff_sigma': 1.0,
            # One warning
            'condition_number': 1e7,  # > 1e6, WARN
        }
        result = evaluate_gates(metrics)
        assert result.overall_status == "PASS"
        assert result.warnings == 1
        assert 'covariance_condition' in result.warning_gates

    def test_multiple_warnings_inconclusive(self):
        """Test multiple warnings returns INCONCLUSIVE."""
        metrics = {
            'injection_bias_sigma': 0.5,
            'null_pass_rate': 0.9,
            'transfer_bias': 0.02,
            'tsz_delta_sigma': 0.5,
            'hartlap_factor': 0.8,
            'map_diff_sigma': 1.0,
            # Multiple warnings
            'condition_number': 1e7,
            'anisotropy_sigma': 4.0,
        }
        result = evaluate_gates(metrics)
        assert result.overall_status == "INCONCLUSIVE"
        assert result.warnings >= 2

    def test_missing_metrics_skip(self):
        """Test missing metrics result in SKIP status."""
        metrics = {}  # Empty
        result = evaluate_gates(metrics, include_warnings=False)
        assert result.skipped == len(CRITICAL_GATES)

    def test_exclude_warnings(self):
        """Test include_warnings=False excludes warning gates."""
        metrics = {'condition_number': 1e7}
        result_with = evaluate_gates(metrics, include_warnings=True)
        result_without = evaluate_gates(metrics, include_warnings=False)

        assert result_with.warnings >= 1 or result_with.skipped > result_without.skipped
        # Warning gates should not appear when excluded
        assert 'covariance_condition' not in result_without.warning_gates


class TestGenerateRerunCommands:
    """Tests for generate_rerun_commands function."""

    def test_empty_lists(self):
        """Test empty lists return empty commands."""
        commands = generate_rerun_commands([], [])
        assert commands == []

    def test_failed_gate_command(self):
        """Test failed gate generates rerun command."""
        commands = generate_rerun_commands(['injection_bias'], [])
        assert len(commands) >= 1
        assert any('injection' in cmd for cmd in commands)

    def test_warning_gate_command(self):
        """Test warning gate generates rerun command."""
        commands = generate_rerun_commands([], ['covariance_condition'])
        assert len(commands) >= 1
        assert any('cov' in cmd.lower() for cmd in commands)


class TestCreateMetricsFromResults:
    """Tests for create_metrics_from_results function."""

    def test_empty_inputs(self):
        """Test empty inputs return empty metrics."""
        metrics = create_metrics_from_results()
        assert metrics == {}

    def test_injection_result(self):
        """Test injection result extraction."""
        class MockInjection:
            bias_sigma = 1.5
        metrics = create_metrics_from_results(injection_result=MockInjection())
        assert metrics['injection_bias_sigma'] == 1.5

    def test_null_result(self):
        """Test null test result extraction."""
        class MockNull:
            n_tests = 10
            n_passed = 8
        metrics = create_metrics_from_results(null_result=MockNull())
        assert metrics['null_pass_rate'] == 0.8

    def test_cov_info(self):
        """Test covariance info extraction."""
        cov_info = {
            'hartlap_factor': 0.85,
            'condition_number': 1e4,
        }
        metrics = create_metrics_from_results(cov_info=cov_info)
        assert metrics['hartlap_factor'] == 0.85
        assert metrics['condition_number'] == 1e4

    def test_map_results(self):
        """Test map consistency extraction."""
        map_results = [
            {'amplitude': 1.0, 'amplitude_err': 0.1},
            {'amplitude': 1.1, 'amplitude_err': 0.1},
        ]
        metrics = create_metrics_from_results(map_results=map_results)
        # diff = 0.1, combined_err = sqrt(0.01 + 0.01) ~ 0.14
        assert 'map_diff_sigma' in metrics
        assert np.isfinite(metrics['map_diff_sigma'])


class TestGateEvaluationResult:
    """Tests for GateEvaluationResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        gate_result = GateResult(
            name='test',
            status=GateStatus.PASS,
            metric=0.5,
            threshold=1.0,
            message='Test',
            is_critical=True,
        )
        result = GateEvaluationResult(
            overall_status="PASS",
            critical_passed=5,
            critical_failed=0,
            warnings=1,
            skipped=0,
            gate_results=[gate_result],
            failed_gates=[],
            warning_gates=['test_warn'],
            recommendation="All good",
            rerun_commands=[],
        )
        d = result.to_dict()
        assert d['overall_status'] == "PASS"
        assert d['critical_passed'] == 5
        assert len(d['gate_results']) == 1
