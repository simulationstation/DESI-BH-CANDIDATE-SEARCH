"""
Gate definitions and evaluation for pipeline quality control.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|------------------------------------------------|-------------|
| G_i         | Gate i result (PASS/FAIL/WARN)                  | categorical |
| τ_i         | Threshold for gate i                            | varies      |
| m_i         | Measured metric for gate i                      | varies      |

Gate Status Logic
-----------------
For each gate:
    PASS if m_i satisfies threshold τ_i
    FAIL if m_i violates threshold and gate is critical
    WARN if m_i violates threshold and gate is non-critical

Overall Status:
    PASS: All critical gates pass, ≤1 warning
    FAIL: Any critical gate fails
    INCONCLUSIVE: Critical pass but multiple warnings or marginal metrics
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Gate evaluation status."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class GateResult:
    """Result of evaluating a single gate."""
    name: str
    status: GateStatus
    metric: float
    threshold: float
    message: str
    is_critical: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'metric': float(self.metric) if np.isfinite(self.metric) else None,
            'threshold': float(self.threshold),
            'message': self.message,
            'is_critical': self.is_critical,
            'details': self.details,
        }


@dataclass
class Gate:
    """
    Gate definition for quality control.

    A gate checks whether a metric satisfies a threshold.
    Critical gates cause FAIL status; non-critical cause WARN.
    """
    name: str
    description: str
    threshold: float
    comparator: str  # 'lt', 'gt', 'le', 'ge', 'eq', 'abs_lt'
    is_critical: bool = True
    metric_key: Optional[str] = None  # Key in metrics dict

    def evaluate(self, metric: float) -> GateStatus:
        """Evaluate gate given a metric value."""
        if not np.isfinite(metric):
            return GateStatus.SKIP

        passed = False
        if self.comparator == 'lt':
            passed = metric < self.threshold
        elif self.comparator == 'gt':
            passed = metric > self.threshold
        elif self.comparator == 'le':
            passed = metric <= self.threshold
        elif self.comparator == 'ge':
            passed = metric >= self.threshold
        elif self.comparator == 'eq':
            passed = np.isclose(metric, self.threshold)
        elif self.comparator == 'abs_lt':
            passed = abs(metric) < self.threshold

        if passed:
            return GateStatus.PASS
        elif self.is_critical:
            return GateStatus.FAIL
        else:
            return GateStatus.WARN


# =============================================================================
# Critical Gates (cause FAIL if violated)
# =============================================================================

CRITICAL_GATES = {
    'injection_bias': Gate(
        name='injection_bias',
        description='Signal injection recovery bias < 2σ',
        threshold=2.0,
        comparator='abs_lt',
        is_critical=True,
        metric_key='injection_bias_sigma',
    ),
    'null_suite_pass_rate': Gate(
        name='null_suite_pass_rate',
        description='Null test suite pass rate ≥ 80%',
        threshold=0.8,
        comparator='ge',
        is_critical=True,
        metric_key='null_pass_rate',
    ),
    'transfer_test': Gate(
        name='transfer_test',
        description='Transfer function test passes (bias < 5%)',
        threshold=0.05,
        comparator='abs_lt',
        is_critical=True,
        metric_key='transfer_bias',
    ),
    'tsz_sweep_stability': Gate(
        name='tsz_sweep_stability',
        description='tSZ mask sweep amplitude stability < 1σ',
        threshold=1.0,
        comparator='lt',
        is_critical=True,
        metric_key='tsz_delta_sigma',
    ),
    'covariance_hartlap': Gate(
        name='covariance_hartlap',
        description='Hartlap factor > 0.5 (K > N_bins + 2)',
        threshold=0.5,
        comparator='gt',
        is_critical=True,
        metric_key='hartlap_factor',
    ),
    'map_consistency': Gate(
        name='map_consistency',
        description='Independent map results consistent within 2σ',
        threshold=2.0,
        comparator='lt',
        is_critical=True,
        metric_key='map_diff_sigma',
    ),
}

# =============================================================================
# Warning Gates (cause WARN if violated)
# =============================================================================

WARNING_GATES = {
    'covariance_condition': Gate(
        name='covariance_condition',
        description='Covariance condition number < 1e6',
        threshold=1e6,
        comparator='lt',
        is_critical=False,
        metric_key='condition_number',
    ),
    'look_elsewhere': Gate(
        name='look_elsewhere',
        description='Look-elsewhere adjusted p-value > 0.01',
        threshold=0.01,
        comparator='gt',
        is_critical=False,
        metric_key='adjusted_pvalue',
    ),
    'anisotropy_residual': Gate(
        name='anisotropy_residual',
        description='Temperature anisotropy residual < 3σ',
        threshold=3.0,
        comparator='lt',
        is_critical=False,
        metric_key='anisotropy_sigma',
    ),
    'weight_leverage': Gate(
        name='weight_leverage',
        description='Weight leverage stability < 1σ',
        threshold=1.0,
        comparator='lt',
        is_critical=False,
        metric_key='weight_leverage_sigma',
    ),
    'split_consistency': Gate(
        name='split_consistency',
        description='z-dependent split consistency < 2σ',
        threshold=2.0,
        comparator='lt',
        is_critical=False,
        metric_key='split_max_diff_sigma',
    ),
    'beam_sensitivity': Gate(
        name='beam_sensitivity',
        description='Beam perturbation sensitivity < 1σ',
        threshold=1.0,
        comparator='lt',
        is_critical=False,
        metric_key='beam_sensitivity_sigma',
    ),
    'ymap_regression': Gate(
        name='ymap_regression',
        description='y-map regression amplitude shift < 2σ',
        threshold=2.0,
        comparator='lt',
        is_critical=False,
        metric_key='ymap_shift_sigma',
    ),
}


@dataclass
class GateEvaluationResult:
    """Result of evaluating all gates."""
    overall_status: str  # PASS, FAIL, INCONCLUSIVE
    critical_passed: int
    critical_failed: int
    warnings: int
    skipped: int
    gate_results: List[GateResult]
    failed_gates: List[str]
    warning_gates: List[str]
    recommendation: str
    rerun_commands: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_status': self.overall_status,
            'critical_passed': self.critical_passed,
            'critical_failed': self.critical_failed,
            'warnings': self.warnings,
            'skipped': self.skipped,
            'gate_results': [g.to_dict() for g in self.gate_results],
            'failed_gates': self.failed_gates,
            'warning_gates': self.warning_gates,
            'recommendation': self.recommendation,
            'rerun_commands': self.rerun_commands,
        }


def evaluate_gates(
    metrics: Dict[str, float],
    include_warnings: bool = True,
) -> GateEvaluationResult:
    """
    Evaluate all gates given a metrics dictionary.

    Parameters
    ----------
    metrics : dict
        Dictionary mapping metric keys to values
    include_warnings : bool
        Whether to include warning gates

    Returns
    -------
    GateEvaluationResult
        Complete evaluation results
    """
    gate_results = []
    failed_gates = []
    warning_gates = []

    critical_passed = 0
    critical_failed = 0
    warnings = 0
    skipped = 0

    # Evaluate critical gates
    for gate_name, gate in CRITICAL_GATES.items():
        metric_value = metrics.get(gate.metric_key, np.nan)
        status = gate.evaluate(metric_value)

        result = GateResult(
            name=gate.name,
            status=status,
            metric=metric_value,
            threshold=gate.threshold,
            message=gate.description,
            is_critical=True,
        )
        gate_results.append(result)

        if status == GateStatus.PASS:
            critical_passed += 1
        elif status == GateStatus.FAIL:
            critical_failed += 1
            failed_gates.append(gate_name)
        elif status == GateStatus.SKIP:
            skipped += 1

    # Evaluate warning gates
    if include_warnings:
        for gate_name, gate in WARNING_GATES.items():
            metric_value = metrics.get(gate.metric_key, np.nan)
            status = gate.evaluate(metric_value)

            result = GateResult(
                name=gate.name,
                status=status,
                metric=metric_value,
                threshold=gate.threshold,
                message=gate.description,
                is_critical=False,
            )
            gate_results.append(result)

            if status == GateStatus.WARN:
                warnings += 1
                warning_gates.append(gate_name)
            elif status == GateStatus.SKIP:
                skipped += 1

    # Determine overall status
    if critical_failed > 0:
        overall_status = "FAIL"
        recommendation = "Address failed critical gates before proceeding."
    elif warnings > 1:
        overall_status = "INCONCLUSIVE"
        recommendation = "Multiple warnings present. Review and consider re-running."
    elif warnings == 1:
        overall_status = "PASS"
        recommendation = "All critical gates passed with minor warning. Safe to proceed."
    else:
        overall_status = "PASS"
        recommendation = "All gates passed. Results are publication-ready."

    # Generate rerun commands based on failures
    rerun_commands = generate_rerun_commands(failed_gates, warning_gates)

    logger.info(f"Gate evaluation: {overall_status} "
                f"(critical: {critical_passed}/{critical_passed + critical_failed}, "
                f"warnings: {warnings})")

    return GateEvaluationResult(
        overall_status=overall_status,
        critical_passed=critical_passed,
        critical_failed=critical_failed,
        warnings=warnings,
        skipped=skipped,
        gate_results=gate_results,
        failed_gates=failed_gates,
        warning_gates=warning_gates,
        recommendation=recommendation,
        rerun_commands=rerun_commands,
    )


def generate_rerun_commands(
    failed_gates: List[str],
    warning_gates: List[str],
) -> List[str]:
    """Generate CLI commands to address gate failures."""
    commands = []

    gate_to_command = {
        'injection_bias': 'python -m desi_ksz.cli injection-test -n 200 --mode template',
        'null_suite_pass_rate': 'python -m desi_ksz.cli null-suite -n 500',
        'transfer_test': 'python -m desi_ksz.cli transfer-test -n 10 --nside 1024',
        'tsz_sweep_stability': 'python -m desi_ksz.cli tsz-sweep --mask-radii 5,10,15,20,25,30',
        'covariance_hartlap': 'python -m desi_ksz.cli covariance --n-regions 200',
        'map_consistency': 'python -m desi_ksz.cli run-phase34 --require-two-maps',
        'covariance_condition': 'python -m desi_ksz.cli auto-cov --target-condition 1e5',
        'look_elsewhere': '# Review aperture/filter choices manually',
        'anisotropy_residual': 'python -m desi_ksz.cli validate-map --run-beam-test',
        'weight_leverage': '# Check for outlier galaxies in catalog',
        'split_consistency': '# Investigate z-dependent systematics',
        'beam_sensitivity': '# Run with perturbed beam parameters',
        'ymap_regression': 'python -m desi_ksz.cli tsz-sweep --with-ymap-regression',
    }

    for gate in failed_gates:
        if gate in gate_to_command:
            commands.append(f"# Fix {gate}:")
            commands.append(gate_to_command[gate])

    for gate in warning_gates:
        if gate in gate_to_command:
            commands.append(f"# Address warning {gate}:")
            commands.append(gate_to_command[gate])

    return commands


def create_metrics_from_results(
    injection_result: Optional[Any] = None,
    null_result: Optional[Any] = None,
    transfer_result: Optional[Any] = None,
    tsz_result: Optional[Any] = None,
    cov_info: Optional[Dict] = None,
    map_results: Optional[List[Dict]] = None,
    referee_results: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Create metrics dictionary from various result objects.

    Parameters
    ----------
    injection_result : InjectionTestResult, optional
    null_result : NullSuiteResult, optional
    transfer_result : TransferFunctionTestResult, optional
    tsz_result : ClusterMaskSweepResult, optional
    cov_info : dict, optional
    map_results : list of dicts, optional
    referee_results : dict, optional

    Returns
    -------
    dict
        Metrics dictionary for gate evaluation
    """
    metrics = {}

    # Injection test metrics
    if injection_result is not None:
        metrics['injection_bias_sigma'] = getattr(injection_result, 'bias_sigma', np.nan)

    # Null test metrics
    if null_result is not None:
        if hasattr(null_result, 'n_tests') and null_result.n_tests > 0:
            metrics['null_pass_rate'] = null_result.n_passed / null_result.n_tests
        else:
            metrics['null_pass_rate'] = np.nan

    # Transfer function metrics
    if transfer_result is not None:
        metrics['transfer_bias'] = getattr(transfer_result, 'bias', np.nan)

    # tSZ sweep metrics
    if tsz_result is not None:
        # Compute max delta amplitude in sigma
        if hasattr(tsz_result, 'amplitudes') and hasattr(tsz_result, 'amplitude_errors'):
            amps = np.array(tsz_result.amplitudes)
            errs = np.array(tsz_result.amplitude_errors)
            valid = np.isfinite(amps) & np.isfinite(errs) & (errs > 0)
            if np.sum(valid) >= 2:
                delta = np.max(amps[valid]) - np.min(amps[valid])
                avg_err = np.mean(errs[valid])
                metrics['tsz_delta_sigma'] = delta / avg_err if avg_err > 0 else np.nan
            else:
                metrics['tsz_delta_sigma'] = np.nan

    # Covariance metrics
    if cov_info is not None:
        metrics['hartlap_factor'] = cov_info.get('hartlap_factor', np.nan)
        metrics['condition_number'] = cov_info.get('condition_number', np.nan)

    # Map consistency metrics
    if map_results is not None and len(map_results) >= 2:
        # Compare amplitudes from different maps
        amps = [r.get('amplitude', np.nan) for r in map_results]
        errs = [r.get('amplitude_err', np.nan) for r in map_results]
        if len(amps) >= 2 and all(np.isfinite(amps[:2])) and all(np.isfinite(errs[:2])):
            diff = abs(amps[0] - amps[1])
            combined_err = np.sqrt(errs[0]**2 + errs[1]**2)
            metrics['map_diff_sigma'] = diff / combined_err if combined_err > 0 else np.nan
        else:
            metrics['map_diff_sigma'] = np.nan

    # Referee check metrics
    if referee_results is not None:
        metrics['adjusted_pvalue'] = referee_results.get('look_elsewhere_pvalue', np.nan)
        metrics['anisotropy_sigma'] = referee_results.get('anisotropy_sigma', np.nan)
        metrics['weight_leverage_sigma'] = referee_results.get('weight_leverage_sigma', np.nan)
        metrics['split_max_diff_sigma'] = referee_results.get('split_max_diff_sigma', np.nan)
        metrics['beam_sensitivity_sigma'] = referee_results.get('beam_sensitivity_sigma', np.nan)
        metrics['ymap_shift_sigma'] = referee_results.get('ymap_shift_sigma', np.nan)

    return metrics
