"""
Results packaging for publication-ready kSZ analysis bundles.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|------------------------------------------------|-------------|
| p(r)        | Pairwise momentum profile                       | μK          |
| A_kSZ       | kSZ amplitude (fitted)                         | dimensionless|
| σ(A)        | Amplitude uncertainty                           | dimensionless|
| S/N         | Detection significance                          | σ           |
| χ²          | Chi-squared statistic                           | dimensionless|
| PTE         | Probability to exceed                           | dimensionless|

Bundle Structure
----------------
results_bundle/
├── plots/               # All publication figures (PDF + PNG)
├── tables/              # CSV tables (p(r), covariance, null tests)
├── configs/             # YAML configuration files used
├── data/                # HDF5/NPY data files
├── manifest.json        # File listing with metadata
├── checksums.sha256     # SHA256 checksums for reproducibility
├── summary.json         # Machine-readable results summary
└── results.md           # Human-readable report (decision + recommendations)
"""

import json
import hashlib
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BundleManifest:
    """Manifest for results bundle."""
    bundle_name: str
    created_at: str
    pipeline_version: str
    git_commit: Optional[str]
    config_hash: str
    files: List[Dict[str, Any]]
    total_size_bytes: int
    n_files: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResultsSummary:
    """Machine-readable results summary."""
    # Overall status
    overall_status: str  # PASS, FAIL, INCONCLUSIVE
    recommendation: str

    # Detection
    detection_significance: float  # σ
    amplitude: float
    amplitude_error: float
    amplitude_snr: float

    # Data summary
    catalog: str
    n_galaxies: int
    z_bins: List[str]
    map_sources: List[str]

    # Quality metrics
    n_critical_gates_passed: int
    n_critical_gates_total: int
    n_warnings: int
    null_test_pass_rate: float
    injection_bias_sigma: float
    transfer_bias_percent: float
    tsz_stability_sigma: float
    hartlap_factor: float

    # Referee checks
    referee_check_results: Dict[str, str]

    # Failed gates (if any)
    failed_gates: List[str]
    warning_gates: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultsPackager:
    """
    Bundle builder for complete kSZ analysis outputs.

    Creates a self-contained results package with:
    - All publication figures
    - Data tables (CSV)
    - Configuration files
    - Checksums for reproducibility
    - Human and machine-readable summaries
    """

    def __init__(self, output_dir: str, bundle_name: Optional[str] = None):
        """
        Initialize packager.

        Parameters
        ----------
        output_dir : str
            Directory to create bundle in
        bundle_name : str, optional
            Name for bundle folder (default: timestamped)
        """
        self.output_dir = Path(output_dir)
        if bundle_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bundle_name = f"ksz_results_{timestamp}"
        self.bundle_name = bundle_name
        self.bundle_path = self.output_dir / bundle_name

        # Subdirectories
        self.plots_dir = self.bundle_path / "plots"
        self.tables_dir = self.bundle_path / "tables"
        self.configs_dir = self.bundle_path / "configs"
        self.data_dir = self.bundle_path / "data"

        # Tracking
        self.files_added: List[Dict[str, Any]] = []
        self.checksums: Dict[str, str] = {}

    def create_bundle(
        self,
        phase34_result: Any,
        gate_result: Any,
        config: Dict[str, Any],
        additional_plots: Optional[List[Path]] = None,
        additional_data: Optional[Dict[str, Path]] = None,
    ) -> Path:
        """
        Create complete results bundle.

        Parameters
        ----------
        phase34_result : Phase34Result
            Complete Phase 3-4 results
        gate_result : GateEvaluationResult
            Gate evaluation results
        config : dict
            Configuration used for run
        additional_plots : list of Path, optional
            Extra plot files to include
        additional_data : dict, optional
            Additional data files {name: path}

        Returns
        -------
        Path
            Path to created bundle
        """
        logger.info(f"Creating results bundle: {self.bundle_path}")

        # Create directory structure
        self._create_directories()

        # Add plots
        self._add_plots(phase34_result, additional_plots)

        # Add tables
        self._add_tables(phase34_result)

        # Add configuration
        self._add_config(config)

        # Add data files
        self._add_data(phase34_result, additional_data)

        # Create summary
        summary = self._create_summary(phase34_result, gate_result, config)
        self._write_summary(summary)

        # Create decision report
        self._write_decision_report(phase34_result, gate_result, summary)

        # Create manifest
        manifest = self._create_manifest(config)
        self._write_manifest(manifest)

        # Write checksums
        self._write_checksums()

        logger.info(f"Bundle created: {self.bundle_path}")
        logger.info(f"  Files: {len(self.files_added)}")
        logger.info(f"  Total size: {self._format_size(manifest.total_size_bytes)}")

        return self.bundle_path

    def _create_directories(self):
        """Create bundle directory structure."""
        self.bundle_path.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

    def _add_plots(
        self,
        result: Any,
        additional: Optional[List[Path]] = None,
    ):
        """Add plot files to bundle."""
        # Copy plots from result
        if hasattr(result, 'plots') and result.plots:
            for plot_name, plot_path in result.plots.items():
                if plot_path and Path(plot_path).exists():
                    self._copy_file(Path(plot_path), self.plots_dir, category="plot")

        # Copy additional plots
        if additional:
            for plot_path in additional:
                if plot_path.exists():
                    self._copy_file(plot_path, self.plots_dir, category="plot")

    def _add_tables(self, result: Any):
        """Add CSV tables to bundle."""
        import csv

        # Pairwise momentum table per z-bin
        if hasattr(result, 'z_bin_results') and result.z_bin_results:
            for zbin_result in result.z_bin_results:
                z_label = zbin_result.z_bin_label.replace(" ", "_").replace("<", "").replace(">", "")
                table_path = self.tables_dir / f"pairwise_momentum_{z_label}.csv"

                with open(table_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['r_Mpc_h', 'p_muK', 'p_err_muK', 'p_theory_muK'])

                    r = np.asarray(zbin_result.r_centers)
                    p = np.asarray(zbin_result.pairwise_momentum)
                    perr = np.asarray(zbin_result.pairwise_momentum_err)
                    pth = np.asarray(zbin_result.theory_template) if zbin_result.theory_template is not None else np.full_like(p, np.nan)

                    for i in range(len(r)):
                        writer.writerow([
                            f"{r[i]:.2f}",
                            f"{p[i]:.4f}",
                            f"{perr[i]:.4f}",
                            f"{pth[i]:.4f}" if np.isfinite(pth[i]) else "nan"
                        ])

                self._register_file(table_path, category="table")

        # Covariance matrix
        if hasattr(result, 'covariance') and result.covariance is not None:
            cov_path = self.tables_dir / "covariance_matrix.csv"
            np.savetxt(cov_path, result.covariance, delimiter=',', fmt='%.6e')
            self._register_file(cov_path, category="table")

        # Null test summary table
        if hasattr(result, 'null_test_result') and result.null_test_result is not None:
            null_path = self.tables_dir / "null_test_summary.csv"
            null_result = result.null_test_result

            with open(null_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['test_name', 'chi2', 'ndof', 'pte', 'passed'])

                if hasattr(null_result, 'test_results'):
                    for test in null_result.test_results:
                        writer.writerow([
                            test.name,
                            f"{test.chi2:.2f}",
                            test.ndof,
                            f"{test.pte:.4f}",
                            "PASS" if test.passed else "FAIL"
                        ])

            self._register_file(null_path, category="table")

        # Gate evaluation table
        if hasattr(result, 'gate_result') and result.gate_result is not None:
            gate_path = self.tables_dir / "gate_evaluation.csv"

            with open(gate_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['gate_name', 'status', 'metric', 'threshold', 'is_critical', 'description'])

                for gr in result.gate_result.gate_results:
                    writer.writerow([
                        gr.name,
                        gr.status.value,
                        f"{gr.metric:.4f}" if np.isfinite(gr.metric) else "N/A",
                        f"{gr.threshold:.4f}",
                        "CRITICAL" if gr.is_critical else "WARNING",
                        gr.message
                    ])

            self._register_file(gate_path, category="table")

        # Amplitude summary table
        if hasattr(result, 'z_bin_results') and result.z_bin_results:
            amp_path = self.tables_dir / "amplitude_summary.csv"

            with open(amp_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['z_bin', 'z_mean', 'amplitude', 'amplitude_err', 'snr', 'n_galaxies', 'n_pairs'])

                for zbin_result in result.z_bin_results:
                    writer.writerow([
                        zbin_result.z_bin_label,
                        f"{zbin_result.z_mean:.3f}",
                        f"{zbin_result.amplitude:.4f}",
                        f"{zbin_result.amplitude_err:.4f}",
                        f"{zbin_result.snr:.2f}",
                        zbin_result.n_galaxies,
                        zbin_result.n_pairs
                    ])

                # Joint amplitude if available
                if hasattr(result, 'joint_amplitude') and result.joint_amplitude is not None:
                    writer.writerow([
                        "JOINT",
                        "-",
                        f"{result.joint_amplitude:.4f}",
                        f"{result.joint_amplitude_err:.4f}",
                        f"{result.joint_snr:.2f}",
                        sum(zr.n_galaxies for zr in result.z_bin_results),
                        sum(zr.n_pairs for zr in result.z_bin_results)
                    ])

            self._register_file(amp_path, category="table")

    def _add_config(self, config: Dict[str, Any]):
        """Add configuration files to bundle."""
        import yaml

        # Main config
        config_path = self.configs_dir / "run_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        self._register_file(config_path, category="config")

    def _add_data(
        self,
        result: Any,
        additional: Optional[Dict[str, Path]] = None,
    ):
        """Add data files (HDF5/NPY) to bundle."""
        # Covariance as numpy
        if hasattr(result, 'covariance') and result.covariance is not None:
            cov_path = self.data_dir / "covariance.npy"
            np.save(cov_path, result.covariance)
            self._register_file(cov_path, category="data")

        # Per-z-bin data
        if hasattr(result, 'z_bin_results') and result.z_bin_results:
            for zbin_result in result.z_bin_results:
                z_label = zbin_result.z_bin_label.replace(" ", "_").replace("<", "").replace(">", "")

                # Save pairwise momentum
                data_path = self.data_dir / f"pairwise_{z_label}.npz"
                np.savez(
                    data_path,
                    r_centers=zbin_result.r_centers,
                    pairwise_momentum=zbin_result.pairwise_momentum,
                    pairwise_momentum_err=zbin_result.pairwise_momentum_err,
                    theory_template=zbin_result.theory_template if zbin_result.theory_template is not None else [],
                    z_mean=zbin_result.z_mean,
                    amplitude=zbin_result.amplitude,
                    amplitude_err=zbin_result.amplitude_err,
                )
                self._register_file(data_path, category="data")

        # Additional data
        if additional:
            for name, path in additional.items():
                if path.exists():
                    self._copy_file(path, self.data_dir, category="data")

    def _create_summary(
        self,
        result: Any,
        gate_result: Any,
        config: Dict[str, Any],
    ) -> ResultsSummary:
        """Create machine-readable results summary."""
        # Extract detection info
        if hasattr(result, 'joint_snr') and result.joint_snr is not None:
            detection_sig = result.joint_snr
            amplitude = result.joint_amplitude
            amplitude_err = result.joint_amplitude_err
        elif hasattr(result, 'z_bin_results') and result.z_bin_results:
            # Use highest SNR z-bin
            best = max(result.z_bin_results, key=lambda x: x.snr)
            detection_sig = best.snr
            amplitude = best.amplitude
            amplitude_err = best.amplitude_err
        else:
            detection_sig = 0.0
            amplitude = 0.0
            amplitude_err = 1.0

        # Z-bins
        z_bins = []
        if hasattr(result, 'z_bin_results'):
            z_bins = [zr.z_bin_label for zr in result.z_bin_results]

        # Map sources
        map_sources = config.get('map_sources', [])
        if isinstance(map_sources, str):
            map_sources = [map_sources]

        # Quality metrics from gate_result
        null_pass_rate = 0.0
        injection_bias = 0.0
        transfer_bias = 0.0
        tsz_stability = 0.0
        hartlap = 0.0

        if hasattr(result, 'metrics') and result.metrics:
            null_pass_rate = result.metrics.get('null_pass_rate', 0.0)
            injection_bias = result.metrics.get('injection_bias_sigma', 0.0)
            transfer_bias = result.metrics.get('transfer_bias', 0.0) * 100
            tsz_stability = result.metrics.get('tsz_delta_sigma', 0.0)
            hartlap = result.metrics.get('hartlap_factor', 0.0)

        # Referee check results
        referee_results = {}
        if hasattr(result, 'referee_results') and result.referee_results:
            for check_name, check_result in result.referee_results.items():
                if hasattr(check_result, 'passed'):
                    referee_results[check_name] = "PASS" if check_result.passed else "FAIL"
                elif isinstance(check_result, dict):
                    referee_results[check_name] = "PASS" if check_result.get('passed', False) else "FAIL"

        return ResultsSummary(
            overall_status=gate_result.overall_status,
            recommendation=gate_result.recommendation,
            detection_significance=detection_sig,
            amplitude=amplitude,
            amplitude_error=amplitude_err,
            amplitude_snr=detection_sig,
            catalog=config.get('catalog', 'unknown'),
            n_galaxies=sum(zr.n_galaxies for zr in result.z_bin_results) if hasattr(result, 'z_bin_results') else 0,
            z_bins=z_bins,
            map_sources=map_sources,
            n_critical_gates_passed=gate_result.critical_passed,
            n_critical_gates_total=gate_result.critical_passed + gate_result.critical_failed,
            n_warnings=gate_result.warnings,
            null_test_pass_rate=null_pass_rate,
            injection_bias_sigma=injection_bias,
            transfer_bias_percent=transfer_bias,
            tsz_stability_sigma=tsz_stability,
            hartlap_factor=hartlap,
            referee_check_results=referee_results,
            failed_gates=gate_result.failed_gates,
            warning_gates=gate_result.warning_gates,
        )

    def _write_summary(self, summary: ResultsSummary):
        """Write machine-readable summary."""
        summary_path = self.bundle_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        self._register_file(summary_path, category="summary")

    def _write_decision_report(
        self,
        result: Any,
        gate_result: Any,
        summary: ResultsSummary,
    ):
        """Write human-readable decision report."""
        report_path = self.bundle_path / "results.md"

        lines = []
        lines.append("# DESI DR1 Pairwise kSZ Analysis Results")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")

        # Decision banner
        status = summary.overall_status
        if status == "PASS":
            lines.append("## Decision: PASS")
            lines.append("")
            lines.append("All critical gates passed. Results are publication-ready.")
        elif status == "FAIL":
            lines.append("## Decision: FAIL")
            lines.append("")
            lines.append("Critical gates failed. Results require investigation before publication.")
        else:
            lines.append("## Decision: INCONCLUSIVE")
            lines.append("")
            lines.append("Multiple warnings present. Manual review recommended.")

        lines.append("")
        lines.append(f"**Recommendation**: {summary.recommendation}")
        lines.append("")

        # Detection summary
        lines.append("---")
        lines.append("")
        lines.append("## Detection Summary")
        lines.append("")
        lines.append(f"- **kSZ Amplitude**: A = {summary.amplitude:.4f} +/- {summary.amplitude_error:.4f}")
        lines.append(f"- **Detection Significance**: {summary.detection_significance:.1f} sigma")
        lines.append(f"- **Catalog**: {summary.catalog}")
        lines.append(f"- **N galaxies**: {summary.n_galaxies:,}")
        lines.append(f"- **Z bins**: {', '.join(summary.z_bins)}")
        lines.append(f"- **Map sources**: {', '.join(summary.map_sources)}")
        lines.append("")

        # Per-bin results
        if hasattr(result, 'z_bin_results') and result.z_bin_results:
            lines.append("### Per-bin Results")
            lines.append("")
            lines.append("| Z bin | z_mean | Amplitude | Error | S/N | N_gal | N_pairs |")
            lines.append("|-------|--------|-----------|-------|-----|-------|---------|")

            for zr in result.z_bin_results:
                lines.append(f"| {zr.z_bin_label} | {zr.z_mean:.3f} | {zr.amplitude:.4f} | {zr.amplitude_err:.4f} | {zr.snr:.1f} | {zr.n_galaxies:,} | {zr.n_pairs:,} |")

            if hasattr(result, 'joint_amplitude') and result.joint_amplitude is not None:
                lines.append(f"| **Joint** | - | **{result.joint_amplitude:.4f}** | **{result.joint_amplitude_err:.4f}** | **{result.joint_snr:.1f}** | - | - |")

            lines.append("")

        # Gate evaluation
        lines.append("---")
        lines.append("")
        lines.append("## Gate Evaluation")
        lines.append("")
        lines.append(f"**Critical Gates**: {summary.n_critical_gates_passed}/{summary.n_critical_gates_total} passed")
        lines.append(f"**Warnings**: {summary.n_warnings}")
        lines.append("")

        if gate_result.failed_gates:
            lines.append("### Failed Critical Gates")
            lines.append("")
            for gate in gate_result.failed_gates:
                lines.append(f"- **{gate}**")
            lines.append("")

        if gate_result.warning_gates:
            lines.append("### Warning Gates")
            lines.append("")
            for gate in gate_result.warning_gates:
                lines.append(f"- {gate}")
            lines.append("")

        # Quality metrics table
        lines.append("### Quality Metrics")
        lines.append("")
        lines.append("| Metric | Value | Threshold | Status |")
        lines.append("|--------|-------|-----------|--------|")
        lines.append(f"| Null test pass rate | {summary.null_test_pass_rate:.1%} | >= 80% | {'PASS' if summary.null_test_pass_rate >= 0.8 else 'FAIL'} |")
        lines.append(f"| Injection bias | {summary.injection_bias_sigma:.2f} sigma | < 2 sigma | {'PASS' if abs(summary.injection_bias_sigma) < 2 else 'FAIL'} |")
        lines.append(f"| Transfer bias | {summary.transfer_bias_percent:.1f}% | < 5% | {'PASS' if abs(summary.transfer_bias_percent) < 5 else 'FAIL'} |")
        lines.append(f"| tSZ stability | {summary.tsz_stability_sigma:.2f} sigma | < 1 sigma | {'PASS' if summary.tsz_stability_sigma < 1 else 'FAIL'} |")
        lines.append(f"| Hartlap factor | {summary.hartlap_factor:.3f} | > 0.5 | {'PASS' if summary.hartlap_factor > 0.5 else 'FAIL'} |")
        lines.append("")

        # Referee checks
        if summary.referee_check_results:
            lines.append("### Referee Attack Checks")
            lines.append("")
            lines.append("| Check | Status |")
            lines.append("|-------|--------|")
            for check, status in summary.referee_check_results.items():
                lines.append(f"| {check} | {status} |")
            lines.append("")

        # Rerun commands
        if gate_result.rerun_commands:
            lines.append("---")
            lines.append("")
            lines.append("## Suggested Remediation Commands")
            lines.append("")
            lines.append("```bash")
            for cmd in gate_result.rerun_commands:
                lines.append(cmd)
            lines.append("```")
            lines.append("")

        # Bundle contents
        lines.append("---")
        lines.append("")
        lines.append("## Bundle Contents")
        lines.append("")
        lines.append("- `plots/` - Publication figures (PDF + PNG)")
        lines.append("- `tables/` - Data tables (CSV)")
        lines.append("- `configs/` - Configuration files (YAML)")
        lines.append("- `data/` - Raw data (NPY/NPZ)")
        lines.append("- `summary.json` - Machine-readable results")
        lines.append("- `checksums.sha256` - File integrity verification")
        lines.append("")

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        self._register_file(report_path, category="report")

    def _create_manifest(self, config: Dict[str, Any]) -> BundleManifest:
        """Create bundle manifest."""
        # Compute config hash
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        # Get git commit if available
        git_commit = None
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()[:8]
        except Exception:
            pass

        # Compute total size
        total_size = sum(f['size_bytes'] for f in self.files_added)

        return BundleManifest(
            bundle_name=self.bundle_name,
            created_at=datetime.now().isoformat(),
            pipeline_version="1.0.0",
            git_commit=git_commit,
            config_hash=config_hash,
            files=self.files_added,
            total_size_bytes=total_size,
            n_files=len(self.files_added),
        )

    def _write_manifest(self, manifest: BundleManifest):
        """Write manifest JSON."""
        manifest_path = self.bundle_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest.to_dict(), f, indent=2)
        # Don't register manifest itself to avoid recursion

    def _write_checksums(self):
        """Write SHA256 checksums file."""
        checksum_path = self.bundle_path / "checksums.sha256"
        with open(checksum_path, 'w') as f:
            for filepath, checksum in sorted(self.checksums.items()):
                # Use relative path
                rel_path = Path(filepath).relative_to(self.bundle_path)
                f.write(f"{checksum}  {rel_path}\n")

    def _copy_file(self, src: Path, dest_dir: Path, category: str):
        """Copy file to bundle and register it."""
        dest_path = dest_dir / src.name
        shutil.copy2(src, dest_path)
        self._register_file(dest_path, category=category)

    def _register_file(self, path: Path, category: str):
        """Register file in manifest and compute checksum."""
        # Compute checksum
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        checksum = sha256.hexdigest()

        self.checksums[str(path)] = checksum

        self.files_added.append({
            'path': str(path.relative_to(self.bundle_path)),
            'category': category,
            'size_bytes': path.stat().st_size,
            'sha256': checksum,
        })

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size for display."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


def create_results_bundle(
    output_dir: str,
    phase34_result: Any,
    gate_result: Any,
    config: Dict[str, Any],
    bundle_name: Optional[str] = None,
    additional_plots: Optional[List[Path]] = None,
    additional_data: Optional[Dict[str, Path]] = None,
) -> Path:
    """
    Convenience function to create results bundle.

    Parameters
    ----------
    output_dir : str
        Directory to create bundle in
    phase34_result : Phase34Result
        Complete Phase 3-4 results
    gate_result : GateEvaluationResult
        Gate evaluation results
    config : dict
        Configuration used for run
    bundle_name : str, optional
        Name for bundle folder
    additional_plots : list of Path, optional
        Extra plot files to include
    additional_data : dict, optional
        Additional data files {name: path}

    Returns
    -------
    Path
        Path to created bundle
    """
    packager = ResultsPackager(output_dir, bundle_name)
    return packager.create_bundle(
        phase34_result=phase34_result,
        gate_result=gate_result,
        config=config,
        additional_plots=additional_plots,
        additional_data=additional_data,
    )
