"""
Full pipeline run for rare_companions multi-experiment search.

This script implements the complete staged pipeline:
- Stage 0: Environment setup and sanity checks
- Stage 1: RV-only scan on all targets
- Stage 2: Fast screening per experiment (E1-E8)
- Stage 3: Deep inference for top candidates
- Stage 4: Final scoring, leaderboards, and report
"""

import os
import sys
import json
import logging
import platform
import subprocess
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DESIDataDiscovery:
    """Discover and download DESI DR1 per-epoch RV data."""

    # Common locations to search for DESI data
    SEARCH_PATHS = [
        "data",
        "./data",
        "../data",
        "/mnt/data",
        "/scratch",
        os.environ.get("DESI_ROOT", ""),
        os.environ.get("DESI_SPECTRO_REDUX", ""),
        os.path.expanduser("~/desi_data"),
        "/home/primary/data",
    ]

    # DESI DR1 public endpoints (NOIRLab Data Lab)
    DESI_DR1_BASE = "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji"

    # Alternative: use rvspecfit per-epoch files
    RVSPECFIT_FILES = [
        "rvpix_exp-main-bright.fits",
        "rvpix_exp-main-dark.fits",
    ]

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.discovered_files = []

    def discover(self) -> List[str]:
        """Search for DESI per-epoch RV files."""
        found = []

        for base_path in self.SEARCH_PATHS:
            if not base_path:
                continue

            base = Path(base_path)
            if not base.exists():
                continue

            # Look for rvspecfit files
            for fname in self.RVSPECFIT_FILES:
                fpath = base / fname
                if fpath.exists():
                    logger.info(f"Found DESI data: {fpath}")
                    found.append(str(fpath))

            # Also check for per-healpix files
            healpix_dir = base / "healpix"
            if healpix_dir.exists():
                fits_files = list(healpix_dir.glob("**/*.fits"))
                if fits_files:
                    logger.info(f"Found {len(fits_files)} healpix files in {healpix_dir}")
                    found.extend([str(f) for f in fits_files[:10]])  # Sample

        self.discovered_files = found
        return found

    def download_sample(self, output_dir: str = "data") -> List[str]:
        """
        Download a sample of DESI DR1 data for testing.

        Note: Full DESI DR1 is very large. This downloads only
        the rvspecfit summary files which contain per-epoch RVs.
        """
        os.makedirs(output_dir, exist_ok=True)
        downloaded = []

        # For real runs, we would download from:
        # https://data.desi.lbl.gov/public/edr/vac/edr/rvspecfit/v1.0/

        logger.warning("DESI data download not implemented - using synthetic data")
        logger.info("For real data, download rvspecfit files from DESI EDR/DR1")

        return downloaded


class StageCheckpoint:
    """Manage stage checkpoints for resumable runs."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.checkpoint_file = os.path.join(run_dir, "checkpoints.json")
        self.checkpoints = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)

    def mark_complete(self, stage: str, metadata: Dict = None):
        self.checkpoints[stage] = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.save()

    def is_complete(self, stage: str) -> bool:
        return self.checkpoints.get(stage, {}).get('completed', False)

    def get_metadata(self, stage: str) -> Dict:
        return self.checkpoints.get(stage, {}).get('metadata', {})


class RunManifest:
    """Create and manage run manifest with environment info."""

    def __init__(self, run_dir: str, config: Any):
        self.run_dir = run_dir
        self.config = config
        self.manifest_path = os.path.join(run_dir, "manifest.json")

    def create(self) -> Dict:
        """Create run manifest with full environment info."""
        manifest = {
            'run_name': self.config.run_name,
            'timestamp': datetime.now().isoformat(),
            'git_hash': self._get_git_hash(),
            'python_version': sys.version,
            'platform': platform.platform(),
            'hostname': platform.node(),
            'cpu_count': os.cpu_count(),
            'random_seed': self.config.random_seed,
            'pip_freeze': self._get_pip_freeze(),
        }

        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def _get_git_hash(self) -> str:
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except:
            return "unknown"

    def _get_pip_freeze(self) -> List[str]:
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout.strip().split('\n')[:50]  # Top 50 packages
        except:
            return []


class FullPipelineRunner:
    """Run the complete rare_companions pipeline."""

    def __init__(self, config, run_dir: str = None):
        from .config import Config, create_run_directory

        self.config = config
        self.run_dir = run_dir or create_run_directory(config)
        self.checkpoint = StageCheckpoint(self.run_dir)
        self.manifest = RunManifest(self.run_dir, config)

        # Set random seed
        np.random.seed(config.random_seed)

        # Setup logging to file
        log_file = os.path.join(self.run_dir, 'logs', 'pipeline.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)

    def run_all(self, skip_completed: bool = True):
        """Run all pipeline stages."""
        logger.info(f"Starting full pipeline run in {self.run_dir}")

        # Stage 0: Environment setup
        if not skip_completed or not self.checkpoint.is_complete('stage0'):
            self.stage0_environment()
        else:
            logger.info("Stage 0 already complete, skipping")

        # Stage 1: RV metrics scan
        if not skip_completed or not self.checkpoint.is_complete('stage1'):
            timeseries_list = self.stage1_rv_scan()
        else:
            logger.info("Stage 1 already complete, loading cached data")
            timeseries_list = self._load_stage1_results()

        if not timeseries_list:
            logger.error("No timeseries data available. Cannot continue.")
            return

        # Stage 2: Fast screening per experiment
        if not skip_completed or not self.checkpoint.is_complete('stage2'):
            experiment_results = self.stage2_fast_screen(timeseries_list)
        else:
            logger.info("Stage 2 already complete, loading cached results")
            experiment_results = self._load_stage2_results()

        # Stage 3: Deep inference (simplified for compute budget)
        if not skip_completed or not self.checkpoint.is_complete('stage3'):
            self.stage3_deep_inference(experiment_results)
        else:
            logger.info("Stage 3 already complete, skipping")

        # Stage 4: Final scoring and report
        self.stage4_final_report(experiment_results)

        logger.info("Pipeline complete!")
        return experiment_results

    def stage0_environment(self):
        """Stage 0: Environment setup and sanity checks."""
        logger.info("=" * 60)
        logger.info("STAGE 0: Environment Setup and Sanity Checks")
        logger.info("=" * 60)

        # Create manifest
        manifest = self.manifest.create()
        logger.info(f"Git hash: {manifest['git_hash']}")
        logger.info(f"Python: {manifest['python_version']}")
        logger.info(f"Platform: {manifest['platform']}")
        logger.info(f"CPU count: {manifest['cpu_count']}")
        logger.info(f"Random seed: {manifest['random_seed']}")

        # Run quick unit tests
        logger.info("Running quick unit tests...")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', '-q', '--tb=no',
                 'rare_companions/tests/', '-x'],
                capture_output=True, text=True, timeout=60,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if result.returncode != 0:
                logger.warning(f"Some tests failed: {result.stdout}")
            else:
                logger.info("Unit tests passed")
        except Exception as e:
            logger.warning(f"Could not run unit tests: {e}")

        # Discover DESI data
        discovery = DESIDataDiscovery()
        found_files = discovery.discover()
        logger.info(f"Found {len(found_files)} DESI data files")

        self.checkpoint.mark_complete('stage0', {
            'manifest': manifest,
            'data_files': found_files
        })

    def stage1_rv_scan(self) -> List:
        """Stage 1: RV metrics scan on all targets."""
        logger.info("=" * 60)
        logger.info("STAGE 1: RV Metrics Scan")
        logger.info("=" * 60)

        from .ingest.desi import DESIRVLoader
        from .ingest.unified import RVTimeSeries, RVEpoch
        from .rv.hardening import RVHardener

        # Try to load real DESI data
        timeseries_list = []

        try:
            bright_path = self.config.data_paths.desi_bright
            dark_path = self.config.data_paths.desi_dark

            if os.path.exists(bright_path) or os.path.exists(dark_path):
                logger.info("Loading DESI per-epoch RV data...")
                loader = DESIRVLoader(bright_path, dark_path)
                n_targets = loader.load(max_rv_err=self.config.selection.max_rv_err)
                logger.info(f"Loaded {n_targets} unique targets from DESI")

                # Get summary stats
                stats = loader.get_summary_stats()
                logger.info(f"  Total epochs: {stats.get('n_total_epochs', 0)}")
                logger.info(f"  Targets with 3+ epochs: {stats.get('n_with_3plus', 0)}")
                logger.info(f"  Targets with 5+ epochs: {stats.get('n_with_5plus', 0)}")

                # Convert to RVTimeSeries objects
                for targetid in loader.get_targets_with_min_epochs(self.config.selection.min_epochs):
                    mjd, rv, rv_err = loader.get_rv_arrays(targetid)

                    # Get metadata from first epoch
                    desi_epochs = loader.get_epochs(targetid)
                    if desi_epochs:
                        first_epoch = desi_epochs[0]
                        teff = first_epoch.teff if not np.isnan(first_epoch.teff) else None
                        logg = first_epoch.logg if not np.isnan(first_epoch.logg) else None
                        feh = first_epoch.feh if not np.isnan(first_epoch.feh) else None

                        # Estimate M1 from Teff if available
                        if teff and teff > 0:
                            # Simple Teff to mass mapping for main sequence
                            if teff > 7000:
                                M1 = 1.5
                            elif teff > 6000:
                                M1 = 1.1
                            elif teff > 5000:
                                M1 = 0.9
                            elif teff > 4000:
                                M1 = 0.7
                            else:
                                M1 = 0.4
                        else:
                            M1 = 0.5  # Default
                    else:
                        M1 = 0.5
                        teff = logg = feh = None

                    # Create RVEpoch objects
                    epochs = [
                        RVEpoch(
                            mjd=mjd[i],
                            rv=rv[i],
                            rv_err=rv_err[i],
                            instrument='DESI',
                            survey='MWS',
                            quality=1.0
                        )
                        for i in range(len(mjd))
                    ]

                    # Create RVTimeSeries
                    ts = RVTimeSeries(
                        targetid=targetid,
                        source_id=None,  # Would need Gaia crossmatch
                        ra=0.0,  # Would need from DESI targeting
                        dec=0.0,
                        epochs=epochs,
                        metadata={
                            'M1': M1,
                            'teff': teff,
                            'logg': logg,
                            'feh': feh,
                            'source': 'DESI_DR1'
                        }
                    )
                    timeseries_list.append(ts)

                logger.info(f"Created {len(timeseries_list)} RVTimeSeries with {self.config.selection.min_epochs}+ epochs")

        except Exception as e:
            logger.error(f"Could not load real DESI data: {e}")
            import traceback
            traceback.print_exc()

        # If no real data, generate synthetic for demonstration
        if not timeseries_list:
            logger.warning("No real DESI data found. Generating synthetic data for demonstration.")
            timeseries_list = self._generate_synthetic_data(n_targets=500)

        logger.info(f"Total targets for Stage 1: {len(timeseries_list)}")

        # Apply basic filters
        filtered = []
        for ts in timeseries_list:
            if ts.n_epochs >= self.config.selection.min_epochs:
                if ts.delta_rv >= self.config.selection.min_delta_rv:
                    filtered.append(ts)

        logger.info(f"After basic filters: {len(filtered)} targets")

        # Compute RV metrics for filtered targets
        hardener = RVHardener(
            min_s_robust=self.config.selection.min_s_robust,
            min_chi2_pvalue=self.config.selection.min_chi2_pvalue
        )

        metrics_data = []
        for ts in filtered:
            metrics = hardener.compute_metrics(ts)
            metrics_data.append({
                'targetid': ts.targetid,
                'source_id': ts.source_id,
                'ra': ts.ra,
                'dec': ts.dec,
                'n_epochs': ts.n_epochs,
                'delta_rv': ts.delta_rv,
                'S': metrics.S,
                'S_robust': metrics.S_robust,
                'S_min_loo': metrics.S_min_loo,
                'chi2_constant': metrics.chi2_constant,
                'chi2_pvalue': metrics.chi2_pvalue,
                'd_max': metrics.d_max,
                'passed_hardening': hardener.passes_hardening(metrics)
            })

        # Save metrics table
        import csv
        metrics_path = os.path.join(self.run_dir, 'stage1_rv_metrics.csv')
        with open(metrics_path, 'w', newline='') as f:
            if metrics_data:
                writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
                writer.writeheader()
                writer.writerows(metrics_data)

        logger.info(f"Saved RV metrics to {metrics_path}")

        self.checkpoint.mark_complete('stage1', {
            'n_targets': len(timeseries_list),
            'n_filtered': len(filtered),
            'metrics_path': metrics_path
        })

        return filtered

    def stage2_fast_screen(self, timeseries_list: List) -> Dict:
        """Stage 2: Fast screening for all experiments."""
        logger.info("=" * 60)
        logger.info("STAGE 2: Fast Screening (All Experiments)")
        logger.info("=" * 60)

        from .experiments.e1_mass_gap import MassGapExperiment
        from .experiments.e2_dark_companions import DarkCompanionExperiment
        from .experiments.e3_dwd_lisa import DWDLISAExperiment
        from .experiments.e4_brown_dwarf import BrownDwarfExperiment
        from .experiments.e5_hierarchical import HierarchicalExperiment
        from .experiments.e6_accretion import AccretionExperiment
        from .experiments.e7_halo_cluster import HaloClusterExperiment
        from .experiments.e8_anomalies import AnomalyExperiment

        experiments = [
            ('E1_mass_gap', MassGapExperiment),
            ('E2_dark_companions', DarkCompanionExperiment),
            ('E3_dwd_lisa', DWDLISAExperiment),
            ('E4_brown_dwarf', BrownDwarfExperiment),
            ('E5_hierarchical', HierarchicalExperiment),
            ('E6_accretion', AccretionExperiment),
            ('E7_halo_cluster', HaloClusterExperiment),
            ('E8_anomalies', AnomalyExperiment),
        ]

        results = {}
        candidates_dir = os.path.join(self.run_dir, 'candidates')
        os.makedirs(candidates_dir, exist_ok=True)

        for exp_name, exp_class in experiments:
            if not self.config.experiments.get(exp_name, type('', (), {'enabled': True})()).enabled:
                logger.info(f"Skipping disabled experiment: {exp_name}")
                continue

            # Check if this experiment was already completed (resumable)
            result_file = os.path.join(candidates_dir, f"{exp_name}_results.json")
            if os.path.exists(result_file):
                logger.info(f"Experiment {exp_name} already completed, loading from checkpoint...")
                try:
                    with open(result_file, 'r') as f:
                        saved_result = json.load(f)
                    # Create a minimal ExperimentResult for compatibility
                    from .experiments.base import ExperimentResult, Candidate
                    result = ExperimentResult(
                        experiment_name=exp_name,
                        n_input=saved_result.get('n_input', 0),
                        n_after_stage1=saved_result.get('n_after_stage1', 0),
                        n_after_stage2=saved_result.get('n_after_stage2', 0),
                        n_final=saved_result.get('n_final', 0),
                        candidates=[],  # Would need to deserialize if needed
                        filter_stats=saved_result.get('filter_stats', {}),
                        runtime_seconds=saved_result.get('runtime_seconds', 0),
                        timestamp=saved_result.get('timestamp', '')
                    )
                    results[exp_name] = result
                    logger.info(f"Loaded {exp_name}: {result.n_final} candidates from checkpoint")
                    continue
                except Exception as e:
                    logger.warning(f"Could not load checkpoint for {exp_name}, re-running: {e}")

            logger.info(f"Running experiment: {exp_name}")
            try:
                exp = exp_class(self.config)
                result = exp.run(timeseries_list, output_dir=candidates_dir)
                results[exp_name] = result
                logger.info(f"{exp_name}: {result.n_final} final candidates")

                # Save per-experiment checkpoint
                self.checkpoint.mark_complete(f'stage2_{exp_name}', {
                    'n_final': result.n_final,
                    'result_file': result_file
                })
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {e}")
                import traceback
                traceback.print_exc()

        # Save stage2 summary
        summary = {exp: r.n_final for exp, r in results.items()}
        summary_path = os.path.join(self.run_dir, 'stage2_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.checkpoint.mark_complete('stage2', {
            'experiments_run': list(results.keys()),
            'summary': summary
        })

        return results

    def stage3_deep_inference(self, experiment_results: Dict):
        """Stage 3: Deep inference for top candidates."""
        logger.info("=" * 60)
        logger.info("STAGE 3: Deep Inference (Top Candidates)")
        logger.info("=" * 60)

        # For compute efficiency, we do simplified deep inference
        # Full MCMC would be run only on the very top candidates

        dossiers_dir = os.path.join(self.run_dir, 'dossiers')
        os.makedirs(dossiers_dir, exist_ok=True)

        top_candidates = []
        for exp_name, result in experiment_results.items():
            for candidate in result.candidates[:10]:  # Top 10 per experiment
                top_candidates.append((exp_name, candidate))

        logger.info(f"Deep inference for {len(top_candidates)} top candidates")

        # Generate basic dossiers (simplified)
        for exp_name, candidate in top_candidates:
            dossier = {
                'targetid': candidate.targetid,
                'experiment': exp_name,
                'ra': candidate.ra,
                'dec': candidate.dec,
                'n_epochs': candidate.n_epochs,
                'delta_rv': candidate.delta_rv,
                'best_period': candidate.best_period,
                'best_K': candidate.best_K,
                'm2_min': candidate.m2_min,
                'physics_score': candidate.physics_score,
                'total_score': candidate.total_score,
                'is_pathological': candidate.is_pathological,
                'pathology_reasons': candidate.pathology_reasons,
                'notes': 'Basic dossier - full MCMC not run'
            }

            dossier_path = os.path.join(
                dossiers_dir, f"{candidate.targetid}_{exp_name}.json"
            )
            with open(dossier_path, 'w') as f:
                json.dump(dossier, f, indent=2)

        self.checkpoint.mark_complete('stage3', {
            'n_dossiers': len(top_candidates)
        })

    def stage4_final_report(self, experiment_results: Dict):
        """Stage 4: Final scoring, leaderboards, and report."""
        logger.info("=" * 60)
        logger.info("STAGE 4: Final Scoring and Report")
        logger.info("=" * 60)

        from .scoring.unified import GlobalLeaderboard
        from .reports.generator import ReportGenerator

        # Build global leaderboard
        leaderboard = GlobalLeaderboard()
        for result in experiment_results.values():
            leaderboard.add_experiment_results(result)

        # Save leaderboard
        leaderboard.save(
            os.path.join(self.run_dir, 'global_leaderboard.csv'),
            os.path.join(self.run_dir, 'global_leaderboard.json')
        )

        # Generate report
        generator = ReportGenerator()
        report = generator.generate(experiment_results, leaderboard)
        report_path = os.path.join(self.run_dir, 'ANALYSIS_REPORT_RARE_REAL.md')
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {report_path}")

        # Print executive summary
        self._print_executive_summary(experiment_results, leaderboard)

        self.checkpoint.mark_complete('stage4', {
            'report_path': report_path
        })

    def _print_executive_summary(self, results: Dict, leaderboard):
        """Print executive summary to console."""
        print("\n" + "=" * 60)
        print("EXECUTIVE SUMMARY")
        print("=" * 60)

        total_input = sum(r.n_input for r in results.values()) // len(results)
        total_final = sum(r.n_final for r in results.values())

        print(f"\nTotal targets scanned: {total_input}")
        print(f"Total candidates across all experiments: {total_final}")

        print("\nYields per experiment:")
        for exp_name, result in results.items():
            print(f"  {exp_name}: {result.n_final} candidates")

        # Top 20 global
        ranked = leaderboard.rank()[:20]
        print("\nTop 20 Global Candidates:")
        print("-" * 80)
        print(f"{'Rank':<5} {'TargetID':<15} {'Score':<8} {'M2_min':<10} {'Experiments'}")
        print("-" * 80)
        for i, entry in enumerate(ranked, 1):
            exps = ','.join(entry.experiments[:2])
            if len(entry.experiments) > 2:
                exps += f"+{len(entry.experiments)-2}"
            print(f"{i:<5} {entry.targetid:<15} {entry.unified_score:<8.3f} "
                  f"{entry.best_m2_min:<10.2f} {exps}")

        # Count pathological
        n_pathological = sum(
            1 for r in results.values()
            for c in r.candidates if c.is_pathological
        )
        print(f"\nCandidates flagged as pathological: {n_pathological}")
        print("=" * 60)

    def _generate_synthetic_data(self, n_targets: int = 500) -> List:
        """Generate synthetic RV timeseries for demonstration."""
        from .ingest.unified import RVTimeSeries, RVEpoch

        np.random.seed(self.config.random_seed)
        timeseries_list = []

        for i in range(n_targets):
            n_epochs = np.random.randint(3, 10)

            # Random orbital parameters
            P = np.random.uniform(5, 200)
            K = np.random.uniform(20, 150)
            gamma = np.random.uniform(-50, 50)
            phi = np.random.uniform(0, 2 * np.pi)

            # Generate observation times
            mjd = np.sort(np.random.uniform(59000, 60500, n_epochs))
            phase = 2 * np.pi * mjd / P + phi
            rv_true = gamma + K * np.sin(phase)

            # Add realistic noise
            rv_err = np.random.uniform(1, 8, n_epochs)
            rv = rv_true + np.random.normal(0, rv_err)

            epochs = [
                RVEpoch(
                    mjd=mjd[j],
                    rv=rv[j],
                    rv_err=rv_err[j],
                    instrument='DESI',
                    survey='MWS',
                    quality=1.0
                )
                for j in range(n_epochs)
            ]

            # Estimate M1 from random spectral type
            M1 = np.random.uniform(0.3, 1.2)

            ts = RVTimeSeries(
                targetid=39000000000 + i,
                source_id=3800000000000000000 + i,
                ra=np.random.uniform(0, 360),
                dec=np.random.uniform(-30, 80),
                epochs=epochs,
                metadata={'M1': M1, 'synthetic': True}
            )
            timeseries_list.append(ts)

        return timeseries_list

    def _load_stage1_results(self) -> List:
        """Load Stage 1 results from checkpoint."""
        # For now, return empty (would need to cache timeseries)
        logger.warning("Stage 1 result loading not fully implemented")
        return []

    def _load_stage2_results(self) -> Dict:
        """Load Stage 2 results from checkpoint."""
        logger.warning("Stage 2 result loading not fully implemented")
        return {}


def main():
    """Main entry point for full pipeline run."""
    from .config import Config

    # Create config
    config = Config()
    config.run_name = "real_run"
    config.budget.stage2_deep_fit_top_k = 200  # Reasonable budget
    config.budget.stage3_dossier_top_n = 50

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/real_run_{timestamp}"

    # Run pipeline
    runner = FullPipelineRunner(config, run_dir)
    runner.run_all()


if __name__ == "__main__":
    main()
