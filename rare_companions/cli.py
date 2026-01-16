"""
Command-line interface for rare_companions pipeline.
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from typing import List, Optional
import json

from .config import Config, load_config, create_run_directory
from .ingest.desi import DESIRVLoader
from .ingest.lamost import LAMOSTLoader
from .ingest.unified import UnifiedRVLoader
from .experiments import EXPERIMENTS
from .experiments.base import ExperimentResult
from .scoring.unified import GlobalLeaderboard
from .reports.generator import ReportGenerator
from .reports.dossier import DossierGenerator

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def run_all(config: Config, output_dir: str = None) -> dict:
    """
    Run all enabled experiments.

    Parameters
    ----------
    config : Config
        Configuration object
    output_dir : str, optional
        Output directory (created if not specified)

    Returns
    -------
    dict
        Results summary
    """
    # Create run directory
    if output_dir is None:
        output_dir = create_run_directory(config)

    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info("Loading DESI data...")
    desi_loader = DESIRVLoader(
        config.data_paths.desi_bright,
        config.data_paths.desi_dark
    )

    try:
        n_loaded = desi_loader.load(max_rv_err=config.selection.max_rv_err)
        logger.info(f"Loaded {n_loaded} DESI targets")
    except Exception as e:
        logger.warning(f"Could not load DESI data: {e}")
        logger.info("Running in smoke test mode with synthetic data")
        return run_smoke_test(config, output_dir)

    # Load LAMOST (optional)
    lamost_loader = None
    if os.path.exists(config.data_paths.lamost_catalog):
        logger.info("Loading LAMOST data...")
        lamost_loader = LAMOSTLoader(config.data_paths.lamost_catalog)
        lamost_loader.load()

    # Create unified loader
    unified = UnifiedRVLoader(desi_loader, lamost_loader)

    # Get all time series
    logger.info("Building time series...")
    timeseries_list = unified.get_all_timeseries(
        min_epochs=config.selection.min_epochs
    )
    logger.info(f"Built {len(timeseries_list)} time series")

    # Run experiments
    experiment_results = {}
    leaderboard = GlobalLeaderboard()

    for exp_name, exp_class in EXPERIMENTS.items():
        exp_config = config.experiments.get(exp_name, {})
        if not getattr(exp_config, 'enabled', True):
            logger.info(f"Skipping disabled experiment: {exp_name}")
            continue

        logger.info(f"Running experiment: {exp_name}")
        experiment = exp_class(config)

        try:
            result = experiment.run(
                timeseries_list,
                output_dir=os.path.join(output_dir, 'candidates')
            )
            experiment_results[exp_name] = result
            leaderboard.add_experiment_results(result)
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            continue

    # Generate global leaderboard
    logger.info("Computing global rankings...")
    ranked = leaderboard.rank()

    # Save leaderboard
    leaderboard.to_csv(os.path.join(output_dir, 'global_leaderboard.csv'))
    leaderboard.to_json(os.path.join(output_dir, 'global_leaderboard.json'))

    # Generate dossiers for top candidates
    logger.info("Generating dossiers...")
    dossier_gen = DossierGenerator(os.path.join(output_dir, 'dossiers'))

    for scored in ranked[:config.budget.stage3_dossier_top_n]:
        dossier_gen.generate(scored.candidate)

    # Generate report
    logger.info("Generating report...")
    report_gen = ReportGenerator(config)
    report_gen.generate(
        experiment_results,
        leaderboard,
        os.path.join(output_dir, 'ANALYSIS_REPORT_RARE.md')
    )

    # Summary
    summary = {
        'run_dir': output_dir,
        'n_input': len(timeseries_list),
        'experiments_run': list(experiment_results.keys()),
        'n_candidates': len(ranked),
        'top_candidates': [s.candidate.targetid for s in ranked[:10]]
    }

    with open(os.path.join(output_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Run complete. Results in {output_dir}")

    return summary


def run_smoke_test(config: Config, output_dir: str = None) -> dict:
    """
    Run a quick smoke test with synthetic data.

    Parameters
    ----------
    config : Config
        Configuration object
    output_dir : str, optional
        Output directory

    Returns
    -------
    dict
        Test results
    """
    import numpy as np
    from .ingest.unified import RVTimeSeries, RVEpoch

    logger.info("Running smoke test with synthetic data...")

    if output_dir is None:
        output_dir = create_run_directory(config)

    # Create synthetic time series
    np.random.seed(config.random_seed)

    timeseries_list = []

    for i in range(100):
        n_epochs = np.random.randint(3, 8)

        # Random parameters
        P = np.random.uniform(5, 100)  # days
        K = np.random.uniform(10, 150)  # km/s
        gamma = np.random.uniform(-50, 50)  # km/s

        # Generate epochs
        mjd = np.sort(np.random.uniform(57000, 60000, n_epochs))

        # Generate RVs
        phase = 2 * np.pi * mjd / P
        rv_true = gamma + K * np.sin(phase)
        rv_err = np.random.uniform(1, 5, n_epochs)
        rv = rv_true + np.random.normal(0, rv_err)

        epochs = [
            RVEpoch(
                mjd=mjd[j],
                rv=rv[j],
                rv_err=rv_err[j],
                instrument='SYNTHETIC',
                survey='smoke_test',
                quality=1.0
            )
            for j in range(n_epochs)
        ]

        ts = RVTimeSeries(
            targetid=1000000 + i,
            source_id=3800000000000000000 + i,
            ra=np.random.uniform(0, 360),
            dec=np.random.uniform(-90, 90),
            epochs=epochs,
            metadata={'M1': np.random.uniform(0.3, 1.5)}
        )

        timeseries_list.append(ts)

    logger.info(f"Created {len(timeseries_list)} synthetic time series")

    # Run experiments
    experiment_results = {}
    leaderboard = GlobalLeaderboard()

    # Run subset of experiments for speed
    test_experiments = ['E1_mass_gap', 'E2_dark_companions', 'E8_anomalies']

    for exp_name in test_experiments:
        if exp_name not in EXPERIMENTS:
            continue

        exp_class = EXPERIMENTS[exp_name]
        logger.info(f"Running experiment: {exp_name}")
        experiment = exp_class(config)

        try:
            result = experiment.run(
                timeseries_list,
                output_dir=os.path.join(output_dir, 'candidates')
            )
            experiment_results[exp_name] = result
            leaderboard.add_experiment_results(result)
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            continue

    # Generate outputs
    ranked = leaderboard.rank()
    leaderboard.to_csv(os.path.join(output_dir, 'global_leaderboard.csv'))
    leaderboard.to_json(os.path.join(output_dir, 'global_leaderboard.json'))

    # Generate report
    report_gen = ReportGenerator(config)
    report_gen.generate(
        experiment_results,
        leaderboard,
        os.path.join(output_dir, 'ANALYSIS_REPORT_RARE.md')
    )

    summary = {
        'run_dir': output_dir,
        'mode': 'smoke_test',
        'n_input': len(timeseries_list),
        'experiments_run': list(experiment_results.keys()),
        'n_candidates': len(ranked),
        'status': 'SUCCESS'
    }

    with open(os.path.join(output_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Smoke test complete. Results in {output_dir}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Rare Companions Multi-Experiment Search Pipeline',
        prog='python -m rare_companions.cli'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # run-all command
    run_parser = subparsers.add_parser('run-all', help='Run all experiments')
    run_parser.add_argument('--config', '-c', required=True,
                           help='Path to config YAML file')
    run_parser.add_argument('--output', '-o', default=None,
                           help='Output directory (auto-generated if not specified)')
    run_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output')

    # smoke-test command
    smoke_parser = subparsers.add_parser('smoke-test', help='Run smoke test')
    smoke_parser.add_argument('--config', '-c', default=None,
                             help='Path to config YAML file')
    smoke_parser.add_argument('--output', '-o', default=None,
                             help='Output directory')
    smoke_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    setup_logging(verbose=getattr(args, 'verbose', False))

    if args.command == 'run-all':
        config = load_config(args.config)
        run_all(config, args.output)

    elif args.command == 'smoke-test':
        if args.config:
            config = load_config(args.config)
        else:
            config = Config()
        run_smoke_test(config, args.output)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
