"""
Configuration management for rare_companions pipeline.
"""

import yaml
import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import subprocess


@dataclass
class DataPaths:
    """Paths to input data files."""
    desi_bright: str = "data/raw/rvpix_exp-main-bright.fits"
    desi_dark: str = "data/raw/rvpix_exp-main-dark.fits"
    desi_special: str = "data/raw/rvpix_exp-special-bright.fits"
    lamost_catalog: str = "data/lamost_dr7_stellar.fits"
    gaia_cache: str = "cache/gaia_cache.pkl"
    output_dir: str = "runs"


@dataclass
class SelectionThresholds:
    """Thresholds for candidate selection."""
    min_epochs: int = 3
    min_delta_rv: float = 20.0  # km/s
    min_s_robust: float = 5.0
    max_rv_err: float = 50.0  # km/s
    min_chi2_pvalue: float = 1e-6  # reject constant RV
    max_ruwe: float = 5.0  # upper limit for astrometric quality


@dataclass
class OrbitConfig:
    """Configuration for orbital fitting."""
    period_min: float = 0.5  # days
    period_max: float = 1000.0  # days
    period_grid_points: int = 500
    mcmc_walkers: int = 32
    mcmc_steps: int = 3000
    mcmc_burnin: int = 500
    eccentricity_max: float = 0.95


@dataclass
class PathologyGuardrails:
    """Guardrails to prevent obviously pathological fits from polluting results."""
    # Maximum allowed K amplitude (km/s) - anything above is flagged pathological
    max_K_amplitude: float = 500.0
    # Maximum allowed M2_min (Msun) before flagging as pathological
    max_m2_min: float = 100.0
    # Minimum epochs required for MCMC (below this, only fast screen)
    min_epochs_for_mcmc: int = 4
    # Minimum epochs for any period reliability claim
    min_epochs_for_period_reliability: int = 5
    # Maximum allowed period relative to baseline
    max_period_to_baseline_ratio: float = 2.0
    # Minimum delta_chi2 improvement over constant model
    min_delta_chi2: float = 10.0
    # Exclude pathological fits from these experiments (send to E8 instead)
    exclude_pathological_from: List[str] = field(default_factory=lambda: [
        'E1_mass_gap', 'E2_dark_companions', 'E4_brown_dwarf'
    ])


@dataclass
class ComputeBudget:
    """Compute budget limits for staged processing."""
    stage1_all_targets: bool = True
    stage2_deep_fit_top_k: int = 100  # per experiment
    stage3_dossier_top_n: int = 20  # per experiment
    max_parallel_workers: int = 8
    injection_recovery_realizations: int = 200


@dataclass
class ExperimentConfig:
    """Configuration for individual experiments."""
    enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration class."""
    run_name: str = "rare_companions_run"
    data_paths: DataPaths = field(default_factory=DataPaths)
    selection: SelectionThresholds = field(default_factory=SelectionThresholds)
    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    budget: ComputeBudget = field(default_factory=ComputeBudget)
    guardrails: PathologyGuardrails = field(default_factory=PathologyGuardrails)
    experiments: Dict[str, ExperimentConfig] = field(default_factory=dict)
    random_seed: int = 42

    def __post_init__(self):
        # Initialize default experiments if not provided
        default_experiments = [
            'E1_mass_gap', 'E2_dark_companions', 'E3_dwd_lisa',
            'E4_brown_dwarf', 'E5_hierarchical', 'E6_accretion',
            'E7_halo_cluster', 'E8_anomalies'
        ]
        for exp in default_experiments:
            if exp not in self.experiments:
                self.experiments[exp] = ExperimentConfig()

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'run_name': self.run_name,
            'data_paths': self.data_paths.__dict__,
            'selection': self.selection.__dict__,
            'orbit': self.orbit.__dict__,
            'budget': self.budget.__dict__,
            'guardrails': {k: v for k, v in self.guardrails.__dict__.items()},
            'experiments': {k: v.__dict__ for k, v in self.experiments.items()},
            'random_seed': self.random_seed
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        if 'run_name' in d:
            config.run_name = d['run_name']
        if 'data_paths' in d:
            config.data_paths = DataPaths(**d['data_paths'])
        if 'selection' in d:
            config.selection = SelectionThresholds(**d['selection'])
        if 'orbit' in d:
            config.orbit = OrbitConfig(**d['orbit'])
        if 'budget' in d:
            config.budget = ComputeBudget(**d['budget'])
        if 'guardrails' in d:
            config.guardrails = PathologyGuardrails(**d['guardrails'])
        if 'experiments' in d:
            config.experiments = {
                k: ExperimentConfig(**v) for k, v in d['experiments'].items()
            }
        if 'random_seed' in d:
            config.random_seed = d['random_seed']
        return config


def load_config(path: str) -> Config:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    return Config.from_dict(d)


def save_config(config: Config, path: str):
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()[:8]
    except:
        return "unknown"


def create_run_directory(config: Config, base_dir: str = "runs") -> str:
    """Create a run directory with config snapshot and metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{config.run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config snapshot
    config_path = os.path.join(run_dir, "config_snapshot.yaml")
    save_config(config, config_path)

    # Save manifest
    manifest = {
        'run_name': config.run_name,
        'timestamp': timestamp,
        'git_hash': get_git_hash(),
        'random_seed': config.random_seed,
        'config_checksum': hashlib.md5(
            json.dumps(config.to_dict(), sort_keys=True).encode()
        ).hexdigest()
    }

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Create subdirectories
    for subdir in ['candidates', 'dossiers', 'plots', 'logs']:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

    return run_dir
