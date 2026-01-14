"""
Run manifest and provenance tracking for reproducibility.

Captures configuration, git state, file checksums, and timing
information for each pipeline run.
"""

import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_config_hash(config: Dict[str, Any], length: int = 16) -> str:
    """
    Compute deterministic hash of configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    length : int
        Length of returned hash (default 16 chars)

    Returns
    -------
    str
        Hex hash string
    """
    # Sort keys for deterministic hashing
    config_str = json.dumps(config, sort_keys=True, default=str)
    full_hash = hashlib.sha256(config_str.encode()).hexdigest()
    return full_hash[:length]


def sha256_file(filepath: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 checksum of a file.

    Parameters
    ----------
    filepath : str
        Path to file
    chunk_size : int
        Read chunk size in bytes

    Returns
    -------
    str
        64-character hex hash
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_info(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get git repository information.

    Parameters
    ----------
    repo_path : str, optional
        Path to git repository (default: current directory)

    Returns
    -------
    dict
        Git info including commit, branch, dirty status
    """
    cwd = repo_path or os.getcwd()

    try:
        # Get current commit
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()

        # Get current branch
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()

        # Check if dirty
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = len(status) > 0

        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty,
            'available': True,
        }

    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            'commit': None,
            'branch': None,
            'dirty': None,
            'available': False,
        }


@dataclass
class RunManifest:
    """
    Manifest tracking all details of a pipeline run.

    Captures configuration, git state, input/output files,
    timing information, and status for reproducibility.
    """

    run_id: str
    timestamp_utc: str
    config_hash: str
    config: Dict[str, Any]
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    command: str = ""
    status: str = "running"
    input_files: Dict[str, str] = field(default_factory=dict)  # path -> checksum
    output_files: Dict[str, str] = field(default_factory=dict)  # path -> checksum
    stage_timings: Dict[str, float] = field(default_factory=dict)  # stage -> seconds
    total_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        command: str = "",
        input_files: Optional[List[str]] = None,
    ) -> "RunManifest":
        """
        Create a new run manifest.

        Parameters
        ----------
        config : dict
            Pipeline configuration
        command : str
            CLI command that was run
        input_files : list of str, optional
            Paths to input files to checksum

        Returns
        -------
        RunManifest
            New manifest instance
        """
        # Generate run ID
        timestamp = datetime.now(timezone.utc)
        run_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{compute_config_hash(config, 8)}"

        # Get git info
        git_info = get_git_info()

        # Checksum input files
        input_checksums = {}
        if input_files:
            for path in input_files:
                if os.path.exists(path):
                    input_checksums[path] = sha256_file(path)

        return cls(
            run_id=run_id,
            timestamp_utc=timestamp.isoformat(),
            config_hash=compute_config_hash(config),
            config=config,
            git_commit=git_info.get('commit'),
            git_branch=git_info.get('branch'),
            git_dirty=git_info.get('dirty'),
            command=command,
            input_files=input_checksums,
        )

    def add_input_file(self, path: str) -> None:
        """Add an input file with its checksum."""
        if os.path.exists(path):
            self.input_files[path] = sha256_file(path)

    def add_output_file(self, path: str) -> None:
        """Add an output file with its checksum."""
        if os.path.exists(path):
            self.output_files[path] = sha256_file(path)

    def mark_completed(self, total_time: Optional[float] = None) -> None:
        """Mark run as completed successfully."""
        self.status = "completed"
        self.total_time_seconds = total_time

    def mark_failed(self, error_message: str) -> None:
        """Mark run as failed with error message."""
        self.status = "failed"
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_id': self.run_id,
            'timestamp_utc': self.timestamp_utc,
            'config_hash': self.config_hash,
            'config': self.config,
            'git_commit': self.git_commit,
            'git_branch': self.git_branch,
            'git_dirty': self.git_dirty,
            'command': self.command,
            'status': self.status,
            'input_files': self.input_files,
            'output_files': self.output_files,
            'stage_timings': self.stage_timings,
            'total_time_seconds': self.total_time_seconds,
            'error_message': self.error_message,
            'metadata': self.metadata,
        }

    def to_json(self, path: str) -> None:
        """Save manifest to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> "RunManifest":
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def __str__(self) -> str:
        status_str = f"[{self.status.upper()}]"
        return (
            f"RunManifest {self.run_id} {status_str}\n"
            f"  Config hash: {self.config_hash}\n"
            f"  Git: {self.git_branch}@{self.git_commit[:8] if self.git_commit else 'N/A'}"
            f"{' (dirty)' if self.git_dirty else ''}\n"
            f"  Stages: {len(self.stage_timings)}\n"
            f"  Total time: {self.total_time_seconds:.1f}s" if self.total_time_seconds else ""
        )


@contextmanager
def stage_timer(manifest: RunManifest, stage_name: str):
    """
    Context manager to time a pipeline stage.

    Parameters
    ----------
    manifest : RunManifest
        Manifest to record timing in
    stage_name : str
        Name of the stage being timed

    Example
    -------
    >>> with stage_timer(manifest, 'compute_pairwise'):
    ...     result = estimator.compute(...)
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        manifest.stage_timings[stage_name] = elapsed


def create_run_manifest(
    config: Dict[str, Any],
    command: str = "",
    input_files: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> RunManifest:
    """
    Convenience function to create and optionally save a run manifest.

    Parameters
    ----------
    config : dict
        Pipeline configuration
    command : str
        CLI command
    input_files : list of str, optional
        Input file paths
    output_dir : str, optional
        Directory to save manifest JSON

    Returns
    -------
    RunManifest
        New manifest instance
    """
    manifest = RunManifest.create(config, command, input_files)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest.to_json(str(output_path / f'manifest_{manifest.run_id}.json'))

    return manifest
