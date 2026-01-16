"""
Base class for experiments.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import json
import os
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

from ..ingest.unified import RVTimeSeries
from ..rv.hardening import RVHardener, RVMetrics
from ..orbit.fast_screen import FastOrbitScreen, FastOrbitResult
from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A candidate from an experiment."""
    targetid: int
    source_id: Optional[int]
    ra: float
    dec: float
    experiment: str

    # RV metrics
    n_epochs: int
    delta_rv: float
    S_robust: float

    # Orbital parameters (fast screen)
    best_period: float
    best_K: float
    mass_function: float
    m2_min: float

    # Probabilities
    prob_ns_or_heavier: float  # Pr(M2 > 1.4)
    prob_bh: float  # Pr(M2 > 3.0)

    # Scores
    physics_score: float
    cleanliness_score: float
    period_reliability_score: float
    followup_score: float
    total_score: float

    # Flags
    passed_hardening: bool
    passed_negative_space: bool
    kill_reasons: List[str] = field(default_factory=list)

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'targetid': self.targetid,
            'source_id': self.source_id,
            'ra': self.ra,
            'dec': self.dec,
            'experiment': self.experiment,
            'n_epochs': self.n_epochs,
            'delta_rv': self.delta_rv,
            'S_robust': self.S_robust,
            'best_period': self.best_period,
            'best_K': self.best_K,
            'mass_function': self.mass_function,
            'm2_min': self.m2_min,
            'prob_ns_or_heavier': self.prob_ns_or_heavier,
            'prob_bh': self.prob_bh,
            'physics_score': self.physics_score,
            'cleanliness_score': self.cleanliness_score,
            'period_reliability_score': self.period_reliability_score,
            'followup_score': self.followup_score,
            'total_score': self.total_score,
            'passed_hardening': self.passed_hardening,
            'passed_negative_space': self.passed_negative_space,
            'kill_reasons': self.kill_reasons,
            'metadata': self.metadata
        }


@dataclass
class ExperimentResult:
    """Result from running an experiment."""
    experiment_name: str
    n_input: int
    n_after_stage1: int
    n_after_stage2: int
    n_final: int

    candidates: List[Candidate]
    filter_stats: Dict[str, int]

    runtime_seconds: float
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'experiment_name': self.experiment_name,
            'n_input': self.n_input,
            'n_after_stage1': self.n_after_stage1,
            'n_after_stage2': self.n_after_stage2,
            'n_final': self.n_final,
            'candidates': [c.to_dict() for c in self.candidates],
            'filter_stats': self.filter_stats,
            'runtime_seconds': self.runtime_seconds,
            'timestamp': self.timestamp
        }

    def save(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)


class BaseExperiment(ABC):
    """
    Base class for all experiments.

    Subclasses must implement:
    - name: experiment identifier
    - description: what the experiment searches for
    - stage1_filter: initial filtering criteria
    - stage2_filter: secondary filtering
    - compute_scores: scoring function
    """

    name: str = "base"
    description: str = "Base experiment class"

    def __init__(self, config: Config):
        self.config = config
        self.hardener = RVHardener(
            min_s_robust=config.selection.min_s_robust,
            min_chi2_pvalue=config.selection.min_chi2_pvalue
        )
        self.orbit_screener = FastOrbitScreen(
            period_min=config.orbit.period_min,
            period_max=config.orbit.period_max,
            n_periods=config.orbit.period_grid_points,
            n_workers=config.budget.max_parallel_workers
        )

    def run(self, timeseries_list: List[RVTimeSeries],
            output_dir: str = None) -> ExperimentResult:
        """
        Run the full experiment pipeline.

        Parameters
        ----------
        timeseries_list : list
            List of RVTimeSeries objects
        output_dir : str, optional
            Directory to save outputs

        Returns
        -------
        ExperimentResult
            Experiment results
        """
        start_time = datetime.now()
        logger.info(f"Running experiment: {self.name}")
        logger.info(f"Input targets: {len(timeseries_list)}")

        filter_stats = {'input': len(timeseries_list)}

        # Stage 1: Basic filtering and fast screen
        stage1_candidates = []
        for ts in timeseries_list:
            candidate = self._process_stage1(ts)
            if candidate is not None:
                stage1_candidates.append(candidate)

        filter_stats['after_stage1'] = len(stage1_candidates)
        logger.info(f"After Stage 1: {len(stage1_candidates)} candidates")

        # Sort by physics score and take top K
        stage1_candidates.sort(key=lambda c: c.physics_score, reverse=True)
        top_k = self.config.budget.stage2_deep_fit_top_k
        stage2_input = stage1_candidates[:top_k]

        # Stage 2: Deep filtering and analysis
        stage2_candidates = []
        for candidate in stage2_input:
            updated = self._process_stage2(candidate)
            if updated is not None:
                stage2_candidates.append(updated)

        filter_stats['after_stage2'] = len(stage2_candidates)
        logger.info(f"After Stage 2: {len(stage2_candidates)} candidates")

        # Final scoring and ranking
        for candidate in stage2_candidates:
            self._compute_final_scores(candidate)

        # Sort by total score
        stage2_candidates.sort(key=lambda c: c.total_score, reverse=True)

        # Take top N
        top_n = self.config.budget.stage3_dossier_top_n
        final_candidates = stage2_candidates[:top_n]

        filter_stats['final'] = len(final_candidates)

        runtime = (datetime.now() - start_time).total_seconds()

        result = ExperimentResult(
            experiment_name=self.name,
            n_input=len(timeseries_list),
            n_after_stage1=len(stage1_candidates),
            n_after_stage2=len(stage2_candidates),
            n_final=len(final_candidates),
            candidates=final_candidates,
            filter_stats=filter_stats,
            runtime_seconds=runtime,
            timestamp=datetime.now().isoformat()
        )

        # Save if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            result.save(os.path.join(output_dir, f"{self.name}_results.json"))

        logger.info(f"Experiment {self.name} complete: {len(final_candidates)} final candidates")

        return result

    def _process_stage1(self, ts: RVTimeSeries) -> Optional[Candidate]:
        """
        Stage 1 processing: basic metrics and fast orbit screen.

        Override stage1_filter() to customize filtering.
        """
        # Basic checks
        if ts.n_epochs < self.config.selection.min_epochs:
            return None

        if ts.delta_rv < self.config.selection.min_delta_rv:
            return None

        # Compute RV metrics
        metrics = self.hardener.compute_metrics(ts)

        # Fast orbit screen
        orbit = self.orbit_screener.search(ts.mjd, ts.rv, ts.rv_err, parallel=False)

        # Compute mass function and M2_min
        P = orbit.best_period
        K = orbit.best_K
        e = 0.0  # Assume circular for fast screen

        f_M = 1.0361e-7 * P * K**3 * (1 - e**2)**1.5

        # Estimate M1 from metadata or assume 0.5 Msun
        M1 = ts.metadata.get('M1', 0.5)

        # Solve for M2_min
        from ..orbit.mass_function import compute_m2_min
        m2_min = compute_m2_min(f_M, M1)

        # Create candidate
        candidate = Candidate(
            targetid=ts.targetid,
            source_id=ts.source_id,
            ra=ts.ra,
            dec=ts.dec,
            experiment=self.name,
            n_epochs=ts.n_epochs,
            delta_rv=ts.delta_rv,
            S_robust=metrics.S_robust,
            best_period=P,
            best_K=K,
            mass_function=f_M,
            m2_min=m2_min,
            prob_ns_or_heavier=0.0,  # Computed later
            prob_bh=0.0,
            physics_score=0.0,
            cleanliness_score=1.0,
            period_reliability_score=0.5,
            followup_score=0.0,
            total_score=0.0,
            passed_hardening=self.hardener.passes_hardening(metrics),
            passed_negative_space=True,  # Check in stage 2
            metadata={
                'rv_metrics': metrics.to_dict(),
                'orbit_screen': {
                    'period': orbit.best_period,
                    'K': orbit.best_K,
                    'chi2': orbit.best_chi2,
                    'delta_chi2': orbit.delta_chi2,
                    'is_multimodal': orbit.is_multimodal
                },
                'M1': M1
            }
        )

        # Apply experiment-specific stage 1 filter
        if not self.stage1_filter(candidate, metrics, orbit):
            return None

        # Compute initial physics score
        candidate.physics_score = self._compute_physics_score(candidate)

        return candidate

    def _process_stage2(self, candidate: Candidate) -> Optional[Candidate]:
        """
        Stage 2 processing: deeper analysis.

        Override stage2_filter() to customize.
        """
        # Apply experiment-specific stage 2 filter
        if not self.stage2_filter(candidate):
            return None

        return candidate

    def _compute_final_scores(self, candidate: Candidate):
        """Compute final scores for ranking."""
        candidate.physics_score = self._compute_physics_score(candidate)
        candidate.followup_score = self._compute_followup_score(candidate)
        candidate.total_score = (
            0.4 * candidate.physics_score +
            0.3 * candidate.cleanliness_score +
            0.2 * candidate.period_reliability_score +
            0.1 * candidate.followup_score
        )

    def _compute_physics_score(self, candidate: Candidate) -> float:
        """Compute physics score based on mass function and probabilities."""
        # Higher score for higher M2_min
        m2_score = min(1.0, candidate.m2_min / 5.0)

        # Higher score for larger RV amplitude
        rv_score = min(1.0, candidate.delta_rv / 200.0)

        # Higher score for higher significance
        sig_score = min(1.0, candidate.S_robust / 50.0)

        return (m2_score + rv_score + sig_score) / 3.0

    def _compute_followup_score(self, candidate: Candidate) -> float:
        """Compute follow-up value score."""
        # Higher score for:
        # - More epochs (better constrained)
        # - Higher M2_min
        # - Passed all filters

        epoch_score = min(1.0, candidate.n_epochs / 10.0)
        m2_score = min(1.0, candidate.m2_min / 3.0)
        filter_score = 1.0 if candidate.passed_hardening else 0.5

        return (epoch_score + m2_score + filter_score) / 3.0

    @abstractmethod
    def stage1_filter(self, candidate: Candidate, metrics: RVMetrics,
                      orbit: FastOrbitResult) -> bool:
        """
        Experiment-specific Stage 1 filter.

        Returns True if candidate passes.
        """
        pass

    @abstractmethod
    def stage2_filter(self, candidate: Candidate) -> bool:
        """
        Experiment-specific Stage 2 filter.

        Returns True if candidate passes.
        """
        pass

    def get_kill_sheet(self, candidate: Candidate) -> Dict:
        """
        Generate "kill sheet" explaining why candidate survived filters.

        Returns dict with filter names and pass/fail status.
        """
        return {
            'experiment': self.name,
            'targetid': candidate.targetid,
            'filters_passed': {
                'min_epochs': candidate.n_epochs >= self.config.selection.min_epochs,
                'min_delta_rv': candidate.delta_rv >= self.config.selection.min_delta_rv,
                'rv_hardening': candidate.passed_hardening,
                'negative_space': candidate.passed_negative_space,
            },
            'kill_reasons': candidate.kill_reasons,
            'scores': {
                'physics': candidate.physics_score,
                'cleanliness': candidate.cleanliness_score,
                'period_reliability': candidate.period_reliability_score,
                'followup': candidate.followup_score,
                'total': candidate.total_score
            }
        }
