"""
Unified scoring and global leaderboard.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
import csv

from ..experiments.base import Candidate, ExperimentResult

logger = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    """Candidate with unified scoring across experiments."""
    candidate: Candidate
    unified_score: float
    rank: int
    experiments: List[str]  # Which experiments found this target
    best_experiment: str
    uncertainty_score: float  # How uncertain is the classification

    def to_dict(self) -> Dict:
        return {
            'targetid': self.candidate.targetid,
            'source_id': self.candidate.source_id,
            'ra': self.candidate.ra,
            'dec': self.candidate.dec,
            'unified_score': self.unified_score,
            'rank': self.rank,
            'experiments': self.experiments,
            'best_experiment': self.best_experiment,
            'uncertainty_score': self.uncertainty_score,
            'm2_min': self.candidate.m2_min,
            'prob_ns_or_heavier': self.candidate.prob_ns_or_heavier,
            'prob_bh': self.candidate.prob_bh,
            'physics_score': self.candidate.physics_score,
            'cleanliness_score': self.candidate.cleanliness_score,
            'period_reliability_score': self.candidate.period_reliability_score,
            'n_epochs': self.candidate.n_epochs,
            'delta_rv': self.candidate.delta_rv,
            'best_period': self.candidate.best_period,
            'best_K': self.candidate.best_K
        }


class GlobalLeaderboard:
    """
    Global leaderboard merging candidates across experiments.
    """

    def __init__(self):
        self.candidates: List[ScoredCandidate] = []
        self._by_targetid: Dict[int, ScoredCandidate] = {}

    def add_experiment_results(self, result: ExperimentResult):
        """Add candidates from an experiment result."""
        for candidate in result.candidates:
            if candidate.targetid in self._by_targetid:
                # Already seen - update if better
                existing = self._by_targetid[candidate.targetid]
                existing.experiments.append(result.experiment_name)

                if candidate.total_score > existing.candidate.total_score:
                    existing.candidate = candidate
                    existing.best_experiment = result.experiment_name
            else:
                # New candidate
                scored = ScoredCandidate(
                    candidate=candidate,
                    unified_score=0.0,
                    rank=0,
                    experiments=[result.experiment_name],
                    best_experiment=result.experiment_name,
                    uncertainty_score=0.0
                )
                self.candidates.append(scored)
                self._by_targetid[candidate.targetid] = scored

    def compute_unified_scores(self):
        """Compute unified scores for all candidates."""
        for scored in self.candidates:
            c = scored.candidate

            # Base score from experiment
            base_score = c.total_score

            # Bonus for appearing in multiple experiments
            multi_exp_bonus = 0.1 * (len(scored.experiments) - 1)

            # Uncertainty based on period reliability and epoch count
            period_unc = 1.0 - c.period_reliability_score
            epoch_unc = 1.0 - min(1.0, c.n_epochs / 10.0)
            scored.uncertainty_score = (period_unc + epoch_unc) / 2.0

            # Final score
            scored.unified_score = min(1.0, base_score + multi_exp_bonus)

    def rank(self) -> List[ScoredCandidate]:
        """Rank all candidates by unified score."""
        self.compute_unified_scores()

        # Sort by score descending
        self.candidates.sort(key=lambda x: x.unified_score, reverse=True)

        # Assign ranks
        for i, scored in enumerate(self.candidates):
            scored.rank = i + 1

        return self.candidates

    def get_top(self, n: int = 50) -> List[ScoredCandidate]:
        """Get top N candidates."""
        return self.rank()[:n]

    def to_csv(self, path: str):
        """Save leaderboard to CSV."""
        ranked = self.rank()

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'rank', 'targetid', 'source_id', 'ra', 'dec',
                'unified_score', 'uncertainty', 'experiments',
                'm2_min', 'prob_ns_or_heavier', 'prob_bh',
                'n_epochs', 'delta_rv', 'best_period', 'best_K'
            ])

            for scored in ranked:
                c = scored.candidate
                writer.writerow([
                    scored.rank,
                    c.targetid,
                    c.source_id or '',
                    f'{c.ra:.6f}',
                    f'{c.dec:.6f}',
                    f'{scored.unified_score:.4f}',
                    f'{scored.uncertainty_score:.4f}',
                    '|'.join(scored.experiments),
                    f'{c.m2_min:.3f}',
                    f'{c.prob_ns_or_heavier:.3f}',
                    f'{c.prob_bh:.3f}',
                    c.n_epochs,
                    f'{c.delta_rv:.2f}',
                    f'{c.best_period:.2f}',
                    f'{c.best_K:.2f}'
                ])

    def to_json(self, path: str):
        """Save leaderboard to JSON."""
        ranked = self.rank()
        data = {
            'n_candidates': len(ranked),
            'candidates': [s.to_dict() for s in ranked]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class UnifiedScorer:
    """
    Compute unified scores across experiments.
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Parameters
        ----------
        weights : dict
            Weights for score components
        """
        self.weights = weights or {
            'physics': 0.35,
            'cleanliness': 0.25,
            'period_reliability': 0.20,
            'followup_value': 0.20
        }

    def score(self, candidate: Candidate) -> float:
        """Compute unified score for a candidate."""
        return (
            self.weights['physics'] * candidate.physics_score +
            self.weights['cleanliness'] * candidate.cleanliness_score +
            self.weights['period_reliability'] * candidate.period_reliability_score +
            self.weights['followup_value'] * candidate.followup_score
        )

    def compare(self, candidates: List[Candidate]) -> List[Tuple[Candidate, float]]:
        """Score and rank a list of candidates."""
        scored = [(c, self.score(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
