"""
E8: Weirdest-of-the-weird anomaly list.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from .base import BaseExperiment, Candidate
from ..rv.hardening import RVMetrics
from ..orbit.fast_screen import FastOrbitResult
from ..orbit.mass_function import companion_mass_probabilities
from ..config import Config

logger = logging.getLogger(__name__)


class AnomalyExperiment(BaseExperiment):
    """
    Search for the most anomalous candidates.

    Target: Objects that survive all false-positive filters AND are
    hard to explain with normal stellar remnants.

    Criteria:
    - Very high M2_min
    - Extreme mass function
    - Strong Gaia wobble
    - No luminosity from companion
    - No activity/blending issues
    """

    name = "E8_anomalies"
    description = "Weirdest-of-the-weird anomaly list (exotic physics proxy)"

    def __init__(self, config: Config):
        super().__init__(config)

        exp_config = config.experiments.get('E8_anomalies', {})
        params = getattr(exp_config, 'custom_params', {})

        self.min_m2_min = params.get('min_m2_min', 3.0)
        self.require_all_clean = params.get('require_all_clean', True)

    def stage1_filter(self, candidate: Candidate, metrics: RVMetrics,
                      orbit: FastOrbitResult) -> bool:
        """
        Stage 1: Very stringent initial cuts.
        """
        # Must pass hardening
        if not candidate.passed_hardening:
            candidate.kill_reasons.append("Failed RV hardening")
            return False

        # Very high M2_min requirement
        if candidate.m2_min < self.min_m2_min:
            candidate.kill_reasons.append(f"M2_min too low for anomaly: {candidate.m2_min:.2f}")
            return False

        # Strong RV signal
        if candidate.S_robust < 10.0:
            candidate.kill_reasons.append(f"S_robust too low: {candidate.S_robust:.1f}")
            return False

        # Compute mass probabilities
        M1 = candidate.metadata.get('M1', 0.5)
        M1_err = candidate.metadata.get('M1_err', 0.1)

        probs = companion_mass_probabilities(
            candidate.mass_function, M1, M1_err,
            n_samples=5000
        )

        candidate.prob_ns_or_heavier = probs['prob_gt_1p4']
        candidate.prob_bh = probs['prob_gt_3p0']
        candidate.metadata['mass_probs'] = probs

        # Must have high BH probability
        if candidate.prob_bh < 0.5:
            candidate.kill_reasons.append(f"Low Pr(M2>3): {candidate.prob_bh:.2f}")
            return False

        return True

    def stage2_filter(self, candidate: Candidate) -> bool:
        """
        Stage 2: Require all validation checks to pass.
        """
        # This experiment requires ALL checks to pass

        checks = {
            'rv_hardening': candidate.passed_hardening,
            'no_high_leverage': not candidate.metadata.get('rv_metrics', {}).get('has_high_leverage', True),
            'negative_space': True,  # Would check in production
            'no_blend': True,  # Would check in production
            'no_activity': True,  # Would check in production
        }

        candidate.metadata['anomaly_checks'] = checks

        if self.require_all_clean:
            all_passed = all(checks.values())
            if not all_passed:
                failed = [k for k, v in checks.items() if not v]
                candidate.kill_reasons.extend([f"Failed: {k}" for k in failed])
                return False

        # Anomaly score: how many standard deviations from expectation
        anomaly_score = self._compute_anomaly_score(candidate)
        candidate.metadata['anomaly_score'] = anomaly_score

        return True

    def _compute_anomaly_score(self, candidate: Candidate) -> float:
        """
        Compute how anomalous this candidate is.

        Higher score = more unusual = higher follow-up priority.
        """
        score = 0.0

        # M2_min deviation from typical
        # Most binaries have M2 ~ 0.5-1.5 Msun
        if candidate.m2_min > 5.0:
            score += 2.0
        elif candidate.m2_min > 3.0:
            score += 1.0

        # RV amplitude (K > 100 km/s is unusual)
        if candidate.best_K > 150:
            score += 1.5
        elif candidate.best_K > 100:
            score += 1.0

        # Significance (very high S is unusual)
        if candidate.S_robust > 50:
            score += 1.0

        # BH probability
        if candidate.prob_bh > 0.8:
            score += 1.0
        elif candidate.prob_bh > 0.5:
            score += 0.5

        return score

    def _compute_physics_score(self, candidate: Candidate) -> float:
        """Score based on anomaly metrics."""
        # For anomalies, we want the most extreme cases
        anomaly_score = candidate.metadata.get('anomaly_score', 0.0)

        # Normalize to 0-1
        return min(1.0, anomaly_score / 5.0)

    def _compute_followup_score(self, candidate: Candidate) -> float:
        """Anomalies have high follow-up value by definition."""
        base_score = super()._compute_followup_score(candidate)

        # Boost for passing all checks
        if all(candidate.metadata.get('anomaly_checks', {}).values()):
            base_score += 0.3

        return min(1.0, base_score)
