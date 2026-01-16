"""
E2: NS/BH dark companion candidates around M/K dwarfs.
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


class DarkCompanionExperiment(BaseExperiment):
    """
    Search for quiescent NS/BH companions around M/K dwarfs.

    Key: Requires "negative space" validation - no IR/UV excess,
    no strong photometric variability.
    """

    name = "E2_dark_companions"
    description = "NS/BH dark companions around M/K dwarfs"

    def __init__(self, config: Config):
        super().__init__(config)

        exp_config = config.experiments.get('E2_dark_companions', {})
        params = getattr(exp_config, 'custom_params', {})

        self.primary_types = params.get('primary_types', ['K', 'M'])
        self.max_ir_excess = params.get('max_ir_excess', 0.15)
        self.max_photometric_var = params.get('max_photometric_var', 0.05)

    def stage1_filter(self, candidate: Candidate, metrics: RVMetrics,
                      orbit: FastOrbitResult) -> bool:
        """
        Stage 1: Basic dark companion criteria.
        """
        # Must pass hardening
        if not candidate.passed_hardening:
            candidate.kill_reasons.append("Failed RV hardening")
            return False

        # Need significant M2_min for NS/BH
        if candidate.m2_min < 1.0:
            candidate.kill_reasons.append(f"M2_min too low for NS/BH: {candidate.m2_min:.2f}")
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

        # Want reasonable probability of compact companion
        if candidate.prob_ns_or_heavier < 0.3:
            candidate.kill_reasons.append(f"Low Pr(M2>1.4): {candidate.prob_ns_or_heavier:.2f}")
            return False

        return True

    def stage2_filter(self, candidate: Candidate) -> bool:
        """
        Stage 2: Check negative space constraints.

        This would query WISE/GALEX/photometry in production.
        For now, we mark for later validation.
        """
        # Mark that negative space needs checking
        candidate.metadata['needs_negative_space_check'] = True

        # Placeholder: would check actual data
        # candidate.passed_negative_space = self._check_negative_space(candidate)

        # For candidates that pass basic criteria, assume clean until verified
        if candidate.prob_ns_or_heavier > 0.5:
            candidate.cleanliness_score = 0.8
        else:
            candidate.cleanliness_score = 0.6

        return True

    def _compute_physics_score(self, candidate: Candidate) -> float:
        """Custom physics score emphasizing compact companion probability."""
        base_score = super()._compute_physics_score(candidate)

        # Bonus for high NS/BH probability
        prob_bonus = candidate.prob_ns_or_heavier * 0.3

        return min(1.0, base_score + prob_bonus)
