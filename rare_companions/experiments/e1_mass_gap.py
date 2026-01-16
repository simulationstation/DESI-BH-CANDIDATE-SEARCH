"""
E1: Mass-gap compact companion candidates (2-5 Msun).
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


class MassGapExperiment(BaseExperiment):
    """
    Search for mass-gap compact companions.

    Targets companions with M2_min in 2-5 Msun range - the "mass gap"
    between neutron stars and typical stellar-mass black holes.
    """

    name = "E1_mass_gap"
    description = "Mass-gap compact companion candidates (2-5 Msun)"

    def __init__(self, config: Config):
        super().__init__(config)

        # Get experiment-specific params
        exp_config = config.experiments.get('E1_mass_gap', {})
        params = getattr(exp_config, 'custom_params', {})

        self.m2_min_lower = params.get('m2_min_lower', 2.0)
        self.m2_min_upper = params.get('m2_min_upper', 5.0)
        self.require_dark = params.get('require_dark', True)

    def stage1_filter(self, candidate: Candidate, metrics: RVMetrics,
                      orbit: FastOrbitResult) -> bool:
        """
        Stage 1 filter for mass-gap candidates.

        Requires:
        - M2_min > 1.5 Msun (to potentially be in mass gap)
        - Passed RV hardening
        """
        # Must have significant mass function
        if candidate.m2_min < 1.5:
            return False

        # Must pass hardening
        if not candidate.passed_hardening:
            candidate.kill_reasons.append("Failed RV hardening")
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

        # For mass gap, we want significant probability in 2-5 Msun
        # Proxy: prob_gt_1p4 > 0.5
        if candidate.prob_ns_or_heavier < 0.5:
            candidate.kill_reasons.append(f"Low Pr(M2>1.4): {candidate.prob_ns_or_heavier:.2f}")
            return False

        return True

    def stage2_filter(self, candidate: Candidate) -> bool:
        """
        Stage 2 filter: confirm mass-gap range.
        """
        # Check if M2_min is in target range
        if candidate.m2_min < self.m2_min_lower:
            candidate.kill_reasons.append(f"M2_min too low: {candidate.m2_min:.2f} < {self.m2_min_lower}")
            return False

        # High end check (M2_min shouldn't be absurdly high suggesting bad fit)
        if candidate.m2_min > 20.0:
            candidate.kill_reasons.append(f"M2_min suspiciously high: {candidate.m2_min:.2f}")
            return False

        # Update physics score for mass-gap specific criteria
        # Higher score if closer to 3-5 Msun range
        if 3.0 <= candidate.m2_min <= 5.0:
            candidate.physics_score *= 1.2

        return True
