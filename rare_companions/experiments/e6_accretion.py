"""
E6: Accretion on/off systems (quiescent XRB candidates).
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from .base import BaseExperiment, Candidate
from ..rv.hardening import RVMetrics
from ..orbit.fast_screen import FastOrbitResult
from ..config import Config

logger = logging.getLogger(__name__)


class AccretionExperiment(BaseExperiment):
    """
    Search for quiescent X-ray binary candidates.

    Target: Dark companion candidates with X-ray/radio detections OR
    deep upper limits constraining accretion.
    """

    name = "E6_accretion"
    description = "Accretion on/off systems (quiescent XRB candidates)"

    def __init__(self, config: Config):
        super().__init__(config)

        exp_config = config.experiments.get('E6_accretion', {})
        params = getattr(exp_config, 'custom_params', {})

        self.require_xray_radio = params.get('require_xray_radio', False)

    def stage1_filter(self, candidate: Candidate, metrics: RVMetrics,
                      orbit: FastOrbitResult) -> bool:
        """
        Stage 1: Select compact companion candidates.
        """
        # Must pass hardening
        if not candidate.passed_hardening:
            candidate.kill_reasons.append("Failed RV hardening")
            return False

        # Need significant M2_min for compact object
        if candidate.m2_min < 0.8:
            candidate.kill_reasons.append(f"M2_min too low for XRB: {candidate.m2_min:.2f}")
            return False

        return True

    def stage2_filter(self, candidate: Candidate) -> bool:
        """
        Stage 2: Check X-ray/radio catalog cross-matches.

        Produces two categories:
        1. "Counterpart detected" - high priority
        2. "Deep upper limits" - quiescent constraints
        """
        # Mark for X-ray/radio checking
        candidate.metadata['needs_xray_radio_check'] = True

        # Placeholder for actual cross-match results
        xray_radio = {
            'has_xray': False,
            'has_radio': False,
            'xray_luminosity': None,
            'xray_upper_limit': 1e30,  # erg/s
            'radio_flux': None,
            'radio_upper_limit': 1.0,  # mJy
        }

        candidate.metadata['xray_radio'] = xray_radio

        # Categorize
        if xray_radio['has_xray'] or xray_radio['has_radio']:
            candidate.metadata['accretion_category'] = 'counterpart_detected'
            candidate.physics_score *= 1.5  # Boost for detection
        else:
            candidate.metadata['accretion_category'] = 'deep_upper_limits'

        return True

    def _compute_physics_score(self, candidate: Candidate) -> float:
        """Score emphasizing X-ray/radio detection or deep limits."""
        base_score = super()._compute_physics_score(candidate)

        xray_radio = candidate.metadata.get('xray_radio', {})

        # Detection boosts score significantly
        if xray_radio.get('has_xray') or xray_radio.get('has_radio'):
            base_score += 0.4

        # Deep upper limits still valuable
        if xray_radio.get('xray_upper_limit', np.inf) < 1e29:
            base_score += 0.2

        return min(1.0, base_score)
