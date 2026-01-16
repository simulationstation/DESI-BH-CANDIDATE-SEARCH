"""
E4: Brown dwarf desert systems.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from .base import BaseExperiment, Candidate
from ..rv.hardening import RVMetrics
from ..orbit.fast_screen import FastOrbitResult
from ..config import Config

logger = logging.getLogger(__name__)


class BrownDwarfExperiment(BaseExperiment):
    """
    Search for brown dwarf desert systems.

    Target: companions with M2_min ~ 13-80 M_Jupiter (0.013-0.08 Msun).
    The "brown dwarf desert" is the dearth of companions in this mass range.
    """

    name = "E4_brown_dwarf"
    description = "Brown dwarf desert systems (13-80 M_Jupiter)"

    # Mass boundaries
    M_JUPITER = 0.000954588  # Solar masses
    BD_MIN = 13 * M_JUPITER  # ~0.012 Msun
    BD_MAX = 80 * M_JUPITER  # ~0.076 Msun
    STELLAR_MIN = 0.08  # Hydrogen burning limit

    def __init__(self, config: Config):
        super().__init__(config)

        exp_config = config.experiments.get('E4_brown_dwarf', {})
        params = getattr(exp_config, 'custom_params', {})

        self.m2_min_lower = params.get('m2_min_lower', self.BD_MIN)
        self.m2_min_upper = params.get('m2_min_upper', self.STELLAR_MIN)

    def stage1_filter(self, candidate: Candidate, metrics: RVMetrics,
                      orbit: FastOrbitResult) -> bool:
        """
        Stage 1: Select brown dwarf mass range.
        """
        # Must pass hardening
        if not candidate.passed_hardening:
            candidate.kill_reasons.append("Failed RV hardening")
            return False

        # Check M2_min is in brown dwarf range
        # Note: M2_min could be in range even if true mass is higher (inclination)
        if candidate.m2_min > self.STELLAR_MIN:
            # M2_min > stellar limit means even edge-on gives stellar companion
            candidate.kill_reasons.append(f"M2_min too high for BD: {candidate.m2_min:.4f}")
            return False

        if candidate.m2_min < self.BD_MIN * 0.5:
            # Very low M2_min likely a planet or noise
            candidate.kill_reasons.append(f"M2_min too low: {candidate.m2_min:.4f}")
            return False

        return True

    def stage2_filter(self, candidate: Candidate) -> bool:
        """
        Stage 2: Verify no IR excess (would indicate luminous companion).
        """
        # In production, check WISE colors
        # BD companions should not produce significant IR excess

        # For now, mark for verification
        candidate.metadata['needs_ir_check'] = True

        # Score based on how well M2_min fits BD range
        if self.BD_MIN <= candidate.m2_min <= self.BD_MAX:
            candidate.physics_score *= 1.3
            candidate.metadata['in_bd_desert'] = True
        else:
            candidate.metadata['in_bd_desert'] = False

        return True

    def _compute_physics_score(self, candidate: Candidate) -> float:
        """Score emphasizing BD mass range."""
        # For BD desert, we want M2_min in specific range
        if self.BD_MIN <= candidate.m2_min <= self.BD_MAX:
            mass_score = 1.0
        elif candidate.m2_min < self.BD_MIN:
            mass_score = candidate.m2_min / self.BD_MIN
        else:
            # Above BD but below stellar
            mass_score = 0.5

        # RV amplitude score (BD should give modest K)
        # K ~ 1-10 km/s for BD companions
        if 1.0 <= candidate.best_K <= 20.0:
            k_score = 1.0
        else:
            k_score = 0.5

        # Significance still matters
        sig_score = min(1.0, candidate.S_robust / 20.0)

        return (mass_score + k_score + sig_score) / 3.0
