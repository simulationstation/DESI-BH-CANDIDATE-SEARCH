"""
E3: Detached double white dwarf / LISA progenitor candidates.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from .base import BaseExperiment, Candidate
from ..rv.hardening import RVMetrics
from ..orbit.fast_screen import FastOrbitResult
from ..config import Config

logger = logging.getLogger(__name__)


class DWDLISAExperiment(BaseExperiment):
    """
    Search for detached double white dwarf systems.

    These are LISA gravitational wave source progenitors.
    Key: WD primary (from CMD/colors), short periods preferred.
    """

    name = "E3_dwd_lisa"
    description = "Detached double white dwarfs / LISA progenitors"

    def __init__(self, config: Config):
        super().__init__(config)

        exp_config = config.experiments.get('E3_dwd_lisa', {})
        params = getattr(exp_config, 'custom_params', {})

        self.require_wd_primary = params.get('require_wd_primary', True)
        self.period_max = params.get('period_max', 10.0)  # days

        # WD CMD selection box (Gaia)
        self.wd_abs_g_min = 10.0
        self.wd_abs_g_max = 16.0
        self.wd_bp_rp_min = -0.5
        self.wd_bp_rp_max = 1.5

    def stage1_filter(self, candidate: Candidate, metrics: RVMetrics,
                      orbit: FastOrbitResult) -> bool:
        """
        Stage 1: WD primary selection and short period preference.
        """
        # Must pass hardening
        if not candidate.passed_hardening:
            candidate.kill_reasons.append("Failed RV hardening")
            return False

        # Prefer shorter periods for LISA band
        if candidate.best_period > self.period_max:
            # Don't reject, but lower priority
            candidate.metadata['long_period_flag'] = True

        # For DWD, M2 can be lower (WD companion)
        # M2_min > 0.2 Msun is sufficient
        if candidate.m2_min < 0.2:
            candidate.kill_reasons.append(f"M2_min too low for WD: {candidate.m2_min:.2f}")
            return False

        # Mark for WD verification in stage 2
        candidate.metadata['needs_wd_verification'] = True

        return True

    def stage2_filter(self, candidate: Candidate) -> bool:
        """
        Stage 2: Verify WD primary from CMD.

        In production, this would check Gaia colors.
        """
        # Check if Gaia data available
        gaia_data = candidate.metadata.get('gaia', {})

        if gaia_data:
            abs_g = gaia_data.get('abs_g')
            bp_rp = gaia_data.get('bp_rp')

            if abs_g and bp_rp:
                is_wd = (
                    self.wd_abs_g_min <= abs_g <= self.wd_abs_g_max and
                    self.wd_bp_rp_min <= bp_rp <= self.wd_bp_rp_max
                )

                if self.require_wd_primary and not is_wd:
                    candidate.kill_reasons.append("Primary not in WD CMD region")
                    return False

                candidate.metadata['is_wd_primary'] = is_wd

        # Score boost for shorter periods (better for LISA)
        if candidate.best_period < 1.0:
            candidate.physics_score *= 1.5
        elif candidate.best_period < 5.0:
            candidate.physics_score *= 1.2

        return True

    def _compute_physics_score(self, candidate: Candidate) -> float:
        """Score emphasizing short period systems."""
        base_score = super()._compute_physics_score(candidate)

        # Period score: shorter is better for LISA
        period_score = max(0, 1.0 - candidate.best_period / 10.0)

        return (base_score + period_score) / 2.0
