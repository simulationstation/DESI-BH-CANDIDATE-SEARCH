"""
E7: Halo/cluster population candidates.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from .base import BaseExperiment, Candidate
from ..rv.hardening import RVMetrics
from ..orbit.fast_screen import FastOrbitResult
from ..config import Config

logger = logging.getLogger(__name__)


class HaloClusterExperiment(BaseExperiment):
    """
    Search for compact companions in halo/cluster populations.

    Target: High-velocity and/or low-metallicity primaries with
    compact companion indicators.
    """

    name = "E7_halo_cluster"
    description = "Halo/cluster population compact companion candidates"

    def __init__(self, config: Config):
        super().__init__(config)

        exp_config = config.experiments.get('E7_halo_cluster', {})
        params = getattr(exp_config, 'custom_params', {})

        self.min_velocity = params.get('min_velocity', 200.0)  # km/s
        self.max_metallicity = params.get('max_metallicity', -1.0)  # [Fe/H]

    def stage1_filter(self, candidate: Candidate, metrics: RVMetrics,
                      orbit: FastOrbitResult) -> bool:
        """
        Stage 1: Basic compact companion criteria.
        """
        # Must pass hardening
        if not candidate.passed_hardening:
            candidate.kill_reasons.append("Failed RV hardening")
            return False

        # Need significant M2_min
        if candidate.m2_min < 0.5:
            candidate.kill_reasons.append(f"M2_min too low: {candidate.m2_min:.2f}")
            return False

        return True

    def stage2_filter(self, candidate: Candidate) -> bool:
        """
        Stage 2: Check kinematics and metallicity.
        """
        # Mark for kinematic checking
        candidate.metadata['needs_kinematic_check'] = True

        # Placeholder for actual Gaia kinematic data
        kinematics = {
            'v_tan': None,  # Transverse velocity km/s
            'v_total': None,  # Total space velocity km/s
            'metallicity': None,  # [Fe/H]
            'population': 'unknown'  # disk, thick_disk, halo
        }

        # Would populate from Gaia data
        gaia_data = candidate.metadata.get('gaia', {})
        if gaia_data:
            kinematics['v_tan'] = gaia_data.get('v_tan')

        # Metallicity from LAMOST/DESI if available
        rv_metadata = candidate.metadata.get('rv_metrics', {})

        candidate.metadata['kinematics'] = kinematics

        # Classify population
        if kinematics['v_total'] and kinematics['v_total'] > self.min_velocity:
            candidate.metadata['is_high_velocity'] = True
        else:
            candidate.metadata['is_high_velocity'] = False

        if kinematics['metallicity'] and kinematics['metallicity'] < self.max_metallicity:
            candidate.metadata['is_metal_poor'] = True
        else:
            candidate.metadata['is_metal_poor'] = False

        # Determine population
        if candidate.metadata.get('is_high_velocity') or candidate.metadata.get('is_metal_poor'):
            kinematics['population'] = 'halo'
        else:
            kinematics['population'] = 'disk'

        return True

    def _compute_physics_score(self, candidate: Candidate) -> float:
        """Score emphasizing halo membership."""
        base_score = super()._compute_physics_score(candidate)

        # Bonus for halo characteristics
        if candidate.metadata.get('is_high_velocity', False):
            base_score += 0.3

        if candidate.metadata.get('is_metal_poor', False):
            base_score += 0.2

        return min(1.0, base_score)
