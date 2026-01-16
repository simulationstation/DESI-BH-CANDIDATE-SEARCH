"""
E5: Hierarchical triples / RV two-component systems.
"""

import numpy as np
from scipy import optimize
from typing import Dict, List, Optional, Tuple
import logging

from .base import BaseExperiment, Candidate
from ..rv.hardening import RVMetrics
from ..orbit.fast_screen import FastOrbitResult
from ..config import Config

logger = logging.getLogger(__name__)


class HierarchicalExperiment(BaseExperiment):
    """
    Search for hierarchical triple systems.

    Target: RV time series inconsistent with single Keplerian:
    - Sinusoid + drift
    - Two periodicities
    """

    name = "E5_hierarchical"
    description = "Hierarchical triples / two-component RV systems"

    def __init__(self, config: Config):
        super().__init__(config)

        exp_config = config.experiments.get('E5_hierarchical', {})
        params = getattr(exp_config, 'custom_params', {})

        self.min_delta_bic = params.get('min_delta_bic', 10.0)

    def stage1_filter(self, candidate: Candidate, metrics: RVMetrics,
                      orbit: FastOrbitResult) -> bool:
        """
        Stage 1: Check for multi-component signature.
        """
        # Must pass basic hardening
        if not candidate.passed_hardening:
            candidate.kill_reasons.append("Failed RV hardening")
            return False

        # Need enough epochs for model comparison
        if candidate.n_epochs < 5:
            candidate.kill_reasons.append(f"Need 5+ epochs for hierarchical: {candidate.n_epochs}")
            return False

        # Check for multimodality in period search
        orbit_data = candidate.metadata.get('orbit_screen', {})
        if orbit_data.get('is_multimodal', False):
            candidate.metadata['multimodal_period'] = True

        return True

    def stage2_filter(self, candidate: Candidate) -> bool:
        """
        Stage 2: Model comparison (single vs two-component).

        Compare:
        1. Single Keplerian
        2. Keplerian + linear trend
        3. Two Keplerians (simplified)
        """
        # Get RV data from metadata (would be stored in production)
        # For now, mark for detailed analysis

        candidate.metadata['needs_model_comparison'] = True

        # Placeholder BIC comparison
        # In production, would actually fit models
        candidate.metadata['model_comparison'] = {
            'single_keplerian_bic': 0.0,
            'keplerian_plus_trend_bic': 0.0,
            'two_keplerian_bic': 0.0,
            'best_model': 'single_keplerian',
            'delta_bic': 0.0
        }

        return True

    def _compute_physics_score(self, candidate: Candidate) -> float:
        """Score based on evidence for hierarchical structure."""
        base_score = super()._compute_physics_score(candidate)

        # Bonus for multimodal period search
        if candidate.metadata.get('multimodal_period', False):
            base_score += 0.2

        # Bonus for model comparison favoring two-component
        model_comp = candidate.metadata.get('model_comparison', {})
        if model_comp.get('best_model') in ['keplerian_plus_trend', 'two_keplerian']:
            base_score += 0.3

        return min(1.0, base_score)

    @staticmethod
    def fit_keplerian_plus_trend(mjd: np.ndarray, rv: np.ndarray,
                                 rv_err: np.ndarray) -> Dict:
        """
        Fit Keplerian orbit plus linear trend.

        Returns fit results including BIC.
        """
        # This would be a full implementation in production
        # Simplified placeholder
        return {
            'success': False,
            'bic': np.inf,
            'params': {}
        }
