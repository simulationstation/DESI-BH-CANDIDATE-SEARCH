"""
Per-candidate dossier generator.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

from ..experiments.base import Candidate
from ..ingest.unified import RVTimeSeries

logger = logging.getLogger(__name__)


class DossierGenerator:
    """
    Generate detailed dossiers for individual candidates.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, candidate: Candidate,
                 timeseries: Optional[RVTimeSeries] = None,
                 kill_sheet: Optional[Dict] = None) -> str:
        """
        Generate a dossier for a candidate.

        Parameters
        ----------
        candidate : Candidate
            The candidate to document
        timeseries : RVTimeSeries, optional
            Original time series data
        kill_sheet : dict, optional
            Filter survival information

        Returns
        -------
        str
            Path to saved dossier
        """
        dossier = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'pipeline_version': '1.0.0'
            },
            'target': {
                'targetid': candidate.targetid,
                'source_id': candidate.source_id,
                'ra': candidate.ra,
                'dec': candidate.dec
            },
            'rv_summary': {
                'n_epochs': candidate.n_epochs,
                'delta_rv': candidate.delta_rv,
                'S_robust': candidate.S_robust,
                'passed_hardening': candidate.passed_hardening
            },
            'orbital_solution': {
                'best_period': candidate.best_period,
                'best_K': candidate.best_K,
                'mass_function': candidate.mass_function,
                'm2_min': candidate.m2_min,
                'note': 'From fast circular screen; MCMC for confirmed period uncertainty'
            },
            'companion_probabilities': {
                'prob_ns_or_heavier': candidate.prob_ns_or_heavier,
                'prob_bh': candidate.prob_bh,
                'note': 'Assuming isotropic inclination distribution'
            },
            'scores': {
                'physics_score': candidate.physics_score,
                'cleanliness_score': candidate.cleanliness_score,
                'period_reliability_score': candidate.period_reliability_score,
                'followup_score': candidate.followup_score,
                'total_score': candidate.total_score
            },
            'validation': {
                'passed_negative_space': candidate.passed_negative_space,
                'kill_reasons': candidate.kill_reasons
            },
            'experiment': candidate.experiment,
            'additional_metadata': candidate.metadata
        }

        if kill_sheet:
            dossier['kill_sheet'] = kill_sheet

        if timeseries:
            dossier['rv_epochs'] = [
                {
                    'mjd': e.mjd,
                    'rv': e.rv,
                    'rv_err': e.rv_err,
                    'instrument': e.instrument,
                    'survey': e.survey
                }
                for e in timeseries.epochs
            ]

        # Save
        filename = f"dossier_{candidate.targetid}.json"
        path = os.path.join(self.output_dir, filename)

        with open(path, 'w') as f:
            json.dump(dossier, f, indent=2, default=str)

        logger.info(f"Saved dossier: {path}")

        return path

    def generate_batch(self, candidates: List[Candidate]) -> List[str]:
        """Generate dossiers for multiple candidates."""
        paths = []
        for candidate in candidates:
            path = self.generate(candidate)
            paths.append(path)
        return paths
