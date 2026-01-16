"""
Report generator for ANALYSIS_REPORT_RARE.md.
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
import logging

from ..experiments.base import ExperimentResult, Candidate
from ..scoring.unified import GlobalLeaderboard, ScoredCandidate
from ..config import Config

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate the main analysis report.
    """

    def __init__(self, config: Config):
        self.config = config

    def generate(self, experiment_results: Dict[str, ExperimentResult],
                 leaderboard: GlobalLeaderboard,
                 output_path: str):
        """
        Generate ANALYSIS_REPORT_RARE.md.

        Parameters
        ----------
        experiment_results : dict
            Results from each experiment
        leaderboard : GlobalLeaderboard
            Global candidate ranking
        output_path : str
            Path to save report
        """
        lines = []

        # Header
        lines.extend(self._header())

        # Executive summary
        lines.extend(self._executive_summary(experiment_results, leaderboard))

        # Data summary
        lines.extend(self._data_summary(experiment_results))

        # Per-experiment results
        for exp_name, result in experiment_results.items():
            lines.extend(self._experiment_section(exp_name, result))

        # Global leaderboard
        lines.extend(self._leaderboard_section(leaderboard))

        # Caveats and limitations
        lines.extend(self._caveats_section())

        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Report saved to {output_path}")

    def _header(self) -> List[str]:
        return [
            "# Rare Companions Multi-Experiment Search Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Pipeline Version:** 1.0.0",
            "",
            "---",
            "",
        ]

    def _executive_summary(self, results: Dict[str, ExperimentResult],
                          leaderboard: GlobalLeaderboard) -> List[str]:
        lines = [
            "## Executive Summary",
            "",
        ]

        # Total candidates
        total_candidates = sum(r.n_final for r in results.values())
        unique_targets = len(leaderboard.candidates)

        lines.extend([
            f"- **Total candidates across all experiments:** {total_candidates}",
            f"- **Unique targets:** {unique_targets}",
            "",
        ])

        # Per-experiment summary
        lines.append("### Experiment Yields")
        lines.append("")
        lines.append("| Experiment | Input | Stage 1 | Stage 2 | Final |")
        lines.append("|------------|-------|---------|---------|-------|")

        for name, result in results.items():
            lines.append(
                f"| {name} | {result.n_input} | {result.n_after_stage1} | "
                f"{result.n_after_stage2} | {result.n_final} |"
            )

        lines.extend(["", "---", ""])

        return lines

    def _data_summary(self, results: Dict[str, ExperimentResult]) -> List[str]:
        # Get total from first experiment that ran
        n_input = 0
        for r in results.values():
            if r.n_input > 0:
                n_input = r.n_input
                break

        lines = [
            "## Data Volume",
            "",
            f"- **Total targets scanned:** {n_input}",
            f"- **Experiments run:** {len(results)}",
            "",
            "---",
            "",
        ]

        return lines

    def _experiment_section(self, name: str, result: ExperimentResult) -> List[str]:
        lines = [
            f"## {name}",
            "",
            f"**Runtime:** {result.runtime_seconds:.1f} seconds",
            "",
            "### Filter Progression",
            "",
            f"- Input: {result.n_input}",
            f"- After Stage 1: {result.n_after_stage1}",
            f"- After Stage 2: {result.n_after_stage2}",
            f"- Final candidates: {result.n_final}",
            "",
        ]

        if result.candidates:
            lines.extend([
                "### Top 10 Candidates",
                "",
                "| Rank | TargetID | M2_min | Pr(NS+) | Pr(BH) | Score |",
                "|------|----------|--------|---------|--------|-------|"
            ])

            for i, c in enumerate(result.candidates[:10]):
                lines.append(
                    f"| {i+1} | {c.targetid} | {c.m2_min:.2f} | "
                    f"{c.prob_ns_or_heavier:.2f} | {c.prob_bh:.2f} | "
                    f"{c.total_score:.3f} |"
                )

            lines.append("")

        lines.extend(["---", ""])

        return lines

    def _leaderboard_section(self, leaderboard: GlobalLeaderboard) -> List[str]:
        ranked = leaderboard.get_top(20)

        lines = [
            "## Global Leaderboard (Top 20)",
            "",
            "Candidates ranked by unified score across all experiments.",
            "",
            "| Rank | TargetID | Score | Uncertainty | Experiments | M2_min | Pr(BH) |",
            "|------|----------|-------|-------------|-------------|--------|--------|"
        ]

        for scored in ranked:
            c = scored.candidate
            exps = ','.join(scored.experiments[:3])
            if len(scored.experiments) > 3:
                exps += f"+{len(scored.experiments)-3}"

            lines.append(
                f"| {scored.rank} | {c.targetid} | {scored.unified_score:.3f} | "
                f"{scored.uncertainty_score:.2f} | {exps} | "
                f"{c.m2_min:.2f} | {c.prob_bh:.2f} |"
            )

        lines.extend(["", "---", ""])

        return lines

    def _caveats_section(self) -> List[str]:
        return [
            "## Caveats and Limitations",
            "",
            "**IMPORTANT: All results are CANDIDATES requiring follow-up confirmation.**",
            "",
            "### Known Limitations",
            "",
            "1. **Period not uniquely determined:** Sparse sampling (typically 3-10 epochs) ",
            "   cannot uniquely constrain orbital periods. The MCMC posteriors reflect this uncertainty.",
            "",
            "2. **Inclination unknown:** All M2 values are MINIMUM masses (sin i = 1). ",
            "   True companion masses are M2 ≥ M2_min.",
            "",
            "3. **Primary mass uncertainty:** M1 estimates rely on spectral type or color-based ",
            "   calibrations with typical uncertainties of 10-20%.",
            "",
            "4. **Blend contamination:** Some candidates may be unresolved blends. ",
            "   Gaia IPD metrics and imaging analysis provide partial constraints.",
            "",
            "5. **No dynamical masses:** Without dense RV monitoring, no dynamical companion ",
            "   masses can be measured. All classifications are probabilistic.",
            "",
            "### What This Pipeline CANNOT Confirm",
            "",
            "- That any candidate IS a black hole or neutron star",
            "- The true orbital period (only constrained ranges)",
            "- The true companion mass (only minimum masses)",
            "",
            "### Recommended Follow-Up",
            "",
            "1. Dense RV monitoring (10-20 epochs over 30-60 days) to:",
            "   - Uniquely determine orbital period",
            "   - Measure precise K and eccentricity",
            "   - Derive dynamical M2_min",
            "",
            "2. High-resolution imaging to rule out blends",
            "",
            "3. X-ray/radio observations to constrain accretion state",
            "",
            "---",
            "",
            "*Report generated automatically by rare_companions pipeline.*",
            "*All results derived from public data and reproducible calculations.*",
            ""
        ]
