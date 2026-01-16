"""
RV hardening and reliability assessment.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from ..ingest.unified import RVTimeSeries, RVEpoch
from .metrics import (
    compute_rv_significance,
    compute_chi2_constant,
    compute_leverage,
    compute_loo_significance,
    check_same_night_consistency,
    compute_cross_survey_offset,
    RVMetrics
)

logger = logging.getLogger(__name__)


class RVHardener:
    """
    Compute hardened RV metrics for candidate validation.

    This class implements a comprehensive suite of tests to assess
    the reliability of RV variability detections.
    """

    def __init__(self, min_s_robust: float = 5.0, min_chi2_pvalue: float = 1e-6):
        """
        Parameters
        ----------
        min_s_robust : float
            Minimum robust significance to pass
        min_chi2_pvalue : float
            Maximum p-value for chi2 test (reject constant RV)
        """
        self.min_s_robust = min_s_robust
        self.min_chi2_pvalue = min_chi2_pvalue

    def compute_metrics(self, ts: RVTimeSeries) -> RVMetrics:
        """
        Compute all RV hardening metrics for a time series.

        Parameters
        ----------
        ts : RVTimeSeries
            Input time series

        Returns
        -------
        RVMetrics
            Complete metrics object
        """
        if ts.n_epochs < 2:
            return self._empty_metrics()

        mjd = ts.mjd
        rv = ts.rv
        rv_err = ts.rv_err
        instruments = [e.instrument for e in ts.epochs]

        # Basic stats
        delta_rv = float(np.max(rv) - np.min(rv))
        median_err = float(np.median(rv_err))

        # Significance
        S = compute_rv_significance(rv, rv_err)

        # Chi-squared test
        chi2, dof, pvalue = compute_chi2_constant(rv, rv_err)
        chi2_reduced = chi2 / dof if dof > 0 else 0.0

        # Leverage
        d, i_max, d_max = compute_leverage(rv, rv_err)
        has_high_leverage = d_max > 3.0 * np.median(d) if len(d) > 2 else False

        # Calculate leverage fraction
        if len(rv) > 2:
            S_without_max = compute_rv_significance(
                np.delete(rv, i_max),
                np.delete(rv_err, i_max)
            )
            leverage_fraction = 1.0 - (S_without_max / S) if S > 0 else 0.0
        else:
            leverage_fraction = 0.0

        # LOO significance
        if ts.n_epochs >= 3:
            S_loo, S_min_loo = compute_loo_significance(rv, rv_err)
        else:
            S_loo = np.array([S])
            S_min_loo = S

        S_robust = S_min_loo

        # Same-night consistency
        same_night_ok, pairs = check_same_night_consistency(mjd, rv, rv_err)

        # Cross-survey offset
        offset = compute_cross_survey_offset(rv, rv_err, instruments)

        # Quality flags
        is_robust = S_robust >= self.min_s_robust
        passed_chi2 = pvalue < self.min_chi2_pvalue

        return RVMetrics(
            n_epochs=ts.n_epochs,
            delta_rv=delta_rv,
            median_rv_err=median_err,
            S=S,
            S_min_loo=S_min_loo,
            S_robust=S_robust,
            chi2_constant=chi2,
            chi2_dof=dof,
            chi2_pvalue=pvalue,
            chi2_reduced=chi2_reduced,
            d_max=d_max,
            high_leverage_epoch=i_max,
            leverage_fraction=leverage_fraction,
            same_night_consistent=same_night_ok,
            cross_survey_offset=offset,
            has_high_leverage=has_high_leverage,
            is_robust=is_robust,
            passed_chi2=passed_chi2
        )

    def _empty_metrics(self) -> RVMetrics:
        """Return empty metrics for invalid input."""
        return RVMetrics(
            n_epochs=0,
            delta_rv=0.0,
            median_rv_err=0.0,
            S=0.0,
            S_min_loo=0.0,
            S_robust=0.0,
            chi2_constant=0.0,
            chi2_dof=0,
            chi2_pvalue=1.0,
            chi2_reduced=0.0,
            d_max=0.0,
            high_leverage_epoch=0,
            leverage_fraction=0.0,
            same_night_consistent=True,
            cross_survey_offset=None,
            has_high_leverage=False,
            is_robust=False,
            passed_chi2=False
        )

    def passes_hardening(self, metrics: RVMetrics) -> bool:
        """
        Check if metrics pass all hardening criteria.

        Parameters
        ----------
        metrics : RVMetrics
            Computed metrics

        Returns
        -------
        bool
            True if passes all tests
        """
        if metrics.n_epochs < 3:
            return False

        if not metrics.is_robust:
            return False

        if not metrics.passed_chi2:
            return False

        if not metrics.same_night_consistent:
            return False

        return True

    def get_failure_reasons(self, metrics: RVMetrics) -> List[str]:
        """
        Get list of reasons why metrics failed hardening.

        Parameters
        ----------
        metrics : RVMetrics
            Computed metrics

        Returns
        -------
        list
            List of failure reason strings
        """
        reasons = []

        if metrics.n_epochs < 3:
            reasons.append(f"Insufficient epochs ({metrics.n_epochs} < 3)")

        if not metrics.is_robust:
            reasons.append(f"S_robust too low ({metrics.S_robust:.1f} < {self.min_s_robust})")

        if not metrics.passed_chi2:
            reasons.append(f"Chi2 p-value too high ({metrics.chi2_pvalue:.2e} > {self.min_chi2_pvalue})")

        if not metrics.same_night_consistent:
            reasons.append("Same-night epochs inconsistent")

        if metrics.has_high_leverage:
            reasons.append(f"High leverage epoch (d_max={metrics.d_max:.1f})")

        return reasons


def harden_rv_series(ts: RVTimeSeries,
                     min_s_robust: float = 5.0,
                     min_chi2_pvalue: float = 1e-6) -> Tuple[RVMetrics, bool, List[str]]:
    """
    Convenience function to compute metrics and check hardening.

    Parameters
    ----------
    ts : RVTimeSeries
        Input time series
    min_s_robust : float
        Minimum robust significance
    min_chi2_pvalue : float
        Maximum p-value threshold

    Returns
    -------
    metrics : RVMetrics
        Computed metrics
    passed : bool
        Whether hardening tests passed
    reasons : list
        Failure reasons if any
    """
    hardener = RVHardener(min_s_robust, min_chi2_pvalue)
    metrics = hardener.compute_metrics(ts)
    passed = hardener.passes_hardening(metrics)
    reasons = hardener.get_failure_reasons(metrics) if not passed else []

    return metrics, passed, reasons
