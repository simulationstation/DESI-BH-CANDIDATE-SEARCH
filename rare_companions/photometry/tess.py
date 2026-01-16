"""
TESS photometric analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import lightkurve
try:
    import lightkurve as lk
    HAS_LIGHTKURVE = True
except ImportError:
    HAS_LIGHTKURVE = False
    logger.warning("lightkurve not installed - TESS analysis limited")


@dataclass
class TESSResult:
    """TESS photometric analysis result."""
    has_data: bool
    n_sectors: int
    n_points: int
    scatter_ppm: float  # Overall scatter in ppm
    amplitude_95_ppm: float  # 95% upper limit on variability
    period_peak: Optional[float]  # Best periodogram period (days)
    period_power: float  # Power at best period
    eclipse_depth_limit: float  # Upper limit on eclipse depth (ppm)
    is_variable: bool
    variability_flag: str  # 'quiet', 'low_var', 'high_var'

    def to_dict(self) -> Dict:
        return {
            'has_data': self.has_data,
            'n_sectors': self.n_sectors,
            'n_points': self.n_points,
            'scatter_ppm': self.scatter_ppm,
            'amplitude_95_ppm': self.amplitude_95_ppm,
            'period_peak': self.period_peak,
            'period_power': self.period_power,
            'eclipse_depth_limit': self.eclipse_depth_limit,
            'is_variable': self.is_variable,
            'variability_flag': self.variability_flag
        }


class TESSAnalyzer:
    """
    Analyze TESS photometry for variability.
    """

    def __init__(self, cache_dir: str = "cache/tess"):
        self.cache_dir = cache_dir

    def analyze(self, ra: float, dec: float,
                search_radius: float = 21.0) -> TESSResult:
        """
        Analyze TESS photometry for a target.

        Parameters
        ----------
        ra : float
            Right ascension (degrees)
        dec : float
            Declination (degrees)
        search_radius : float
            Search radius (arcsec)

        Returns
        -------
        TESSResult
            Analysis results
        """
        if not HAS_LIGHTKURVE:
            return self._empty_result()

        try:
            # Search for TESS data
            search = lk.search_lightcurve(
                f"{ra} {dec}",
                mission='TESS',
                radius=search_radius
            )

            if len(search) == 0:
                return self._empty_result()

            # Download and stitch light curves
            lc_collection = search.download_all()
            if lc_collection is None or len(lc_collection) == 0:
                return self._empty_result()

            lc = lc_collection.stitch()

            # Remove NaNs and outliers
            lc = lc.remove_nans().remove_outliers(sigma=5)

            if len(lc.flux) < 100:
                return self._empty_result()

            # Basic statistics
            flux = lc.flux.value
            flux_err = lc.flux_err.value if hasattr(lc, 'flux_err') else np.zeros_like(flux)

            median_flux = np.median(flux)
            scatter = np.std(flux) / median_flux * 1e6  # ppm

            # 95% amplitude
            flux_norm = flux / median_flux
            amplitude_95 = np.percentile(np.abs(flux_norm - 1), 95) * 1e6

            # Periodogram
            try:
                pg = lc.to_periodogram(method='bls', minimum_period=0.5, maximum_period=30)
                period_peak = float(pg.period_at_max_power.value)
                period_power = float(pg.max_power.value)
            except:
                period_peak = None
                period_power = 0.0

            # Eclipse depth limit (3-sigma)
            eclipse_limit = 3 * scatter

            # Variability classification
            if scatter < 5000:  # < 0.5%
                variability_flag = 'quiet'
                is_variable = False
            elif scatter < 20000:  # < 2%
                variability_flag = 'low_var'
                is_variable = False
            else:
                variability_flag = 'high_var'
                is_variable = True

            return TESSResult(
                has_data=True,
                n_sectors=len(search),
                n_points=len(flux),
                scatter_ppm=scatter,
                amplitude_95_ppm=amplitude_95,
                period_peak=period_peak,
                period_power=period_power,
                eclipse_depth_limit=eclipse_limit,
                is_variable=is_variable,
                variability_flag=variability_flag
            )

        except Exception as e:
            logger.warning(f"TESS analysis failed: {e}")
            return self._empty_result()

    def _empty_result(self) -> TESSResult:
        """Return empty result for no data or errors."""
        return TESSResult(
            has_data=False,
            n_sectors=0,
            n_points=0,
            scatter_ppm=0.0,
            amplitude_95_ppm=0.0,
            period_peak=None,
            period_power=0.0,
            eclipse_depth_limit=0.0,
            is_variable=False,
            variability_flag='no_data'
        )


def get_tess_variability(ra: float, dec: float) -> TESSResult:
    """
    Convenience function for TESS analysis.

    Parameters
    ----------
    ra : float
        Right ascension (degrees)
    dec : float
        Declination (degrees)

    Returns
    -------
    TESSResult
        Analysis results
    """
    analyzer = TESSAnalyzer()
    return analyzer.analyze(ra, dec)
