"""
ZTF photometric analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import requests

logger = logging.getLogger(__name__)


@dataclass
class ZTFResult:
    """ZTF photometric analysis result."""
    has_data: bool
    n_epochs_g: int
    n_epochs_r: int
    n_epochs_i: int
    baseline_days: float
    scatter_g_mmag: float
    scatter_r_mmag: float
    amplitude_95_mmag: float
    is_variable: bool
    variability_flag: str

    def to_dict(self) -> Dict:
        return {
            'has_data': self.has_data,
            'n_epochs_g': self.n_epochs_g,
            'n_epochs_r': self.n_epochs_r,
            'n_epochs_i': self.n_epochs_i,
            'baseline_days': self.baseline_days,
            'scatter_g_mmag': self.scatter_g_mmag,
            'scatter_r_mmag': self.scatter_r_mmag,
            'amplitude_95_mmag': self.amplitude_95_mmag,
            'is_variable': self.is_variable,
            'variability_flag': self.variability_flag
        }


class ZTFAnalyzer:
    """
    Analyze ZTF photometry for variability.
    """

    ZTF_API = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

    def __init__(self, cache_dir: str = "cache/ztf"):
        self.cache_dir = cache_dir

    def analyze(self, ra: float, dec: float,
                radius_arcsec: float = 3.0) -> ZTFResult:
        """
        Analyze ZTF photometry for a target.

        Parameters
        ----------
        ra : float
            Right ascension (degrees)
        dec : float
            Declination (degrees)
        radius_arcsec : float
            Search radius (arcsec)

        Returns
        -------
        ZTFResult
            Analysis results
        """
        try:
            # Query ZTF via IRSA
            data = self._query_ztf(ra, dec, radius_arcsec)

            if data is None or len(data) == 0:
                return self._empty_result()

            # Separate by filter
            g_data = [d for d in data if d.get('filtercode') == 'zg']
            r_data = [d for d in data if d.get('filtercode') == 'zr']
            i_data = [d for d in data if d.get('filtercode') == 'zi']

            n_g = len(g_data)
            n_r = len(r_data)
            n_i = len(i_data)

            if n_g + n_r + n_i < 10:
                return self._empty_result()

            # Compute statistics
            all_mjd = []
            all_mag = []

            scatter_g = 0.0
            scatter_r = 0.0

            if n_g >= 5:
                mjd_g = np.array([d['mjd'] for d in g_data])
                mag_g = np.array([d['mag'] for d in g_data])
                scatter_g = np.std(mag_g) * 1000  # mmag
                all_mjd.extend(mjd_g)
                all_mag.extend(mag_g)

            if n_r >= 5:
                mjd_r = np.array([d['mjd'] for d in r_data])
                mag_r = np.array([d['mag'] for d in r_data])
                scatter_r = np.std(mag_r) * 1000  # mmag
                all_mjd.extend(mjd_r)
                all_mag.extend(mag_r)

            if len(all_mjd) < 10:
                return self._empty_result()

            all_mjd = np.array(all_mjd)
            all_mag = np.array(all_mag)

            baseline = all_mjd.max() - all_mjd.min()

            # Normalize magnitudes per band and compute amplitude
            # Use maximum scatter
            max_scatter = max(scatter_g, scatter_r) if scatter_g > 0 or scatter_r > 0 else 0.0

            # 95% amplitude estimate
            amplitude_95 = np.percentile(np.abs(all_mag - np.median(all_mag)), 95) * 1000

            # Variability classification
            if max_scatter < 10:  # < 10 mmag
                variability_flag = 'quiet'
                is_variable = False
            elif max_scatter < 50:  # < 50 mmag
                variability_flag = 'low_var'
                is_variable = False
            else:
                variability_flag = 'high_var'
                is_variable = True

            return ZTFResult(
                has_data=True,
                n_epochs_g=n_g,
                n_epochs_r=n_r,
                n_epochs_i=n_i,
                baseline_days=baseline,
                scatter_g_mmag=scatter_g,
                scatter_r_mmag=scatter_r,
                amplitude_95_mmag=amplitude_95,
                is_variable=is_variable,
                variability_flag=variability_flag
            )

        except Exception as e:
            logger.warning(f"ZTF analysis failed: {e}")
            return self._empty_result()

    def _query_ztf(self, ra: float, dec: float,
                   radius_arcsec: float) -> Optional[List[Dict]]:
        """Query ZTF light curve data."""
        try:
            params = {
                'POS': f'CIRCLE {ra} {dec} {radius_arcsec/3600}',
                'FORMAT': 'json',
                'BAD_CATFLAGS_MASK': 32768
            }

            response = requests.get(self.ZTF_API, params=params, timeout=30)

            if response.status_code != 200:
                return None

            data = response.json()

            # Parse response (format varies)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'data' in data:
                return data['data']

            return None

        except Exception as e:
            logger.debug(f"ZTF query failed: {e}")
            return None

    def _empty_result(self) -> ZTFResult:
        """Return empty result."""
        return ZTFResult(
            has_data=False,
            n_epochs_g=0,
            n_epochs_r=0,
            n_epochs_i=0,
            baseline_days=0.0,
            scatter_g_mmag=0.0,
            scatter_r_mmag=0.0,
            amplitude_95_mmag=0.0,
            is_variable=False,
            variability_flag='no_data'
        )


def get_ztf_variability(ra: float, dec: float) -> ZTFResult:
    """
    Convenience function for ZTF analysis.

    Parameters
    ----------
    ra : float
        Right ascension (degrees)
    dec : float
        Declination (degrees)

    Returns
    -------
    ZTFResult
        Analysis results
    """
    analyzer = ZTFAnalyzer()
    return analyzer.analyze(ra, dec)
