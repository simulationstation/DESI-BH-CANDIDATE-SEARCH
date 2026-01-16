"""
WISE and GALEX catalog queries for SED analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import astroquery
try:
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False


@dataclass
class SEDResult:
    """SED analysis from WISE and GALEX."""
    # WISE
    has_wise: bool
    w1_mag: Optional[float]
    w2_mag: Optional[float]
    w3_mag: Optional[float]
    w4_mag: Optional[float]
    w1_w2: Optional[float]  # Color
    w2_w3: Optional[float]

    # GALEX
    has_galex: bool
    nuv_mag: Optional[float]
    fuv_mag: Optional[float]
    nuv_fuv: Optional[float]  # Color

    # 2MASS
    has_2mass: bool
    j_mag: Optional[float]
    h_mag: Optional[float]
    k_mag: Optional[float]
    j_k: Optional[float]

    # Derived quantities
    ir_excess: bool  # W1-W2 > 0.15 suggests companion/disk
    uv_excess: bool  # GALEX detection suggests hot companion
    sed_flag: str  # 'normal', 'ir_excess', 'uv_excess', 'double_excess'

    def to_dict(self) -> Dict:
        return {
            'has_wise': self.has_wise,
            'w1_mag': self.w1_mag,
            'w2_mag': self.w2_mag,
            'w3_mag': self.w3_mag,
            'w4_mag': self.w4_mag,
            'w1_w2': self.w1_w2,
            'w2_w3': self.w2_w3,
            'has_galex': self.has_galex,
            'nuv_mag': self.nuv_mag,
            'fuv_mag': self.fuv_mag,
            'nuv_fuv': self.nuv_fuv,
            'has_2mass': self.has_2mass,
            'j_mag': self.j_mag,
            'h_mag': self.h_mag,
            'k_mag': self.k_mag,
            'j_k': self.j_k,
            'ir_excess': self.ir_excess,
            'uv_excess': self.uv_excess,
            'sed_flag': self.sed_flag
        }


class WISEGALEXAnalyzer:
    """
    Query WISE, GALEX, and 2MASS for SED analysis.
    """

    # Thresholds for excess detection
    IR_EXCESS_THRESHOLD = 0.15  # W1-W2 > 0.15 suggests companion
    UV_DETECTION_LIMIT = 23.0  # NUV magnitude limit

    def __init__(self, cache_dir: str = "cache/sed"):
        self.cache_dir = cache_dir

    def analyze(self, ra: float, dec: float) -> SEDResult:
        """
        Query WISE, GALEX, and 2MASS.

        Parameters
        ----------
        ra : float
            Right ascension (degrees)
        dec : float
            Declination (degrees)

        Returns
        -------
        SEDResult
            SED analysis results
        """
        if not HAS_ASTROQUERY:
            return self._empty_result()

        wise_result = self._query_wise(ra, dec)
        galex_result = self._query_galex(ra, dec)
        twomass_result = self._query_2mass(ra, dec)

        # Determine excess flags
        ir_excess = False
        if wise_result['w1_w2'] is not None:
            ir_excess = wise_result['w1_w2'] > self.IR_EXCESS_THRESHOLD

        uv_excess = galex_result['has_detection']

        # SED flag
        if ir_excess and uv_excess:
            sed_flag = 'double_excess'
        elif ir_excess:
            sed_flag = 'ir_excess'
        elif uv_excess:
            sed_flag = 'uv_excess'
        else:
            sed_flag = 'normal'

        return SEDResult(
            has_wise=wise_result['has_data'],
            w1_mag=wise_result['w1'],
            w2_mag=wise_result['w2'],
            w3_mag=wise_result['w3'],
            w4_mag=wise_result['w4'],
            w1_w2=wise_result['w1_w2'],
            w2_w3=wise_result['w2_w3'],
            has_galex=galex_result['has_detection'],
            nuv_mag=galex_result['nuv'],
            fuv_mag=galex_result['fuv'],
            nuv_fuv=galex_result['nuv_fuv'],
            has_2mass=twomass_result['has_data'],
            j_mag=twomass_result['j'],
            h_mag=twomass_result['h'],
            k_mag=twomass_result['k'],
            j_k=twomass_result['j_k'],
            ir_excess=ir_excess,
            uv_excess=uv_excess,
            sed_flag=sed_flag
        )

    def _query_wise(self, ra: float, dec: float) -> Dict:
        """Query AllWISE catalog."""
        result = {
            'has_data': False,
            'w1': None, 'w2': None, 'w3': None, 'w4': None,
            'w1_w2': None, 'w2_w3': None
        }

        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            v = Vizier(columns=['*'], row_limit=1)

            wise = v.query_region(coord, radius=3*u.arcsec, catalog='II/328/allwise')

            if wise and len(wise) > 0 and len(wise[0]) > 0:
                row = wise[0][0]
                result['has_data'] = True

                for band, col in [('w1', 'W1mag'), ('w2', 'W2mag'),
                                  ('w3', 'W3mag'), ('w4', 'W4mag')]:
                    if col in wise[0].colnames:
                        val = row[col]
                        if val and not np.ma.is_masked(val):
                            result[band] = float(val)

                if result['w1'] and result['w2']:
                    result['w1_w2'] = result['w1'] - result['w2']

                if result['w2'] and result['w3']:
                    result['w2_w3'] = result['w2'] - result['w3']

        except Exception as e:
            logger.debug(f"WISE query failed: {e}")

        return result

    def _query_galex(self, ra: float, dec: float) -> Dict:
        """Query GALEX catalog."""
        result = {
            'has_detection': False,
            'nuv': None, 'fuv': None, 'nuv_fuv': None
        }

        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            v = Vizier(columns=['*'], row_limit=1)

            galex = v.query_region(coord, radius=5*u.arcsec, catalog='II/335/galex_ais')

            if galex and len(galex) > 0 and len(galex[0]) > 0:
                row = galex[0][0]

                # NUV
                if 'NUVmag' in galex[0].colnames:
                    val = row['NUVmag']
                    if val and not np.ma.is_masked(val) and float(val) < self.UV_DETECTION_LIMIT:
                        result['nuv'] = float(val)
                        result['has_detection'] = True

                # FUV
                if 'FUVmag' in galex[0].colnames:
                    val = row['FUVmag']
                    if val and not np.ma.is_masked(val) and float(val) < self.UV_DETECTION_LIMIT:
                        result['fuv'] = float(val)
                        result['has_detection'] = True

                if result['nuv'] and result['fuv']:
                    result['nuv_fuv'] = result['nuv'] - result['fuv']

        except Exception as e:
            logger.debug(f"GALEX query failed: {e}")

        return result

    def _query_2mass(self, ra: float, dec: float) -> Dict:
        """Query 2MASS catalog."""
        result = {
            'has_data': False,
            'j': None, 'h': None, 'k': None, 'j_k': None
        }

        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            v = Vizier(columns=['*'], row_limit=1)

            twomass = v.query_region(coord, radius=3*u.arcsec, catalog='II/246/out')

            if twomass and len(twomass) > 0 and len(twomass[0]) > 0:
                row = twomass[0][0]
                result['has_data'] = True

                for band, col in [('j', 'Jmag'), ('h', 'Hmag'), ('k', 'Kmag')]:
                    if col in twomass[0].colnames:
                        val = row[col]
                        if val and not np.ma.is_masked(val):
                            result[band] = float(val)

                if result['j'] and result['k']:
                    result['j_k'] = result['j'] - result['k']

        except Exception as e:
            logger.debug(f"2MASS query failed: {e}")

        return result

    def _empty_result(self) -> SEDResult:
        """Return empty result."""
        return SEDResult(
            has_wise=False,
            w1_mag=None, w2_mag=None, w3_mag=None, w4_mag=None,
            w1_w2=None, w2_w3=None,
            has_galex=False,
            nuv_mag=None, fuv_mag=None, nuv_fuv=None,
            has_2mass=False,
            j_mag=None, h_mag=None, k_mag=None, j_k=None,
            ir_excess=False, uv_excess=False, sed_flag='no_data'
        )
