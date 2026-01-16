"""
X-ray and radio catalog cross-matching.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import astroquery
try:
    from astroquery.vizier import Vizier
    from astroquery.heasarc import Heasarc
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False


@dataclass
class XrayRadioResult:
    """X-ray and radio cross-match results."""
    has_xray_detection: bool
    has_radio_detection: bool

    # X-ray
    xray_catalog: Optional[str]
    xray_flux: Optional[float]  # erg/s/cm^2
    xray_luminosity: Optional[float]  # erg/s (if distance known)
    xray_upper_limit: float  # erg/s (typical survey limit)

    # Radio
    radio_catalog: Optional[str]
    radio_flux: Optional[float]  # mJy
    radio_luminosity: Optional[float]  # erg/s/Hz (if distance known)
    radio_upper_limit: float  # mJy (typical survey limit)

    # Assessment
    consistent_with_quiescent: bool
    notes: str

    def to_dict(self) -> Dict:
        return {
            'has_xray_detection': self.has_xray_detection,
            'has_radio_detection': self.has_radio_detection,
            'xray_catalog': self.xray_catalog,
            'xray_flux': self.xray_flux,
            'xray_luminosity': self.xray_luminosity,
            'xray_upper_limit': self.xray_upper_limit,
            'radio_catalog': self.radio_catalog,
            'radio_flux': self.radio_flux,
            'radio_luminosity': self.radio_luminosity,
            'radio_upper_limit': self.radio_upper_limit,
            'consistent_with_quiescent': self.consistent_with_quiescent,
            'notes': self.notes
        }


class XrayRadioAnalyzer:
    """
    Cross-match with X-ray and radio catalogs.
    """

    # Survey limits (approximate)
    ROSAT_LIMIT = 1e-13  # erg/s/cm^2 (0.1-2.4 keV)
    XMM_LIMIT = 1e-14  # erg/s/cm^2
    CHANDRA_LIMIT = 1e-15  # erg/s/cm^2
    NVSS_LIMIT = 2.5  # mJy
    FIRST_LIMIT = 1.0  # mJy
    VLASS_LIMIT = 0.12  # mJy

    def __init__(self, cache_dir: str = "cache/xray_radio"):
        self.cache_dir = cache_dir

    def analyze(self, ra: float, dec: float,
                distance_pc: float = None) -> XrayRadioResult:
        """
        Cross-match with X-ray and radio catalogs.

        Parameters
        ----------
        ra : float
            Right ascension (degrees)
        dec : float
            Declination (degrees)
        distance_pc : float, optional
            Distance in parsecs (for luminosity calculation)

        Returns
        -------
        XrayRadioResult
            Cross-match results
        """
        if not HAS_ASTROQUERY:
            return self._empty_result()

        xray_result = self._query_xray(ra, dec, distance_pc)
        radio_result = self._query_radio(ra, dec, distance_pc)

        # Combine results
        has_xray = xray_result['detected']
        has_radio = radio_result['detected']

        # Determine if consistent with quiescent compact object
        # Quiescent NS/BH typically have L_X < 10^32 erg/s
        notes = []

        if has_xray:
            if xray_result['luminosity'] and xray_result['luminosity'] > 1e32:
                notes.append("X-ray luminosity suggests active accretion")
                consistent = False
            else:
                notes.append("X-ray detection consistent with quiescent")
                consistent = True
        else:
            notes.append("No X-ray detection (upper limit only)")
            consistent = True

        if has_radio:
            notes.append(f"Radio detection in {radio_result['catalog']}")
            # Radio could indicate jet activity
            if radio_result['flux'] > 10:  # > 10 mJy
                notes.append("Strong radio source - possible jet")

        return XrayRadioResult(
            has_xray_detection=has_xray,
            has_radio_detection=has_radio,
            xray_catalog=xray_result['catalog'],
            xray_flux=xray_result['flux'],
            xray_luminosity=xray_result['luminosity'],
            xray_upper_limit=self._get_xray_upper_limit(distance_pc),
            radio_catalog=radio_result['catalog'],
            radio_flux=radio_result['flux'],
            radio_luminosity=radio_result['luminosity'],
            radio_upper_limit=self.FIRST_LIMIT,
            consistent_with_quiescent=consistent,
            notes="; ".join(notes)
        )

    def _query_xray(self, ra: float, dec: float,
                    distance_pc: float = None) -> Dict:
        """Query X-ray catalogs."""
        result = {
            'detected': False,
            'catalog': None,
            'flux': None,
            'luminosity': None
        }

        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

            # Query ROSAT
            v = Vizier(columns=['*'], row_limit=5)
            rosat = v.query_region(coord, radius=30*u.arcsec, catalog='IX/10A/1rxs')

            if rosat and len(rosat) > 0 and len(rosat[0]) > 0:
                result['detected'] = True
                result['catalog'] = 'ROSAT'
                # ROSAT count rate to flux (approximate)
                if 'Count' in rosat[0].colnames:
                    count_rate = float(rosat[0]['Count'][0])
                    result['flux'] = count_rate * 1e-11  # Very rough conversion

            # Query XMM if no ROSAT
            if not result['detected']:
                xmm = v.query_region(coord, radius=10*u.arcsec, catalog='IX/65/xmm4d13s')
                if xmm and len(xmm) > 0 and len(xmm[0]) > 0:
                    result['detected'] = True
                    result['catalog'] = 'XMM-Newton'
                    if 'Flux8' in xmm[0].colnames:
                        result['flux'] = float(xmm[0]['Flux8'][0])

            # Calculate luminosity if distance known
            if result['flux'] and distance_pc:
                d_cm = distance_pc * 3.086e18  # pc to cm
                result['luminosity'] = 4 * np.pi * d_cm**2 * result['flux']

        except Exception as e:
            logger.debug(f"X-ray query failed: {e}")

        return result

    def _query_radio(self, ra: float, dec: float,
                     distance_pc: float = None) -> Dict:
        """Query radio catalogs."""
        result = {
            'detected': False,
            'catalog': None,
            'flux': None,
            'luminosity': None
        }

        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            v = Vizier(columns=['*'], row_limit=5)

            # Query NVSS
            nvss = v.query_region(coord, radius=30*u.arcsec, catalog='VIII/65/nvss')
            if nvss and len(nvss) > 0 and len(nvss[0]) > 0:
                result['detected'] = True
                result['catalog'] = 'NVSS'
                if 'S1.4' in nvss[0].colnames:
                    result['flux'] = float(nvss[0]['S1.4'][0])

            # Query FIRST if no NVSS
            if not result['detected']:
                first = v.query_region(coord, radius=10*u.arcsec, catalog='VIII/92/first14')
                if first and len(first) > 0 and len(first[0]) > 0:
                    result['detected'] = True
                    result['catalog'] = 'FIRST'
                    if 'Fpeak' in first[0].colnames:
                        result['flux'] = float(first[0]['Fpeak'][0])

            # Calculate luminosity if distance known
            if result['flux'] and distance_pc:
                d_cm = distance_pc * 3.086e18
                # Convert mJy to erg/s/cm^2/Hz
                flux_cgs = result['flux'] * 1e-26
                result['luminosity'] = 4 * np.pi * d_cm**2 * flux_cgs

        except Exception as e:
            logger.debug(f"Radio query failed: {e}")

        return result

    def _get_xray_upper_limit(self, distance_pc: float = None) -> float:
        """Get X-ray luminosity upper limit."""
        # Use Chandra limit as deepest
        flux_limit = self.CHANDRA_LIMIT

        if distance_pc:
            d_cm = distance_pc * 3.086e18
            return 4 * np.pi * d_cm**2 * flux_limit

        # Return flux limit if no distance
        return flux_limit

    def _empty_result(self) -> XrayRadioResult:
        """Return empty result."""
        return XrayRadioResult(
            has_xray_detection=False,
            has_radio_detection=False,
            xray_catalog=None,
            xray_flux=None,
            xray_luminosity=None,
            xray_upper_limit=self.CHANDRA_LIMIT,
            radio_catalog=None,
            radio_flux=None,
            radio_luminosity=None,
            radio_upper_limit=self.FIRST_LIMIT,
            consistent_with_quiescent=True,
            notes="No queries performed"
        )
