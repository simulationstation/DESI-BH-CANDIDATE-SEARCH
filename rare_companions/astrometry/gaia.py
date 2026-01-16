"""
Gaia astrometry analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import astroquery
try:
    from astroquery.gaia import Gaia
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False
    logger.warning("astroquery not installed - Gaia queries limited")


@dataclass
class GaiaResult:
    """Gaia astrometric analysis result."""
    source_id: Optional[int]
    has_data: bool

    # Basic astrometry
    ra: float
    dec: float
    parallax: float
    parallax_err: float
    pmra: float
    pmdec: float

    # Photometry
    phot_g_mean_mag: float
    bp_rp: float
    abs_g: Optional[float]

    # Quality metrics
    ruwe: float
    aen: float  # Astrometric excess noise
    aen_sig: float  # AEN significance
    ipd_frac_multi_peak: float
    ipd_gof_harmonic_amp: float

    # Derived flags
    is_astrometric_binary: bool  # High RUWE/AEN
    is_blended: bool  # High ipd_frac_multi_peak
    parallax_reliable: bool

    # Kinematics (if distance available)
    distance_pc: Optional[float]
    v_tan: Optional[float]  # Transverse velocity km/s

    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'has_data': self.has_data,
            'ra': self.ra,
            'dec': self.dec,
            'parallax': self.parallax,
            'parallax_err': self.parallax_err,
            'pmra': self.pmra,
            'pmdec': self.pmdec,
            'phot_g_mean_mag': self.phot_g_mean_mag,
            'bp_rp': self.bp_rp,
            'abs_g': self.abs_g,
            'ruwe': self.ruwe,
            'aen': self.aen,
            'aen_sig': self.aen_sig,
            'ipd_frac_multi_peak': self.ipd_frac_multi_peak,
            'ipd_gof_harmonic_amp': self.ipd_gof_harmonic_amp,
            'is_astrometric_binary': self.is_astrometric_binary,
            'is_blended': self.is_blended,
            'parallax_reliable': self.parallax_reliable,
            'distance_pc': self.distance_pc,
            'v_tan': self.v_tan
        }


class GaiaAnalyzer:
    """
    Query and analyze Gaia DR3 data.
    """

    # Quality thresholds
    RUWE_THRESHOLD = 1.4  # Above this suggests non-single
    IPD_BLEND_THRESHOLD = 20.0  # % above this suggests blend
    PARALLAX_SNR_MIN = 5.0  # Minimum for reliable parallax

    def __init__(self, cache_dir: str = "cache/gaia"):
        self.cache_dir = cache_dir
        self._cache: Dict[int, GaiaResult] = {}

    def query_by_source_id(self, source_id: int) -> GaiaResult:
        """
        Query Gaia by source_id.

        Parameters
        ----------
        source_id : int
            Gaia DR3 source_id

        Returns
        -------
        GaiaResult
            Query results
        """
        if source_id in self._cache:
            return self._cache[source_id]

        if not HAS_ASTROQUERY:
            return self._empty_result(source_id)

        try:
            query = f"""
            SELECT
                source_id, ra, dec, parallax, parallax_error,
                pmra, pmdec, phot_g_mean_mag, bp_rp,
                ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
                ipd_frac_multi_peak, ipd_gof_harmonic_amplitude
            FROM gaiadr3.gaia_source
            WHERE source_id = {source_id}
            """

            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            job = Gaia.launch_job(query)
            result = job.get_results()

            if len(result) == 0:
                return self._empty_result(source_id)

            row = result[0]
            gaia_result = self._parse_row(row)
            self._cache[source_id] = gaia_result

            return gaia_result

        except Exception as e:
            logger.warning(f"Gaia query failed for {source_id}: {e}")
            return self._empty_result(source_id)

    def query_by_coord(self, ra: float, dec: float,
                       radius_arcsec: float = 3.0) -> GaiaResult:
        """
        Query Gaia by coordinate.

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
        GaiaResult
            Query results (best match)
        """
        if not HAS_ASTROQUERY:
            return self._empty_result()

        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

            query = f"""
            SELECT
                source_id, ra, dec, parallax, parallax_error,
                pmra, pmdec, phot_g_mean_mag, bp_rp,
                ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
                ipd_frac_multi_peak, ipd_gof_harmonic_amplitude,
                DISTANCE(
                    POINT('ICRS', ra, dec),
                    POINT('ICRS', {ra}, {dec})
                ) AS ang_sep
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600})
            )
            ORDER BY ang_sep ASC
            LIMIT 1
            """

            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            job = Gaia.launch_job(query)
            result = job.get_results()

            if len(result) == 0:
                return self._empty_result()

            row = result[0]
            gaia_result = self._parse_row(row)

            if gaia_result.source_id:
                self._cache[gaia_result.source_id] = gaia_result

            return gaia_result

        except Exception as e:
            logger.warning(f"Gaia coordinate query failed: {e}")
            return self._empty_result()

    def _parse_row(self, row) -> GaiaResult:
        """Parse Gaia query result row."""
        source_id = int(row['source_id'])
        ra = float(row['ra'])
        dec = float(row['dec'])
        parallax = float(row['parallax']) if row['parallax'] else 0.0
        parallax_err = float(row['parallax_error']) if row['parallax_error'] else 999.0
        pmra = float(row['pmra']) if row['pmra'] else 0.0
        pmdec = float(row['pmdec']) if row['pmdec'] else 0.0
        g_mag = float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] else 99.0
        bp_rp = float(row['bp_rp']) if row['bp_rp'] else 0.0
        ruwe = float(row['ruwe']) if row['ruwe'] else 1.0
        aen = float(row['astrometric_excess_noise']) if row['astrometric_excess_noise'] else 0.0
        aen_sig = float(row['astrometric_excess_noise_sig']) if row['astrometric_excess_noise_sig'] else 0.0
        ipd_multi = float(row['ipd_frac_multi_peak']) if row['ipd_frac_multi_peak'] else 0.0
        ipd_harm = float(row['ipd_gof_harmonic_amplitude']) if row['ipd_gof_harmonic_amplitude'] else 0.0

        # Derived quantities
        parallax_snr = parallax / parallax_err if parallax_err > 0 else 0.0
        parallax_reliable = parallax_snr >= self.PARALLAX_SNR_MIN

        # Distance
        if parallax_reliable and parallax > 0:
            distance_pc = 1000.0 / parallax
            abs_g = g_mag - 5 * np.log10(distance_pc) + 5
        else:
            distance_pc = None
            abs_g = None

        # Transverse velocity
        if distance_pc and (pmra != 0 or pmdec != 0):
            pm_total = np.sqrt(pmra**2 + pmdec**2)  # mas/yr
            v_tan = 4.74 * pm_total * distance_pc / 1000  # km/s
        else:
            v_tan = None

        # Flags
        is_astrometric_binary = ruwe > self.RUWE_THRESHOLD or aen_sig > 5
        is_blended = ipd_multi > self.IPD_BLEND_THRESHOLD

        return GaiaResult(
            source_id=source_id,
            has_data=True,
            ra=ra,
            dec=dec,
            parallax=parallax,
            parallax_err=parallax_err,
            pmra=pmra,
            pmdec=pmdec,
            phot_g_mean_mag=g_mag,
            bp_rp=bp_rp,
            abs_g=abs_g,
            ruwe=ruwe,
            aen=aen,
            aen_sig=aen_sig,
            ipd_frac_multi_peak=ipd_multi,
            ipd_gof_harmonic_amp=ipd_harm,
            is_astrometric_binary=is_astrometric_binary,
            is_blended=is_blended,
            parallax_reliable=parallax_reliable,
            distance_pc=distance_pc,
            v_tan=v_tan
        )

    def _empty_result(self, source_id: int = None) -> GaiaResult:
        """Return empty result."""
        return GaiaResult(
            source_id=source_id,
            has_data=False,
            ra=0.0, dec=0.0,
            parallax=0.0, parallax_err=999.0,
            pmra=0.0, pmdec=0.0,
            phot_g_mean_mag=99.0, bp_rp=0.0, abs_g=None,
            ruwe=1.0, aen=0.0, aen_sig=0.0,
            ipd_frac_multi_peak=0.0, ipd_gof_harmonic_amp=0.0,
            is_astrometric_binary=False, is_blended=False,
            parallax_reliable=False,
            distance_pc=None, v_tan=None
        )


def query_gaia_source(source_id: int) -> GaiaResult:
    """
    Convenience function to query Gaia.

    Parameters
    ----------
    source_id : int
        Gaia DR3 source_id

    Returns
    -------
    GaiaResult
        Query results
    """
    analyzer = GaiaAnalyzer()
    return analyzer.query_by_source_id(source_id)
