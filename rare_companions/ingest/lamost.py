"""
LAMOST DR7 RV data ingestion.
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Dict, List, Optional, Tuple
import os
import logging
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class LAMOSTEpoch:
    """Single LAMOST RV epoch."""
    mjd: float
    rv: float  # km/s
    rv_err: float  # km/s
    obsid: int
    ra: float
    dec: float
    snr_g: float
    teff: float
    logg: float
    feh: float
    spectral_type: str


class LAMOSTLoader:
    """Load and process LAMOST DR7 RV data."""

    # LAMOST systematic floor for M dwarfs
    SYSTEMATIC_FLOOR_MDWARF = 3.0  # km/s

    def __init__(self, catalog_path: Optional[str] = None, cache_dir: str = "cache"):
        self.catalog_path = catalog_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self._epochs_by_coord: Dict[str, List[LAMOSTEpoch]] = {}
        self._loaded = False

    def _coord_key(self, ra: float, dec: float, tol_arcsec: float = 3.0) -> str:
        """Create a coordinate key for matching."""
        # Round to tolerance
        ra_round = round(ra * 3600 / tol_arcsec) * tol_arcsec / 3600
        dec_round = round(dec * 3600 / tol_arcsec) * tol_arcsec / 3600
        return f"{ra_round:.6f}_{dec_round:.6f}"

    def load(self) -> int:
        """
        Load LAMOST catalog data.

        Returns number of unique sources loaded.
        """
        if self._loaded:
            return len(self._epochs_by_coord)

        if self.catalog_path and os.path.exists(self.catalog_path):
            logger.info(f"Loading LAMOST catalog from {self.catalog_path}...")
            try:
                with fits.open(self.catalog_path) as hdu:
                    data = Table(hdu[1].data)

                for row in data:
                    epoch = LAMOSTEpoch(
                        mjd=float(row.get('MJD', row.get('mjd', 0))),
                        rv=float(row.get('RV', row.get('rv', 0))),
                        rv_err=float(row.get('RV_ERR', row.get('rv_err', 5.0))),
                        obsid=int(row.get('OBSID', row.get('obsid', 0))),
                        ra=float(row.get('RA', row.get('ra', 0))),
                        dec=float(row.get('DEC', row.get('dec', 0))),
                        snr_g=float(row.get('SNR_G', row.get('snrg', 0))),
                        teff=float(row.get('TEFF', row.get('teff', np.nan))),
                        logg=float(row.get('LOGG', row.get('logg', np.nan))),
                        feh=float(row.get('FEH', row.get('feh', np.nan))),
                        spectral_type=str(row.get('SUBCLASS', row.get('subclass', '')))
                    )

                    key = self._coord_key(epoch.ra, epoch.dec)
                    if key not in self._epochs_by_coord:
                        self._epochs_by_coord[key] = []
                    self._epochs_by_coord[key].append(epoch)

                self._loaded = True
                logger.info(f"Loaded {len(self._epochs_by_coord)} LAMOST sources")

            except Exception as e:
                logger.error(f"Error loading LAMOST catalog: {e}")

        return len(self._epochs_by_coord)

    def query_by_coord(self, ra: float, dec: float,
                       radius_arcsec: float = 3.0) -> List[LAMOSTEpoch]:
        """
        Query LAMOST epochs by coordinate.

        If local catalog loaded, uses that. Otherwise queries VizieR.
        """
        if self._loaded:
            key = self._coord_key(ra, dec, tol_arcsec=radius_arcsec)
            return self._epochs_by_coord.get(key, [])

        # Query VizieR if no local catalog
        return self._query_vizier(ra, dec, radius_arcsec)

    def _query_vizier(self, ra: float, dec: float,
                      radius_arcsec: float = 3.0) -> List[LAMOSTEpoch]:
        """Query LAMOST data from VizieR."""
        try:
            from astroquery.vizier import Vizier

            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

            v = Vizier(columns=['*'], row_limit=100)
            result = v.query_region(
                coord,
                radius=radius_arcsec * u.arcsec,
                catalog='V/164/dr7'  # LAMOST DR7
            )

            if not result or len(result) == 0:
                return []

            epochs = []
            for row in result[0]:
                try:
                    epoch = LAMOSTEpoch(
                        mjd=float(row.get('MJD', 0)),
                        rv=float(row.get('RV', 0)),
                        rv_err=max(float(row.get('e_RV', 5.0)), self.SYSTEMATIC_FLOOR_MDWARF),
                        obsid=int(row.get('ObsID', 0)),
                        ra=float(row.get('RAJ2000', ra)),
                        dec=float(row.get('DEJ2000', dec)),
                        snr_g=float(row.get('snrg', 0)),
                        teff=float(row.get('Teff', np.nan)),
                        logg=float(row.get('logg', np.nan)),
                        feh=float(row.get('[Fe/H]', np.nan)),
                        spectral_type=str(row.get('SubClass', ''))
                    )
                    epochs.append(epoch)
                except Exception as e:
                    logger.debug(f"Error parsing LAMOST row: {e}")
                    continue

            return epochs

        except Exception as e:
            logger.warning(f"VizieR query failed for LAMOST: {e}")
            return []

    def get_effective_rv_err(self, epoch: LAMOSTEpoch) -> float:
        """
        Get effective RV error including systematic floor.

        For M dwarfs, LAMOST has a ~3 km/s systematic floor.
        """
        nominal = epoch.rv_err
        if epoch.spectral_type.lower().startswith('m') or epoch.spectral_type.lower().startswith('dm'):
            return np.sqrt(nominal**2 + self.SYSTEMATIC_FLOOR_MDWARF**2)
        return nominal


def load_lamost_epochs(catalog_path: Optional[str] = None) -> LAMOSTLoader:
    """
    Convenience function to load LAMOST data.

    Parameters
    ----------
    catalog_path : str, optional
        Path to local LAMOST catalog file

    Returns
    -------
    LAMOSTLoader
        Loaded data object
    """
    loader = LAMOSTLoader(catalog_path)
    loader.load()
    return loader
