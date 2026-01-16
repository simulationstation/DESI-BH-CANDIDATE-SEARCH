"""
DESI DR1 per-epoch RV data ingestion.
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from typing import Dict, List, Optional, Tuple
import os
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DESIEpoch:
    """Single DESI RV epoch."""
    mjd: float
    rv: float  # km/s
    rv_err: float  # km/s
    targetid: int
    survey: str
    program: str
    healpix: int
    snr: float
    teff: float
    logg: float
    feh: float
    quality_flags: int


class DESIRVLoader:
    """Load and process DESI DR1 per-epoch RV data."""

    def __init__(self, bright_path: str, dark_path: str, cache_dir: str = "cache"):
        self.bright_path = bright_path
        self.dark_path = dark_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self._epochs_by_target: Dict[int, List[DESIEpoch]] = {}
        self._metadata: Dict[int, Dict] = {}
        self._loaded = False

    def load(self, max_rv_err: float = 50.0) -> int:
        """
        Load DESI RV data from FITS files.

        Returns number of unique targets loaded.
        """
        if self._loaded:
            return len(self._epochs_by_target)

        logger.info("Loading DESI DR1 RV data...")

        all_epochs = []

        for path, program in [(self.bright_path, 'bright'), (self.dark_path, 'dark')]:
            if not os.path.exists(path):
                logger.warning(f"DESI file not found: {path}")
                continue

            logger.info(f"Loading {path}...")
            try:
                with fits.open(path) as hdu:
                    # DESI rvspecfit files have RVTAB and FIBERMAP extensions
                    # RVTAB has VRAD, VRAD_ERR, TEFF, LOGG, FEH
                    # FIBERMAP has MJD, TARGET_RA, TARGET_DEC, etc.
                    # Tables are row-aligned

                    rvtab = hdu['RVTAB'].data
                    fibermap = hdu['FIBERMAP'].data

                    n_rows = len(rvtab)
                    logger.info(f"  Found {n_rows} rows")

                    # Build mask for quality filtering
                    mask = np.ones(n_rows, dtype=bool)
                    mask &= np.isfinite(rvtab['VRAD'])
                    mask &= np.isfinite(rvtab['VRAD_ERR'])
                    mask &= rvtab['VRAD_ERR'] > 0
                    mask &= rvtab['VRAD_ERR'] < max_rv_err
                    mask &= np.abs(rvtab['VRAD']) < 500  # Exclude extreme values

                    # Apply mask
                    indices = np.where(mask)[0]
                    logger.info(f"  After quality filter: {len(indices)} epochs")

                    # Extract data efficiently
                    for idx in indices:
                        epoch = DESIEpoch(
                            mjd=float(fibermap['MJD'][idx]),
                            rv=float(rvtab['VRAD'][idx]),
                            rv_err=float(rvtab['VRAD_ERR'][idx]),
                            targetid=int(rvtab['TARGETID'][idx]),
                            survey='DESI',
                            program=program,
                            healpix=int(rvtab['HEALPIX'][idx]) if 'HEALPIX' in rvtab.dtype.names else 0,
                            snr=float(rvtab['SN_R'][idx]) if 'SN_R' in rvtab.dtype.names else 0,
                            teff=float(rvtab['TEFF'][idx]) if 'TEFF' in rvtab.dtype.names else np.nan,
                            logg=float(rvtab['LOGG'][idx]) if 'LOGG' in rvtab.dtype.names else np.nan,
                            feh=float(rvtab['FEH'][idx]) if 'FEH' in rvtab.dtype.names else np.nan,
                            quality_flags=int(rvtab['RVS_WARN'][idx]) if 'RVS_WARN' in rvtab.dtype.names else 0
                        )
                        all_epochs.append(epoch)

                    logger.info(f"  Loaded {len(indices)} epochs from {program}")

            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Group epochs by targetid
        for epoch in all_epochs:
            if epoch.targetid not in self._epochs_by_target:
                self._epochs_by_target[epoch.targetid] = []
            self._epochs_by_target[epoch.targetid].append(epoch)

        # Sort epochs by MJD for each target
        for targetid in self._epochs_by_target:
            self._epochs_by_target[targetid].sort(key=lambda e: e.mjd)

        self._loaded = True
        logger.info(f"Loaded {len(all_epochs)} total epochs for {len(self._epochs_by_target)} unique targets")

        return len(self._epochs_by_target)

    def get_epochs(self, targetid: int) -> List[DESIEpoch]:
        """Get all epochs for a target."""
        return self._epochs_by_target.get(targetid, [])

    def get_all_targetids(self) -> List[int]:
        """Get list of all target IDs."""
        return list(self._epochs_by_target.keys())

    def get_targets_with_min_epochs(self, min_epochs: int = 3) -> List[int]:
        """Get targets with at least min_epochs observations."""
        return [
            tid for tid, epochs in self._epochs_by_target.items()
            if len(epochs) >= min_epochs
        ]

    def get_rv_arrays(self, targetid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get MJD, RV, RV_err arrays for a target."""
        epochs = self.get_epochs(targetid)
        if not epochs:
            return np.array([]), np.array([]), np.array([])

        mjd = np.array([e.mjd for e in epochs])
        rv = np.array([e.rv for e in epochs])
        rv_err = np.array([e.rv_err for e in epochs])

        return mjd, rv, rv_err

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of loaded data."""
        if not self._loaded:
            return {}

        n_epochs_per_target = [len(e) for e in self._epochs_by_target.values()]

        return {
            'n_targets': len(self._epochs_by_target),
            'n_total_epochs': sum(n_epochs_per_target),
            'n_epochs_median': np.median(n_epochs_per_target) if n_epochs_per_target else 0,
            'n_epochs_max': max(n_epochs_per_target) if n_epochs_per_target else 0,
            'n_with_3plus': sum(1 for n in n_epochs_per_target if n >= 3),
            'n_with_5plus': sum(1 for n in n_epochs_per_target if n >= 5),
        }


def load_desi_rv_epochs(bright_path: str, dark_path: str,
                        max_rv_err: float = 50.0) -> DESIRVLoader:
    """
    Convenience function to load DESI RV data.

    Parameters
    ----------
    bright_path : str
        Path to DESI bright program file
    dark_path : str
        Path to DESI dark program file
    max_rv_err : float
        Maximum RV error to include (km/s)

    Returns
    -------
    DESIRVLoader
        Loaded data object
    """
    loader = DESIRVLoader(bright_path, dark_path)
    loader.load(max_rv_err=max_rv_err)
    return loader
