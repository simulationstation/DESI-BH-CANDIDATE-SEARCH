"""
Unified RV time series representation.
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import json

from .desi import DESIRVLoader, DESIEpoch
from .lamost import LAMOSTLoader, LAMOSTEpoch

logger = logging.getLogger(__name__)


@dataclass
class RVEpoch:
    """Unified RV epoch from any source."""
    mjd: float
    rv: float  # km/s
    rv_err: float  # km/s
    instrument: str  # e.g., 'DESI', 'LAMOST'
    survey: str  # e.g., 'MWS', 'DR7'
    quality: float  # 0-1 quality score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RVTimeSeries:
    """
    Unified RV time series for a single target.

    This is the central data structure passed to all analysis modules.
    """
    targetid: int
    source_id: Optional[int]  # Gaia DR3 source_id
    ra: float
    dec: float
    epochs: List[RVEpoch] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)

    @property
    def mjd(self) -> np.ndarray:
        return np.array([e.mjd for e in self.epochs])

    @property
    def rv(self) -> np.ndarray:
        return np.array([e.rv for e in self.epochs])

    @property
    def rv_err(self) -> np.ndarray:
        return np.array([e.rv_err for e in self.epochs])

    @property
    def baseline(self) -> float:
        """Time baseline in days."""
        if self.n_epochs < 2:
            return 0.0
        return float(np.max(self.mjd) - np.min(self.mjd))

    @property
    def delta_rv(self) -> float:
        """Maximum RV swing."""
        if self.n_epochs < 2:
            return 0.0
        return float(np.max(self.rv) - np.min(self.rv))

    @property
    def instruments(self) -> List[str]:
        """List of unique instruments."""
        return list(set(e.instrument for e in self.epochs))

    def get_epochs_by_instrument(self, instrument: str) -> List[RVEpoch]:
        """Get epochs from a specific instrument."""
        return [e for e in self.epochs if e.instrument == instrument]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'targetid': self.targetid,
            'source_id': self.source_id,
            'ra': self.ra,
            'dec': self.dec,
            'n_epochs': self.n_epochs,
            'baseline_days': self.baseline,
            'delta_rv_kms': self.delta_rv,
            'instruments': self.instruments,
            'epochs': [
                {
                    'mjd': e.mjd,
                    'rv': e.rv,
                    'rv_err': e.rv_err,
                    'instrument': e.instrument,
                    'survey': e.survey,
                    'quality': e.quality
                }
                for e in self.epochs
            ],
            'metadata': self.metadata
        }

    def to_json(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> 'RVTimeSeries':
        """Create from dictionary."""
        epochs = [
            RVEpoch(
                mjd=e['mjd'],
                rv=e['rv'],
                rv_err=e['rv_err'],
                instrument=e['instrument'],
                survey=e['survey'],
                quality=e.get('quality', 1.0)
            )
            for e in d.get('epochs', [])
        ]
        return cls(
            targetid=d['targetid'],
            source_id=d.get('source_id'),
            ra=d['ra'],
            dec=d['dec'],
            epochs=epochs,
            metadata=d.get('metadata', {})
        )


class UnifiedRVLoader:
    """
    Load and merge RV data from multiple sources.
    """

    def __init__(self, desi_loader: Optional[DESIRVLoader] = None,
                 lamost_loader: Optional[LAMOSTLoader] = None):
        self.desi = desi_loader
        self.lamost = lamost_loader

        self._gaia_crossmatch: Dict[int, int] = {}  # targetid -> source_id
        self._target_coords: Dict[int, Tuple[float, float]] = {}  # targetid -> (ra, dec)

    def set_gaia_crossmatch(self, crossmatch: Dict[int, int]):
        """Set DESI targetid to Gaia source_id mapping."""
        self._gaia_crossmatch = crossmatch

    def set_target_coords(self, coords: Dict[int, Tuple[float, float]]):
        """Set target coordinates from DESI metadata."""
        self._target_coords = coords

    def get_rv_timeseries(self, targetid: int,
                          ra: Optional[float] = None,
                          dec: Optional[float] = None,
                          include_lamost: bool = True) -> RVTimeSeries:
        """
        Get unified RV time series for a target.

        Parameters
        ----------
        targetid : int
            DESI target ID
        ra, dec : float, optional
            Coordinates (if not in cached metadata)
        include_lamost : bool
            Whether to include LAMOST epochs

        Returns
        -------
        RVTimeSeries
            Unified time series object
        """
        epochs = []

        # Get coordinates
        if ra is None or dec is None:
            if targetid in self._target_coords:
                ra, dec = self._target_coords[targetid]
            else:
                ra, dec = 0.0, 0.0

        # Get DESI epochs
        if self.desi:
            desi_epochs = self.desi.get_epochs(targetid)
            for de in desi_epochs:
                epoch = RVEpoch(
                    mjd=de.mjd,
                    rv=de.rv,
                    rv_err=de.rv_err,
                    instrument='DESI',
                    survey=de.program,
                    quality=1.0 if de.quality_flags == 0 else 0.5,
                    metadata={
                        'snr': de.snr,
                        'teff': de.teff,
                        'logg': de.logg,
                        'feh': de.feh
                    }
                )
                epochs.append(epoch)

                # Update coordinates from first DESI epoch if not set
                if ra == 0.0 and dec == 0.0:
                    # DESI epochs don't have coordinates, would need from metadata
                    pass

        # Get LAMOST epochs
        if include_lamost and self.lamost and ra != 0.0:
            lamost_epochs = self.lamost.query_by_coord(ra, dec)
            for le in lamost_epochs:
                # Apply systematic floor for LAMOST
                effective_err = self.lamost.get_effective_rv_err(le)

                epoch = RVEpoch(
                    mjd=le.mjd,
                    rv=le.rv,
                    rv_err=effective_err,
                    instrument='LAMOST',
                    survey='DR7',
                    quality=min(1.0, le.snr_g / 50.0) if le.snr_g > 0 else 0.5,
                    metadata={
                        'snr_g': le.snr_g,
                        'teff': le.teff,
                        'logg': le.logg,
                        'feh': le.feh,
                        'spectral_type': le.spectral_type
                    }
                )
                epochs.append(epoch)

                # Update coordinates from LAMOST
                if ra == 0.0:
                    ra, dec = le.ra, le.dec

        # Sort by MJD
        epochs.sort(key=lambda e: e.mjd)

        # Get Gaia source_id
        source_id = self._gaia_crossmatch.get(targetid)

        return RVTimeSeries(
            targetid=targetid,
            source_id=source_id,
            ra=ra,
            dec=dec,
            epochs=epochs
        )

    def get_all_timeseries(self, min_epochs: int = 3,
                           include_lamost: bool = True) -> List[RVTimeSeries]:
        """
        Get time series for all targets meeting minimum epoch requirement.

        Parameters
        ----------
        min_epochs : int
            Minimum number of epochs required
        include_lamost : bool
            Whether to include LAMOST epochs

        Returns
        -------
        list
            List of RVTimeSeries objects
        """
        if not self.desi:
            return []

        targetids = self.desi.get_targets_with_min_epochs(min_epochs)
        logger.info(f"Getting time series for {len(targetids)} targets with >={min_epochs} epochs")

        timeseries = []
        for targetid in targetids:
            ts = self.get_rv_timeseries(targetid, include_lamost=include_lamost)
            if ts.n_epochs >= min_epochs:
                timeseries.append(ts)

        return timeseries


def merge_rv_sources(desi_epochs: List[DESIEpoch],
                     lamost_epochs: List[LAMOSTEpoch],
                     lamost_loader: Optional[LAMOSTLoader] = None) -> List[RVEpoch]:
    """
    Merge RV epochs from multiple sources into unified format.

    Parameters
    ----------
    desi_epochs : list
        DESI epochs
    lamost_epochs : list
        LAMOST epochs
    lamost_loader : LAMOSTLoader, optional
        For computing effective errors

    Returns
    -------
    list
        Merged and sorted RVEpoch list
    """
    merged = []

    for de in desi_epochs:
        epoch = RVEpoch(
            mjd=de.mjd,
            rv=de.rv,
            rv_err=de.rv_err,
            instrument='DESI',
            survey=de.program,
            quality=1.0 if de.quality_flags == 0 else 0.5
        )
        merged.append(epoch)

    for le in lamost_epochs:
        if lamost_loader:
            effective_err = lamost_loader.get_effective_rv_err(le)
        else:
            effective_err = max(le.rv_err, 3.0)  # Default floor

        epoch = RVEpoch(
            mjd=le.mjd,
            rv=le.rv,
            rv_err=effective_err,
            instrument='LAMOST',
            survey='DR7',
            quality=min(1.0, le.snr_g / 50.0) if le.snr_g > 0 else 0.5
        )
        merged.append(epoch)

    # Sort by MJD
    merged.sort(key=lambda e: e.mjd)

    return merged
