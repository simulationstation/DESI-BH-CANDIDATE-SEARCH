"""
Multi-frequency map set abstraction for kSZ analysis.

Provides unified access to multi-frequency CMB maps and supports
creating null tests via difference maps.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|------------------------------------------------|-------------|
| T_ν         | CMB temperature at frequency ν                  | μK          |
| ΔT_null     | Frequency difference null map                   | μK          |
| w_ν         | Inverse-variance weight at frequency ν          | μK⁻²        |
| T_coadd     | Inverse-variance weighted coadd                 | μK          |

Frequency Null Test
-------------------
The kSZ signal has a blackbody spectrum (same at all frequencies), while
tSZ and foregrounds have frequency-dependent spectra. The null map:

    ΔT_null = T_f1 - T_f2

should contain zero kSZ signal if calibration is correct.

References
----------
- ACT DR6 collaboration
- Planck 2018 component separation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    hp = None
    HEALPY_AVAILABLE = False

try:
    from pixell import enmap
    PIXELL_AVAILABLE = True
except ImportError:
    enmap = None
    PIXELL_AVAILABLE = False


@dataclass
class FrequencyMap:
    """Single-frequency map with metadata."""

    data: np.ndarray
    frequency_ghz: float
    beam_fwhm_arcmin: float
    ivar: Optional[np.ndarray] = None  # Inverse variance
    mask: Optional[np.ndarray] = None
    is_healpix: bool = True
    wcs: Any = None  # For CAR maps
    units: str = "uK_CMB"
    source: str = "unknown"

    def __post_init__(self):
        """Validate map."""
        if self.data is None:
            raise ValueError("Map data cannot be None")

        if self.ivar is not None and self.ivar.shape != self.data.shape:
            raise ValueError(f"ivar shape {self.ivar.shape} != data shape {self.data.shape}")

    @property
    def nside(self) -> Optional[int]:
        """HEALPix nside if applicable."""
        if self.is_healpix and HEALPY_AVAILABLE:
            return hp.npix2nside(len(self.data.ravel()))
        return None

    @property
    def valid_fraction(self) -> float:
        """Fraction of unmasked pixels."""
        if self.mask is not None:
            return np.mean(self.mask > 0.5)
        return np.mean(np.isfinite(self.data))


@dataclass
class MapSetResult:
    """Result container for map set operations."""

    data: np.ndarray
    frequencies_used: List[float]
    operation: str  # 'coadd', 'difference', 'weighted_mean'
    weights_applied: bool
    effective_beam_fwhm_arcmin: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MapSet:
    """
    Multi-frequency map set with null test support.

    Provides unified interface to access multiple frequency maps and
    perform operations like coaddition and null map creation.

    Examples
    --------
    >>> mapset = MapSet()
    >>> mapset.add_map(map_150, frequency_ghz=150)
    >>> mapset.add_map(map_090, frequency_ghz=90)
    >>> null_map = mapset.create_null_map(freq1=150, freq2=90)
    >>> coadd = mapset.create_coadd()
    """

    def __init__(self, name: str = "default"):
        """
        Initialize map set.

        Parameters
        ----------
        name : str
            Name for this map set (e.g., 'ACT_DR6', 'Planck_PR4')
        """
        self.name = name
        self._maps: Dict[float, FrequencyMap] = {}
        self._combined_mask: Optional[np.ndarray] = None

    def add_map(
        self,
        data: np.ndarray,
        frequency_ghz: float,
        beam_fwhm_arcmin: float = 1.4,
        ivar: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        is_healpix: bool = True,
        wcs: Any = None,
        source: str = "unknown",
    ) -> None:
        """
        Add a frequency map to the set.

        Parameters
        ----------
        data : np.ndarray
            Map data (HEALPix or CAR)
        frequency_ghz : float
            Central frequency in GHz
        beam_fwhm_arcmin : float
            Beam FWHM in arcmin
        ivar : np.ndarray, optional
            Inverse variance map
        mask : np.ndarray, optional
            Binary mask (1=valid)
        is_healpix : bool
            True for HEALPix, False for CAR
        wcs : WCS, optional
            WCS for CAR maps
        source : str
            Data source description
        """
        freq_map = FrequencyMap(
            data=data,
            frequency_ghz=frequency_ghz,
            beam_fwhm_arcmin=beam_fwhm_arcmin,
            ivar=ivar,
            mask=mask,
            is_healpix=is_healpix,
            wcs=wcs,
            source=source,
        )

        self._maps[frequency_ghz] = freq_map
        self._combined_mask = None  # Reset combined mask

        logger.info(f"Added {frequency_ghz} GHz map from {source} "
                   f"(beam={beam_fwhm_arcmin}', valid={freq_map.valid_fraction:.1%})")

    @property
    def frequencies(self) -> List[float]:
        """List of available frequencies in GHz."""
        return sorted(self._maps.keys())

    @property
    def n_frequencies(self) -> int:
        """Number of frequency maps."""
        return len(self._maps)

    def get_map(self, frequency_ghz: float) -> FrequencyMap:
        """Get map at specified frequency."""
        if frequency_ghz not in self._maps:
            raise KeyError(f"No map at {frequency_ghz} GHz. "
                          f"Available: {self.frequencies}")
        return self._maps[frequency_ghz]

    def get_combined_mask(self) -> np.ndarray:
        """Get intersection of all individual masks."""
        if self._combined_mask is not None:
            return self._combined_mask

        if len(self._maps) == 0:
            raise ValueError("No maps in set")

        # Start with all ones
        first_map = list(self._maps.values())[0]
        combined = np.ones(first_map.data.shape, dtype=np.float32)

        for freq_map in self._maps.values():
            if freq_map.mask is not None:
                combined *= freq_map.mask
            # Also mask NaN/inf values
            combined *= np.isfinite(freq_map.data).astype(np.float32)

        self._combined_mask = combined
        return combined

    def create_null_map(
        self,
        freq1: float,
        freq2: float,
        calibration_factor: float = 1.0,
    ) -> MapSetResult:
        """
        Create null map from frequency difference.

        Creates: ΔT = T_freq1 - calibration_factor × T_freq2

        For a perfect null (kSZ cancels), calibration_factor should
        account for any relative calibration offset between maps.

        Parameters
        ----------
        freq1, freq2 : float
            Frequencies in GHz
        calibration_factor : float
            Relative calibration adjustment for freq2

        Returns
        -------
        MapSetResult
            Difference map with metadata
        """
        map1 = self.get_map(freq1)
        map2 = self.get_map(freq2)

        if map1.data.shape != map2.data.shape:
            raise ValueError(f"Map shapes don't match: "
                           f"{map1.data.shape} vs {map2.data.shape}")

        # Create difference
        null_data = map1.data - calibration_factor * map2.data

        # Propagate uncertainties if available
        if map1.ivar is not None and map2.ivar is not None:
            # Var(A-B) = Var(A) + Var(B) for independent maps
            combined_var = 1.0 / map1.ivar + calibration_factor**2 / map2.ivar
            # Avoid division by zero
            combined_var = np.maximum(combined_var, 1e-20)
        else:
            combined_var = None

        # Effective beam: larger of the two (conservative)
        eff_beam = max(map1.beam_fwhm_arcmin, map2.beam_fwhm_arcmin)

        logger.info(f"Created null map: {freq1} GHz - {calibration_factor}×{freq2} GHz")

        return MapSetResult(
            data=null_data,
            frequencies_used=[freq1, freq2],
            operation='difference',
            weights_applied=False,
            effective_beam_fwhm_arcmin=eff_beam,
            metadata={
                'freq1': freq1,
                'freq2': freq2,
                'calibration_factor': calibration_factor,
                'combined_variance': combined_var,
            }
        )

    def create_coadd(
        self,
        frequencies: Optional[List[float]] = None,
        use_ivar_weights: bool = True,
    ) -> MapSetResult:
        """
        Create inverse-variance weighted coadd.

        T_coadd = Σ_ν (w_ν × T_ν) / Σ_ν w_ν

        Parameters
        ----------
        frequencies : list of float, optional
            Frequencies to include (default: all)
        use_ivar_weights : bool
            If True and ivar available, use inverse-variance weighting

        Returns
        -------
        MapSetResult
            Coadded map
        """
        if frequencies is None:
            frequencies = self.frequencies

        if len(frequencies) == 0:
            raise ValueError("No frequencies specified")

        # Get first map for shape
        first_map = self.get_map(frequencies[0])
        shape = first_map.data.shape

        sum_weighted = np.zeros(shape, dtype=np.float64)
        sum_weights = np.zeros(shape, dtype=np.float64)

        min_beam = np.inf

        for freq in frequencies:
            freq_map = self.get_map(freq)

            if freq_map.data.shape != shape:
                raise ValueError(f"Shape mismatch at {freq} GHz")

            # Determine weights
            if use_ivar_weights and freq_map.ivar is not None:
                weights = freq_map.ivar
            else:
                weights = np.ones(shape, dtype=np.float64)

            # Apply mask
            if freq_map.mask is not None:
                weights = weights * freq_map.mask

            # Handle NaN
            valid = np.isfinite(freq_map.data) & np.isfinite(weights)
            weights = np.where(valid, weights, 0)
            data = np.where(valid, freq_map.data, 0)

            sum_weighted += weights * data
            sum_weights += weights

            min_beam = min(min_beam, freq_map.beam_fwhm_arcmin)

        # Avoid division by zero
        sum_weights = np.maximum(sum_weights, 1e-20)
        coadd = sum_weighted / sum_weights

        # Coadd ivar is sum of individual ivars
        coadd_ivar = sum_weights

        logger.info(f"Created coadd from {len(frequencies)} frequencies: {frequencies}")

        return MapSetResult(
            data=coadd,
            frequencies_used=frequencies,
            operation='coadd',
            weights_applied=use_ivar_weights,
            effective_beam_fwhm_arcmin=min_beam if np.isfinite(min_beam) else None,
            metadata={
                'coadd_ivar': coadd_ivar,
                'n_frequencies': len(frequencies),
            }
        )

    def extract_temperatures(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        frequency_ghz: Optional[float] = None,
        use_coadd: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract temperatures at galaxy positions.

        Parameters
        ----------
        ra, dec : np.ndarray
            Galaxy coordinates in degrees
        frequency_ghz : float, optional
            Specific frequency to use
        use_coadd : bool
            If True, use coadd instead of single frequency

        Returns
        -------
        temperatures : np.ndarray
            Temperature at each position
        weights : np.ndarray
            Inverse variance weights
        """
        if use_coadd:
            result = self.create_coadd()
            data = result.data
            ivar = result.metadata.get('coadd_ivar')
        elif frequency_ghz is not None:
            freq_map = self.get_map(frequency_ghz)
            data = freq_map.data
            ivar = freq_map.ivar
        else:
            # Use highest frequency by default
            freq = max(self.frequencies)
            freq_map = self.get_map(freq)
            data = freq_map.data
            ivar = freq_map.ivar

        # Check if HEALPix
        first_map = list(self._maps.values())[0]
        if first_map.is_healpix:
            if not HEALPY_AVAILABLE:
                raise ImportError("healpy required for HEALPix maps")

            nside = hp.npix2nside(len(data.ravel()))
            theta = np.radians(90.0 - dec)
            phi = np.radians(ra)
            pix = hp.ang2pix(nside, theta, phi)

            temperatures = data.ravel()[pix]

            if ivar is not None:
                weights = ivar.ravel()[pix]
            else:
                weights = np.ones_like(temperatures)
        else:
            # CAR map - use pixell
            if not PIXELL_AVAILABLE:
                raise ImportError("pixell required for CAR maps")

            coords = np.deg2rad(np.array([dec, ra]))
            temperatures = data.at(coords)

            if ivar is not None:
                weights = ivar.at(coords)
            else:
                weights = np.ones_like(temperatures)

        return temperatures, weights

    def validate(self) -> Dict[str, Any]:
        """Validate map set consistency."""
        issues = []

        if len(self._maps) == 0:
            issues.append("No maps in set")
            return {'valid': False, 'issues': issues}

        # Check shape consistency
        shapes = [m.data.shape for m in self._maps.values()]
        if len(set(str(s) for s in shapes)) > 1:
            issues.append(f"Inconsistent shapes: {shapes}")

        # Check HEALPix vs CAR consistency
        types = [m.is_healpix for m in self._maps.values()]
        if len(set(types)) > 1:
            issues.append("Mixed HEALPix and CAR maps")

        # Check mask coverage
        combined = self.get_combined_mask()
        f_valid = np.mean(combined > 0.5)
        if f_valid < 0.1:
            issues.append(f"Very low valid fraction: {f_valid:.1%}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'n_frequencies': self.n_frequencies,
            'frequencies': self.frequencies,
            'valid_fraction': float(f_valid),
        }

    def summary(self) -> str:
        """Generate summary string."""
        lines = [f"MapSet '{self.name}':"]
        lines.append(f"  Frequencies: {self.frequencies} GHz")

        for freq, fmap in sorted(self._maps.items()):
            lines.append(f"  {freq} GHz: beam={fmap.beam_fwhm_arcmin}', "
                        f"valid={fmap.valid_fraction:.1%}, source={fmap.source}")

        return "\n".join(lines)


def load_act_mapset(
    map_dir: str,
    frequencies: List[int] = [90, 150],
    load_ivar: bool = True,
) -> MapSet:
    """
    Load ACT DR6 multi-frequency map set.

    Parameters
    ----------
    map_dir : str
        Directory containing ACT maps
    frequencies : list of int
        Frequencies to load (90, 150, 220)
    load_ivar : bool
        Whether to load inverse variance maps

    Returns
    -------
    MapSet
        Loaded map set
    """
    import os

    mapset = MapSet(name="ACT_DR6")

    beam_fwhm = {90: 2.1, 150: 1.4, 220: 1.0}  # arcmin

    for freq in frequencies:
        map_path = os.path.join(map_dir, f"act_dr6.02_coadd_f{freq:03d}_map.fits")
        ivar_path = os.path.join(map_dir, f"act_dr6.02_coadd_f{freq:03d}_ivar.fits")

        if not os.path.exists(map_path):
            logger.warning(f"Map not found: {map_path}")
            continue

        # Load with pixell or healpy depending on format
        if PIXELL_AVAILABLE:
            try:
                data = enmap.read_map(map_path)
                is_healpix = False
                wcs = data.wcs
                data = np.array(data)

                ivar = None
                if load_ivar and os.path.exists(ivar_path):
                    ivar = np.array(enmap.read_map(ivar_path))

                mapset.add_map(
                    data=data,
                    frequency_ghz=float(freq),
                    beam_fwhm_arcmin=beam_fwhm.get(freq, 1.5),
                    ivar=ivar,
                    is_healpix=False,
                    wcs=wcs,
                    source=f"ACT_DR6_f{freq}",
                )
                continue
            except Exception as e:
                logger.debug(f"pixell load failed: {e}, trying healpy")

        if HEALPY_AVAILABLE:
            try:
                data = hp.read_map(map_path)

                ivar = None
                if load_ivar and os.path.exists(ivar_path):
                    ivar = hp.read_map(ivar_path)

                mapset.add_map(
                    data=data,
                    frequency_ghz=float(freq),
                    beam_fwhm_arcmin=beam_fwhm.get(freq, 1.5),
                    ivar=ivar,
                    is_healpix=True,
                    source=f"ACT_DR6_f{freq}",
                )
            except Exception as e:
                logger.error(f"Failed to load {map_path}: {e}")

    return mapset


def load_planck_mapset(
    map_dir: str,
    components: List[str] = ["commander", "smica"],
) -> MapSet:
    """
    Load Planck PR4 component-separated map set.

    Parameters
    ----------
    map_dir : str
        Directory containing Planck maps
    components : list of str
        Component separation methods to load

    Returns
    -------
    MapSet
        Loaded map set (frequency set to method code for identification)
    """
    import os

    mapset = MapSet(name="Planck_PR4")

    # Use frequency codes for component methods
    freq_codes = {"commander": 100.1, "smica": 100.2, "nilc": 100.3, "sevem": 100.4}
    beam_fwhm = {"commander": 5.0, "smica": 5.0, "nilc": 5.0, "sevem": 5.0}

    for comp in components:
        map_path = os.path.join(map_dir, f"npipe6v20_{comp}_full_map_n2048.fits")

        if not os.path.exists(map_path):
            logger.warning(f"Map not found: {map_path}")
            continue

        if not HEALPY_AVAILABLE:
            logger.error("healpy required for Planck maps")
            continue

        try:
            data = hp.read_map(map_path)

            mapset.add_map(
                data=data,
                frequency_ghz=freq_codes.get(comp, 100.0),
                beam_fwhm_arcmin=beam_fwhm.get(comp, 5.0),
                is_healpix=True,
                source=f"Planck_PR4_{comp}",
            )
        except Exception as e:
            logger.error(f"Failed to load {map_path}: {e}")

    return mapset
