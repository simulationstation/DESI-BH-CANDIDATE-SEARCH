"""
Interface for mock catalogs and simulations.

Provides hooks for loading DESI mocks (AbacusSummit, EZmocks)
with velocity information for kSZ validation.
"""

from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MockCatalog:
    """Container for mock catalog data."""
    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    vlos: np.ndarray  # Line-of-sight velocity (km/s)
    weights: np.ndarray
    mock_id: str


class MockCatalogInterface:
    """
    Interface for loading mock galaxy catalogs.

    Supports DESI AbacusSummit and EZmock formats.

    Parameters
    ----------
    mock_type : str
        Mock type: 'abacus', 'ezmock', 'custom'
    mock_dir : Path
        Directory containing mock files

    Examples
    --------
    >>> interface = MockCatalogInterface('abacus', mock_dir='mocks/')
    >>> mock = interface.load_mock(realization=0, tracer='LRG')
    >>> v_los = mock.vlos
    """

    def __init__(
        self,
        mock_type: str = "abacus",
        mock_dir: Optional[Union[str, Path]] = None,
    ):
        self.mock_type = mock_type
        self.mock_dir = Path(mock_dir) if mock_dir else None

    def load_mock(
        self,
        realization: int = 0,
        tracer: str = "LRG",
        region: str = "N",
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> MockCatalog:
        """
        Load a mock catalog realization.

        Parameters
        ----------
        realization : int
            Mock realization number
        tracer : str
            Galaxy tracer type
        region : str
            Sky region (N or S)
        z_min, z_max : float, optional
            Redshift cuts

        Returns
        -------
        MockCatalog
            Mock catalog with velocities
        """
        if self.mock_type == "abacus":
            return self._load_abacus_mock(realization, tracer, region, z_min, z_max)
        elif self.mock_type == "ezmock":
            return self._load_ezmock(realization, tracer, region, z_min, z_max)
        else:
            raise ValueError(f"Unknown mock type: {self.mock_type}")

    def _load_abacus_mock(
        self,
        realization: int,
        tracer: str,
        region: str,
        z_min: Optional[float],
        z_max: Optional[float],
    ) -> MockCatalog:
        """Load AbacusSummit HOD mock."""
        if self.mock_dir is None:
            raise ValueError("mock_dir must be set for AbacusSummit mocks")

        # Expected file pattern (adjust based on actual format)
        file_pattern = f"AbacusSummit_{tracer}_{region}_mock{realization:04d}.fits"
        mock_file = self.mock_dir / file_pattern

        if not mock_file.exists():
            # Try alternative pattern
            file_pattern = f"cutsky_{tracer}_{region}_ph{realization:03d}.fits"
            mock_file = self.mock_dir / file_pattern

        if not mock_file.exists():
            raise FileNotFoundError(f"Mock file not found: {mock_file}")

        logger.info(f"Loading mock: {mock_file}")

        try:
            import fitsio
            data = fitsio.read(mock_file)
        except ImportError:
            from astropy.io import fits
            with fits.open(mock_file) as hdul:
                data = hdul[1].data

        # Extract columns
        ra = data["RA"]
        dec = data["DEC"]
        z = data["Z"]

        # Velocity columns - try different naming conventions
        vlos = None
        for col in ["VLOS", "VZ", "VELOCITY_LOS", "V_LOS"]:
            if col in data.dtype.names:
                vlos = data[col]
                break

        if vlos is None:
            logger.warning("No velocity column found in mock - using zeros")
            vlos = np.zeros(len(ra))

        # Weights
        weights = np.ones(len(ra))
        for col in ["WEIGHT", "WEIGHT_FKP"]:
            if col in data.dtype.names:
                weights = data[col]
                break

        # Apply redshift cuts
        mask = np.ones(len(ra), dtype=bool)
        if z_min is not None:
            mask &= z >= z_min
        if z_max is not None:
            mask &= z < z_max

        return MockCatalog(
            ra=ra[mask],
            dec=dec[mask],
            z=z[mask],
            vlos=vlos[mask],
            weights=weights[mask],
            mock_id=f"abacus_{tracer}_{region}_{realization}",
        )

    def _load_ezmock(
        self,
        realization: int,
        tracer: str,
        region: str,
        z_min: Optional[float],
        z_max: Optional[float],
    ) -> MockCatalog:
        """Load EZmock catalog."""
        if self.mock_dir is None:
            raise ValueError("mock_dir must be set for EZmocks")

        file_pattern = f"EZmock_{tracer}_{region}_{realization:04d}.fits"
        mock_file = self.mock_dir / file_pattern

        if not mock_file.exists():
            raise FileNotFoundError(f"Mock file not found: {mock_file}")

        logger.info(f"Loading EZmock: {mock_file}")

        try:
            import fitsio
            data = fitsio.read(mock_file)
        except ImportError:
            from astropy.io import fits
            with fits.open(mock_file) as hdul:
                data = hdul[1].data

        ra = data["RA"]
        dec = data["DEC"]
        z = data["Z"]
        vlos = data.get("VLOS", np.zeros(len(ra)))
        weights = np.ones(len(ra))

        mask = np.ones(len(ra), dtype=bool)
        if z_min is not None:
            mask &= z >= z_min
        if z_max is not None:
            mask &= z < z_max

        return MockCatalog(
            ra=ra[mask],
            dec=dec[mask],
            z=z[mask],
            vlos=vlos[mask],
            weights=weights[mask],
            mock_id=f"ezmock_{tracer}_{region}_{realization}",
        )

    def get_available_realizations(
        self,
        tracer: str = "LRG",
        region: str = "N",
    ) -> List[int]:
        """List available mock realizations."""
        if self.mock_dir is None:
            return []

        # Find matching files
        patterns = [
            f"*{tracer}*{region}*mock*.fits",
            f"*{tracer}*{region}*ph*.fits",
        ]

        available = []
        for pattern in patterns:
            for f in self.mock_dir.glob(pattern):
                # Extract realization number from filename
                import re
                match = re.search(r"(\d{3,4})", f.name)
                if match:
                    available.append(int(match.group(1)))

        return sorted(set(available))


def load_mock_velocities(
    mock_file: Union[str, Path],
    velocity_column: str = "VLOS",
) -> np.ndarray:
    """
    Load velocity data from mock catalog.

    Parameters
    ----------
    mock_file : str or Path
        Path to mock catalog
    velocity_column : str
        Name of velocity column

    Returns
    -------
    np.ndarray
        Line-of-sight velocities in km/s
    """
    mock_file = Path(mock_file)

    try:
        import fitsio
        data = fitsio.read(mock_file, columns=[velocity_column])
        return data[velocity_column]
    except ImportError:
        from astropy.io import fits
        with fits.open(mock_file) as hdul:
            return hdul[1].data[velocity_column]


def compute_expected_ksz_from_mock(
    mock: MockCatalog,
    tau_bar: float = 1e-4,
    T_CMB: float = 2.7255e6,  # muK
) -> np.ndarray:
    """
    Compute expected kSZ temperature from mock velocities.

    T_kSZ = -τ × v_los / c × T_CMB

    Parameters
    ----------
    mock : MockCatalog
        Mock catalog with velocities
    tau_bar : float
        Mean optical depth
    T_CMB : float
        CMB temperature in muK

    Returns
    -------
    np.ndarray
        Expected kSZ temperatures in muK
    """
    c_km_s = 299792.458  # km/s
    T_ksz = -tau_bar * mock.vlos / c_km_s * T_CMB
    return T_ksz
