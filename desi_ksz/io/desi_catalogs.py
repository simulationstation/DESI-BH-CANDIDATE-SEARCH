"""
DESI DR1 LSS galaxy catalog loading and handling.

This module provides classes for loading and managing DESI Large-Scale
Structure (LSS) galaxy catalogs, including BGS, LRG, and ELG tracers.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| RA          | Right Ascension                                 | degrees     |
| DEC         | Declination                                     | degrees     |
| Z           | Spectroscopic redshift                          | dimensionless|
| chi         | Comoving distance                               | Mpc/h       |
| w_tot       | Total weight = w_sys * w_comp * w_zfail * w_fkp | dimensionless|
| w_sys       | Imaging systematics weight                      | dimensionless|
| w_comp      | Completeness weight                             | dimensionless|
| w_zfail     | Redshift failure weight                         | dimensionless|
| w_fkp       | FKP optimal weight                              | dimensionless|
| N_eff       | Effective number of galaxies = (sum w)^2 / sum w^2 | dimensionless|

References
----------
- DESI Collaboration 2024, AJ, 168, 58 (DESI DR1)
- Feldman, Kaiser, Peacock 1994, ApJ, 426, 23 (FKP weighting)
"""

from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import logging

# Try fitsio first (faster), fall back to astropy
try:
    import fitsio
    USE_FITSIO = True
except ImportError:
    from astropy.io import fits
    USE_FITSIO = False

logger = logging.getLogger(__name__)


@dataclass
class DESIGalaxyCatalog:
    """
    Container for DESI galaxy catalog data.

    This class handles loading, storing, and manipulating DESI LSS
    galaxy catalogs with proper weighting and selection.

    Attributes
    ----------
    tracer : str
        Galaxy tracer type (BGS_BRIGHT, BGS_FAINT, LRG, ELG_LOP, QSO)
    ra : np.ndarray
        Right Ascension in degrees
    dec : np.ndarray
        Declination in degrees
    z : np.ndarray
        Spectroscopic redshift
    weights : np.ndarray
        Total combined weights
    targetid : np.ndarray
        DESI target identifiers
    chi : np.ndarray, optional
        Comoving distances (computed on demand)
    positions : np.ndarray, optional
        3D comoving Cartesian positions (computed on demand)

    Examples
    --------
    >>> catalog = load_desi_catalog('LRG', data_dir='data/ksz/catalogs/')
    >>> catalog_subset = catalog.select_redshift_bin(0.4, 0.6)
    >>> print(f"N_eff = {catalog_subset.effective_number:.0f}")
    """

    tracer: str
    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    weights: np.ndarray
    targetid: np.ndarray
    weight_sys: np.ndarray = field(default_factory=lambda: np.array([]))
    weight_comp: np.ndarray = field(default_factory=lambda: np.array([]))
    weight_zfail: np.ndarray = field(default_factory=lambda: np.array([]))
    weight_fkp: np.ndarray = field(default_factory=lambda: np.array([]))
    _chi: Optional[np.ndarray] = field(default=None, repr=False)
    _positions: Optional[np.ndarray] = field(default=None, repr=False)
    _cosmology: Optional[Any] = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.ra)

    @property
    def n_galaxies(self) -> int:
        """Number of galaxies in catalog."""
        return len(self.ra)

    @property
    def effective_number(self) -> float:
        """
        Effective number of galaxies accounting for weights.

        N_eff = (sum w_i)^2 / sum w_i^2
        """
        sum_w = np.sum(self.weights)
        sum_w2 = np.sum(self.weights**2)
        if sum_w2 == 0:
            return 0.0
        return sum_w**2 / sum_w2

    @property
    def z_mean(self) -> float:
        """Mean redshift (weighted)."""
        return np.average(self.z, weights=self.weights)

    @property
    def z_median(self) -> float:
        """Median redshift."""
        return np.median(self.z)

    def compute_comoving_distances(
        self,
        cosmology: Optional[Any] = None,
        h: float = 0.6736,
        Omega_m: float = 0.3153,
    ) -> np.ndarray:
        """
        Compute comoving distances from redshifts.

        Parameters
        ----------
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmology object. If None, uses flat LCDM with given h, Omega_m.
        h : float
            Hubble parameter (used if cosmology is None)
        Omega_m : float
            Matter density (used if cosmology is None)

        Returns
        -------
        np.ndarray
            Comoving distances in Mpc/h
        """
        if self._chi is not None and self._cosmology is not None:
            return self._chi

        try:
            from astropy.cosmology import FlatLambdaCDM
            import astropy.units as u

            if cosmology is None:
                cosmology = FlatLambdaCDM(H0=100 * h, Om0=Omega_m)

            # Compute comoving distance (returns Mpc, we want Mpc/h)
            chi = cosmology.comoving_distance(self.z).to(u.Mpc).value * h

        except ImportError:
            # Fallback: simple integration for flat LCDM
            logger.warning("astropy not available, using simple distance computation")
            chi = self._compute_chi_simple(self.z, h, Omega_m)

        self._chi = chi
        self._cosmology = cosmology
        return chi

    @staticmethod
    def _compute_chi_simple(
        z: np.ndarray, h: float = 0.6736, Omega_m: float = 0.3153
    ) -> np.ndarray:
        """
        Simple comoving distance computation for flat LCDM.

        chi(z) = (c/H_0) * integral_0^z dz' / E(z')
        E(z) = sqrt(Omega_m * (1+z)^3 + Omega_Lambda)
        """
        from scipy.integrate import quad

        c_over_H0 = 2997.92458 / h  # Mpc/h

        def integrand(zp):
            Omega_Lambda = 1.0 - Omega_m
            Ez = np.sqrt(Omega_m * (1 + zp) ** 3 + Omega_Lambda)
            return 1.0 / Ez

        chi = np.zeros_like(z)
        for i, zi in enumerate(z):
            chi[i], _ = quad(integrand, 0, zi)
        chi *= c_over_H0

        return chi

    def compute_positions(
        self,
        cosmology: Optional[Any] = None,
        h: float = 0.6736,
        Omega_m: float = 0.3153,
    ) -> np.ndarray:
        """
        Compute 3D comoving Cartesian positions.

        Parameters
        ----------
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmology object for distance computation
        h : float
            Hubble parameter
        Omega_m : float
            Matter density

        Returns
        -------
        np.ndarray
            Shape (N, 3) array of (x, y, z) positions in Mpc/h
        """
        if self._positions is not None:
            return self._positions

        # Get comoving distances
        chi = self.compute_comoving_distances(cosmology, h, Omega_m)

        # Convert to Cartesian
        ra_rad = np.deg2rad(self.ra)
        dec_rad = np.deg2rad(self.dec)

        x = chi * np.cos(dec_rad) * np.cos(ra_rad)
        y = chi * np.cos(dec_rad) * np.sin(ra_rad)
        z = chi * np.sin(dec_rad)

        self._positions = np.column_stack([x, y, z])
        return self._positions

    @property
    def positions(self) -> np.ndarray:
        """3D comoving Cartesian positions in Mpc/h."""
        if self._positions is None:
            self.compute_positions()
        return self._positions

    @property
    def chi(self) -> np.ndarray:
        """Comoving distances in Mpc/h."""
        if self._chi is None:
            self.compute_comoving_distances()
        return self._chi

    @property
    def angular_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (RA, DEC) in radians."""
        return np.deg2rad(self.ra), np.deg2rad(self.dec)

    def select_redshift_bin(
        self, z_min: float, z_max: float
    ) -> "DESIGalaxyCatalog":
        """
        Return subset in redshift range [z_min, z_max).

        Parameters
        ----------
        z_min : float
            Minimum redshift (inclusive)
        z_max : float
            Maximum redshift (exclusive)

        Returns
        -------
        DESIGalaxyCatalog
            New catalog with selected galaxies
        """
        mask = (self.z >= z_min) & (self.z < z_max)
        return self._apply_mask(mask)

    def select_sky_region(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
    ) -> "DESIGalaxyCatalog":
        """
        Return subset in sky region.

        Parameters
        ----------
        ra_min, ra_max : float
            RA bounds in degrees
        dec_min, dec_max : float
            DEC bounds in degrees

        Returns
        -------
        DESIGalaxyCatalog
            New catalog with selected galaxies
        """
        mask = (
            (self.ra >= ra_min)
            & (self.ra < ra_max)
            & (self.dec >= dec_min)
            & (self.dec < dec_max)
        )
        return self._apply_mask(mask)

    def _apply_mask(self, mask: np.ndarray) -> "DESIGalaxyCatalog":
        """Apply boolean mask to create subset catalog."""
        return DESIGalaxyCatalog(
            tracer=self.tracer,
            ra=self.ra[mask],
            dec=self.dec[mask],
            z=self.z[mask],
            weights=self.weights[mask],
            targetid=self.targetid[mask],
            weight_sys=self.weight_sys[mask] if len(self.weight_sys) else np.array([]),
            weight_comp=self.weight_comp[mask] if len(self.weight_comp) else np.array([]),
            weight_zfail=self.weight_zfail[mask] if len(self.weight_zfail) else np.array([]),
            weight_fkp=self.weight_fkp[mask] if len(self.weight_fkp) else np.array([]),
            _chi=self._chi[mask] if self._chi is not None else None,
            _positions=self._positions[mask] if self._positions is not None else None,
            _cosmology=self._cosmology,
        )

    def downsample(
        self, fraction: float, random_seed: int = 42
    ) -> "DESIGalaxyCatalog":
        """
        Randomly downsample catalog.

        Parameters
        ----------
        fraction : float
            Fraction of galaxies to keep (0 < fraction <= 1)
        random_seed : int
            Random seed for reproducibility

        Returns
        -------
        DESIGalaxyCatalog
            Downsampled catalog
        """
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        rng = np.random.default_rng(random_seed)
        n_keep = int(len(self) * fraction)
        indices = rng.choice(len(self), size=n_keep, replace=False)
        mask = np.zeros(len(self), dtype=bool)
        mask[indices] = True

        return self._apply_mask(mask)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of catalog."""
        return {
            "tracer": self.tracer,
            "n_galaxies": self.n_galaxies,
            "n_eff": self.effective_number,
            "z_min": float(np.min(self.z)),
            "z_max": float(np.max(self.z)),
            "z_mean": float(self.z_mean),
            "z_median": float(self.z_median),
            "ra_range": (float(np.min(self.ra)), float(np.max(self.ra))),
            "dec_range": (float(np.min(self.dec)), float(np.max(self.dec))),
            "sum_weights": float(np.sum(self.weights)),
        }


def load_desi_catalog(
    tracer: str,
    data_dir: Union[str, Path],
    regions: List[str] = ["N", "S"],
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    apply_weights: bool = True,
) -> DESIGalaxyCatalog:
    """
    Load DESI LSS galaxy catalog.

    Parameters
    ----------
    tracer : str
        Galaxy tracer: BGS_BRIGHT, BGS_FAINT, LRG, ELG_LOP, QSO
    data_dir : str or Path
        Directory containing DESI LSS catalog files
    regions : list of str
        Galactic cap regions to load: ['N'], ['S'], or ['N', 'S']
    z_min : float, optional
        Minimum redshift cut
    z_max : float, optional
        Maximum redshift cut
    apply_weights : bool
        Whether to compute combined weights

    Returns
    -------
    DESIGalaxyCatalog
        Loaded galaxy catalog

    Notes
    -----
    Expected file naming: {tracer}_{region}_clustering.dat.fits
    e.g., LRG_N_clustering.dat.fits

    The total weight is computed as:
        w_tot = WEIGHT_SYS * WEIGHT_COMP * WEIGHT_ZFAIL * WEIGHT_FKP
    """
    data_dir = Path(data_dir)

    all_ra = []
    all_dec = []
    all_z = []
    all_targetid = []
    all_w_sys = []
    all_w_comp = []
    all_w_zfail = []
    all_w_fkp = []

    for region in regions:
        # Try different file naming conventions
        file_patterns = [
            f"{tracer}_{region}_clustering.dat.fits",
            f"{tracer}_{region}_full.dat.fits",
            f"{tracer}_{region}.fits",
        ]

        fits_file = None
        for pattern in file_patterns:
            candidate = data_dir / pattern
            if candidate.exists():
                fits_file = candidate
                break

        if fits_file is None:
            logger.warning(f"No catalog file found for {tracer} region {region}")
            continue

        logger.info(f"Loading {fits_file}")

        # Read FITS file
        if USE_FITSIO:
            data = fitsio.read(fits_file)
            # Column access
            ra = data["RA"]
            dec = data["DEC"]
            z = data["Z"]
            targetid = data["TARGETID"] if "TARGETID" in data.dtype.names else np.arange(len(ra))

            # Weight columns (may have different names)
            w_sys = _get_column(data, ["WEIGHT_SYS", "WEIGHT_SYSTOT"], default=1.0)
            w_comp = _get_column(data, ["WEIGHT_COMP"], default=1.0)
            w_zfail = _get_column(data, ["WEIGHT_ZFAIL"], default=1.0)
            w_fkp = _get_column(data, ["WEIGHT_FKP", "WEIGHT"], default=1.0)

        else:
            with fits.open(fits_file) as hdul:
                data = hdul[1].data
                ra = data["RA"]
                dec = data["DEC"]
                z = data["Z"]
                targetid = data["TARGETID"] if "TARGETID" in data.columns.names else np.arange(len(ra))

                w_sys = _get_column_astropy(data, ["WEIGHT_SYS", "WEIGHT_SYSTOT"], default=1.0)
                w_comp = _get_column_astropy(data, ["WEIGHT_COMP"], default=1.0)
                w_zfail = _get_column_astropy(data, ["WEIGHT_ZFAIL"], default=1.0)
                w_fkp = _get_column_astropy(data, ["WEIGHT_FKP", "WEIGHT"], default=1.0)

        all_ra.append(ra)
        all_dec.append(dec)
        all_z.append(z)
        all_targetid.append(targetid)
        all_w_sys.append(w_sys)
        all_w_comp.append(w_comp)
        all_w_zfail.append(w_zfail)
        all_w_fkp.append(w_fkp)

    if not all_ra:
        raise FileNotFoundError(f"No catalog files found for {tracer} in {data_dir}")

    # Concatenate all regions
    ra = np.concatenate(all_ra)
    dec = np.concatenate(all_dec)
    z = np.concatenate(all_z)
    targetid = np.concatenate(all_targetid)
    w_sys = np.concatenate(all_w_sys)
    w_comp = np.concatenate(all_w_comp)
    w_zfail = np.concatenate(all_w_zfail)
    w_fkp = np.concatenate(all_w_fkp)

    # Compute total weight
    if apply_weights:
        weights = w_sys * w_comp * w_zfail * w_fkp
    else:
        weights = np.ones(len(ra))

    # Create catalog
    catalog = DESIGalaxyCatalog(
        tracer=tracer,
        ra=ra,
        dec=dec,
        z=z,
        weights=weights,
        targetid=targetid,
        weight_sys=w_sys,
        weight_comp=w_comp,
        weight_zfail=w_zfail,
        weight_fkp=w_fkp,
    )

    # Apply redshift cuts if specified
    if z_min is not None or z_max is not None:
        z_min = z_min if z_min is not None else 0.0
        z_max = z_max if z_max is not None else 10.0
        catalog = catalog.select_redshift_bin(z_min, z_max)

    logger.info(f"Loaded {len(catalog)} galaxies, N_eff = {catalog.effective_number:.0f}")
    return catalog


def load_desi_randoms(
    tracer: str,
    data_dir: Union[str, Path],
    regions: List[str] = ["N", "S"],
    n_randoms: int = 1,
    fraction: float = 1.0,
    random_seed: int = 42,
) -> DESIGalaxyCatalog:
    """
    Load DESI random catalog(s).

    Parameters
    ----------
    tracer : str
        Galaxy tracer (same as data catalog)
    data_dir : str or Path
        Directory containing random catalog files
    regions : list of str
        Galactic cap regions
    n_randoms : int
        Number of random files to load (0-17 available)
    fraction : float
        Fraction of randoms to use (for speed)
    random_seed : int
        Random seed for downsampling

    Returns
    -------
    DESIGalaxyCatalog
        Random catalog (weights set to 1)
    """
    data_dir = Path(data_dir)

    all_ra = []
    all_dec = []
    all_z = []

    for region in regions:
        for i in range(n_randoms):
            # Try different file naming conventions
            file_patterns = [
                f"{tracer}_{region}_{i}_clustering.ran.fits",
                f"{tracer}_{region}_{i}_full.ran.fits",
                f"{tracer}_{region}_{i}.ran.fits",
            ]

            fits_file = None
            for pattern in file_patterns:
                candidate = data_dir / pattern
                if candidate.exists():
                    fits_file = candidate
                    break

            if fits_file is None:
                continue

            logger.info(f"Loading randoms: {fits_file}")

            if USE_FITSIO:
                data = fitsio.read(fits_file)
                ra = data["RA"]
                dec = data["DEC"]
                z = data["Z"]
            else:
                with fits.open(fits_file) as hdul:
                    data = hdul[1].data
                    ra = data["RA"]
                    dec = data["DEC"]
                    z = data["Z"]

            all_ra.append(ra)
            all_dec.append(dec)
            all_z.append(z)

    if not all_ra:
        raise FileNotFoundError(f"No random files found for {tracer} in {data_dir}")

    ra = np.concatenate(all_ra)
    dec = np.concatenate(all_dec)
    z = np.concatenate(all_z)

    catalog = DESIGalaxyCatalog(
        tracer=f"{tracer}_randoms",
        ra=ra,
        dec=dec,
        z=z,
        weights=np.ones(len(ra)),
        targetid=np.arange(len(ra)),
    )

    # Downsample if requested
    if fraction < 1.0:
        catalog = catalog.downsample(fraction, random_seed)

    logger.info(f"Loaded {len(catalog)} random points")
    return catalog


def _get_column(
    data: np.ndarray,
    column_names: List[str],
    default: float = 1.0,
) -> np.ndarray:
    """Get column from fitsio record array with fallback names."""
    for name in column_names:
        if name in data.dtype.names:
            col = data[name]
            # Handle masked/invalid values
            if hasattr(col, "filled"):
                col = col.filled(default)
            return np.asarray(col, dtype=np.float64)
    return np.full(len(data), default, dtype=np.float64)


def _get_column_astropy(
    data,
    column_names: List[str],
    default: float = 1.0,
) -> np.ndarray:
    """Get column from astropy FITS data with fallback names."""
    for name in column_names:
        if name in data.columns.names:
            col = data[name]
            return np.asarray(col, dtype=np.float64)
    return np.full(len(data), default, dtype=np.float64)
