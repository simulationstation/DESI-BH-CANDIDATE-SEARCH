"""
CMB realization generation for covariance estimation and validation.

Provides interfaces to generate Gaussian CMB realizations from power spectra
for use in Monte Carlo covariance estimation and null test validation.
"""

from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    hp = None
    HEALPY_AVAILABLE = False


@dataclass
class CMBRealizationParams:
    """Parameters for CMB realization generation."""
    nside: int = 2048
    lmax: int = 4000
    beam_fwhm_arcmin: float = 1.4
    noise_level_uk_arcmin: float = 10.0
    include_cmb: bool = True
    include_noise: bool = True
    random_seed: Optional[int] = None


class CMBSimulator:
    """
    Generate Gaussian CMB + noise realizations.

    Parameters
    ----------
    power_spectrum : np.ndarray or str
        CMB power spectrum C_ell (in μK² units) or path to file
    params : CMBRealizationParams, optional
        Simulation parameters

    Examples
    --------
    >>> sim = CMBSimulator(power_spectrum='camb_planck2018.dat')
    >>> T_map = sim.generate_realization()
    >>> T_at_pos = sim.get_temperatures_at_positions(ra, dec)
    """

    def __init__(
        self,
        power_spectrum: Optional[np.ndarray] = None,
        params: Optional[CMBRealizationParams] = None,
    ):
        if not HEALPY_AVAILABLE:
            raise ImportError("healpy is required for CMB simulations")

        self.params = params or CMBRealizationParams()

        # Load or generate power spectrum
        if power_spectrum is not None:
            if isinstance(power_spectrum, str):
                self.cl = self._load_power_spectrum(power_spectrum)
            else:
                self.cl = np.asarray(power_spectrum)
        else:
            self.cl = self._default_power_spectrum()

        # Ensure cl extends to lmax
        if len(self.cl) < self.params.lmax + 1:
            # Pad with zeros or extrapolate
            cl_extended = np.zeros(self.params.lmax + 1)
            cl_extended[:len(self.cl)] = self.cl
            self.cl = cl_extended

        # Precompute beam and noise
        self._compute_beam()
        self._compute_noise_cl()

        self.rng = np.random.default_rng(self.params.random_seed)

    def _load_power_spectrum(self, filename: str) -> np.ndarray:
        """Load power spectrum from file."""
        try:
            data = np.loadtxt(filename)
            if data.ndim == 1:
                return data
            else:
                # Assume CAMB format: ell, D_ell or ell, C_ell
                ell = data[:, 0].astype(int)
                cl = data[:, 1]

                # Convert D_ell to C_ell if needed
                if cl[2] > 1e3:  # D_ell ~ 6000 μK² at ell=2
                    cl = cl * 2 * np.pi / (ell * (ell + 1))

                # Reindex to start at ell=0
                cl_full = np.zeros(int(ell.max()) + 1)
                cl_full[ell] = cl
                return cl_full

        except Exception as e:
            logger.warning(f"Could not load power spectrum: {e}")
            return self._default_power_spectrum()

    def _default_power_spectrum(self) -> np.ndarray:
        """Generate approximate CMB power spectrum."""
        ell = np.arange(self.params.lmax + 1)

        # Approximate primary CMB power spectrum
        # Based on Planck best-fit cosmology
        cl = np.zeros(self.params.lmax + 1)

        # Sachs-Wolfe plateau + acoustic peaks (simplified)
        for l in range(2, self.params.lmax + 1):
            # SW plateau + decay
            cl[l] = 6000 * (l / 200) ** (-0.15) * np.exp(-l / 2000)

            # Add acoustic peaks (very simplified)
            for peak in [220, 540, 810]:
                cl[l] += 2000 * np.exp(-0.5 * ((l - peak) / 50) ** 2)

        # Convert from D_ell to C_ell
        cl[2:] *= 2 * np.pi / (ell[2:] * (ell[2:] + 1))

        return cl

    def _compute_beam(self) -> None:
        """Compute beam window function."""
        fwhm_rad = self.params.beam_fwhm_arcmin * np.pi / (180 * 60)
        sigma = fwhm_rad / np.sqrt(8 * np.log(2))

        ell = np.arange(self.params.lmax + 1)
        self.beam = np.exp(-0.5 * ell * (ell + 1) * sigma ** 2)

    def _compute_noise_cl(self) -> None:
        """Compute noise power spectrum."""
        # White noise: N_ell = (σ_pix)² / Ω_pix
        # For noise in μK·arcmin: N_ell = (σ_arcmin)² × (arcmin²/sr)

        noise_uk_arcmin = self.params.noise_level_uk_arcmin
        arcmin_to_rad = np.pi / (180 * 60)

        # N_ell is flat in ell for white noise
        self.noise_cl = (noise_uk_arcmin * arcmin_to_rad) ** 2 * np.ones(self.params.lmax + 1)

    def generate_realization(
        self,
        seed: Optional[int] = None,
        return_components: bool = False,
    ) -> np.ndarray:
        """
        Generate a CMB + noise realization.

        Parameters
        ----------
        seed : int, optional
            Random seed for this realization
        return_components : bool
            If True, return (total, cmb, noise)

        Returns
        -------
        T_map : np.ndarray
            Temperature map (HEALPix RING ordering)
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        npix = hp.nside2npix(self.params.nside)

        # CMB component
        if self.params.include_cmb:
            cl_beam = self.cl * self.beam ** 2
            cmb_map = hp.synfast(
                cl_beam, self.params.nside, lmax=self.params.lmax,
                new=True, verbose=False
            )
        else:
            cmb_map = np.zeros(npix)

        # Noise component
        if self.params.include_noise:
            noise_map = hp.synfast(
                self.noise_cl, self.params.nside, lmax=self.params.lmax,
                new=True, verbose=False
            )
        else:
            noise_map = np.zeros(npix)

        total_map = cmb_map + noise_map

        if return_components:
            return total_map, cmb_map, noise_map
        return total_map

    def get_temperatures_at_positions(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        T_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract temperatures at given sky positions.

        Parameters
        ----------
        ra : np.ndarray
            Right ascension in degrees
        dec : np.ndarray
            Declination in degrees
        T_map : np.ndarray, optional
            Temperature map (generates new realization if None)

        Returns
        -------
        T : np.ndarray
            Temperatures at positions (μK)
        """
        if T_map is None:
            T_map = self.generate_realization()

        # Convert to theta, phi
        theta = np.radians(90 - dec)
        phi = np.radians(ra)

        # Get pixel indices
        pix = hp.ang2pix(self.params.nside, theta, phi, nest=False)

        return T_map[pix]


def generate_cmb_realization(
    nside: int = 2048,
    power_spectrum: Optional[np.ndarray] = None,
    beam_fwhm_arcmin: float = 1.4,
    noise_level_uk_arcmin: float = 10.0,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a single CMB + noise realization.

    Convenience function for simple use cases.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE parameter
    power_spectrum : np.ndarray, optional
        CMB C_ell (uses default if None)
    beam_fwhm_arcmin : float
        Beam FWHM in arcmin
    noise_level_uk_arcmin : float
        Noise level in μK·arcmin
    random_seed : int, optional
        Random seed

    Returns
    -------
    T_map : np.ndarray
        Temperature map (HEALPix RING ordering)
    """
    params = CMBRealizationParams(
        nside=nside,
        beam_fwhm_arcmin=beam_fwhm_arcmin,
        noise_level_uk_arcmin=noise_level_uk_arcmin,
        random_seed=random_seed,
    )

    sim = CMBSimulator(power_spectrum=power_spectrum, params=params)
    return sim.generate_realization()


def generate_batch_realizations(
    n_realizations: int,
    nside: int = 2048,
    power_spectrum: Optional[np.ndarray] = None,
    beam_fwhm_arcmin: float = 1.4,
    noise_level_uk_arcmin: float = 10.0,
    base_seed: int = 42,
    output_func=None,
) -> List[np.ndarray]:
    """
    Generate multiple CMB realizations.

    Parameters
    ----------
    n_realizations : int
        Number of realizations to generate
    nside : int
        HEALPix NSIDE
    power_spectrum : np.ndarray, optional
        CMB C_ell
    beam_fwhm_arcmin : float
        Beam FWHM
    noise_level_uk_arcmin : float
        Noise level
    base_seed : int
        Base random seed
    output_func : callable, optional
        Function to call with each realization (for streaming)

    Returns
    -------
    maps : list of np.ndarray
        Generated maps (or empty list if output_func used)
    """
    params = CMBRealizationParams(
        nside=nside,
        beam_fwhm_arcmin=beam_fwhm_arcmin,
        noise_level_uk_arcmin=noise_level_uk_arcmin,
    )

    sim = CMBSimulator(power_spectrum=power_spectrum, params=params)

    maps = []
    for i in range(n_realizations):
        seed = base_seed + i
        T_map = sim.generate_realization(seed=seed)

        if output_func is not None:
            output_func(i, T_map)
        else:
            maps.append(T_map)

        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1}/{n_realizations} realizations")

    return maps
