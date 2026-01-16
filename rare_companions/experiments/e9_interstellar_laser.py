"""
EXPERIMENT E9: THE "INTERSTELLAR LASER" SEARCH (SKY FIBER ANOMALIES)

Technosignature search using DESI "Sky Fibers" - fibers calibrated to point at
empty space to measure sky glow.

We search for Monochromatic Optical Pulses (Lasers) that appear in a single fiber
but are not present in the background sky spectrum of neighboring fibers.

HYPOTHESIS:
A sufficiently advanced civilization using high-powered lasers for communication
or propulsion will create a spectral signature that:
- Is spatially unresolved (appears in 1 fiber, not the whole camera)
- Is spectrally unresolved (line width ≈ instrument resolution)
- Is non-terrestrial (does not match airglow lines like [OI] 5577, OH, Sodium)

Author: Claude Code (E9 Experiment)
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
import urllib.request
import urllib.error
import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime
import warnings
import csv

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# =============================================================================
# KNOWN SKY LINE DATABASE (The "Anti-Airglow Shield")
# =============================================================================

# Known atmospheric emission lines to reject (wavelength in Angstroms)
AIRGLOW_LINES = {
    # Oxygen forbidden lines
    'OI_5577': 5577.3,
    'OI_6300': 6300.3,
    'OI_6364': 6363.8,

    # Sodium D lines
    'NaD1': 5895.9,
    'NaD2': 5889.9,

    # Mercury (from streetlights/light pollution)
    'Hg_4047': 4046.6,
    'Hg_4358': 4358.3,
    'Hg_5461': 5460.7,
    'Hg_5769': 5769.6,
    'Hg_5790': 5790.7,

    # OH Meinel bands (major bands in 7000-9000 A range)
    'OH_6834': 6834.0,
    'OH_6871': 6871.0,
    'OH_6912': 6912.0,
    'OH_6950': 6950.0,
    'OH_6989': 6989.0,
    'OH_7244': 7244.0,
    'OH_7276': 7276.0,
    'OH_7316': 7316.0,
    'OH_7340': 7340.0,
    'OH_7369': 7369.0,
    'OH_7524': 7524.0,
    'OH_7556': 7556.0,
    'OH_7586': 7586.0,
    'OH_7621': 7621.0,
    'OH_7651': 7651.0,
    'OH_7750': 7750.0,
    'OH_7794': 7794.0,
    'OH_7821': 7821.0,
    'OH_7853': 7853.0,
    'OH_7914': 7914.0,
    'OH_7964': 7964.0,
    'OH_7993': 7993.0,
    'OH_8025': 8025.0,
    'OH_8063': 8063.0,
    'OH_8101': 8101.0,
    'OH_8264': 8264.0,
    'OH_8298': 8298.0,
    'OH_8344': 8344.0,
    'OH_8382': 8382.0,
    'OH_8430': 8430.0,
    'OH_8505': 8505.0,
    'OH_8627': 8627.0,
    'OH_8696': 8696.0,
    'OH_8761': 8761.0,
    'OH_8827': 8827.0,
    'OH_8886': 8886.0,
    'OH_8943': 8943.0,

    # Telluric absorption band edges (can cause artifacts)
    'H2O_A_band_start': 7160.0,
    'H2O_A_band_end': 7320.0,
    'H2O_B_band_start': 8120.0,
    'H2O_B_band_end': 8350.0,
    'O2_A_band': 7620.0,
    'O2_B_band': 6870.0,
}

# Generate list of wavelengths to reject (within ± REJECTION_WINDOW Angstroms)
REJECTION_WINDOW = 2.0  # Angstroms

def get_rejection_wavelengths() -> np.ndarray:
    """Get sorted array of airglow line wavelengths."""
    return np.array(sorted(AIRGLOW_LINES.values()))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LaserCandidate:
    """A candidate laser detection from a sky fiber."""
    fiber_id: int
    targetid: int
    ra: float
    dec: float
    wavelength_peak: float  # Angstroms
    flux_peak: float
    snr: float
    fwhm_pixels: float
    mjd: float
    exposure_id: int
    tile_id: int
    petal: int

    # Metadata
    n_neighboring_detections: int
    median_sky_at_wavelength: float
    residual_significance: float

    # Flags
    is_near_airglow: bool
    is_spatially_isolated: bool
    passed_all_cuts: bool
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'FIBER_ID': self.fiber_id,
            'TARGETID': self.targetid,
            'RA': self.ra,
            'DEC': self.dec,
            'WAVELENGTH_PEAK': self.wavelength_peak,
            'FLUX_PEAK': self.flux_peak,
            'SNR': self.snr,
            'FWHM_PIXELS': self.fwhm_pixels,
            'MJD': self.mjd,
            'EXPOSURE_ID': self.exposure_id,
            'TILE_ID': self.tile_id,
            'PETAL': self.petal,
            'N_NEIGHBORS': self.n_neighboring_detections,
            'MEDIAN_SKY': self.median_sky_at_wavelength,
            'RESIDUAL_SIG': self.residual_significance,
            'NEAR_AIRGLOW': self.is_near_airglow,
            'SPATIALLY_ISOLATED': self.is_spatially_isolated,
            'PASSED_ALL_CUTS': self.passed_all_cuts,
            'REJECTION_REASONS': ';'.join(self.rejection_reasons)
        }


@dataclass
class ExposureData:
    """Container for a single exposure's sky fiber data."""
    exposure_id: int
    tile_id: int
    mjd: float
    wavelength: np.ndarray  # Common wavelength grid
    sky_flux: np.ndarray  # Shape: (n_fibers, n_wavelength)
    sky_ivar: np.ndarray  # Shape: (n_fibers, n_wavelength)
    fiber_ids: np.ndarray
    fiber_ra: np.ndarray
    fiber_dec: np.ndarray
    fiber_x: np.ndarray  # Focal plane X
    fiber_y: np.ndarray  # Focal plane Y
    fiber_petal: np.ndarray
    targetids: np.ndarray


# =============================================================================
# DESI DATA ACCESS
# =============================================================================

class DESISkyFiberLoader:
    """Load DESI sky fiber spectra from DR1."""

    DESI_BASE_URL = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"

    def __init__(self, cache_dir: str = "data/e9_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_tile_list(self, max_tiles: int = None) -> List[int]:
        """Get list of available tile IDs from DR1."""
        # For DR1, we can use a manifest or scan the directory
        # Here we use a known subset of tiles for efficiency

        # DR1 has thousands of tiles - we'll start with a sample
        # These are known good tiles from the main survey
        sample_tiles = [
            80605, 80606, 80607, 80608, 80609, 80610,
            80611, 80612, 80613, 80614, 80615, 80616,
            80617, 80618, 80619, 80620, 80621, 80622,
            80623, 80624, 80625, 80626, 80627, 80628,
            80629, 80630, 80631, 80632, 80633, 80634,
            80635, 80636, 80637, 80638, 80639, 80640,
            1000, 1001, 1002, 1003, 1004, 1005,
            2000, 2001, 2002, 2003, 2004, 2005,
            3000, 3001, 3002, 3003, 3004, 3005,
        ]

        if max_tiles:
            return sample_tiles[:max_tiles]
        return sample_tiles

    def download_cframe(self, tile_id: int, night: str, expid: int,
                        camera: str = 'b0') -> Optional[Path]:
        """
        Download a cframe file from DESI DR1.

        cframe files contain calibrated spectra with sky subtraction info.
        """
        # Construct URL for cumulative spectra
        # Format: tiles/cumulative/{TILEID}/{NIGHT}/cframe-{CAMERA}-{EXPID}.fits
        filename = f"cframe-{camera}-{expid:08d}.fits"
        url = f"{self.DESI_BASE_URL}/tiles/cumulative/{tile_id}/{night}/{filename}"

        local_path = self.cache_dir / f"tile_{tile_id}" / night / filename

        if local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.debug(f"Downloading {url}")
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except urllib.error.HTTPError as e:
            logger.debug(f"HTTP error downloading {url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error downloading {url}: {e}")
            return None

    def load_spectra_file(self, filepath: Path) -> Optional[Dict]:
        """
        Load a DESI spectra file and extract sky fiber data.

        Works with coadd or cframe files.
        """
        try:
            with fits.open(filepath) as hdu:
                # Check available extensions
                ext_names = [h.name for h in hdu]

                # DESI spectra files have:
                # - FIBERMAP: fiber metadata
                # - B_WAVELENGTH, R_WAVELENGTH, Z_WAVELENGTH: wavelength grids
                # - B_FLUX, R_FLUX, Z_FLUX: flux arrays
                # - B_IVAR, R_IVAR, Z_IVAR: inverse variance

                if 'FIBERMAP' not in ext_names:
                    logger.debug(f"No FIBERMAP in {filepath}")
                    return None

                fibermap = hdu['FIBERMAP'].data

                # Filter for SKY fibers
                if 'OBJTYPE' in fibermap.dtype.names:
                    sky_mask = fibermap['OBJTYPE'] == 'SKY'
                elif 'TARGETID' in fibermap.dtype.names:
                    # For some files, use negative TARGETID for sky
                    sky_mask = fibermap['TARGETID'] < 0
                else:
                    # Assume all are sky if no OBJTYPE
                    sky_mask = np.ones(len(fibermap), dtype=bool)

                n_sky = np.sum(sky_mask)
                if n_sky == 0:
                    logger.debug(f"No sky fibers in {filepath}")
                    return None

                logger.debug(f"Found {n_sky} sky fibers in {filepath}")

                # Get wavelength and flux data
                # Try different possible extension names
                data = {
                    'fibermap': fibermap[sky_mask],
                    'sky_mask': sky_mask,
                    'filepath': filepath
                }

                # Load spectra for each arm (B, R, Z)
                for arm in ['B', 'R', 'Z']:
                    wave_key = f'{arm}_WAVELENGTH'
                    flux_key = f'{arm}_FLUX'
                    ivar_key = f'{arm}_IVAR'

                    if wave_key in ext_names:
                        data[f'{arm.lower()}_wave'] = hdu[wave_key].data
                        data[f'{arm.lower()}_flux'] = hdu[flux_key].data[sky_mask]
                        data[f'{arm.lower()}_ivar'] = hdu[ivar_key].data[sky_mask]

                # Also try single-arm format
                if 'WAVELENGTH' in ext_names:
                    data['wave'] = hdu['WAVELENGTH'].data
                    data['flux'] = hdu['FLUX'].data[sky_mask]
                    data['ivar'] = hdu['IVAR'].data[sky_mask]

                return data

        except Exception as e:
            logger.debug(f"Error loading {filepath}: {e}")
            return None

    def load_healpix_spectra(self, healpix: int, survey: str = 'main',
                             program: str = 'bright') -> Optional[Dict]:
        """
        Load spectra from healpix-organized coadd files.

        This is the primary organization for DR1 spectra.
        """
        # Format: healpix/{SURVEY}/{PROGRAM}/{PIX//100}/{PIX}/coadd-{SURVEY}-{PROGRAM}-{PIX}.fits
        pix_dir = healpix // 100
        filename = f"coadd-{survey}-{program}-{healpix}.fits"
        url = f"{self.DESI_BASE_URL}/healpix/{survey}/{program}/{pix_dir}/{healpix}/{filename}"

        local_path = self.cache_dir / "healpix" / survey / program / str(pix_dir) / str(healpix) / filename

        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                logger.info(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                logger.debug(f"Could not download {url}: {e}")
                return None

        return self.load_spectra_file(local_path)


# =============================================================================
# THE DIFFERENCE ENGINE
# =============================================================================

class DifferenceEngine:
    """
    The core algorithm for detecting anomalies in sky fiber spectra.

    For each exposure:
    1. Calculate median sky spectrum across all sky fibers
    2. Subtract median from each individual fiber
    3. Analyze residuals for anomalies
    """

    def __init__(self, sigma_threshold: float = 10.0,
                 max_fwhm_pixels: float = 3.0,
                 rejection_window: float = 2.0):
        self.sigma_threshold = sigma_threshold
        self.max_fwhm_pixels = max_fwhm_pixels
        self.rejection_window = rejection_window
        self.airglow_wavelengths = get_rejection_wavelengths()

    def compute_median_sky(self, flux: np.ndarray, ivar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute median sky spectrum across fibers.

        Parameters
        ----------
        flux : array (n_fibers, n_wavelength)
            Flux values
        ivar : array (n_fibers, n_wavelength)
            Inverse variance values

        Returns
        -------
        median_flux : array (n_wavelength,)
            Median sky spectrum
        mad : array (n_wavelength,)
            Median absolute deviation (robust sigma)
        """
        # Use weighted median where possible
        with np.errstate(invalid='ignore', divide='ignore'):
            # Simple median for robustness
            median_flux = np.nanmedian(flux, axis=0)

            # MAD for robust sigma estimate
            mad = np.nanmedian(np.abs(flux - median_flux), axis=0) * 1.4826

        return median_flux, mad

    def compute_residuals(self, flux: np.ndarray, ivar: np.ndarray,
                          median_flux: np.ndarray, mad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute residual spectra and their significance.

        Parameters
        ----------
        flux : array (n_fibers, n_wavelength)
        ivar : array (n_fibers, n_wavelength)
        median_flux : array (n_wavelength,)
        mad : array (n_wavelength,)

        Returns
        -------
        residuals : array (n_fibers, n_wavelength)
            Residual flux (flux - median)
        significance : array (n_fibers, n_wavelength)
            Residual / uncertainty in sigma units
        """
        residuals = flux - median_flux

        # Combine MAD with individual fiber variance
        with np.errstate(invalid='ignore', divide='ignore'):
            fiber_sigma = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf)
            total_sigma = np.sqrt(mad**2 + fiber_sigma**2)
            significance = residuals / total_sigma

        return residuals, significance

    def find_peaks(self, significance: np.ndarray, wavelength: np.ndarray,
                   fiber_idx: int) -> List[Dict]:
        """
        Find significant peaks in a single fiber's residual spectrum.

        Parameters
        ----------
        significance : array (n_wavelength,)
            Significance in sigma units
        wavelength : array (n_wavelength,)
            Wavelength grid
        fiber_idx : int
            Fiber index for bookkeeping

        Returns
        -------
        peaks : list of dict
            Peak information for each detected peak
        """
        peaks = []

        # Mask bad data
        good_mask = np.isfinite(significance) & (significance > 0)
        if np.sum(good_mask) < 10:
            return peaks

        # Find pixels above threshold
        above_thresh = significance > self.sigma_threshold

        if not np.any(above_thresh):
            return peaks

        # Find contiguous regions above threshold
        diff = np.diff(above_thresh.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if above_thresh[0]:
            starts = np.concatenate([[0], starts])
        if above_thresh[-1]:
            ends = np.concatenate([ends, [len(above_thresh)]])

        for start, end in zip(starts, ends):
            region = significance[start:end]
            wave_region = wavelength[start:end]

            if len(region) == 0:
                continue

            # Find peak within region
            peak_idx = np.argmax(region)
            peak_wave = wave_region[peak_idx]
            peak_sig = region[peak_idx]

            # Estimate FWHM (in pixels)
            fwhm_pixels = end - start

            # Only keep narrow peaks (laser-like)
            if fwhm_pixels <= self.max_fwhm_pixels:
                peaks.append({
                    'wavelength': float(peak_wave),
                    'significance': float(peak_sig),
                    'fwhm_pixels': float(fwhm_pixels),
                    'fiber_idx': fiber_idx,
                    'start_idx': start,
                    'end_idx': end
                })

        return peaks

    def is_near_airglow(self, wavelength: float) -> Tuple[bool, str]:
        """Check if wavelength is near a known airglow line."""
        for name, line_wave in AIRGLOW_LINES.items():
            if abs(wavelength - line_wave) < self.rejection_window:
                return True, name
        return False, ""

    def process_exposure(self, wavelength: np.ndarray, flux: np.ndarray,
                         ivar: np.ndarray, fibermap: np.ndarray,
                         exposure_info: Dict) -> List[LaserCandidate]:
        """
        Process a single exposure's worth of sky fiber data.

        This is THE DIFFERENCE ENGINE.

        Parameters
        ----------
        wavelength : array (n_wavelength,)
            Common wavelength grid
        flux : array (n_fibers, n_wavelength)
            Sky fiber flux
        ivar : array (n_fibers, n_wavelength)
            Inverse variance
        fibermap : structured array
            Fiber metadata
        exposure_info : dict
            Exposure metadata (MJD, exposure ID, etc.)

        Returns
        -------
        candidates : list of LaserCandidate
            Detected laser candidates
        """
        n_fibers, n_wave = flux.shape

        if n_fibers < 3:
            logger.debug("Too few fibers for difference engine")
            return []

        # Step 1: Compute median sky spectrum
        median_flux, mad = self.compute_median_sky(flux, ivar)

        # Step 2: Compute residuals and significance
        residuals, significance = self.compute_residuals(flux, ivar, median_flux, mad)

        # Step 3: Find peaks in each fiber
        all_peaks = []
        for i in range(n_fibers):
            peaks = self.find_peaks(significance[i], wavelength, i)
            all_peaks.extend(peaks)

        if not all_peaks:
            return []

        logger.debug(f"Found {len(all_peaks)} initial peaks in exposure")

        # Step 4: Apply filters and create candidates
        candidates = []

        # Group peaks by wavelength for spatial isolation check
        for peak in all_peaks:
            fiber_idx = peak['fiber_idx']
            wave = peak['wavelength']

            # Get fiber info
            fiber_info = fibermap[fiber_idx]

            # Check for airglow contamination
            is_airglow, airglow_name = self.is_near_airglow(wave)

            # Count neighboring detections at similar wavelength
            n_neighbors = sum(
                1 for p in all_peaks
                if p['fiber_idx'] != fiber_idx
                and abs(p['wavelength'] - wave) < 5.0  # Within 5 Angstroms
            )

            # Spatial isolation: signal in only 1 fiber
            is_isolated = n_neighbors <= 1

            # Build rejection reasons
            rejection_reasons = []
            if is_airglow:
                rejection_reasons.append(f"Near airglow line {airglow_name}")
            if not is_isolated:
                rejection_reasons.append(f"Signal in {n_neighbors + 1} fibers (not isolated)")

            passed_all = len(rejection_reasons) == 0

            # Extract fiber coordinates
            ra = float(fiber_info['TARGET_RA']) if 'TARGET_RA' in fiber_info.dtype.names else 0.0
            dec = float(fiber_info['TARGET_DEC']) if 'TARGET_DEC' in fiber_info.dtype.names else 0.0
            targetid = int(fiber_info['TARGETID']) if 'TARGETID' in fiber_info.dtype.names else -1
            fiber_id = int(fiber_info['FIBER']) if 'FIBER' in fiber_info.dtype.names else fiber_idx
            petal = int(fiber_info['PETAL_LOC']) if 'PETAL_LOC' in fiber_info.dtype.names else 0

            # Get flux at peak
            wave_idx = np.argmin(np.abs(wavelength - wave))
            flux_peak = float(flux[fiber_idx, wave_idx])
            median_at_wave = float(median_flux[wave_idx])

            candidate = LaserCandidate(
                fiber_id=fiber_id,
                targetid=targetid,
                ra=ra,
                dec=dec,
                wavelength_peak=wave,
                flux_peak=flux_peak,
                snr=peak['significance'],
                fwhm_pixels=peak['fwhm_pixels'],
                mjd=exposure_info.get('mjd', 0.0),
                exposure_id=exposure_info.get('expid', 0),
                tile_id=exposure_info.get('tileid', 0),
                petal=petal,
                n_neighboring_detections=n_neighbors,
                median_sky_at_wavelength=median_at_wave,
                residual_significance=peak['significance'],
                is_near_airglow=is_airglow,
                is_spatially_isolated=is_isolated,
                passed_all_cuts=passed_all,
                rejection_reasons=rejection_reasons
            )

            candidates.append(candidate)

        return candidates


# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

def process_single_file(args: Tuple[Path, Dict]) -> List[LaserCandidate]:
    """
    Process a single spectra file for laser candidates.

    Worker function for parallel processing.
    """
    filepath, config = args

    try:
        loader = DESISkyFiberLoader(cache_dir=config['cache_dir'])
        engine = DifferenceEngine(
            sigma_threshold=config['sigma_threshold'],
            max_fwhm_pixels=config['max_fwhm_pixels'],
            rejection_window=config['rejection_window']
        )

        data = loader.load_spectra_file(filepath)
        if data is None:
            return []

        candidates = []

        # Process each spectral arm
        for arm in ['b', 'r', 'z', '']:
            wave_key = f'{arm}_wave' if arm else 'wave'
            flux_key = f'{arm}_flux' if arm else 'flux'
            ivar_key = f'{arm}_ivar' if arm else 'ivar'

            if wave_key not in data:
                continue

            exposure_info = {
                'mjd': data['fibermap']['MJD'][0] if 'MJD' in data['fibermap'].dtype.names else 0,
                'expid': 0,
                'tileid': 0
            }

            arm_candidates = engine.process_exposure(
                wavelength=data[wave_key],
                flux=data[flux_key],
                ivar=data[ivar_key],
                fibermap=data['fibermap'],
                exposure_info=exposure_info
            )

            candidates.extend(arm_candidates)

        return candidates

    except Exception as e:
        logger.debug(f"Error processing {filepath}: {e}")
        return []


class LaserSearchPipeline:
    """
    Main pipeline for the E9 Interstellar Laser Search.

    Orchestrates parallel processing of DESI sky fiber spectra.
    """

    def __init__(self, output_dir: str = "data/e9_results",
                 cache_dir: str = "data/e9_cache",
                 sigma_threshold: float = 10.0,
                 max_fwhm_pixels: float = 3.0,
                 rejection_window: float = 2.0,
                 n_workers: int = None):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.sigma_threshold = sigma_threshold
        self.max_fwhm_pixels = max_fwhm_pixels
        self.rejection_window = rejection_window

        self.n_workers = n_workers or max(1, cpu_count() - 1)

        self.loader = DESISkyFiberLoader(cache_dir=str(cache_dir))
        self.engine = DifferenceEngine(
            sigma_threshold=sigma_threshold,
            max_fwhm_pixels=max_fwhm_pixels,
            rejection_window=rejection_window
        )

        self.all_candidates: List[LaserCandidate] = []
        self.stats = {
            'files_processed': 0,
            'fibers_processed': 0,
            'initial_detections': 0,
            'passed_all_cuts': 0,
            'rejected_airglow': 0,
            'rejected_spatial': 0
        }

    def run_healpix_search(self, healpix_list: List[int],
                          survey: str = 'main',
                          program: str = 'bright') -> List[LaserCandidate]:
        """
        Run laser search on a list of healpix pixels.

        Parameters
        ----------
        healpix_list : list of int
            Healpix pixel IDs to process
        survey : str
            DESI survey (main, sv1, sv2, sv3)
        program : str
            Program (bright, dark)

        Returns
        -------
        candidates : list of LaserCandidate
            All detected candidates
        """
        logger.info(f"Starting E9 Laser Search on {len(healpix_list)} healpix pixels")
        logger.info(f"Using {self.n_workers} parallel workers")
        logger.info(f"Detection threshold: {self.sigma_threshold} sigma")

        start_time = datetime.now()

        # Download all files first
        file_paths = []
        for hpx in healpix_list:
            data = self.loader.load_healpix_spectra(hpx, survey, program)
            if data is not None:
                file_paths.append(data['filepath'])

        logger.info(f"Found {len(file_paths)} valid files to process")

        if not file_paths:
            logger.warning("No valid files found!")
            return []

        # Prepare config for workers
        config = {
            'cache_dir': str(self.cache_dir),
            'sigma_threshold': self.sigma_threshold,
            'max_fwhm_pixels': self.max_fwhm_pixels,
            'rejection_window': self.rejection_window
        }

        # Process in parallel
        all_candidates = []

        if self.n_workers > 1 and len(file_paths) > 1:
            # Parallel processing
            args_list = [(fp, config) for fp in file_paths]

            with Pool(processes=self.n_workers) as pool:
                results = pool.map(process_single_file, args_list)

            for result in results:
                all_candidates.extend(result)
                self.stats['files_processed'] += 1
        else:
            # Sequential for debugging
            for filepath in file_paths:
                candidates = process_single_file((filepath, config))
                all_candidates.extend(candidates)
                self.stats['files_processed'] += 1

        # Update statistics
        self.stats['initial_detections'] = len(all_candidates)
        self.stats['passed_all_cuts'] = sum(1 for c in all_candidates if c.passed_all_cuts)
        self.stats['rejected_airglow'] = sum(1 for c in all_candidates if c.is_near_airglow)
        self.stats['rejected_spatial'] = sum(1 for c in all_candidates if not c.is_spatially_isolated)

        self.all_candidates = all_candidates

        runtime = (datetime.now() - start_time).total_seconds()

        logger.info(f"E9 Search Complete in {runtime:.1f} seconds")
        logger.info(f"  Files processed: {self.stats['files_processed']}")
        logger.info(f"  Initial detections: {self.stats['initial_detections']}")
        logger.info(f"  Passed all cuts: {self.stats['passed_all_cuts']}")
        logger.info(f"  Rejected (airglow): {self.stats['rejected_airglow']}")
        logger.info(f"  Rejected (not isolated): {self.stats['rejected_spatial']}")

        return all_candidates

    def run_from_local_files(self, file_list: List[Path]) -> List[LaserCandidate]:
        """
        Run laser search on a list of local spectra files.

        Parameters
        ----------
        file_list : list of Path
            Paths to DESI spectra files

        Returns
        -------
        candidates : list of LaserCandidate
        """
        logger.info(f"Processing {len(file_list)} local files with {self.n_workers} workers")

        start_time = datetime.now()

        config = {
            'cache_dir': str(self.cache_dir),
            'sigma_threshold': self.sigma_threshold,
            'max_fwhm_pixels': self.max_fwhm_pixels,
            'rejection_window': self.rejection_window
        }

        all_candidates = []

        if self.n_workers > 1 and len(file_list) > 1:
            args_list = [(fp, config) for fp in file_list]
            with Pool(processes=self.n_workers) as pool:
                results = pool.map(process_single_file, args_list)
            for result in results:
                all_candidates.extend(result)
                self.stats['files_processed'] += 1
        else:
            for filepath in file_list:
                candidates = process_single_file((filepath, config))
                all_candidates.extend(candidates)
                self.stats['files_processed'] += 1
                if self.stats['files_processed'] % 10 == 0:
                    logger.info(f"  Processed {self.stats['files_processed']}/{len(file_list)} files")

        self.stats['initial_detections'] = len(all_candidates)
        self.stats['passed_all_cuts'] = sum(1 for c in all_candidates if c.passed_all_cuts)
        self.stats['rejected_airglow'] = sum(1 for c in all_candidates if c.is_near_airglow)
        self.stats['rejected_spatial'] = sum(1 for c in all_candidates if not c.is_spatially_isolated)

        self.all_candidates = all_candidates

        runtime = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processing complete in {runtime:.1f}s")

        return all_candidates

    def save_results(self, filename: str = "laser_candidates.csv"):
        """Save all candidates to CSV."""
        output_path = self.output_dir / filename

        # Filter to candidates that passed all cuts (or save all for analysis)
        passed = [c for c in self.all_candidates if c.passed_all_cuts]

        logger.info(f"Saving {len(passed)} candidates to {output_path}")

        if not passed:
            logger.warning("No candidates passed all cuts!")
            # Save all candidates for debugging
            passed = self.all_candidates
            output_path = self.output_dir / "laser_candidates_all.csv"

        if not passed:
            logger.warning("No candidates at all!")
            return

        # Write CSV
        fieldnames = list(passed[0].to_dict().keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in passed:
                writer.writerow(candidate.to_dict())

        logger.info(f"Results saved to {output_path}")

        # Also save statistics
        stats_path = self.output_dir / "search_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Statistics saved to {stats_path}")

    def get_top_candidates(self, n: int = 10) -> List[LaserCandidate]:
        """Get top N candidates by SNR that passed all cuts."""
        passed = [c for c in self.all_candidates if c.passed_all_cuts]
        passed.sort(key=lambda c: c.snr, reverse=True)
        return passed[:n]


# =============================================================================
# SYNTHETIC DATA FOR TESTING
# =============================================================================

def generate_synthetic_exposure(n_fibers: int = 100, n_wavelength: int = 4000,
                               inject_laser: bool = True,
                               laser_wavelength: float = 6500.0,
                               laser_snr: float = 15.0) -> Dict:
    """
    Generate synthetic sky fiber data for testing.

    Parameters
    ----------
    n_fibers : int
        Number of sky fibers
    n_wavelength : int
        Number of wavelength bins
    inject_laser : bool
        Whether to inject a fake laser signal
    laser_wavelength : float
        Wavelength of injected laser (Angstroms)
    laser_snr : float
        Signal-to-noise ratio of injected laser

    Returns
    -------
    dict
        Synthetic data in same format as real data
    """
    # Create wavelength grid (typical DESI range)
    wavelength = np.linspace(3600, 9800, n_wavelength)

    # Generate sky continuum with some features
    sky_continuum = 100 + 50 * np.exp(-((wavelength - 5577) / 100)**2)  # OI glow
    sky_continuum += 30 * np.exp(-((wavelength - 6300) / 50)**2)  # OI red

    # Add OH forest in red
    for w in np.arange(7000, 9000, 100):
        sky_continuum += 20 * np.exp(-((wavelength - w) / 10)**2)

    # Generate individual fiber spectra with noise
    np.random.seed(42)
    noise_level = 10.0

    flux = np.zeros((n_fibers, n_wavelength))
    ivar = np.zeros((n_fibers, n_wavelength))

    for i in range(n_fibers):
        # Add Poisson-like noise
        fiber_noise = np.random.normal(0, noise_level, n_wavelength)
        flux[i] = sky_continuum + fiber_noise
        ivar[i] = 1.0 / (noise_level**2 + 0.1 * np.abs(flux[i]))

    # Inject laser signal in ONE fiber
    if inject_laser:
        laser_fiber = 42  # Answer to everything
        laser_idx = np.argmin(np.abs(wavelength - laser_wavelength))

        # Inject narrow spike (1-2 pixels wide)
        laser_amplitude = laser_snr * noise_level
        flux[laser_fiber, laser_idx] += laser_amplitude
        flux[laser_fiber, laser_idx + 1] += laser_amplitude * 0.3  # Slight wing

        logger.info(f"Injected laser at λ={laser_wavelength}Å in fiber {laser_fiber} with SNR={laser_snr}")

    # Create fibermap
    fibermap = np.zeros(n_fibers, dtype=[
        ('FIBER', 'i4'),
        ('TARGETID', 'i8'),
        ('TARGET_RA', 'f8'),
        ('TARGET_DEC', 'f8'),
        ('PETAL_LOC', 'i4'),
        ('MJD', 'f8'),
        ('OBJTYPE', 'U10')
    ])

    for i in range(n_fibers):
        fibermap[i]['FIBER'] = i
        fibermap[i]['TARGETID'] = -1000 - i  # Negative = sky
        fibermap[i]['TARGET_RA'] = 180.0 + np.random.uniform(-1, 1)
        fibermap[i]['TARGET_DEC'] = 30.0 + np.random.uniform(-1, 1)
        fibermap[i]['PETAL_LOC'] = i // 10
        fibermap[i]['MJD'] = 59000 + np.random.uniform(0, 100)
        fibermap[i]['OBJTYPE'] = 'SKY'

    return {
        'wave': wavelength,
        'flux': flux,
        'ivar': ivar,
        'fibermap': fibermap,
        'injected_laser': inject_laser,
        'laser_wavelength': laser_wavelength if inject_laser else None,
        'laser_fiber': 42 if inject_laser else None
    }


def run_smoke_test():
    """
    Run a smoke test with synthetic data to verify the pipeline works.
    """
    print("=" * 70)
    print("E9 INTERSTELLAR LASER SEARCH - SMOKE TEST")
    print("=" * 70)

    # Generate synthetic data with an injected laser
    print("\n[1] Generating synthetic sky fiber data...")
    synthetic_data = generate_synthetic_exposure(
        n_fibers=100,
        n_wavelength=4000,
        inject_laser=True,
        laser_wavelength=6500.0,  # Away from airglow
        laser_snr=15.0
    )

    print(f"    Generated {synthetic_data['flux'].shape[0]} fibers x {synthetic_data['flux'].shape[1]} wavelengths")

    # Initialize the Difference Engine
    print("\n[2] Initializing Difference Engine...")
    engine = DifferenceEngine(
        sigma_threshold=10.0,
        max_fwhm_pixels=3.0,
        rejection_window=2.0
    )

    # Run detection
    print("\n[3] Running laser detection...")
    exposure_info = {'mjd': 59000, 'expid': 1, 'tileid': 0}

    candidates = engine.process_exposure(
        wavelength=synthetic_data['wave'],
        flux=synthetic_data['flux'],
        ivar=synthetic_data['ivar'],
        fibermap=synthetic_data['fibermap'],
        exposure_info=exposure_info
    )

    print(f"    Found {len(candidates)} candidates")

    # Check results
    print("\n[4] Results:")

    passed = [c for c in candidates if c.passed_all_cuts]
    print(f"    Candidates passing all cuts: {len(passed)}")

    if passed:
        print("\n    TOP CANDIDATE:")
        top = max(passed, key=lambda c: c.snr)
        print(f"      Wavelength: {top.wavelength_peak:.2f} Å")
        print(f"      SNR: {top.snr:.1f}")
        print(f"      FWHM: {top.fwhm_pixels:.1f} pixels")
        print(f"      Fiber: {top.fiber_id}")
        print(f"      Spatially isolated: {top.is_spatially_isolated}")
        print(f"      Near airglow: {top.is_near_airglow}")

        # Check if we found the injected laser
        if synthetic_data['injected_laser']:
            expected_wave = synthetic_data['laser_wavelength']
            expected_fiber = synthetic_data['laser_fiber']

            found = any(
                abs(c.wavelength_peak - expected_wave) < 10 and c.fiber_id == expected_fiber
                for c in passed
            )

            if found:
                print("\n    ✓ SUCCESS: Detected injected laser signal!")
            else:
                print("\n    ✗ WARNING: Did not detect injected laser")
                print(f"      Expected: λ={expected_wave}Å in fiber {expected_fiber}")
    else:
        print("    No candidates passed all cuts")

        if candidates:
            print("\n    All candidates (for debugging):")
            for c in candidates[:5]:
                print(f"      λ={c.wavelength_peak:.1f}Å, SNR={c.snr:.1f}, reasons: {c.rejection_reasons}")

    # Test airglow rejection
    print("\n[5] Testing airglow rejection...")
    synthetic_airglow = generate_synthetic_exposure(
        n_fibers=100,
        n_wavelength=4000,
        inject_laser=True,
        laser_wavelength=5577.3,  # Right on OI airglow
        laser_snr=20.0
    )

    candidates_airglow = engine.process_exposure(
        wavelength=synthetic_airglow['wave'],
        flux=synthetic_airglow['flux'],
        ivar=synthetic_airglow['ivar'],
        fibermap=synthetic_airglow['fibermap'],
        exposure_info=exposure_info
    )

    passed_airglow = [c for c in candidates_airglow if c.passed_all_cuts]
    rejected_airglow = [c for c in candidates_airglow if c.is_near_airglow]

    print(f"    Signal injected at OI 5577Å")
    print(f"    Total detections: {len(candidates_airglow)}")
    print(f"    Rejected as airglow: {len(rejected_airglow)}")
    print(f"    Passed all cuts: {len(passed_airglow)}")

    if rejected_airglow and not passed_airglow:
        print("    ✓ SUCCESS: Airglow filter working correctly!")
    else:
        print("    ✗ WARNING: Airglow filter may not be working")

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)

    return len(passed) > 0


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main entry point for E9 Interstellar Laser Search.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="E9: Interstellar Laser Search in DESI Sky Fibers"
    )
    parser.add_argument('--mode', choices=['smoke', 'search', 'local'],
                        default='smoke', help='Run mode')
    parser.add_argument('--n-healpix', type=int, default=10,
                        help='Number of healpix pixels to search')
    parser.add_argument('--sigma', type=float, default=10.0,
                        help='Detection threshold in sigma')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--output', type=str, default='data/e9_results',
                        help='Output directory')
    parser.add_argument('--local-files', type=str, nargs='+',
                        help='Local spectra files to process')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.mode == 'smoke':
        # Run smoke test
        success = run_smoke_test()
        return 0 if success else 1

    elif args.mode == 'search':
        # Run healpix search
        pipeline = LaserSearchPipeline(
            output_dir=args.output,
            sigma_threshold=args.sigma,
            n_workers=args.workers
        )

        # Get healpix list
        healpix_list = list(range(10000, 10000 + args.n_healpix))

        candidates = pipeline.run_healpix_search(healpix_list)
        pipeline.save_results()

        # Print top candidates
        top = pipeline.get_top_candidates(10)
        if top:
            print("\nTOP 10 LASER CANDIDATES:")
            print("-" * 80)
            for i, c in enumerate(top):
                print(f"{i+1}. λ={c.wavelength_peak:.2f}Å  SNR={c.snr:.1f}  "
                      f"RA={c.ra:.4f}  DEC={c.dec:.4f}  MJD={c.mjd:.2f}")

        return 0

    elif args.mode == 'local':
        # Process local files
        if not args.local_files:
            print("ERROR: --local-files required for local mode")
            return 1

        pipeline = LaserSearchPipeline(
            output_dir=args.output,
            sigma_threshold=args.sigma,
            n_workers=args.workers
        )

        file_list = [Path(f) for f in args.local_files]
        candidates = pipeline.run_from_local_files(file_list)
        pipeline.save_results()

        return 0


if __name__ == '__main__':
    exit(main())
