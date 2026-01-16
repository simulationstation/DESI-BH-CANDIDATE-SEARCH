"""
Legacy Survey imaging analysis for blend detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

# Try to import image processing
try:
    from astropy.io import fits
    from scipy import ndimage
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False


@dataclass
class LegacyResult:
    """Legacy Survey imaging analysis result."""
    has_data: bool
    bands_available: List[str]

    # PSF shape metrics
    ellipticity: float  # 0 = round
    asymmetry: float  # 0 = symmetric
    fwhm_arcsec: float

    # Source detection
    n_secondary_peaks: int
    secondary_peak_flux_ratio: float  # Brightest secondary / primary

    # Blend assessment
    blend_score: float  # 0-1, higher = more likely blend
    blend_flag: str  # 'clean', 'possible_blend', 'likely_blend'

    def to_dict(self) -> Dict:
        return {
            'has_data': self.has_data,
            'bands_available': self.bands_available,
            'ellipticity': self.ellipticity,
            'asymmetry': self.asymmetry,
            'fwhm_arcsec': self.fwhm_arcsec,
            'n_secondary_peaks': self.n_secondary_peaks,
            'secondary_peak_flux_ratio': self.secondary_peak_flux_ratio,
            'blend_score': self.blend_score,
            'blend_flag': self.blend_flag
        }


class LegacyAnalyzer:
    """
    Analyze Legacy Survey imaging for blend detection.
    """

    CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
    PIXEL_SCALE = 0.262  # arcsec/pixel

    def __init__(self, cache_dir: str = "cache/legacy"):
        self.cache_dir = cache_dir

    def analyze(self, ra: float, dec: float,
                size_arcsec: float = 30.0) -> LegacyResult:
        """
        Analyze Legacy Survey imaging for a target.

        Parameters
        ----------
        ra : float
            Right ascension (degrees)
        dec : float
            Declination (degrees)
        size_arcsec : float
            Cutout size (arcsec)

        Returns
        -------
        LegacyResult
            Analysis results
        """
        if not HAS_IMAGING:
            return self._empty_result()

        try:
            # Download cutouts
            size_pix = int(size_arcsec / self.PIXEL_SCALE)

            images = {}
            bands_available = []

            for band in ['g', 'r', 'z']:
                img = self._download_cutout(ra, dec, size_pix, band)
                if img is not None:
                    images[band] = img
                    bands_available.append(band)

            if not images:
                return self._empty_result()

            # Use r-band preferentially, else g or z
            if 'r' in images:
                primary_image = images['r']
            elif 'g' in images:
                primary_image = images['g']
            else:
                primary_image = images[bands_available[0]]

            # Analyze PSF shape
            ellipticity, asymmetry, fwhm = self._analyze_psf(primary_image)

            # Find secondary peaks
            n_secondary, flux_ratio = self._find_secondary_peaks(primary_image)

            # Compute blend score
            blend_score = self._compute_blend_score(
                ellipticity, asymmetry, n_secondary, flux_ratio
            )

            # Classify
            if blend_score < 0.3:
                blend_flag = 'clean'
            elif blend_score < 0.6:
                blend_flag = 'possible_blend'
            else:
                blend_flag = 'likely_blend'

            return LegacyResult(
                has_data=True,
                bands_available=bands_available,
                ellipticity=ellipticity,
                asymmetry=asymmetry,
                fwhm_arcsec=fwhm * self.PIXEL_SCALE,
                n_secondary_peaks=n_secondary,
                secondary_peak_flux_ratio=flux_ratio,
                blend_score=blend_score,
                blend_flag=blend_flag
            )

        except Exception as e:
            logger.warning(f"Legacy analysis failed: {e}")
            return self._empty_result()

    def _download_cutout(self, ra: float, dec: float,
                         size: int, band: str) -> Optional[np.ndarray]:
        """Download cutout from Legacy Survey."""
        try:
            params = {
                'ra': ra,
                'dec': dec,
                'size': size,
                'layer': 'ls-dr10',
                'pixscale': self.PIXEL_SCALE,
                'bands': band
            }

            response = requests.get(self.CUTOUT_URL, params=params, timeout=30)

            if response.status_code != 200:
                return None

            with fits.open(BytesIO(response.content)) as hdu:
                data = hdu[0].data
                if data is not None and data.ndim >= 2:
                    # Handle 3D arrays (multiple bands)
                    if data.ndim == 3:
                        data = data[0]
                    return data.astype(float)

            return None

        except Exception as e:
            logger.debug(f"Failed to download {band} cutout: {e}")
            return None

    def _analyze_psf(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Analyze PSF shape."""
        # Find centroid
        y, x = ndimage.center_of_mass(image)
        cy, cx = int(y), int(x)

        # Extract central region
        half = min(10, image.shape[0] // 4, image.shape[1] // 4)
        y0, y1 = max(0, cy - half), min(image.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(image.shape[1], cx + half)
        cutout = image[y0:y1, x0:x1]

        if cutout.size == 0:
            return 0.0, 0.0, 3.0

        # Compute moments
        total = np.sum(cutout)
        if total <= 0:
            return 0.0, 0.0, 3.0

        yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
        cy_local = np.sum(yy * cutout) / total
        cx_local = np.sum(xx * cutout) / total

        # Second moments
        m20 = np.sum((yy - cy_local)**2 * cutout) / total
        m02 = np.sum((xx - cx_local)**2 * cutout) / total
        m11 = np.sum((yy - cy_local) * (xx - cx_local) * cutout) / total

        # Ellipticity from moments
        e1 = (m20 - m02) / (m20 + m02) if (m20 + m02) > 0 else 0
        e2 = 2 * m11 / (m20 + m02) if (m20 + m02) > 0 else 0
        ellipticity = np.sqrt(e1**2 + e2**2)

        # FWHM estimate
        sigma = np.sqrt((m20 + m02) / 2)
        fwhm = 2.355 * sigma

        # Asymmetry: flip and compare
        flipped = np.flip(np.flip(cutout, 0), 1)
        diff = np.abs(cutout - flipped)
        asymmetry = np.sum(diff) / (2 * total) if total > 0 else 0

        return float(ellipticity), float(asymmetry), float(fwhm)

    def _find_secondary_peaks(self, image: np.ndarray,
                              threshold: float = 0.1) -> Tuple[int, float]:
        """Find secondary peaks in image."""
        # Smooth to reduce noise
        smoothed = ndimage.gaussian_filter(image, sigma=1.5)

        # Find local maxima
        max_filter = ndimage.maximum_filter(smoothed, size=5)
        peaks = (smoothed == max_filter) & (smoothed > threshold * np.max(smoothed))

        # Label peaks
        labeled, n_peaks = ndimage.label(peaks)

        if n_peaks <= 1:
            return 0, 0.0

        # Get peak fluxes
        peak_fluxes = []
        for i in range(1, n_peaks + 1):
            mask = labeled == i
            peak_fluxes.append(np.max(smoothed[mask]))

        peak_fluxes = sorted(peak_fluxes, reverse=True)

        # Number of secondary peaks
        n_secondary = n_peaks - 1

        # Flux ratio (second brightest / brightest)
        if len(peak_fluxes) >= 2:
            flux_ratio = peak_fluxes[1] / peak_fluxes[0]
        else:
            flux_ratio = 0.0

        return n_secondary, float(flux_ratio)

    def _compute_blend_score(self, ellipticity: float, asymmetry: float,
                             n_secondary: int, flux_ratio: float) -> float:
        """Compute overall blend score (0-1)."""
        score = 0.0

        # Ellipticity contribution
        if ellipticity > 0.1:
            score += min(0.3, ellipticity)

        # Asymmetry contribution
        if asymmetry > 0.1:
            score += min(0.3, asymmetry)

        # Secondary peaks contribution
        if n_secondary > 0:
            score += min(0.3, 0.1 * n_secondary + 0.2 * flux_ratio)

        return min(1.0, score)

    def _empty_result(self) -> LegacyResult:
        """Return empty result."""
        return LegacyResult(
            has_data=False,
            bands_available=[],
            ellipticity=0.0,
            asymmetry=0.0,
            fwhm_arcsec=0.0,
            n_secondary_peaks=0,
            secondary_peak_flux_ratio=0.0,
            blend_score=0.0,
            blend_flag='no_data'
        )


def check_for_blend(ra: float, dec: float) -> LegacyResult:
    """
    Convenience function to check for blends.

    Parameters
    ----------
    ra : float
        Right ascension (degrees)
    dec : float
        Declination (degrees)

    Returns
    -------
    LegacyResult
        Analysis results
    """
    analyzer = LegacyAnalyzer()
    return analyzer.analyze(ra, dec)
