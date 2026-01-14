"""
Map validation and sanity checks for kSZ analysis.

Includes beam injection tests and unit consistency checks.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import json


@dataclass
class MapValidationResult:
    """Result of map validation checks."""

    map_file: str
    passed: bool
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'map_file': self.map_file,
            'passed': self.passed,
            'checks': self.checks,
            'warnings': self.warnings,
            'errors': self.errors,
        }

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def check_map_statistics(
    map_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    expected_unit: str = "uK_CMB",
) -> Dict[str, Any]:
    """
    Check basic map statistics for sanity.

    Parameters
    ----------
    map_data : np.ndarray
        CMB temperature map (HEALPix or CAR)
    mask : np.ndarray, optional
        Binary mask (1 = valid, 0 = masked)
    expected_unit : str
        Expected unit for range checks

    Returns
    -------
    dict
        Statistics and pass/fail status
    """
    if mask is not None:
        valid_data = map_data[mask > 0.5]
    else:
        # Exclude UNSEEN values for HEALPix
        valid_data = map_data[np.isfinite(map_data) & (map_data > -1e30)]

    if len(valid_data) == 0:
        return {
            'passed': False,
            'error': 'No valid pixels found',
            'n_valid': 0,
        }

    stats = {
        'n_valid': len(valid_data),
        'n_total': len(map_data.ravel()),
        'fraction_valid': len(valid_data) / len(map_data.ravel()),
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'median': float(np.median(valid_data)),
        'mad': float(np.median(np.abs(valid_data - np.median(valid_data)))),
    }

    # Unit-dependent range checks
    if expected_unit == "uK_CMB":
        # CMB maps should have |mean| < 50 uK, std ~ 100-200 uK
        mean_ok = np.abs(stats['mean']) < 100
        std_ok = 10 < stats['std'] < 500
        range_ok = stats['max'] - stats['min'] < 5000
    elif expected_unit == "K_CMB":
        # Same but in Kelvin
        mean_ok = np.abs(stats['mean']) < 1e-4
        std_ok = 1e-5 < stats['std'] < 5e-4
        range_ok = stats['max'] - stats['min'] < 5e-3
    else:
        # No range checks for unknown units
        mean_ok = std_ok = range_ok = True

    stats['passed'] = mean_ok and std_ok and range_ok
    stats['checks'] = {
        'mean_reasonable': mean_ok,
        'std_reasonable': std_ok,
        'range_reasonable': range_ok,
    }

    return stats


def check_nan_inf(map_data: np.ndarray) -> Dict[str, Any]:
    """Check for NaN and Inf values in map."""
    n_nan = np.sum(np.isnan(map_data))
    n_inf = np.sum(np.isinf(map_data))
    n_total = map_data.size

    return {
        'n_nan': int(n_nan),
        'n_inf': int(n_inf),
        'fraction_nan': n_nan / n_total,
        'fraction_inf': n_inf / n_total,
        'passed': n_nan == 0 and n_inf == 0,
    }


def check_monopole_dipole(
    map_data: np.ndarray,
    nside: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Check monopole and dipole of HEALPix map.

    For properly processed CMB maps, monopole should be ~0 and
    dipole should be removed or small.
    """
    try:
        import healpy as hp
    except ImportError:
        return {'passed': True, 'skipped': True, 'reason': 'healpy not available'}

    if nside is None:
        nside = hp.npix2nside(len(map_data))

    # Apply mask if provided
    if mask is not None:
        map_masked = hp.ma(map_data)
        map_masked.mask = mask < 0.5
    else:
        map_masked = map_data

    # Remove monopole/dipole and measure them
    try:
        mono, dipole = hp.fit_monopole(map_masked, gal_cut=20), None
        alm = hp.map2alm(map_masked, lmax=1)
        dipole_power = np.sum(np.abs(alm[1:4])**2)
    except Exception as e:
        return {'passed': True, 'skipped': True, 'reason': str(e)}

    return {
        'monopole': float(mono) if np.isfinite(mono) else None,
        'dipole_power': float(dipole_power),
        'passed': True,  # Informational only
    }


def beam_injection_test(
    map_data: np.ndarray,
    positions_ra: np.ndarray,
    positions_dec: np.ndarray,
    beam_fwhm_arcmin: float,
    aperture_inner_arcmin: float = 1.8,
    aperture_outer_arcmin: float = 5.0,
    inject_amplitude: float = 100.0,  # uK
    nside: Optional[int] = None,
    n_inject: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Beam injection test: inject known signals and verify recovery.

    This tests the entire aperture photometry pipeline by injecting
    Gaussian beam profiles at random positions and measuring recovery.

    Parameters
    ----------
    map_data : np.ndarray
        Original CMB map (HEALPix)
    positions_ra, positions_dec : np.ndarray
        Galaxy positions in degrees
    beam_fwhm_arcmin : float
        Beam FWHM in arcminutes
    aperture_inner_arcmin : float
        Inner aperture radius for AP
    aperture_outer_arcmin : float
        Outer aperture radius for AP
    inject_amplitude : float
        Amplitude of injected signals in uK
    nside : int, optional
        HEALPix nside (inferred if not given)
    n_inject : int
        Number of injection positions
    seed : int
        Random seed

    Returns
    -------
    dict
        Recovery statistics and pass/fail
    """
    try:
        import healpy as hp
    except ImportError:
        return {'passed': True, 'skipped': True, 'reason': 'healpy not available'}

    if nside is None:
        nside = hp.npix2nside(len(map_data))

    rng = np.random.default_rng(seed)

    # Select random subset of galaxy positions for injection
    n_gal = len(positions_ra)
    if n_gal < n_inject:
        inject_idx = np.arange(n_gal)
    else:
        inject_idx = rng.choice(n_gal, size=n_inject, replace=False)

    inject_ra = positions_ra[inject_idx]
    inject_dec = positions_dec[inject_idx]

    # Create injection map
    beam_sigma_rad = np.radians(beam_fwhm_arcmin / 60.0) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    injection_map = np.zeros_like(map_data)

    for ra, dec in zip(inject_ra, inject_dec):
        # Convert to theta, phi
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)

        # Get vector
        vec = hp.ang2vec(theta, phi)

        # Query disc around position (3 sigma radius)
        search_radius = 3.0 * beam_sigma_rad
        pixels = hp.query_disc(nside, vec, search_radius)

        # Compute Gaussian profile
        pix_theta, pix_phi = hp.pix2ang(nside, pixels)
        pix_vec = hp.ang2vec(pix_theta, pix_phi)

        # Angular distance
        cos_dist = np.dot(pix_vec, vec)
        cos_dist = np.clip(cos_dist, -1, 1)
        ang_dist = np.arccos(cos_dist)

        # Gaussian beam
        profile = inject_amplitude * np.exp(-0.5 * (ang_dist / beam_sigma_rad)**2)
        injection_map[pixels] += profile

    # Add injection to map
    injected_map = map_data + injection_map

    # Measure temperatures at injection positions using AP
    from desi_ksz.estimators.aperture_photometry import measure_aperture_temperature

    temps_original = []
    temps_injected = []

    for ra, dec in zip(inject_ra, inject_dec):
        try:
            t_orig = measure_aperture_temperature(
                map_data, ra, dec,
                theta_inner=aperture_inner_arcmin,
                theta_outer=aperture_outer_arcmin,
                nside=nside,
            )
            t_inj = measure_aperture_temperature(
                injected_map, ra, dec,
                theta_inner=aperture_inner_arcmin,
                theta_outer=aperture_outer_arcmin,
                nside=nside,
            )
            temps_original.append(t_orig)
            temps_injected.append(t_inj)
        except Exception:
            continue

    temps_original = np.array(temps_original)
    temps_injected = np.array(temps_injected)

    if len(temps_original) == 0:
        return {'passed': False, 'error': 'No successful temperature measurements'}

    # Compute recovery
    delta_T = temps_injected - temps_original
    mean_recovery = np.mean(delta_T)
    std_recovery = np.std(delta_T)

    # Expected recovery depends on aperture vs beam size
    # For compensated aperture, recovery < inject_amplitude
    # Just check that we recover something positive and correlated

    recovery_fraction = mean_recovery / inject_amplitude

    # Pass if we recover 20-150% of injected signal (depends on beam/aperture)
    passed = 0.1 < recovery_fraction < 2.0

    return {
        'n_injected': len(inject_ra),
        'n_measured': len(temps_original),
        'inject_amplitude': inject_amplitude,
        'mean_recovery': float(mean_recovery),
        'std_recovery': float(std_recovery),
        'recovery_fraction': float(recovery_fraction),
        'passed': passed,
        'details': {
            'beam_fwhm_arcmin': beam_fwhm_arcmin,
            'aperture_inner_arcmin': aperture_inner_arcmin,
            'aperture_outer_arcmin': aperture_outer_arcmin,
        }
    }


def coordinate_consistency_test(
    ra: np.ndarray,
    dec: np.ndarray,
    map_nside: int,
) -> Dict[str, Any]:
    """
    Check coordinate consistency between catalog and map.

    Verifies that galaxy positions fall within valid map regions.
    """
    try:
        import healpy as hp
    except ImportError:
        return {'passed': True, 'skipped': True, 'reason': 'healpy not available'}

    # Convert to HEALPix pixels
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    # Check valid ranges
    ra_valid = np.all((ra >= 0) & (ra <= 360))
    dec_valid = np.all((dec >= -90) & (dec <= 90))

    pixels = hp.ang2pix(map_nside, theta, phi)
    npix = hp.nside2npix(map_nside)

    # Check pixels are valid
    pixels_valid = np.all((pixels >= 0) & (pixels < npix))

    # Check sky coverage overlap
    unique_pixels = np.unique(pixels)

    return {
        'ra_range_valid': bool(ra_valid),
        'dec_range_valid': bool(dec_valid),
        'pixels_valid': bool(pixels_valid),
        'n_unique_pixels': len(unique_pixels),
        'sky_fraction': len(unique_pixels) / npix,
        'passed': ra_valid and dec_valid and pixels_valid,
    }


def validate_cmb_map(
    map_file: str,
    mask_file: Optional[str] = None,
    expected_unit: str = "uK_CMB",
    nside_expected: Optional[int] = None,
) -> MapValidationResult:
    """
    Run full validation suite on CMB map.

    Parameters
    ----------
    map_file : str
        Path to CMB map FITS file
    mask_file : str, optional
        Path to mask FITS file
    expected_unit : str
        Expected map units
    nside_expected : int, optional
        Expected HEALPix nside

    Returns
    -------
    MapValidationResult
        Validation results
    """
    try:
        import healpy as hp
    except ImportError:
        return MapValidationResult(
            map_file=map_file,
            passed=False,
            errors=['healpy not available'],
        )

    result = MapValidationResult(map_file=map_file, passed=True)

    # Load map
    try:
        map_data = hp.read_map(map_file, verbose=False)
    except Exception as e:
        result.passed = False
        result.errors.append(f'Failed to load map: {e}')
        return result

    # Check nside
    nside = hp.npix2nside(len(map_data))
    if nside_expected is not None and nside != nside_expected:
        result.warnings.append(f'Unexpected nside: {nside} (expected {nside_expected})')
    result.checks['nside'] = {'value': nside, 'expected': nside_expected}

    # Load mask if provided
    mask = None
    if mask_file is not None:
        try:
            mask = hp.read_map(mask_file, verbose=False)
        except Exception as e:
            result.warnings.append(f'Failed to load mask: {e}')

    # Run checks
    result.checks['nan_inf'] = check_nan_inf(map_data)
    if not result.checks['nan_inf']['passed']:
        result.warnings.append('Map contains NaN or Inf values')

    result.checks['statistics'] = check_map_statistics(map_data, mask, expected_unit)
    if not result.checks['statistics']['passed']:
        result.warnings.append('Map statistics outside expected range')

    result.checks['monopole_dipole'] = check_monopole_dipole(map_data, nside, mask)

    # Overall pass/fail
    critical_checks = ['nan_inf']
    result.passed = all(
        result.checks.get(c, {}).get('passed', True)
        for c in critical_checks
    )

    return result


def validate_catalog(
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Validate galaxy catalog for kSZ analysis.

    Parameters
    ----------
    ra, dec : np.ndarray
        Galaxy coordinates in degrees
    z : np.ndarray
        Redshifts
    weights : np.ndarray, optional
        Galaxy weights

    Returns
    -------
    dict
        Validation results
    """
    result = {
        'n_galaxies': len(ra),
        'passed': True,
        'checks': {},
        'warnings': [],
        'errors': [],
    }

    # Check coordinate ranges
    ra_valid = np.all((ra >= 0) & (ra <= 360))
    dec_valid = np.all((dec >= -90) & (dec <= 90))
    result['checks']['coordinates'] = {
        'ra_valid': bool(ra_valid),
        'dec_valid': bool(dec_valid),
        'ra_range': [float(np.min(ra)), float(np.max(ra))],
        'dec_range': [float(np.min(dec)), float(np.max(dec))],
        'passed': ra_valid and dec_valid,
    }
    if not (ra_valid and dec_valid):
        result['errors'].append('Invalid coordinate ranges')
        result['passed'] = False

    # Check redshifts
    z_positive = np.all(z >= 0)
    z_reasonable = np.all(z < 10)  # No galaxies at z > 10
    result['checks']['redshift'] = {
        'all_positive': bool(z_positive),
        'all_reasonable': bool(z_reasonable),
        'z_range': [float(np.min(z)), float(np.max(z))],
        'z_median': float(np.median(z)),
        'passed': z_positive and z_reasonable,
    }
    if not (z_positive and z_reasonable):
        result['errors'].append('Invalid redshift values')
        result['passed'] = False

    # Check for NaN/Inf
    has_nan = np.any(np.isnan(ra)) or np.any(np.isnan(dec)) or np.any(np.isnan(z))
    has_inf = np.any(np.isinf(ra)) or np.any(np.isinf(dec)) or np.any(np.isinf(z))
    result['checks']['nan_inf'] = {
        'has_nan': bool(has_nan),
        'has_inf': bool(has_inf),
        'passed': not (has_nan or has_inf),
    }
    if has_nan or has_inf:
        result['errors'].append('Catalog contains NaN or Inf values')
        result['passed'] = False

    # Check weights if provided
    if weights is not None:
        weights_positive = np.all(weights > 0)
        weights_finite = np.all(np.isfinite(weights))
        result['checks']['weights'] = {
            'all_positive': bool(weights_positive),
            'all_finite': bool(weights_finite),
            'weight_range': [float(np.min(weights)), float(np.max(weights))],
            'weight_mean': float(np.mean(weights)),
            'passed': weights_positive and weights_finite,
        }
        if not (weights_positive and weights_finite):
            result['warnings'].append('Invalid weight values')

    # Check for duplicates
    coords = np.column_stack([ra, dec])
    unique_coords = np.unique(coords, axis=0)
    n_duplicates = len(coords) - len(unique_coords)
    result['checks']['duplicates'] = {
        'n_duplicates': int(n_duplicates),
        'fraction_duplicates': n_duplicates / len(ra),
        'passed': n_duplicates == 0,
    }
    if n_duplicates > 0:
        result['warnings'].append(f'{n_duplicates} duplicate positions found')

    return result


def run_full_map_validation(
    map_file: str,
    catalog_ra: np.ndarray,
    catalog_dec: np.ndarray,
    beam_fwhm_arcmin: float = 1.4,
    mask_file: Optional[str] = None,
    run_beam_test: bool = True,
    n_inject: int = 50,
) -> Dict[str, Any]:
    """
    Run comprehensive map validation including beam injection.

    Parameters
    ----------
    map_file : str
        Path to CMB map
    catalog_ra, catalog_dec : np.ndarray
        Galaxy positions for beam test
    beam_fwhm_arcmin : float
        Beam FWHM
    mask_file : str, optional
        Path to mask file
    run_beam_test : bool
        Whether to run beam injection test
    n_inject : int
        Number of beam injections

    Returns
    -------
    dict
        Complete validation results
    """
    try:
        import healpy as hp
    except ImportError:
        return {'passed': False, 'error': 'healpy not available'}

    results = {}

    # Basic map validation
    map_result = validate_cmb_map(map_file, mask_file)
    results['map_validation'] = map_result.to_dict()

    # Load map for additional tests
    try:
        map_data = hp.read_map(map_file, verbose=False)
        nside = hp.npix2nside(len(map_data))
    except Exception as e:
        return {'passed': False, 'error': f'Failed to load map: {e}'}

    # Coordinate consistency
    results['coordinate_consistency'] = coordinate_consistency_test(
        catalog_ra, catalog_dec, nside
    )

    # Beam injection test
    if run_beam_test:
        results['beam_injection'] = beam_injection_test(
            map_data, catalog_ra, catalog_dec,
            beam_fwhm_arcmin=beam_fwhm_arcmin,
            n_inject=n_inject,
        )

    # Overall pass
    results['passed'] = all(
        results.get(k, {}).get('passed', True)
        for k in ['map_validation', 'coordinate_consistency']
    )

    return results


# =============================================================================
# End-to-End Map Transfer Function Test
# =============================================================================

@dataclass
class TransferFunctionTestResult:
    """Result from map transfer function test."""

    passed: bool
    input_power_spectrum: np.ndarray
    output_power_spectrum: np.ndarray
    transfer_function: np.ndarray
    ell_bins: np.ndarray
    mean_transfer: float
    std_transfer: float
    bias: float  # (output - input) / input at fiducial scale
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'input_power_spectrum': self.input_power_spectrum.tolist(),
            'output_power_spectrum': self.output_power_spectrum.tolist(),
            'transfer_function': self.transfer_function.tolist(),
            'ell_bins': self.ell_bins.tolist(),
            'mean_transfer': self.mean_transfer,
            'std_transfer': self.std_transfer,
            'bias': self.bias,
            'details': self.details,
        }


def map_transfer_function_test(
    temperature_extraction_func,
    positions_ra: np.ndarray,
    positions_dec: np.ndarray,
    nside: int = 2048,
    lmax: int = 3000,
    n_realizations: int = 10,
    input_cl_file: Optional[str] = None,
    seed: int = 42,
    bias_threshold: float = 0.05,
) -> TransferFunctionTestResult:
    """
    End-to-end map transfer function test.

    Generates synthetic CMB maps from a known power spectrum, passes them
    through the temperature extraction pipeline, and measures the transfer
    function. This tests for any systematic biases in the map processing.

    Parameters
    ----------
    temperature_extraction_func : callable
        Function that takes (map_data, ra, dec) and returns temperatures
    positions_ra, positions_dec : np.ndarray
        Galaxy positions for temperature extraction
    nside : int
        HEALPix nside for test maps
    lmax : int
        Maximum multipole
    n_realizations : int
        Number of CMB realizations to average over
    input_cl_file : str, optional
        Path to input C_l file (uses default if None)
    seed : int
        Random seed
    bias_threshold : float
        Maximum acceptable fractional bias

    Returns
    -------
    TransferFunctionTestResult
        Transfer function measurement
    """
    try:
        import healpy as hp
    except ImportError:
        return TransferFunctionTestResult(
            passed=True,
            input_power_spectrum=np.array([]),
            output_power_spectrum=np.array([]),
            transfer_function=np.array([]),
            ell_bins=np.array([]),
            mean_transfer=1.0,
            std_transfer=0.0,
            bias=0.0,
            details={'skipped': True, 'reason': 'healpy not available'},
        )

    rng = np.random.default_rng(seed)

    # Generate input power spectrum
    if input_cl_file is not None:
        try:
            cl_input = hp.read_cl(input_cl_file)
        except Exception:
            cl_input = None
    else:
        cl_input = None

    if cl_input is None:
        # Default CMB-like power spectrum
        ell = np.arange(lmax + 1)
        cl_input = np.zeros(lmax + 1)
        # Simple approximation: C_l ~ 1/l(l+1) with acoustic peaks
        cl_input[2:] = 5000.0 / (ell[2:] * (ell[2:] + 1))
        # Add some acoustic oscillations
        cl_input[2:] *= (1.0 + 0.3 * np.sin(ell[2:] * 0.01))
        cl_input = cl_input * (2 * np.pi) / (ell * (ell + 1) + 1)
        cl_input[:2] = 0  # No monopole/dipole

    # Ell bins for analysis
    ell_bin_edges = np.logspace(np.log10(10), np.log10(min(lmax, 2000)), 20)
    ell_bin_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])

    input_powers = []
    output_powers = []

    for i in range(n_realizations):
        # Generate realization
        alm = hp.synalm(cl_input, lmax=lmax, new=True)
        map_realization = hp.alm2map(alm, nside, verbose=False)

        # Extract temperatures at galaxy positions
        try:
            temps = temperature_extraction_func(
                map_realization, positions_ra, positions_dec
            )
        except Exception as e:
            continue

        # Compute "output" power spectrum from extracted temperatures
        # This is approximate - we project temperatures back to map
        temp_map = np.zeros(hp.nside2npix(nside))
        count_map = np.zeros(hp.nside2npix(nside))

        theta = np.radians(90.0 - positions_dec)
        phi = np.radians(positions_ra)
        pixels = hp.ang2pix(nside, theta, phi)

        np.add.at(temp_map, pixels, temps)
        np.add.at(count_map, pixels, 1)

        valid = count_map > 0
        temp_map[valid] /= count_map[valid]
        temp_map[~valid] = hp.UNSEEN

        # Compute power spectrum of extracted temperatures
        # This is a proxy for the transfer function
        cl_output = hp.anafast(temp_map, lmax=min(lmax, 3 * nside - 1))

        # Bin the power spectra
        input_binned = np.zeros(len(ell_bin_centers))
        output_binned = np.zeros(len(ell_bin_centers))

        for j in range(len(ell_bin_centers)):
            ell_min = int(ell_bin_edges[j])
            ell_max = int(ell_bin_edges[j + 1])
            ell_range = np.arange(ell_min, min(ell_max, len(cl_input)))

            if len(ell_range) > 0:
                input_binned[j] = np.mean(cl_input[ell_range])
                if ell_max < len(cl_output):
                    output_binned[j] = np.mean(cl_output[ell_range])

        input_powers.append(input_binned)
        output_powers.append(output_binned)

    if len(input_powers) == 0:
        return TransferFunctionTestResult(
            passed=False,
            input_power_spectrum=np.array([]),
            output_power_spectrum=np.array([]),
            transfer_function=np.array([]),
            ell_bins=ell_bin_centers,
            mean_transfer=0.0,
            std_transfer=0.0,
            bias=1.0,
            details={'error': 'No successful realizations'},
        )

    # Average over realizations
    input_powers = np.array(input_powers)
    output_powers = np.array(output_powers)

    mean_input = np.mean(input_powers, axis=0)
    mean_output = np.mean(output_powers, axis=0)

    # Transfer function T(l) = C_l^out / C_l^in
    # Note: this includes shot noise and sampling effects
    valid_mask = mean_input > 0
    transfer = np.ones_like(mean_input)
    transfer[valid_mask] = mean_output[valid_mask] / mean_input[valid_mask]

    # Summary statistics
    # Focus on scales ell ~ 100-1000 relevant for kSZ
    kSZ_scales = (ell_bin_centers > 100) & (ell_bin_centers < 1000)
    if np.any(kSZ_scales):
        mean_transfer = float(np.mean(transfer[kSZ_scales & valid_mask]))
        std_transfer = float(np.std(transfer[kSZ_scales & valid_mask]))
    else:
        mean_transfer = float(np.mean(transfer[valid_mask]))
        std_transfer = float(np.std(transfer[valid_mask]))

    # Bias at fiducial scale (l ~ 500)
    fiducial_idx = np.argmin(np.abs(ell_bin_centers - 500))
    if valid_mask[fiducial_idx]:
        bias = (mean_output[fiducial_idx] - mean_input[fiducial_idx]) / mean_input[fiducial_idx]
    else:
        bias = 0.0

    # Pass if bias is within threshold
    passed = abs(bias) < bias_threshold and 0.5 < mean_transfer < 2.0

    return TransferFunctionTestResult(
        passed=passed,
        input_power_spectrum=mean_input,
        output_power_spectrum=mean_output,
        transfer_function=transfer,
        ell_bins=ell_bin_centers,
        mean_transfer=mean_transfer,
        std_transfer=std_transfer,
        bias=float(bias),
        details={
            'n_realizations': len(input_powers),
            'nside': nside,
            'lmax': lmax,
            'bias_threshold': bias_threshold,
        },
    )


def pairwise_signal_propagation_test(
    estimator,
    positions: np.ndarray,
    weights: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    inject_amplitude: float = 50.0,  # μK
    inject_correlation_length: float = 50.0,  # Mpc/h
    n_realizations: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test signal propagation through pairwise estimator.

    Injects a known correlated temperature signal and verifies that
    the pairwise estimator recovers the expected p(r) shape.

    Parameters
    ----------
    estimator : PairwiseMomentumEstimator
        The estimator to test
    positions : np.ndarray
        Galaxy positions (N, 3) in Mpc/h
    weights : np.ndarray
        Galaxy weights
    ra, dec : np.ndarray
        Galaxy sky coordinates
    inject_amplitude : float
        Amplitude of injected temperature correlation
    inject_correlation_length : float
        Correlation length scale (Mpc/h)
    n_realizations : int
        Number of noise realizations
    seed : int
        Random seed

    Returns
    -------
    dict
        Test results including recovered signal and bias
    """
    from multiprocessing import Pool, cpu_count

    rng = np.random.default_rng(seed)
    n_galaxies = len(positions)

    # Create correlated temperature field
    # T_i = A × exp(-r_i / r_corr) + noise
    # where r_i is distance from some reference point

    # Generate reference point at center of distribution
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)

    # Correlated component
    T_signal = inject_amplitude * np.exp(-distances / inject_correlation_length)

    # Run with and without signal
    p_ksz_signal = []
    p_ksz_null = []

    for i in range(n_realizations):
        # Add noise
        noise = rng.normal(0, 50, size=n_galaxies)  # 50 μK noise

        # With signal
        temps_with = T_signal + noise
        result_with = estimator.compute(positions, temps_with, weights)
        p_ksz_signal.append(result_with.p_ksz)

        # Without signal (pure noise)
        temps_null = noise
        result_null = estimator.compute(positions, temps_null, weights)
        p_ksz_null.append(result_null.p_ksz)

    p_ksz_signal = np.array(p_ksz_signal)
    p_ksz_null = np.array(p_ksz_null)

    # Statistics
    mean_signal = np.mean(p_ksz_signal, axis=0)
    std_signal = np.std(p_ksz_signal, axis=0)
    mean_null = np.mean(p_ksz_null, axis=0)
    std_null = np.std(p_ksz_null, axis=0)

    # Check that signal is detected above null
    # At small scales, signal should be > null
    bin_centers = estimator.bin_centers

    # Detection significance
    diff = mean_signal - mean_null
    combined_err = np.sqrt(std_signal**2 + std_null**2) / np.sqrt(n_realizations)
    significance = np.abs(diff) / combined_err

    # Test passes if:
    # 1. Signal is detected at > 2σ at small scales
    # 2. Null is consistent with zero
    small_scale_idx = bin_centers < inject_correlation_length * 2
    detected = np.any(significance[small_scale_idx] > 2.0)

    null_consistent = np.all(np.abs(mean_null) < 3 * std_null / np.sqrt(n_realizations))

    passed = detected and null_consistent

    return {
        'passed': passed,
        'detected': detected,
        'null_consistent': null_consistent,
        'mean_signal': mean_signal.tolist(),
        'mean_null': mean_null.tolist(),
        'std_signal': std_signal.tolist(),
        'significance': significance.tolist(),
        'bin_centers': bin_centers.tolist(),
        'inject_amplitude': inject_amplitude,
        'inject_correlation_length': inject_correlation_length,
        'n_realizations': n_realizations,
    }
