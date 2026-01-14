"""
Default configuration values for kSZ analysis pipeline.

Symbol Table
------------
| Symbol          | Definition                                  | Units       |
|-----------------|---------------------------------------------|-------------|
| r               | Comoving pair separation                    | Mpc/h       |
| z               | Spectroscopic redshift                      | dimensionless|
| theta_inner     | Inner aperture radius                       | arcmin      |
| theta_outer     | Outer aperture radius                       | arcmin      |
| N_jk            | Number of jackknife regions                 | dimensionless|
| h               | Hubble parameter H_0 / (100 km/s/Mpc)       | dimensionless|
| Omega_m         | Matter density parameter                    | dimensionless|
| Omega_b         | Baryon density parameter                    | dimensionless|
| sigma_8         | Matter fluctuation amplitude at 8 Mpc/h     | dimensionless|
| n_s             | Scalar spectral index                       | dimensionless|
| T_CMB           | CMB temperature                             | K           |
"""

import numpy as np

# =============================================================================
# Separation Bins (Mpc/h)
# =============================================================================
# Default linear bins from 5 to 150 Mpc/h
# Signal peaks at 20-50 Mpc/h; extend to larger scales for covariance stability
DEFAULT_SEPARATION_BINS = np.linspace(5.0, 150.0, 16)  # 15 bins

# Alternative logarithmic binning for scale-dependent analysis
SEPARATION_BINS_LOG = np.logspace(np.log10(3.0), np.log10(200.0), 18)

# =============================================================================
# Redshift Bins for Tomography
# =============================================================================
# BGS: low-z (0.1 < z < 0.4)
# LRG: intermediate-z (0.4 < z < 0.8)
# ELG: high-z (0.8 < z < 1.6)

DEFAULT_REDSHIFT_BINS = {
    "BGS_BRIGHT": [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)],
    "BGS_FAINT": [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)],
    "LRG": [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)],
    "ELG_LOP": [(0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)],
}

# =============================================================================
# Aperture Photometry Parameters
# =============================================================================
# Compensated aperture filter: T_AP = T_inner - T_annulus
# Inner radius chosen to match typical halo virial radius at z~0.5
# Outer radius for background subtraction

DEFAULT_APERTURE_INNER = 1.8  # arcmin (corresponds to ~0.5 Mpc at z=0.5)
DEFAULT_APERTURE_OUTER = 5.0  # arcmin (annulus outer edge)

# Grid of apertures for optimization (arcmin)
APERTURE_GRID = [1.0, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]

# Fixed physical apertures (Mpc/h) - converted to angular at each z
PHYSICAL_APERTURES = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

# =============================================================================
# Covariance Estimation
# =============================================================================
DEFAULT_JACKKNIFE_REGIONS = 100  # Number of spatial jackknife regions
MIN_JACKKNIFE_REGIONS = 50      # Minimum for stable covariance
MAX_JACKKNIFE_REGIONS = 200     # Maximum (diminishing returns beyond this)

# HEALPix nside for defining jackknife regions
JACKKNIFE_NSIDE = 16  # ~3.7 deg² per pixel

# =============================================================================
# Fiducial Cosmology (Planck 2018 + DESI fiducial)
# =============================================================================
COSMOLOGY_PARAMS = {
    "h": 0.6736,              # Hubble parameter
    "Omega_m": 0.3153,        # Total matter density
    "Omega_b": 0.0493,        # Baryon density
    "Omega_cdm": 0.2660,      # CDM density (Omega_m - Omega_b)
    "sigma_8": 0.8111,        # Matter fluctuation amplitude
    "n_s": 0.9649,            # Scalar spectral index
    "tau": 0.0544,            # Optical depth to reionization
    "T_CMB": 2.7255,          # CMB temperature in Kelvin
}

# Speed of light in km/s
C_LIGHT_KM_S = 299792.458

# =============================================================================
# CMB Map Parameters
# =============================================================================
# ACT DR6 default settings
ACT_DR6_CONFIG = {
    "frequencies": [90, 150, 220],  # GHz
    "default_frequency": 150,
    "nside_equiv": 8192,            # Equivalent HEALPix nside for resolution
    "beam_fwhm": {                  # arcmin
        90: 2.2,
        150: 1.4,
        220: 1.0,
    },
    "pixel_size": 0.5,              # arcmin
}

# Planck PR4 default settings
PLANCK_PR4_CONFIG = {
    "components": ["commander", "sevem", "nilc", "smica"],
    "default_component": "commander",
    "nside": 2048,
    "beam_fwhm": 5.0,               # arcmin (effective)
}

# =============================================================================
# DESI LSS Catalog Parameters
# =============================================================================
DESI_LSS_CONFIG = {
    "tracers": ["BGS_BRIGHT", "BGS_FAINT", "LRG", "ELG_LOP", "QSO"],
    "regions": ["N", "S"],          # North and South Galactic Cap
    "version": "v1.5",              # LSS catalog version
    "weight_columns": [
        "WEIGHT_SYS",               # Imaging systematics
        "WEIGHT_COMP",              # Completeness
        "WEIGHT_ZFAIL",             # Redshift failure
        "WEIGHT_FKP",               # FKP optimal weight
    ],
}

# =============================================================================
# Inference Parameters
# =============================================================================
INFERENCE_DEFAULTS = {
    "n_walkers": 32,                # emcee walkers
    "n_samples": 10000,             # MCMC samples
    "n_burnin": 2000,               # Burn-in samples
    "prior_A_ksz": (-5.0, 5.0),     # Flat prior bounds on A_kSZ
    "prior_f_sigma8": (0.0, 2.0),   # Flat prior bounds on f*sigma_8
}

# =============================================================================
# Null Test Configuration
# =============================================================================
NULL_TEST_CONFIG = {
    "n_realizations": 1000,         # Random realizations for shuffle tests
    "pte_threshold": 0.05,          # PTE threshold for pass/fail
    "rotation_angles": [30, 60, 90, 120, 150, 180],  # degrees for CMB rotation
}

# =============================================================================
# Output Paths (relative to run directory)
# =============================================================================
OUTPUT_STRUCTURE = {
    "catalogs": "data/ksz/catalogs/",
    "maps": "data/ksz/maps/",
    "masks": "data/ksz/masks/",
    "filtered": "data/ksz/filtered/",
    "temperatures": "data/ksz/temperatures/",
    "pairwise": "data/ksz/pairwise/",
    "covariance": "data/ksz/cov/",
    "null_tests": "data/ksz/nulls/",
    "chains": "data/ksz/chains/",
    "plots": "plots/ksz/",
    "results": "data/ksz/results/",
}
