"""
pytest fixtures for kSZ analysis tests.

Provides mock data and reusable test components.
"""

import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(random_seed):
    """Numpy random generator."""
    return np.random.default_rng(random_seed)


@pytest.fixture
def mock_positions(rng):
    """
    Generate mock galaxy positions in comoving coordinates.

    Returns 1000 galaxies in a ~500 Mpc/h box.
    """
    n_gal = 1000
    box_size = 500.0  # Mpc/h

    # Random positions
    positions = rng.uniform(0, box_size, (n_gal, 3))

    return positions


@pytest.fixture
def mock_temperatures(rng, mock_positions):
    """
    Generate mock CMB temperatures at galaxy positions.

    Returns temperatures with realistic variance (~100 μK rms).
    """
    n_gal = len(mock_positions)
    return rng.standard_normal(n_gal) * 100.0  # μK


@pytest.fixture
def mock_weights(mock_positions):
    """
    Generate mock galaxy weights.

    All weights = 1 for simplicity.
    """
    n_gal = len(mock_positions)
    return np.ones(n_gal)


@pytest.fixture
def mock_catalog(rng):
    """
    Generate mock DESI-like catalog data.

    Returns dict with ra, dec, z, weights.
    """
    n_gal = 500

    # Random sky positions
    ra = rng.uniform(0, 360, n_gal)
    dec = rng.uniform(-60, 60, n_gal)  # Not full sky

    # Redshifts with realistic distribution
    z = rng.exponential(0.3, n_gal)
    z = np.clip(z, 0.01, 1.5)

    # Weights
    weights = np.ones(n_gal)

    return {
        'ra': ra,
        'dec': dec,
        'z': z,
        'weights': weights,
        'targetid': np.arange(n_gal),
    }


@pytest.fixture
def separation_bins():
    """Default separation bins for pairwise estimator."""
    return np.linspace(20, 150, 14)  # 13 bins from 20-150 Mpc/h


@pytest.fixture
def mock_pksz_data(separation_bins, rng):
    """
    Generate mock p(r) measurement.

    Returns r_centers, p_ksz, p_ksz_err with realistic values.
    """
    n_bins = len(separation_bins) - 1
    r_centers = 0.5 * (separation_bins[:-1] + separation_bins[1:])

    # Mock signal with ~1 μK amplitude, decaying with r
    p_ksz = 1.0 * np.exp(-r_centers / 100) + rng.standard_normal(n_bins) * 0.2

    # Errors ~20% of typical signal
    p_ksz_err = 0.2 * np.ones(n_bins)

    return {
        'r_centers': r_centers,
        'p_ksz': p_ksz,
        'p_ksz_err': p_ksz_err,
    }


@pytest.fixture
def mock_covariance(separation_bins, rng):
    """
    Generate mock covariance matrix.

    Returns positive-definite covariance with reasonable structure.
    """
    n_bins = len(separation_bins) - 1

    # Start with diagonal
    diag = 0.04 * np.ones(n_bins)  # σ ~ 0.2 μK

    # Add off-diagonal correlations
    corr = np.eye(n_bins)
    for i in range(n_bins):
        for j in range(n_bins):
            if i != j:
                corr[i, j] = 0.3 * np.exp(-abs(i - j) / 3)

    cov = np.outer(np.sqrt(diag), np.sqrt(diag)) * corr

    return cov


@pytest.fixture
def mock_theory_template(separation_bins):
    """
    Generate mock theory template.

    Returns p_theory(r) normalized to A_kSZ = 1.
    """
    r_centers = 0.5 * (separation_bins[:-1] + separation_bins[1:])

    # Simple decaying template
    p_theory = np.exp(-r_centers / 100)

    return p_theory


@pytest.fixture
def mock_healpix_map(rng):
    """
    Generate mock HEALPix temperature map.

    Returns NSIDE=64 map for fast tests.
    """
    nside = 64
    npix = 12 * nside ** 2

    # Random Gaussian map with ~100 μK rms
    T_map = rng.standard_normal(npix) * 100.0

    return {
        'data': T_map,
        'nside': nside,
    }


# Markers for tests that need optional dependencies
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_healpy: mark test as requiring healpy"
    )
    config.addinivalue_line(
        "markers", "requires_pixell: mark test as requiring pixell"
    )
    config.addinivalue_line(
        "markers", "requires_emcee: mark test as requiring emcee"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
