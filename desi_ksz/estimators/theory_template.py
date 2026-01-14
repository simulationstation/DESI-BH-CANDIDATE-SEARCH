"""
Theory template for pairwise kSZ momentum.

This module computes theoretical predictions for the pairwise kSZ
momentum signal using linear theory and the mean pairwise velocity.

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|-------------------------------------------------|-------------|
| p_th(r)     | Theory prediction for pairwise momentum         | muK         |
| v_12(r)     | Mean pairwise velocity                          | km/s        |
| ξ(r)        | Two-point correlation function                  | dimensionless|
| ξ̄(r)        | Volume-averaged correlation function            | dimensionless|
| f(z)        | Growth rate d ln D / d ln a                     | dimensionless|
| H(z)        | Hubble parameter at redshift z                  | km/s/Mpc    |
| a           | Scale factor = 1/(1+z)                          | dimensionless|
| τ̄           | Mean optical depth                              | dimensionless|
| T_CMB       | CMB temperature                                 | K (or muK)  |
| c           | Speed of light                                  | km/s        |

Theory
------
The pairwise kSZ momentum is related to the mean pairwise velocity:

    p_theory(r, z) = -τ̄ × T_CMB × v_12(r, z) / c

The mean pairwise velocity from linear theory (Kaiser 1987):

    v_12(r, z) = -(2/3) × f(z) × H(z) × a × r × ξ̄(r, z) / [1 + ξ(r, z)]

where:
    ξ̄(r) = (3/r³) ∫₀ʳ ξ(r') r'² dr' is the volume-averaged correlation function
    f(z) ≈ Ω_m(z)^0.55 is the growth rate

References
----------
- Kaiser, N. 1987, MNRAS, 227, 1
- Ferreira, P. G., et al. 1999, ApJ, 515, L1
- Sheth, R. K. 1996, MNRAS, 279, 1310
"""

from typing import Optional, Tuple, Callable
import numpy as np
import logging
from scipy.integrate import quad
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# Physical constants
C_LIGHT_KM_S = 299792.458  # km/s
T_CMB_MUKELVIN = 2.7255e6  # muK


def compute_theory_template(
    r: np.ndarray,
    z: float,
    tau_bar: float = 1e-4,
    cosmology: Optional[dict] = None,
    xi_func: Optional[Callable] = None,
) -> np.ndarray:
    """
    Compute theory template for pairwise kSZ momentum.

    Parameters
    ----------
    r : np.ndarray
        Comoving separations in Mpc/h
    z : float
        Redshift
    tau_bar : float
        Mean optical depth (default: 1e-4, typical for LRGs)
    cosmology : dict, optional
        Cosmology parameters. If None, uses default.
    xi_func : callable, optional
        Custom correlation function ξ(r). If None, uses default model.

    Returns
    -------
    np.ndarray
        Theory prediction p(r) in muK

    Notes
    -----
    The default tau_bar is uncertain and depends on:
    - Halo mass and gas profile
    - Galaxy-halo connection model
    - This should be marginalized over in parameter inference
    """
    if cosmology is None:
        cosmology = {
            "h": 0.6736,
            "Omega_m": 0.3153,
            "sigma_8": 0.8111,
        }

    # Compute mean pairwise velocity
    v_12 = compute_mean_pairwise_velocity(r, z, cosmology, xi_func)

    # Convert to temperature: p = -tau * T_CMB * v_12 / c
    p_theory = -tau_bar * T_CMB_MUKELVIN * v_12 / C_LIGHT_KM_S

    return p_theory


def compute_mean_pairwise_velocity(
    r: np.ndarray,
    z: float,
    cosmology: Optional[dict] = None,
    xi_func: Optional[Callable] = None,
) -> np.ndarray:
    """
    Compute mean pairwise velocity from linear theory.

    The mean pairwise velocity (Sheth 1996, Kaiser 1987):

        v_12(r) = -(2/3) × f × H × a × r × ξ̄(r) / [1 + ξ(r)]

    Parameters
    ----------
    r : np.ndarray
        Comoving separations in Mpc/h
    z : float
        Redshift
    cosmology : dict, optional
        Cosmology parameters
    xi_func : callable, optional
        Correlation function ξ(r) in Mpc/h

    Returns
    -------
    np.ndarray
        Mean pairwise velocity in km/s (negative = infall)
    """
    if cosmology is None:
        cosmology = {
            "h": 0.6736,
            "Omega_m": 0.3153,
            "sigma_8": 0.8111,
        }

    h = cosmology["h"]
    Omega_m = cosmology["Omega_m"]

    # Scale factor
    a = 1.0 / (1.0 + z)

    # Hubble parameter H(z) in km/s/(Mpc/h)
    # H(z) = H_0 * E(z) where H_0 = 100*h km/s/Mpc
    Omega_Lambda = 1.0 - Omega_m
    Ez = np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
    Hz = 100.0 * Ez  # km/s/(Mpc/h)

    # Growth rate f ≈ Omega_m(z)^0.55
    Omega_m_z = Omega_m * (1 + z)**3 / Ez**2
    f_growth = Omega_m_z**0.55

    # Get correlation function
    if xi_func is None:
        xi_func = _default_xi_model(z, cosmology)

    # Compute xi and xi_bar at each r
    xi = xi_func(r)
    xi_bar = compute_xi_bar(r, xi_func)

    # Mean pairwise velocity
    # v_12 = -(2/3) * f * H * a * r * xi_bar / (1 + xi)
    v_12 = -(2.0 / 3.0) * f_growth * Hz * a * r * xi_bar / (1.0 + xi)

    return v_12


def compute_xi_bar(
    r: np.ndarray,
    xi_func: Callable,
) -> np.ndarray:
    """
    Compute volume-averaged correlation function.

    ξ̄(r) = (3/r³) ∫₀ʳ ξ(r') r'² dr'

    Parameters
    ----------
    r : np.ndarray
        Radii in Mpc/h
    xi_func : callable
        Correlation function ξ(r)

    Returns
    -------
    np.ndarray
        Volume-averaged ξ̄(r)
    """
    xi_bar = np.zeros_like(r)

    for i, ri in enumerate(r):
        if ri <= 0:
            xi_bar[i] = 0.0
            continue

        # Integrate 3/r^3 * int_0^r xi(r') r'^2 dr'
        def integrand(rp):
            return xi_func(rp) * rp**2

        integral, _ = quad(integrand, 0, ri, limit=100)
        xi_bar[i] = 3.0 * integral / ri**3

    return xi_bar


def compute_correlation_function(
    r: np.ndarray,
    z: float = 0.5,
    cosmology: Optional[dict] = None,
    use_camb: bool = True,
) -> np.ndarray:
    """
    Compute matter correlation function ξ(r).

    Parameters
    ----------
    r : np.ndarray
        Separations in Mpc/h
    z : float
        Redshift
    cosmology : dict, optional
        Cosmology parameters
    use_camb : bool
        Try to use CAMB for P(k). If False or unavailable, use fitting formula.

    Returns
    -------
    np.ndarray
        Correlation function ξ(r)
    """
    if use_camb:
        try:
            return _compute_xi_from_camb(r, z, cosmology)
        except ImportError:
            logger.warning("CAMB not available, using fitting formula")

    # Use fitting formula as fallback
    return _xi_fitting_formula(r, z, cosmology)


def _default_xi_model(z: float, cosmology: dict) -> Callable:
    """
    Create default correlation function model.

    Uses a power-law approximation calibrated to simulations.
    """
    # Simple power-law model: xi(r) = (r_0/r)^gamma
    # with r_0 ~ 5 Mpc/h, gamma ~ 1.8 at z=0
    # Scale with growth factor

    # Rough growth factor scaling
    Omega_m = cosmology.get("Omega_m", 0.3)
    sigma_8 = cosmology.get("sigma_8", 0.8)

    # Growth factor D(z) relative to z=0
    D_z = _growth_factor(z, Omega_m)

    # Correlation length scales with sigma_8 and D(z)
    r0 = 5.0 * (sigma_8 / 0.8) * (D_z / 1.0)  # Mpc/h
    gamma = 1.8

    def xi(r):
        r = np.atleast_1d(r)
        result = np.zeros_like(r)
        mask = r > 0
        result[mask] = (r0 / r[mask])**gamma
        # Limit at small r to avoid infinity
        result = np.minimum(result, 1000.0)
        return result

    return xi


def _growth_factor(z: float, Omega_m: float) -> float:
    """
    Approximate growth factor D(z) / D(0).

    Uses the fitting formula from Carroll, Press & Turner (1992).
    """
    Omega_Lambda = 1.0 - Omega_m
    a = 1.0 / (1.0 + z)

    # Omega_m(z) and Omega_Lambda(z)
    Omega_m_z = Omega_m / (Omega_m + Omega_Lambda * a**3)
    Omega_L_z = Omega_Lambda * a**3 / (Omega_m + Omega_Lambda * a**3)

    # Growth factor approximation
    D_z = (5.0 / 2.0) * Omega_m_z / (
        Omega_m_z**(4.0/7.0) - Omega_L_z +
        (1.0 + Omega_m_z / 2.0) * (1.0 + Omega_L_z / 70.0)
    )

    # Same at z=0
    D_0 = (5.0 / 2.0) * Omega_m / (
        Omega_m**(4.0/7.0) - Omega_Lambda +
        (1.0 + Omega_m / 2.0) * (1.0 + Omega_Lambda / 70.0)
    )

    return D_z / D_0 * a  # Include scale factor


def _xi_fitting_formula(
    r: np.ndarray,
    z: float,
    cosmology: Optional[dict] = None,
) -> np.ndarray:
    """
    Fitting formula for correlation function.

    Based on Eisenstein & Hu 1999 BAO model.
    """
    if cosmology is None:
        cosmology = {"Omega_m": 0.3, "sigma_8": 0.8, "h": 0.7}

    Omega_m = cosmology.get("Omega_m", 0.3)
    sigma_8 = cosmology.get("sigma_8", 0.8)
    h = cosmology.get("h", 0.7)

    # Growth factor
    D_z = _growth_factor(z, Omega_m)

    # Amplitude
    A = (sigma_8 / 0.8)**2 * D_z**2

    # Power-law part
    r0 = 5.0  # Mpc/h
    gamma = 1.8
    xi_pl = A * (r0 / np.maximum(r, 0.1))**gamma

    # BAO feature (simplified Gaussian)
    r_bao = 105.0  # Mpc/h (BAO scale)
    sigma_bao = 10.0  # Mpc/h (smoothing)
    A_bao = 0.05 * A  # BAO amplitude
    xi_bao = A_bao * np.exp(-0.5 * ((r - r_bao) / sigma_bao)**2)

    return xi_pl + xi_bao


def _compute_xi_from_camb(
    r: np.ndarray,
    z: float,
    cosmology: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute correlation function using CAMB power spectrum.
    """
    import camb

    if cosmology is None:
        cosmology = {"h": 0.6736, "Omega_m": 0.3153, "Omega_b": 0.0493,
                     "sigma_8": 0.8111, "n_s": 0.9649}

    h = cosmology["h"]
    H0 = 100.0 * h

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0,
        ombh2=cosmology.get("Omega_b", 0.0493) * h**2,
        omch2=(cosmology.get("Omega_m", 0.3) - cosmology.get("Omega_b", 0.05)) * h**2,
    )
    pars.InitPower.set_params(
        As=2.1e-9,  # Will be rescaled by sigma_8
        ns=cosmology.get("n_s", 0.965),
    )

    # Get matter power spectrum
    pars.set_matter_power(redshifts=[z], kmax=10.0)
    results = camb.get_results(pars)
    kh, _, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10.0, npoints=500)

    # P(k) in (Mpc/h)^3
    pk = pk[0]  # First redshift

    # Rescale to target sigma_8
    sigma_8_camb = results.get_sigma8()[0]
    pk *= (cosmology.get("sigma_8", 0.8) / sigma_8_camb)**2

    # Fourier transform to get xi(r)
    # xi(r) = 1/(2*pi^2) * int k^2 P(k) sin(kr)/(kr) dk
    xi = np.zeros_like(r)
    for i, ri in enumerate(r):
        if ri <= 0:
            continue

        # Integrate numerically
        kr = kh * ri
        integrand = kh**2 * pk * np.sinc(kr / np.pi)  # np.sinc includes pi factor
        xi[i] = np.trapz(integrand, kh) / (2.0 * np.pi**2) * np.pi  # Correct for sinc normalization

    return xi


def rescale_template_amplitude(
    p_theory: np.ndarray,
    A_ksz: float,
) -> np.ndarray:
    """
    Rescale theory template by amplitude parameter.

    Parameters
    ----------
    p_theory : np.ndarray
        Fiducial theory template
    A_ksz : float
        Amplitude parameter (1 = fiducial, >1 = enhanced, <1 = suppressed)

    Returns
    -------
    np.ndarray
        Rescaled template
    """
    return A_ksz * p_theory
