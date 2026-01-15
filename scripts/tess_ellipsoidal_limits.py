#!/usr/bin/env python3
"""
tess_ellipsoidal_limits.py - Compute upper limits on ellipsoidal variation amplitude

Uses TESS light curve scatter to constrain binary parameters.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ellipsoidal_amplitude(q, R1_over_a, u=0.6, g=0.3):
    """
    Compute ellipsoidal variation semi-amplitude.

    A_ellip ≈ α × q × (R₁/a)³ × sin²i

    where α depends on limb darkening (u) and gravity darkening (g).

    Symbol table:
    | Symbol  | Definition                              | Units    |
    |---------|----------------------------------------|----------|
    | A_ellip | Ellipsoidal semi-amplitude             | fraction |
    | q       | Mass ratio M₂/M₁                       | -        |
    | R₁      | Primary star radius                    | R_sun    |
    | a       | Orbital semi-major axis                | R_sun    |
    | i       | Orbital inclination                    | rad      |
    | u       | Limb darkening coefficient             | -        |
    | g       | Gravity darkening coefficient          | -        |

    For main sequence star with u ≈ 0.6, g ≈ 0.3:
    α ≈ 0.15 × (15 + u)(1 + g) / (3 - u) ≈ 3.0

    Returns: A_ellip for sin(i) = 1 (edge-on)
    """
    # Morris & Naftilan (1993) approximation
    alpha = 0.15 * (15 + u) * (1 + g) / (3 - u)

    A_ellip = alpha * q * (R1_over_a)**3

    return A_ellip

def kepler_third_law_a(P_days, M1_msun, M2_msun):
    """
    Compute semi-major axis from Kepler's third law.

    a³ = G(M₁+M₂)P² / (4π²)

    Returns: a in solar radii
    """
    G = 6.674e-11  # m³/kg/s²
    M_SUN = 1.989e30  # kg
    R_SUN = 6.957e8  # m
    DAY_S = 86400

    P_s = P_days * DAY_S
    M_tot = (M1_msun + M2_msun) * M_SUN

    a_m = (G * M_tot * P_s**2 / (4 * np.pi**2))**(1/3)
    a_rsun = a_m / R_SUN

    return a_rsun

def tess_scatter_to_amplitude_limit(scatter_ppt, confidence=0.95):
    """
    Convert TESS scatter to upper limit on periodic signal.

    For Gaussian noise, a periodic signal with amplitude A would
    be detected at significance A/σ.

    At 95% confidence (2σ), upper limit is:
    A_max = 2 × scatter / sqrt(N_in_bin)

    For TESS with ~37000 points and ~50 phase bins:
    N_per_bin ≈ 740
    sqrt(N) ≈ 27

    Symbol table:
    | Symbol      | Definition                           | Units |
    |-------------|--------------------------------------|-------|
    | σ_scatter   | RMS scatter in light curve           | ppt   |
    | A_max       | Upper limit on amplitude             | ppt   |
    | N           | Number of points per phase bin       | -     |
    """
    # From analyze_tess_photometry.py output
    N_total = 37832
    N_bins = 50  # Reasonable binning
    N_per_bin = N_total / N_bins

    # 2σ upper limit
    sigma_factor = 2.0 if confidence == 0.95 else 3.0
    A_max = sigma_factor * scatter_ppt / np.sqrt(N_per_bin)

    return A_max

def main():
    print("=" * 70)
    print("TESS ELLIPSOIDAL VARIATION CONSTRAINTS")
    print("=" * 70)
    print()

    # TESS data for Gaia DR3 3802130935635096832
    # From analyze_tess_photometry.py:
    tess_scatter_ppt = 6.32  # parts per thousand
    n_points = 37832

    print("TESS LIGHT CURVE PROPERTIES:")
    print("-" * 50)
    print(f"  Data points:    {n_points}")
    print(f"  Scatter:        {tess_scatter_ppt:.2f} ppt")
    print(f"  Peak LS power:  0.0014 (no significant period)")
    print()

    # Compute upper limit on ellipsoidal amplitude
    A_max_ppt = tess_scatter_to_amplitude_limit(tess_scatter_ppt)
    A_max_frac = A_max_ppt / 1000  # convert to fraction

    print("AMPLITUDE UPPER LIMIT:")
    print("-" * 50)
    print(f"  A_ellip (95% CL) < {A_max_ppt:.3f} ppt = {A_max_frac:.5f}")
    print(f"  A_ellip (99% CL) < {tess_scatter_to_amplitude_limit(tess_scatter_ppt, 0.99):.3f} ppt")
    print()

    # What this rules out
    print("ELLIPSOIDAL AMPLITUDE SCALING:")
    print("-" * 50)
    print("  A_ellip ≈ 3.0 × q × (R₁/a)³ × sin²i")
    print()
    print("  Symbol table:")
    print("  | Symbol | Definition              | Value/Range |")
    print("  |--------|-------------------------|-------------|")
    print("  | q      | Mass ratio M₂/M₁        | ~1-5        |")
    print("  | R₁     | Primary radius          | ~0.5-1 R_sun|")
    print("  | a      | Semi-major axis         | ~20-80 R_sun|")
    print("  | i      | Inclination             | unknown     |")
    print()

    # Compute constraints for grid of parameters
    print("CONSTRAINT TABLE: R₁/a VALUES ALLOWED BY TESS NON-DETECTION")
    print("-" * 70)

    q_values = [0.5, 1.0, 2.0, 3.0, 5.0]

    # For edge-on (sin i = 1), what R1/a is ruled out?
    print(f"For edge-on orbit (sin i = 1), A_max = {A_max_frac:.5f}:")
    print()
    print(f"{'q':<8} {'(R₁/a)_max':<14} {'Comment':<40}")
    print("-" * 62)

    for q in q_values:
        # Solve: 3.0 * q * (R1/a)^3 = A_max
        R1_over_a_max = (A_max_frac / (3.0 * q))**(1/3)

        # For K dwarf, R1 ~ 0.7 R_sun
        # If R1/a < R1_over_a_max, then a > R1/R1_over_a_max
        R1_assumed = 0.7  # R_sun for K dwarf
        a_min = R1_assumed / R1_over_a_max

        comment = f"a > {a_min:.1f} R_sun (for R₁=0.7 R_sun)"
        print(f"{q:<8.1f} {R1_over_a_max:<14.4f} {comment:<40}")

    print()
    print("INTERPRETATION:")
    print("-" * 50)
    print("  The TESS non-detection constrains R₁/a.")
    print("  For typical orbital separations (a ~ 30-60 R_sun),")
    print("  R₁/a ~ 0.01-0.02, giving A_ellip ~ 0.0001-0.001 (0.1-1 ppt)")
    print()
    print("  This is CONSISTENT with non-detection:")
    print("    Expected A_ellip < TESS scatter")
    print()
    print("  TESS DOES NOT rule out any plausible binary configuration")
    print("  for periods > 10 days with non-giant primary.")
    print()

    # What TESS DOES constrain
    print("WHAT TESS DOES CONSTRAIN:")
    print("-" * 50)
    print("  ✓ No deep eclipses (no edge-on transiting system)")
    print("  ✓ No contact binary (would show large ellipsoidal)")
    print("  ✓ No short-period (P < 1 day) close binary")
    print()
    print("  ✗ Does NOT constrain detached binaries with P > 10 days")
    print("  ✗ Does NOT distinguish WD/NS/BH companion types")
    print()

    # Create visualization
    create_constraint_plot(A_max_frac)

def create_constraint_plot(A_max_frac):
    """Create visualization of TESS constraints in q-P space."""

    R1 = 0.7  # R_sun (K dwarf assumption)
    M1 = 0.7  # M_sun

    P_range = np.linspace(5, 100, 50)
    q_range = np.linspace(0.5, 5, 50)

    P_grid, q_grid = np.meshgrid(P_range, q_range)
    A_ellip_grid = np.zeros_like(P_grid)

    for i in range(len(q_range)):
        for j in range(len(P_range)):
            q = q_range[i]
            P = P_range[j]
            M2 = q * M1
            a = kepler_third_law_a(P, M1, M2)
            A_ellip_grid[i, j] = ellipsoidal_amplitude(q, R1/a)

    # Convert to ppt
    A_ellip_ppt = A_ellip_grid * 1000

    fig, ax = plt.subplots(figsize=(10, 7))

    # Contour plot
    levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    cs = ax.contour(P_grid, q_grid, A_ellip_ppt, levels=levels, colors='blue')
    ax.clabel(cs, inline=True, fontsize=10, fmt='%.1f ppt')

    # Mark TESS detection threshold
    threshold_ppt = A_max_frac * 1000
    ax.contour(P_grid, q_grid, A_ellip_ppt, levels=[threshold_ppt],
               colors='red', linewidths=2, linestyles='--')

    # Mark plausible region
    ax.axvspan(25, 80, alpha=0.1, color='green', label='Likely P range')
    ax.axhspan(1.0, 3.0, alpha=0.1, color='orange', label='Likely q range')

    ax.set_xlabel('Orbital Period (days)', fontsize=12)
    ax.set_ylabel('Mass Ratio q = M₂/M₁', fontsize=12)
    ax.set_title(f'Ellipsoidal Amplitude (ppt) for R₁={R1} R$_\\odot$, M₁={M1} M$_\\odot$\n'
                 f'Red dashed: TESS 95% upper limit ({threshold_ppt:.2f} ppt)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tess_ellipsoidal_constraints.png', dpi=150)
    plt.close()
    print("Saved: tess_ellipsoidal_constraints.png")

if __name__ == "__main__":
    main()
