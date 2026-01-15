#!/usr/bin/env python3
"""
roche_geometry.py - Roche geometry and physical stability analysis

Computes:
1. Semi-major axis from Kepler's third law
2. Roche lobe radius using Eggleton approximation
3. Filling factor to verify system is detached
4. Expected ellipsoidal variation amplitude
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Physical constants
G_SI = 6.674e-11      # m^3 kg^-1 s^-2
M_SUN = 1.989e30      # kg
R_SUN = 6.957e8       # m
DAY_S = 86400         # seconds

# Primary star properties (M0 dwarf from LAMOST)
M1_MSUN = 0.5         # M_sun
R1_RSUN = 0.6         # R_sun (typical for M0 dwarf)
R1_RSUN_ERR = 0.1     # uncertainty


def kepler_semi_major_axis(P_days, M1_msun, M2_msun):
    """
    Compute semi-major axis from Kepler's third law.

    a^3 = G(M1 + M2) P^2 / (4 pi^2)

    Returns a in solar radii.
    """
    P_s = P_days * DAY_S
    M_tot = (M1_msun + M2_msun) * M_SUN

    a_m = (G_SI * M_tot * P_s**2 / (4 * np.pi**2))**(1/3)
    a_rsun = a_m / R_SUN

    return a_rsun


def eggleton_roche_lobe(q):
    """
    Eggleton (1983) approximation for Roche lobe radius.

    r_L / a = 0.49 q^(2/3) / [0.6 q^(2/3) + ln(1 + q^(1/3))]

    where q = M_donor / M_accretor (here: q = M1 / M2)

    Returns r_L / a (dimensionless)
    """
    q23 = q**(2/3)
    q13 = q**(1/3)

    r_L_over_a = 0.49 * q23 / (0.6 * q23 + np.log(1 + q13))

    return r_L_over_a


def filling_factor(R1_rsun, a_rsun, q):
    """
    Compute Roche lobe filling factor.

    f = R1 / R_L,1

    f < 1: detached (no mass transfer)
    f ~ 1: contact/semi-detached (mass transfer likely)
    f > 1: overflow (Roche lobe overflow)
    """
    r_L_over_a = eggleton_roche_lobe(q)
    R_L1_rsun = r_L_over_a * a_rsun

    f = R1_rsun / R_L1_rsun

    return f, R_L1_rsun


def ellipsoidal_amplitude(q, R1_over_a, u=0.6, g=0.3):
    """
    Compute ellipsoidal variation semi-amplitude.

    A_ellip ≈ α × q × (R1/a)³ × sin²i

    For u=0.6 (limb darkening), g=0.3 (gravity darkening):
    α ≈ 3.0

    Returns amplitude as a fraction (multiply by 1e6 for ppm).
    """
    alpha = 0.15 * (15 + u) * (1 + g) / (3 - u)
    A_ellip = alpha * q * R1_over_a**3

    return A_ellip


def analyze_roche_geometry():
    """Perform Roche geometry analysis for the candidate."""

    print("=" * 70)
    print("ROCHE GEOMETRY AND PHYSICAL STABILITY")
    print("=" * 70)
    print(f"Target: Gaia DR3 3802130935635096832")
    print()

    results = {
        'primary': {
            'M1_msun': M1_MSUN,
            'R1_rsun': R1_RSUN,
            'R1_rsun_err': R1_RSUN_ERR,
            'spectral_type': 'dM0',
        },
        'configurations': [],
    }

    # Test configurations based on orbit posterior
    # From MCMC: P ranges from ~10-50 days, M2_min from ~1.5-4 M_sun
    test_configs = [
        {'P': 15.9, 'M2': 2.62, 'label': 'Best circular fit'},
        {'P': 10, 'M2': 1.5, 'label': 'Short P, low M2'},
        {'P': 20, 'M2': 2.0, 'label': 'Moderate P'},
        {'P': 30, 'M2': 2.5, 'label': 'Longer P'},
        {'P': 50, 'M2': 3.5, 'label': 'Long P, high M2'},
        {'P': 80, 'M2': 4.5, 'label': 'Very long P'},
    ]

    print("PRIMARY STAR PROPERTIES:")
    print("-" * 50)
    print(f"  Spectral type: dM0 (LAMOST)")
    print(f"  Mass: M1 = {M1_MSUN:.2f} M_sun")
    print(f"  Radius: R1 = {R1_RSUN:.2f} ± {R1_RSUN_ERR:.2f} R_sun")
    print()

    print("ROCHE GEOMETRY FOR DIFFERENT CONFIGURATIONS:")
    print("-" * 70)
    print(f"{'Config':<20} {'P (d)':<8} {'M2 (M☉)':<10} {'a (R☉)':<10} {'R_L1 (R☉)':<12} {'f':<8} {'A_ellip':<10}")
    print("-" * 70)

    for cfg in test_configs:
        P = cfg['P']
        M2 = cfg['M2']
        label = cfg['label']

        # Compute semi-major axis
        a = kepler_semi_major_axis(P, M1_MSUN, M2)

        # Compute Roche lobe radius
        q = M1_MSUN / M2  # q < 1 since M2 > M1
        f, R_L1 = filling_factor(R1_RSUN, a, q)

        # Compute ellipsoidal amplitude
        R1_over_a = R1_RSUN / a
        A_ellip = ellipsoidal_amplitude(M2 / M1_MSUN, R1_over_a)  # Note: q for ellipsoidal is M2/M1
        A_ellip_ppm = A_ellip * 1e6

        print(f"  {label:<20} {P:<8.1f} {M2:<10.2f} {a:<10.1f} {R_L1:<12.2f} {f:<8.3f} {A_ellip_ppm:<10.0f} ppm")

        results['configurations'].append({
            'label': label,
            'P_days': P,
            'M2_msun': M2,
            'a_rsun': a,
            'R_L1_rsun': R_L1,
            'filling_factor': f,
            'A_ellip_ppm': A_ellip_ppm,
            'detached': f < 1.0,
        })

    print()

    # Summary
    print("SUMMARY:")
    print("-" * 50)

    all_detached = all(cfg['filling_factor'] < 1.0 for cfg in results['configurations'])
    min_f = min(cfg['filling_factor'] for cfg in results['configurations'])
    max_f = max(cfg['filling_factor'] for cfg in results['configurations'])
    max_A = max(cfg['A_ellip_ppm'] for cfg in results['configurations'])

    print(f"  Filling factor range: {min_f:.3f} - {max_f:.3f}")
    print(f"  All configurations: {'DETACHED' if all_detached else 'SOME OVERFLOW'}")
    print(f"  Maximum ellipsoidal amplitude: {max_A:.0f} ppm")
    print()

    if all_detached:
        print("  CONCLUSION: The system is DEEPLY DETACHED for all plausible")
        print("  orbital configurations. The primary star is well within its")
        print("  Roche lobe, consistent with the lack of ellipsoidal variations")
        print("  in the TESS light curve.")
    else:
        print("  WARNING: Some configurations show Roche lobe overflow!")
        print("  These can likely be ruled out by the flat TESS light curve.")

    print()

    # TESS detection threshold comparison
    print("TESS DETECTION THRESHOLD COMPARISON:")
    print("-" * 50)

    TESS_SCATTER_PPM = 6320  # From analyze_tess_photometry.py
    TESS_DETECTION_LIMIT = TESS_SCATTER_PPM / np.sqrt(37832 / 50) * 2  # 95% CL, 50 phase bins

    print(f"  TESS scatter: {TESS_SCATTER_PPM:.0f} ppm")
    print(f"  TESS 95% detection limit: ~{TESS_DETECTION_LIMIT:.0f} ppm")
    print()

    detectable = [cfg for cfg in results['configurations'] if cfg['A_ellip_ppm'] > TESS_DETECTION_LIMIT]

    if len(detectable) == 0:
        print("  NONE of the plausible configurations would produce detectable")
        print("  ellipsoidal variations in TESS. The non-detection is EXPECTED.")
    else:
        print(f"  {len(detectable)} configuration(s) would be detectable:")
        for cfg in detectable:
            print(f"    - {cfg['label']}: {cfg['A_ellip_ppm']:.0f} ppm")

    results['tess_comparison'] = {
        'tess_scatter_ppm': TESS_SCATTER_PPM,
        'tess_detection_limit_ppm': TESS_DETECTION_LIMIT,
        'n_detectable': len(detectable),
    }
    print()

    # Create visualization
    create_roche_plot(results)

    # Save results
    with open('roche_geometry_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved: roche_geometry_results.json")

    return results


def create_roche_plot(results):
    """Create visualization of Roche geometry constraints."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract data
    configs = results['configurations']
    P_vals = [c['P_days'] for c in configs]
    f_vals = [c['filling_factor'] for c in configs]
    A_vals = [c['A_ellip_ppm'] for c in configs]
    a_vals = [c['a_rsun'] for c in configs]

    # Panel 1: Semi-major axis vs Period
    ax1 = axes[0]
    P_grid = np.linspace(5, 100, 100)

    for M2, color, label in [(1.5, 'blue', 'M2=1.5'), (2.5, 'green', 'M2=2.5'),
                              (3.5, 'orange', 'M2=3.5'), (4.5, 'red', 'M2=4.5')]:
        a_grid = [kepler_semi_major_axis(P, M1_MSUN, M2) for P in P_grid]
        ax1.plot(P_grid, a_grid, color=color, lw=2, label=f'{label} M☉')

    ax1.axhline(R1_RSUN, color='black', linestyle='--', label=f'R1 = {R1_RSUN} R☉')
    ax1.set_xlabel('Period (days)', fontsize=12)
    ax1.set_ylabel('Semi-major axis (R☉)', fontsize=12)
    ax1.set_title('Orbital Separation', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 150)

    # Panel 2: Filling factor vs Period
    ax2 = axes[1]

    for M2, color, label in [(1.5, 'blue', 'M2=1.5'), (2.5, 'green', 'M2=2.5'),
                              (3.5, 'orange', 'M2=3.5'), (4.5, 'red', 'M2=4.5')]:
        f_grid = []
        for P in P_grid:
            a = kepler_semi_major_axis(P, M1_MSUN, M2)
            q = M1_MSUN / M2
            f, _ = filling_factor(R1_RSUN, a, q)
            f_grid.append(f)
        ax2.plot(P_grid, f_grid, color=color, lw=2, label=f'{label} M☉')

    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Roche limit')
    ax2.axhline(0.5, color='orange', linestyle=':', label='50% filled')
    ax2.set_xlabel('Period (days)', fontsize=12)
    ax2.set_ylabel('Filling factor f = R1/R_L1', fontsize=12)
    ax2.set_title('Roche Lobe Filling', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.5)

    # Panel 3: Ellipsoidal amplitude vs Period
    ax3 = axes[2]

    TESS_LIMIT = results['tess_comparison']['tess_detection_limit_ppm']

    for M2, color, label in [(1.5, 'blue', 'M2=1.5'), (2.5, 'green', 'M2=2.5'),
                              (3.5, 'orange', 'M2=3.5'), (4.5, 'red', 'M2=4.5')]:
        A_grid = []
        for P in P_grid:
            a = kepler_semi_major_axis(P, M1_MSUN, M2)
            A = ellipsoidal_amplitude(M2 / M1_MSUN, R1_RSUN / a) * 1e6
            A_grid.append(A)
        ax3.plot(P_grid, A_grid, color=color, lw=2, label=f'{label} M☉')

    ax3.axhline(TESS_LIMIT, color='purple', linestyle='--', linewidth=2,
                label=f'TESS limit ({TESS_LIMIT:.0f} ppm)')
    ax3.set_xlabel('Period (days)', fontsize=12)
    ax3.set_ylabel('Ellipsoidal amplitude (ppm)', fontsize=12)
    ax3.set_title('Expected Ellipsoidal Variation', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.set_ylim(1, 10000)

    plt.tight_layout()
    plt.savefig('roche_geometry_plot.png', dpi=150)
    plt.close()
    print("Saved: roche_geometry_plot.png")


if __name__ == "__main__":
    analyze_roche_geometry()
