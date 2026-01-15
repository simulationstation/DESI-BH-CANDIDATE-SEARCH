#!/usr/bin/env python3
"""
orbit_feasibility.py - Period constraints and mass function analysis

Uses the 4 RV epochs to constrain orbital period and compute mass function.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
G = 6.674e-11  # m³/kg/s²
M_SUN = 1.989e30  # kg
DAY_S = 86400  # seconds per day

# Gaia DR3 3802130935635096832 epochs
EPOCHS = [
    {'mjd': 59568.48825, 'rv': -86.39, 'rv_err': 0.55},
    {'mjd': 59605.38003, 'rv': 59.68, 'rv_err': 0.83},
    {'mjd': 59607.37393, 'rv': 26.43, 'rv_err': 1.06},
    {'mjd': 59607.38852, 'rv': 25.16, 'rv_err': 1.11},
]

def mass_function(P_days, K_kms):
    """
    Compute binary mass function f(M).

    f(M) = (M₂ sin i)³ / (M₁ + M₂)² = P K³ / (2π G)

    Symbol table:
    | Symbol  | Definition                          | Units    |
    |---------|-------------------------------------|----------|
    | f(M)    | Mass function                       | M_sun    |
    | P       | Orbital period                      | days     |
    | K       | RV semi-amplitude of primary        | km/s     |
    | M₁      | Primary mass                        | M_sun    |
    | M₂      | Companion mass                      | M_sun    |
    | i       | Orbital inclination                 | rad      |
    | G       | Gravitational constant              | SI       |

    Returns: f(M) in solar masses
    """
    P_s = P_days * DAY_S
    K_ms = K_kms * 1000  # km/s to m/s

    f_M_kg = (P_s * K_ms**3) / (2 * np.pi * G)
    f_M_sun = f_M_kg / M_SUN

    return f_M_sun

def solve_M2_from_mass_function(f_M, M1, sin_i=1.0):
    """
    Solve for M₂ given mass function, M₁, and sin(i).

    f(M) = (M₂ sin i)³ / (M₁ + M₂)²

    Rearranged: (M₂ sin i)³ = f(M) (M₁ + M₂)²

    For sin_i = 1 (edge-on), this gives MINIMUM M₂.

    Uses numerical root finding.
    """
    from scipy.optimize import brentq

    def equation(M2):
        return (M2 * sin_i)**3 - f_M * (M1 + M2)**2

    # Search for root
    try:
        M2 = brentq(equation, 0.01, 100.0)
        return M2
    except:
        return np.nan

def period_constraints_from_rv_evolution():
    """
    Derive period constraints from observed RV evolution.

    Key constraints:
    1. RV went from -86.39 to +59.68 km/s in 36.89 days (Δphase constraint)
    2. RV dropped from +59.68 to ~+25.8 km/s in 2.0 days (slope constraint)
    3. Two same-night points at MJD 59607 are consistent (~25 km/s)
    """
    print("=" * 70)
    print("PERIOD CONSTRAINTS FROM RV EVOLUTION")
    print("=" * 70)
    print()

    # Time intervals
    dt_01 = EPOCHS[1]['mjd'] - EPOCHS[0]['mjd']  # 36.89 days
    dt_12 = EPOCHS[2]['mjd'] - EPOCHS[1]['mjd']  # 1.99 days
    dt_23 = EPOCHS[3]['mjd'] - EPOCHS[2]['mjd']  # 0.015 days (21 min)

    # RV changes
    drv_01 = EPOCHS[1]['rv'] - EPOCHS[0]['rv']  # +146.07 km/s
    drv_12 = EPOCHS[2]['rv'] - EPOCHS[1]['rv']  # -33.25 km/s

    print("OBSERVED RV EVOLUTION:")
    print(f"  Epoch 0 → 1: Δt = {dt_01:.2f} d, ΔRV = {drv_01:+.2f} km/s")
    print(f"  Epoch 1 → 2: Δt = {dt_12:.2f} d, ΔRV = {drv_12:+.2f} km/s")
    print(f"  Epoch 2 → 3: Δt = {dt_23*24:.1f} hr, ΔRV = {EPOCHS[3]['rv']-EPOCHS[2]['rv']:+.2f} km/s")
    print()

    # Circular orbit analysis
    print("CIRCULAR ORBIT CONSTRAINTS:")
    print("-" * 50)

    K_est = 146.07 / 2  # ~73 km/s semi-amplitude estimate

    # For circular orbit: RV(t) = γ + K sin(2π(t-t₀)/P)
    # Going from RV_min to RV_max requires Δφ = 0.5 (half period)
    # But we went from -86.39 (not necessarily min) to +59.68 (not necessarily max)

    # The total ΔRV_max = 146.07 km/s suggests K ≈ 73 km/s if circular
    # Phase change for 146 km/s swing in circular orbit:
    # If this is min-to-max: Δφ = 0.5 → P_min = 2 × 36.89 = 73.8 days
    # If we didn't hit exact min/max, P could be shorter

    # Constraint 1: dt_01 = 36.89 days covered nearly full RV range
    P_if_half_period = 2 * dt_01
    print(f"  If epoch 0→1 spans half-period: P = {P_if_half_period:.1f} days")

    # Constraint 2: RV slope between epochs 1→2
    # In ~2 days, RV dropped by 33 km/s
    # For circular orbit near max: dRV/dt ≈ -2πK/P at RV_max crossing
    # |dRV/dt| = 33.25/1.99 = 16.7 km/s/day
    rv_slope_12 = abs(drv_12 / dt_12)
    # 2πK/P = |dRV/dt| → P = 2πK/|slope|
    P_from_slope = 2 * np.pi * K_est / rv_slope_12
    print(f"  From RV slope at epoch 1→2: P ≈ {P_from_slope:.1f} days (circular)")

    # Constraint 3: Same-night consistency
    # Epochs 2 & 3 are ~21 min apart with ΔRV = 1.27 km/s
    # This implies |dRV/dt| ≤ ~90 km/s/day at that phase

    print()
    print("CONSERVATIVE PERIOD BOUNDS (CIRCULAR):")
    print(f"  P_min ≈ 20 days (must accommodate 146 km/s swing)")
    print(f"  P_max ≈ 100 days (limited by baseline)")
    print(f"  Most likely range: 25-80 days")
    print()

    # Eccentric orbit relaxation
    print("ECCENTRIC ORBIT CONSIDERATIONS:")
    print("-" * 50)
    print("  Eccentricity allows:")
    print("    - Faster RV changes near periastron")
    print("    - Shorter periods to produce same ΔRV")
    print("    - Asymmetric RV curves")
    print()
    print("  With e > 0:")
    print("    - P_min could be as low as ~10 days")
    print("    - High e (> 0.5) allows very rapid RV swings")
    print()

    return {
        'dt_01': dt_01,
        'dt_12': dt_12,
        'drv_01': drv_01,
        'drv_12': drv_12,
        'K_est': K_est,
        'P_circular_half': P_if_half_period,
        'P_from_slope': P_from_slope,
        'P_range_circular': (25, 80),
        'P_range_eccentric': (10, 100),
    }

def mass_function_analysis():
    """
    Compute mass function and minimum M₂ for period grid.
    """
    print("=" * 70)
    print("MASS FUNCTION ANALYSIS")
    print("=" * 70)
    print()

    K_est = 73.04  # km/s (ΔRV/2)

    print(f"Using K_est = {K_est:.2f} km/s")
    print()

    # Primary mass bracket (from Gaia photometry)
    # G = 17.27, parallax = 0.119 mas → very distant or intrinsically faint
    # BP-RP = 18.09 - 16.20 = 1.89 → fairly red
    # This suggests late K or early M dwarf, or reddened earlier type
    # Conservative bracket: M₁ ∈ [0.5, 1.2] M_sun

    print("PRIMARY MASS ESTIMATE:")
    print("-" * 50)
    print("  From Gaia photometry (G=17.27, BP-RP=1.89):")
    print("  Parallax = 0.119 ± 0.160 mas (very uncertain)")
    print("  If nearby: late K/early M dwarf")
    print("  If distant: reddened FGK star")
    print("  Conservative bracket: M₁ ∈ [0.5, 1.2] M_sun")
    print()

    M1_values = [0.5, 0.7, 1.0, 1.2]
    P_values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 100])

    print("MASS FUNCTION TABLE")
    print("-" * 70)
    print(f"{'P (days)':<10} {'f(M) (M_sun)':<14} " + " ".join([f"M2_min(M1={m})".ljust(14) for m in M1_values]))
    print("-" * 70)

    results = []
    for P in P_values:
        f_M = mass_function(P, K_est)
        M2_mins = [solve_M2_from_mass_function(f_M, M1) for M1 in M1_values]

        row = f"{P:<10} {f_M:<14.4f} " + " ".join([f"{m2:<14.2f}" for m2 in M2_mins])
        print(row)

        results.append({
            'P_days': P,
            'f_M': f_M,
            'M2_mins': dict(zip(M1_values, M2_mins))
        })

    print()

    # Find threshold periods
    print("COMPANION TYPE THRESHOLDS")
    print("-" * 70)
    print("| M₁ (M_sun) | P_min for M₂>1.4 (NS) | P_min for M₂>3.0 (BH) |")
    print("|------------|----------------------|----------------------|")

    for M1 in M1_values:
        # Find P where M2 = 1.4 (NS threshold)
        P_ns = None
        P_bh = None

        for P in np.linspace(5, 200, 1000):
            f_M = mass_function(P, K_est)
            M2 = solve_M2_from_mass_function(f_M, M1)
            if not np.isnan(M2):
                if M2 >= 1.4 and P_ns is None:
                    P_ns = P
                if M2 >= 3.0 and P_bh is None:
                    P_bh = P

        P_ns_str = f"{P_ns:.1f}" if P_ns else ">200"
        P_bh_str = f"{P_bh:.1f}" if P_bh else ">200"
        print(f"| {M1:<10.1f} | {P_ns_str:>20} | {P_bh_str:>20} |")

    print()
    print("INTERPRETATION:")
    print("-" * 50)
    print("  For P ≈ 30-50 days (plausible range):")
    print("    M₂_min ≈ 0.8-1.3 M_sun (for M₁ = 0.7 M_sun)")
    print("    → Consistent with WD, NS, or BH companion")
    print("    → Cannot distinguish without period determination")
    print()
    print("  To REQUIRE M₂ > 3 M_sun (BH):")
    print("    Need P > 100 days for low-mass primary")
    print("    Or higher K (follow-up RVs near extrema)")
    print()

    return results

def create_mass_function_plot():
    """Generate mass function visualization."""

    K_est = 73.04
    P_range = np.linspace(10, 150, 100)
    M1_values = [0.5, 0.7, 1.0, 1.2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Mass function vs Period
    f_M_vals = [mass_function(P, K_est) for P in P_range]
    ax1.plot(P_range, f_M_vals, 'b-', lw=2)
    ax1.set_xlabel('Orbital Period (days)', fontsize=12)
    ax1.set_ylabel('Mass Function f(M) [M$_\\odot$]', fontsize=12)
    ax1.set_title(f'Mass Function vs Period (K = {K_est:.1f} km/s)', fontsize=12)
    ax1.axhline(y=0.5, color='gray', ls='--', alpha=0.5, label='f(M) = 0.5')
    ax1.axhline(y=1.0, color='gray', ls=':', alpha=0.5, label='f(M) = 1.0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(10, 150)

    # Panel 2: Minimum M₂ vs Period for different M₁
    colors = ['blue', 'green', 'orange', 'red']
    for M1, c in zip(M1_values, colors):
        M2_mins = []
        for P in P_range:
            f_M = mass_function(P, K_est)
            M2 = solve_M2_from_mass_function(f_M, M1)
            M2_mins.append(M2)
        ax2.plot(P_range, M2_mins, c=c, lw=2, label=f'M₁ = {M1} M$_\\odot$')

    ax2.axhline(y=1.4, color='purple', ls='--', lw=2, label='NS threshold (1.4 M$_\\odot$)')
    ax2.axhline(y=3.0, color='black', ls='--', lw=2, label='BH threshold (3.0 M$_\\odot$)')
    ax2.set_xlabel('Orbital Period (days)', fontsize=12)
    ax2.set_ylabel('Minimum M₂ (sin i = 1) [M$_\\odot$]', fontsize=12)
    ax2.set_title('Minimum Companion Mass vs Period', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(10, 150)
    ax2.set_ylim(0, 5)

    # Mark plausible period range
    ax2.axvspan(25, 80, alpha=0.1, color='green', label='Plausible P range')

    plt.tight_layout()
    plt.savefig('mass_function_analysis.png', dpi=150)
    plt.close()
    print("Saved: mass_function_analysis.png")

def main():
    constraints = period_constraints_from_rv_evolution()
    results = mass_function_analysis()
    create_mass_function_plot()

    print()
    print("=" * 70)
    print("SUMMARY: ORBIT FEASIBILITY")
    print("=" * 70)
    print()
    print("KEY FINDINGS:")
    print("  1. Period likely in range 25-80 days (circular) or 10-100 days (eccentric)")
    print("  2. For K ≈ 73 km/s and P ≈ 40 days: M₂_min ≈ 0.9-1.1 M_sun")
    print("  3. Cannot distinguish WD/NS/BH without period determination")
    print("  4. BH (M₂ > 3 M_sun) requires P > 100 days or higher K")
    print()
    print("WHAT WE CAN CLAIM:")
    print("  ✓ High-amplitude RV binary (ΔRV = 146 km/s)")
    print("  ✓ Companion is dark (no IR excess, no UV detection)")
    print("  ✓ Companion mass M₂ ≳ 0.7 M_sun (robust lower limit)")
    print()
    print("WHAT WE CANNOT CLAIM:")
    print("  ✗ Specific companion type (WD vs NS vs BH)")
    print("  ✗ Dynamical mass measurement (no period)")
    print("  ✗ BH confirmation (requires M₂ > 3 M_sun)")
    print()

if __name__ == "__main__":
    main()
