#!/usr/bin/env python3
"""
distance_analysis.py - Distance tension analysis

Compares:
1. Spectrophotometric distance from LAMOST dM0 classification
2. Gaia DR3 parallax distance

Quantifies tension and explains RUWE implications.
"""

import numpy as np
import json

# Gaia DR3 data for 3802130935635096832
GAIA_DATA = {
    'source_id': 3802130935635096832,
    'ra': 164.5235,
    'dec': -1.6602,
    'parallax': 0.119,          # mas
    'parallax_error': 0.160,    # mas
    'G': 17.272505,
    'BP': 18.087648,
    'RP': 16.199726,
    'bp_rp': 1.888,             # BP - RP color
    'ruwe': 1.9535599946975708,
    'astrometric_excess_noise': 0.532,  # mas
    'astrometric_excess_noise_sig': 16.49,
}

# LAMOST classification
LAMOST_DATA = {
    'spectral_type': 'dM0',
    'class': 'STAR',
    'rv_kms': -49.36,
    'rv_err_kms': 2.79,
}

# M0 dwarf reference values (from Pecaut & Mamajek 2013, updated)
M0_DWARF = {
    'Teff': 3850,       # K
    'M_G': 8.8,         # Absolute G magnitude (range 8.5-9.2)
    'M_G_err': 0.4,     # uncertainty
    'BC_G': -0.5,       # Bolometric correction
    'R_star': 0.62,     # R_sun
    'M_star': 0.57,     # M_sun (range 0.5-0.6)
    'M_star_err': 0.05,
}


def parallax_to_distance(parallax_mas, parallax_err_mas):
    """
    Convert parallax to distance with error propagation.

    d = 1000 / parallax [pc]

    For parallax with large fractional error, use Bayesian inference
    with exponentially decreasing space density prior.
    """
    if parallax_mas <= 0:
        return np.nan, np.nan, "Negative or zero parallax"

    # Simple inversion (valid for small fractional errors)
    d_pc = 1000 / parallax_mas
    d_err_pc = d_pc * (parallax_err_mas / parallax_mas)

    # Check if parallax is significant
    snr = parallax_mas / parallax_err_mas

    if snr < 2:
        note = f"Low SNR parallax ({snr:.2f}); distance poorly constrained"
    elif snr < 5:
        note = f"Moderate SNR parallax ({snr:.2f}); distance uncertain"
    else:
        note = f"Good SNR parallax ({snr:.2f})"

    return d_pc, d_err_pc, note


def spectrophotometric_distance(G_app, M_G, M_G_err, A_G=0):
    """
    Compute spectrophotometric distance from apparent and absolute magnitude.

    m - M = 5 * log10(d/10pc) + A

    d = 10^((m - M - A + 5) / 5) [pc]
    """
    dist_mod = G_app - M_G - A_G
    d_pc = 10 ** ((dist_mod + 5) / 5)

    # Error propagation (dominant error from M_G uncertainty)
    d_err_pc = d_pc * M_G_err * np.log(10) / 5

    return d_pc, d_err_pc


def analyze_distance_tension():
    """Perform full distance tension analysis."""

    print("=" * 70)
    print("DISTANCE TENSION ANALYSIS")
    print("=" * 70)
    print(f"Target: Gaia DR3 {GAIA_DATA['source_id']}")
    print()

    results = {}

    # 1. Gaia parallax distance
    print("1. GAIA DR3 PARALLAX DISTANCE")
    print("-" * 50)

    plx = GAIA_DATA['parallax']
    plx_err = GAIA_DATA['parallax_error']

    print(f"  Parallax: {plx:.3f} ± {plx_err:.3f} mas")
    print(f"  Parallax SNR: {plx/plx_err:.2f}")

    d_plx, d_plx_err, note = parallax_to_distance(plx, plx_err)
    print(f"  Distance (naive): {d_plx:.0f} ± {d_plx_err:.0f} pc")
    print(f"  Note: {note}")
    print()

    # The parallax is consistent with infinity at 1σ
    if plx - plx_err <= 0:
        print("  WARNING: Parallax is consistent with zero (d → ∞) at 1σ!")
        print("  The Gaia parallax does not meaningfully constrain the distance.")

    results['parallax'] = {
        'parallax_mas': plx,
        'parallax_err_mas': plx_err,
        'parallax_snr': plx / plx_err,
        'distance_pc': d_plx,
        'distance_err_pc': d_plx_err,
        'note': note,
    }
    print()

    # 2. Spectrophotometric distance
    print("2. SPECTROPHOTOMETRIC DISTANCE (LAMOST dM0)")
    print("-" * 50)

    G_app = GAIA_DATA['G']
    M_G = M0_DWARF['M_G']
    M_G_err = M0_DWARF['M_G_err']

    print(f"  LAMOST spectral type: {LAMOST_DATA['spectral_type']}")
    print(f"  Apparent G: {G_app:.3f}")
    print(f"  M0 dwarf M_G: {M_G:.1f} ± {M_G_err:.1f}")

    d_spec, d_spec_err = spectrophotometric_distance(G_app, M_G, M_G_err)
    print(f"  Spectrophotometric distance: {d_spec:.0f} ± {d_spec_err:.0f} pc")
    print()

    # Check for extinction
    bp_rp = GAIA_DATA['bp_rp']
    bp_rp_M0 = 2.0  # Expected BP-RP for M0 dwarf
    E_bp_rp = bp_rp - bp_rp_M0
    print(f"  Observed BP-RP: {bp_rp:.3f}")
    print(f"  Expected BP-RP (M0): ~{bp_rp_M0:.1f}")
    print(f"  Color excess E(BP-RP): {E_bp_rp:.3f}")

    if E_bp_rp < -0.2:
        print("  Note: Star is BLUER than expected for M0. Could be:")
        print("        - Earlier spectral type (K dwarf contamination)")
        print("        - Companion contributing blue light")
        print("        - Classification uncertainty")
    elif E_bp_rp > 0.2:
        print("  Note: Star is REDDER than expected. Possible extinction.")
        A_G_est = E_bp_rp * 1.5  # Rough conversion
        d_spec_ext, d_spec_ext_err = spectrophotometric_distance(G_app, M_G, M_G_err, A_G=A_G_est)
        print(f"  If A_G ~ {A_G_est:.2f}: d ~ {d_spec_ext:.0f} pc")

    results['spectrophotometric'] = {
        'spectral_type': LAMOST_DATA['spectral_type'],
        'G_app': G_app,
        'M_G': M_G,
        'M_G_err': M_G_err,
        'distance_pc': d_spec,
        'distance_err_pc': d_spec_err,
        'bp_rp_observed': bp_rp,
        'bp_rp_expected': bp_rp_M0,
    }
    print()

    # 3. Distance tension
    print("3. DISTANCE TENSION")
    print("-" * 50)

    # Compare distances
    if np.isfinite(d_plx) and d_plx_err > 0:
        tension_sigma = abs(d_plx - d_spec) / np.sqrt(d_plx_err**2 + d_spec_err**2)
        print(f"  Parallax distance: {d_plx:.0f} ± {d_plx_err:.0f} pc")
        print(f"  Spectrophotometric: {d_spec:.0f} ± {d_spec_err:.0f} pc")
        print(f"  Tension: {tension_sigma:.1f}σ")

        if tension_sigma < 1:
            print("  Status: CONSISTENT")
        elif tension_sigma < 2:
            print("  Status: MILD TENSION")
        else:
            print("  Status: SIGNIFICANT TENSION")
    else:
        print("  Parallax distance effectively unconstrained.")
        print(f"  Spectrophotometric: {d_spec:.0f} ± {d_spec_err:.0f} pc (ADOPTED)")
        tension_sigma = np.nan

    results['tension'] = {
        'd_parallax_pc': d_plx,
        'd_spectrophot_pc': d_spec,
        'tension_sigma': tension_sigma if np.isfinite(tension_sigma) else None,
    }
    print()

    # 4. RUWE and astrometric noise
    print("4. RUWE AND ASTROMETRIC EXCESS NOISE")
    print("-" * 50)

    ruwe = GAIA_DATA['ruwe']
    aen = GAIA_DATA['astrometric_excess_noise']
    aen_sig = GAIA_DATA['astrometric_excess_noise_sig']

    print(f"  RUWE: {ruwe:.4f}")
    print(f"  Astrometric Excess Noise: {aen:.3f} mas")
    print(f"  AEN Significance: {aen_sig:.2f}")
    print()

    if ruwe > 1.4:
        print("  RUWE > 1.4 indicates POOR astrometric fit.")
        print("  Possible causes:")
        print("    - Unresolved binary (photocenter wobble)")
        print("    - Source confusion/blending")
        print("    - Extended source")
        print()
        print("  For this target:")
        print("    - Legacy Survey shows isolated point source (no blending)")
        print("    - High RV variability suggests binary")
        print("    - RUWE consistent with orbital photocenter motion")

    if aen_sig > 2:
        print()
        print(f"  AEN significance ({aen_sig:.1f}) >> 2 confirms astrometric anomaly.")

    results['astrometry'] = {
        'ruwe': ruwe,
        'aen_mas': aen,
        'aen_sig': aen_sig,
    }
    print()

    # 5. Expected photocenter wobble
    print("5. EXPECTED PHOTOCENTER WOBBLE")
    print("-" * 50)

    # For a binary, the photocenter wobble is:
    # a_phot = a * q / (1 + q) * (L2 - L1) / (L1 + L2)
    # For dark companion (L2 << L1): a_phot ≈ a * q / (1 + q) = a * M2 / (M1 + M2)

    # From MCMC best fit: P ~ 16 d, M1 ~ 0.5, M2 ~ 2.6 M_sun
    P_days = 16.0
    M1 = 0.5
    M2 = 2.6
    d_pc = d_spec  # Use spectrophotometric distance

    # Semi-major axis from Kepler's third law
    G = 6.674e-11
    M_SUN = 1.989e30
    AU_M = 1.496e11
    DAY_S = 86400

    P_s = P_days * DAY_S
    M_tot = (M1 + M2) * M_SUN
    a_m = (G * M_tot * P_s**2 / (4 * np.pi**2))**(1/3)
    a_AU = a_m / AU_M

    # Primary's orbit around center of mass
    a1_AU = a_AU * M2 / (M1 + M2)

    # Photocenter wobble (dark companion, all light from primary)
    # a_phot = a1 (for completely dark companion)
    a_phot_AU = a1_AU
    a_phot_mas = a_phot_AU / d_pc * 1000  # mas

    print(f"  Assumed parameters:")
    print(f"    P = {P_days:.0f} days")
    print(f"    M1 = {M1:.2f} M_sun (M0 dwarf)")
    print(f"    M2 = {M2:.1f} M_sun (dark companion)")
    print(f"    d = {d_pc:.0f} pc (spectrophotometric)")
    print()
    print(f"  Orbital semi-major axis: a = {a_AU:.4f} AU = {a_m/6.957e8:.1f} R_sun")
    print(f"  Primary's orbit: a1 = {a1_AU:.4f} AU")
    print(f"  Expected photocenter wobble: {a_phot_mas:.2f} mas")
    print()

    if a_phot_mas > 0.1:
        print(f"  The expected wobble ({a_phot_mas:.2f} mas) is SIGNIFICANT")
        print(f"  compared to Gaia's per-observation precision (~0.1-0.3 mas for G~17).")
        print(f"  This explains the elevated RUWE ({ruwe:.2f}) and AEN ({aen:.2f} mas).")
    else:
        print(f"  The expected wobble ({a_phot_mas:.2f} mas) is small.")

    results['photocenter_wobble'] = {
        'P_days': P_days,
        'M1_msun': M1,
        'M2_msun': M2,
        'd_pc': d_pc,
        'a_AU': a_AU,
        'a1_AU': a1_AU,
        'a_phot_mas': a_phot_mas,
    }
    print()

    # 6. Adopted distance
    print("6. ADOPTED DISTANCE")
    print("-" * 50)

    print(f"  Given the unreliable Gaia parallax (SNR = {plx/plx_err:.2f}),")
    print(f"  we adopt the spectrophotometric distance:")
    print()
    print(f"  d = {d_spec:.0f} ± {d_spec_err:.0f} pc")
    print()
    print(f"  This is anchored by the LAMOST dM0 classification and")
    print(f"  M0 dwarf absolute magnitude M_G = {M_G:.1f} ± {M_G_err:.1f}.")

    results['adopted_distance'] = {
        'd_pc': d_spec,
        'd_err_pc': d_spec_err,
        'method': 'spectrophotometric (LAMOST dM0)',
    }
    print()

    # 7. Summary for paper
    print("=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print()
    print("  The Gaia DR3 parallax (0.119 ± 0.160 mas) has SNR < 1 and")
    print("  is consistent with zero, providing no meaningful distance")
    print("  constraint. We adopt a spectrophotometric distance of")
    print(f"  d = {d_spec:.0f} ± {d_spec_err:.0f} pc based on the LAMOST dM0")
    print("  classification and standard M0 dwarf absolute magnitudes.")
    print()
    print("  The elevated RUWE (1.95) and astrometric excess noise")
    print("  (0.53 mas, 16.5σ) are consistent with unmodeled orbital")
    print(f"  photocenter motion. For a {M2:.1f} M_sun dark companion at")
    print(f"  P ~ {P_days:.0f} days, the expected wobble is ~{a_phot_mas:.1f} mas,")
    print("  sufficient to corrupt the 5-parameter astrometric solution")
    print("  but not detectable as a full orbital solution in DR3.")
    print()

    # Save results
    with open('distance_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved: distance_analysis_results.json")

    return results


if __name__ == "__main__":
    analyze_distance_tension()
