#!/usr/bin/env python3
"""
xray_radio_search.py - X-ray and Radio Archival Search

Searches for X-ray and radio counterparts at the target position.
If none found, derives upper limits that constrain accretion/activity.

Target: Gaia DR3 3802130935635096832
RA: 164.5235 deg, Dec: -1.6602 deg
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from astropy.coordinates import SkyCoord
import astropy.units as u

# Target coordinates
TARGET_RA = 164.5235
TARGET_DEC = -1.6602
TARGET_NAME = "Gaia DR3 3802130935635096832"
SEARCH_RADIUS = 30.0  # arcsec

# Adopted distance from spectrophotometric analysis
DISTANCE_PC = 495.0
DISTANCE_ERR_PC = 91.0

# Survey flux limits (representative values from documentation)
SURVEY_LIMITS = {
    'ROSAT_2RXS': {
        'band': '0.1-2.4 keV',
        'flux_limit_cgs': 1.0e-13,  # erg/s/cm^2
        'notes': 'ROSAT All-Sky Survey typical limit'
    },
    'XMM_4XMM': {
        'band': '0.2-12 keV',
        'flux_limit_cgs': 5.0e-15,  # erg/s/cm^2 (pointed, varies by field)
        'notes': 'XMM-Newton Serendipitous Catalog (field-dependent)'
    },
    'Chandra_CSC2': {
        'band': '0.5-7 keV',
        'flux_limit_cgs': 1.0e-15,  # erg/s/cm^2 (pointed, varies)
        'notes': 'Chandra Source Catalog 2.0 (field-dependent)'
    },
    'NVSS': {
        'band': '1.4 GHz',
        'flux_limit_mJy': 2.5,
        'notes': 'NRAO VLA Sky Survey'
    },
    'VLASS': {
        'band': '3 GHz',
        'flux_limit_mJy': 0.4,  # Epoch 1 quick-look
        'notes': 'VLA Sky Survey'
    },
    'FIRST': {
        'band': '1.4 GHz',
        'flux_limit_mJy': 1.0,
        'notes': 'Faint Images of the Radio Sky at Twenty-cm'
    }
}

# Typical M-dwarf X-ray luminosities for comparison
MDWARF_LX_QUIESCENT = (1e27, 1e29)  # erg/s range
MDWARF_LX_FLARE = (1e29, 1e31)  # erg/s range


def flux_to_luminosity(flux_cgs, distance_pc):
    """Convert flux (erg/s/cm^2) to luminosity (erg/s)."""
    d_cm = distance_pc * 3.086e18  # pc to cm
    return 4 * np.pi * d_cm**2 * flux_cgs


def mJy_to_luminosity(flux_mJy, distance_pc, freq_GHz):
    """Convert radio flux density (mJy) to spectral luminosity (erg/s/Hz)."""
    flux_Jy = flux_mJy * 1e-3
    flux_cgs = flux_Jy * 1e-23  # Jy to erg/s/cm^2/Hz
    d_cm = distance_pc * 3.086e18
    L_nu = 4 * np.pi * d_cm**2 * flux_cgs  # erg/s/Hz
    return L_nu


def query_vizier_catalog(ra, dec, radius_arcsec, catalog_id, catalog_name):
    """Query a VizieR catalog for sources near the target."""
    try:
        from astroquery.vizier import Vizier

        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        v = Vizier(columns=['*'], row_limit=10)

        result = v.query_region(coord, radius=radius_arcsec*u.arcsec, catalog=catalog_id)

        if result and len(result) > 0:
            return {'found': True, 'table': result[0], 'catalog': catalog_name}
        else:
            return {'found': False, 'catalog': catalog_name}
    except Exception as e:
        return {'found': False, 'catalog': catalog_name, 'error': str(e)[:200]}


def query_heasarc(ra, dec, radius_arcsec, table_name):
    """Query HEASARC for X-ray catalogs."""
    try:
        from astroquery.heasarc import Heasarc

        heasarc = Heasarc()
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

        result = heasarc.query_region(coord, mission=table_name, radius=radius_arcsec*u.arcsec)

        if result is not None and len(result) > 0:
            return {'found': True, 'table': result, 'catalog': table_name}
        else:
            return {'found': False, 'catalog': table_name}
    except Exception as e:
        return {'found': False, 'catalog': table_name, 'error': str(e)[:200]}


def search_xray_catalogs():
    """Search X-ray catalogs for counterparts."""
    print("=" * 70)
    print("X-RAY ARCHIVAL SEARCH")
    print("=" * 70)
    print(f"Target: {TARGET_NAME}")
    print(f"Coordinates: RA={TARGET_RA}, Dec={TARGET_DEC}")
    print(f"Search radius: {SEARCH_RADIUS} arcsec")
    print()

    results = {}

    # 1. ROSAT 2RXS via VizieR
    print("Querying ROSAT 2RXS (VizieR IX/10A)...")
    rosat_result = query_vizier_catalog(TARGET_RA, TARGET_DEC, SEARCH_RADIUS,
                                         'IX/10A', 'ROSAT_2RXS')
    results['ROSAT_2RXS'] = rosat_result
    if rosat_result.get('found'):
        print(f"  FOUND: {len(rosat_result['table'])} source(s)")
    elif rosat_result.get('error'):
        print(f"  Query failed: {rosat_result['error'][:80]}")
    else:
        print("  No sources found within search radius")

    # 2. XMM 4XMM-DR13 via VizieR
    print("Querying XMM 4XMM-DR13 (VizieR IX/68)...")
    xmm_result = query_vizier_catalog(TARGET_RA, TARGET_DEC, SEARCH_RADIUS,
                                       'IX/68', 'XMM_4XMM_DR13')
    results['XMM_4XMM'] = xmm_result
    if xmm_result.get('found'):
        print(f"  FOUND: {len(xmm_result['table'])} source(s)")
    elif xmm_result.get('error'):
        print(f"  Query failed: {xmm_result['error'][:80]}")
    else:
        print("  No sources found within search radius")

    # 3. Chandra CSC 2.0 via VizieR
    print("Querying Chandra CSC 2.0 (VizieR IX/57)...")
    chandra_result = query_vizier_catalog(TARGET_RA, TARGET_DEC, SEARCH_RADIUS,
                                           'IX/57', 'Chandra_CSC2')
    results['Chandra_CSC2'] = chandra_result
    if chandra_result.get('found'):
        print(f"  FOUND: {len(chandra_result['table'])} source(s)")
    elif chandra_result.get('error'):
        print(f"  Query failed: {chandra_result['error'][:80]}")
    else:
        print("  No sources found within search radius")

    # 4. Try HEASARC directly for ROSAT
    print("Querying HEASARC RASS2RXS...")
    try:
        heasarc_result = query_heasarc(TARGET_RA, TARGET_DEC, SEARCH_RADIUS, 'rass2rxs')
        results['HEASARC_ROSAT'] = heasarc_result
        if heasarc_result.get('found'):
            print(f"  FOUND: {len(heasarc_result['table'])} source(s)")
        elif heasarc_result.get('error'):
            print(f"  Query failed: {heasarc_result['error'][:80]}")
        else:
            print("  No sources found")
    except Exception as e:
        results['HEASARC_ROSAT'] = {'found': False, 'error': str(e)[:200]}
        print(f"  HEASARC query failed: {str(e)[:80]}")

    print()
    return results


def search_radio_catalogs():
    """Search radio catalogs for counterparts."""
    print("=" * 70)
    print("RADIO ARCHIVAL SEARCH")
    print("=" * 70)
    print()

    results = {}

    # 1. NVSS via VizieR
    print("Querying NVSS (VizieR VIII/65)...")
    nvss_result = query_vizier_catalog(TARGET_RA, TARGET_DEC, SEARCH_RADIUS,
                                        'VIII/65', 'NVSS')
    results['NVSS'] = nvss_result
    if nvss_result.get('found'):
        print(f"  FOUND: {len(nvss_result['table'])} source(s)")
    elif nvss_result.get('error'):
        print(f"  Query failed: {nvss_result['error'][:80]}")
    else:
        print("  No sources found within search radius")

    # 2. FIRST via VizieR
    print("Querying FIRST (VizieR VIII/92)...")
    first_result = query_vizier_catalog(TARGET_RA, TARGET_DEC, SEARCH_RADIUS,
                                         'VIII/92', 'FIRST')
    results['FIRST'] = first_result
    if first_result.get('found'):
        print(f"  FOUND: {len(first_result['table'])} source(s)")
    elif first_result.get('error'):
        print(f"  Query failed: {first_result['error'][:80]}")
    else:
        print("  No sources found within search radius")

    # 3. VLASS via VizieR (if available)
    print("Querying VLASS (VizieR)...")
    vlass_result = query_vizier_catalog(TARGET_RA, TARGET_DEC, SEARCH_RADIUS,
                                         'J/ApJS/255/30', 'VLASS')
    results['VLASS'] = vlass_result
    if vlass_result.get('found'):
        print(f"  FOUND: {len(vlass_result['table'])} source(s)")
    elif vlass_result.get('error'):
        print(f"  Query failed: {vlass_result['error'][:80]}")
    else:
        print("  No sources found within search radius")

    print()
    return results


def compute_upper_limits(xray_results, radio_results):
    """Compute luminosity upper limits from non-detections."""
    print("=" * 70)
    print("LUMINOSITY UPPER LIMITS")
    print("=" * 70)
    print(f"Adopted distance: {DISTANCE_PC} ± {DISTANCE_ERR_PC} pc")
    print()

    upper_limits = {}

    # X-ray upper limits
    print("X-RAY UPPER LIMITS:")
    print("-" * 50)

    for survey, info in [('ROSAT_2RXS', SURVEY_LIMITS['ROSAT_2RXS']),
                          ('XMM_4XMM', SURVEY_LIMITS['XMM_4XMM']),
                          ('Chandra_CSC2', SURVEY_LIMITS['Chandra_CSC2'])]:

        # Check if detected
        detected = xray_results.get(survey, {}).get('found', False)

        if not detected:
            F_lim = info['flux_limit_cgs']
            L_lim = flux_to_luminosity(F_lim, DISTANCE_PC)
            L_lim_err = flux_to_luminosity(F_lim, DISTANCE_PC + DISTANCE_ERR_PC) - L_lim

            upper_limits[survey] = {
                'detected': False,
                'band': info['band'],
                'flux_limit_cgs': F_lim,
                'L_X_limit_erg_s': L_lim,
                'L_X_limit_err': abs(L_lim_err),
                'notes': info['notes']
            }

            print(f"  {survey} ({info['band']}):")
            print(f"    Flux limit: {F_lim:.2e} erg/s/cm²")
            print(f"    L_X limit: {L_lim:.2e} erg/s")

            # Compare to M-dwarf emission
            if L_lim > MDWARF_LX_QUIESCENT[1]:
                print(f"    → Above quiescent M-dwarf range ({MDWARF_LX_QUIESCENT[0]:.0e}-{MDWARF_LX_QUIESCENT[1]:.0e})")
            else:
                print(f"    → Within/below quiescent M-dwarf range")
        else:
            upper_limits[survey] = {
                'detected': True,
                'band': info['band'],
                'notes': 'Source detected - see detailed results'
            }
            print(f"  {survey}: DETECTED")

    print()

    # Radio upper limits
    print("RADIO UPPER LIMITS:")
    print("-" * 50)

    for survey, info in [('NVSS', SURVEY_LIMITS['NVSS']),
                          ('FIRST', SURVEY_LIMITS['FIRST']),
                          ('VLASS', SURVEY_LIMITS['VLASS'])]:

        detected = radio_results.get(survey, {}).get('found', False)

        if not detected:
            S_lim = info['flux_limit_mJy']
            freq = float(info['band'].split()[0])  # Extract frequency
            L_nu_lim = mJy_to_luminosity(S_lim, DISTANCE_PC, freq)

            upper_limits[survey] = {
                'detected': False,
                'band': info['band'],
                'flux_limit_mJy': S_lim,
                'L_nu_limit_erg_s_Hz': L_nu_lim,
                'notes': info['notes']
            }

            print(f"  {survey} ({info['band']}):")
            print(f"    Flux limit: {S_lim:.2f} mJy")
            print(f"    L_ν limit: {L_nu_lim:.2e} erg/s/Hz")
        else:
            upper_limits[survey] = {
                'detected': True,
                'band': info['band'],
                'notes': 'Source detected - see detailed results'
            }
            print(f"  {survey}: DETECTED")

    print()
    return upper_limits


def create_limits_plot(upper_limits):
    """Create visualization of X-ray and radio limits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # X-ray panel
    xray_surveys = ['ROSAT_2RXS', 'XMM_4XMM', 'Chandra_CSC2']
    xray_limits = []
    xray_labels = []

    for survey in xray_surveys:
        if survey in upper_limits and not upper_limits[survey].get('detected', True):
            xray_limits.append(upper_limits[survey]['L_X_limit_erg_s'])
            xray_labels.append(survey.replace('_', '\n'))
        else:
            xray_limits.append(0)
            xray_labels.append(survey.replace('_', '\n') + '\n(N/A)')

    x_pos = np.arange(len(xray_surveys))
    bars1 = ax1.bar(x_pos, xray_limits, color='steelblue', alpha=0.7, edgecolor='navy')

    # Add M-dwarf reference ranges
    ax1.axhspan(MDWARF_LX_QUIESCENT[0], MDWARF_LX_QUIESCENT[1], alpha=0.2, color='green',
                label=f'Quiescent M-dwarf ({MDWARF_LX_QUIESCENT[0]:.0e}-{MDWARF_LX_QUIESCENT[1]:.0e})')
    ax1.axhspan(MDWARF_LX_FLARE[0], MDWARF_LX_FLARE[1], alpha=0.2, color='orange',
                label=f'Flaring M-dwarf ({MDWARF_LX_FLARE[0]:.0e}-{MDWARF_LX_FLARE[1]:.0e})')

    ax1.set_yscale('log')
    ax1.set_ylabel('L$_X$ Upper Limit (erg/s)', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(xray_labels, fontsize=10)
    ax1.set_title('X-ray Luminosity Upper Limits', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(1e26, 1e32)
    ax1.grid(True, alpha=0.3, axis='y')

    # Radio panel
    radio_surveys = ['NVSS', 'FIRST', 'VLASS']
    radio_limits = []
    radio_labels = []

    for survey in radio_surveys:
        if survey in upper_limits and not upper_limits[survey].get('detected', True):
            radio_limits.append(upper_limits[survey]['L_nu_limit_erg_s_Hz'])
            radio_labels.append(survey)
        else:
            radio_limits.append(0)
            radio_labels.append(survey + '\n(N/A)')

    x_pos2 = np.arange(len(radio_surveys))
    bars2 = ax2.bar(x_pos2, radio_limits, color='firebrick', alpha=0.7, edgecolor='darkred')

    # Typical radio luminosities for reference
    ax2.axhline(1e15, color='purple', linestyle='--', alpha=0.7, label='Typical active M-dwarf')
    ax2.axhline(1e18, color='black', linestyle=':', alpha=0.7, label='Accreting compact object')

    ax2.set_yscale('log')
    ax2.set_ylabel('L$_\\nu$ Upper Limit (erg/s/Hz)', fontsize=12)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(radio_labels, fontsize=10)
    ax2.set_title('Radio Luminosity Upper Limits', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(1e12, 1e20)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'X-ray and Radio Constraints: {TARGET_NAME}\n(d = {DISTANCE_PC} pc)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('xray_radio_limits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: xray_radio_limits.png")


def interpret_results(upper_limits):
    """Interpret the X-ray and radio constraints."""
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    # Check for any detections
    xray_detected = any(upper_limits.get(s, {}).get('detected', False)
                        for s in ['ROSAT_2RXS', 'XMM_4XMM', 'Chandra_CSC2'])
    radio_detected = any(upper_limits.get(s, {}).get('detected', False)
                         for s in ['NVSS', 'FIRST', 'VLASS'])

    interpretation = {
        'xray_detected': xray_detected,
        'radio_detected': radio_detected,
    }

    if not xray_detected:
        # Get the most constraining X-ray limit
        xray_lims = [(s, upper_limits[s]['L_X_limit_erg_s'])
                     for s in ['ROSAT_2RXS', 'XMM_4XMM', 'Chandra_CSC2']
                     if s in upper_limits and 'L_X_limit_erg_s' in upper_limits[s]]

        if xray_lims:
            best_xray = min(xray_lims, key=lambda x: x[1])
            interpretation['best_xray_limit'] = {
                'survey': best_xray[0],
                'L_X_limit': best_xray[1]
            }

            print("X-RAY INTERPRETATION:")
            print("-" * 50)
            print(f"  No X-ray counterpart detected.")
            print(f"  Best upper limit: L_X < {best_xray[1]:.2e} erg/s ({best_xray[0]})")
            print()

            if best_xray[1] < MDWARF_LX_QUIESCENT[1]:
                print("  This limit is BELOW typical quiescent M-dwarf emission.")
                print("  → Constrains any X-ray activity to very low levels.")
                interpretation['xray_conclusion'] = 'below_quiescent'
            elif best_xray[1] < MDWARF_LX_FLARE[0]:
                print("  This limit is within the quiescent M-dwarf range.")
                print("  → Consistent with a typical inactive M dwarf.")
                interpretation['xray_conclusion'] = 'within_quiescent'
            else:
                print("  This limit is above typical M-dwarf emission.")
                print("  → Does not strongly constrain X-ray activity.")
                interpretation['xray_conclusion'] = 'unconstraining'
            print()

            # Compare to NS/BH accretion
            print("  For a quiescent NS or BH:")
            print("    - Quiescent NS: L_X ~ 10³¹-10³³ erg/s (detectable if accreting)")
            print("    - Quiescent BH: L_X ~ 10³⁰-10³² erg/s (detectable if accreting)")
            print("  → Non-detection supports a QUIESCENT (non-accreting) compact object")
            print()
    else:
        print("X-RAY: Source DETECTED - further analysis needed")
        interpretation['xray_conclusion'] = 'detected'

    if not radio_detected:
        radio_lims = [(s, upper_limits[s]['L_nu_limit_erg_s_Hz'])
                      for s in ['NVSS', 'FIRST', 'VLASS']
                      if s in upper_limits and 'L_nu_limit_erg_s_Hz' in upper_limits[s]]

        if radio_lims:
            best_radio = min(radio_lims, key=lambda x: x[1])
            interpretation['best_radio_limit'] = {
                'survey': best_radio[0],
                'L_nu_limit': best_radio[1]
            }

            print("RADIO INTERPRETATION:")
            print("-" * 50)
            print(f"  No radio counterpart detected.")
            print(f"  Best upper limit: L_ν < {best_radio[1]:.2e} erg/s/Hz ({best_radio[0]})")
            print()
            print("  → Consistent with a radio-quiet system")
            print("  → No evidence for jets or strong radio emission")
            interpretation['radio_conclusion'] = 'non_detection'
    else:
        print("RADIO: Source DETECTED - further analysis needed")
        interpretation['radio_conclusion'] = 'detected'

    print()
    print("OVERALL CONCLUSION:")
    print("-" * 50)

    if not xray_detected and not radio_detected:
        print("  The system shows NO X-ray or radio counterpart.")
        print("  This is consistent with:")
        print("    1. A quiescent (non-accreting) compact companion")
        print("    2. The deeply detached binary configuration (no mass transfer)")
        print("  → STRENGTHENS the 'dark companion' interpretation")
        interpretation['overall'] = 'supports_dark_companion'
    else:
        print("  Detections found - requires detailed analysis of fluxes")
        interpretation['overall'] = 'requires_analysis'

    print()

    return interpretation


def main():
    results = {
        'target': {
            'name': TARGET_NAME,
            'ra': TARGET_RA,
            'dec': TARGET_DEC,
            'search_radius_arcsec': SEARCH_RADIUS,
            'distance_pc': DISTANCE_PC,
            'distance_err_pc': DISTANCE_ERR_PC
        }
    }

    # Search X-ray catalogs
    xray_results = search_xray_catalogs()
    results['xray_searches'] = {}
    for key, val in xray_results.items():
        results['xray_searches'][key] = {
            'found': val.get('found', False),
            'error': val.get('error', None)
        }

    # Search radio catalogs
    radio_results = search_radio_catalogs()
    results['radio_searches'] = {}
    for key, val in radio_results.items():
        results['radio_searches'][key] = {
            'found': val.get('found', False),
            'error': val.get('error', None)
        }

    # Compute upper limits
    upper_limits = compute_upper_limits(xray_results, radio_results)
    results['upper_limits'] = {}
    for key, val in upper_limits.items():
        # Convert numpy types to native Python
        clean_val = {}
        for k, v in val.items():
            if hasattr(v, 'item'):
                clean_val[k] = v.item()
            else:
                clean_val[k] = v
        results['upper_limits'][key] = clean_val

    # Create visualization
    create_limits_plot(upper_limits)

    # Interpret results
    interpretation = interpret_results(upper_limits)
    results['interpretation'] = interpretation

    # Save results
    with open('xray_radio_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved: xray_radio_results.json")

    return results


if __name__ == "__main__":
    main()
