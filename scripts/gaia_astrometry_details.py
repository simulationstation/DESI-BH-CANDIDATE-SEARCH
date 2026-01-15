#!/usr/bin/env python3
"""
gaia_astrometry_details.py - Deep Gaia Astrometry Check

Retrieves and analyzes Gaia DR3 astrometric quality flags including
ipd_frac_multi_peak and ipd_gof_harmonic_amplitude to determine
whether the source appears as a single unresolved photocenter wobble
or a resolved double.

Target: Gaia DR3 3802130935635096832
"""

import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Target information
GAIA_SOURCE_ID = 3802130935635096832
TARGET_RA = 164.5235
TARGET_DEC = -1.6602
TARGET_NAME = f"Gaia DR3 {GAIA_SOURCE_ID}"

# Columns to retrieve
GAIA_COLUMNS = [
    'source_id',
    'ra', 'dec',
    'parallax', 'parallax_error',
    'pmra', 'pmra_error',
    'pmdec', 'pmdec_error',
    'ruwe',
    'astrometric_excess_noise',
    'astrometric_excess_noise_sig',
    'astrometric_params_solved',
    'astrometric_n_good_obs_al',
    'astrometric_chi2_al',
    'astrometric_gof_al',
    'ipd_frac_multi_peak',
    'ipd_gof_harmonic_amplitude',
    'ipd_frac_odd_win',
    'phot_g_mean_mag',
    'phot_bp_mean_mag',
    'phot_rp_mean_mag',
    'bp_rp',
    'visibility_periods_used',
    'duplicated_source',
]


def query_gaia_dr3():
    """Query Gaia DR3 for detailed astrometric parameters."""
    print("=" * 70)
    print("GAIA DR3 ASTROMETRY QUERY")
    print("=" * 70)
    print(f"Target: {TARGET_NAME}")
    print(f"Source ID: {GAIA_SOURCE_ID}")
    print()

    result = {'status': 'unknown', 'data': {}}

    # Try astroquery first
    try:
        from astroquery.gaia import Gaia

        # Build column list
        columns = ', '.join(GAIA_COLUMNS)

        query = f"""
        SELECT {columns}
        FROM gaiadr3.gaia_source
        WHERE source_id = {GAIA_SOURCE_ID}
        """

        print("Executing Gaia TAP query...")
        job = Gaia.launch_job(query)
        table = job.get_results()

        if len(table) > 0:
            print(f"  Found source in Gaia DR3")
            result['status'] = 'success'
            result['method'] = 'astroquery_tap'

            # Extract all columns
            row = table[0]
            for col in GAIA_COLUMNS:
                if col in table.colnames:
                    val = row[col]
                    if hasattr(val, 'item'):
                        val = val.item()
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        result['data'][col] = val
        else:
            print("  Source not found in Gaia DR3")
            result['status'] = 'not_found'

    except ImportError:
        print("  astroquery.gaia not available, trying VizieR...")
        result['method'] = 'vizier_fallback'

    except Exception as e:
        print(f"  Gaia TAP query failed: {str(e)[:80]}")
        result['error'] = str(e)[:200]

    # Fallback to VizieR if needed
    if result['status'] != 'success':
        try:
            from astroquery.vizier import Vizier
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            print("Trying VizieR Gaia DR3...")
            coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')

            v = Vizier(columns=['**'], row_limit=5)  # All columns
            vizier_result = v.query_region(coord, radius=2*u.arcsec, catalog='I/355/gaiadr3')

            if vizier_result and len(vizier_result) > 0:
                table = vizier_result[0]

                # Find our source by source_id
                for row in table:
                    if 'Source' in table.colnames:
                        src_id = row['Source']
                        if src_id == GAIA_SOURCE_ID:
                            print(f"  Found source via VizieR")
                            result['status'] = 'vizier_success'
                            result['method'] = 'vizier'

                            # Extract available columns
                            for col in table.colnames:
                                val = row[col]
                                if hasattr(val, 'item'):
                                    val = val.item()
                                if val is not None:
                                    result['data'][col] = val
                            break

                if result['status'] != 'vizier_success':
                    # Take closest source
                    row = table[0]
                    print(f"  Using closest source from VizieR")
                    result['status'] = 'vizier_closest'
                    for col in table.colnames:
                        val = row[col]
                        if hasattr(val, 'item'):
                            val = val.item()
                        if val is not None:
                            result['data'][col] = val
            else:
                print("  No sources found via VizieR")
                result['status'] = 'not_found'

        except Exception as e:
            print(f"  VizieR query failed: {str(e)[:80]}")
            result['status'] = 'query_failed'
            result['error'] = str(e)[:200]

    print()
    return result


def analyze_astrometry(gaia_data):
    """Analyze astrometric quality flags."""
    print("=" * 70)
    print("ASTROMETRIC QUALITY ANALYSIS")
    print("=" * 70)
    print()

    data = gaia_data.get('data', {})
    analysis = {'parameters': {}, 'flags': {}, 'interpretation': {}}

    # Key parameters
    params_to_check = [
        ('parallax', 'mas', 'Parallax'),
        ('parallax_error', 'mas', 'Parallax error'),
        ('ruwe', '', 'RUWE'),
        ('astrometric_excess_noise', 'mas', 'Astrometric excess noise'),
        ('astrometric_excess_noise_sig', '', 'AEN significance'),
        ('ipd_frac_multi_peak', '', 'IPD frac multi-peak'),
        ('ipd_gof_harmonic_amplitude', '', 'IPD GoF harmonic amp'),
        ('phot_g_mean_mag', 'mag', 'G magnitude'),
        ('astrometric_params_solved', '', 'Params solved'),
        ('visibility_periods_used', '', 'Visibility periods'),
    ]

    print("KEY ASTROMETRIC PARAMETERS:")
    print("-" * 60)

    for param, unit, description in params_to_check:
        if param in data:
            val = data[param]
            analysis['parameters'][param] = val
            unit_str = f" {unit}" if unit else ""
            print(f"  {description:<30}: {val}{unit_str}")

    print()

    # Flag analysis
    print("FLAG INTERPRETATION:")
    print("-" * 60)

    # RUWE
    ruwe = data.get('ruwe', None)
    if ruwe is not None:
        if ruwe > 1.4:
            print(f"  RUWE = {ruwe:.3f} > 1.4: POOR astrometric fit")
            print("    → Indicates non-single-star behavior")
            analysis['flags']['ruwe'] = 'elevated'
        else:
            print(f"  RUWE = {ruwe:.3f} < 1.4: Good astrometric fit")
            analysis['flags']['ruwe'] = 'normal'

    # Astrometric excess noise
    aen = data.get('astrometric_excess_noise', None)
    aen_sig = data.get('astrometric_excess_noise_sig', None)
    if aen is not None and aen_sig is not None:
        if aen_sig > 2:
            print(f"  AEN = {aen:.3f} mas ({aen_sig:.1f}σ): SIGNIFICANT excess noise")
            print("    → Unmodeled astrometric motion present")
            analysis['flags']['aen'] = 'significant'
        else:
            print(f"  AEN = {aen:.3f} mas ({aen_sig:.1f}σ): Not significant")
            analysis['flags']['aen'] = 'normal'

    # IPD frac multi-peak - KEY for resolved vs unresolved
    ipd_multi = data.get('ipd_frac_multi_peak', None)
    if ipd_multi is not None:
        print()
        print("  IPD FRAC MULTI-PEAK (Critical for resolved double test):")
        if ipd_multi < 5:
            print(f"    Value = {ipd_multi:.1f}% < 5%: Source appears SINGLE")
            print("    → NOT a resolved visual double")
            print("    → Consistent with unresolved photocenter wobble")
            analysis['flags']['ipd_multi_peak'] = 'single_source'
            analysis['interpretation']['resolved_double'] = False
        elif ipd_multi < 20:
            print(f"    Value = {ipd_multi:.1f}%: Some multi-peak detections")
            print("    → Possible resolved companion or extended source")
            analysis['flags']['ipd_multi_peak'] = 'some_multi'
            analysis['interpretation']['resolved_double'] = 'uncertain'
        else:
            print(f"    Value = {ipd_multi:.1f}% > 20%: Frequent multi-peak")
            print("    → Likely a resolved double or extended source")
            analysis['flags']['ipd_multi_peak'] = 'multi_peak'
            analysis['interpretation']['resolved_double'] = True
    else:
        print("  IPD frac multi-peak: NOT AVAILABLE")
        print("    → Cannot definitively test for resolved double")
        analysis['flags']['ipd_multi_peak'] = 'unavailable'
        analysis['interpretation']['resolved_double'] = 'unknown'

    # IPD GoF harmonic amplitude
    ipd_gof = data.get('ipd_gof_harmonic_amplitude', None)
    if ipd_gof is not None:
        if ipd_gof > 0.1:
            print(f"  IPD GoF harmonic amp = {ipd_gof:.3f} > 0.1: Periodic pattern")
            analysis['flags']['ipd_gof_harmonic'] = 'elevated'
        else:
            print(f"  IPD GoF harmonic amp = {ipd_gof:.3f}: Normal")
            analysis['flags']['ipd_gof_harmonic'] = 'normal'

    # Duplicated source flag
    dup = data.get('duplicated_source', None)
    if dup is not None:
        if dup:
            print(f"  Duplicated source: YES - caution advised")
            analysis['flags']['duplicated'] = True
        else:
            print(f"  Duplicated source: NO")
            analysis['flags']['duplicated'] = False

    print()

    # Overall interpretation
    print("OVERALL INTERPRETATION:")
    print("-" * 60)

    if analysis['flags'].get('ruwe') == 'elevated' and analysis['flags'].get('aen') == 'significant':
        print("  ✓ RUWE and AEN both indicate non-single-star astrometry")

        if analysis['interpretation'].get('resolved_double') == False:
            print("  ✓ IPD frac multi-peak shows source is NOT resolved")
            print()
            print("  CONCLUSION: The Gaia astrometry is consistent with an")
            print("  UNRESOLVED PHOTOCENTER WOBBLE from an unseen companion.")
            print("  This SUPPORTS the dark companion hypothesis.")
            analysis['interpretation']['conclusion'] = 'unresolved_wobble'
            analysis['interpretation']['supports_dark_companion'] = True
        elif analysis['interpretation'].get('resolved_double') == True:
            print("  ✗ IPD indicates source may be resolved")
            print()
            print("  CAUTION: The source may be a resolved visual double.")
            analysis['interpretation']['conclusion'] = 'possible_resolved'
            analysis['interpretation']['supports_dark_companion'] = False
        else:
            print("  ? IPD frac multi-peak not available")
            print()
            print("  Cannot definitively rule out resolved double.")
            analysis['interpretation']['conclusion'] = 'inconclusive'
            analysis['interpretation']['supports_dark_companion'] = 'uncertain'
    else:
        print("  Astrometry appears normal - no strong evidence for companion")
        analysis['interpretation']['conclusion'] = 'normal_astrometry'
        analysis['interpretation']['supports_dark_companion'] = 'neutral'

    print()

    return analysis


def create_summary_notes(gaia_data, analysis):
    """Create summary notes for the report."""
    notes = []

    notes.append("# Gaia DR3 Astrometric Analysis Notes")
    notes.append(f"\n## Target: {TARGET_NAME}\n")

    data = gaia_data.get('data', {})

    # Key values
    notes.append("## Key Astrometric Values\n")
    notes.append(f"- RUWE: {data.get('ruwe', 'N/A')}")
    notes.append(f"- AEN: {data.get('astrometric_excess_noise', 'N/A')} mas")
    notes.append(f"- AEN significance: {data.get('astrometric_excess_noise_sig', 'N/A')}")
    notes.append(f"- IPD frac multi-peak: {data.get('ipd_frac_multi_peak', 'N/A')}")
    notes.append(f"- IPD GoF harmonic amplitude: {data.get('ipd_gof_harmonic_amplitude', 'N/A')}")

    # Interpretation
    notes.append("\n## Interpretation\n")
    interp = analysis.get('interpretation', {})

    if interp.get('resolved_double') == False:
        notes.append("The low ipd_frac_multi_peak indicates the source is NOT a resolved")
        notes.append("visual double. The elevated RUWE and AEN are consistent with an")
        notes.append("unresolved photocenter wobble from an unseen companion.")
        notes.append("\n**Conclusion: Supports dark companion hypothesis.**")
    elif interp.get('resolved_double') == True:
        notes.append("High ipd_frac_multi_peak suggests possible resolved companion.")
        notes.append("Further investigation needed.")
    else:
        notes.append("IPD flags not available; cannot definitively test for resolved double.")

    return '\n'.join(notes)


def main():
    # Query Gaia DR3
    gaia_result = query_gaia_dr3()

    # If query succeeded, analyze
    if gaia_result['status'] in ['success', 'vizier_success', 'vizier_closest']:
        analysis = analyze_astrometry(gaia_result)
    else:
        print("Query failed - using fallback values from existing analysis")
        # Use values we already know from distance_analysis.py
        gaia_result['data'] = {
            'source_id': GAIA_SOURCE_ID,
            'parallax': 0.119,
            'parallax_error': 0.160,
            'ruwe': 1.9535599946975708,
            'astrometric_excess_noise': 0.532,
            'astrometric_excess_noise_sig': 16.49,
            'phot_g_mean_mag': 17.27,
        }
        gaia_result['status'] = 'fallback'
        analysis = analyze_astrometry(gaia_result)

    # Create summary notes
    notes = create_summary_notes(gaia_result, analysis)

    # Save notes
    with open('gaia_astrometry_notes.md', 'w') as f:
        f.write(notes)
    print("Saved: gaia_astrometry_notes.md")

    # Compile results
    results = {
        'target': TARGET_NAME,
        'source_id': GAIA_SOURCE_ID,
        'query_status': gaia_result['status'],
        'gaia_data': {k: v for k, v in gaia_result.get('data', {}).items()
                      if not isinstance(v, np.ndarray)},
        'analysis': analysis,
    }

    # Save results
    with open('gaia_astrometry_details.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved: gaia_astrometry_details.json")

    return results


if __name__ == "__main__":
    main()
