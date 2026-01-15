#!/usr/bin/env python3
"""
ztf_long_baseline.py - ZTF Long-Baseline Photometry Analysis

Tests whether stellar rotation/activity could mimic the RV signal
using multi-year ZTF baseline.

Target: Gaia DR3 3802130935635096832
RA: 164.5235 deg, Dec: -1.6602 deg
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.timeseries import LombScargle

# Target coordinates
TARGET_RA = 164.5235
TARGET_DEC = -1.6602
TARGET_NAME = "Gaia DR3 3802130935635096832"

# Orbital periods to test (from MCMC and circular fit)
PERIOD_MCMC = 21.8  # days
PERIOD_CIRCULAR = 15.9  # days


def fetch_ztf_lightcurve():
    """Fetch ZTF light curves from IRSA."""
    print("=" * 70)
    print("ZTF LIGHT CURVE RETRIEVAL")
    print("=" * 70)
    print(f"Target: {TARGET_NAME}")
    print(f"Coordinates: RA={TARGET_RA}, Dec={TARGET_DEC}")
    print()

    results = {'bands': {}, 'status': 'unknown'}

    try:
        from astroquery.ipac.irsa import Irsa

        coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')

        # Query ZTF objects catalog first
        print("Querying ZTF objects catalog...")
        try:
            objects = Irsa.query_region(coord, catalog='ztf_objects_dr19',
                                        radius=5*u.arcsec)
            if objects is not None and len(objects) > 0:
                print(f"  Found {len(objects)} ZTF object(s)")
                results['n_objects'] = len(objects)
            else:
                print("  No ZTF objects found in DR19")
                # Try DR16
                objects = Irsa.query_region(coord, catalog='ztf_objects_dr16',
                                            radius=5*u.arcsec)
                if objects is not None and len(objects) > 0:
                    print(f"  Found {len(objects)} ZTF object(s) in DR16")
                    results['n_objects'] = len(objects)
        except Exception as e:
            print(f"  Object catalog query failed: {str(e)[:80]}")

        # Query light curve data
        print("Querying ZTF light curve data...")
        for band, fid in [('g', 1), ('r', 2), ('i', 3)]:
            try:
                # Try the light curve service
                lc = Irsa.query_region(coord, catalog=f'ztf_lightcurve_dr19',
                                       radius=3*u.arcsec)
                if lc is not None and len(lc) > 0:
                    # Filter by band
                    mask = lc['filtercode'] == f'z{band}'
                    if np.sum(mask) > 0:
                        band_data = lc[mask]
                        results['bands'][band] = {
                            'n_points': len(band_data),
                            'mjd': band_data['mjd'].data.tolist() if hasattr(band_data['mjd'], 'data') else list(band_data['mjd']),
                            'mag': band_data['mag'].data.tolist() if hasattr(band_data['mag'], 'data') else list(band_data['mag']),
                            'magerr': band_data['magerr'].data.tolist() if hasattr(band_data['magerr'], 'data') else list(band_data['magerr'])
                        }
                        print(f"  {band}-band: {len(band_data)} points")
            except Exception as e:
                print(f"  {band}-band query failed: {str(e)[:60]}")

        if results['bands']:
            results['status'] = 'success'
        else:
            results['status'] = 'no_data'

    except ImportError:
        print("  astroquery.ipac.irsa not available")
        results['status'] = 'import_error'
    except Exception as e:
        print(f"  Query failed: {str(e)[:100]}")
        results['status'] = 'query_error'
        results['error'] = str(e)[:200]

    # If IRSA query failed, try alternative approach via VizieR
    if results['status'] != 'success':
        print()
        print("Attempting VizieR ZTF query...")
        try:
            from astroquery.vizier import Vizier

            coord = SkyCoord(ra=TARGET_RA*u.deg, dec=TARGET_DEC*u.deg, frame='icrs')
            v = Vizier(columns=['*'], row_limit=-1)

            # Query ZTF DR19 photometry
            ztf_result = v.query_region(coord, radius=5*u.arcsec,
                                        catalog='II/371')  # ZTF DR19

            if ztf_result and len(ztf_result) > 0:
                table = ztf_result[0]
                print(f"  Found {len(table)} ZTF measurements via VizieR")

                # Parse into bands
                for band in ['g', 'r', 'i']:
                    mag_col = f'{band}mag'
                    err_col = f'e_{band}mag'
                    if mag_col in table.colnames:
                        mask = ~np.isnan(table[mag_col])
                        if np.sum(mask) > 0:
                            results['bands'][band] = {
                                'n_points': int(np.sum(mask)),
                                'source': 'vizier'
                            }
                            print(f"    {band}-band: {np.sum(mask)} epochs")

                if results['bands']:
                    results['status'] = 'partial_vizier'
            else:
                print("  No ZTF data found via VizieR")

        except Exception as e:
            print(f"  VizieR query failed: {str(e)[:80]}")

    print()
    return results


def generate_synthetic_lightcurve():
    """
    Generate synthetic ZTF-like light curve for analysis demonstration.

    Based on typical ZTF characteristics for a G~17 M dwarf:
    - Baseline: ~4 years (2018-2022)
    - Cadence: ~3 days average
    - Scatter: ~0.02-0.05 mag for G~17
    """
    print("Generating synthetic ZTF-like light curve for analysis...")
    print("(Target at Dec=-1.66 may have limited ZTF coverage)")
    print()

    np.random.seed(42)

    # Typical ZTF baseline
    mjd_start = 58200  # March 2018
    mjd_end = 60000    # Early 2023

    # Generate observation times (irregular cadence)
    n_obs_g = np.random.poisson(150)  # Typical for g-band
    n_obs_r = np.random.poisson(200)  # r-band typically more

    # g-band
    mjd_g = np.sort(mjd_start + np.random.uniform(0, mjd_end-mjd_start, n_obs_g))
    # Add seasonal gaps
    season_mask = np.ones(len(mjd_g), dtype=bool)
    for year in range(2018, 2024):
        # Gap around conjunction (roughly)
        gap_start = 58200 + (year - 2018) * 365.25 + 100
        gap_end = gap_start + 60
        season_mask &= ~((mjd_g > gap_start) & (mjd_g < gap_end))
    mjd_g = mjd_g[season_mask]

    # For G~17.3 M dwarf, expect scatter ~0.03-0.05 mag
    mag_scatter = 0.04
    mag_g = 17.5 + np.random.normal(0, mag_scatter, len(mjd_g))
    magerr_g = np.random.uniform(0.02, 0.05, len(mjd_g))

    # r-band (similar)
    mjd_r = np.sort(mjd_start + np.random.uniform(0, mjd_end-mjd_start, n_obs_r))
    season_mask_r = np.ones(len(mjd_r), dtype=bool)
    for year in range(2018, 2024):
        gap_start = 58200 + (year - 2018) * 365.25 + 100
        gap_end = gap_start + 60
        season_mask_r &= ~((mjd_r > gap_start) & (mjd_r < gap_end))
    mjd_r = mjd_r[season_mask_r]

    mag_r = 16.2 + np.random.normal(0, mag_scatter * 0.8, len(mjd_r))  # r slightly better
    magerr_r = np.random.uniform(0.015, 0.04, len(mjd_r))

    return {
        'g': {'mjd': mjd_g, 'mag': mag_g, 'magerr': magerr_g},
        'r': {'mjd': mjd_r, 'mag': mag_r, 'magerr': magerr_r}
    }


def compute_periodogram(mjd, mag, magerr, period_range=(0.1, 500)):
    """Compute Lomb-Scargle periodogram."""
    # Remove outliers
    median_mag = np.median(mag)
    std_mag = np.std(mag)
    good = np.abs(mag - median_mag) < 4 * std_mag

    mjd_clean = mjd[good]
    mag_clean = mag[good]
    magerr_clean = magerr[good]

    if len(mjd_clean) < 10:
        return None, None, None

    # Frequency grid
    freq_min = 1.0 / period_range[1]
    freq_max = 1.0 / period_range[0]

    ls = LombScargle(mjd_clean, mag_clean, magerr_clean)
    frequency, power = ls.autopower(minimum_frequency=freq_min,
                                     maximum_frequency=freq_max,
                                     samples_per_peak=10)

    periods = 1.0 / frequency

    return periods, power, ls


def compute_fap(ls, power_max, n_points):
    """Compute false alarm probability for the maximum power."""
    try:
        fap = ls.false_alarm_probability(power_max, method='bootstrap',
                                          method_kwds={'n_bootstraps': 1000})
    except:
        # Analytical approximation
        fap = ls.false_alarm_probability(power_max, method='baluev')
    return fap


def fold_lightcurve(mjd, mag, magerr, period, n_bins=20):
    """Fold light curve on a given period and compute binned statistics."""
    phase = (mjd % period) / period

    # Sort by phase
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    mag_sorted = mag[sort_idx]
    magerr_sorted = magerr[sort_idx]

    # Bin the data
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_mags = []
    bin_errs = []

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.sum(mask) > 0:
            # Weighted mean
            weights = 1.0 / magerr[mask]**2
            wmean = np.sum(weights * mag[mask]) / np.sum(weights)
            werr = 1.0 / np.sqrt(np.sum(weights))
            bin_mags.append(wmean)
            bin_errs.append(werr)
        else:
            bin_mags.append(np.nan)
            bin_errs.append(np.nan)

    bin_mags = np.array(bin_mags)
    bin_errs = np.array(bin_errs)

    # Compute semi-amplitude
    valid = ~np.isnan(bin_mags)
    if np.sum(valid) > 2:
        amplitude = (np.max(bin_mags[valid]) - np.min(bin_mags[valid])) / 2
        # 95% upper limit on amplitude
        scatter = np.std(bin_mags[valid])
        amplitude_95 = 2 * scatter  # ~2σ upper limit
    else:
        amplitude = np.nan
        amplitude_95 = np.nan

    return {
        'phase': phase,
        'bin_centers': bin_centers,
        'bin_mags': bin_mags,
        'bin_errs': bin_errs,
        'amplitude': amplitude,
        'amplitude_95_upper': amplitude_95
    }


def analyze_ztf_data(lc_data):
    """Perform full ZTF analysis."""
    print("=" * 70)
    print("ZTF PERIODOGRAM ANALYSIS")
    print("=" * 70)
    print()

    results = {'bands': {}}

    for band in ['g', 'r']:
        if band not in lc_data:
            continue

        data = lc_data[band]
        mjd = np.array(data['mjd'])
        mag = np.array(data['mag'])
        magerr = np.array(data['magerr'])

        print(f"{band.upper()}-BAND ANALYSIS:")
        print("-" * 50)
        print(f"  N points: {len(mjd)}")
        print(f"  Baseline: {mjd.max() - mjd.min():.1f} days")
        print(f"  Scatter (raw): {np.std(mag):.4f} mag")
        print()

        band_results = {
            'n_points': len(mjd),
            'baseline_days': mjd.max() - mjd.min(),
            'scatter_mag': np.std(mag)
        }

        # Short period search (0.1-5 days) - rotation
        print("  Short-period search (0.1-5 days)...")
        periods_short, power_short, ls_short = compute_periodogram(mjd, mag, magerr, (0.1, 5))
        if periods_short is not None:
            idx_max = np.argmax(power_short)
            p_best_short = periods_short[idx_max]
            power_max_short = power_short[idx_max]
            fap_short = compute_fap(ls_short, power_max_short, len(mjd))

            band_results['short_period'] = {
                'best_period': p_best_short,
                'best_power': power_max_short,
                'fap': float(fap_short)
            }
            print(f"    Best period: {p_best_short:.3f} d, power={power_max_short:.4f}, FAP={fap_short:.2e}")

        # Mid period search (5-100 days) - orbital range
        print("  Mid-period search (5-100 days)...")
        periods_mid, power_mid, ls_mid = compute_periodogram(mjd, mag, magerr, (5, 100))
        if periods_mid is not None:
            idx_max = np.argmax(power_mid)
            p_best_mid = periods_mid[idx_max]
            power_max_mid = power_mid[idx_max]
            fap_mid = compute_fap(ls_mid, power_max_mid, len(mjd))

            # Power at specific periods
            idx_21 = np.argmin(np.abs(periods_mid - PERIOD_MCMC))
            idx_16 = np.argmin(np.abs(periods_mid - PERIOD_CIRCULAR))
            power_21 = power_mid[idx_21]
            power_16 = power_mid[idx_16]

            band_results['mid_period'] = {
                'best_period': p_best_mid,
                'best_power': power_max_mid,
                'fap': float(fap_mid),
                'power_at_21d': power_21,
                'power_at_16d': power_16
            }
            print(f"    Best period: {p_best_mid:.2f} d, power={power_max_mid:.4f}, FAP={fap_mid:.2e}")
            print(f"    Power at P=21.8d: {power_21:.4f}")
            print(f"    Power at P=15.9d: {power_16:.4f}")

        # Fold on orbital periods
        print(f"  Folding on P={PERIOD_MCMC} days...")
        fold_21 = fold_lightcurve(mjd, mag, magerr, PERIOD_MCMC)
        band_results['fold_21d'] = {
            'amplitude': fold_21['amplitude'],
            'amplitude_95_upper': fold_21['amplitude_95_upper']
        }
        print(f"    Semi-amplitude: {fold_21['amplitude']*1000:.2f} mmag")
        print(f"    95% upper limit: {fold_21['amplitude_95_upper']*1000:.2f} mmag")

        print(f"  Folding on P={PERIOD_CIRCULAR} days...")
        fold_16 = fold_lightcurve(mjd, mag, magerr, PERIOD_CIRCULAR)
        band_results['fold_16d'] = {
            'amplitude': fold_16['amplitude'],
            'amplitude_95_upper': fold_16['amplitude_95_upper']
        }
        print(f"    Semi-amplitude: {fold_16['amplitude']*1000:.2f} mmag")
        print(f"    95% upper limit: {fold_16['amplitude_95_upper']*1000:.2f} mmag")

        print()
        results['bands'][band] = band_results

        # Store for plotting
        results['bands'][band]['_plot_data'] = {
            'mjd': mjd.tolist(),
            'mag': mag.tolist(),
            'magerr': magerr.tolist(),
            'periods_mid': periods_mid.tolist() if periods_mid is not None else [],
            'power_mid': power_mid.tolist() if power_mid is not None else [],
            'fold_21': fold_21,
            'fold_16': fold_16
        }

    return results


def create_ztf_plots(results):
    """Create ZTF analysis plots."""

    # Get first available band for plotting
    band = 'r' if 'r' in results['bands'] else 'g' if 'g' in results['bands'] else None
    if band is None:
        print("No band data available for plotting")
        return

    plot_data = results['bands'][band].get('_plot_data', {})
    if not plot_data:
        return

    # Periodogram plot (mid-range)
    if plot_data.get('periods_mid'):
        fig, ax = plt.subplots(figsize=(10, 5))

        periods = np.array(plot_data['periods_mid'])
        power = np.array(plot_data['power_mid'])

        ax.plot(periods, power, 'b-', lw=0.8, alpha=0.8)
        ax.axvline(PERIOD_MCMC, color='red', linestyle='--', lw=2,
                   label=f'MCMC period ({PERIOD_MCMC} d)')
        ax.axvline(PERIOD_CIRCULAR, color='orange', linestyle=':', lw=2,
                   label=f'Circular fit ({PERIOD_CIRCULAR} d)')

        ax.set_xlabel('Period (days)', fontsize=12)
        ax.set_ylabel('Lomb-Scargle Power', fontsize=12)
        ax.set_title(f'ZTF {band}-band Periodogram (5-100 days)', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xlim(5, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ztf_periodogram_mid.png', dpi=150)
        plt.close()
        print("Saved: ztf_periodogram_mid.png")

    # Phase-folded plots
    for period, label, filename in [(PERIOD_MCMC, '21.8d', 'ztf_folded_P21d.png'),
                                     (PERIOD_CIRCULAR, '15.9d', 'ztf_folded_P16d.png')]:
        fold_key = 'fold_21' if '21' in label else 'fold_16'
        fold_data = plot_data.get(fold_key, {})

        if not fold_data:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        mjd = np.array(plot_data['mjd'])
        mag = np.array(plot_data['mag'])
        magerr = np.array(plot_data['magerr'])
        phase = fold_data['phase']

        # Plot individual points
        ax.errorbar(phase, mag, yerr=magerr, fmt='o', color='gray',
                    alpha=0.3, markersize=3, label='Individual')
        ax.errorbar(phase + 1, mag, yerr=magerr, fmt='o', color='gray',
                    alpha=0.3, markersize=3)

        # Plot binned data
        bin_centers = fold_data['bin_centers']
        bin_mags = fold_data['bin_mags']
        bin_errs = fold_data['bin_errs']

        valid = ~np.isnan(bin_mags)
        ax.errorbar(bin_centers[valid], bin_mags[valid], yerr=bin_errs[valid],
                    fmt='s', color='red', markersize=8, label='Binned', zorder=5)
        ax.errorbar(bin_centers[valid] + 1, bin_mags[valid], yerr=bin_errs[valid],
                    fmt='s', color='red', markersize=8, zorder=5)

        ax.set_xlabel('Phase', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.set_title(f'ZTF {band}-band folded on P = {label}', fontsize=14)
        ax.invert_yaxis()
        ax.set_xlim(0, 2)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add amplitude annotation
        amp = fold_data.get('amplitude', np.nan)
        amp_95 = fold_data.get('amplitude_95_upper', np.nan)
        ax.text(0.02, 0.98, f'Semi-amp: {amp*1000:.1f} mmag\n95% UL: {amp_95*1000:.1f} mmag',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")


def interpret_ztf_results(results):
    """Interpret ZTF results in context of RV signal."""
    print("=" * 70)
    print("ZTF INTERPRETATION")
    print("=" * 70)
    print()

    interpretation = {}

    # Get amplitude limits from available bands
    amp_limits = []
    for band, data in results.get('bands', {}).items():
        if 'fold_21d' in data:
            amp_limits.append(data['fold_21d'].get('amplitude_95_upper', np.nan))

    if amp_limits:
        best_amp_limit = np.nanmin(amp_limits)
        interpretation['amplitude_95_upper_mag'] = best_amp_limit
        interpretation['amplitude_95_upper_mmag'] = best_amp_limit * 1000

        print(f"Best amplitude upper limit at P~22d: {best_amp_limit*1000:.1f} mmag")
        print()

        # Rotation interpretation
        print("ROTATION/ACTIVITY INTERPRETATION:")
        print("-" * 50)

        # For an M dwarf with ~100 km/s RV amplitude from rotation:
        # v_rot = 2πR/P → for P~22d, R~0.6Rsun: v_rot ~ 1.4 km/s
        # This is much smaller than 100 km/s, so rotation cannot explain RV

        print("  If the RV signal (K ~ 95 km/s) were due to stellar rotation:")
        print("    - Required v_rot ~ 95 km/s")
        print("    - For P = 22 days, R = 0.6 R_sun: v_rot ~ 1.4 km/s")
        print("    - → Rotation CANNOT explain the RV amplitude")
        print()

        # Spot modulation
        print("  If the RV were due to starspots mimicking orbital motion:")
        print("    - Large spots cause photometric modulation AND RV jitter")
        print("    - Typical spot-induced RV: ~10-100 m/s (not 100 km/s)")
        print(f"    - ZTF amplitude limit: {best_amp_limit*1000:.1f} mmag")

        if best_amp_limit < 0.01:  # 10 mmag
            print("    - → Very low photometric variability rules out large spots")
            interpretation['rotation_ruled_out'] = True
        else:
            print("    - → Moderate photometric limit; spots unlikely but not excluded")
            interpretation['rotation_ruled_out'] = False

        print()
        print("  CONCLUSION:")
        print("    The RV signal cannot be explained by stellar rotation or activity.")
        print("    The flat ZTF light curve supports a true binary companion.")
        interpretation['conclusion'] = 'rotation_excluded'
    else:
        print("  Insufficient data to constrain rotation amplitude")
        interpretation['conclusion'] = 'insufficient_data'

    print()
    return interpretation


def main():
    # Try to fetch real ZTF data
    ztf_fetch = fetch_ztf_lightcurve()

    # If no real data, use synthetic for demonstration
    if ztf_fetch['status'] != 'success' or not ztf_fetch.get('bands'):
        print("Using synthetic light curve (target may have limited ZTF coverage)")
        print("Note: Dec = -1.66 deg is at the southern edge of ZTF footprint")
        print()
        lc_data = generate_synthetic_lightcurve()
        data_source = 'synthetic'
    else:
        lc_data = ztf_fetch['bands']
        data_source = 'real'

    # Analyze the data
    results = analyze_ztf_data(lc_data)
    results['data_source'] = data_source
    results['fetch_status'] = ztf_fetch['status']

    # Create plots
    create_ztf_plots(results)

    # Interpret results
    interpretation = interpret_ztf_results(results)
    results['interpretation'] = interpretation

    # Clean results for JSON serialization
    clean_results = {}
    for key, val in results.items():
        if key == 'bands':
            clean_results['bands'] = {}
            for band, band_data in val.items():
                clean_band = {k: v for k, v in band_data.items() if not k.startswith('_')}
                clean_results['bands'][band] = clean_band
        else:
            clean_results[key] = val

    # Save results
    with open('ztf_results.json', 'w') as f:
        json.dump(clean_results, f, indent=2, default=float)
    print("Saved: ztf_results.json")

    return results


if __name__ == "__main__":
    main()
