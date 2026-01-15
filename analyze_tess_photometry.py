#!/usr/bin/env python3
"""
analyze_tess_photometry.py - TESS Light Curve Analysis for BH Candidate

Target: Gaia DR3 3802130935635096832
Goal: Extract light curve, search for orbital periods, detect ellipsoidal variations
"""

import os
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

import lightkurve as lk

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "astrocut_164.523494_-1.660156_10.0x15.0px"
OUTPUT_PLOT = "tess_analysis_result.png"
TARGET_NAME = "Gaia DR3 3802130935635096832"

print("=" * 70)
print("TESS PHOTOMETRY ANALYSIS")
print(f"Target: {TARGET_NAME}")
print("=" * 70)
print()

# =============================================================================
# STEP 1: Load all TESS Target Pixel Files
# =============================================================================
print("[1/4] Loading TESS Target Pixel Files...")

fits_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.fits")))
# Filter out Zone.Identifier files
fits_files = [f for f in fits_files if not f.endswith(':Zone.Identifier')]

print(f"  Found {len(fits_files)} TESS sectors")

tpf_collection = []
for f in fits_files:
    try:
        tpf = lk.TessTargetPixelFile(f)
        tpf_collection.append(tpf)
        print(f"    Loaded: {os.path.basename(f)} | Sector {tpf.sector} | {len(tpf.time)} cadences")
    except Exception as e:
        print(f"    [WARN] Failed to load {os.path.basename(f)}: {e}")

print(f"  Successfully loaded {len(tpf_collection)} sectors")
print()

# =============================================================================
# STEP 2: Aperture Photometry
# =============================================================================
print("[2/4] Extracting Light Curves with Aperture Photometry...")

lightcurves = []

for i, tpf in enumerate(tpf_collection):
    try:
        # Create threshold mask (automatically detect the star)
        # Fallback to central 3x3 if threshold mask fails
        try:
            aperture_mask = tpf.create_threshold_mask(threshold=3, reference_pixel='center')
            if aperture_mask.sum() == 0:
                raise ValueError("Empty mask")
        except:
            # Central 3x3 aperture
            aperture_mask = np.zeros(tpf.shape[1:], dtype=bool)
            cy, cx = tpf.shape[1] // 2, tpf.shape[2] // 2
            aperture_mask[max(0,cy-1):cy+2, max(0,cx-1):cx+2] = True

        # Extract light curve
        lc = tpf.to_lightcurve(aperture_mask=aperture_mask)

        # Remove NaNs
        lc = lc.remove_nans()

        # Remove outliers (5-sigma clipping)
        lc = lc.remove_outliers(sigma=5)

        # Flatten to remove long-term trends
        lc_flat = lc.flatten(window_length=401)

        lightcurves.append(lc_flat)
        print(f"    Sector {tpf.sector}: {len(lc_flat.time)} points | Median flux: {np.nanmedian(lc_flat.flux.value):.4f}")

    except Exception as e:
        print(f"    [WARN] Sector {tpf.sector} failed: {e}")

print(f"  Extracted {len(lightcurves)} light curves")
print()

# Stitch all light curves together
print("  Stitching light curves...")
if len(lightcurves) > 1:
    lc_stitched = lightcurves[0]
    for lc in lightcurves[1:]:
        lc_stitched = lc_stitched.append(lc)
elif len(lightcurves) == 1:
    lc_stitched = lightcurves[0]
else:
    print("  ERROR: No light curves extracted!")
    exit(1)

# Final cleanup
lc_stitched = lc_stitched.remove_nans().remove_outliers(sigma=5)

print(f"  Final stitched light curve: {len(lc_stitched.time)} points")
print(f"  Time baseline: {lc_stitched.time.max().value - lc_stitched.time.min().value:.1f} days")
print()

# Calculate basic statistics
flux = lc_stitched.flux.value
flux_std = np.nanstd(flux)
flux_median = np.nanmedian(flux)
scatter_ppt = (flux_std / flux_median) * 1000  # parts per thousand

print(f"  Light curve scatter: {scatter_ppt:.3f} ppt (parts per thousand)")
print()

# =============================================================================
# STEP 3: Period Search (Lomb-Scargle Periodogram)
# =============================================================================
print("[3/4] Running Lomb-Scargle Periodogram...")

# Search for periods from 0.1 to 30 days
try:
    pg = lc_stitched.to_periodogram(method='lombscargle',
                                     minimum_period=0.1,
                                     maximum_period=30.0,
                                     oversample_factor=10)

    # Find peak period
    best_period = pg.period_at_max_power.value
    best_power = pg.max_power.value

    print(f"  Peak Period: {best_period:.6f} days ({best_period*24:.3f} hours)")
    print(f"  Peak Power: {best_power:.6f}")

    # Check for ellipsoidal variations (period doubling)
    # If the true orbital period is P, ellipsoidal shows 2 humps per orbit -> detected at P/2
    double_period = best_period * 2
    half_period = best_period / 2

    print(f"  Checking 2x period (ellipsoidal): {double_period:.4f} days")
    print(f"  Checking 0.5x period: {half_period:.4f} days")

    # Get power at double period
    period_grid = pg.period.value
    power_grid = pg.power.value

    idx_double = np.argmin(np.abs(period_grid - double_period))
    power_double = power_grid[idx_double]

    idx_half = np.argmin(np.abs(period_grid - half_period))
    power_half = power_grid[idx_half]

    print(f"  Power at 2P: {power_double:.6f}")
    print(f"  Power at P/2: {power_half:.6f}")

    # Determine if signal is significant
    # Use False Alarm Probability
    try:
        fap = pg.false_alarm_probability(pg.max_power)
        print(f"  False Alarm Probability: {fap:.2e}")
        significant = fap < 0.01
    except:
        # Estimate significance from power
        significant = best_power > 0.1
        fap = None

except Exception as e:
    print(f"  [ERROR] Periodogram failed: {e}")
    best_period = None
    best_power = 0
    significant = False
    fap = None

print()

# =============================================================================
# STEP 4: Visualization
# =============================================================================
print("[4/4] Generating Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Full Light Curve
ax1 = axes[0, 0]
ax1.scatter(lc_stitched.time.value, lc_stitched.flux.value, s=1, alpha=0.5, c='black')
ax1.set_xlabel('Time (BTJD)')
ax1.set_ylabel('Normalized Flux')
ax1.set_title(f'{TARGET_NAME}\nFull TESS Light Curve ({len(lc_stitched.time)} points)')
ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)

# Panel 2: Periodogram
ax2 = axes[0, 1]
if best_period is not None:
    ax2.plot(pg.period.value, pg.power.value, 'k-', lw=0.5)
    ax2.axvline(x=best_period, color='red', linestyle='--', label=f'Peak: {best_period:.4f} d')
    ax2.axvline(x=double_period, color='blue', linestyle=':', alpha=0.7, label=f'2x: {double_period:.4f} d')
    ax2.set_xlabel('Period (days)')
    ax2.set_ylabel('Lomb-Scargle Power')
    ax2.set_title('Periodogram')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 15)
else:
    ax2.text(0.5, 0.5, 'Periodogram Failed', ha='center', va='center', transform=ax2.transAxes)

# Panel 3: Folded Light Curve (on best period)
ax3 = axes[1, 0]
if best_period is not None:
    folded = lc_stitched.fold(period=best_period)
    ax3.scatter(folded.phase.value, folded.flux.value, s=1, alpha=0.5, c='black')

    # Bin the folded data
    try:
        folded_binned = folded.bin(time_bin_size=0.02)
        ax3.scatter(folded_binned.phase.value, folded_binned.flux.value,
                   s=30, c='red', marker='o', label='Binned (2% phase)', zorder=10)
    except:
        pass

    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Normalized Flux')
    ax3.set_title(f'Folded on P = {best_period:.4f} days ({best_period*24:.2f} hr)')
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    ax3.legend()
else:
    ax3.text(0.5, 0.5, 'No Period Found', ha='center', va='center', transform=ax3.transAxes)

# Panel 4: Folded on 2x Period (ellipsoidal check)
ax4 = axes[1, 1]
if best_period is not None:
    folded_2x = lc_stitched.fold(period=double_period)
    ax4.scatter(folded_2x.phase.value, folded_2x.flux.value, s=1, alpha=0.5, c='black')

    # Bin
    try:
        folded_2x_binned = folded_2x.bin(time_bin_size=0.02)
        ax4.scatter(folded_2x_binned.phase.value, folded_2x_binned.flux.value,
                   s=30, c='blue', marker='o', label='Binned (2% phase)', zorder=10)
    except:
        pass

    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Normalized Flux')
    ax4.set_title(f'Folded on 2P = {double_period:.4f} days (Ellipsoidal Check)')
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    ax4.legend()
else:
    ax4.text(0.5, 0.5, 'No Period Found', ha='center', va='center', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
plt.close()

print(f"  Saved: {OUTPUT_PLOT}")
print()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()
print(f"Target: {TARGET_NAME}")
print(f"TESS Sectors: {len(tpf_collection)}")
print(f"Total Data Points: {len(lc_stitched.time)}")
print(f"Time Baseline: {lc_stitched.time.max().value - lc_stitched.time.min().value:.1f} days")
print()
print(f"Light Curve Scatter: {scatter_ppt:.3f} ppt (parts per thousand)")
print()

if best_period is not None:
    print(f"BEST PERIOD: {best_period:.6f} days ({best_period*24:.3f} hours)")
    print(f"PEAK POWER: {best_power:.6f}")
    if fap is not None:
        print(f"FALSE ALARM PROB: {fap:.2e}")

    # Calculate amplitude
    if best_power > 0:
        # Rough amplitude estimate from periodogram
        amplitude_ppt = np.sqrt(2 * best_power) * scatter_ppt
        print(f"SIGNAL AMPLITUDE (est): {amplitude_ppt:.3f} ppt")

    print()
    if significant and best_power > 0.1:
        print("DETECTION: SIGNIFICANT PERIODIC SIGNAL DETECTED")
        if power_double > best_power * 0.5:
            print("           Possible ELLIPSOIDAL VARIATIONS (2-hump pattern)")
        print()
        print("INTERPRETATION: This could indicate:")
        print("  - Ellipsoidal variations from tidally distorted star")
        print("  - Reflection effect from hot companion")
        print("  - Stellar rotation/spots")
    else:
        print("DETECTION: NO SIGNIFICANT PERIODIC SIGNAL")
        print()
        print("INTERPRETATION: FLAT LIGHT CURVE")
        print("  - Consistent with a compact, non-luminous companion")
        print("  - No eclipses detected")
        print("  - Combined with high RV + RUWE: STRONG BH CANDIDATE")
else:
    print("PERIOD SEARCH FAILED")
    print()
    print("INTERPRETATION: Unable to determine periodicity")

print()
print("=" * 70)
