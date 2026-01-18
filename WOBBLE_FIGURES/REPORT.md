# Wobble Imaging Analysis Report

**Target:** Gaia DR3 3802130935635096832
**Analysis Date:** 2026-01-17

## Target Properties (Gaia DR3)

| Property | Value |
|----------|-------|
| RA (J2016) | 164.5235° |
| Dec (J2016) | -1.6602° |
| G magnitude | 17.27 |
| BP-RP color | 1.89 |
| Parallax | 0.12 ± 0.16 mas |
| Distance (assumed) | ~1 kpc |
| pmRA | -7.60 ± 0.17 mas/yr |
| pmDec | 3.00 ± 0.13 mas/yr |
| RUWE | 1.95 |
| Astrometric excess noise sig | 16.5 |

### Gaia Astrometric Flags

The elevated RUWE (1.95 > 1.4 threshold) and high astrometric excess noise significance (16.5) suggest the Gaia astrometry is perturbed. This is consistent with either:
- An unseen companion causing astrometric wobble
- Unresolved binarity or source confusion
- Problematic astrometry due to crowding

## Multi-Epoch Imaging

**Survey:** Pan-STARRS1 (PS1)
**Image Type:** Warp (single-exposure) cutouts
**Cutout Size:** 30" × 30" (120 × 120 pixels)
**Filters:** g, r, i

| Parameter | Value |
|-----------|-------|
| Total images queried | 71 |
| Images downloaded | 50 |
| Valid measurements | 38 |
| After outlier rejection | 32 |
| MJD range | 55242 - 56999 |
| Baseline | 4.81 years |

## Centroid Measurements

Centroids were measured using 2D Gaussian fitting in each PS1 warp image. Outlier rejection removed frames with large centroid deviations (>3 MAD).

### Wobble Statistics

| Axis | RMS (mas) | Range (mas) |
|------|-----------|-------------|
| X (RA) | 16.2 | 71.9 |
| Y (Dec) | 33.7 | 142.1 |
| Total | 37.4 | - |

### Proper Motion (from PS1 centroids)

| Axis | Rate (mas/yr) |
|------|---------------|
| X | 4.7 |
| Y | 9.2 |

Note: These differ from Gaia because PS1 centroid precision is ~50-100 mas for G=17 sources, and systematic effects may dominate.

## Wobble Detection Assessment

The measured centroid RMS of 37.4 mas is **below the typical PS1 centroid precision** (~100 mas for a G=17.3 source).

**Conclusion:** The PS1 multi-epoch imaging does not provide sufficient precision to detect or constrain astrometric wobble at the level implied by the Gaia excess noise.

### Upper Limits

At the assumed distance of 1 kpc:
- Wobble amplitude < 37.4 mas corresponds to < 37.4 AU physical separation
- This is not constraining for typical compact object companions

## Figures

1. **wobble_timeseries.png** - X and Y centroid offsets vs time
2. **centroid_track.png** - Centroid positions in X-Y plane with time coloring
3. **residual_histograms.png** - Distribution of X and Y residuals
4. **centroid_by_filter.png** - Centroid positions colored by filter
5. **multi_epoch_panels.png** - 3×3 panel of representative epochs
6. **blink_animation.gif** - Animated blink of 20 epochs

## Recommendations

1. **Gaia epoch astrometry:** The elevated RUWE and astrometric excess noise are more sensitive probes of wobble than ground-based imaging. Request Gaia epoch astrometry if available.

2. **High-precision imaging:** HST or AO-assisted ground imaging could achieve ~1-5 mas precision, sufficient to detect wobble at the ~0.9 mas level (Gaia excess noise).

3. **Radial velocity monitoring:** Spectroscopic RV monitoring would provide an independent constraint on companion mass and period.

## Data Products

| File | Description |
|------|-------------|
| `wobble_limits.json` | Quantitative wobble limits |
| `astrometry_timeseries.csv` | Centroid measurements |
| `gaia_query.json` | Full Gaia DR3 query results |
| `ps1_metadata.json` | PS1 image metadata |
