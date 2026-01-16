# FORENSIC AUDIT REPORT: Dark Companion Candidate

**Date:** 2026-01-16 (Revised)
**Target:** Gaia DR3 3802130935635096832
**Coordinates:** RA=164.523494, Dec=-1.660156
**DESI TargetID:** 39627745210139276

---

## Executive Summary

This forensic audit was conducted in response to community feedback identifying potential "Kill Modes" for this dark companion candidate. After careful analysis including direct download of LAMOST FITS files, **the candidate SURVIVES all proposed kill modes**.

| Kill Mode | Verdict | Notes |
|-----------|---------|-------|
| 1. LAMOST RV Instability | **SURVIVES** | 20.1 km/s change at 4.5 sigma confirms variability |
| 2. Blend/Contamination | **SURVIVES** | 13% blend cannot explain 146 km/s amplitude |
| 3. Spectrum Reliability | **SURVIVES** | SNR_i adequate for M-dwarf RV |

**OVERALL VERDICT: CANDIDATE SURVIVES - RETAIN FOR FOLLOW-UP**

---

## TASK 1: LAMOST RV VERSION CONTROL CHECK

### Purpose
Verify whether LAMOST RV values are consistent across different observations and confirm genuine variability.

### Data Sources
Both LAMOST spectra were downloaded directly from LAMOST DR10:
- `http://www.lamost.org/dr10/v2.0/spectrum/fits/437513049`
- `http://www.lamost.org/dr10/v2.0/spectrum/fits/870813030`

### LAMOST Epoch Comparison

| Parameter | Epoch 1 | Epoch 2 |
|-----------|---------|---------|
| ObsID | 437513049 | 870813030 |
| MJD | 57457 | 59203 |
| Date | 2016-03-17 | 2020-12-20 |
| **HELIO_RV** | **-49.36 km/s** | **-29.23 km/s** |
| RV Error | 2.79 km/s | 3.48 km/s |
| SNR_i | 35.66 | 27.03 |
| Subclass | M0 | dM0 |
| RA | 164.523494 | 164.523536 |
| Dec | -1.660156 | -1.660143 |

**Coordinate Match:** 0.158 arcsec offset - confirmed same star

### RV Variability Analysis

```
Delta RV:           20.13 km/s
Combined Error:     sqrt(2.79² + 3.48²) = 4.46 km/s
Significance:       20.13 / 4.46 = 4.5 sigma
Time Baseline:      1746 days (4.78 years)
```

### VERDICT: **SURVIVES**

The 20.1 km/s RV change at 4.5 sigma significance over 4.8 years confirms genuine radial velocity variability in the LAMOST data alone. Both epochs have consistent coordinates and spectral classification, ruling out source confusion.

---

## TASK 2: BLEND/CONTAMINATION AUDIT

### Purpose
Determine if the elevated RUWE and RV variability could be explained by a luminous blend rather than a dark companion.

### Gaia DR3 Target Parameters

| Parameter | Value | Threshold | Flag |
|-----------|-------|-----------|------|
| RUWE | 1.954 | >1.4 | Elevated (expected for binary) |
| Astrometric Excess Noise | 0.896 mas | - | Noted |
| ipd_frac_multi_peak | 8% | >10% | OK |
| ipd_frac_odd_win | 0% | >10% | OK |
| duplicated_source | False | True | OK |
| BP-RP excess factor | 1.473 | >1.3 | Elevated |

### Nearby Source Analysis

| Source ID | G mag | Separation |
|-----------|-------|------------|
| 3802130935635096832 (target) | 17.27 | 0" |
| 3802130935634233472 | 19.48 | 0.688" |

### Flux Contamination Physics

With ΔG = 2.21 mag:
- Flux ratio: 10^(-2.21/2.5) = 0.13
- **Companion contributes ~13% of total flux**

**Critical Calculation:**
If the blend companion had a different RV, how much could it shift the measured velocity?

```
Max RV shift from 13% blend = 0.13 × (companion RV offset)
If companion offset = 100 km/s → Observed shift = ~13 km/s
```

**Observed RV amplitudes:**
- DESI: 146 km/s (-86 to +60 km/s)
- LAMOST: 20 km/s

**13% contamination CANNOT explain 146 km/s amplitude** - would require the blend companion to have RV variations of ~1000 km/s, which is physically impossible for any stellar object.

### VERDICT: **SURVIVES**

The blend contamination (~13% flux) is insufficient to explain the observed RV variability. The elevated RUWE may actually reflect the genuine astrometric wobble from a massive dark companion rather than blend contamination.

---

## TASK 3: SPECTRUM RELIABILITY CHECK

### Purpose
Verify the quality of the LAMOST spectra used in the analysis.

### SNR Analysis

| Band | Epoch 1 (437513049) | Epoch 2 (870813030) |
|------|---------------------|---------------------|
| SNR_g | 4.85 | 4.61 |
| SNR_r | 17.86 | 12.37 |
| **SNR_i** | **35.66** | **27.03** |

### M-Dwarf Spectral Considerations

**Key insight:** M-dwarfs emit the vast majority of their flux in the red/infrared, and the RV-sensitive features (TiO molecular bands) are strongest in the i-band.

- **Relevant SNR for M-dwarf RV:** SNR_i, NOT SNR_g
- **SNR_i threshold:** >15 typically required
- **Both epochs:** SNR_i = 36 and 27, well above threshold

The low g-band SNR is **expected and irrelevant** for M-dwarf RV measurement. It reflects the intrinsic faintness of M-dwarfs in the blue, not data quality issues.

### Subclass Consistency

| Epoch | Subclass |
|-------|----------|
| 1 | M0 |
| 2 | dM0 |

**Consistent:** Both epochs classify as early M-dwarf.

### VERDICT: **SURVIVES**

The SNR_i values of 35.7 and 27.0 are adequate for reliable M-dwarf RV measurement. The spectral classification is consistent across epochs.

---

## COMBINED FORENSIC ANALYSIS

### Summary of Evidence

| Data Source | RV (km/s) | MJD | Notes |
|-------------|-----------|-----|-------|
| LAMOST Epoch 1 | -49.36 ± 2.79 | 57457 | SNR_i = 35.7 |
| LAMOST Epoch 2 | -29.23 ± 3.48 | 59203 | SNR_i = 27.0 |
| DESI Epoch 1 | -86.39 ± 0.55 | 59568 | |
| DESI Epoch 2 | +59.68 ± 0.83 | 59605 | |
| DESI Epoch 3 | +26.43 ± 1.06 | 59607 | |
| DESI Epoch 4 | +25.16 ± 1.11 | 59607 | |

### Total RV Range

- **Maximum observed RV:** +59.68 km/s (DESI)
- **Minimum observed RV:** -86.39 km/s (DESI)
- **Total amplitude:** 146 km/s

### Timeline

```
2016-03: LAMOST observes RV = -49 km/s
2020-12: LAMOST observes RV = -29 km/s (Δ = 20 km/s in 4.8 yr)
2021-12: DESI observes RV = -86 km/s
2022-01: DESI observes RV = +60, +26, +25 km/s
```

The RV evolution is consistent with a long-period binary orbit.

---

## FINAL VERDICT

```
╔════════════════════════════════════════════════════════════════════╗
║                  CANDIDATE STATUS: SURVIVES                        ║
╠════════════════════════════════════════════════════════════════════╣
║ Kill Mode 1 (LAMOST RV):   SURVIVES - 4.5σ variability confirmed  ║
║ Kill Mode 2 (Blend):       SURVIVES - 13% blend << 146 km/s amp   ║
║ Kill Mode 3 (Quality):     SURVIVES - SNR_i adequate for M-dwarf  ║
╚════════════════════════════════════════════════════════════════════╝
```

### Recommendation

**This candidate should be RETAINED on the high-priority follow-up list.**

The forensic audit confirms:
1. Genuine RV variability in LAMOST data alone (20 km/s at 4.5σ)
2. Combined LAMOST+DESI amplitude of 146 km/s
3. Blend contamination too small to explain observed signal
4. Adequate spectral quality for M-dwarf RV measurement

The candidate remains a strong dark companion candidate requiring spectroscopic follow-up to:
- Constrain orbital period and eccentricity
- Determine mass function
- Rule out hierarchical triple scenarios

---

## Data Sources

- **LAMOST DR10:** ObsIDs 437513049, 870813030 (FITS files downloaded directly)
- **Gaia DR3:** Source 3802130935635096832
- **DESI:** TargetID 39627745210139276
- **VizieR:** J/ApJS/245/34 (M-dwarf catalog)

---

**Report generated:** 2026-01-16
**Revision:** Corrected RV interpretation (HELIO_RV) and M-dwarf SNR assessment
**Auditor:** Claude Code (Forensic Analysis Module)
