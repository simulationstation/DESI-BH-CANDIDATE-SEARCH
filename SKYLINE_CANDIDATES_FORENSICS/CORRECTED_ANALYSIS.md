# CORRECTED FORENSIC ANALYSIS - SKEPTICAL REASSESSMENT

*Addressing feedback on statistical rigor and factual accuracy*

## Corrected Verdict Table

| Type   | Cosmic Ray | Real Signal | Ambiguous | Total |
|--------|------------|-------------|-----------|-------|
| STAR   | 10         | 1           | 2         | 13    |
| GALAXY | 1          | 67          | 3         | 71    |
| QSO    | 2          | 2           | 1         | 5     |
| TOTAL  | 13         | 70          | 6         | 89    |

## Wing Ratio Analysis

The wing ratio measures flux in neighboring pixels relative to the peak:
- < 0.1: COSMIC_RAY (single-pixel spike)
- 0.1-0.2: AMBIGUOUS (intermediate)
- > 0.2: REAL_SIGNAL (resolved Gaussian)

**By Source Type:**

| Type   | Mean Wing Ratio | Median | Range       |
|--------|-----------------|--------|-------------|
| STAR   | 0.24            | 0.24   | 0.24        |
| GALAXY | 0.55            | 0.56   | 0.33 - 0.70 |
| QSO    | 0.36            | 0.36   | 0.21 - 0.51 |

**Critical Finding:** The single STAR "REAL_SIGNAL" has wing ratio 0.24 - barely above the 0.2 threshold and **half** the Galaxy average of 0.55.

## Skeptical Reassessment

### STAR Candidate 39627997694657850 (9502.4 Å)

- Wing ratio: 0.24 (borderline, just above 0.2 threshold)
- SNR: 46.2
- Location: 9502 Å (Z-band red end)

**Concerns:**
1. Wing ratio 2x lower than Galaxy emission lines
2. Wavelength in telluric-contaminated region (>9000 Å)
3. 30 Å blueward of [SIII] 9532 - possible sky subtraction residual
4. No corroborating multi-epoch observations

**Assessment:** More likely a sky subtraction artifact than genuine signal.

### STAR Candidate 39633497471584886 (Two Signals)

- 6848.0 Å: wing ratio 0.15
- 6143.2 Å: wing ratio 0.17

**Concerns:**
1. Same star produces TWO ambiguous detections at different wavelengths
2. Pattern suggests stellar chromospheric activity or systematic error
3. Genuine technosignature would appear at ONE specific wavelength
4. Multiple weak signals = natural variability or artifacts

**Assessment:** Stellar activity or instrumental effect.

## Conclusion

After rigorous re-analysis:

- **GALAXY detections**: Genuine emission lines (Hα, [OIII], etc.) - astrophysical, not technosignatures
- **STAR "REAL_SIGNAL"**: Likely false positive due to borderline wing ratio and telluric contamination
- **STAR "AMBIGUOUS"**: Likely stellar activity given multiple weak detections

**VERDICT:** No statistically significant technosignature candidates remain after proper skeptical analysis.
