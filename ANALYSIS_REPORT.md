# Comprehensive Analysis Report: Gaia DR3 3802130935635096832

**Date:** 2026-01-15
**Target:** Gaia DR3 3802130935635096832
**DESI TargetID:** 39627745210139276

---

## A) CLAIM LEDGER

| ID | Status | Claim | Location | Artifact | Notes |
|----|--------|-------|----------|----------|-------|
| 1 | **WRONG** | S_robust = 100.0 | Table 1, p6 | compute_rv_dossier.py | Paper: 100.0, Repo: **19.78**. 5x discrepancy. |
| 2 | VERIFIED | ΔRV ≈ 146 km/s | Abstract | compute_rv_dossier.py | Computed: 146.07 km/s |
| 3 | VERIFIED | RUWE = 1.95 | Sec 5.1.3 | validation_results_full.csv | Gaia: 1.9536 |
| 4 | VERIFIED | σ_AEN = 16.5 | Sec 5.1.3 | validation_results_full.csv | Gaia: 16.49 |
| 5 | VERIFIED | W1-W2 = 0.052 | Sec 5.1.3 | validation_results_full.csv | WISE: 0.052 |
| 6 | VERIFIED | K ≈ 73 km/s | Sec 6 | compute_rv_dossier.py | K_est = 73.04 km/s |
| 7 | VERIFIED | N = 4 epochs | Table 1 | FITS data | Confirmed |
| 8 | VERIFIED | Baseline = 39 days | Sec 5.1 | compute_rv_dossier.py | 38.90 days |
| 9 | VERIFIED | "Violent RV variations" | Sec 5.1 | N/A | Qualitative, reasonable |
| 10 | UNVERIFIED | GALEX rules out hot WD | Sec 5.1.2 | Manual inspection | Need reproducible script |
| 11 | VERIFIED | TESS no eclipses | Sec 5.1.3 | analyze_tess_photometry.py | LS power = 0.0014 |
| 12 | **WRONG** | "High-confidence dark companion" | Sec 4 | compute_rv_dossier.py | d_max=113, single epoch dominates |
| 13 | UNVERIFIED | Primary ~0.7 M_sun K dwarf | Sec 6 | None | Parallax too uncertain |
| 14 | **WRONG** | "Massive companion" required | Sec 6 | orbit_feasibility.py | M2_min ~0.7-1.3 M_sun, could be WD |
| 15 | **WRONG** | Verdict: "Dark Companion" | Table 1 | orbit_feasibility.py | Should be "Candidate" |

**Summary:** 9 VERIFIED, 2 UNVERIFIED, **4 WRONG**

---

## B) CANDIDATE DOSSIER

### Per-Epoch RV Table (EXACT VALUES)

| # | MJD | RV (km/s) | σRV (km/s) |
|---|-----|-----------|------------|
| 0 | 59568.48825 | -86.39 | 0.55 |
| 1 | 59605.38003 | +59.68 | 0.83 |
| 2 | 59607.37393 | +26.43 | 1.06 |
| 3 | 59607.38852 | +25.16 | 1.11 |

### Derived Quantities

| Quantity | Value | Definition |
|----------|-------|------------|
| ΔRV_max | 146.07 km/s | max(RV) - min(RV) |
| K_est | 73.04 km/s | ΔRV_max / 2 |
| RV_wmean | -24.00 km/s | Weighted mean RV |
| χ²(constant) | 27257.21 | Chi-squared for constant RV model |
| χ²_reduced | 9085.74 | Constant RV model strongly rejected |
| MJD span | 38.90 days | Observational baseline |

### RV Selection Metrics (Pipeline Definitions)

**Equation: S (Global Significance)**
```
S = ΔRV_max / sqrt(Σᵢ σ²_RV,i)
```

| Symbol | Definition | Value |
|--------|------------|-------|
| S | Global RV significance | 79.84 |
| ΔRV_max | Max RV range | 146.07 km/s |
| σ_RV,i | Error on epoch i | 0.55-1.11 km/s |

**Equation: S_min_LOO (Leave-One-Out)**
```
S_min_LOO = min over all i of S(epochs excluding i)
```

| Symbol | Definition | Value |
|--------|------------|-------|
| S_min_LOO | Min LOO significance | **19.78** |
| Dropped epoch | Epoch giving minimum | 0 (MJD 59568) |

**Equation: S_robust**
```
S_robust = min(S, S_min_LOO)
```

| Symbol | Definition | Value |
|--------|------------|-------|
| S_robust | Conservative significance | **19.78** |

**Equation: d_max (Leverage)**
```
d_max = max over all i of |RV_i - RV_wmean| / σ_RV,i
```

| Symbol | Definition | Value |
|--------|------------|-------|
| d_max | Maximum leverage | **113.44** |
| High leverage? | d_max > 100 | **YES** |
| Dominant epoch | Epoch with highest leverage | 0 |

### Night Consistency Check

Epochs 2 & 3 observed 21 minutes apart:
- ΔRV = 1.27 ± 1.53 km/s
- Significance = 0.83σ
- **CONSISTENT** (< 3σ)

### Critical Finding

**The paper claims S_robust = 100, but the correct value is S_robust = 19.78.**

The first epoch (MJD 59568.48825, RV = -86.39 km/s) has **extreme leverage** (d_max = 113). When this single epoch is removed, the remaining 3 epochs span only 60 km/s with S = 19.78.

---

## C) ORBIT FEASIBILITY

### Period Constraints from RV Evolution

| Transition | Δt (days) | ΔRV (km/s) | Constraint |
|------------|-----------|------------|------------|
| Epoch 0→1 | 36.89 | +146.07 | Major swing |
| Epoch 1→2 | 1.99 | -33.25 | Slope constraint |
| Epoch 2→3 | 0.015 | -1.27 | Night consistency |

**Circular orbit bounds:**
- If epochs 0→1 span half-period: P ≈ 74 days
- From RV slope at epoch 1→2: P ≈ 28 days
- **Conservative range: P = 25-80 days**

**Eccentric orbit relaxation:**
- High eccentricity allows faster RV swings
- P could be as short as ~10 days with e > 0.5

### Mass Function Analysis

**Equation:**
```
f(M) = (M₂ sin i)³ / (M₁ + M₂)² = P K³ / (2π G)
```

| Symbol | Definition | Units |
|--------|------------|-------|
| f(M) | Mass function | M_sun |
| P | Orbital period | days |
| K | RV semi-amplitude | km/s |
| M₁ | Primary mass | M_sun |
| M₂ | Companion mass | M_sun |
| i | Inclination | rad |

Using K_est = 73.04 km/s:

| P (days) | f(M) (M_sun) | M2_min (M1=0.5) | M2_min (M1=0.7) | M2_min (M1=1.0) |
|----------|--------------|-----------------|-----------------|-----------------|
| 20 | 0.81 | 1.46 | 1.64 | 1.89 |
| 30 | 1.21 | 1.92 | 2.14 | 2.42 |
| 40 | 1.61 | 2.37 | 2.60 | 2.91 |
| 50 | 2.02 | 2.80 | 3.05 | 3.39 |
| 80 | 3.23 | 4.07 | 4.35 | 4.74 |

### Companion Type Thresholds

| M₁ (M_sun) | P_min for NS (M₂>1.4) | P_min for BH (M₂>3.0) |
|------------|----------------------|----------------------|
| 0.5 | 18.9 days | 54.8 days |
| 0.7 | 15.5 days | 48.9 days |
| 1.0 | 11.8 days | 41.9 days |

**Key Finding:** For P ≈ 30-50 days (plausible range), M₂_min ≈ 1.6-3.0 M_sun. This is consistent with WD, NS, OR BH. **Cannot distinguish without period.**

---

## D) NEGATIVE SPACE REALITY CHECK

### 1. WISE/2MASS (Infrared)

**Result:** W1 - W2 = 0.052 mag (consistent with zero)

**What this rules out:**
- ✗ M dwarf companion (would show W1-W2 > 0.2)
- ✗ Brown dwarf companion
- ✗ Circumstellar dust disk

**What this does NOT rule out:**
- ✓ White dwarf (negligible IR flux)
- ✓ Neutron star (zero IR flux)
- ✓ Black hole (zero IR flux)

### 2. GALEX (Ultraviolet)

**Result:** Non-detection in NUV

**What this rules out:**
- ✗ Hot WD (T > 10,000 K would be NUV-bright)
- ✗ Young WD (recently formed, still hot)

**What this does NOT rule out:**
- ✓ Cool WD (T < 6,000 K)
- ✓ Old WD (T ~ 4,000-5,000 K)
- ✓ NS/BH (no UV emission)

**Why cool WDs are invisible:**
A 5000 K WD with R ~ 0.01 R_sun at 1 kpc has G ~ 28 mag, far below detection.

### 3. TESS (Photometry)

**Result:**
- 37,832 data points
- Scatter = 6.32 ppt
- Peak LS power = 0.0014 (not significant)
- No eclipses detected

**Upper limit on ellipsoidal amplitude:**
```
A_ellip (95% CL) < 0.46 ppt
```

**What this constrains:**
- ✓ No deep eclipses (no edge-on transiting system)
- ✓ No contact binary
- ✓ No short-period (P < 1 day) binary

**What this does NOT constrain:**
- ✗ Detached binaries with P > 10 days (expected A_ellip < 0.1 ppt)
- ✗ Cannot distinguish WD/NS/BH

### 4. Legacy Survey Imaging

**Status:** Manual inspection only (Figures 1a, 1b in paper)
**Finding:** Source appears isolated, no blending

**Recommendation:** Add reproducible script to save cutouts and measure PSF residuals.

---

## E) PAPER SURGERY PLAN

### Critical Fixes Required

#### 1. Table 1: S_robust Value
**Location:** Page 6, Table 1
**Current:** S_robust = 100.0
**Correct:** S_robust = 19.78
**Action:** Replace "100.0" with "19.8" and add footnote explaining high-leverage flag

#### 2. Table 1: Verdict Column
**Location:** Page 6, Table 1
**Current:** "Dark Companion"
**Correct:** "Dark Companion Candidate"
**Action:** Change verdict and add caveat about period uncertainty

#### 3. Section 4 Title/Language
**Location:** Section 4, Page 2
**Current:** "high-confidence dark companion"
**Correct:** "candidate dark companion system"
**Action:** Remove "high-confidence" throughout

#### 4. Section 6 Discussion
**Location:** Page 6, paragraph 2
**Current:** "the high velocity semi-amplitude (K ≈ 73 km/s) requires a massive companion"
**Correct:** "the high velocity semi-amplitude (K ≈ 73 km/s) is consistent with a massive companion, though the minimum mass depends on the unknown period"

#### 5. Add Limitations Section (NEW)
**Location:** After Section 6, before Conclusion
**Content:**
```
7 Limitations

Several important caveats apply to this candidate:

1. No Period Determination: Without an orbital period, we cannot compute
   the mass function to derive a dynamical companion mass. The 4 DESI
   epochs constrain the period to approximately 25-80 days but do not
   uniquely determine it.

2. High-Leverage Epoch: The RV significance is dominated by a single
   epoch (MJD 59568.48825). The leave-one-out robust significance
   S_robust = 19.8 is substantially lower than the global S = 79.8.

3. Companion Type Ambiguity: The negative-space analysis rules out
   M-dwarf and hot WD companions but cannot distinguish between a
   cool white dwarf, neutron star, or black hole.

4. Primary Mass Uncertainty: The parallax (0.12 ± 0.16 mas) is too
   uncertain to constrain the primary mass spectroscopically. We
   assume M₁ ~ 0.5-1.2 M_sun based on colors.

5. RUWE Interpretation: While RUWE = 1.95 indicates astrometric
   non-single-star behavior, it is not a direct mass measurement
   and could arise from other causes.
```

#### 6. Add Per-Epoch RV Table (NEW)
**Location:** Section 5.1 or Appendix
**Content:**
```
Table 2: Per-Epoch Radial Velocities for Gaia DR3 3802130935635096832

| Epoch | MJD | RV (km/s) | σRV (km/s) |
|-------|-----|-----------|------------|
| 1 | 59568.48825 | -86.39 | 0.55 |
| 2 | 59605.38003 | +59.68 | 0.83 |
| 3 | 59607.37393 | +26.43 | 1.06 |
| 4 | 59607.38852 | +25.16 | 1.11 |
```

#### 7. Add Metrics Table (NEW)
**Location:** Section 5.1
**Content:**
```
Table 3: RV Variability Metrics

| Metric | Value | Note |
|--------|-------|------|
| S (global) | 79.8 | |
| S_min_LOO | 19.8 | Epoch 0 dropped |
| S_robust | 19.8 | min(S, S_LOO) |
| d_max | 113.4 | High leverage |
```

---

## F) REPO IMPROVEMENTS

### Required New Scripts

#### 1. scripts/compute_rv_dossier.py ✓ CREATED
- Computes S, S_LOO, S_robust, d_max for any target
- Outputs JSON dossier with all metrics
- Includes night consistency check

#### 2. scripts/orbit_feasibility.py ✓ CREATED
- Period constraint analysis from RV evolution
- Mass function computation
- Companion type threshold table
- Output: mass_function_analysis.png

#### 3. scripts/tess_ellipsoidal_limits.py ✓ CREATED
- Converts TESS scatter to amplitude upper limit
- Computes ellipsoidal constraints in (q, P) space
- Output: tess_ellipsoidal_constraints.png

#### 4. scripts/sed_companion_limits.py ✓ CREATED
- SED fitting from Gaia+2MASS+WISE
- Companion flux upper limits
- Output: sed_analysis.png

#### 5. scripts/claims_checker.py ✓ CREATED
- Validates paper claims against repo values
- Produces claim ledger table
- Flags discrepancies

#### 6. scripts/galex_check.py (TO CREATE)
```python
# Query GALEX archive for target
# Save cutout image
# Compute detection limit
# Output: galex_cutout.png, galex_limits.json
```

#### 7. scripts/legacy_imaging_check.py (TO CREATE)
```python
# Fetch Legacy Survey cutout
# Measure PSF residuals
# Check for blending
# Output: legacy_cutout.png, psf_residuals.json
```

#### 8. scripts/export_candidate_table.py (TO CREATE)
```python
# Generate publication-ready tables
# Include all metrics with uncertainties
# LaTeX format output
```

#### 9. scripts/validate_all_candidates.py (TO CREATE)
```python
# Run full validation pipeline on all candidates
# Flag high-leverage cases
# Generate summary report
```

#### 10. Makefile target: make validate
```makefile
validate:
    python scripts/compute_rv_dossier.py
    python scripts/claims_checker.py
    python scripts/orbit_feasibility.py
```

---

## G) GO/NO-GO RECOMMENDATION

### **GO** — with mandatory corrections

The paper can proceed to submission IF AND ONLY IF:

1. **Table 1 is corrected:** S_robust = 19.8 (not 100.0)

2. **Verdict is downgraded:** "Dark Companion" → "Dark Companion Candidate"

3. **Limitations section is added** explaining:
   - No period → no dynamical mass
   - Single epoch dominates significance
   - Cannot rule out cool WD

4. **Per-epoch RV table is added** for reproducibility

5. **Language is moderated:**
   - Remove "high-confidence"
   - Remove "massive companion confirmed"
   - Add "candidate" qualifier throughout

### Defensible Claims After Correction

✓ High-amplitude RV binary candidate (ΔRV = 146 km/s)
✓ Significant astrometric wobble (RUWE = 1.95)
✓ No infrared excess (W1-W2 = 0.052)
✓ No hot WD companion (GALEX non-detection)
✓ No eclipses or large ellipsoidal variations (TESS)
✓ Companion is optically dark
✓ Follow-up observations required to determine period and mass

### Non-Defensible Claims (Must Remove)

✗ "Dark Companion" as definitive verdict
✗ S_robust = 100
✗ "Massive companion" without period
✗ BH/NS classification without dynamical mass

---

## Appendix: Artifact Locations

| Artifact | Path |
|----------|------|
| RV dossier script | scripts/compute_rv_dossier.py |
| Orbit analysis | scripts/orbit_feasibility.py |
| TESS constraints | scripts/tess_ellipsoidal_limits.py |
| SED analysis | scripts/sed_companion_limits.py |
| Claims checker | scripts/claims_checker.py |
| Validation results | validation_results_full.csv |
| TESS plot | tess_analysis_result.png |
| Mass function plot | mass_function_analysis.png |
| SED plot | sed_analysis.png |
| Dossier JSON | candidate_dossier.json |
