# DESI BLEND-AWARE RV RE-MEASUREMENT REPORT v3

**Date:** 2026-01-16
**Target:** Gaia DR3 3802130935635096832
**DESI TARGETID:** 39627745210139276

---

## Executive Summary

This report presents a rigorous blend-aware analysis of DESI DR1 spectra with **methodological improvements over v2**:

1. **RV uncertainties properly renormalized** by sqrt(χ²_red)
2. **BIC computed on data likelihood only** (no priors mixed in)
3. **Marginalization over amplitude and continuum polynomial** at each velocity
4. **Separate R and Z band fits** for wavelength-dependent diagnostics
5. **Explicit neighbor-only and component-switching tests**

### Key Facts

| Property | Value |
|----------|-------|
| Neighbor separation | 0.688" |
| Neighbor ΔG | 2.21 mag |
| Expected flux ratio (G) | 0.130 |
| Allowed flux ratio range | [0.02, 0.3] |
| Catalog RV swing | 146 km/s |
| Fitted RV swing | 142 km/s |

### Verdict Table

| Check | Result | Notes |
|-------|--------|-------|
| 1. RV swing robust across arms/templates | **INCONCLUSIVE** | Swing: 142 km/s |
| 2. Neighbor-only fits competitive | **FAIL** | χ² ratios shown below |
| 3. Two-component per-epoch favored | **FAIL** | ΔBIC values shown below |
| 4. Cross-epoch constant-v2 favored | **FAIL** | ΔBIC=-1436.9 |
| 5. Blend switching plausible | **INCONCLUSIVE** | See analysis below |

**OVERALL: COMPROMISED**

Evidence suggests blend contamination may affect RV measurements.

---

## Data Provenance

### DESI Spectra

| Epoch | MJD | File | SHA256 |
|-------|-----|------|--------|
| Epoch1 | 59568.488 | coadd_20268_20211219_p2.fits | 1f3aefba5e346da2... |
| Epoch2 | 59605.380 | coadd_24976_20220125_p7.fits | 87af27aa867295ee... |
| Epoch3 | 59607.380 | coadd_23137_20220127_p0.fits | 0ee485970c0b14e0... |

### Template Grid

**Primary templates (logg=4.5):** Teff = [3400, 3600, 3800, 4000] K
**Neighbor templates (logg=5.0):** Teff = [2800, 3000, 3200, 3400] K

---

## Analysis 1: Single-Star Primary-Only Fits

For each epoch and wavelength region, we fit:
   F(λ) = A × T_primary(v) + poly(λ)

where poly(λ) is a degree-2 polynomial to absorb continuum mismatch.

### Results by Epoch (Combined R+Z)

| Epoch | Catalog RV | Fitted RV | σ_formal | σ_renorm | χ²_red | Best Teff |
|-------|------------|-----------|----------|----------|--------|-----------|
| Epoch1 | -86.4 | -80.9 | 0.39 | 1.0 | 6.4 | 3600K |
| Epoch2 | +59.7 | +61.2 | 1.18 | 2.2 | 3.4 | 3600K |
| Epoch3 | +25.8 | +23.8 | 0.64 | 1.3 | 4.4 | 3600K |

**Fitted RV swing: 142 km/s** (catalog: 146 km/s)

---

## Analysis 2: Neighbor-Only Fits

Same method but using neighbor (cooler) templates.

### Results by Epoch (Combined R+Z)

| Epoch | Neighbor-only RV | χ² | χ²_red | Primary χ² | Ratio |
|-------|------------------|-----|--------|------------|-------|
| Epoch1 | -80.6 | 17039 | 6.4 | 17077 | 1.00 |
| Epoch2 | +61.1 | 8863 | 3.4 | 8806 | 1.01 |
| Epoch3 | +24.2 | 11789 | 4.4 | 11831 | 1.00 |

**Verdict:** FAIL

If ratio > 1.5 for all epochs, neighbor-only template is a poor fit → PASS.

---

## Analysis 3: Two-Component Per-Epoch Fits

For each epoch, fit:
   F(λ) = A × [T_primary(v1) + b × T_neighbor(v2)] + poly(λ)

with b constrained to [0.02, 0.3].

### Results (Combined R+Z)

| Epoch | v1 | v2 | b | ΔBIC | Two-comp preferred? |
|-------|----|----|---|------|---------------------|
| Epoch1 | -80.9 | +23.9 | 0.300 | -1719.1 | YES |
| Epoch2 | +72.8 | -0.0 | 0.300 | -614.8 | YES |
| Epoch3 | +24.0 | -59.0 | 0.300 | -1037.0 | YES |

**Verdict:** FAIL

ΔBIC < -6 means two-component model is strongly preferred (bad for single-star hypothesis).

---

## Analysis 4: Cross-Epoch Constant-v2 Model

Joint fit across all epochs with:
- v2 (neighbor velocity) **shared** across epochs
- b (flux ratio) **shared** across epochs
- v1 (primary velocity) **free per epoch**

This tests whether the RV swing could be explained by a static background neighbor.

### Results

| Parameter | Value |
|-----------|-------|
| v2 (shared) | +0.0 km/s |
| b (shared) | 0.300 |
| v1 (Epoch1) | -81.1 km/s |
| v1 (Epoch2) | +63.3 km/s |
| v1 (Epoch3) | +40.5 km/s |
| Total χ² | 36259.5 |
| χ²_red | 4.58 |

### Model Comparison

| Model | BIC |
|-------|-----|
| Primary-only | 37849.1 |
| Constant-v2 blend | 36412.2 |
| **ΔBIC** | **-1436.9** |

**Verdict:** FAIL

ΔBIC < -6 with b in plausible range means constant-v2 blend model is favored → component switching is a serious explanation.

---

## Analysis 5: Wavelength-Dependent RV Diagnostics

Compare RV measured from R band (6000-7550 Å) vs Z band (7700-8800 Å).

| Epoch | RV_R | σ_R | RV_Z | σ_Z | Δ(Z-R) | Significance |
|-------|------|-----|------|-----|--------|--------------|
| Epoch1 | -79.0 | 1.6 | -87.0 | 8.0 | -8.1 | -1.0σ |
| Epoch2 | +73.5 | 6.4 | +57.9 | 3.9 | -15.6 | -2.1σ |
| Epoch3 | -40.9 | 2.6 | +25.9 | 6.2 | +66.7 | 9.9σ |

Large systematic differences (> 3σ) between arms would indicate wavelength-dependent RV shifts, a signature of blending.

---

## Final Verdict Summary

```
╔════════════════════════════════════════════════════════════════════╗
║              DESI BLEND-AWARE ANALYSIS v3 RESULTS                  ║
╠════════════════════════════════════════════════════════════════════╣
║ 1. RV swing robust across arms/templates:  INCONCLUSIVE             ║
║ 2. Neighbor-only fits competitive:         FAIL                     ║
║ 3. Two-component per-epoch favored:        FAIL                     ║
║ 4. Cross-epoch constant-v2 favored:        FAIL                     ║
║ 5. Blend switching plausible:              INCONCLUSIVE             ║
╠════════════════════════════════════════════════════════════════════╣
║ OVERALL VERDICT:                           COMPROMISED              ║
╚════════════════════════════════════════════════════════════════════╝
```

### Interpretation

The DESI RV measurements are **COMPROMISED** by potential blend effects:

1. Neighbor-only or two-component models show competitive fits
2. Component switching may explain some of the observed RV variability
3. Further investigation with high-resolution spectroscopy is required

**Cannot definitively claim gravitational companion without ruling out blend scenarios.**

---

## Output Files

| File | Description |
|------|-------------|
| `desi_epoch_rv_refit_v3.json` | Per-epoch RV fits with uncertainties |
| `desi_model_comparison_v3.json` | Two-component and arm-split results |
| `desi_constant_v2_fit_v3.json` | Cross-epoch constant-v2 model |
| `figures/chi2_vs_v_by_epoch_v3.png` | χ²(v) curves |
| `figures/rv_by_method_v3.png` | RV comparison with error bars |
| `figures/neighbor_only_vs_primary_v3.png` | Template comparison |
| `figures/constant_v2_model_comparison_v3.png` | Model BIC comparison |
| `figures/arm_split_rv_v3.png` | R vs Z band RVs |

---

## Methodological Notes

### A) RV Uncertainty Renormalization

When χ²_red >> 1 (model mismatch), formal uncertainties underestimate true errors.
We report both:
- **σ_formal**: from Δχ² = 1 criterion
- **σ_renorm**: σ_formal × sqrt(χ²_red)

### B) BIC Computation

BIC = χ²_data + k × ln(n)

where χ²_data is the data-only chi-squared (no prior penalties), k is the number of parameters, and n is the number of data points.

### C) Continuum Marginalization

At each trial velocity, we analytically marginalize over:
- Amplitude A
- Degree-2 polynomial continuum

This ensures χ²(v) curves are smooth and stable.

### D) Flux Ratio Constraints

The flux ratio b is constrained to [0.02, 0.3], broader than the G-band expectation (~0.13) to allow for:
- Color differences between bands
- Seeing variations
- Fiber coupling differences

---

**Report generated:** 2026-01-16 11:48:02
**Analysis by:** Claude Code (v3 methodology)
**Templates:** PHOENIX-ACES models from Göttingen

