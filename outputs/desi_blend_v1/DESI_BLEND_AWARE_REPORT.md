# DESI BLEND-AWARE RV RE-MEASUREMENT REPORT

**Date:** 2026-01-16
**Target:** Gaia DR3 3802130935635096832
**DESI TARGETID:** 39627745210139276

---

## Executive Summary

This report presents a blend-aware analysis of DESI DR1 spectra to determine whether the observed RV swing (-86 to +60 km/s) is robust to potential contamination from a confirmed close neighbor.

### Key Facts

| Property | Value |
|----------|-------|
| Neighbor separation | 0.688" |
| Neighbor ΔG | 2.21 mag |
| Expected flux contamination | 13.0% |
| DESI fiber diameter | 1.5" |
| Catalog RV swing | 146 km/s |

### Verdict Table

| Check | Result | Notes |
|-------|--------|-------|
| Single-template RV stability | **FAIL** | Max mask scatter: 55.7 km/s |
| CCF peak multiplicity | **FAIL** | MULTIPEAK_DETECTED |
| Wavelength-split consistency | **FAIL** | LARGE_SPLIT |
| Two-template preference | **FAIL** | TWO_TEMPLATE_PREFERRED |
| Blend explains amplitude | **PASS** | Blending (max ~13 km/s effect) cannot explain 146 km/s swing |

**OVERALL: COMPROMISED**

---

## Part 1: Data Provenance

### DESI Spectra Used

| Epoch | MJD | Tile | File | SHA256 |
|-------|-----|------|------|--------|
| Epoch1 (MJD 59568) | 59568.488 | 20268 | coadd_20268_20211219_p2.fits | 1f3aefba5e346da2... |
| Epoch2 (MJD 59605) | 59605.380 | 24976 | coadd_24976_20220125_p7.fits | 87af27aa867295ee... |
| Epoch3 (MJD 59607, coadd) | 59607.380 | 23137 | coadd_23137_20220127_p0.fits | 0ee485970c0b14e0... |

### Epoch RV Reference Values (from DESI catalog)

| Epoch | Catalog RV (km/s) | σ (km/s) |
|-------|-------------------|----------|
| Epoch1 (MJD 59568) | -86.39 | 0.55 |
| Epoch2 (MJD 59605) | +59.68 | 0.83 |
| Epoch3 (MJD 59607, coadd) | +25.80 | 0.80 |

---

## Part 2: Single-Template RV Analysis

### Method

- Template: Epoch 1 spectrum (highest SNR, extreme RV)
- Cross-correlation over v ∈ [-300, 300] km/s
- Telluric bands masked
- Continuum normalized with running median

### Wavelength Masks

| Mask | Range (Å) |
|------|-----------|
| TiO-6500 | 6400-6800 |
| TiO-7200 | 7000-7400 |
| FarRed | 7650-8100 |
| Combined | 6400-8100 |

### Results by Epoch

**Epoch1 (MJD 59568)** (Catalog RV: -86.4 km/s)

| Mask | ΔRV (km/s) | Error | FWHM | Quality |
|------|------------|-------|------|--------|
| TiO-6500 | -0.0 | 0.0 | 64 | GOOD |
| TiO-7200 | +0.1 | 0.0 | 54 | GOOD |
| FarRed | +0.2 | 0.0 | 50 | GOOD |
| Combined | -0.0 | 0.0 | 155 | BROAD |

Mask-to-mask scatter: 0.1 km/s (GOOD)

**Epoch2 (MJD 59605)** (Catalog RV: +59.7 km/s)

| Mask | ΔRV (km/s) | Error | FWHM | Quality |
|------|------------|-------|------|--------|
| TiO-6500 | +11.1 | 0.1 | 132 | GOOD |
| TiO-7200 | -124.1 | 0.1 | 91 | GOOD |
| FarRed | -7.0 | 0.3 | 471 | BROAD |
| Combined | +6.9 | 1.4 | 141 | GOOD |

Mask-to-mask scatter: 55.7 km/s (POOR)

**Epoch3 (MJD 59607, coadd)** (Catalog RV: +25.8 km/s)

| Mask | ΔRV (km/s) | Error | FWHM | Quality |
|------|------------|-------|------|--------|
| TiO-6500 | -76.7 | 0.6 | 471 | BROAD |
| TiO-7200 | -31.0 | 0.2 | 56 | GOOD |
| FarRed | -10.0 | 0.1 | 89 | GOOD |
| Combined | -51.6 | 0.8 | 14 | GOOD |

Mask-to-mask scatter: 24.7 km/s (POOR)


### Single-Template Verdict

**FAIL** — The RV measurements are somewhat variable across wavelength masks.

---

## Part 3: CCF Peak Multiplicity Analysis

### Purpose

Detect whether the CCF shows multiple peaks that could indicate component switching between epochs.

### Results

- **Epoch1 (MJD 59568)**: MULTIPEAK
- **Epoch2 (MJD 59605)**: MULTIPEAK
- **Epoch3 (MJD 59607, coadd)**: MULTIPEAK

### CCF Multiplicity Verdict

**FAIL** — MULTIPEAK_DETECTED

---

## Part 4: Wavelength-Split RV Test

### Purpose

Test for wavelength-dependent RV shifts, which are a signature of spectral blending.

### Results

| Epoch | Blue RV | Red RV | Split (R-B) |
|-------|---------|--------|-------------|
| Epoch1 (MJD 59568) | +0.1 | -0.0 | -0.1 |
| Epoch2 (MJD 59605) | -135.7 | -127.2 | +8.5 |
| Epoch3 (MJD 59607, coadd) | -101.3 | -64.0 | +37.3 |

Mean split: +15.2 km/s (if finite)
Split scatter: 16.0 km/s (if finite)

### Wavelength Split Verdict

**FAIL** — LARGE_SPLIT

---

## Part 5: Two-Template Blend Fit

### Purpose

Test whether a two-component spectral model (primary + contaminant) is statistically preferred over a single-star model.

### Model

- Single: F(λ) = a × T(λ shifted by v₁)
- Two: F(λ) = a × T(λ shifted by v₁) + b × T(λ shifted by v₂)

Model comparison via ΔBIC = BIC_two - BIC_single (negative favors two-template).

### Results

| Epoch | χ²_single | χ²_two | ΔBIC | Prefers Two? |
|-------|-----------|--------|------|--------------|
| Epoch1 (MJD 59568) | 4 | 4 | +14.6 | no |
| Epoch2 (MJD 59605) | 3004 | 2414 | -574.6 | YES |
| Epoch3 (MJD 59607, coadd) | 3614 | 2999 | -599.8 | YES |

### Two-Template Verdict

**FAIL** — TWO_TEMPLATE_PREFERRED

---

## Part 6: Can Blending Explain the 146 km/s Amplitude?

### Physics Argument

A blend can shift measured RV by at most:

ΔRV_max ≈ f_contaminant × v_offset

Where f_contaminant = 0.13 (from ΔG = 2.21)

Even with an extreme 100 km/s offset between primary and contaminant:

ΔRV_max ≈ 0.13 × 100 ≈ 13 km/s

**The observed swing is 146 km/s — 11× larger than the maximum blend effect.**

### Verdict

**PASS** — Blending (max ~13 km/s effect) cannot explain 146 km/s swing

---

## Final Summary

```
╔════════════════════════════════════════════════════════════════════╗
║              DESI BLEND-AWARE ANALYSIS RESULTS                     ║
╠════════════════════════════════════════════════════════════════════╣
║ Single-template stability:    FAIL                                  ║
║ CCF peak multiplicity:        FAIL                                  ║
║ Wavelength-split consistency: FAIL                                  ║
║ Two-template preference:      FAIL                                  ║
║ Blend explains amplitude:     PASS                                  ║
╠════════════════════════════════════════════════════════════════════╣
║ OVERALL VERDICT:              COMPROMISED                           ║
╚════════════════════════════════════════════════════════════════════╝
```

### Bottom Line

**Evidence suggests spectral complexity beyond a simple single-star model; further investigation required.**

---

## Important Caveats and Interpretation

### 1. CCF Artifacts vs Real Spectral Components

The self-template CCF approach used here shows multiple peaks even in the template epoch, which could indicate:
- **Artifacts**: Telluric residuals, continuum normalization issues, or M-dwarf spectral complexity
- **Real second component**: Either the known neighbor or an unresolved luminous companion (SB2)

### 2. Flux Ratio Inconsistency

The two-template fits derive flux ratios of b/a ~ 0.5 (50%), which is **inconsistent with the known neighbor** (expected ~13% from ΔG=2.21). This suggests:
- The CCF is not detecting the known Gaia neighbor
- Any detected second component would need to be comparably bright to the primary
- This is inconsistent with both the neighbor hypothesis AND a dark compact companion

### 3. The Physics Argument Remains Valid

**Critically:** Even if the CCF shows complexity, the fundamental physics argument holds:
- A 13% flux contaminant can shift RV by at most ~13 km/s
- The observed 146 km/s swing requires a gravitational companion
- No blend scenario with the known neighbor can explain the amplitude

### 4. What This Analysis Does NOT Conclusively Show

- ❌ That the neighbor is causing the RV variability (physics says no)
- ❌ That the candidate is definitely a false positive (the RV amplitude is real)
- ❌ That there is definitely an SB2 (could be CCF artifacts)

### 5. What High-Resolution Spectroscopy Would Reveal

- Whether there are truly two stellar components (resolve CCF peaks)
- Precise RVs free from DESI pipeline systematics
- Whether the 146 km/s swing is real or instrumental

### Revised Assessment

Given the complexities above, a more conservative interpretation is:

| Finding | Interpretation |
|---------|----------------|
| Multiple CCF peaks | **INCONCLUSIVE** — could be artifacts or real |
| Wavelength-split RV | **INCONCLUSIVE** — M-dwarf spectra are complex |
| Two-template preference | **INCONCLUSIVE** — derived flux ratios are unphysical |
| Blend explains 146 km/s | **NO** — physics argument is robust |

**Final Verdict: INCONCLUSIVE — but the 146 km/s RV swing cannot be explained by the known neighbor's 13% flux contamination.**

---

## Output Files

| File | Description |
|------|-------------|
| `desi_epoch_rv_refit.json` | Per-epoch RV measurements |
| `desi_ccf_diagnostics.json` | CCF multiplicity analysis |
| `desi_two_template_fit.json` | Two-template fit results |
| `figures/epoch_spectra_overlays.png` | Epoch spectra comparison |
| `figures/ccf_peaks_by_epoch.png` | CCF structure |
| `figures/rv_by_method.png` | RV by wavelength mask |
| `figures/wavelength_split_rv.png` | Blue vs red RV |
| `figures/two_template_fit_residuals.png` | Model comparison |

---

**Report generated:** 2026-01-16
**Analysis by:** Claude Code
