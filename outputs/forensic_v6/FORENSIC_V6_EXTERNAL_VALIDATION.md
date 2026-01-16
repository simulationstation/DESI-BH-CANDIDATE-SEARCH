# FORENSIC V6: EXTERNAL VALIDATION REPORT

**Date:** 2026-01-16 (Updated)
**Target:** Gaia DR3 3802130935635096832
**Coordinates:** RA=164.523494°, Dec=-1.660156°

---

## Executive Summary

This report presents an independent validation of the v5 forensic audit using **real data only**.

### Verdict Table

| Check | Result | Confidence |
|-------|--------|------------|
| **Neighbor Detection** | **PASS** | High — confirmed in Gaia DR3 + EDR3 |
| **LAMOST RV Variability (header)** | Suggestive | Headers show 30.7 km/s difference |
| **LAMOST RV Variability (CCF refit)** | **INCONCLUSIVE** | Refit shows -6.3 ± 4.6 km/s (1.4σ), inconsistent with headers |

### Key Finding

The CCF-based RV refit does **NOT** confirm the header ΔRV. Different wavelength regions yield highly discrepant velocities (-14 to +4.5 km/s), suggesting either:
1. Blend contamination affecting different spectral regions differently
2. Systematic issues with the LAMOST pipeline RVs
3. Template mismatch in cross-correlation

**The LAMOST variability claim requires further investigation before it can be considered confirmed.**

---

## Part 1: LAMOST RV Analysis

### Data Sources

| ObsID | Date | MJD | SHA256 (truncated) |
|-------|------|-----|--------------------|
| 437513049 | 2016-03-10 | 57457 | `58aed377d194...` |
| 870813030 | 2020-12-20 | 59203 | `0888fb557fab...` |

### FITS Header Values

| Parameter | Epoch 1 | Epoch 2 | Difference |
|-----------|---------|---------|------------|
| HELIO_RV | +1.48 km/s | -29.23 km/s | **-30.71 km/s** |
| Z | -0.00016466 | -4.691e-05 | — |
| Z × c | -49.36 km/s | -14.06 km/s | — |
| SNR_i | 35.6 | 27.0 | — |

**V5 Error:** Used Z×c for epoch 1 but HELIO_RV for epoch 2 — inconsistent comparison.

### CCF Refit Results (v2 - Telluric Masked)

Cross-correlation was performed with:
- Telluric masking (O₂, H₂O bands)
- FWHM sanity check (reject >250 km/s)
- Robust continuum normalization

| Region | Wavelength (Å) | ΔRV (km/s) | Error | FWHM | Status |
|--------|----------------|------------|-------|------|--------|
| TiO-6500 | 6350-6850 | **-13.97** | 0.50 | 217 | GOOD |
| TiO-7050 | 6980-7150 | **+4.49** | 0.50 | 149 | GOOD |
| TiO-7450 | 7350-7580 | — | — | — | FAILED |
| TiO-7900 | 7710-8100 | — | — | — | FAILED |
| CaII-triplet | 8360-8600 | — | — | — | FAILED |
| Combined-Red | 6350-7150 | **-9.57** | 0.50 | 190 | GOOD |

**Problem:** The valid regions show a **18 km/s spread** (-14 to +4.5 km/s).

### Combined Result

```
Weighted ΔRV: -6.3 ± 4.6 km/s
Significance: 1.4σ
Header ΔRV: -30.7 km/s
Difference: 24 km/s (refit ≠ header)
```

### Interpretation

The CCF refit is **internally inconsistent** and does **not** match the header values:

1. **Region-to-region scatter (8 km/s)** exceeds typical M-dwarf RV precision
2. **Combined ΔRV (-6.3 km/s)** is 24 km/s away from header ΔRV (-30.7 km/s)
3. **Only 3/6 regions** produced valid fits

Possible explanations:
- **Blend contamination:** The 0.688" neighbor may affect different wavelength regions differently
- **Pipeline issues:** LAMOST HELIO_RV may be unreliable for this target
- **Template mismatch:** Self-template CCF may not be appropriate

### LAMOST Verdict

| Method | ΔRV | Significance | Verdict |
|--------|-----|--------------|---------|
| Header HELIO_RV | -30.7 km/s | N/A (pipeline) | Suggestive |
| CCF Refit | -6.3 ± 4.6 km/s | 1.4σ | **INCONCLUSIVE** |

**The LAMOST variability cannot be independently confirmed from spectra at this time.**

---

## Part 2: Neighbor Confirmation

### Gaia Cross-DR Results

| Catalog | Target Found | Neighbor Found | Separation | ΔG |
|---------|--------------|----------------|------------|-----|
| Gaia DR3 | ✅ | ✅ | **0.688"** | **2.21** |
| Gaia EDR3 | ✅ | ✅ | 0.688" | 2.21 |
| Gaia DR2 | — | — | Query failed | — |

### Neighbor Properties

| Property | Value |
|----------|-------|
| Source ID | 3802130935634233472 |
| Separation | 0.688" (exact match to claim) |
| G magnitude | 19.48 |
| ΔG from target | 2.21 mag (exact match) |
| BP/RP | No data (faint) |
| Parallax/PM | No data |

### Other Catalogs

| Catalog | Result |
|---------|--------|
| SDSS DR16 | Target only (neighbor below limit) |
| 2MASS | Target only |
| Pan-STARRS | Query failed |

### Neighbor Verdict

**PASS** — The neighbor is:
- Confirmed in Gaia DR3 and EDR3
- Persistent across data releases
- Parameters exactly match v5 claim (0.688", ΔG=2.21)

---

## Part 3: Implications

### What This Means for the Candidate

1. **DESI RV variability (146 km/s)** is the primary signal — this was not re-analyzed here
2. **LAMOST adds limited support** — headers suggest variability but CCF refit is inconclusive
3. **Neighbor is confirmed** — blend-aware analysis of DESI is mandatory
4. **The "dark companion" hypothesis** cannot be strengthened by LAMOST until RVs are properly extracted

### Recommended Next Steps

1. **Blend-aware DESI RV extraction:**
   - Check CCFs for multi-peak structure
   - Test template dependence
   - Look for wavelength-dependent RV shifts

2. **LAMOST re-analysis:**
   - Use proper M-dwarf template library
   - Test whether the 0.688" neighbor could cause the wavelength-dependent ΔRV

3. **Do not claim "confirmed variability" from LAMOST** until the CCF discrepancy is resolved

---

## Final Summary

```
╔════════════════════════════════════════════════════════════════════╗
║              FORENSIC V6 EXTERNAL VALIDATION                       ║
╠════════════════════════════════════════════════════════════════════╣
║ Neighbor Detection:          PASS (confirmed in DR3 + EDR3)        ║
║ LAMOST Header ΔRV:           -30.7 km/s (suggestive)               ║
║ LAMOST CCF Refit:            INCONCLUSIVE (-6.3 ± 4.6 km/s, 1.4σ) ║
╠════════════════════════════════════════════════════════════════════╣
║ CCF regions highly discrepant: -14 to +4.5 km/s                    ║
║ Refit does NOT match headers                                       ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## Reproducibility

```bash
# Run LAMOST RV analysis (v2 with telluric masking)
python scripts/forensic_v6_lamost_rv_v2.py

# Run neighbor confirmation
python scripts/forensic_v6_neighbor_check.py
```

### Output Files

| File | Description |
|------|-------------|
| `lamost_rv_refit_v2.json` | CCF refit results with telluric masking |
| `neighbor_catalog_crosscheck.json` | Multi-catalog neighbor results |
| `figures/lamost_ccf_diagnostics_v2.png` | CCF by wavelength region |
| `figures/lamost_epoch_overlay_v2.png` | Spectrum comparison |

---

**Report generated:** 2026-01-16
**Auditor:** Claude Code
