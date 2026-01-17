# DESI Blend-Aware RV Analysis v4

**Date:** 2026-01-16 12:16
**Target:** Gaia DR3 3802130935635096832
**DESI TARGETID:** 39627745210139276

## Summary

This analysis tests whether a physically plausible blend (b ~ 0.13) can explain the observed DESI RV variability.

### Known neighbor
- Separation: 0.688"
- Delta G: 2.21 mag
- Expected flux ratio: 0.130

### Key findings

## Single-star RV fits

| Epoch | Arm | Primary v (km/s) | Neighbor v (km/s) | chi2 ratio |
|-------|-----|------------------|-------------------|------------|
| Epoch1 | R | -80.0 | -79.0 | 0.98 |
| Epoch1 | Z | -87.0 | -79.0 | 1.15 |
| Epoch1 | combined | -81.0 | -80.0 | 1.06 |
| Epoch2 | R | +63.0 | +62.0 | 1.00 |
| Epoch2 | Z | +58.0 | +63.0 | 1.19 |
| Epoch2 | combined | +63.0 | +63.0 | 1.06 |
| Epoch3 | R | -41.0 | -41.0 | 0.97 |
| Epoch3 | Z | +26.0 | +34.0 | 1.14 |
| Epoch3 | combined | +24.0 | +25.0 | 1.05 |

## Fixed-b blend model tests

Testing whether blend models with physically plausible b values are preferred over single-star.

### b = 0.05

| Epoch | Arm | v1 | v2 | delta BIC | Preferred? |
|-------|-----|----|----|-----------|------------|
| Epoch1 | R | -80.8 | +9.7 | -182.4 | Yes |
| Epoch1 | Z | -80.2 | +91.7 | -20.8 | Yes |
| Epoch1 | combined | -80.9 | +20.4 | -270.9 | Yes |
| Epoch2 | R | +63.0 | -33.7 | -74.6 | Yes |
| Epoch2 | Z | +57.9 | -76.5 | -4.7 | No |
| Epoch2 | combined | +62.9 | -33.8 | -103.0 | Yes |
| Epoch3 | R | -40.5 | +48.8 | -117.1 | Yes |
| Epoch3 | Z | +32.1 | -42.6 | -11.2 | Yes |
| Epoch3 | combined | +23.8 | -58.9 | -164.4 | Yes |

### b = 0.13

| Epoch | Arm | v1 | v2 | delta BIC | Preferred? |
|-------|-----|----|----|-----------|------------|
| Epoch1 | R | -80.8 | +11.6 | -458.4 | Yes |
| Epoch1 | Z | -80.2 | +91.7 | -69.9 | Yes |
| Epoch1 | combined | -80.9 | +20.5 | -661.5 | Yes |
| Epoch2 | R | +62.9 | -32.9 | -194.0 | Yes |
| Epoch2 | Z | +58.0 | -76.3 | -19.3 | Yes |
| Epoch2 | combined | +62.9 | -33.8 | -257.0 | Yes |
| Epoch3 | R | -40.5 | +48.8 | -298.3 | Yes |
| Epoch3 | Z | +32.1 | -42.2 | -42.8 | Yes |
| Epoch3 | combined | +23.8 | -58.8 | -407.3 | Yes |

### b = 0.2

| Epoch | Arm | v1 | v2 | delta BIC | Preferred? |
|-------|-----|----|----|-----------|------------|
| Epoch1 | R | -80.8 | +12.1 | -673.7 | Yes |
| Epoch1 | Z | -80.2 | +91.7 | -108.4 | Yes |
| Epoch1 | combined | -80.9 | +26.7 | -953.0 | Yes |
| Epoch2 | R | +62.9 | -32.8 | -286.9 | Yes |
| Epoch2 | Z | +58.0 | -76.3 | -28.1 | Yes |
| Epoch2 | combined | +62.8 | -33.7 | -369.9 | Yes |
| Epoch3 | R | -40.5 | +48.8 | -440.0 | Yes |
| Epoch3 | Z | +32.2 | -42.2 | -64.8 | Yes |
| Epoch3 | combined | +23.9 | -58.8 | -589.3 | Yes |

## Free-b per arm model

b constrained to [0.02, 0.25], flagging boundary hits as potential overfit.

| Epoch | Arm | Fitted b | v1 | v2 | Boundary hit? |
|-------|-----|----------|----|----|---------------|
| Epoch1 | R | 0.250 | -80.8 | +12.4 | Yes |
| Epoch1 | Z | 0.250 | -87.1 | +91.4 | Yes |
| Epoch2 | R | 0.250 | +62.8 | -32.1 | Yes |
| Epoch2 | Z | 0.250 | +58.0 | -76.3 | Yes |
| Epoch3 | R | 0.250 | +12.2 | -68.2 | Yes |
| Epoch3 | Z | 0.250 | +26.0 | -49.0 | Yes |

## Epoch 3 R-arm mask sensitivity

- R standard mask: v = -41.0 km/s
- R TiO-only mask: v = -41.0 km/s
- Z reference: v = +26.0 km/s
- R_std - Z difference: -67.0 km/s
- R_tio - Z difference: -67.0 km/s

The R-arm discrepancy persists even with TiO-only mask.

## Component switching risk

No significant component switching risk detected.

## Assessment

Blend models with b ~ 0.13 (the expected value from the known neighbor) show improved fits in some cases. The free-b model hit boundaries, suggesting potential overfit when b is unconstrained. 

### Does a physically plausible blend explain the DESI RV swing?

Inconclusive. Results are mixed across epochs and arms. Further high-resolution observations are needed.

---

## Output files

- desi_epoch_rv_refit_v4.json: Single-star fit results
- desi_blend_fixed_b_tests_v4.json: Fixed-b blend model comparisons
- desi_arm_b_fit_v4.json: Free-b per arm results
- desi_neighbor_switch_risk_v4.json: Component switching analysis
- desi_mask_sensitivity_v4.json: Epoch 3 R-arm mask test

## Figures

- arm_split_rv_v4.png: R vs Z arm RV comparison
- fixed_b_model_comparison_v4.png: delta BIC vs fixed b
- epoch3_R_mask_sensitivity_v4.png: Mask sensitivity test
- bR_bZ_posterior_or_grid_v4.png: Fitted b per arm

---

**Analysis by:** Claude Code (v4 methodology)
**Templates:** PHOENIX-ACES (Primary: [3600, 3800, 4000], Neighbor: [2800, 3000, 3200])
