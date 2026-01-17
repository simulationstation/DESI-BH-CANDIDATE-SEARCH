# E8 Anomalies Candidates Summary

Generated: 2026-01-15

## Overview

E8_anomalies searches for transient phenomena, cataclysmic variables, and other anomalous RV signatures that don't fit standard binary models.

**Total candidates**: 50
**Plausible candidates (K < 150 km/s, 0.5 < M2 < 10 M☉)**: 11

## Top 15 Candidates

| # | TargetID | M2_min | Period | K | ΔRV | N_ep | Score | Status |
|---|----------|--------|--------|---|-----|------|-------|--------|
| 1 | 39627624485487722 | 8.94 | 32.5d | 128 | 59 | 7 | 0.84 | ❌ DISCARDED (no coords) |
| 2 | 39627691564992866 | 13.55 | 106.6d | 102 | 40 | 7 | 0.84 | ⚠️ Review |
| 3 | 39627775883085095 | 12.89 | 1.3d | 430 | 82 | 5 | 0.84 | ❌ HIGH K (SB2?) |
| 4 | 39627872045893843 | 19.49 | 14.5d | 224 | 423 | 5 | 0.84 | ❌ HIGH K (SB2?) |
| 5 | 39632935875250494 | 48.15 | 8.0d | 385 | 488 | 5 | 0.84 | ❌ EXTREME MASS |
| 6 | 39627800600117994 | 32.09 | 214.7d | 111 | 26 | 5 | 0.80 | ⚠️ Review |
| 7 | 39627860838712972 | 5.42 | 25.8d | 112 | 49 | 5 | 0.80 | ✓ Plausible |
| 8 | 39627634937694383 | 3.22 | 9.0d | 124 | 120 | 5 | 0.80 | ✓ Plausible |
| 9 | 39632980964018604 | 13.86 | 105.0d | 103 | 36 | 5 | 0.80 | ⚠️ Review |
| 10 | 39627884519756202 | 67.70 | 7.5d | 438 | 90 | 4 | 0.80 | ❌ EXTREME MASS |
| 11 | 39628127814550862 | 31.80 | 27.5d | 219 | 21 | 4 | 0.80 | ❌ HIGH K |
| 12 | 39633232601286378 | 9.70 | 7.0d | 224 | 31 | 3 | 0.80 | ❌ LOW EPOCHS |
| 13 | 39627885153091837 | 40.51 | 105.0d | 154 | 40 | 4 | 0.80 | ❌ EXTREME MASS |
| 14 | 39633358875001295 | 34.01 | 52.9d | 181 | 40 | 4 | 0.80 | ❌ HIGH K |
| 15 | 39628502240072918 | 9.18 | 16.1d | 164 | 22 | 4 | 0.80 | ❌ HIGH K |

## Recommended for Follow-up

Based on reasonable parameters (K < 150 km/s, plausible masses):

1. **39627860838712972** - M2=5.4 M☉, K=112 km/s, P=25.8d
2. **39627634937694383** - M2=3.2 M☉, K=124 km/s, P=9.0d
3. **39627691564992866** - M2=13.6 M☉, K=102 km/s, P=106.6d (high mass needs verification)

## Files in this folder

1. **E8_anomalies_results.json** - Full E8 results (50 candidates)
2. **e8_candidates.csv** - All candidates with flags
3. **e8_plausible_candidates.json** - 11 filtered candidates
4. **E8_SUMMARY.md** - This file
5. **39627624485487722_E8_anomalies.json** - Dossier (discarded)
6. **39627634937694383_E8_anomalies.json** - Dossier (plausible)

## Warning Signs (Gemini's Criteria)

- **K > 200 km/s**: Almost certainly SB2 (double-lined binary)
- **M2_min > 20 M☉**: Physically implausible for stellar-mass BH
- **N_epochs < 5**: Insufficient sampling
- **RA/Dec = 0.0**: Metadata error, cannot verify with Gaia
