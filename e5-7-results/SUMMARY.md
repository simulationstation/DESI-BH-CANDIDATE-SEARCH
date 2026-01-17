# E5-E7 Candidates Analysis Summary

Generated: 2026-01-15

## Overview

| Experiment | Description | Candidates |
|------------|-------------|------------|
| E5_hierarchical | Hierarchical triple systems | 50 |
| E6_accretion | Accreting compact objects | 50 |
| E7_halo_cluster | Halo/cluster BH candidates | 50 |

**Total unique candidates across E5-E7: 72**
**Candidates with M2_min > 3 M☉: 47**

## Top 15 by M2_min

| TargetID | M2_min (M☉) | Period (d) | K (km/s) | ΔRV (km/s) | N_epochs | Score | Experiments |
|----------|-------------|------------|----------|------------|----------|-------|-------------|
| 39627612133547418 | 100.00 | 2.10 | 1216.2 | 498.7 | 5 | 0.68 | E6,E7 |
| 39627638236758216 | 94.69 | 4.12 | 725.4 | 460.8 | 5 | 0.68 | E6,E7 |
| 39627617463088366 | 67.33 | 5.52 | 584.9 | 514.1 | 6 | 0.71 | E6,E7 |
| 39627603643633849 | 51.59 | 4.17 | 579.9 | 343.5 | 5 | 0.73 | E5,E6,E7 |
| 39632935875250494 | 48.15 | 8.00 | 385.2 | 487.7 | 5 | 0.86 | E5,E6,E7 |
| 39627580969583367 | 42.22 | 1.32 | 868.1 | 355.6 | 5 | 0.68 | E6,E7 |
| 39627621676973265 | 28.69 | 9.36 | 307.9 | 425.9 | 7 | 0.73 | E6,E7 |
| 39627629802538179 | 26.54 | 3.04 | 577.1 | 344.0 | 6 | 0.73 | E5,E6,E7 |
| 39627872045893843 | 19.49 | 14.49 | 223.7 | 422.7 | 5 | 0.86 | E5,E6,E7 |
| 39627805893329991 | 15.33 | 8.25 | 252.0 | 412.2 | 5 | 0.80 | E5,E6,E7 |
| 39627630360899782 | 14.93 | 11.97 | 218.8 | 332.9 | 8 | 0.73 | E6,E7 |
| 39627609933098277 | 14.82 | 6.48 | 297.0 | 300.3 | 6 | 0.73 | E5,E6,E7 |
| 39627621676973144 | 13.53 | 5.11 | 338.0 | 280.8 | 5 | 0.72 | E6,E7 |
| 39627609933098340 | 12.70 | 5.05 | 334.1 | 317.0 | 5 | 0.72 | E5,E6,E7 |
| 39627609967764672 | 11.93 | 1.63 | 516.7 | 324.7 | 5 | 0.73 | E5,E6,E7 |

## Files in this folder

1. **E5_hierarchical_results.json** - Full E5 results (50 candidates)
2. **E6_accretion_results.json** - Full E6 results (50 candidates)
3. **E7_halo_cluster_results.json** - Full E7 results (50 candidates)
4. **all_unique_candidates.csv** - All 72 unique candidates, sorted by M2_min
5. **top_bh_candidates.json** - 47 candidates with M2_min > 3 M☉
6. **SUMMARY.md** - This file

## Key Fields

- **targetid**: DESI target identifier
- **m2_min**: Minimum companion mass assuming M1=0.7 M☉ and i=90°
- **best_period**: Best-fit orbital period in days
- **best_K**: RV semi-amplitude in km/s
- **delta_rv**: Peak-to-peak RV variation in km/s
- **n_epochs**: Number of RV measurements
- **total_score**: Combined quality score (0-1)
- **mass_function**: f(M) in solar masses
- **S_robust**: Robust significance (leave-one-out)
- **is_pathological**: True if candidate shows pathological behavior

## Cautions

1. **Very high M2_min values (>20 M☉)** may indicate:
   - Hierarchical systems with blended spectra
   - Measurement errors
   - Actual intermediate-mass BHs (rare)

2. **Short periods (<3 days) with high K** need verification for:
   - Contact binaries
   - Spectroscopic blends

3. **Candidates appearing in all 3 experiments** are most robust

## Recommended Follow-up

1. Cross-match with Gaia DR3 for RUWE, parallax, proper motion
2. Check ZTF/TESS photometry for eclipses/variability
3. Verify with additional RV epochs
4. Check for emission lines (accretion signatures)
