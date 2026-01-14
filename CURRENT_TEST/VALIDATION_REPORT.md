# VALIDATION REPORT

## Fixes Applied

### ISSUE A (CRITICAL): Gaia IDs Preserved
- **Fixed**: `triage_rv_candidates.py` now reads `SOURCE_ID` from GAIA HDU (HDU 4), not RVTAB
- **Fixed**: Sentinel value `-9999` treated as missing (not coerced to 0)
- **Fixed**: CSV output uses empty string for missing Gaia IDs

### ISSUE B (CRITICAL): Per-Measurement Quality Cuts Consistent
- **Fixed**: `extract_epochs_for_targets()` now applies the same cuts as `analyze_rv_candidates.py`:
  - `np.isfinite(RV) & np.isfinite(RV_ERR)`
  - `VRAD_ERR < 10 km/s`
  - `|VRAD| < 500 km/s`
- **Impact**: Dark survey Tier A dropped from 90 → 70 (bad epochs filtered)
- **Impact**: d_max max dropped from 765 → 196 (extreme RV epochs removed)

### ISSUE C (QUALITY): Missing Gaia IDs Handled Properly
- **Fixed**: All CSV writers use empty string, not "0"
- **Fixed**: `build_priority_packet.py` dedup logic checks `is None`, not `== 0`

---

## Validation Checklist

### ✓ Gaia SOURCE_ID Presence

| Survey | Total Rows | Rows with Gaia ID | Rows Missing |
|--------|------------|-------------------|--------------|
| Bright | 200 | 199 | 1 |
| Dark | 200 | 197 | 3 |
| Priority A Master | 39 | 39 | 0 |

**Sample Gaia IDs from priorityA_master.csv:**
```
39628001431785529 → 2759088365339967488
39632991214896712 → 1480681355298504960
39633437979575384 → 1584997005586641280
39627714713356667 → 3826086648304166400
```

### ✓ No "0" or "-9999" Values in Gaia Column
- Verified: Empty string used for truly missing Gaia IDs
- No numeric placeholders for missing values

### ✓ d_max Distribution Changed (Due to Filtering)

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Dark max d_max | 765.52 | 196.31 |
| Dark median d_max | ~72.5 | ~42.4 |

**Reason**: Epochs with |VRAD| > 500 km/s are now filtered, removing extreme outliers.

### ✓ Top 10 Priority A Master Candidates

| Rank | TARGETID | Gaia ID | Survey | N | S_robust | d_max |
|------|----------|---------|--------|---|----------|-------|
| 1 | 39628001431785529 | 2759088365339967488 | dark | 4 | 91.1 | 101.2 |
| 2 | 39632991214896712 | 1480681355298504960 | bright | 3 | 81.1 | 302.7 |
| 3 | 39633437979575384 | 1584997005586641280 | bright | 3 | 71.7 | 105.3 |
| 4 | 39627714713356667 | 3826086648304166400 | dark | 4 | 55.4 | 55.3 |
| 5 | 39627830035744797 | 3891388499304470656 | bright | 3 | 52.2 | 108.5 |
| 6 | 39627681263782079 | 6914501041337922944 | bright | 3 | 49.1 | 79.2 |
| 7 | 39633025553665365 | 1375654252266254080 | bright | 3 | 44.8 | 219.7 |
| 8 | 39627720727987709 | 3827093418703158272 | dark | 7 | 43.8 | 110.1 |
| 9 | 39627782317149427 | 3652971286995183488 | dark | 5 | 42.7 | 196.3 |
| 10 | 39627977599746474 | 2751843515722446208 | dark | 4 | 37.8 | 61.1 |

All top 10 have:
- ✓ Valid Gaia SOURCE_ID (non-empty)
- ✓ N_epochs >= 3
- ✓ Reasonable MJD_span (3-355 days)
- ✓ Diagnostics computed using ONLY filtered epochs

### ✓ Priority A Candidate Counts

| Category | Count |
|----------|-------|
| Bright Priority A | 11 |
| Dark Priority A | 28 |
| Master (deduped) | 39 |

Count is in expected range (tens, not thousands, not zero).

### ✓ Red Flags Summary

| Flag | Count | Notes |
|------|-------|-------|
| d_max > 100 | 14/39 | Expected: real high-amplitude RV variability |
| LOO-drops | 20/39 | Signal partially depends on single epoch |
| N=3 (minimum) | 23/39 | Most candidates at minimum epoch threshold |

---

## Files in CURRENT_TEST/

```
CURRENT_TEST/
├── analyze_rv_candidates.py
├── triage_rv_candidates.py
├── build_priority_packet.py
├── audit_dmax.py
├── compute_quantiles.py
├── triage_candidates_bright.csv
├── triage_candidates_dark.csv
├── priorityA_bright.csv
├── priorityA_dark.csv
├── priorityA_master.csv
├── observer_packet_priorityA_master.csv
├── README_packet.md
├── VALIDATION_REPORT.md
└── plots/
    └── (top 10 PDFs)
```

---

## Commands Executed

```bash
python3 triage_rv_candidates.py --survey bright
python3 triage_rv_candidates.py --survey dark
python3 build_priority_packet.py
```

---

## Conclusion

All three critical issues have been fixed:
1. Gaia IDs are properly extracted and preserved
2. Per-measurement quality cuts are consistently applied
3. Missing Gaia IDs use empty string, not numeric placeholder

The pipeline is now production-ready for the specified use case.
