# Priority A RV Variable Candidates - Observer Packet

## Summary

This packet contains 39 Priority A candidates selected from DESI DR1
Milky Way Survey radial velocity measurements. These are sources showing
statistically significant RV variability.

**IMPORTANT**: This list includes RV-variable objects of **mixed physical origin**.
Known variable classes (eclipsing binaries, RR Lyrae pulsators, QSOs) have been
identified via SIMBAD cross-matching and labeled with a `variable_class` column.
A **follow-up-only subset** (`priorityA_followup_only.csv`) is provided containing
only `candidate_companion` sources to avoid wasting telescope time on known variables.

---

## What This List Is NOT

1. **Not a compact-object catalog**: No claim is made about companion masses or types.
   These are RV-variable sources only.

2. **Not a population inference**: This is not a complete or unbiased sample.
   Selection effects from DESI targeting, observing cadence, and RV precision
   are not characterized.

3. **Not complete**: Many genuine binaries will be missed due to:
   - Insufficient epoch coverage
   - RV precision limits
   - Face-on orbital inclinations
   - Long orbital periods

4. **Not discovery claims**: Some sources may already be known variables in
   catalogs not checked (e.g., LAMOST, APOGEE, Gaia DR4).

---

## Recommended Use

1. **Follow-up spectroscopy**: Use `priorityA_followup_only.csv` for new observations.
   This subset excludes known pulsators (RR Lyrae), AGN/QSOs, and previously
   catalogued binaries.

2. **Retrieve coordinates and magnitudes**: Use the Gaia SOURCE_ID column to
   query Gaia DR3 for coordinates, proper motions, parallaxes, and photometry.

3. **Check additional catalogs**: Before observing, query LAMOST, APOGEE, and
   other RV surveys for existing multi-epoch data.

4. **Confirm variability first**: The first follow-up observation should verify
   that RV has changed from DESI measurements before investing in orbital monitoring.

---

## Candidate Counts

| Category | Count |
|----------|-------|
| Total Priority A | 39 |
| Known binaries (EB*, BY*) | 3 |
| Pulsating stars (RR Lyrae) | 11 |
| AGN/QSO | 4 |
| **Candidate companions (follow-up worthy)** | **21** |

| Survey | Priority A Total |
|--------|-----------------|
| Bright | 11 |
| Dark | 28 |
| Master (deduped) | 39 |

---

## Selection Criteria

Candidates passed these data-driven gates:

1. **Tier A only**: N_epochs >= 3 (at least 3 RV measurements)
2. **S_robust >= 10**: Conservative significance score
   - S_robust = min(S, S_min_LOO) where S_min_LOO is the minimum
     significance when any single epoch is dropped
   - This guards against single-epoch flukes
3. **MJD_span >= 0.5 days**: Observations span at least half a day
4. **Per-measurement quality cuts**: VRAD_ERR < 10 km/s, |VRAD| < 500 km/s

---

## Variable Class Definitions

| Class | SIMBAD Types | Meaning |
|-------|-------------|---------|
| `candidate_companion` | None or generic (*) | Primary follow-up targets |
| `known_binary` | EB*, BY*, SB* | Already known binary systems |
| `pulsating_star` | RR* | Intrinsic RV variability, not orbital |
| `AGN_QSO` | QSO, AGN | Extragalactic, not stellar |

---

## Metric Definitions

| Metric | Definition |
|--------|------------|
| S | ΔRV_max / sqrt(Σ σ_i²) - original significance |
| S_robust | min(S, S_min_LOO) - conservative significance |
| S_min_LOO | Minimum S when any single epoch is dropped |
| d_max | max_i |RV_i - median(RV)| / σ_i - outlier leverage |
| ΔRV | max(RV) - min(RV) in km/s |

---

## Flags in Notes Column

- **LOO-stable**: S_min_LOO / S > 0.8 (signal robust to dropping any epoch)
- **LOO-moderate**: 0.5 < S_min_LOO / S <= 0.8
- **LOO-drops**: S_min_LOO / S <= 0.5 (signal depends on one epoch)
- **high-leverage(d=X)**: d_max > 100 (one epoch far from median)
- **N=X**: Number of epochs (N=3 is minimum for Tier A)

---

## Files in This Packet

| File | Description |
|------|-------------|
| `priorityA_master_annotated_classes.csv` | Full list with variable_class column |
| `priorityA_followup_only.csv` | **Follow-up subset** (candidate_companion only) |
| `priorityA_bright.csv` | Bright survey candidates (original) |
| `priorityA_dark.csv` | Dark survey candidates (original) |
| `priorityA_master.csv` | Merged master list (original) |
| `NSS_SIMBAD_CROSSCHECK_REPORT_REVISED.md` | Cross-match methodology and results |
| `plots/*.pdf` | RV vs MJD plots for top 20 master candidates |

---

## Data Provenance

- **Source**: DESI DR1 Milky Way Survey "iron" VAC
- **Files**: rvpix_exp-main-bright.fits, rvpix_exp-main-dark.fits
- **URL**: https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0/
- **Processing**: triage_rv_candidates.py → build_priority_packet.py → add_variable_class.py
- **Cross-match**: Gaia DR3 NSS (TAP), SIMBAD (TAP)

---

## Caveats

1. **No orbit fitting performed**: ΔRV is max-min, not orbital amplitude
2. **No stellar parameter cuts**: No filtering by Teff, logg, [Fe/H], etc.
3. **Missing Gaia IDs**: Some targets have no Gaia SOURCE_ID (empty in CSV)
4. **VRAD_ERR is quoted pipeline uncertainty**: May not reflect true errors
5. **SIMBAD may be incomplete**: Absence of classification does not imply novelty
6. **Gaia NSS not checked**: 0/39 had NSS entries (expected due to incompleteness)

---

Generated by build_priority_packet.py + add_variable_class.py
Date: 2026-01-13
