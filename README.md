# DESI DR1 MWS RV-Variable Candidate Search

Pipeline for identifying stellar radial velocity (RV) variable candidates from DESI Data Release 1 Milky Way Survey data. These are sources showing statistically significant RV variability that may warrant spectroscopic follow-up.

---

## Final Results: 21 Validated Follow-Up Candidates

After rigorous filtering, quality control, and cross-matching, we identify **21 stellar RV-variable candidates** suitable for follow-up spectroscopy.

### Selection Summary

| Stage | Count | Description |
|-------|-------|-------------|
| DESI DR1 MWS sources | ~5.4M | All RV measurements (bright + dark surveys) |
| Multi-epoch (N ≥ 2) | ~180k | Sources with multiple observations |
| Tier A (N ≥ 3) | ~100 | Minimum 3 epochs for robust statistics |
| Priority A (S_robust ≥ 10) | 39 | High-significance RV variability |
| After SIMBAD filtering | 21 | Excluding known QSOs, RR Lyrae, binaries |
| **Deep-dive validated** | **21** | All pass quality checks |

### What This List Is

- **RV-variable stellar sources** with statistically significant velocity changes
- **Candidates for follow-up** — not confirmed compact object companions
- **Validated against Gaia DR3 and LAMOST** for additional context

### What This List Is NOT

- Not a compact-object catalog (no mass inference)
- Not a complete sample (selection effects not characterized)
- Not discovery claims (some may be known variables in unchecked catalogs)

---

## The 21 Validated Candidates

| Rank | TARGETID | Gaia SOURCE_ID | N | S_robust | ΔRV (km/s) | MJD span | Gaia Flag | LAMOST |
|------|----------|----------------|---|----------|------------|----------|-----------|--------|
| 3 | 39633437979575384 | 1584997005586641280 | 3 | 71.7 | 98.6 | 355.0 | OK | - |
| 5 | 39627830035744797 | 3891388499304470656 | 3 | 52.2 | 76.0 | 94.8 | OK/VAR | - |
| 6 | 39627681263782079 | 6914501041337922944 | 3 | 49.1 | 96.9 | 3.0 | OK | - |
| 8 | 39627720727987709 | 3827093418703158272 | 7 | 43.8 | 93.4 | 102.7 | OK | - |
| 11 | 39627721168388315 | 3787512447507460352 | 3 | 35.6 | 330.5 | 122.7 | OK | - |
| 12 | 39628408488985455 | 1880759723584022528 | 3 | 33.4 | 119.6 | 13.0 | OK | - |
| 13 | 39627690751300586 | 3780868403682988032 | 8 | 30.5 | 66.6 | 109.7 | OK | - |
| 15 | 39627634937694383 | 3244963210786057984 | 4 | 30.2 | 114.3 | 78.8 | OK | - |
| 16 | 39627751707116493 | 3683541768991487104 | 4 | 29.1 | 80.5 | 46.9 | OK | - |
| 18 | 39627793041985967 | 3843110008879900032 | 6 | 28.8 | 42.8 | 87.8 | OK | - |
| 22 | 39627829624701236 | 3856238482657768576 | 3 | 20.6 | 100.2 | 14.9 | OK | - |
| 23 | 39627745210139276 | 3802130935635096832 | 4 | 19.8 | 146.1 | 38.9 | SUSPECT | YES |
| 24 | 39628051222364390 | 602361299180812800 | 3 | 19.6 | 32.4 | 139.6 | OK | - |
| 26 | 39627836088124572 | 3891796860498960768 | 3 | 17.2 | 64.0 | 23.9 | OK | YES |
| 27 | 39627624326103210 | 3823628002865715328 | 5 | 16.7 | 70.9 | 79.8 | OK | - |
| 29 | 39627817100513868 | 3078449763965033472 | 3 | 16.1 | 381.8 | 9.0 | OK | - |
| 30 | 39633325534479674 | 1563820480355540352 | 3 | 15.2 | 66.9 | 29.0 | OK | - |
| 31 | 39627981798249854 | 4442920432493231872 | 5 | 15.0 | 111.0 | 12.0 | OK | - |
| 32 | 39627788696688328 | 4410518344516247424 | 3 | 13.2 | 51.9 | 16.9 | OK | - |
| 33 | 39627892669286666 | 1732847539605117824 | 3 | 12.6 | 111.8 | 4.0 | OK | - |
| 37 | 39632981127598108 | 1473318613122088448 | 3 | 11.9 | 39.5 | 30.9 | OK | - |

### Column Definitions

- **N**: Number of DESI RV epochs passing quality cuts
- **S_robust**: Conservative significance = min(S, S_min_LOO) where S = ΔRV / σ_combined
- **ΔRV**: Maximum RV range (km/s)
- **MJD span**: Time baseline of observations (days)
- **Gaia Flag**: `OK` = RUWE ≤ 1.4; `SUSPECT` = RUWE > 1.4; `/VAR` = photometric variability
- **LAMOST**: `YES` = has LAMOST DR5 spectrum within 3 arcsec

### Notable Targets

1. **Rank 3** (39633437979575384): Highest S_robust (71.7), 355-day baseline
2. **Rank 5** (39627830035744797): Gaia photometric variable flag set
3. **Rank 11** (39627721168388315): Largest ΔRV (330.5 km/s)
4. **Rank 23** (39627745210139276): Elevated Gaia RUWE (1.95), has LAMOST match
5. **Rank 29** (39627817100513868): Very large ΔRV (381.8 km/s) over 9 days

---

## Validation Summary

### Gaia DR3 Cross-Match

| Metric | Count |
|--------|-------|
| Gaia queries succeeded | 21/21 |
| RUWE > 1.4 (astrometric anomaly) | 1 |
| Photometric variability flag | 1 |
| Gaia RVS data available | 0 |

### LAMOST DR5 Cross-Match

| Metric | Count |
|--------|-------|
| LAMOST matches (3 arcsec) | 2 |
| No LAMOST coverage | 19 |

### SIMBAD Classification (Excluded from Follow-Up)

| Class | Count | Action |
|-------|-------|--------|
| QSO/AGN | 4 | Excluded (extragalactic) |
| RR Lyrae (RR*) | 11 | Excluded (pulsating) |
| Eclipsing Binary (EB*) | 2 | Excluded (known binary) |
| BY Draconis (BY*) | 1 | Excluded (known binary) |
| **Remaining candidates** | **21** | **Follow-up worthy** |

---

## Metric Definitions

| Metric | Definition |
|--------|------------|
| S | ΔRV_max / sqrt(Σ σ_i²) — significance of RV variation |
| S_robust | min(S, S_min_LOO) — conservative significance after leave-one-out |
| S_min_LOO | Minimum S when any single epoch is dropped |
| d_max | max_i \|RV_i - median(RV)\| / σ_i — outlier leverage metric |
| ΔRV | max(RV) - min(RV) in km/s |

---

## Output Files

### Primary Outputs

| File | Description |
|------|-------------|
| `data/derived/priorityA_followup_only.csv` | 21 validated candidates |
| `data/derived/priorityA_followup_only_annotated_gaia_lamost.csv` | With Gaia + LAMOST annotations |
| `data/derived/deep_dive_clean_subset.csv` | Deep-dive validated candidates |

### Reports

| File | Description |
|------|-------------|
| `data/derived/VALIDATION_GAIA_LAMOST_REPORT.md` | Gaia + LAMOST cross-match report |
| `data/derived/NSS_SIMBAD_CROSSCHECK_REPORT_REVISED.md` | SIMBAD classification report |
| `data/derived/observer_packet/README_packet.md` | Observer-ready documentation |

### Plots

| Directory | Description |
|-----------|-------------|
| `data/derived/deep_dive_clean_plots/` | RV vs MJD for top 10 candidates |
| `data/derived/plots/` | RV plots for all Priority A candidates |

---

## Quick Start

```bash
# 1. Download DESI DR1 MWS RV data
./fetch_desi_dr1_mws_rv.sh

# 2. Run full analysis pipeline
python analyze_rv_candidates.py

# 3. Triage and build priority packet
python triage_rv_candidates.py --survey bright
python triage_rv_candidates.py --survey dark
python build_priority_packet.py

# 4. Add variable class annotations
python add_variable_class.py

# 5. Cross-match with Gaia and LAMOST
python validate_gaia_lamost.py
```

---

## Data Provenance

- **Source**: DESI DR1 Milky Way Survey "iron" VAC
- **Files**: rvpix_exp-main-bright.fits, rvpix_exp-main-dark.fits
- **URL**: https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0/
- **Cross-match**: Gaia DR3 (ESA TAP), SIMBAD (CDS), LAMOST DR5 (VizieR)

---

## Quality Cuts Applied

### Per-Measurement Cuts

- `VRAD_ERR < 10 km/s` — reject high-error measurements
- `|VRAD| < 500 km/s` — reject extreme/non-physical velocities
- `finite(VRAD) & finite(VRAD_ERR)` — reject NaN values

### Per-Source Cuts

- `N_epochs ≥ 3` — minimum epochs for robust statistics (Tier A)
- `S_robust ≥ 10` — high-significance variability (Priority A)
- `MJD_span ≥ 0.5 days` — exclude same-night duplicates

---

## Caveats

1. **No orbit fitting performed** — ΔRV is max-min, not orbital amplitude
2. **No stellar parameter cuts** — no filtering by Teff, logg, [Fe/H]
3. **SIMBAD may be incomplete** — absence of classification ≠ novelty
4. **Single-epoch LAMOST RVs** — comparison to DESI requires care
5. **Gaia RUWE > 1.4** — may indicate binarity OR astrometric issues

---

## References

- [DESI DR1 MWS VAC Documentation](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws/)
- [DESI DR1 Stellar Catalogue Paper](https://arxiv.org/abs/2505.14787)
- [Gaia DR3 Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [LAMOST DR5 at VizieR](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=V/164)

---

## License

This pipeline is for use with publicly released DESI data. See DESI data policies for usage terms.

---

*Pipeline developed 2026-01-13*
