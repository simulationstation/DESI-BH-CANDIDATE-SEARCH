# Branch Audit (main)

## Scope
This audit focuses on preparing the main branch for the DESI DR1 dark companion candidate work (Gaia DR3 3802130935635096832) and removing unrelated material.

## Cleanup Performed
- Removed the kSZ project from this branch (moved out of scope for the main dark companion work). Deleted:
  - `desi_ksz/`
  - `ksz_iteration0.py`, `ksz_iteration1.py`, `ksz_iteration2.py`, `ksz_null_test.py`
  - `run_ksz.sh`, `run_real_ksz_analysis.py`, `run_real_ksz_jackknife.py`, `run_real_ksz_parallel.py`
  - `requirements-ksz.txt`

## Open Issues / Known Limitations
- **Large data not included**: The core analysis depends on ~8.8 GB of external data (DESI RV files, DESI spectra, PHOENIX templates). These are now documented in `README.md`, but they are not present in the repo and must be fetched by reviewers.
- **DESI arm discrepancy**: A persistent R–Z arm RV discrepancy remains unresolved (documented in the analysis reports and README). This is a known systematic that prevents a definitive dynamical claim from public data alone.
- **Blend contamination**: A Gaia-resolved neighbor at 0.688" separation contaminates the DESI fiber. Blend-aware remeasurement is documented, but high-resolution follow-up is still required to fully resolve this.

## Notes for Reviewers
- The analysis outputs and reports in `outputs/` and the `ANALYSIS_REPORT_v*.md` files are intended to enable auditability without full re-processing.
- All scripts referenced in the README remain in place for reproducibility once the large data files are obtained.
