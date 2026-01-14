# DESI DR1 Pairwise kSZ Analysis Pipeline

Publication-grade pairwise kinetic Sunyaev-Zel'dovich (kSZ) measurement using DESI DR1 galaxies cross-correlated with CMB temperature maps.

---

## Overview

The pairwise kSZ effect measures the momentum of galaxy pairs through the CMB temperature decrement/increment caused by Compton scattering. This pipeline measures:

```
p̂(r) = Σ_{ij} w_ij (T_i - T_j) c_ij / Σ_{ij} w_ij c_ij²
```

where:
- `T_i` = CMB temperature at galaxy i position (μK)
- `c_ij` = geometric weight = ½ r̂_ij · (r̂_i - r̂_j)
- `w_ij` = combined galaxy weights
- `r` = comoving pair separation (Mpc/h)

---

## Key Features

- **Tomographic analysis**: Minimum 3 redshift bins for LRG
- **Multi-map cross-check**: Planck PR4 + ACT DR6 consistency
- **Automated gating**: PASS/FAIL/INCONCLUSIVE decision system
- **Full systematics suite**: 8+ null tests, injection recovery, tSZ controls
- **Publication-ready outputs**: Bundled results with checksums

---

## Pipeline Architecture

```
desi_ksz/
├── cli.py                  # Click-based CLI (35+ commands)
├── config/                 # Configuration schemas and presets
├── io/                     # DESI catalogs + CMB map I/O
├── selection/              # Quality cuts, z-bins, weights
├── maps/                   # HEALPix/CAR utilities, filtering, masking
├── estimators/             # Pairwise momentum, aperture photometry
├── covariance/             # Jackknife, Hartlap correction
├── systematics/            # Null tests, tSZ leakage, referee checks
├── sims/                   # Injection tests, CMB realizations
├── inference/              # Likelihood, MCMC
├── plots/                  # Publication figures
├── runner/                 # Phase 3-4 execution driver, gates
├── results/                # Results packager
└── tests/                  # Comprehensive test suite
```

---

## Quick Start

### Planck-Only Run (Minimum Viable)

```bash
# Prepare data
python -m desi_ksz.cli ingest-desi --tracer LRG --output data/ksz/catalogs
python -m desi_ksz.cli ingest-maps --source planck_pr4 --output data/ksz/maps

# Run Phase 3-4 measurement
python -m desi_ksz.cli run-phase34 \
    --catalog-file data/ksz/catalogs/LRG_catalog.npz \
    --map-file data/ksz/maps/planck_pr4_smica.fits \
    --tracer LRG \
    --z-bins 0.4,0.5,0.6,0.7 \
    --jackknife-k 100 \
    --n-injection 100 \
    --n-null 500 \
    --tsz-mask-radii 5,10,15,20 \
    --output data/ksz/phase34_planck

# Package results
python -m desi_ksz.cli package-results \
    --result-file data/ksz/phase34_planck/phase34_result.json \
    --output data/ksz/bundle_planck
```

### ACT + Planck Run (Full Cross-Check)

```bash
# Prepare ACT maps
python -m desi_ksz.cli ingest-maps --source act_dr6 --output data/ksz/maps

# Run with two independent maps
python -m desi_ksz.cli run-phase34 \
    --catalog-file data/ksz/catalogs/LRG_catalog.npz \
    --map-file data/ksz/maps/act_dr6_coadd_f150.fits \
    --map-file-secondary data/ksz/maps/planck_pr4_smica.fits \
    --tracer LRG \
    --z-bins 0.4,0.5,0.6,0.7,0.8 \
    --jackknife-k 50,100,200 \
    --n-injection 200 \
    --n-null 1000 \
    --tsz-mask-radii 5,10,15,20,25 \
    --run-referee-checks \
    --output data/ksz/phase34_act_planck

# Package with full bundle
python -m desi_ksz.cli package-results \
    --result-file data/ksz/phase34_act_planck/phase34_result.json \
    --bundle-name ksz_lrg_act_planck_v1 \
    --output data/ksz/bundle_final
```

---

## Gate System

The pipeline enforces quality gates with automatic PASS/FAIL/INCONCLUSIVE decisions:

### Critical Gates (Cause FAIL)

| Gate | Threshold | Description |
|------|-----------|-------------|
| `injection_bias` | \|bias\| < 2σ | Signal injection recovery unbiased |
| `null_suite_pass_rate` | ≥ 80% | Null test suite pass rate |
| `transfer_test` | \|bias\| < 5% | Map transfer function test |
| `tsz_sweep_stability` | Δ < 1σ | tSZ mask sweep amplitude stable |
| `covariance_hartlap` | α > 0.5 | Hartlap factor valid |
| `map_consistency` | Δ < 2σ | ACT vs Planck consistent |

### Warning Gates (Cause WARN)

| Gate | Threshold | Description |
|------|-----------|-------------|
| `covariance_condition` | κ < 10⁶ | Covariance condition number |
| `look_elsewhere` | p_adj > 0.01 | Look-elsewhere correction |
| `anisotropy_residual` | < 3σ | Temperature anisotropy check |
| `weight_leverage` | < 1σ | Weight leverage stability |
| `split_consistency` | < 2σ | z-dependent split consistency |
| `beam_sensitivity` | < 1σ | Beam perturbation sensitivity |
| `ymap_regression` | < 2σ | y-map regression shift |

### Decision Logic

- **PASS**: All critical gates pass, ≤1 warning
- **FAIL**: Any critical gate fails
- **INCONCLUSIVE**: Critical pass but multiple warnings

---

## Results Bundle Structure

```
ksz_results_YYYYMMDD_HHMMSS/
├── plots/                  # Publication figures (PDF + PNG)
├── tables/                 # CSV data tables
│   ├── pairwise_momentum_*.csv
│   ├── amplitude_summary.csv
│   ├── null_test_summary.csv
│   └── gate_evaluation.csv
├── configs/                # YAML configuration used
├── data/                   # NPY/NPZ data files
├── manifest.json           # File listing with metadata
├── checksums.sha256        # SHA256 for reproducibility
├── summary.json            # Machine-readable results
└── results.md              # Human-readable decision report
```

---

## Non-Negotiable Requirements

The pipeline enforces these measurement requirements:

1. **Two independent map products** (ACT + Planck)
2. **LRG tomography with ≥3 z-bins**
3. **Detection significance explicitly reported**
4. **Jackknife covariance** with K ∈ {50, 100, 200}
5. **tSZ leakage controls** via cluster mask sweep
6. **Transfer function test** (< 5% bias)

---

## Referee Attack Checks

Five additional checks for robustness:

1. **Look-elsewhere**: Sidak correction for trials factor
2. **Anisotropy**: Spherical harmonic fitting (dipole/quadrupole)
3. **Weight leverage**: Remove top 1% highest-weight galaxies
4. **z-split**: Sky density quartile consistency
5. **Beam sensitivity**: ±5% FWHM perturbation

---

## CLI Commands Reference

```bash
# Pipeline stages
python -m desi_ksz.cli pipeline --config config.yaml
python -m desi_ksz.cli ingest-desi --tracer LRG
python -m desi_ksz.cli ingest-maps --source act_dr6
python -m desi_ksz.cli compute-pairwise --z-bins 0.4,0.5,0.6,0.7
python -m desi_ksz.cli covariance --method jackknife --n-regions 100

# Validation
python -m desi_ksz.cli injection-test -c catalog.npz -n 100
python -m desi_ksz.cli null-suite -c catalog.npz -n 500
python -m desi_ksz.cli transfer-test -c catalog.npz --nside 512
python -m desi_ksz.cli tsz-sweep -c catalog.npz --mask-radii 0,5,10,15,20

# Phase 3-4 execution
python -m desi_ksz.cli run-phase34 --catalog-file cat.npz --map-file map.fits
python -m desi_ksz.cli package-results --result-file result.json

# Utilities
python -m desi_ksz.cli info
python -m desi_ksz.cli validate-config --config config.yaml
```

---

## Data Requirements

| Dataset | Source | Size |
|---------|--------|------|
| DESI LRG | data.desi.lbl.gov/public/dr1 | ~5 GB |
| Planck PR4 | NERSC / PLA | ~1.5 GB |
| ACT DR6 | LAMBDA / NERSC | ~15 GB |

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
astropy>=5.3
healpy>=1.16
pixell>=0.20  # for ACT CAR maps
fitsio>=1.2
emcee>=3.1
click>=8.1
pyyaml>=6.0
h5py>=3.8
matplotlib>=3.7
```

Install with:
```bash
pip install -r requirements-ksz.txt
```

---

## Running Tests

```bash
# All tests
pytest desi_ksz/tests/ -v

# Specific modules
pytest desi_ksz/tests/test_gates.py -v
pytest desi_ksz/tests/test_packager.py -v
pytest desi_ksz/tests/test_referee_checks.py -v
pytest desi_ksz/tests/test_estimators.py -v
```

---

## Pairwise kSZ Theory

The pairwise momentum estimator measures the mean relative velocity of galaxy pairs:

```
p(r) = -τ̄ · T_CMB · v_12(r) / c
```

where:
- `τ̄` = mean optical depth of halos
- `T_CMB` = 2.725 K
- `v_12(r)` = mean pairwise velocity at separation r

The linear theory prediction:
```
v_12(r,z) = -(2/3) · f(z) H(z) a r ξ̄(r,z) / [1 + ξ(r,z)]
```

where `f(z) ≈ Ω_m(z)^0.55` is the growth rate.

---

## References

- [DESI DR1 LSS Catalogs](https://data.desi.lbl.gov/public/dr1/survey/catalogs/)
- [ACT DR6 Maps](https://lambda.gsfc.nasa.gov/product/act/)
- [Planck PR4](https://wiki.cosmos.esa.int/planck-legacy-archive/)
- Hand et al. (2012): First pairwise kSZ detection
- Schaan et al. (2021): ACT × BOSS kSZ measurement
- Planck Collaboration (2016): Planck kSZ constraints

---

## License

This pipeline is for use with publicly released DESI data. See DESI data policies for usage terms.

---

## Citation

If you use this pipeline, please cite:
- DESI Collaboration (2024) for DR1 data
- Relevant CMB map papers (Planck PR4, ACT DR6)

---

*Pipeline developed 2026-01-14*
