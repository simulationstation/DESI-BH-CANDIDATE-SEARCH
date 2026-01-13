# DESI DR1 MWS Radial Velocity Pipeline for Compact Object Candidate Search

Pipeline for downloading and analyzing DESI Data Release 1 Milky Way Survey radial velocity data to identify candidate compact object companions (black holes, neutron stars) via RV variability.

## Quick Start

```bash
# 1. Download minimal test data (~129 MB)
chmod +x fetch_desi_dr1_mws_rv.sh
./fetch_desi_dr1_mws_rv.sh --quick

# 2. Run smoke test
python smoke_test_desi_mws_rv.py --data-root data

# 3. View results
cat data/derived/smoke_top20.csv
```

## Data Products

### Selected Files

| File | Size | Description | Why Needed |
|------|------|-------------|------------|
| `mwsall-pix-iron.fits` | 13.1 GB | Combined MWS catalog with Gaia crossmatch | Best/primary observation per source, Gaia IDs |
| `rvpix_exp-main-bright.fits` | 5.6 GB | Single-epoch RV (main survey, bright) | Per-epoch RV for multi-epoch ΔRV analysis |
| `rvpix_exp-main-dark.fits` | 2.9 GB | Single-epoch RV (main survey, dark) | Additional single-epoch measurements |
| `rvpix_exp-special-bright.fits` | 129 MB | Single-epoch RV (special program) | Quick-mode smoke test file |

### URLs and Checksums

Base URL: `https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0/`

```
mwsall-pix-iron.fits
  URL: https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0/mwsall-pix-iron.fits
  Size: 13,051,186,560 bytes
  SHA256: eae3b31807c58ac340c257e06f66935d01bb81698a08108e6ca57a372cab76b5

rvpix_exp-main-bright.fits
  URL: https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0/rv_output/240521/rvpix_exp-main-bright.fits
  Size: ~6,013,747,200 bytes

rvpix_exp-special-bright.fits (quick mode)
  URL: https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0/rv_output/240521/rvpix_exp-special-bright.fits
  Size: ~135,266,880 bytes
```

## File Structure and Column Names

### Single-Epoch Files (`rvpix_exp-*.fits`)

| HDU | Name | Key Columns | Description |
|-----|------|-------------|-------------|
| 1 | RVTAB | `VRAD`, `VRAD_ERR`, `TARGETID` | RVSpecFit RV measurements |
| 2 | FIBERMAP | `TARGETID`, `MJD`, `EXPID`, `NIGHT`, `TILEID` | Observation metadata |
| 3 | SCORES | `TSNR2_*`, SNR columns | Quality metrics |
| 4 | GAIA | `SOURCE_ID` | Gaia DR3 crossmatch |

### Combined Catalog (`mwsall-pix-iron.fits`)

| HDU | Name | Key Columns | Description |
|-----|------|-------------|-------------|
| 1 | RVTAB | `VRAD`, `VRAD_ERR`, `TARGETID`, `PRIMARY` | Best RVSpecFit per source |
| 2 | SPTAB | `RV_ADOP`, `RV_ERR`, `TEFF`, `LOGG`, `FEH` | FERRE stellar parameters |
| 3 | FIBERMAP | Targeting columns | Target information |
| 4 | SCORES | Quality metrics | Spectral quality |
| 5 | GAIA | `SOURCE_ID`, all Gaia DR3 columns | Full Gaia crossmatch |

### Key Column Descriptions

- **VRAD**: Heliocentric radial velocity from RVSpecFit (km/s)
- **VRAD_ERR**: RV uncertainty (km/s)
- **TARGETID**: Unique DESI target identifier (int64)
- **SOURCE_ID**: Gaia DR3 source_id (int64)
- **MJD**: Modified Julian Date of observation
- **EXPID**: DESI exposure ID
- **NIGHT**: Observation night (YYYYMMDD format)
- **PRIMARY**: Boolean flag for best observation (highest R-band SNR)

## Directory Layout

```
data/
├── raw/                    # Downloaded FITS files
│   ├── mwsall-pix-iron.fits
│   └── rvpix_exp-*.fits
├── derived/                # Analysis outputs
│   └── smoke_top20.csv
├── logs/                   # Download logs
│   └── download.log
└── manifests/              # Download manifests
    └── download_manifest_*.txt
```

## Usage

### Download Options

```bash
# Full download (~16 GB) - main survey files
./fetch_desi_dr1_mws_rv.sh

# Quick mode (~129 MB) - minimal test file
./fetch_desi_dr1_mws_rv.sh --quick

# Verify existing downloads
./fetch_desi_dr1_mws_rv.sh --verify

# Custom data directory
./fetch_desi_dr1_mws_rv.sh --data-root /path/to/data
```

### Smoke Test Options

```bash
# Basic run
python smoke_test_desi_mws_rv.py --data-root data

# Limit rows for faster testing
python smoke_test_desi_mws_rv.py --data-root data --max-rows 1000000

# Require more epochs
python smoke_test_desi_mws_rv.py --data-root data --min-epochs 3

# Output more candidates
python smoke_test_desi_mws_rv.py --data-root data --top-n 50
```

## Verification: Confirming True Multi-Epoch Data

The smoke test automatically checks for **true** multi-epoch observations:

1. **Multiple spectrographs caveat**: DESI uses 10 spectrographs simultaneously. A single observation can produce multiple table rows with the same MJD but different fibers. These are NOT multi-epoch.

2. **True multi-epoch**: Different observation nights (MJD difference > 0.5 days).

3. **The test reports**:
   - How many sources have observations on different nights
   - Warning if most "multi-epoch" data is actually same-night

## Gotchas and Notes

### 1. Duplicate Rows
- Single-epoch files may have multiple rows per TARGETID from different exposures
- Use EXPID or MJD to distinguish unique observations
- The combined catalog (`mwsall-pix-iron.fits`) has one row per TARGETID with `PRIMARY=True` for best observation

### 2. Heliocentric Corrections
- `VRAD` is already heliocentric-corrected (Earth motion removed)
- Safe to directly compare RVs from different epochs

### 3. Multi-Spectrograph Observations
- Same night observations from multiple spectrographs should NOT be treated as multi-epoch
- Filter by requiring MJD difference > 0.5 days between epochs

### 4. Gaia Source ID Availability
- The GAIA HDU in rvpix_exp files contains crossmatched Gaia DR3 source_id
- Crossmatch radius: 1 arcsecond
- Some targets may not have Gaia matches (faint sources, crowded fields)

### 5. Quality Flags
- Check `VRAD_ERR` - large errors (>10 km/s) may indicate poor fits
- The `TSNR2_*` columns in SCORES indicate spectral quality
- RV values with |VRAD| > 500 km/s are likely non-MW or problematic

### 6. Survey/Program Combinations
DR1 includes multiple surveys with different targeting:
- `main-bright`: Bright stars (G < 19)
- `main-dark`: Fainter stars
- `sv1/sv2/sv3`: Survey validation phases
- `special`: Special targeting programs

## Joining Tables (If Needed)

If Gaia IDs are not in your chosen file, join via TARGETID:

```python
import fitsio

# Load single-epoch RVs
rv_data = fitsio.read('rvpix_exp-main-bright.fits', ext='RVTAB')

# Load Gaia crossmatch from combined catalog
gaia_data = fitsio.read('mwsall-pix-iron.fits', ext='GAIA')
fibermap = fitsio.read('mwsall-pix-iron.fits', ext='FIBERMAP')

# Join on TARGETID
# Create lookup: TARGETID -> SOURCE_ID
targetid_to_gaia = dict(zip(fibermap['TARGETID'], gaia_data['SOURCE_ID']))
```

## Requirements

- Python 3.8+
- numpy
- fitsio (preferred) or astropy
- aria2c (preferred) or wget

Install dependencies:
```bash
pip install numpy fitsio
# or
pip install numpy astropy
```

## References

- [DESI DR1 MWS VAC Documentation](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws/)
- [DESI DR1 Stellar Catalogue Paper](https://arxiv.org/abs/2505.14787)
- [DESI MWS DR1 Data Model](https://desi-mws-dr1-datamodel.readthedocs.io/)
- [DESI DR1 Tutorials](https://github.com/desimilkyway/dr1_tutorials)

## License

This pipeline is for use with publicly released DESI data. See DESI data policies for usage terms.
