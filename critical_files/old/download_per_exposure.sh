#!/bin/bash
# Download DESI per-exposure spectra for target 39627745210139276
# Correct paths: redux/iron/exposures/{NIGHT}/{EXPID}/cframe-{cam}-{EXPID}.fits.gz

set -e

mkdir -p data/per_exposure

BASE_URL="https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/exposures"

# EXPID 114768, Night 20211219, Petal 2, Fiber 1377
echo "Downloading EXPID 114768 (Night 20211219, Petal 2)..."
wget -q -O "data/per_exposure/cframe-b2-00114768.fits.gz" "${BASE_URL}/20211219/00114768/cframe-b2-00114768.fits.gz"
wget -q -O "data/per_exposure/cframe-r2-00114768.fits.gz" "${BASE_URL}/20211219/00114768/cframe-r2-00114768.fits.gz"
wget -q -O "data/per_exposure/cframe-z2-00114768.fits.gz" "${BASE_URL}/20211219/00114768/cframe-z2-00114768.fits.gz"

# EXPID 120194, Night 20220125, Petal 7, Fiber 3816
echo "Downloading EXPID 120194 (Night 20220125, Petal 7)..."
wget -q -O "data/per_exposure/cframe-b7-00120194.fits.gz" "${BASE_URL}/20220125/00120194/cframe-b7-00120194.fits.gz"
wget -q -O "data/per_exposure/cframe-r7-00120194.fits.gz" "${BASE_URL}/20220125/00120194/cframe-r7-00120194.fits.gz"
wget -q -O "data/per_exposure/cframe-z7-00120194.fits.gz" "${BASE_URL}/20220125/00120194/cframe-z7-00120194.fits.gz"

# EXPID 120449, Night 20220127, Petal 0, Fiber 68
echo "Downloading EXPID 120449 (Night 20220127, Petal 0)..."
wget -q -O "data/per_exposure/cframe-b0-00120449.fits.gz" "${BASE_URL}/20220127/00120449/cframe-b0-00120449.fits.gz"
wget -q -O "data/per_exposure/cframe-r0-00120449.fits.gz" "${BASE_URL}/20220127/00120449/cframe-r0-00120449.fits.gz"
wget -q -O "data/per_exposure/cframe-z0-00120449.fits.gz" "${BASE_URL}/20220127/00120449/cframe-z0-00120449.fits.gz"

# EXPID 120450, Night 20220127, Petal 0, Fiber 68
echo "Downloading EXPID 120450 (Night 20220127, Petal 0)..."
wget -q -O "data/per_exposure/cframe-b0-00120450.fits.gz" "${BASE_URL}/20220127/00120450/cframe-b0-00120450.fits.gz"
wget -q -O "data/per_exposure/cframe-r0-00120450.fits.gz" "${BASE_URL}/20220127/00120450/cframe-r0-00120450.fits.gz"
wget -q -O "data/per_exposure/cframe-z0-00120450.fits.gz" "${BASE_URL}/20220127/00120450/cframe-z0-00120450.fits.gz"

echo ""
echo "Download complete. Decompressing..."
gunzip -f data/per_exposure/*.gz

echo ""
echo "Files saved to data/per_exposure/:"
ls -la data/per_exposure/

echo ""
echo "Fiber indices for each exposure:"
echo "  EXPID 114768 (Petal 2): FIBER = 1377"
echo "  EXPID 120194 (Petal 7): FIBER = 3816"
echo "  EXPID 120449 (Petal 0): FIBER = 68"
echo "  EXPID 120450 (Petal 0): FIBER = 68"
