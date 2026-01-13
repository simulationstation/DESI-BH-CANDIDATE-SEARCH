#!/usr/bin/env bash
# =============================================================================
# fetch_desi_dr1_mws_rv.sh
# Downloads DESI DR1 Milky Way Survey RV products for compact-object companion search
#
# Usage:
#   ./fetch_desi_dr1_mws_rv.sh          # Full download (~16 GB)
#   ./fetch_desi_dr1_mws_rv.sh --quick  # Quick mode (~129 MB for smoke test)
#
# Requires: aria2c (preferred) or wget
# =============================================================================

set -euo pipefail

# Configuration
BASE_URL="https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0"
DATA_ROOT="${DATA_ROOT:-./data}"
QUICK_MODE=false
VERIFY_ONLY=false
ARIA2C_OPTS="-c -x 4 -s 4 --retry-wait=5 --max-tries=10"
WGET_OPTS="-c --retry-connrefused --tries=10 --waitretry=5"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK_MODE=true; shift ;;
        --verify) VERIFY_ONLY=true; shift ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--quick] [--verify] [--data-root PATH]"
            echo "  --quick      Download only minimal files for smoke test (~129 MB)"
            echo "  --verify     Verify existing files without downloading"
            echo "  --data-root  Set data root directory (default: ./data)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create directory structure
mkdir -p "${DATA_ROOT}/raw" "${DATA_ROOT}/derived" "${DATA_ROOT}/logs" "${DATA_ROOT}/manifests"

# Define files to download
# File format: URL|SIZE_BYTES|SHA256|DESCRIPTION
if [ "$QUICK_MODE" = true ]; then
    # Quick mode: smallest single-epoch file that still has multi-epoch data
    # AUDIT FIX: Correct file size is 129271680 bytes (verified via HEAD request)
    FILES=(
        "${BASE_URL}/rv_output/240521/rvpix_exp-special-bright.fits|129271680||Single-epoch RV (special-bright, ~123MB)"
    )
    echo "==> Quick mode: downloading minimal test file (~129 MB)"
else
    # Full mode: main survey single-epoch files + combined catalog
    FILES=(
        "${BASE_URL}/mwsall-pix-iron.fits|13051186560|eae3b31807c58ac340c257e06f66935d01bb81698a08108e6ca57a372cab76b5|Combined MWS catalog with Gaia (~13.1GB)"
        "${BASE_URL}/rv_output/240521/rvpix_exp-main-bright.fits|6013747200||Single-epoch RV main-bright (~5.6GB)"
        "${BASE_URL}/rv_output/240521/rvpix_exp-main-dark.fits|3113779200||Single-epoch RV main-dark (~2.9GB)"
    )
    echo "==> Full mode: downloading main survey files (~16 GB total)"
fi

# Detect download tool
if command -v aria2c &> /dev/null; then
    DOWNLOADER="aria2c"
    echo "==> Using aria2c for download"
elif command -v wget &> /dev/null; then
    DOWNLOADER="wget"
    echo "==> Using wget for download"
else
    echo "ERROR: Neither aria2c nor wget found. Please install one."
    exit 1
fi

# Create manifest
MANIFEST="${DATA_ROOT}/manifests/download_manifest_$(date +%Y%m%d_%H%M%S).txt"
echo "# DESI DR1 MWS RV Download Manifest" > "$MANIFEST"
echo "# Created: $(date -Iseconds)" >> "$MANIFEST"
echo "# Mode: $([ "$QUICK_MODE" = true ] && echo "quick" || echo "full")" >> "$MANIFEST"
echo "#" >> "$MANIFEST"
echo "# URL|EXPECTED_SIZE|SHA256|DESCRIPTION|LOCAL_PATH|ACTUAL_SIZE|STATUS" >> "$MANIFEST"

# Download function
download_file() {
    local url="$1"
    local expected_size="$2"
    local sha256="$3"
    local desc="$4"
    local filename
    filename=$(basename "$url")
    local dest="${DATA_ROOT}/raw/${filename}"

    echo ""
    echo "==> Downloading: ${filename}"
    echo "    Description: ${desc}"
    echo "    Expected size: ${expected_size} bytes"

    if [ "$VERIFY_ONLY" = true ]; then
        if [ -f "$dest" ]; then
            actual_size=$(stat --printf="%s" "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null)
            echo "    File exists: ${actual_size} bytes"
        else
            echo "    File NOT FOUND"
        fi
        return
    fi

    # Download
    if [ "$DOWNLOADER" = "aria2c" ]; then
        aria2c $ARIA2C_OPTS -d "${DATA_ROOT}/raw" -o "$filename" "$url" 2>&1 | tee -a "${DATA_ROOT}/logs/download.log"
    else
        wget $WGET_OPTS -O "$dest" "$url" 2>&1 | tee -a "${DATA_ROOT}/logs/download.log"
    fi

    # Verify size
    if [ -f "$dest" ]; then
        actual_size=$(stat --printf="%s" "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null)
        echo "    Downloaded: ${actual_size} bytes"

        # Verify SHA256 if available
        if [ -n "$sha256" ]; then
            echo "    Verifying SHA256..."
            computed_sha=$(sha256sum "$dest" | cut -d' ' -f1)
            if [ "$computed_sha" = "$sha256" ]; then
                echo "    SHA256: VERIFIED"
                status="OK"
            else
                echo "    SHA256: MISMATCH (expected: ${sha256}, got: ${computed_sha})"
                status="SHA256_MISMATCH"
            fi
        else
            # Verify FITS header is readable
            echo "    Verifying FITS header..."
            if python3 -c "import fitsio; fitsio.FITS('${dest}'); print('    FITS header: OK')" 2>/dev/null; then
                status="OK"
            elif python3 -c "from astropy.io import fits; fits.open('${dest}'); print('    FITS header: OK')" 2>/dev/null; then
                status="OK"
            else
                echo "    WARNING: Could not verify FITS header"
                status="UNVERIFIED"
            fi
        fi

        # Log to manifest
        echo "${url}|${expected_size}|${sha256}|${desc}|${dest}|${actual_size}|${status}" >> "$MANIFEST"
    else
        echo "    ERROR: Download failed"
        echo "${url}|${expected_size}|${sha256}|${desc}|${dest}||FAILED" >> "$MANIFEST"
        return 1
    fi
}

# Download all files
echo ""
echo "============================================================"
echo "Starting DESI DR1 MWS RV Download"
echo "============================================================"
echo "Data root: ${DATA_ROOT}"
echo "Manifest: ${MANIFEST}"
echo ""

for file_spec in "${FILES[@]}"; do
    IFS='|' read -r url size sha256 desc <<< "$file_spec"
    download_file "$url" "$size" "$sha256" "$desc"
done

echo ""
echo "============================================================"
echo "Download Complete"
echo "============================================================"
echo "Files saved to: ${DATA_ROOT}/raw/"
echo "Manifest: ${MANIFEST}"
echo ""
echo "Next steps:"
echo "  1. Run smoke test: python smoke_test_desi_mws_rv.py --data-root ${DATA_ROOT}"
echo ""

# Print summary
echo "Downloaded files:"
ls -lh "${DATA_ROOT}/raw/"*.fits 2>/dev/null || echo "  (no files yet)"
