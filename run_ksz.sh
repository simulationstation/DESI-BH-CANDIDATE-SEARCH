#!/bin/bash
# DESI kSZ Analysis Pipeline Runner
#
# Usage:
#   ./run_ksz.sh                    # Run full pipeline with defaults
#   ./run_ksz.sh --tracer BGS       # Specify tracer
#   ./run_ksz.sh --help             # Show help
#
# Prerequisites:
#   pip install -r requirements-ksz.txt
#
# Data:
#   Download DESI LSS catalogs and CMB maps first, or use --skip-download

set -e  # Exit on error

# Default values
TRACER="LRG"
CMB_SOURCE="act_dr6"
OUTPUT_DIR="data/ksz"
CONFIG_FILE=""
SKIP_DOWNLOAD=false
N_REGIONS=100
N_REALIZATIONS=1000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tracer)
            TRACER="$2"
            shift 2
            ;;
        --cmb-source)
            CMB_SOURCE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --n-regions)
            N_REGIONS="$2"
            shift 2
            ;;
        --n-realizations)
            N_REALIZATIONS="$2"
            shift 2
            ;;
        --help)
            echo "DESI kSZ Analysis Pipeline"
            echo ""
            echo "Usage: ./run_ksz.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tracer TRACER       Galaxy tracer (BGS, LRG, ELG, QSO) [default: LRG]"
            echo "  --cmb-source SOURCE   CMB map source (act_dr6, planck_pr4) [default: act_dr6]"
            echo "  --output DIR          Output directory [default: data/ksz]"
            echo "  --config FILE         YAML configuration file"
            echo "  --skip-download       Skip data download step"
            echo "  --n-regions N         Number of jackknife regions [default: 100]"
            echo "  --n-realizations N    Number of null test realizations [default: 1000]"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "DESI kSZ Analysis Pipeline"
echo "=============================================="
echo "Tracer: $TRACER"
echo "CMB Source: $CMB_SOURCE"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Python module is available
if ! python3 -c "import desi_ksz" 2>/dev/null; then
    echo "Warning: desi_ksz module not found in PYTHONPATH"
    echo "Adding current directory to PYTHONPATH"
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
fi

# Build CLI command
CLI_CMD="python3 -m desi_ksz.cli"

# If config file provided, use pipeline command
if [[ -n "$CONFIG_FILE" ]]; then
    echo "Using configuration file: $CONFIG_FILE"
    $CLI_CMD pipeline --config "$CONFIG_FILE" --output "$OUTPUT_DIR"
else
    # Run individual steps

    echo ""
    echo "[Step 1/10] Ingesting DESI catalogs..."
    $CLI_CMD ingest-desi --tracer "$TRACER" --output "$OUTPUT_DIR/catalogs"

    echo ""
    echo "[Step 2/10] Ingesting CMB maps..."
    $CLI_CMD ingest-maps --source "$CMB_SOURCE" --output "$OUTPUT_DIR/maps"

    echo ""
    echo "[Step 3/10] Creating masks..."
    $CLI_CMD make-masks --output "$OUTPUT_DIR/masks"

    echo ""
    echo "[Step 4/10] Filtering maps..."
    $CLI_CMD filter-maps --filter-type matched --output "$OUTPUT_DIR/filtered"

    echo ""
    echo "[Step 5/10] Measuring temperatures..."
    $CLI_CMD measure-temps --output "$OUTPUT_DIR/temperatures.h5"

    echo ""
    echo "[Step 6/10] Computing pairwise momentum..."
    $CLI_CMD compute-pairwise --output "$OUTPUT_DIR/pairwise"

    echo ""
    echo "[Step 7/10] Estimating covariance..."
    $CLI_CMD covariance --method jackknife --n-regions "$N_REGIONS" --output "$OUTPUT_DIR/covariance"

    echo ""
    echo "[Step 8/10] Running null tests..."
    $CLI_CMD null-tests --n-realizations "$N_REALIZATIONS" --output "$OUTPUT_DIR/nulls"

    echo ""
    echo "[Step 9/10] Running inference..."
    $CLI_CMD inference --method analytic --output "$OUTPUT_DIR/chains"

    echo ""
    echo "[Step 10/10] Generating plots..."
    $CLI_CMD make-plots --output "$OUTPUT_DIR/plots"
fi

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Results written to: $OUTPUT_DIR"
echo "=============================================="
