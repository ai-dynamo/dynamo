#!/bin/bash

# Post-process script wrapper for performance sweep results
# This script processes directories containing performance sweep results and extracts
# throughput data and deployment configuration, creating JSON summary files.

set -e

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/post_process.py"

# Function to show usage
usage() {
    echo "Usage: $0 <base_path> [--output-dir <output_directory>]"
    echo ""
    echo "Arguments:"
    echo "  base_path           Base directory containing performance sweep results"
    echo "  --output-dir DIR    Output directory for JSON files (default: same as input)"
    echo ""
    echo "Examples:"
    echo "  $0 ./8150-1024-20250806_174027"
    echo "  $0 ./results --output-dir ./summaries"
    echo ""
    echo "This script will:"
    echo "  1. Find all subdirectories matching pattern 'ctx*_gen*_*'"
    echo "  2. Extract throughput data from genai_perf_artifacts"
    echo "  3. Create JSON files with results for each subdirectory"
    exit 1
}

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed"
    exit 1
fi

# Parse arguments
if [ $# -eq 0 ]; then
    usage
fi

BASE_PATH="$1"
shift

OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate base path
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Base path '$BASE_PATH' does not exist or is not a directory"
    exit 1
fi

# Build command
CMD="python3 \"$PYTHON_SCRIPT\" \"$BASE_PATH\""
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

echo "Running post-process script..."
echo "Command: $CMD"
echo ""

# Execute the Python script
eval $CMD

echo ""
echo "Post-processing completed!" 