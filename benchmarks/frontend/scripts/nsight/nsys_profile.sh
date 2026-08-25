#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Nsight Systems profiling wrapper for dynamo frontend.
# Captures NVTX ranges and CPU samples. Context switches are disabled
# (--cpuctxsw=none) to reduce overhead.
#
# Prerequisites:
#   - nsys (Nsight Systems CLI) installed
#   - Binary built with: cargo build --profile profiling --features dynamo-runtime/nvtx
#
# Usage:
#   ./nsys_profile.sh <binary> [args...]
#   ./nsys_profile.sh --duration 60 <binary> [args...]
#   DURATION=30 ./nsys_profile.sh target/profiling/dynamo-frontend

set -euo pipefail

DURATION="${DURATION:-30}"
OUTPUT_PREFIX="dynamo_frontend_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-.}"

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)    DURATION="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --output)      OUTPUT_PREFIX="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] <binary> [binary-args...]"
            echo ""
            echo "Options:"
            echo "  --duration N      Profile duration in seconds (default: 30)"
            echo "  --output-dir DIR  Output directory (default: .)"
            echo "  --output PREFIX   Output file prefix (default: dynamo_frontend_<timestamp>)"
            echo ""
            echo "Environment:"
            echo "  DYN_NVTX=1 is set automatically"
            echo ""
            echo "Build the binary first:"
            echo "  cargo build --profile profiling --features dynamo-runtime/nvtx"
            exit 0
            ;;
        *)  break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "ERROR: No binary specified."
    echo "Usage: $0 [OPTIONS] <binary> [binary-args...]"
    exit 1
fi

BINARY="$1"
shift

if ! command -v nsys &>/dev/null; then
    echo "ERROR: nsys not found. Install Nsight Systems."
    exit 1
fi

if ! command -v "$BINARY" &>/dev/null && [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found or not executable: $BINARY"
    echo "Build with: cargo build --profile profiling --features dynamo-runtime/nvtx"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# One switch for both layers: the Rust runtime (lib/runtime/src/nvtx.rs) and
# the Python components (dynamo.common.utils.nvtx_utils) both read this.
export DYN_NVTX=1

# nvtx_utils imports the nvtx package eagerly once the switch is on, so a Python
# target dies at startup without it. Warned rather than fatal: this script also
# profiles pure-Rust binaries, which need nothing from Python.
if ! python3 -c "import nvtx" 2>/dev/null; then
    echo "WARNING: the nvtx package is not installed. DYN_NVTX=1 is exported above, so a"
    echo "         Python target will raise ImportError at startup."
    echo "         Install: pip install 'ai-dynamo[profiling]'"
fi

echo "Profiling: $BINARY $*"
echo "Duration: ${DURATION}s"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_PREFIX}.nsys-rep"

nsys profile \
    --trace=osrt,nvtx \
    --sample=cpu \
    --cpuctxsw=none \
    --output="${OUTPUT_DIR}/${OUTPUT_PREFIX}" \
    --duration="$DURATION" \
    --force-overwrite=true \
    "$BINARY" "$@"

echo ""
echo "Profile saved: ${OUTPUT_DIR}/${OUTPUT_PREFIX}.nsys-rep"
echo "View with: nsys-ui ${OUTPUT_DIR}/${OUTPUT_PREFIX}.nsys-rep"
