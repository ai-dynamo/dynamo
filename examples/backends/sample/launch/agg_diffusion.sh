#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving with the sample diffusion (raw-media) backend.
# Exercises the DiffusionEngine path (frontend -> RawEngineAdapter ->
# SampleDiffusionEngine) end-to-end on CPU — no GPU, no model.
# GPUs: 0 (CPU-only)

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default values
MODEL_NAME="${MODEL_NAME:-sample-diffusion-model}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-name <name>  Specify model name (default: $MODEL_NAME)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Any additional options are passed through to sample_diffusion_main."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Sample Diffusion Aggregated Serving" "$MODEL_NAME" "$HTTP_PORT"

# run frontend
python3 -m dynamo.frontend &

# run sample diffusion worker (serves /v1/images/generations)
python3 -m dynamo.common.backend.sample_diffusion_main \
  --model-name "$MODEL_NAME" \
  --endpoint-types images \
  "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
