#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Encode-Aggregated (E/Agg) disaggregated serving with the sample backend.
#
# Spawns dynamo.frontend plus one Encode worker and one Aggregated worker.
# The Encode worker runs the synthetic vision-encoder forward pass
# (_run_encoder) and emits a terminal chunk with encoder_result.
# The frontend's EncodeRouter forwards that payload as encoder_result on
# the downstream request; the Aggregated worker validates its shape.
#
# NOTE: End-to-end encoder handoff requires the frontend to support an
# EncodeRouter that reads encoder_result from the encode terminal and
# injects it into the downstream request.  Until that is landed, this
# script validates the per-worker lifecycle and E/Agg topology discovery.
#
# GPUs: 0 (CPU-only, useful for CI smoke tests of the encode disagg path).

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL_NAME="${MODEL_NAME:-sample-model}"

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
            echo "Any additional options are passed through to both sample workers."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap 'echo Cleaning up...; kill 0' EXIT

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Sample Encode-Aggregated Serving (CPU-only)" "$MODEL_NAME" "$HTTP_PORT"

# run frontend
python3 -m dynamo.frontend &

# Encode worker: runs _run_encoder, emits terminal chunk with encoder_result.
# Registered as WorkerType::Encode so the frontend can route to it.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.common.backend.sample_main \
  --model-name "$MODEL_NAME" \
  --component sample-encode \
  --disaggregation-mode encode \
  "${EXTRA_ARGS[@]}" &

# Aggregated worker: validates encoder_result forwarded by the frontend.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
python3 -m dynamo.common.backend.sample_main \
  --model-name "$MODEL_NAME" \
  --component sample-aggregated \
  --disaggregation-mode agg \
  "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
