#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only Encode/Prefill/Decode multimodal handoff smoke for the sample backend.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../../../..")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL_NAME="${MODEL_NAME:-$REPO_ROOT/lib/llm/tests/data/sample-models/TinyLlama_v1.1}"
NAMESPACE="${NAMESPACE:-dynamo}"
ENCODE_COMPONENT="${ENCODE_COMPONENT:-sample-multimodal-encode}"
PREFILL_COMPONENT="${PREFILL_COMPONENT:-sample-multimodal-prefill}"
DECODE_COMPONENT="${DECODE_COMPONENT:-sample-multimodal-decode}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model-name NAME] [--namespace NAMESPACE] [WORKER OPTIONS]"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# This direct worker smoke intentionally has no frontend. print_launch_banner
# always advertises a frontend URL, so use a scoped message instead.
echo "Running direct Encode -> Prefill -> Decode multimodal handoff smoke with $MODEL_NAME"

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.common.backend.sample_main \
  --model-name "$MODEL_NAME" \
  --namespace "$NAMESPACE" \
  --component "$ENCODE_COMPONENT" \
  --disaggregation-mode encode \
  --disable-kv-routing \
  "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
python3 -m dynamo.common.backend.sample_main \
  --model-name "$MODEL_NAME" \
  --namespace "$NAMESPACE" \
  --component "$PREFILL_COMPONENT" \
  --disaggregation-mode prefill \
  --route-to-encoder \
  --disable-kv-routing \
  "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT3:-8083} \
python3 -m dynamo.common.backend.sample_main \
  --model-name "$MODEL_NAME" \
  --namespace "$NAMESPACE" \
  --component "$DECODE_COMPONENT" \
  --disaggregation-mode decode \
  --disable-kv-routing \
  "${EXTRA_ARGS[@]}" &

python3 "$SCRIPT_DIR/multimodal_smoke_client.py" \
  --mode epd \
  --model-name "$MODEL_NAME" \
  --namespace "$NAMESPACE" \
  --encode-component "$ENCODE_COMPONENT" \
  --prefill-component "$PREFILL_COMPONENT" \
  --decode-component "$DECODE_COMPONENT" &

wait_any_exit
