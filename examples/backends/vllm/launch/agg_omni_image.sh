#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="black-forest-labs/FLUX.1-dev"

# Torch XPU requires a newer oneCCL symbol than some environments expose by
# default (e.g. 2021.15). Prefer oneCCL 2021.17 when available.
if [[ -f /opt/intel/oneapi/ccl/2021.17/lib/libccl.so.1 ]]; then
  export LD_LIBRARY_PATH="/opt/intel/oneapi/ccl/2021.17/lib:${LD_LIBRARY_PATH:-}"
fi

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching vLLM-Omni Image Generation (1 GPU)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
curl -s -X POST http://localhost:${HTTP_PORT}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "prompt": "A red apple on a white table",
    "size": "1024x1024",
    "nvext": {
      "num_inference_steps": 30,
      "guidance_scale": 7.5,
      "seed": 42
    }
  }' | jq
CURL
DYN_LORA_ENABLED=true python -m dynamo.frontend 2>&1 | tee  "frontend.log" &
FRONTEND_PID=$!

sleep 2

echo "Starting Omni worker..."
TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
DYN_LORA_ENABLED=true DYN_OMNI_ENFORCE_EAGER=true DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --output-modalities image \
  --media-output-fs-url file:///tmp/dynamo_media \
  --enforce-eager \
  --enable-lora \
    --tensor-parallel-size 2 \
  --max-lora-rank 64 \
    "${EXTRA_ARGS[@]}" 2>&1 | tee  "omni_worker.log" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
