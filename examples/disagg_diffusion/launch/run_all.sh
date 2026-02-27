#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated Diffusion POC — all-in-one launcher
#
# Starts three stage workers (Encoder, Denoiser, VAE) as background
# processes, waits for them to be ready, then runs the orchestrator.
#
# Requirements:
#   - 3 GPUs (or 1 large GPU >48 GB — set SINGLE_GPU=1)
#   - diffusers, transformers, torch, dynamo runtime, uvloop
#
# Usage:
#   bash launch/run_all.sh [MODEL] [PROMPT]
#
# Examples:
#   bash launch/run_all.sh
#   bash launch/run_all.sh black-forest-labs/FLUX.1-schnell "A sunset over mountains"
#   SINGLE_GPU=1 bash launch/run_all.sh   # all stages on GPU 0

set -euo pipefail

export MODEL_PATH="${1:-black-forest-labs/FLUX.1-schnell}"
export PROMPT="${2:-A photo of a cat sitting on a windowsill}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKERS_DIR="${SCRIPT_DIR}/phase1_workers"
ORCH_DIR="${SCRIPT_DIR}/phase2_orchestrator"

echo "============================================"
echo "  Disaggregated Diffusion POC"
echo "  Model:  ${MODEL_PATH}"
echo "  Prompt: ${PROMPT}"
echo "============================================"
echo ""

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down workers …"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT

if [[ "${SINGLE_GPU:-0}" == "1" ]]; then
    GPU_ENC=0; GPU_DEN=0; GPU_VAE=0
else
    GPU_ENC=0; GPU_DEN=1; GPU_VAE=2
fi

# --- Start workers --------------------------------------------------------

echo "[1/3] Starting Encoder Worker (GPU ${GPU_ENC}) …"
CUDA_VISIBLE_DEVICES=${GPU_ENC} python "${WORKERS_DIR}/encoder_worker.py" \
    2>&1 | sed 's/^/  [encoder]  /' &
PIDS+=($!)

echo "[2/3] Starting Denoiser Worker (GPU ${GPU_DEN}) …"
CUDA_VISIBLE_DEVICES=${GPU_DEN} python "${WORKERS_DIR}/denoiser_worker.py" \
    2>&1 | sed 's/^/  [denoiser] /' &
PIDS+=($!)

echo "[3/3] Starting VAE Worker (GPU ${GPU_VAE}) …"
CUDA_VISIBLE_DEVICES=${GPU_VAE} python "${WORKERS_DIR}/vae_worker.py" \
    2>&1 | sed 's/^/  [vae]      /' &
PIDS+=($!)

# Wait for model loading.  In production use Dynamo health checks.
WAIT_SECS="${WAIT_SECS:-90}"
echo ""
echo "Waiting ${WAIT_SECS}s for workers to load models …"
sleep "${WAIT_SECS}"

# --- Run orchestrator -----------------------------------------------------

echo ""
echo "Running orchestrator …"
echo ""

export OUTPUT="/tmp/disagg_output.png"
python "${ORCH_DIR}/run_disagg.py"
