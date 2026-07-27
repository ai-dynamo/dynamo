#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Explicitly unset PROMETHEUS_MULTIPROC_DIR to let Dynamo manage it internally
unset PROMETHEUS_MULTIPROC_DIR

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

export VLLM_TARGET_DEVICE=xpu

# Device affinity: Use auto-selected device via ZE_AFFINITY_MASK if set by test framework,
# otherwise default to device 0
ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export ZE_AFFINITY_MASK

MODEL="Qwen/Qwen3-0.6B"

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# Non-profiled XPU fallback: cap at 0.75 to leave headroom for the Level Zero
# driver/runtime, whose allocations vLLM's accounting doesn't track. The profiler
# path supplies its own --gpu-memory-utilization 0.01 via $GPU_MEM_ARGS.
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"

print_launch_banner "Launching Aggregated Serving (1 XPU) + LMCache MP" "$MODEL" "$HTTP_PORT"

# XPU currently uses LMCache MP's non-GPU transfer path. Current LMCache releases
# call this --transfer-mode; newer dev builds call it --supported-transfer-mode.
if lmcache server --help 2>&1 | grep -q -- '--supported-transfer-mode'; then
  LMCACHE_TRANSFER_MODE_ARG=(--supported-transfer-mode non_gpu)
else
  LMCACHE_TRANSFER_MODE_ARG=(--transfer-mode non_gpu)
fi

# Start the LMCache MP server (out-of-process cache engine).
lmcache server \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB:-16}" --eviction-policy LRU \
  "${LMCACHE_TRANSFER_MODE_ARG[@]}" \
  --port "$LMCACHE_PORT" --http-port "$LMCACHE_HTTP_PORT" &

# Wait until the server's HTTP admin endpoint is healthy before launching workers.
for _ in $(seq 1 60); do
  curl -sf "http://localhost:$LMCACHE_HTTP_PORT/healthcheck" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "http://localhost:$LMCACHE_HTTP_PORT/healthcheck" >/dev/null 2>&1 || {
  echo "lmcache server failed to become healthy on :$LMCACHE_HTTP_PORT" >&2
  exit 1
}

python -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  python -m dynamo.vllm --model "$MODEL" --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_CONCURRENT_SEQS" \
  --block-size "${BLOCK_SIZE:-64}" \
  ${GPU_MEM_ARGS:---gpu-memory-utilization 0.75} \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT}}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit