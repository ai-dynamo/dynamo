#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# NUMA-pinned vLLM instance launcher for the qwen3-tp2-experiments bundle.
# Single entrypoint used by all three scenarios — TP, GPU list, NUMA node,
# port, optional disagg role, and the feature set rendered by kvbmctl are all
# parameters.
#
# Render shape mirrors disagg-bringup/launch-prefill.sh:
#   kvbmctl config vllm --hub http://127.0.0.1:1337 --features <csv> \
#       [--role prefill|decode] --kv-connector-module-path kvbm.v2.vllm.connector
# emits the three flags vLLM needs: --block-size --max-model-len --kv-transfer-config.
#
# Usage:
#   PORT=8000 GPUS=0,1 NUMA=0 TP=2 FEATURES=indexer,p2p ROLE= \
#     bash launch-instance.sh
#   PORT=8000 GPUS=2   NUMA=1 TP=1 FEATURES=disagg      ROLE=prefill \
#     bash launch-instance.sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env.sh"
. "$SCRIPT_DIR/numa-lib.sh"
. "$KVBM_REPO/.claude/skills/kvbm-hub-bringup/hub-lib.sh"

PORT=${PORT:?"PORT required"}
GPUS=${GPUS:?"GPUS required (e.g. 0,1)"}
NUMA=${NUMA:?"NUMA required (e.g. 0)"}
TP=${TP:?"TP required (1 or 2)"}
FEATURES=${FEATURES:?"FEATURES required (e.g. indexer,p2p or disagg)"}
ROLE=${ROLE:-}                                    # optional: prefill | decode | empty
MEMUTIL=${MEMUTIL:-}                              # optional override
if [ -z "$MEMUTIL" ]; then
  if [ "$TP" -eq 2 ]; then MEMUTIL="$KVBM_TP2_GPU_MEMUTIL"; else MEMUTIL="$KVBM_TP1_GPU_MEMUTIL"; fi
fi
HUB_URL=${KVBM_HUB_URL:-http://127.0.0.1:$KVBM_HUB_DISCOVERY_PORT}

# Render --block-size / --max-model-len / --kv-transfer-config from the live hub.
ROLE_ARGS=()
[ -n "$ROLE" ] && ROLE_ARGS=( --role "$ROLE" )
KV_RENDERED=$(kvbm_hub_render_vllm "$KVBM_KVBMCTL_BIN" "$HUB_URL" "$FEATURES" \
    "${ROLE_ARGS[@]}" \
    --kv-connector-module-path "$KVBM_CONNECTOR_MODULE_PATH") \
  || { echo "kvbmctl render failed (is the hub up at $HUB_URL with --features $FEATURES?)" >&2; exit 1; }
eval "KV_ARGS=( $KV_RENDERED )"

export CUDA_VISIBLE_DEVICES="$GPUS"
export DYN_KVBM_CPU_CACHE_GB="$KVBM_CPU_CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export KVBM_SKIP_VLLM_VERSION_CHECK=${KVBM_SKIP_VLLM_VERSION_CHECK:-1}

# NIXL .libs discovery (mirrors disagg-bringup/launch-prefill.sh:48-58).
NIXL_LIBS=${KVBM_NIXL_LIBS:-}
if [ -z "$NIXL_LIBS" ]; then
    for cand in "$KVBM_VENV"/lib/python*/site-packages/.nixl_cu12.mesonpy.libs \
                "$KVBM_VENV"/lib/python*/site-packages/.nixl_cu13.mesonpy.libs; do
        [ -d "$cand" ] && NIXL_LIBS="$cand" && break
    done
fi
if [ -n "$NIXL_LIBS" ]; then
    export LD_LIBRARY_PATH="$NIXL_LIBS:$NIXL_LIBS/plugins:${LD_LIBRARY_PATH:-}"
    export NIXL_PLUGIN_DIR="$NIXL_LIBS/plugins"
fi

# Pin the entire vLLM process tree (api_server + EngineCore subproc + the
# TP=2 NCCL workers it forks) to the right NUMA node. CUDA_VISIBLE_DEVICES
# already constrains the GPUs; numactl adds CPU + memory locality.
NUMA_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
  NUMA_PREFIX=( numactl --cpunodebind="$NUMA" --membind="$NUMA" )
else
  echo "[launch-instance] WARN: numactl not found — NUMA pinning disabled" >&2
fi

# Optional YaRN rope scaling — required when KVBM_HUB_MAX_SEQ_LEN exceeds the
# model's native max_position_embeddings. vLLM 0.20.x doesn't expose --rope-scaling
# directly; we use --hf-overrides to inject both the extended position cap and the
# rope_scaling block. Defaults set for Qwen3-32B (native 40960) → 131072 via YaRN
# factor 3.2. Set KVBM_HF_OVERRIDES="" to disable (model already covers target).
HF_OVERRIDE_ARGS=()
KVBM_HF_OVERRIDES=${KVBM_HF_OVERRIDES-'{"max_position_embeddings":131072,"rope_scaling":{"rope_type":"yarn","factor":3.2,"original_max_position_embeddings":40960}}'}
if [ -n "$KVBM_HF_OVERRIDES" ]; then
  HF_OVERRIDE_ARGS=( --hf-overrides "$KVBM_HF_OVERRIDES" )
fi

echo "[launch-instance] port=$PORT gpus=$GPUS numa=$NUMA tp=$TP features=$FEATURES role=${ROLE:-<none>} memutil=$MEMUTIL"
echo "[launch-instance] kv args: ${KV_ARGS[*]}"
echo "[launch-instance] hf overrides: ${HF_OVERRIDE_ARGS[*]:-<none>}"
echo "[launch-instance] numa prefix: ${NUMA_PREFIX[*]:-<none>}"

exec "${NUMA_PREFIX[@]}" \
  "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
    --model "$KVBM_MODEL" \
    --served-model-name "$KVBM_MODEL" \
    --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEMUTIL" \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port "$PORT" \
    "${HF_OVERRIDE_ARGS[@]}" \
    "${KV_ARGS[@]}"
