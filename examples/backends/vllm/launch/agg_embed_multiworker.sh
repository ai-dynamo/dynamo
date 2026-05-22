#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-worker aggregated embedding serving.
#
# Spawns 1 frontend + 2 embedding workers, each on its own GPU and its own
# DYN_SYSTEM_PORT so per-worker metrics can be scraped independently.
#
# Used by the multi-worker embedding tests to verify:
#   1. Same-model load balancing — pass the same MODEL twice; the frontend
#      should weighted-randomly distribute requests across both workers.
#   2. Multi-model dispatch       — pass two different MODELs; the
#      name-keyed router (lib/llm/src/discovery/model_manager.rs) should
#      send each request only to the worker registered for that model.
#
# GPUs: 2
#
# Usage:
#   agg_embed_multiworker.sh MODEL1 MODEL2 [EXTRA_DYNAMO_VLLM_ARGS...]
#
# EXTRA args (after the two model positions) are forwarded verbatim to
# *both* dynamo.vllm worker processes. The current launch matches the
# single-worker ``agg_embed.sh`` script (``--runner pooling``,
# ``--dtype float32``, MEAN pooler config, ``--max-model-len 2048``,
# ``--no-enable-prefix-caching``).

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_vllm_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MODEL1 MODEL2 [extra dynamo.vllm args...]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  # Same-model load balance test:" >&2
    echo "  $0 Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Embedding-0.6B" >&2
    echo "" >&2
    echo "  # Multi-model dispatch test:" >&2
    echo "  $0 Qwen/Qwen3-Embedding-0.6B BAAI/bge-small-en-v1.5" >&2
    exit 2
fi

MODEL1="$1"
MODEL2="$2"
shift 2
EXTRA_ARGS=("$@")

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT1="${DYN_SYSTEM_PORT1:-8081}"
SYSTEM_PORT2="${DYN_SYSTEM_PORT2:-8082}"

print_launch_banner --no-curl "Launching Multi-Worker Embeddings (2 GPUs)" "${MODEL1} + ${MODEL2}" "$HTTP_PORT"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/embeddings \\
    -H 'Content-Type: application/json' \\
    -d '{"model": "${MODEL1}", "input": "Hello, world!"}'

  # Per-worker metrics:
  curl http://localhost:${SYSTEM_PORT1}/metrics  # worker on GPU 0 (${MODEL1})
  curl http://localhost:${SYSTEM_PORT2}/metrics  # worker on GPU 1 (${MODEL2})
CURL

# Frontend — same routing layer as the single-worker case; the name-keyed
# DashMap lookup in lib/llm/src/discovery/model_manager.rs handles both the
# same-model fan-out (weighted-random across matching workers) and the
# multi-model dispatch (model field selects the set of eligible workers).
python3 -m dynamo.frontend &

# Tunable: see agg_embed.sh for the rationale.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

# Common worker args. Mirrors agg_embed.sh — the embedding worker handler
# is the same; only the per-worker model + GPU + system port differ.
common_worker_args=(
    --embedding-worker
    --runner pooling
    --dtype float32
    --pooler-config '{"pooling_type": "MEAN", "use_activation": false}'
    --max-model-len "$MAX_MODEL_LEN"
    --no-enable-prefix-caching
    --trust-remote-code
)

# Each worker registers under its OWN Dynamo NAMESPACE.
#
# Why namespaces and not just unique components or endpoints: the frontend
# keys ``Model.worker_sets`` by ``(namespace, model_type)`` (see
# ``worker_set_key`` in ``lib/llm/src/discovery/watcher.rs``), and
# ``add_worker_set`` is an insert-overwrite on that key. Two workers
# sharing a namespace -- regardless of how their endpoint paths differ --
# both hash to the same ``ws_key``, and the second registration silently
# replaces the first ``WorkerSet`` (along with its push_router). Only the
# last-registered worker survives in the routing table; the other one
# stays alive but is orphaned from the frontend's
# ``select_worker_set_with`` selector. The original "one model per
# endpoint" symptom was a sibling consequence of the same collision --
# this resolves both by giving each worker its own namespace, so both
# ``WorkerSet``s coexist and ``select_worker_set_with`` does its
# weighted-random fan-out as designed.
#
# Endpoint format is ``namespace.component.endpoint`` (dots, not slashes).
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${SYSTEM_PORT1} \
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model "$MODEL1" \
    --endpoint embed-worker-1.vllm.generate \
    "${common_worker_args[@]}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${SYSTEM_PORT2} \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model "$MODEL2" \
    --endpoint embed-worker-2.vllm.generate \
    "${common_worker_args[@]}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
