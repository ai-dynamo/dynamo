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
# GPUs: 2 -- the first two devices of the inherited
#           CUDA_VISIBLE_DEVICES, so a caller that assigns a
#           specific pair (the VRAM-aware test scheduler does) is
#           honoured rather than overridden.
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
# Fall through to ``DYN_SYSTEM_PORT`` (the single-worker convention used by
# sibling launch scripts like ``agg_embed.sh``) for worker 1 so callers
# that only set the non-numbered env var still drive the first worker's
# port. ``SYSTEM_PORT2`` stays numbered-only -- there's no single-worker
# equivalent for it.
SYSTEM_PORT1="${DYN_SYSTEM_PORT1:-${DYN_SYSTEM_PORT:-8081}}"
SYSTEM_PORT2="${DYN_SYSTEM_PORT2:-8082}"

# Pin each worker to one device of the set we were GIVEN, not to a hardcoded
# 0 and 1.
#
# CUDA_VISIBLE_DEVICES is absolute: a child that sets it to "0" gets physical
# GPU 0, whatever the parent's value was -- the parent's restriction is
# replaced, not compounded. So hardcoding 0/1 here is only correct while the
# caller happens to have handed us exactly those two devices. The VRAM-aware
# test scheduler (tests/utils/pytest_parallel_gpu.py) allocates a gpu_2 test a
# gang of any two devices on the node and exports them in
# CUDA_VISIBLE_DEVICES, so on a node with more than two GPUs it can hand us
# "2,3" -- and hardcoded 0/1 would then run both workers on devices the
# scheduler has reserved for other tests, silently double-booking their VRAM.
#
# Unset means "no restriction" (a manual run on an unrestricted host), which
# keeps the historical 0,1 behaviour. An explicitly EMPTY value is different --
# in CUDA it means "no devices at all" -- so it must fail the check below rather
# than fall back to two GPUs; hence ``-`` and not ``:-``.
IFS=',' read -r -a _VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES-0,1}"
if [ -z "${CUDA_VISIBLE_DEVICES-0,1}" ] || [ "${#_VISIBLE_GPUS[@]}" -lt 2 ]; then
    echo "ERROR: this script needs 2 GPUs;" \
         "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-<unset>}' exposes ${#_VISIBLE_GPUS[@]}." >&2
    exit 1
fi
WORKER1_GPU="${_VISIBLE_GPUS[0]}"
WORKER2_GPU="${_VISIBLE_GPUS[1]}"

print_launch_banner --no-curl "Launching Multi-Worker Embeddings (2 GPUs)" "${MODEL1} + ${MODEL2}" "$HTTP_PORT"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/embeddings \\
    -H 'Content-Type: application/json' \\
    -d '{"model": "${MODEL1}", "input": "Hello, world!"}'

  # Per-worker metrics:
  curl http://localhost:${SYSTEM_PORT1}/metrics  # worker on the 1st visible GPU (${MODEL1})
  curl http://localhost:${SYSTEM_PORT2}/metrics  # worker on the 2nd visible GPU (${MODEL2})
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
CUDA_VISIBLE_DEVICES="${WORKER1_GPU}" python3 -m dynamo.vllm \
    --model "$MODEL1" \
    --endpoint embed-worker-1.vllm.generate \
    "${common_worker_args[@]}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${SYSTEM_PORT2} \
CUDA_VISIBLE_DEVICES="${WORKER2_GPU}" python3 -m dynamo.vllm \
    --model "$MODEL2" \
    --endpoint embed-worker-2.vllm.generate \
    "${common_worker_args[@]}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
