#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch one vLLM prefill endpoint for cross-cluster disaggregated serving.
# Run disagg_multi_cluster_decode.sh on the decode cluster with the same
# namespace, model, block size, and KV connector configuration.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
PREFILL_GPUS="${PREFILL_GPUS:-0}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
KV_EVENTS_PORT="${KV_EVENTS_PORT:-20081}"
KV_EVENTS_CONFIG="${KV_EVENTS_CONFIG:-{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT}\",\"enable_kv_cache_events\":true}}"
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}}"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

pick_worker_module dynamo.vllm dynamo.vllm.unified_main "$@"
set -- "${REMAINING_ARGS[@]}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ETCD_ENDPOINTS=<url> NATS_SERVER=<url> \
       VLLM_NIXL_SIDE_CHANNEL_HOST=<reachable-ip> \
       ./disagg_multi_cluster_prefill.sh [--unified] [vLLM arguments]

The worker uses etcd discovery, the TCP request plane, a local vLLM ZMQ event
publisher bridged to Dynamo's NATS event plane, and NIXL/UCX KV transfer. Set
PREFILL_GPUS and PREFILL_TP_SIZE for a tensor-parallel endpoint.
DYN_TCP_RPC_HOST defaults to VLLM_NIXL_SIDE_CHANNEL_HOST.
EOF
    exit 0
fi

: "${ETCD_ENDPOINTS:?ETCD_ENDPOINTS must be reachable from both clusters}"
: "${NATS_SERVER:?NATS_SERVER must be reachable from both clusters}"
: "${VLLM_NIXL_SIDE_CHANNEL_HOST:?Set this to the prefill address reachable from the decode cluster}"

export DYN_NAMESPACE="${DYN_NAMESPACE:-dynamo-cross-cluster}"
export DYN_DISCOVERY_BACKEND="etcd"
export DYN_REQUEST_PLANE="tcp"
export DYN_EVENT_PLANE="nats"
export DYN_TCP_RPC_HOST="${DYN_TCP_RPC_HOST:-$VLLM_NIXL_SIDE_CHANNEL_HOST}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# UCX needs an AM-capable network transport and a CUDA transport when the NIXL
# buffer remains in GPU memory. UCX_NET_DEVICES may be set by the operator when
# a multi-NIC host needs an explicit TCP interface.
export UCX_TLS="${UCX_TLS:-tcp,cuda_copy,self}"
export UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-tcp}"
export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-1024}"

VLLM_NIXL_SIDE_CHANNEL_PORT="${VLLM_NIXL_SIDE_CHANNEL_PORT:-20097}"
PREFILL_TCP_PORT="${PREFILL_TCP_PORT:-8792}"
PREFILL_SYSTEM_PORT="${PREFILL_SYSTEM_PORT:-8082}"

worker_args=(
    --model "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --tensor-parallel-size "$PREFILL_TP_SIZE"
    --block-size "$BLOCK_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --disaggregation-mode prefill
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
    --kv-events-config "$KV_EVENTS_CONFIG"
)
if [[ -n "$MAX_MODEL_LEN" ]]; then
    worker_args+=(--max-model-len "$MAX_MODEL_LEN")
fi

child_pids=()
cleanup() {
    trap - EXIT INT TERM
    echo "Stopping cross-cluster prefill"
    for pid in "${child_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${child_pids[@]}" 2>/dev/null || true
}
on_signal() {
    cleanup
    exit 0
}
trap cleanup EXIT
trap on_signal INT TERM

DYN_TCP_RPC_PORT="$PREFILL_TCP_PORT" \
DYN_SYSTEM_PORT="$PREFILL_SYSTEM_PORT" \
CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$VLLM_NIXL_SIDE_CHANNEL_PORT" \
python3 -m "$WORKER_MODULE" "${worker_args[@]}" "$@" &
child_pids+=("$!")

# The shared wait_any_exit helper signals the entire process group, which also
# terminates an enclosing srun. Keep cleanup scoped to this launcher's children.
wait_rc=0
wait -n || wait_rc=$?
exit "$wait_rc"
