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

# Launch the frontend and one vLLM decode endpoint for cross-cluster
# disaggregated serving. Run disagg_multi_cluster_prefill.sh on the prefill
# cluster with matching configuration.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
DECODE_GPUS="${DECODE_GPUS:-0}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}}"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

pick_worker_module dynamo.vllm dynamo.vllm.unified_main "$@"
set -- "${REMAINING_ARGS[@]}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ETCD_ENDPOINTS=<url> NATS_SERVER=<url> \
       VLLM_NIXL_SIDE_CHANNEL_HOST=<reachable-ip> \
       ./disagg_multi_cluster_decode.sh [--unified] [vLLM arguments]

The frontend uses strict disaggregated routing: requests fail until a prefill
endpoint is registered. Set DECODE_GPUS and DECODE_TP_SIZE for a tensor-parallel
decode endpoint. DYN_TCP_RPC_HOST defaults to VLLM_NIXL_SIDE_CHANNEL_HOST.
EOF
    exit 0
fi

: "${ETCD_ENDPOINTS:?ETCD_ENDPOINTS must be reachable from both clusters}"
: "${NATS_SERVER:?NATS_SERVER must be reachable from both clusters}"
: "${VLLM_NIXL_SIDE_CHANNEL_HOST:?Set this to the decode node address}"

export DYN_NAMESPACE="${DYN_NAMESPACE:-dynamo-cross-cluster}"
export DYN_DISCOVERY_BACKEND="etcd"
export DYN_REQUEST_PLANE="tcp"
export DYN_EVENT_PLANE="nats"
export DYN_TCP_RPC_HOST="${DYN_TCP_RPC_HOST:-$VLLM_NIXL_SIDE_CHANNEL_HOST}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export UCX_TLS="${UCX_TLS:-tcp,cuda_copy,self}"
export UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-tcp}"
export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-1024}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
DECODE_TCP_PORT="${DECODE_TCP_PORT:-8791}"
DECODE_SYSTEM_PORT="${DECODE_SYSTEM_PORT:-8081}"
VLLM_NIXL_SIDE_CHANNEL_PORT="${VLLM_NIXL_SIDE_CHANNEL_PORT:-20097}"

worker_args=(
    --model "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --tensor-parallel-size "$DECODE_TP_SIZE"
    --block-size "$BLOCK_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --disaggregation-mode decode
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
)
if [[ -n "$MAX_MODEL_LEN" ]]; then
    worker_args+=(--max-model-len "$MAX_MODEL_LEN")
fi

child_pids=()
cleanup() {
    trap - EXIT INT TERM
    echo "Stopping cross-cluster decode and frontend"
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

# The frontend only exposes HTTP in this example. Let its internal request-plane
# server bind an OS-assigned port; the decode and prefill workers retain fixed
# ports because traffic crosses process and cluster boundaries to reach them.
python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --router-mode kv \
    --router-reset-states \
    --enforce-disagg &
child_pids+=("$!")

DYN_TCP_RPC_PORT="$DECODE_TCP_PORT" \
DYN_SYSTEM_PORT="$DECODE_SYSTEM_PORT" \
CUDA_VISIBLE_DEVICES="$DECODE_GPUS" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$VLLM_NIXL_SIDE_CHANNEL_PORT" \
python3 -m "$WORKER_MODULE" "${worker_args[@]}" "$@" &
child_pids+=("$!")

# The shared wait_any_exit helper signals the entire process group, which also
# terminates an enclosing srun. Keep cleanup scoped to this launcher's children.
wait_rc=0
wait -n || wait_rc=$?
exit "$wait_rc"
