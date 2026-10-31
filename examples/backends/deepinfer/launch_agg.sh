#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated DeepSeek-V4.1-Flash on DeepInfer: the frontend and one dynamo.deepinfer worker
# (four DP ranks with EP4) in one container on a 4-GPU GB200 or GB300 node. No etcd or NATS:
# file-based discovery and the ZMQ event plane. Run it inside the DeepInfer image
# (examples/backends/deepinfer/Dockerfile) from a checkout of this repository.
# GPUs: 4
#
# The container needs RDMA devices for NCCL GIN (see README.md). Arguments are appended to the
# worker command line, so they override the defaults below.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# No gpu_utils.sh: DeepInfer sizes its KV cache on whole GPUs and has no VRAM budget flag.
source "$SCRIPT_DIR/../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL_PATH="${MODEL_PATH:-${DEEPINFER_HF_MODEL_DIR:-/shared-model-cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}"
NUM_GPUS="${NUM_GPUS:-4}"
FRONTEND_PYTHON="${FRONTEND_PYTHON:-/opt/dynamo-frontend/bin/python}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
EXTRA_ARGS=("$@")

# Check before installing the kill-process-group trap, so a failed check cannot kill the caller.
if [[ ! -r "$MODEL_PATH/config.json" ]]; then
    echo "No checkpoint at $MODEL_PATH; set MODEL_PATH" >&2
    exit 1
fi
# EP4 fetches the host-memory Engram tables over NCCL GIN, which needs RDMA devices. Only
# NICs attached to the pod create uverbs nodes; /sys/class/infiniband lists host NICs anyway.
if ! compgen -G "/dev/infiniband/uverbs*" > /dev/null; then
    echo "No RDMA devices in this container; DeepInfer EP4 needs NCCL GIN" >&2
    exit 1
fi

trap 'echo Cleaning up...; kill 0' EXIT

# Pod network settings from GKE GB200; values already in the environment win.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
# GKE's rdma-0..3 are isolated rails, so GIN across NICs fails. Keep every rank on one NIC.
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"  # one GIN communicator per rail (GKE multi-plane RDMA)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

# CPU-pinning shim (sitecustomize.py) for containers whose CPU set is smaller than the node.
SHIM_DIR="$SCRIPT_DIR/cpu_affinity"
export DEEPINFER_LOCAL_NGPUS="${DEEPINFER_LOCAL_NGPUS:-$NUM_GPUS}"

export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"
export DYN_FILE_KV="${DYN_FILE_KV:-$(mktemp -d)}"

print_launch_banner "Launching DeepInfer Aggregated Serving ($NUM_GPUS GPUs)" \
    "$SERVED_MODEL_NAME" "$HTTP_PORT" \
    "Checkpoint:  $MODEL_PATH" \
    "Discovery:   $DYN_DISCOVERY_BACKEND ($DYN_FILE_KV)"

# The frontend venv must not see the worker's DeepInfer and dynamo.deepinfer paths.
PYTHONPATH="" "$FRONTEND_PYTHON" -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --router-mode kv &

PYTHONPATH="$SHIM_DIR${PYTHONPATH:+:$PYTHONPATH}" \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    python -m dynamo.deepinfer \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --dp-size "$NUM_GPUS" \
    --ep-size "$NUM_GPUS" \
    --kv-cache-manager prefix \
    --dyn-reasoning-parser deepseek_v41 \
    --dyn-tool-call-parser deepseek_v41 \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
