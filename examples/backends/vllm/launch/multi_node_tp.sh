#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Multi-node TP deployment with dynamo.vllm
#
# Single script for both head and worker roles.
#
# Usage:
#   Head node:
#     bash multi_node_tp.sh --head --head-ip 10.0.0.1
#
#   Worker node:
#     bash multi_node_tp.sh --worker --head-ip 10.0.0.1
#
# Prerequisites:
#   - 8 GPUs per node
#   - Head: NATS and etcd running (on this node or reachable)
#   - Worker: torch.distributed connectivity to head node
#   - Worker: head node must be started first

set -e
trap 'echo "Cleaning up..."; kill 0' EXIT

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${TENSOR_PARALLEL_SIZE:-16}"
NNODES="${NNODES:-2}"
ROLE=""
HEAD_IP=""

usage() {
  echo "Usage: $0 (--head | --worker) --head-ip <IP>"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --head)   ROLE="head";   shift ;;
    --worker) ROLE="worker"; shift ;;
    --head-ip)
      HEAD_IP="$2"
      shift 2
      ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "${ROLE}" ]]    && { echo "Error: specify --head or --worker"; usage; }
[[ -z "${HEAD_IP}" ]] && { echo "Error: --head-ip is required"; usage; }

if [[ "${ROLE}" == "head" ]]; then

  echo "Starting Dynamo frontend..."
  python3 -m dynamo.frontend &

  echo "Starting dynamo.vllm head node (TP=${TP}, nnodes=${NNODES}, node-rank=0)..."
  python3 -m dynamo.vllm \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --nnodes "${NNODES}" \
    --node-rank 0 \
    --master-addr "${HEAD_IP}" \
    --enforce-eager &

  wait
else
  echo "Starting dynamo.vllm headless worker (TP=${TP}, nnodes=${NNODES}, node-rank=1)..."
  python3 -m dynamo.vllm \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --nnodes "${NNODES}" \
    --node-rank 1 \
    --master-addr "${HEAD_IP}" \
    --enforce-eager \
    --headless
fi
