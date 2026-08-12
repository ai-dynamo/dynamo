#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Performance-only single-H100 topology:
#   frontend -> custom Encode worker (vision encoder + LinearEmbedsAdapter)
#            -> aggregated Qwen2.5-1.5B PD worker
#
# The Qwen2.5-VL-3B vision output is truncated from 2048 to 1536 columns. This
# is not a trained projection and therefore makes no quality/parity claim.

set -e
trap 'kill 0 2>/dev/null || true' EXIT

MODEL="${DYN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.qwen2_5_vl_benchmark_encoder.Qwen2_5VLBenchmarkEncoder}"
WORKER_GPU="${DYN_WORKER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${DYN_MAX_NUM_SEQS:-64}"
PD_GPU_MEMORY_UTILIZATION="${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JINJA_TEMPLATE="${DYN_CUSTOM_JINJA_TEMPLATE:-$SCRIPT_DIR/../templates/qwen_vl.jinja}"

export DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200
export DYN_QWEN2_VL_ENCODER_MODEL="${DYN_QWEN2_VL_ENCODER_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE="${DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE:-1536}"
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY="${DYN_QWEN2_VL_PREPROCESS_CONCURRENCY:-64}"
export DYN_QWEN2_VL_MAX_BATCH_PATCHES="${DYN_QWEN2_VL_MAX_BATCH_PATCHES:-82944}"
export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS="${DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS:-1,2,4,8,16,32,64}"
export DYN_QWEN2_VL_MAX_BATCH_ITEMS="${DYN_QWEN2_VL_MAX_BATCH_ITEMS:-64}"
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES="${DYN_QWEN2_VL_GRAPH_IMAGE_SIZES:-300x300,500x500}"

echo >&2 "WARNING: performance-only 2048-to-1536 vision-output truncation; no quality claim."
echo >&2 "BENCHMARK_GPU_MEMORY_UTILIZATION=$PD_GPU_MEMORY_UTILIZATION"
python -m dynamo.frontend --http-port "$HTTP_PORT" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
CUDA_VISIBLE_DEVICES="$WORKER_GPU" \
python -m dynamo.vllm \
  --model "$MODEL" \
  --enable-multimodal \
  --enable-prompt-embeds \
  --disaggregation-mode encode \
  --custom-encoder-routing-mode frontend \
  --custom-encoder-class "$ENCODER_CLASS" \
  --embedding-transfer-mode nixl-write \
  --max-model-len "$MAX_MODEL_LEN" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
CUDA_VISIBLE_DEVICES="$WORKER_GPU" \
python -m dynamo.vllm \
  --model "$MODEL" \
  --enable-multimodal \
  --enable-prompt-embeds \
  --disaggregation-mode agg \
  --custom-encoder-routing-mode frontend \
  --embedding-transfer-mode nixl-write \
  --custom-jinja-template "$JINJA_TEMPLATE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$PD_GPU_MEMORY_UTILIZATION" \
  --no-enable-prefix-caching &

wait -n
