#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Performance-only Qwen2.5 benchmark topology: run the complete 3B-VL vision
# tower, truncate its 2048-wide output to 1536 columns, and feed the result to
# the Qwen2.5-1.5B decoder. The truncation is not a trained projection and this
# workflow makes no output-quality or model-parity claim.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DYN_MODEL="${DYN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export DYN_QWEN2_VL_ENCODER_MODEL="${DYN_QWEN2_VL_ENCODER_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE="${DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE:-1536}"
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY="${DYN_QWEN2_VL_PREPROCESS_CONCURRENCY:-64}"
export DYN_QWEN2_VL_MAX_BATCH_COST="${DYN_QWEN2_VL_MAX_BATCH_COST:-64}"
export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS="${DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS:-1,2,4,8,16,32,64}"
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES="${DYN_QWEN2_VL_GRAPH_IMAGE_SIZES:-500x500}"
export DYN_MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-2048}"
export DYN_MAX_NUM_SEQS="${DYN_MAX_NUM_SEQS:-64}"
export DYN_VLLM_GPU_MEMORY_UTILIZATION="${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.4}"

echo >&2 "WARNING: performance-only 2048-to-1536 vision-output truncation; no quality claim."
exec "$SCRIPT_DIR/agg_qwen2_vl.sh" --max-num-seqs "$DYN_MAX_NUM_SEQS" "$@"
