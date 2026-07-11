#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the full Qwen3-VL-2B checkpoint in vLLM while Dynamo supplies image
# embeddings from a second, in-process copy of the checkpoint's vision tower.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DYN_MODEL="${DYN_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
export DYN_ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.qwen3_vl_vision_encoder.Qwen3VLVisionEncoder}"
# Leave room for the separately loaded ViT/projector until automatic encoder
# reservation is available.
export DYN_VLLM_GPU_MEMORY_UTILIZATION="${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.7}"

exec "$SCRIPT_DIR/agg_custom.sh" "$@"
