#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the language tower from the Qwen2.5-VL-3B checkpoint in vLLM while Dynamo
# supplies image embeddings from an in-process copy of its vision tower.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DYN_MODEL="${DYN_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export DYN_ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.qwen2_vl_vision_encoder.Qwen2VLVisionEncoder}"
# Leave room for the separately loaded ViT/projector until automatic encoder
# reservation is available.
export DYN_VLLM_GPU_MEMORY_UTILIZATION="${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.7}"

# Keep prefix caching explicit: repeated identical prompt blocks can still reuse
# decoder KV even though the custom image spans are embeddings.
exec "$SCRIPT_DIR/agg_custom.sh" --enable-prefix-caching "$@"
