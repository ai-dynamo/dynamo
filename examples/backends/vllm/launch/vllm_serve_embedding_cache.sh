#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
CAPACITY_GB=10
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --multimodal-embedding-cache-capacity-gb)
            CAPACITY_GB="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Need vLLM main or v0.17+
EC_ARGS=()
if [[ "$CAPACITY_GB" != "0" ]]; then
    EC_ARGS=(--ec-transfer-config "{
        \"ec_role\": \"ec_both\",
        \"ec_connector\": \"DynamoMultimodalEmbeddingCacheConnector\",
        \"ec_connector_module_path\": \"dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector\",
        \"ec_connector_extra_config\": {\"multimodal_embedding_cache_capacity_gb\": $CAPACITY_GB}
    }")
fi

GPU_MEM_UTIL=".9"
KV_BYTES="${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:-}"
if [[ -n "$KV_BYTES" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization 0.01"
else
    GPU_MEM_ARGS="--gpu-memory-utilization $GPU_MEM_UTIL"
fi

CUDA_VISIBLE_DEVICES=2 \
vllm serve "$MODEL" \
    --enable-log-requests \
    --max-model-len 16384 \
    $GPU_MEM_ARGS \
    "${EC_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
