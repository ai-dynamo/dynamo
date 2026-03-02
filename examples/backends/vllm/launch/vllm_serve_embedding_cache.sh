#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


CAPACITY_GB=10
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --multimodal-embedding-cache-capacity-gb)
            CAPACITY_GB="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

EC_ARGS=()
if [[ "$CAPACITY_GB" != "0" ]]; then
    EC_ARGS=(--ec-transfer-config "{
        \"ec_role\": \"ec_both\",
        \"ec_connector\": \"DynamoMultimodalEmbeddingCacheConnector\",
        \"ec_connector_module_path\": \"dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector\",
        \"ec_connector_extra_config\": {\"multimodal_embedding_cache_capacity_gb\": $CAPACITY_GB}
    }")
fi

CUDA_VISIBLE_DEVICES=2 \
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --enable-log-requests \
    --max-model-len 16384 \
    --gpu-memory-utilization .9 \
    "${EC_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"