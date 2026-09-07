# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime capability metadata for vLLM's native Generate API."""

import json

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, WorkerType

VLLM_GENERATE_CAPABILITY = "vllm_inference_v1_generate"
VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY = "vllm_enable_tower_connector_lora"


def publish_engine_generate_capability(
    runtime_config: ModelRuntimeConfig,
    model_input: ModelInput,
    model_type: ModelType,
    worker_type: WorkerType,
    tower_connector_lora_enabled: bool,
) -> bool:
    """Publish native Generate support and its MM-routing-relevant config."""
    if model_input != ModelInput.Tokens:
        return False
    if worker_type == WorkerType.Prefill:
        supported = model_type == ModelType.Prefill
    else:
        supported = worker_type in (WorkerType.Decode, WorkerType.Aggregated) and (
            model_type.supports_chat() or model_type == ModelType.Completions
        )
    if not supported:
        return False

    runtime_config.set_engine_specific(
        VLLM_GENERATE_CAPABILITY,
        json.dumps(True),
    )
    runtime_config.set_engine_specific(
        VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY,
        json.dumps(tower_connector_lora_enabled),
    )
    return True
