# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declare the request-override semantics of the Python vLLM adapter."""

import json
from typing import TYPE_CHECKING

from dynamo.llm import ModelInput, ModelType, WorkerType

if TYPE_CHECKING:
    from dynamo.llm import ModelRuntimeConfig

PREFIX_CACHE_READ_POLICY_KEY = "prefix_cache_read_policy"


def publish_prefix_cache_read_policy(
    runtime_config: "ModelRuntimeConfig",
    model_input: ModelInput,
    model_type: ModelType,
    worker_type: WorkerType,
) -> bool:
    """Advertise only the aggregated token adapter whose request policy is known.

    This contract describes build_sampling_params request overrides. It does not
    infer model defaults, native Generate, pooling, or remote-prefill behavior.
    """
    if (
        model_input != ModelInput.Tokens
        or worker_type != WorkerType.Aggregated
        or not (model_type.supports_chat() or model_type == ModelType.Completions)
    ):
        return False
    runtime_config.set_engine_specific(
        PREFIX_CACHE_READ_POLICY_KEY, json.dumps("vllm_python_v1")
    )
    return True
