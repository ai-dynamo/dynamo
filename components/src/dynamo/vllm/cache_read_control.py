# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advertise the Python vLLM adapter's typed cache-read control."""

import json
from typing import TYPE_CHECKING

from dynamo.llm import ModelInput, ModelType, WorkerType

if TYPE_CHECKING:
    from dynamo.llm import ModelRuntimeConfig

SUPPORTS_PREFIX_CACHE_READ_CONTROL_KEY = "supports_prefix_cache_read_control"


def publish_prefix_cache_read_control_capability(
    runtime_config: "ModelRuntimeConfig",
    model_input: ModelInput,
    model_type: ModelType,
    worker_type: WorkerType,
) -> bool:
    """Declare honoring the frontend's decision on supported token adapters.

    This capability does not define request normalization or imply support on
    native Generate, pooling, text-input, or disaggregated execution paths.
    """
    if (
        model_input != ModelInput.Tokens
        or worker_type != WorkerType.Aggregated
        or not (model_type.supports_chat() or model_type == ModelType.Completions)
    ):
        return False
    runtime_config.set_engine_specific(
        SUPPORTS_PREFIX_CACHE_READ_CONTROL_KEY, json.dumps(True)
    )
    return True
