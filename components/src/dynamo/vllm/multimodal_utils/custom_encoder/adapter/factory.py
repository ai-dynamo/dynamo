# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a custom encoder adapter for the resolved decoder."""

from typing import Any

from dynamo.vllm.multimodal_utils.custom_encoder.adapter.base import (
    CustomEncoderAdapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.linear import (
    LinearEmbedsAdapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.qwen3_vl import (
    Qwen3VLNativeAdapter,
    _is_native_qwen_vlm,
)
from dynamo.vllm.multimodal_utils.custom_encoder.backend.base import (
    VisionEncoderBackend,
)
from dynamo.vllm.multimodal_utils.model_config import (
    _is_multimodal_model,
    _model_architectures,
)


def create_custom_encoder_adapter(
    backend: VisionEncoderBackend[Any, Any, Any],
    model_config: Any,
    engine_args: Any,
) -> CustomEncoderAdapter[Any]:
    """Create the adapter selected by the resolved downstream decoder."""

    if model_config is None:
        raise ValueError("CustomEncoder requires the resolved vLLM ModelConfig")
    if _is_native_qwen_vlm(model_config):
        return Qwen3VLNativeAdapter(model_config)

    if _is_multimodal_model(model_config):
        raise ValueError(
            "CustomEncoder does not support this multimodal decoder architecture: "
            f"{_model_architectures(model_config)}"
        )
    return LinearEmbedsAdapter(backend, model_config, engine_args)
