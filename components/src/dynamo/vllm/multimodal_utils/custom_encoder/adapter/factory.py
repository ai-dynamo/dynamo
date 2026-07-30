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
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.model_config import (
    _is_multimodal_model,
    _model_architectures,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.qwen3_vl import (
    _QWEN3_VL_ARCHITECTURES,
    Qwen3VLNativeAdapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.backend.base import (
    VisionEncoderBackend,
)


def create_custom_encoder_adapter(
    backend: VisionEncoderBackend[Any, Any, Any],
    model_config: Any,
    engine_args: Any,
) -> CustomEncoderAdapter[Any]:
    """Create the adapter selected by the resolved downstream decoder."""

    if model_config is None:
        raise ValueError("CustomEncoder requires the resolved vLLM ModelConfig")
    architectures = _model_architectures(model_config)
    qwen3_vl_architectures = [
        architecture
        for architecture in architectures
        if architecture in _QWEN3_VL_ARCHITECTURES
    ]
    if qwen3_vl_architectures:
        return Qwen3VLNativeAdapter()

    if _is_multimodal_model(model_config):
        raise ValueError(
            "CustomEncoder does not support this multimodal decoder architecture: "
            f"{architectures}"
        )
    return LinearEmbedsAdapter(backend, model_config, engine_args)
