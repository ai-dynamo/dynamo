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
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.qwen2_vl import (
    _QWEN2_ARCHITECTURES,
    _Qwen2VLNativeAdapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.backend.base import (
    VisionEncoderBackend,
)


def create_custom_encoder_adapter(
    backend: VisionEncoderBackend[Any, Any, Any],
    model_config: Any,
    engine_args: Any,
    vllm_config: Any | None = None,
) -> CustomEncoderAdapter[Any]:
    """Create the adapter selected by the resolved downstream decoder."""

    if model_config is None:
        raise ValueError("CustomEncoder requires the resolved vLLM ModelConfig")
    architectures = _model_architectures(model_config)
    qwen_architectures = [
        architecture
        for architecture in architectures
        if architecture in _QWEN2_ARCHITECTURES
    ]
    if qwen_architectures:
        if len(qwen_architectures) != 1:
            raise ValueError(
                "Qwen CustomEncoder requires exactly one supported architecture, "
                f"got {architectures}"
            )
        return _Qwen2VLNativeAdapter(model_config, engine_args, vllm_config)

    if _is_multimodal_model(model_config):
        raise ValueError(
            "CustomEncoder does not support this multimodal decoder architecture: "
            f"{architectures}"
        )
    return LinearEmbedsAdapter(backend, model_config, engine_args)
