# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters from custom encoder artifacts to decoder prompts."""

from dynamo.vllm.multimodal_utils.custom_encoder.adapter.base import (
    CustomEncoderAdapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.factory import (
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.linear import (
    LinearEmbedsAdapter,
    LinearVisualPrompt,
    build_mixed_embeds,
    build_mixed_layout,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.qwen3_vl import (
    Qwen3VLImageEncoding,
)

__all__ = [
    "build_mixed_embeds",
    "build_mixed_layout",
    "CustomEncoderAdapter",
    "create_custom_encoder_adapter",
    "LinearEmbedsAdapter",
    "LinearVisualPrompt",
    "Qwen3VLImageEncoding",
]
