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
    build_mixed_embeds,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.qwen3_vl import (
    Qwen3VLImageEncoding,
)

__all__ = [
    "build_mixed_embeds",
    "CustomEncoderAdapter",
    "create_custom_encoder_adapter",
    "Qwen3VLImageEncoding",
]
