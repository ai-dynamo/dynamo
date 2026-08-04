# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom encoder interfaces and Dynamo-owned runtime drivers."""

from dynamo.vllm.multimodal_utils.custom_encoder.adapter import (
    CustomEncoderAdapter,
    build_mixed_embeds,
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.async_encoder import AsyncVisionEncoder
from dynamo.vllm.multimodal_utils.custom_encoder.backend import (
    ItemT,
    Preprocessed,
    RawT,
    VisionEncoderBackend,
)

__all__ = [
    "AsyncVisionEncoder",
    "build_mixed_embeds",
    "CustomEncoderAdapter",
    "create_custom_encoder_adapter",
    "ItemT",
    "Preprocessed",
    "RawT",
    "VisionEncoderBackend",
]
