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
from dynamo.vllm.multimodal_utils.custom_encoder.handoff import (
    HandoffReplayGuard,
    LinearEmbedsHandoffV1,
    receive_linear_embeds_prompt,
    stage_linear_embeds_prompt,
)
from dynamo.vllm.multimodal_utils.custom_encoder.loader import (
    extract_custom_encoder_image_urls,
    load_custom_encoder,
)

__all__ = [
    "AsyncVisionEncoder",
    "build_mixed_embeds",
    "CustomEncoderAdapter",
    "create_custom_encoder_adapter",
    "extract_custom_encoder_image_urls",
    "ItemT",
    "HandoffReplayGuard",
    "LinearEmbedsHandoffV1",
    "load_custom_encoder",
    "Preprocessed",
    "RawT",
    "receive_linear_embeds_prompt",
    "stage_linear_embeds_prompt",
    "VisionEncoderBackend",
]
