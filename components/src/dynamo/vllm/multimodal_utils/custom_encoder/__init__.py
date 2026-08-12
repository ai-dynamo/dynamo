# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom encoder interfaces and Dynamo-owned runtime drivers."""

from dynamo.vllm.multimodal_utils.custom_encoder.adapter import (
    CustomEncoderAdapter,
    LinearEmbedsAdapter,
    LinearVisualPrompt,
    Qwen3VLImageEncoding,
    build_mixed_embeds,
    build_mixed_layout,
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.async_encoder import AsyncVisionEncoder
from dynamo.vllm.multimodal_utils.custom_encoder.backend import (
    ArtifactT,
    ItemT,
    Preprocessed,
    RawT,
    VisionEncoderBackend,
)
from dynamo.vllm.multimodal_utils.custom_encoder.handoff import (
    HandoffReplayGuard,
    LinearVisualHandoffV1,
    receive_linear_visual_prompt,
    stage_linear_visual_prompt,
)
from dynamo.vllm.multimodal_utils.custom_encoder.loader import (
    extract_custom_encoder_image_urls,
    load_custom_encoder,
)

__all__ = [
    "AsyncVisionEncoder",
    "ArtifactT",
    "build_mixed_embeds",
    "build_mixed_layout",
    "CustomEncoderAdapter",
    "create_custom_encoder_adapter",
    "extract_custom_encoder_image_urls",
    "ItemT",
    "HandoffReplayGuard",
    "LinearVisualHandoffV1",
    "LinearEmbedsAdapter",
    "LinearVisualPrompt",
    "load_custom_encoder",
    "Preprocessed",
    "Qwen3VLImageEncoding",
    "RawT",
    "receive_linear_visual_prompt",
    "stage_linear_visual_prompt",
    "VisionEncoderBackend",
]
