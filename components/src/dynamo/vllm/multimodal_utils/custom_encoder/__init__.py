# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom encoder interfaces and Dynamo-owned runtime drivers."""

from dynamo.vllm.multimodal_utils.custom_encoder.adapter import (
    CustomEncoderAdapter,
    Qwen3VLImageEncoding,
    build_mixed_embeds,
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.artifact_transfer import (
    CUSTOM_ENCODER_ARTIFACTS_KEY,
    ReceivedCustomEncoderArtifacts,
    create_custom_encoder_artifact_receiver,
    create_custom_encoder_artifact_sender,
    export_custom_encoder_artifacts,
    has_transferred_custom_encoder_artifacts,
    receive_custom_encoder_artifacts,
)
from dynamo.vllm.multimodal_utils.custom_encoder.async_encoder import AsyncVisionEncoder
from dynamo.vllm.multimodal_utils.custom_encoder.backend import (
    ArtifactT,
    ItemT,
    Preprocessed,
    RawT,
    VisionEncoderBackend,
)

__all__ = [
    "AsyncVisionEncoder",
    "ArtifactT",
    "build_mixed_embeds",
    "CUSTOM_ENCODER_ARTIFACTS_KEY",
    "CustomEncoderAdapter",
    "create_custom_encoder_artifact_receiver",
    "create_custom_encoder_artifact_sender",
    "create_custom_encoder_adapter",
    "export_custom_encoder_artifacts",
    "has_transferred_custom_encoder_artifacts",
    "ItemT",
    "Preprocessed",
    "Qwen3VLImageEncoding",
    "RawT",
    "ReceivedCustomEncoderArtifacts",
    "receive_custom_encoder_artifacts",
    "VisionEncoderBackend",
]
