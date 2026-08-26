# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author-facing custom encoder backend contract."""

from dynamo.vllm.multimodal_utils.custom_encoder.backend.base import (
    ArtifactT,
    ItemT,
    Preprocessed,
    RawT,
    VisionEncoderBackend,
)
from dynamo.vllm.multimodal_utils.custom_encoder.backend.loader import (
    resolve_vision_encoder_backend_class,
)

__all__ = [
    "ArtifactT",
    "ItemT",
    "Preprocessed",
    "RawT",
    "resolve_vision_encoder_backend_class",
    "VisionEncoderBackend",
]
