# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder resource construction for the remote user ensemble."""

from __future__ import annotations

from typing import Any

from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
)
from examples.custom_backend.user_ensemble.stages import EncoderStage


def build_encoder_stage(
    model: str,
    backend_type: type[VisionEncoderBackend[Any, Any, Any]],
    *,
    name: str,
) -> tuple[EncoderStage, AsyncVisionEncoder[Any, Any, Any]]:
    """Construct and load the independently hosted encoder stage."""

    backend = backend_type()
    image_token_id = getattr(backend, "image_token_id", None)
    if not isinstance(image_token_id, int) or isinstance(image_token_id, bool):
        raise ValueError("external encoder requires an integer image_token_id")
    encoder = AsyncVisionEncoder(backend, name=name)
    try:
        encoder.load(model)
    except BaseException:
        encoder.shutdown()
        raise
    return EncoderStage(encoder, image_token_id), encoder
