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
from dynamo.vllm.multimodal_utils.custom_encoder.backend.base import (
    VisionEncoderBackend,
)


def create_custom_encoder_adapter(
    backend: VisionEncoderBackend,
    model_config: Any,
    engine_args: Any,
    vllm_config: Any | None = None,
) -> CustomEncoderAdapter:
    """Create the adapter selected by the resolved downstream decoder.

    The first slice supports text-only decoders. ``vllm_config`` is accepted at
    this stable factory boundary for model-specific adapters added later.
    """

    del vllm_config
    return LinearEmbedsAdapter(backend, model_config, engine_args)
