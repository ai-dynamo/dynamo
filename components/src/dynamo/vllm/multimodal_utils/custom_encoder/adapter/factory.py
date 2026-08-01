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
) -> CustomEncoderAdapter:
    """Create the adapter selected by the resolved downstream decoder."""

    return LinearEmbedsAdapter(backend, model_config, engine_args)
