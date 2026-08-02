# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for vLLM backend.

DisaggregationMode is defined in dynamo.common.constants and re-exported here
so that existing imports from dynamo.vllm.constants continue to work.
"""

from enum import Enum

from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode


class CustomEncoderRoutingMode(str, Enum):
    """Owner of the custom-encoder invocation for an image request."""

    INLINE = "inline"
    FRONTEND = "frontend"


__all__ = [
    "DisaggregationMode",
    "EmbeddingTransferMode",
    "CustomEncoderRoutingMode",
]
