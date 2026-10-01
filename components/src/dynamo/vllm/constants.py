# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for vLLM backend.

DisaggregationMode is defined in dynamo.common.constants and re-exported here
so that existing imports from dynamo.vllm.constants continue to work.
"""

from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode

# vLLM load formats served by the ModelExpress plugin.
MX_LOAD_FORMATS = frozenset({"modelexpress", "mx"})

__all__ = [
    "DisaggregationMode",
    "EmbeddingTransferMode",
    "MX_LOAD_FORMATS",
]
