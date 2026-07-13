# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for vLLM backend.

DisaggregationMode is defined in dynamo.common.constants and re-exported here
so that existing imports from dynamo.vllm.constants continue to work.
"""

from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode

# vLLM exposes cache salt as an otherwise-untyped string in KV-event
# ``extra_keys``. The event decoder recognizes this prefix and restores the
# caller-visible namespace before hashing.
DYNAMO_CACHE_SALT_PREFIX = "dynamo-cache-salt:"

__all__ = [
    "DYNAMO_CACHE_SALT_PREFIX",
    "DisaggregationMode",
    "EmbeddingTransferMode",
]
