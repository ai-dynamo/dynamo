# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for vLLM backend.

DisaggregationMode is defined in dynamo.common.constants and re-exported here
so that existing imports from dynamo.vllm.constants continue to work.
"""

from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode

INTERNAL_FINAL_ONLY_EXTRA_ARG = "dynamo_internal_final_only"

__all__ = [
    "DisaggregationMode",
    "EmbeddingTransferMode",
    "INTERNAL_FINAL_ONLY_EXTRA_ARG",
]
