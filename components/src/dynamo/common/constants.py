# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for Dynamo backends."""

from enum import Enum


class DisaggregationMode(Enum):
    """Disaggregation mode for LLM workers.

    Examples:
        >>> from dynamo.common.constants import DisaggregationMode
        >>> DisaggregationMode.PREFILL.value
        'prefill'
        >>> DisaggregationMode("agg") == DisaggregationMode.AGGREGATED
        True
    """

    AGGREGATED = "agg"
    PREFILL = "prefill"
    DECODE = "decode"


class EmbeddingTransferMode(Enum):
    """Embedding transfer mode for LLM workers.

    Examples:
        >>> from dynamo.common.constants import EmbeddingTransferMode
        >>> EmbeddingTransferMode.NIXL_WRITE.value
        'nixl-write'
        >>> EmbeddingTransferMode("local") == EmbeddingTransferMode.LOCAL
        True
    """

    LOCAL = "local"
    NIXL_WRITE = "nixl-write"
    NIXL_READ = "nixl-read"
