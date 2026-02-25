# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for Dynamo backends."""

from enum import Enum


class DisaggregationMode(Enum):
    """Disaggregation mode for LLM workers."""

    AGGREGATED = "agg"
    PREFILL = "prefill"
    DECODE = "decode"
    # AFD (Attention-FFN Disaggregation) modes - decode phase only
    ATTENTION = "attention"  # Attention worker (stateful, KV-cache dominated)
    FFN = "ffn"  # FFN worker (stateless, compute-intensive)
