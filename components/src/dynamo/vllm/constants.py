# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for vLLM backend.

This module defines enums and constants used throughout the vllm module.
"""

from enum import Enum


class DisaggregationMode(Enum):
    """Disaggregation mode for vLLM workers."""

    AGGREGATED = "prefill_and_decode"
    PREFILL = "prefill"
    DECODE = "decode"
