# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for Dynamo backends."""

from enum import Enum

ROUTER_HINT_RUNTIME_CAPABILITY_KEY = "router_hint"
ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY = "router_hint_worker_type"
ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY = (
    "router_hint_source_control_endpoints"
)


class DisaggregationMode(Enum):
    """Disaggregation mode for LLM workers."""

    AGGREGATED = "agg"
    PREFILL = "prefill"
    DECODE = "decode"
    ENCODE = "encode"


class EmbeddingTransferMode(Enum):
    """Embedding transfer mode for LLM workers."""

    LOCAL = "local"
    NIXL_WRITE = "nixl-write"
    NIXL_READ = "nixl-read"
