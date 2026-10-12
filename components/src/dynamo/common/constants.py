# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for Dynamo backends."""

from enum import Enum

# TODO: Rename these constants and wire values with the matching KVCC names.
KV_HINT_TRANSFER_CAPABILITY_KEY = "router_hint"
KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY = "router_hint_worker_type"
KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY = (
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
