# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for Dynamo backends."""

from enum import Enum

KV_HINT_TRANSFER_CAPABILITY_KEY = "kv_hint.transfer.v1"
KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY = "kv_hint_transfer_worker_type"
KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY = (
    "kv_hint_transfer_source_control_endpoints"
)
KV_HINT_PROTOCOL_VERSION = "0.1"
KV_SOURCE_LOCATIONS_ACTION_TYPE = "kv.source_locations"
KV_SOURCE_LOCATIONS_ACTION_VERSION = "1.0"


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
