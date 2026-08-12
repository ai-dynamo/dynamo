# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for Dynamo backends."""

from enum import Enum

KV_HINT_MIGRATE_CAPABILITY_KEY = "kv_hint.migrate.v1"
KV_HINT_MIGRATE_WORKER_TYPE_RUNTIME_KEY = "kv_hint_migrate_worker_type"
KV_HINT_MIGRATE_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY = (
    "kv_hint_migrate_source_control_endpoints"
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
