# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo KVBM v2 Framework Integrations

This package re-exports the Rust v2 bindings and provides Python integration utilities.
"""

# Re-export all v2 bindings from Rust
# Note: kvbm._core is a Rust extension module, so we import v2 as a submodule attribute
try:
    from kvbm._core import v2 as _v2

    is_available = _v2.is_available
    KvbmRuntime = _v2.KvbmRuntime
    KvbmVllmConfig = _v2.KvbmVllmConfig
    ConnectorLeader = _v2.ConnectorLeader
    ConnectorWorker = _v2.ConnectorWorker
    KvbmRequest = _v2.KvbmRequest
    SchedulerOutput = _v2.SchedulerOutput
    Tensor = _v2.Tensor

    _V2_CORE_AVAILABLE = True
except ImportError:
    # Provide stubs when the v2 feature is not compiled at all.
    def is_available() -> bool:
        return False

    from kvbm._feature_stubs import _make_feature_stub

    KvbmRuntime = _make_feature_stub("KvbmRuntime", "v2")
    KvbmVllmConfig = _make_feature_stub("KvbmVllmConfig", "v2")
    ConnectorLeader = _make_feature_stub("ConnectorLeader", "v2")
    ConnectorWorker = _make_feature_stub("ConnectorWorker", "v2")
    KvbmRequest = _make_feature_stub("KvbmRequest", "v2")
    SchedulerOutput = _make_feature_stub("SchedulerOutput", "v2")
    Tensor = _make_feature_stub("Tensor", "v2")
    _V2_CORE_AVAILABLE = False

# Rust scheduler symbols are exported separately because `pub mod scheduler`
# is currently commented out in lib/bindings/kvbm/src/v2/mod.rs (TODO:
# scheduler integration types not yet ported into the decomposed kvbm-*
# crates). When that port lands these will become non-None automatically;
# until then the absence is non-fatal — DynamoScheduler degrades to a vLLM
# passthrough and KV transfer offload still routes through ConnectorLeader/
# ConnectorWorker above.
try:
    from kvbm._core import v2 as _v2_for_scheduler  # noqa: F401

    RustScheduler = _v2_for_scheduler.RustScheduler
    SchedulerConfig = _v2_for_scheduler.SchedulerConfig
    RequestStatus = _v2_for_scheduler.RequestStatus
    _V2_SCHEDULER_AVAILABLE = True
except (ImportError, AttributeError):
    RustScheduler = None
    SchedulerConfig = None
    RequestStatus = None
    _V2_SCHEDULER_AVAILABLE = False

__all__ = [
    "is_available",
    "KvbmRuntime",
    "KvbmVllmConfig",
    "ConnectorLeader",
    "ConnectorWorker",
    "KvbmRequest",
    "SchedulerOutput",
    "Tensor",
    "RustScheduler",
    "SchedulerConfig",
    "RequestStatus",
    "_V2_CORE_AVAILABLE",
    "_V2_SCHEDULER_AVAILABLE",
]
