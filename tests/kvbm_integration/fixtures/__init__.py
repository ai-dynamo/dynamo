# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-layer integration test fixtures for KVBM (deps / server / eval).

See tests/kvbm_integration/README.md for the layered architecture and the
KVBM_EXTERNAL_BASE_URL env-var contract.
"""

from .deps import DepsHandle, kvbm_deps
from .eval import AggDeterminismTester, kvbm_tester, make_determinism_tester
from .server import (
    KvbmModelConfig,
    KvbmServerManager,
    KvbmServerSpec,
    ServerHandle,
    build_kv_transfer_config,
    kvbm_server,
    kvbm_server_spec,
)

__all__ = [
    "AggDeterminismTester",
    "DepsHandle",
    "KvbmModelConfig",
    "KvbmServerManager",
    "KvbmServerSpec",
    "ServerHandle",
    "build_kv_transfer_config",
    "kvbm_deps",
    "kvbm_server",
    "kvbm_server_spec",
    "kvbm_tester",
    "make_determinism_tester",
]
