# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer A: dependency bring-up.

v1 wraps the existing reuse-or-spawn `runtime_services` fixture
(NATS + etcd). v2 (agg) returns an empty handle — discovery defaults to
None per `lib/kvbm-config/src/messenger.rs:43` so single-process agg
mode needs no external services.

External-attach mode: when `KVBM_EXTERNAL_BASE_URL` is set the fixture
short-circuits to an empty handle for both versions — the long-lived
external server already brought up its own deps.
"""

import os
from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class DepsHandle:
    """Layer-A handle.

    For v1 the underlying NATS+etcd are reachable via the env vars set by
    the wrapped `runtime_services` fixture (`NATS_SERVER`, `ETCD_ENDPOINTS`).
    For v2 agg both are `None`. Phase 2 doesn't expose runtime objects on
    this handle — the env vars are the contract used by vllm and the
    KVBM connector.
    """

    version: str
    nats_url: Optional[str] = None
    etcd_endpoints: Optional[str] = None


@pytest.fixture(scope="function")
def kvbm_deps(request, kvbm_server_spec) -> DepsHandle:
    """Dependency bring-up for the chosen KVBM version.

    v1: pulls in `runtime_services` via `request.getfixturevalue` so we only
    spawn NATS+etcd when actually needed.
    v2 agg: returns an empty handle; no external services required.
    External-attach mode: empty handle for both versions.
    """
    version = kvbm_server_spec.kvbm_version

    if os.environ.get("KVBM_EXTERNAL_BASE_URL"):
        return DepsHandle(version=version)

    if version == "v1":
        # Triggers reuse-or-spawn of NATS+etcd; sets NATS_SERVER / ETCD_ENDPOINTS env vars.
        request.getfixturevalue("runtime_services")
        return DepsHandle(
            version="v1",
            nats_url=os.environ.get("NATS_SERVER"),
            etcd_endpoints=os.environ.get("ETCD_ENDPOINTS"),
        )

    if version == "v2":
        return DepsHandle(version="v2")

    raise ValueError(f"unknown kvbm_version: {version!r}")
