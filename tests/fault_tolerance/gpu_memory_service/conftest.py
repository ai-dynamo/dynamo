# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for GPU Memory Service tests."""

import pytest

# Skip collection entirely if gpu_memory_service is not installed
try:
    import gpu_memory_service  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]

from tests.utils.port_utils import allocate_port, deallocate_ports


@pytest.fixture
def gms_ports():
    """Allocate ports for GMS tests.

    Returns a dict with ports for:
    - frontend: Frontend HTTP port
    - shadow_system: System port for shadow/primary engine
    - primary_system: System port for primary engine (failover test only)
    - shadow_kv_event: KV event port for shadow engine
    - primary_kv_event: KV event port for primary engine
    - shadow_nixl: NIXL side channel port for shadow engine
    - primary_nixl: NIXL side channel port for primary engine
    """
    ports = [allocate_port(p) for p in [8200, 8100, 8101, 20080, 20081, 20096, 20097]]
    yield {
        "frontend": ports[0],
        "shadow_system": ports[1],
        "primary_system": ports[2],
        "shadow_kv_event": ports[3],
        "primary_kv_event": ports[4],
        "shadow_nixl": ports[5],
        "primary_nixl": ports[6],
    }
    deallocate_ports(ports)
