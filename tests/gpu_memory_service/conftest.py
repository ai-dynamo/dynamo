# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for GPU Memory Service tests."""

import pytest

# Skip collection entirely if gpu_memory_service is not installed.
# This package lives under nested common/ and integration/ subdirectories, so
# we ignore those directories directly instead of only matching test files next
# to this conftest.
try:
    import gpu_memory_service  # noqa: F401
except ImportError:
    collect_ignore = ["common", "integration"]

from tests.utils.port_utils import allocate_port, deallocate_ports


@pytest.fixture
def gms_ports():
    """Allocate ports for GMS tests.

    Returns a dict with ports for:
    - frontend: Frontend HTTP port
    - shadow_system: System port for the first shadow engine
    - shadow2_system: System port for the second shadow engine
    - primary_system: System port for primary engine (failover test only)
    - shadow_kv_event: KV event port for the first shadow engine (vLLM)
    - shadow2_kv_event: KV event port for the second shadow engine (vLLM)
    - primary_kv_event: KV event port for primary engine (vLLM)
    - shadow_nixl: NIXL side channel port for the first shadow engine (vLLM)
    - shadow2_nixl: NIXL side channel port for the second shadow engine (vLLM)
    - primary_nixl: NIXL side channel port for primary engine (vLLM)
    - shadow_sglang: SGLang HTTP port for the first shadow engine
    - shadow2_sglang: SGLang HTTP port for the second shadow engine
    - primary_sglang: SGLang HTTP port for primary engine
    """
    ports = [
        allocate_port(p)
        for p in [
            8200,
            8100,
            8101,
            8102,
            20080,
            20081,
            20082,
            20096,
            20097,
            20098,
            30000,
            30001,
            30002,
        ]
    ]
    yield {
        "frontend": ports[0],
        "shadow_system": ports[1],
        "primary_system": ports[2],
        "shadow2_system": ports[3],
        "shadow_kv_event": ports[4],
        "primary_kv_event": ports[5],
        "shadow2_kv_event": ports[6],
        "shadow_nixl": ports[7],
        "primary_nixl": ports[8],
        "shadow2_nixl": ports[9],
        "shadow_sglang": ports[10],
        "primary_sglang": ports[11],
        "shadow2_sglang": ports[12],
    }
    deallocate_ports(ports)
