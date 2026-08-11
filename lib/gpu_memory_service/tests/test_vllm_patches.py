# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for vLLM GMS monkey-patches."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

from gpu_memory_service.client.torch import allocator
from gpu_memory_service.integrations.vllm import patches

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture
def nixl_connector(monkeypatch):
    """Expose only vLLM's canonical v1.nixl package, not the removed module."""

    class NixlConnector:
        def __init__(self):
            self.registered = []

        def register_kv_caches(self, kv_caches):
            self.registered.append(kv_caches)
            return "registered"

    module_name = "vllm.distributed.kv_transfer.kv_connector.v1.nixl"
    module = ModuleType(module_name)
    module.NixlConnector = NixlConnector
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.delitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
        raising=False,
    )
    monkeypatch.setattr(patches, "_register_kv_caches_patched", False)
    patches.patch_register_kv_caches()

    assert patches._register_kv_caches_patched
    return NixlConnector


def test_register_kv_caches_patch_defers_scratch_registration(
    monkeypatch, nixl_connector
):
    manager = object()
    monkeypatch.setattr(
        allocator, "get_gms_client_memory_manager", lambda _name: manager
    )
    monkeypatch.setattr(allocator, "is_scratch", lambda value: value is manager)

    connector = nixl_connector()
    kv_caches = {"layer.0": object()}

    assert connector.register_kv_caches(kv_caches) is None
    assert connector._scratch_kv_pending is kv_caches
    assert connector.registered == []


def test_register_kv_caches_patch_calls_vllm_for_real_backing(
    monkeypatch, nixl_connector
):
    monkeypatch.setattr(
        allocator,
        "get_gms_client_memory_manager",
        lambda _name: SimpleNamespace(),
    )
    monkeypatch.setattr(allocator, "is_scratch", lambda _manager: False)

    connector = nixl_connector()
    kv_caches = {"layer.0": object()}

    assert connector.register_kv_caches(kv_caches) == "registered"
    assert connector.registered == [kv_caches]
    assert not hasattr(connector, "_scratch_kv_pending")


def test_register_kv_caches_patch_fails_closed_on_state_lookup_error(
    monkeypatch, nixl_connector
):
    def fail_lookup(_name):
        raise RuntimeError("manager unavailable")

    monkeypatch.setattr(allocator, "get_gms_client_memory_manager", fail_lookup)

    connector = nixl_connector()
    with pytest.raises(RuntimeError, match="manager unavailable"):
        connector.register_kv_caches({"layer.0": object()})

    assert connector.registered == []
