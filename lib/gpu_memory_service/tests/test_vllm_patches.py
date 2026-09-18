# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused API regression coverage for vLLM GMS monkey-patches."""

import sys
from types import ModuleType

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

from gpu_memory_service.integrations.vllm import patches

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def test_nixl_base_patch_covers_pull_and_push_registration(monkeypatch):
    """Both NIXL modes defer registration through the vLLM 0.29 base."""

    class NixlBaseConnector:
        def register_kv_caches(self, kv_caches):
            raise AssertionError("scratch-backed KV caches must be deferred")

    class NixlPullConnector(NixlBaseConnector):
        pass

    class NixlPushConnector(NixlBaseConnector):
        pass

    module = ModuleType(patches._NIXL_MODULE)
    module.NixlBaseConnector = NixlBaseConnector
    module.NixlPullConnector = NixlPullConnector
    module.NixlPushConnector = NixlPushConnector
    module.NixlConnector = NixlPullConnector
    monkeypatch.setitem(sys.modules, patches._NIXL_MODULE, module)
    monkeypatch.setattr(patches, "_register_kv_caches_patched", False)
    monkeypatch.setattr(patches, "get_gms_client_memory_manager", lambda _tag: object())
    monkeypatch.setattr(patches, "is_scratch", lambda _manager: True)

    patches.patch_register_kv_caches()

    pull = NixlPullConnector()
    push = NixlPushConnector()
    kv_caches = {"layer.0": object()}
    pull.register_kv_caches(kv_caches)
    push.register_kv_caches(kv_caches)

    assert pull._scratch_kv_pending is kv_caches
    assert push._scratch_kv_pending is kv_caches
