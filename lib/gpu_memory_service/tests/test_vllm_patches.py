# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for vLLM GMS monkey-patches."""

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


def test_patch_register_kv_caches_uses_canonical_vllm_connector(monkeypatch):
    """Patch NixlConnector from vLLM's canonical v1.nixl package."""

    class NixlConnector:
        def register_kv_caches(self, kv_caches):
            return kv_caches

    module_name = "vllm.distributed.kv_transfer.kv_connector.v1.nixl"
    module = ModuleType(module_name)
    module.NixlConnector = NixlConnector
    monkeypatch.setitem(sys.modules, module_name, module)
    original_register = NixlConnector.register_kv_caches
    monkeypatch.setattr(patches, "_register_kv_caches_patched", False)

    patches.patch_register_kv_caches()

    assert patches._register_kv_caches_patched
    assert NixlConnector.register_kv_caches is not original_register
