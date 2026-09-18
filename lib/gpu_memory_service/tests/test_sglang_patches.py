# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for SGLang ModelRunner memory accounting patches."""

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

from gpu_memory_service.integrations.sglang import patches

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def _patch_model_runner(monkeypatch, model_runner, preloaded_weights_bytes):
    module_name = "sglang.srt.model_executor.model_runner"
    module = ModuleType(module_name)
    module.ModelRunner = model_runner
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(patches, "_model_runner_patched", False)
    monkeypatch.setattr(
        patches,
        "get_gms_memory_saver_impl",
        lambda: SimpleNamespace(preloaded_weights_bytes=preloaded_weights_bytes),
    )
    patches.patch_model_runner()


def test_patch_model_runner_adjusts_persistent_baseline_once(monkeypatch):
    class ModelRunner:
        def alloc_memory_pool(self, memory_pool_config=None):
            self.calls.append((self.pre_model_load_memory, memory_pool_config))
            return memory_pool_config

    _patch_model_runner(monkeypatch, ModelRunner, 2 << 30)
    patched_method = ModelRunner.alloc_memory_pool
    patches.patch_model_runner()
    monkeypatch.setattr(patches, "_model_runner_patched", False)
    patches.patch_model_runner()

    runner = ModelRunner()
    runner.pre_model_load_memory = 10.0
    runner.calls = []
    positional_config = object()
    keyword_config = object()

    assert runner.alloc_memory_pool(positional_config) is positional_config
    assert runner.alloc_memory_pool(memory_pool_config=keyword_config) is keyword_config
    assert ModelRunner.alloc_memory_pool is patched_method
    assert runner.pre_model_load_memory == 12.0
    assert runner.calls == [(12.0, positional_config), (12.0, keyword_config)]


def test_patch_model_runner_leaves_baseline_unchanged_without_preload(monkeypatch):
    class ModelRunner:
        def alloc_memory_pool(self):
            return self.pre_model_load_memory

    _patch_model_runner(monkeypatch, ModelRunner, 0)
    runner = ModelRunner()
    runner.pre_model_load_memory = 10.0

    assert runner.alloc_memory_pool() == 10.0
    assert runner.pre_model_load_memory == 10.0


def test_shared_kv_shadow_adopts_geometry_without_memory_profile(monkeypatch):
    from gpu_memory_service.integrations.common import kv_lease_client
    from gpu_memory_service.integrations.sglang import kv_identity
    from sglang.srt.mem_cache import kv_cache_configurator
    from sglang.srt.model_executor import pool_configurator

    class FakeKVCacheConfigurator:
        page_size = 64
        gpu_id = 0
        server_args = SimpleNamespace(mem_fraction_static=0.8)

        def _resolve_memory_pool_config(self, _pre_model_load_memory):
            raise AssertionError("shadow reattach must not profile free HBM")

        def resolve_max_num_reqs(self, tokens):
            assert tokens == 9 * 64
            return 7

    class FakePoolConfigurator:
        def calculate_pool_sizes_from_max_tokens(self, tokens, page_size):
            assert (tokens, page_size) == (9 * 64, 64)
            return SimpleNamespace(
                max_total_num_tokens=tokens,
                max_running_requests=None,
                mem_fraction_static=None,
            )

        def finalize_with_max_running_requests(self, config):
            return config

    monkeypatch.setattr(
        kv_cache_configurator, "KVCacheConfigurator", FakeKVCacheConfigurator
    )
    monkeypatch.setattr(
        pool_configurator,
        "create_memory_pool_configurator",
        lambda _configurator: FakePoolConfigurator(),
    )
    monkeypatch.setattr(kv_identity, "shared_kv_enabled", lambda: True)
    monkeypatch.setattr(kv_lease_client, "kv_leases_enabled", lambda _engine: True)
    monkeypatch.setattr(
        kv_lease_client,
        "read_kv_lease_namespace_total_blocks",
        lambda *_args, **_kwargs: ("sglang:gpu0:page-pool", 10),
    )
    monkeypatch.setattr(patches, "_kv_pool_geometry_patched", False)

    patches.patch_shared_kv_pool_geometry()
    config = FakeKVCacheConfigurator()._resolve_memory_pool_config(180.0)

    assert config.max_total_num_tokens == 9 * 64
    assert config.max_running_requests == 7
    assert config.mem_fraction_static == 0.8
