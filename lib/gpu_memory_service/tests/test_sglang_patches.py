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
