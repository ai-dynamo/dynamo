# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types

import pytest
from gpu_memory_service.integrations.sglang import patches as sglang_patches

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def test_patch_model_runner_rewrites_total_gpu_memory(monkeypatch):
    fake_sglang = types.ModuleType("sglang")
    fake_srt = types.ModuleType("sglang.srt")
    fake_executor = types.ModuleType("sglang.srt.model_executor")
    fake_model_runner = types.ModuleType("sglang.srt.model_executor.model_runner")

    class FakeModelRunner:
        def init_memory_pool(self, total_gpu_memory):
            self.seen_total_gpu_memory = total_gpu_memory
            return total_gpu_memory

    fake_model_runner.ModelRunner = FakeModelRunner
    fake_sglang.srt = fake_srt
    fake_srt.model_executor = fake_executor
    fake_executor.model_runner = fake_model_runner

    fake_memory_saver = types.ModuleType(
        "gpu_memory_service.integrations.sglang.memory_saver"
    )

    class FakeImpl:
        def get_imported_weights_bytes(self):
            return 8 << 30

    fake_memory_saver.get_gms_memory_saver_impl = lambda: FakeImpl()

    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", fake_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor", fake_executor)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner",
        fake_model_runner,
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.integrations.sglang.memory_saver",
        fake_memory_saver,
    )
    monkeypatch.setattr(sglang_patches, "_model_runner_patched", False)
    monkeypatch.delattr(FakeModelRunner, "_gms_patched", raising=False)
    monkeypatch.setattr(
        sglang_patches.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        sglang_patches.torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(total_memory=80 * (1 << 30)),
    )

    sglang_patches.patch_model_runner()

    runner = FakeModelRunner()
    assert runner.init_memory_pool(40.0) == pytest.approx(80.0)
    assert runner.seen_total_gpu_memory == pytest.approx(80.0)
