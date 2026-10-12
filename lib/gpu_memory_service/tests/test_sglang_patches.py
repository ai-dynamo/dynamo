# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for SGLang ModelRunner memory accounting patches."""

import importlib
import logging
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

import torch
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


def test_patch_static_state_for_gms_patches_weight_updater(monkeypatch):
    try:
        weight_updater = importlib.import_module(
            "sglang.srt.managers.scheduler_components.weight_updater"
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "sglang",
            "sglang.srt",
            "sglang.srt.managers",
            "sglang.srt.managers.scheduler_components",
            "sglang.srt.managers.scheduler_components.weight_updater",
        }:
            raise
        pytest.skip(f"SGLang weight_updater is unavailable: {exc.name}")

    def original_export(model):
        return {"buffers": ["original"]}

    def original_import(model, static_params):
        raise AssertionError("original _import_static_state called")

    monkeypatch.setattr(patches, "_static_state_patched", False)
    monkeypatch.setattr(weight_updater, "_export_static_state", original_export)
    monkeypatch.setattr(weight_updater, "_import_static_state", original_import)

    patches.patch_static_state_for_gms()

    model = torch.nn.Linear(1, 1)
    model.register_buffer("stable", torch.zeros(1))
    assert weight_updater._export_static_state is not original_export
    assert weight_updater._import_static_state is not original_import
    assert weight_updater._export_static_state(model) == {"buffers": []}
    assert weight_updater._import_static_state(model, {"buffers": []}) is None


def _install_fake_sglang(monkeypatch):
    def original_export(model):
        return {"buffers": ["original"]}

    def original_import(model, static_params):
        raise AssertionError("original _import_static_state called")

    mixin_name = "sglang.srt.managers.scheduler_update_weights_mixin"
    mixin = ModuleType(mixin_name)
    mixin._export_static_state = original_export
    mixin._import_static_state = original_import
    managers = ModuleType("sglang.srt.managers")
    managers.scheduler_update_weights_mixin = mixin
    for name, module in (
        ("sglang", ModuleType("sglang")),
        ("sglang.srt", ModuleType("sglang.srt")),
        ("sglang.srt.managers", managers),
        (mixin_name, mixin),
    ):
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(patches, "_static_state_patched", False)
    return mixin, original_export


@pytest.mark.parametrize("has_components_package", [False, True])
def test_patch_static_state_for_gms_falls_back_to_legacy_mixin(
    monkeypatch, has_components_package
):
    mixin, original_export = _install_fake_sglang(monkeypatch)
    components_name = "sglang.srt.managers.scheduler_components"
    if has_components_package:
        components = ModuleType(components_name)
        components.__path__ = []
        monkeypatch.setitem(sys.modules, components_name, components)
    else:
        monkeypatch.delitem(sys.modules, components_name, False)
    monkeypatch.delitem(sys.modules, f"{components_name}.weight_updater", False)

    patches.patch_static_state_for_gms()

    assert patches._static_state_patched
    assert mixin._export_static_state is not original_export
    assert mixin._export_static_state(object()) == {"buffers": []}
    assert mixin._import_static_state(object(), {"buffers": []}) is None


def test_patch_static_state_for_gms_keeps_weight_updater_dependency_errors(
    monkeypatch, tmp_path, caplog
):
    mixin, original_export = _install_fake_sglang(monkeypatch)
    (tmp_path / "weight_updater.py").write_text("import gms_test_missing_dependency\n")
    components = ModuleType("sglang.srt.managers.scheduler_components")
    components.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, components.__name__, components)
    monkeypatch.delitem(sys.modules, f"{components.__name__}.weight_updater", False)

    # Capture directly so SGLang's logger settings cannot hide the warning.
    monkeypatch.setattr(patches.logger, "disabled", False)
    monkeypatch.setattr(patches.logger, "propagate", False)
    monkeypatch.setattr(patches.logger, "handlers", [caplog.handler])
    with caplog.at_level(logging.WARNING, logger=patches.logger.name):
        patches.patch_static_state_for_gms()

    assert not patches._static_state_patched
    assert mixin._export_static_state is original_export
    assert "gms_test_missing_dependency" in caplog.text
