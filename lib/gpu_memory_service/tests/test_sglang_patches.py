# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the SGLang GMS patches.

Covers ModelRunner memory accounting and the torch_memory_saver hook_mode
claim that decides whether GMS installs its own impl.
"""

import sys
from contextlib import contextmanager
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

from gpu_memory_service.common import vmm as gms_vmm
from gpu_memory_service.common.vmm import VMMDeviceType
from gpu_memory_service.integrations.common import utils as gms_common_utils
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


def test_patch_model_runner_skips_when_upstream_accounts(monkeypatch):
    """SGLang >=0.5.21 accounts for preloaded weights itself.

    Applying the legacy patch on top would add the weight bytes to
    pre_model_load_memory twice and oversize the KV cache.
    """

    class ModelRunner:
        def account_preloaded_weights(self, preloaded_weights_bytes):
            self.pre_model_load_memory += preloaded_weights_bytes / (1 << 30)

        def alloc_memory_pool(self, memory_pool_config=None):
            return self.pre_model_load_memory

    original_method = ModelRunner.alloc_memory_pool
    _patch_model_runner(monkeypatch, ModelRunner, 2 << 30)

    assert ModelRunner.alloc_memory_pool is original_method
    assert not hasattr(ModelRunner, "_gms_patched")

    runner = ModelRunner()
    runner.pre_model_load_memory = 10.0
    runner.account_preloaded_weights(2 << 30)

    assert runner.alloc_memory_pool() == 12.0
    assert runner.pre_model_load_memory == 12.0


class _FakeGMSImpl:
    """Stand-in for GMSMemorySaverImpl, which would otherwise dial a server."""

    def __init__(self, device_index, mode, ro_connect_timeout_ms):
        self.device_index = device_index
        self.mode = mode
        self.ro_connect_timeout_ms = ro_connect_timeout_ms
        self.allocators = {
            "weights": SimpleNamespace(granted_lock_type=SimpleNamespace(name="RW"))
        }


@pytest.fixture
def memory_saver_stub(monkeypatch):
    """Run patch_torch_memory_saver() against a stubbed torch_memory_saver.

    The package is absent from the unit-test image and the patch silently
    returns when it cannot be imported, so without these stubs every
    assertion about hook_mode would pass vacuously.
    """
    original_calls = []

    class TorchMemorySaver:
        def __init__(self, **ctor_kwargs):
            self._impl = None
            self._impl_ctor_kwargs = dict(ctor_kwargs)

        def _ensure_initialized(self):
            original_calls.append(self)
            self._impl = "upstream-impl"

    entrypoint = ModuleType("torch_memory_saver.entrypoint")
    entrypoint.TorchMemorySaver = TorchMemorySaver

    @contextmanager
    def configure_subprocess():
        yield

    tms = ModuleType("torch_memory_saver")
    tms.entrypoint = entrypoint
    tms.TorchMemorySaver = TorchMemorySaver
    tms.torch_memory_saver = TorchMemorySaver()
    tms.configure_subprocess = configure_subprocess

    monkeypatch.setitem(sys.modules, "torch_memory_saver", tms)
    monkeypatch.setitem(sys.modules, "torch_memory_saver.entrypoint", entrypoint)
    monkeypatch.setattr(patches, "_torch_memory_saver_patched", False)
    monkeypatch.setattr(patches, "GMSMemorySaverImpl", _FakeGMSImpl)
    monkeypatch.setattr(
        gms_common_utils,
        "torch_device",
        lambda: SimpleNamespace(current_device=lambda: 0),
    )

    return SimpleNamespace(saver_cls=TorchMemorySaver, original_calls=original_calls)


@pytest.mark.parametrize(
    "hook_mode, device_type, expect_gms",
    [
        # SGLang forces hook_mode="torch" at import time on XPU because the
        # LD_PRELOAD mode is CUDA/HIP-only. That leaves no way to request
        # "gms", so the patch claims the saver anyway -- otherwise
        # GMSModelLoader loads with no impl and the load fails.
        ("torch", VMMDeviceType.XPU, True),
        # On CUDA nothing forces the value, so an explicit "torch" stays a
        # real opt-out.
        ("torch", VMMDeviceType.CUDA, False),
        (None, VMMDeviceType.CUDA, True),
        ("gms", VMMDeviceType.CUDA, True),
    ],
)
def test_torch_memory_saver_hook_mode_claim(
    memory_saver_stub, monkeypatch, hook_mode, device_type, expect_gms
):
    monkeypatch.setattr(gms_vmm, "get_vmm_device_type", lambda: device_type)
    patches.patch_torch_memory_saver()

    saver = memory_saver_stub.saver_cls(hook_mode=hook_mode)
    saver._ensure_initialized()

    if expect_gms:
        assert isinstance(saver._impl, _FakeGMSImpl)
        assert saver.gms_impl is saver._impl
        assert saver._impl.device_index == 0
        assert memory_saver_stub.original_calls == []
        # The patched path drops the ctor kwargs once it owns the impl.
        assert not hasattr(saver, "_impl_ctor_kwargs")
    else:
        assert saver._impl == "upstream-impl"
        assert saver.gms_impl is None
        assert memory_saver_stub.original_calls == [saver]


def test_torch_memory_saver_skips_already_initialized(memory_saver_stub, monkeypatch):
    monkeypatch.setattr(gms_vmm, "get_vmm_device_type", lambda: VMMDeviceType.XPU)
    patches.patch_torch_memory_saver()

    saver = memory_saver_stub.saver_cls(hook_mode="torch")
    saver._impl = "pre-existing"
    saver._ensure_initialized()

    assert saver._impl == "pre-existing"
    assert memory_saver_stub.original_calls == []
