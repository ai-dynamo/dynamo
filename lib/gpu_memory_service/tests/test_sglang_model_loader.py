# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for GMSModelLoader's preloaded-weight accounting.

SGLang reads ``loader.preloaded_weights_bytes`` through
``ModelRunner.preloaded_weights_bytes`` and feeds it to
``account_preloaded_weights()`` before the KV pool is sized, so the value this
loader reports in each lock mode is load-bearing for startup correctness.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

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
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.vmm import VMMDeviceType

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
]

MODULE = "gpu_memory_service.integrations.sglang.model_loader"
WEIGHT_BYTES = 3 << 30


@dataclass
class _LoadConfig:
    """Stand-in for sglang's LoadConfig.

    strip_gms_model_loader_config() calls dataclasses.replace() on it, so it
    has to be a real dataclass rather than a SimpleNamespace.
    """

    load_format: Any = "gms"
    model_loader_extra_config: dict = field(default_factory=dict)


class _StubBaseModelLoader(ABC):
    """Mirror of sglang.srt.model_loader.loader.BaseModelLoader (0.5.21).

    Two upstream details are reproduced deliberately because GMSModelLoader
    depends on both: the class-level ``preloaded_weights_bytes = 0`` default
    that write mode falls through to, and the abstract ``download_model`` /
    ``load_model`` that make a loader missing either one uninstantiable.
    """

    preloaded_weights_bytes: int = 0

    def __init__(self, load_config):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, *, model_config, device_config):
        raise NotImplementedError


class _FakeModel:
    def __init__(self, name: str):
        self.name = name
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


@pytest.fixture
def gms(monkeypatch):
    """Import GMSModelLoader against a minimal SGLang stub.

    SGLang is absent from the unit-test image but model_loader imports
    BaseModelLoader at module scope. The sys.modules entries go through
    monkeypatch so a real sglang (present in the runtime image) is never
    shadowed past this test.
    """
    default_loaders: list = []

    class StubDefaultModelLoader(_StubBaseModelLoader):
        def __init__(self, load_config):
            super().__init__(load_config)
            self.downloaded: list = []
            self.loaded: list = []
            default_loaders.append(self)

        def download_model(self, model_config) -> None:
            self.downloaded.append(model_config)

        def load_model(self, *, model_config, device_config):
            self.loaded.append((model_config, device_config))
            return _FakeModel("from-disk")

    loader_mod = ModuleType("sglang.srt.model_loader.loader")
    loader_mod.BaseModelLoader = _StubBaseModelLoader
    loader_mod.DefaultModelLoader = StubDefaultModelLoader

    model_loader_pkg = ModuleType("sglang.srt.model_loader")
    model_loader_pkg.loader = loader_mod

    srt_pkg = ModuleType("sglang.srt")
    srt_pkg.model_loader = model_loader_pkg

    sglang_pkg = ModuleType("sglang")
    sglang_pkg.srt = srt_pkg

    for name, module in (
        ("sglang", sglang_pkg),
        ("sglang.srt", srt_pkg),
        ("sglang.srt.model_loader", model_loader_pkg),
        ("sglang.srt.model_loader.loader", loader_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop(MODULE, None)
    try:
        module = importlib.import_module(MODULE)
        yield SimpleNamespace(
            module=module,
            loader_cls=module.GMSModelLoader,
            default_loaders=default_loaders,
        )
    finally:
        sys.modules.pop(MODULE, None)


def _install_impl(gms, monkeypatch, granted_lock_type):
    """Install a fake memory-saver impl and return it."""
    allocator = SimpleNamespace(
        granted_lock_type=granted_lock_type,
        total_bytes=WEIGHT_BYTES,
    )
    impl = SimpleNamespace(
        allocators={"weights": allocator},
        imported_weights_bytes=0,
        preloaded_weights_bytes=0,
        finalized=[],
    )
    impl.finalize_write_mode = impl.finalized.append
    monkeypatch.setattr(gms.module, "get_gms_memory_saver_impl", lambda: impl)
    return impl


def test_loader_implements_base_model_loader_contract(gms):
    """GMSModelLoader must be a concrete BaseModelLoader.

    Dropping the base class removes the ``preloaded_weights_bytes`` default
    that SGLang reads off every loader; dropping ``download_model`` leaves an
    abstract class that cannot be instantiated at all.
    """
    loader = gms.loader_cls(_LoadConfig())

    assert isinstance(loader, _StubBaseModelLoader)
    assert not inspect.isabstract(gms.loader_cls)
    assert loader.preloaded_weights_bytes == 0


def test_download_model_delegates_and_strips_gms_config(gms):
    loader = gms.loader_cls(
        _LoadConfig(model_loader_extra_config={"gms_read_only": True, "keep": 1})
    )

    loader.download_model("model-config")

    (default_loader,) = gms.default_loaders
    assert default_loader.downloaded == ["model-config"]
    assert default_loader.load_config.load_format == "auto"
    assert default_loader.load_config.model_loader_extra_config == {"keep": 1}


def test_load_model_requires_initialized_impl(gms, monkeypatch):
    monkeypatch.setattr(gms.module, "get_gms_memory_saver_impl", lambda: None)
    loader = gms.loader_cls(_LoadConfig())

    with pytest.raises(RuntimeError, match="GMS impl not initialized"):
        loader.load_model(model_config="mc", device_config="dc")


@pytest.mark.parametrize(
    "granted_lock_type", [GrantedLockType.RW, GrantedLockType.RW_DATA]
)
def test_write_mode_reports_no_preloaded_weights(gms, monkeypatch, granted_lock_type):
    """A writer loads weights *after* the baseline snapshot.

    Upstream's KV formula already subtracts them, so the loader must leave
    preloaded_weights_bytes at the inherited 0; reporting the weight bytes
    here would add them back a second time and oversize the pool.
    """
    impl = _install_impl(gms, monkeypatch, granted_lock_type)
    loader = gms.loader_cls(_LoadConfig())

    model = loader.load_model(model_config="mc", device_config="dc")

    (default_loader,) = gms.default_loaders
    assert default_loader.loaded == [("mc", "dc")]
    assert impl.finalized == [model]
    assert loader.preloaded_weights_bytes == 0


@pytest.mark.parametrize(
    "device_type, expected_loader_bytes",
    [
        # CUDA's get_available_gpu_memory() is device-wide, so the pre-load
        # baseline really was depressed by the GMS-resident weights and
        # SGLang's add-back restores it.
        (VMMDeviceType.CUDA, WEIGHT_BYTES),
        # XPU's probe is total_memory - torch.xpu.memory_allocated(), which
        # never saw the VMM mappings, so the baseline was never depressed and
        # the add-back would inflate it instead.
        (VMMDeviceType.XPU, 0),
    ],
)
def test_read_mode_preloaded_weights_depend_on_device(
    gms, monkeypatch, device_type, expected_loader_bytes
):
    impl = _install_impl(gms, monkeypatch, GrantedLockType.RO)
    model = _FakeModel("from-gms")
    materialized: list = []

    monkeypatch.setattr(gms_vmm, "get_vmm_device_type", lambda: device_type)
    monkeypatch.setattr(
        gms.module, "torch_device", lambda: SimpleNamespace(current_device=lambda: 0)
    )
    monkeypatch.setattr(
        gms.module,
        "materialize_module_from_gms",
        lambda allocator, module, device_index: materialized.append(
            (allocator, module, device_index)
        ),
    )
    monkeypatch.setattr(
        gms.loader_cls,
        "_create_meta_model",
        lambda self, model_config, device_config: model,
    )

    loader = gms.loader_cls(_LoadConfig())
    returned = loader.load_model(model_config="mc", device_config="dc")

    assert returned is model
    assert model.eval_called
    assert gms.default_loaders == []
    assert materialized == [(impl.allocators["weights"], model, 0)]

    # The impl always carries the true byte count: the XPU avail-mem probe
    # patch reads it from there, while SGLang reads the loader attribute.
    assert impl.imported_weights_bytes == WEIGHT_BYTES
    assert impl.preloaded_weights_bytes == WEIGHT_BYTES
    assert loader.preloaded_weights_bytes == expected_loader_bytes
