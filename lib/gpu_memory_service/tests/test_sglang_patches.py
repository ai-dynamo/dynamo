# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the SGLang GMS patches.

Covers the XPU KV sizing probe rebind and the torch_memory_saver hook_mode
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


def _install_probe_modules(monkeypatch, free_gb):
    """Fake the SGLang modules that define and from-import the probe."""

    def get_available_gpu_memory(*args, **kwargs):
        return free_gb

    modules = {}
    for name in (
        "sglang.srt.utils.common",  # definition site
        "sglang.srt.utils",  # `from .common import *` re-export
        "sglang.srt.fake_consumer",  # an existing from-import binding
    ):
        module = ModuleType(name)
        module.get_available_gpu_memory = get_available_gpu_memory
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module

    unrelated = ModuleType("sglang.srt.fake_unrelated")
    unrelated.get_available_gpu_memory = lambda *args, **kwargs: -1.0
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)
    modules[unrelated.__name__] = unrelated

    monkeypatch.setattr(patches, "_xpu_kv_cache_sizing_patched", False)
    return get_available_gpu_memory, modules


def test_xpu_kv_cache_sizing_rebinds_every_reference(monkeypatch):
    """Every binding of the probe gets the GMS adjustment, not a fixed list.

    Covers the definition site, a package re-export, an existing from-import
    binding and a from-import executed after the patch. A different function
    that happens to share the name is left alone.
    """
    original, modules = _install_probe_modules(monkeypatch, free_gb=10.0)
    monkeypatch.setattr(gms_vmm, "get_vmm_device_type", lambda: VMMDeviceType.XPU)
    impl = SimpleNamespace(preloaded_weights_bytes=0)
    monkeypatch.setattr(patches, "get_gms_memory_saver_impl", lambda: impl)

    patches.patch_xpu_kv_cache_sizing_for_gms()

    late = {}
    exec("from sglang.srt.utils import get_available_gpu_memory", late)
    probes = [
        m.get_available_gpu_memory
        for n, m in modules.items()
        if n != "sglang.srt.fake_unrelated"
    ]
    probes.append(late["get_available_gpu_memory"])
    assert all(probe is not original for probe in probes)

    # Writer / pre-load: nothing imported yet, so no adjustment.
    assert [probe() for probe in probes] == [10.0] * len(probes)

    # Reader after import: GMS-resident weights are subtracted everywhere.
    impl.preloaded_weights_bytes = 2 << 30
    assert [probe() for probe in probes] == [8.0] * len(probes)

    assert modules["sglang.srt.fake_unrelated"].get_available_gpu_memory() == -1.0


def test_xpu_kv_cache_sizing_is_noop_on_cuda(monkeypatch):
    original, modules = _install_probe_modules(monkeypatch, free_gb=10.0)
    monkeypatch.setattr(gms_vmm, "get_vmm_device_type", lambda: VMMDeviceType.CUDA)

    patches.patch_xpu_kv_cache_sizing_for_gms()

    for name, module in modules.items():
        if name != "sglang.srt.fake_unrelated":
            assert module.get_available_gpu_memory is original


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
    # The real package is installed in the XPU image and is shadowed for the
    # duration of this test. Keep __path__ so any submodule import attempted
    # meanwhile raises ModuleNotFoundError rather than a bare AttributeError.
    tms.__path__ = []
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
