# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-GPU VMM ownership, admission and rollback tests with a fake CUDA driver."""

import ctypes
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_helper_relative = Path("lib/bindings/kvbm/python/kvbm/vllm_integration/gpu_mnnvl.py")
_helper_file = next(
    (
        parent / _helper_relative
        for parent in Path(__file__).resolve().parents
        if (parent / _helper_relative).is_file()
    ),
    None,
)
if _helper_file is None:
    _package = importlib.util.find_spec("kvbm")
    if _package is None or _package.submodule_search_locations is None:
        raise ModuleNotFoundError(
            "KVBM fabric helper source is unavailable", name="kvbm"
        )
    _locations = list(_package.submodule_search_locations)
    if len(_locations) != 1:
        raise RuntimeError("One unambiguous KVBM helper source is required")
    _helper_file = Path(_locations[0]) / "vllm_integration/gpu_mnnvl.py"
_helper_spec = importlib.util.spec_from_file_location(
    __name__ + "._gpu_mnnvl", _helper_file
)
if _helper_spec is None or _helper_spec.loader is None:
    raise RuntimeError("KVBM fabric helper source cannot be loaded")
m = importlib.util.module_from_spec(_helper_spec)
sys.modules[_helper_spec.name] = m
_helper_spec.loader.exec_module(m)
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.kvbm,
    pytest.mark.gpu_0,
]


class FakeDriver:
    def __init__(self):
        self.calls = []
        self.fail = None
        self.identity = m.FabricIdentity(
            "GPU-00000000-0000-0000-0000-000000000001", "1" * 32, 0
        )
        self.properties_from_handle = self.properties(0)

    def record(self, name, *args):
        self.calls.append((name, *args))
        if self.fail == name:
            self.fail = None
            raise RuntimeError("injected " + name)

    def device(self):
        return 0

    def gpu_uuid(self, device):
        return self.identity.gpu_uuid

    def retain_context(self, device):
        self.record("retain", device)
        return 77

    @contextmanager
    def context(self, context):
        self.record("push", context)
        try:
            yield
        finally:
            self.record("pop", context)

    def release_context(self, device):
        self.record("release-context", device)

    def properties(self, device):
        return m._CudaDriver.properties(None, device)

    def granularity(self, properties):
        self.record(
            "granularity",
            properties.type,
            properties.requestedHandleTypes,
            properties.location.type,
        )
        return 8

    def create(self, size, properties):
        self.record("create", size, properties.requestedHandleTypes)
        return 90

    def allocation_properties(self, handle):
        self.record("properties", handle)
        return self.properties_from_handle

    def reserve(self, size, granularity):
        self.record("reserve", size, granularity)
        return 4096

    def map(self, pointer, size, handle):
        self.record("map", pointer, size, handle)

    def set_access(self, pointer, size, device):
        self.record("access", pointer, size, device)

    def zero(self, pointer, size):
        self.record("zero", pointer, size)

    def synchronize(self):
        self.record("sync")

    def export(self, handle):
        self.record("export", handle)
        return b"h" * 64

    def import_handle(self, value):
        if type(value) is not bytes or len(value) != 64:
            raise ValueError("exact64 bytes")
        self.record("import", value)
        return 91

    def unmap(self, pointer, size):
        self.record("unmap", pointer, size)

    def free_address(self, pointer, size):
        self.record("free", pointer, size)

    def release(self, handle):
        self.record("release", handle)


class SignatureFunction:
    """Model ctypes default-int conversion when a fresh function has no ABI."""

    def __init__(self, library, name):
        self.library, self.name = library, name
        self.argtypes = None
        self.restype = ctypes.c_int

    def __call__(self, *arguments):
        converted = []
        for index, value in enumerate(arguments):
            if isinstance(value, (int, ctypes.c_void_p)):
                scalar = value if isinstance(value, int) else value.value
                kind = ctypes.c_int if self.argtypes is None else self.argtypes[index]
                converted.append(kind(scalar).value)
            else:
                converted.append(value)
        self.library.calls.append((self.name, self, converted))
        if self.name == "nvmlDeviceGetHandleByUUID":
            arguments[1]._obj.value = self.library.high_handle
        elif self.name == "cuMemImportFromShareableHandle":
            arguments[0]._obj.value = self.library.high_handle
        elif self.name == "nvmlDeviceGetGpuFabricInfoV":
            info = arguments[1]._obj
            info.state, info.status, info.cliqueId = 3, 0, 0
            info.clusterUuid[:] = b"\x11" * 16
        return 0


class SignatureLibrary:
    """CDLL indexing returns fresh functions; attributes cache each function."""

    high_handle = 0x123456789ABC

    def __init__(self):
        self.functions, self.calls, self.index_lookups = {}, [], []

    def __getitem__(self, name):
        self.index_lookups.append(name)
        return SignatureFunction(self, name)

    def __getattr__(self, name):
        if name not in self.functions:
            self.functions[name] = SignatureFunction(self, name)
        return self.functions[name]


def test_cuda_driver_reuses_declared_signatures_for_high_64bit_addresses(monkeypatch):
    library = SignatureLibrary()
    monkeypatch.setattr(m.ctypes, "CDLL", lambda name: library)
    driver = m._CudaDriver()
    pointer, handle, extent = 0x123456789ABC, 0x23456789ABCD, 1 << 33
    driver.map(pointer, extent, handle)
    driver.free_address(pointer, extent)
    driver.release(handle)
    assert not library.index_lookups
    by_name = {
        name: (function, arguments) for name, function, arguments in library.calls
    }
    assert by_name["cuMemMap"][1] == [pointer, extent, 0, handle, 0]
    assert by_name["cuMemAddressFree"][1] == [pointer, extent]
    assert by_name["cuMemRelease"][1] == [handle]
    for name, (function, _) in by_name.items():
        assert function is library.functions[name]
        assert function.argtypes is not None


def test_nvml_observation_uses_configured_cached_functions_for_device_handle(
    monkeypatch,
):
    library = SignatureLibrary()
    monkeypatch.setattr(m.ctypes, "CDLL", lambda name: library)
    driver = FakeDriver()
    identity = m.observe_fabric(device=0, driver=driver)
    assert identity == driver.identity
    assert not library.index_lookups
    query = [
        (function, values)
        for name, function, values in library.calls
        if name == "nvmlDeviceGetGpuFabricInfoV"
    ]
    assert len(query) == 1 and query[0][1][0] == library.high_handle
    for name, function, _ in library.calls:
        assert function is library.functions[name]
        assert function.argtypes is not None


def test_effective_domain_matches_same_clique_not_gpu_uuid_and_separates_other_profiles():
    first = m.FabricIdentity("GPU-00000000-0000-0000-0000-000000000001", "1" * 32, 0)
    second = m.FabricIdentity("GPU-00000000-0000-0000-0000-000000000002", "1" * 32, 0)
    domain = m.effective_domain("layout-v1", first.domain)
    assert domain == m.effective_domain("layout-v1", second.domain)
    assert domain != m.effective_domain("other-layout", first.domain)
    assert domain != m.effective_domain("layout-v1", "mnnvl:" + "2" * 32 + ":0")
    assert domain != m.effective_domain("layout-v1", "mnnvl:" + "1" * 32 + ":1")
    assert domain != "layout-v1"


@pytest.mark.parametrize(
    "domain",
    [
        "mnnvl:" + "0" * 32 + ":0",
        "mnnvl:" + "1" * 32 + ":00",
        "mnnvl:" + "1" * 32 + ":-1",
        "mnnvl:" + "1" * 32 + ":4294967296",
        "tcp",
        None,
    ],
)
def test_uncanonical_or_missing_fabric_domain_is_not_a_capability(domain):
    with pytest.raises(ValueError):
        m.effective_domain("layout-v1", domain)


def test_ctypes_layout_matches_pinned_cuda_and_nvml_struct_declarations():
    assert ctypes.sizeof(m._AllocationProperties) == 32
    assert ctypes.sizeof(m._Access) == 12
    assert ctypes.sizeof(m._FabricInfo) == 36


@pytest.mark.parametrize("layout_domain", ["bad\x00layout", "", " padded ", None])
def test_invalid_raw_layout_domain_cannot_be_hidden_by_compatibility_hash(
    layout_domain,
):
    with pytest.raises(ValueError):
        m.effective_domain(layout_domain, "mnnvl:" + "1" * 32 + ":0")
