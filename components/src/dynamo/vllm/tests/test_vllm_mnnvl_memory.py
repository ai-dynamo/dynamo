# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-GPU VMM ownership, admission and rollback tests with a fake CUDA driver."""

import ctypes
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

# This helper is stdlib-only. Marker collection must not import the optional
# KVBM/vLLM packages. Flat component staging resolves the installed package path
# without executing its __init__; missing or broken source still fails loudly.
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


def test_driver_import_requests_exact_fabric_kind_and_64_handle_bytes(monkeypatch):
    library = SignatureLibrary()
    monkeypatch.setattr(m.ctypes, "CDLL", lambda name: library)
    driver = m._CudaDriver()
    assert driver.import_handle(b"h" * 64) == library.high_handle
    imported = [
        values
        for name, _, values in library.calls
        if name == "cuMemImportFromShareableHandle"
    ]
    assert len(imported) == 1
    assert imported[0][2] == 8
    assert bytes(imported[0][1]._obj) == b"h" * 64
    for invalid in (b"h" * 63, b"h" * 65, "h" * 64):
        with pytest.raises(ValueError):
            driver.import_handle(invalid)
    assert (
        sum(name == "cuMemImportFromShareableHandle" for name, _, _ in library.calls)
        == 1
    )


@pytest.fixture
def driver(monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(m, "observe_fabric", lambda *args, **kwargs: driver.identity)
    monkeypatch.setattr(m, "_RETAINED_FAILED_FABRIC_ALLOCATIONS", [])
    return driver


def test_vmm_allocates_only_device_fabric_and_drains_before_last_release(driver):
    owner = m.FabricAllocation((3, 3), driver=driver)
    assert owner.logical_bytes == 9 and owner.allocation_bytes == 16
    assert ("granularity", 1, 8, 1) in driver.calls
    assert ("create", 16, 8) in driver.calls
    assert ("map", 4096, 16, 90) in driver.calls
    assert ("access", 4096, 16, 0) in driver.calls
    assert owner.fabric_handle == b"h" * 64
    interface = owner.__cuda_array_interface__
    assert interface == {
        "shape": (3, 3),
        "strides": (3, 1),
        "typestr": "|i1",
        "data": (4096, False),
        "version": 3,
    }
    driver.calls.clear()
    del owner
    assert driver.calls == [
        ("push", 77),
        ("sync",),
        ("unmap", 4096, 16),
        ("free", 4096, 16),
        ("release", 90),
        ("pop", 77),
        ("release-context", 0),
    ]


def test_import_does_not_zero_or_reexport_the_shared_bank(driver):
    owner = m.FabricAllocation(
        (3, 3), driver=driver, imported_handle=b"z" * 64, allocation_bytes=16
    )
    assert owner.fabric_handle == b"z" * 64
    names = [r[0] for r in driver.calls]
    assert "import" in names and "create" not in names
    assert "zero" not in names and "export" not in names
    del owner


@pytest.mark.parametrize(
    "shape", [(0, 1), (1, 0), (True, 2), (1 << 63, 2), [1, 2], (1,)]
)
def test_bad_extent_fails_before_loading_native_libraries(monkeypatch, shape):
    monkeypatch.setattr(
        m, "_CudaDriver", lambda: pytest.fail("No native library may load")
    )
    with pytest.raises(ValueError):
        m.FabricAllocation(shape)


@pytest.mark.parametrize(
    "stage", ["create", "reserve", "map", "access", "zero", "sync", "export"]
)
def test_partial_allocation_failure_releases_only_resources_actually_acquired(
    driver, stage
):
    driver.fail = stage
    with pytest.raises(RuntimeError, match="injected " + stage):
        m.FabricAllocation((3, 3), driver=driver)
    names = [r[0] for r in driver.calls]
    assert names[-1] == "release-context"
    if stage == "create":
        assert "release" not in names and "free" not in names
    elif stage == "reserve":
        assert "release" in names and "free" not in names
    elif stage == "map":
        assert "free" in names and "release" in names and "unmap" not in names
    else:
        assert names.index("unmap") < names.index("free") < names.index("release")


@pytest.mark.parametrize("kind", ["host", "not-pinned", "bad-import-size"])
def test_import_admission_rejects_wrong_physical_properties_or_extent(driver, kind):
    if kind == "host":
        driver.properties_from_handle.location.type = 2
    elif kind == "not-pinned":
        driver.properties_from_handle.type = 0
    with pytest.raises((ValueError, RuntimeError)):
        m.FabricAllocation(
            (3, 3),
            driver=driver,
            imported_handle=b"h" * 64,
            allocation_bytes=8 if kind == "bad-import-size" else 16,
        )
    assert "map" not in [r[0] for r in driver.calls]


@pytest.mark.parametrize("flags", [0, 1])
def test_created_bank_without_actual_fabric_capability_rejects_before_mapping(
    driver, flags
):
    driver.properties_from_handle.requestedHandleTypes = flags
    with pytest.raises(RuntimeError, match="DEVICE/FABRIC"):
        m.FabricAllocation((3, 3), driver=driver)
    assert "map" not in [row[0] for row in driver.calls]
    assert ("release", 90) in driver.calls


def test_successful_typed_import_accepts_device_handle_with_cleared_export_flags(
    driver,
):
    driver.properties_from_handle.requestedHandleTypes = 0
    owner = m.FabricAllocation(
        (3, 3), driver=driver, imported_handle=b"h" * 64, allocation_bytes=16
    )
    assert ("import", b"h" * 64) in driver.calls
    assert ("map", 4096, 16, 91) in driver.calls
    assert not {"create", "export", "zero"} & {row[0] for row in driver.calls}
    del owner


def test_failed_typed_import_never_maps_or_falls_back_to_other_storage(driver):
    driver.properties_from_handle.requestedHandleTypes = 0
    driver.fail = "import"
    with pytest.raises(RuntimeError, match="injected import"):
        m.FabricAllocation(
            (3, 3), driver=driver, imported_handle=b"h" * 64, allocation_bytes=16
        )
    assert not {"map", "create", "export", "zero"} & {row[0] for row in driver.calls}


def test_failed_local_quiescence_retains_live_mapping_without_forced_reuse(driver):
    owner = m.FabricAllocation((1, 8), driver=driver)
    driver.calls.clear()
    driver.fail = "sync"
    del owner
    assert len(m._RETAINED_FAILED_FABRIC_ALLOCATIONS) == 1
    retained = m._RETAINED_FAILED_FABRIC_ALLOCATIONS[0]
    assert retained._mapped and retained.pointer == 4096 and retained.handle == 90
    assert not {"unmap", "free", "release", "release-context"} & {
        r[0] for r in driver.calls
    }


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
