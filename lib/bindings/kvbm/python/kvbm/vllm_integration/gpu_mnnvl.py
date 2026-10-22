# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in FABRIC VMM storage; importing this module does not load CUDA or NVML.

Only allocation handles/addresses use host memory. KV bytes are mapped DEVICE
memory. Each PyTorch CUDA-array-interface storage retains its allocation owner;
the owner must outlive every local view and all native transfer leases.
"""

import ctypes
import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)
_FABRIC = 8
_DEVICE = 1
_PINNED = 1
_READ_WRITE = 3
_FABRIC_HANDLE_BYTES = 64
_MAX_BYTES = min((1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1, (1 << 63) - 1)
_RETAINED_FAILED_FABRIC_ALLOCATIONS = []


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocationProperties(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocationFlags),
    ]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


class _FabricInfo(ctypes.Structure):
    # NVML nvmlGpuFabricInfo_v2_t, also present in nvidia-ml-py13.580.65.
    _fields_ = [
        ("version", ctypes.c_uint),
        ("clusterUuid", ctypes.c_ubyte * 16),
        ("status", ctypes.c_uint),
        ("cliqueId", ctypes.c_uint),
        ("state", ctypes.c_uint),
        ("healthMask", ctypes.c_uint),
    ]


def _gpu_uuid(value):
    if not isinstance(value, str):
        raise ValueError("A physical GPU UUID is required")
    return "GPU-" + str(uuid.UUID(value.removeprefix("GPU-")))


@dataclass(frozen=True)
class FabricIdentity:
    gpu_uuid: str
    cluster_uuid: str
    clique_id: int
    health_mask: int = 0

    def __post_init__(self):
        if self.gpu_uuid != _gpu_uuid(self.gpu_uuid):
            raise ValueError("Canonical physical GPU UUID required")
        if (
            not isinstance(self.cluster_uuid, str)
            or len(self.cluster_uuid) != 32
            or any(c not in "0123456789abcdef" for c in self.cluster_uuid)
            or int(self.cluster_uuid, 16) == 0
            or type(self.clique_id) is not int
            or not 0 <= self.clique_id <= 0xFFFFFFFF
        ):
            raise ValueError("Completed nonzero MNNVL cluster and u32 clique required")

    @property
    def domain(self):
        return f"mnnvl:{self.cluster_uuid}:{self.clique_id}"


def effective_domain(layout_domain, fabric_domain):
    """Bind the existing exact router domain to layout, transport and fabric."""
    if (
        not isinstance(layout_domain, str)
        or not layout_domain
        or layout_domain != layout_domain.strip()
        or "\x00" in layout_domain
    ):
        raise ValueError("Explicit unpadded certified layout domain required")
    parts = fabric_domain.split(":") if isinstance(fabric_domain, str) else []
    if (
        len(parts) != 3
        or parts[0] != "mnnvl"
        or not parts[2].isascii()
        or not parts[2].isdecimal()
    ):
        raise ValueError("Canonical observed MNNVL fabric domain required")
    identity = FabricIdentity(
        "GPU-00000000-0000-0000-0000-000000000001", parts[1], int(parts[2])
    )
    if identity.domain != fabric_domain:
        raise ValueError("Canonical observed MNNVL fabric domain required")
    body = {
        "schema": 1,
        "layout_domain": layout_domain,
        "gpu_transport": "mnnvl",
        "fabric_domain": fabric_domain,
    }
    return (
        "mnnvl-sha256:"
        + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _shape_bytes(shape):
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(type(n) is not int or n <= 0 for n in shape)
    ):
        raise ValueError("Two positive integer GPU bank dimensions required")
    count = shape[0] * shape[1]
    if count > _MAX_BYTES:
        raise ValueError("GPU bank extent overflows the driver/Torch range")
    return count


class _CudaDriver:
    def __init__(self):
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("MNNVL VMM requires a64-bit CUDA process")
        self.lib = ctypes.CDLL("libcuda.so.1")
        ptr, size, handle, device = (
            ctypes.c_ulonglong,
            ctypes.c_size_t,
            ctypes.c_ulonglong,
            ctypes.c_int,
        )
        signatures = {
            "cuInit": [ctypes.c_uint],
            "cuDeviceGet": [ctypes.POINTER(device), ctypes.c_int],
            "cuCtxGetDevice": [ctypes.POINTER(device)],
            "cuDeviceGetUuid_v2": [ctypes.c_void_p, device],
            "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), device],
            "cuDevicePrimaryCtxRelease_v2": [device],
            "cuCtxPushCurrent_v2": [ctypes.c_void_p],
            "cuCtxPopCurrent_v2": [ctypes.POINTER(ctypes.c_void_p)],
            "cuCtxSynchronize": [],
            "cuMemGetAllocationGranularity": [
                ctypes.POINTER(size),
                ctypes.POINTER(_AllocationProperties),
                ctypes.c_int,
            ],
            "cuMemCreate": [
                ctypes.POINTER(handle),
                size,
                ctypes.POINTER(_AllocationProperties),
                ctypes.c_ulonglong,
            ],
            "cuMemGetAllocationPropertiesFromHandle": [
                ctypes.POINTER(_AllocationProperties),
                handle,
            ],
            "cuMemAddressReserve": [
                ctypes.POINTER(ptr),
                size,
                size,
                ptr,
                ctypes.c_ulonglong,
            ],
            "cuMemMap": [ptr, size, size, handle, ctypes.c_ulonglong],
            "cuMemSetAccess": [ptr, size, ctypes.POINTER(_Access), size],
            "cuMemsetD8_v2": [ptr, ctypes.c_ubyte, size],
            "cuMemExportToShareableHandle": [
                ctypes.c_void_p,
                handle,
                ctypes.c_int,
                ctypes.c_ulonglong,
            ],
            "cuMemImportFromShareableHandle": [
                ctypes.POINTER(handle),
                ctypes.c_void_p,
                ctypes.c_int,
            ],
            "cuMemUnmap": [ptr, size],
            "cuMemAddressFree": [ptr, size],
            "cuMemRelease": [handle],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self.call("cuInit", 0)

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result != 0:
            raise RuntimeError(
                f"MNNVL CUDA {name} failed ({result}); FABRIC/IMEX access is required; no fallback"
            )

    def device(self):
        value = ctypes.c_int()
        result = self.lib.cuCtxGetDevice(ctypes.byref(value))
        if result == 201:  # No current context in the TP1 registration process.
            self.call("cuDeviceGet", ctypes.byref(value), 0)
        elif result != 0:
            raise RuntimeError(f"MNNVL current CUDA device query failed ({result})")
        return value.value

    def gpu_uuid(self, device):
        value = (ctypes.c_ubyte * 16)()
        self.call("cuDeviceGetUuid_v2", ctypes.byref(value), device)
        return "GPU-" + str(uuid.UUID(bytes=bytes(value)))

    def retain_context(self, device):
        value = ctypes.c_void_p()
        self.call("cuDevicePrimaryCtxRetain", ctypes.byref(value), device)
        return value.value

    @contextmanager
    def context(self, context):
        self.call("cuCtxPushCurrent_v2", context)
        try:
            yield
        finally:
            previous = ctypes.c_void_p()
            self.call("cuCtxPopCurrent_v2", ctypes.byref(previous))

    def release_context(self, device):
        self.call("cuDevicePrimaryCtxRelease_v2", device)

    def properties(self, device):
        value = _AllocationProperties()
        value.type, value.requestedHandleTypes = _PINNED, _FABRIC
        value.location.type, value.location.id = _DEVICE, device
        return value

    def granularity(self, properties):
        value = ctypes.c_size_t()
        self.call(
            "cuMemGetAllocationGranularity",
            ctypes.byref(value),
            ctypes.byref(properties),
            0,
        )
        return value.value

    def create(self, size, properties):
        value = ctypes.c_ulonglong()
        self.call("cuMemCreate", ctypes.byref(value), size, ctypes.byref(properties), 0)
        return value.value

    def allocation_properties(self, handle):
        value = _AllocationProperties()
        self.call("cuMemGetAllocationPropertiesFromHandle", ctypes.byref(value), handle)
        return value

    def reserve(self, size, granularity):
        value = ctypes.c_ulonglong()
        self.call("cuMemAddressReserve", ctypes.byref(value), size, granularity, 0, 0)
        return value.value

    def map(self, pointer, size, handle):
        self.call("cuMemMap", pointer, size, 0, handle, 0)

    def set_access(self, pointer, size, device):
        value = _Access(_Location(_DEVICE, device), _READ_WRITE)
        self.call("cuMemSetAccess", pointer, size, ctypes.byref(value), 1)

    def zero(self, pointer, size):
        self.call("cuMemsetD8_v2", pointer, 0, size)

    def synchronize(self):
        self.call("cuCtxSynchronize")

    def export(self, handle):
        value = (ctypes.c_ubyte * _FABRIC_HANDLE_BYTES)()
        self.call(
            "cuMemExportToShareableHandle", ctypes.byref(value), handle, _FABRIC, 0
        )
        return bytes(value)

    def import_handle(self, value):
        if not isinstance(value, bytes) or len(value) != _FABRIC_HANDLE_BYTES:
            raise ValueError("Exactly64 opaque FABRIC handle bytes required")
        raw = (ctypes.c_ubyte * _FABRIC_HANDLE_BYTES).from_buffer_copy(value)
        handle = ctypes.c_ulonglong()
        self.call(
            "cuMemImportFromShareableHandle",
            ctypes.byref(handle),
            ctypes.byref(raw),
            _FABRIC,
        )
        return handle.value

    def unmap(self, pointer, size):
        self.call("cuMemUnmap", pointer, size)

    def free_address(self, pointer, size):
        self.call("cuMemAddressFree", pointer, size)

    def release(self, handle):
        self.call("cuMemRelease", handle)


def observe_fabric(device=None, *, driver=None):
    """Observe actual assigned GPU/NVML fabric; device presence is not IMEX proof."""
    driver = _CudaDriver() if driver is None else driver
    device = driver.device() if device is None else device
    gpu = driver.gpu_uuid(device)
    library = ctypes.CDLL("libnvidia-ml.so.1")
    signatures = {
        "nvmlInit_v2": [],
        "nvmlShutdown": [],
        "nvmlDeviceGetHandleByUUID": [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
        "nvmlDeviceGetGpuFabricInfoV": [ctypes.c_void_p, ctypes.POINTER(_FabricInfo)],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, ctypes.c_uint
    if library.nvmlInit_v2() != 0:
        raise RuntimeError("NVML initialization is required for MNNVL")
    try:
        handle = ctypes.c_void_p()
        if library.nvmlDeviceGetHandleByUUID(gpu.encode(), ctypes.byref(handle)) != 0:
            raise RuntimeError("Actual assigned GPU UUID is absent from NVML")
        info = _FabricInfo()
        info.version = 0x02000024
        if (
            library.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(info)) != 0
            or info.state != 3
            or info.status != 0
        ):
            raise RuntimeError("MNNVL requires actual NVML fabric Completed/Success")
        return FabricIdentity(
            gpu, bytes(info.clusterUuid).hex(), int(info.cliqueId), int(info.healthMask)
        )
    finally:
        if library.nvmlShutdown() != 0:
            raise RuntimeError("NVML shutdown failed after MNNVL fabric observation")


class FabricAllocation:
    """One VMM mapping retained by every CUDA-array-interface tensor/view.

    No public forced-close exists: the final storage reference releases this
    owner. Native transfer owners must retain it until actual quiescence.
    """

    def __init__(
        self,
        shape,
        *,
        device=None,
        driver=None,
        imported_handle=None,
        allocation_bytes=None,
    ):
        self.pointer = self.handle = self.context_handle = 0
        self._mapped = self._released = False
        self._driver = None
        self.fabric_handle = None
        self.allocation_bytes = 0
        self.device = device
        self.shape = shape
        self.logical_bytes = _shape_bytes(shape)
        self._driver = _CudaDriver() if driver is None else driver
        self.device = self._driver.device() if device is None else device
        self.identity = observe_fabric(self.device, driver=self._driver)
        if self._driver.gpu_uuid(self.device) != self.identity.gpu_uuid:
            raise ValueError("VMM bank must use its actual observed physical GPU")
        self.context_handle = self._driver.retain_context(self.device)
        try:
            with self._driver.context(self.context_handle):
                properties = self._driver.properties(self.device)
                granularity = self._driver.granularity(properties)
                if type(granularity) is not int or granularity <= 0:
                    raise RuntimeError(
                        "Positive CUDA VMM allocation granularity required"
                    )
                rounded = (
                    (self.logical_bytes + granularity - 1) // granularity
                ) * granularity
                if rounded > _MAX_BYTES:
                    raise ValueError(
                        "Rounded VMM bank overflows the driver/Torch range"
                    )
                self.allocation_bytes = rounded
                if imported_handle is None:
                    self.handle = self._driver.create(rounded, properties)
                else:
                    if type(allocation_bytes) is not int or allocation_bytes != rounded:
                        raise ValueError("Imported FABRIC bank rounded extent differs")
                    self.handle = self._driver.import_handle(imported_handle)
                actual = self._driver.allocation_properties(self.handle)
                if (
                    actual.type != _PINNED
                    or actual.location.type != _DEVICE
                    or (
                        imported_handle is None
                        and not actual.requestedHandleTypes & _FABRIC
                    )
                ):
                    raise RuntimeError(
                        "MNNVL bank requires actual DEVICE/FABRIC VMM storage"
                    )
                # CUDA clears export permissions on imported handles. Their
                # successful cuMemImport(..., FABRIC) above proves the handle
                # kind; physical PINNED/DEVICE admission remains mandatory.
                # New manager banks still require FABRIC and a real export.
                self.pointer = self._driver.reserve(rounded, granularity)
                self._driver.map(self.pointer, rounded, self.handle)
                self._mapped = True
                self._driver.set_access(self.pointer, rounded, self.device)
                if imported_handle is None:
                    self._driver.zero(self.pointer, rounded)
                self._driver.synchronize()
                self.fabric_handle = (
                    self._driver.export(self.handle)
                    if imported_handle is None
                    else imported_handle
                )
                if len(self.fabric_handle) != _FABRIC_HANDLE_BYTES:
                    raise RuntimeError("CUDA did not export a complete FABRIC handle")
        except Exception:
            self._release_or_retain()
            raise

    @property
    def __cuda_array_interface__(self):
        if self._released or not self._mapped:
            raise RuntimeError("FABRIC mapping is not live")
        return {
            "shape": self.shape,
            "strides": (self.shape[1], 1),
            "typestr": "|i1",
            "data": (self.pointer, False),
            "version": 3,
        }

    def _release(self):
        if self._released or self._driver is None:
            return
        if self.context_handle:
            with self._driver.context(self.context_handle):
                # Also protect queued local operations issued through a surviving
                # Tensor view. Remote DMA quiescence is the native owner's duty.
                self._driver.synchronize()
                if self._mapped:
                    self._driver.unmap(self.pointer, self.allocation_bytes)
                    self._mapped = False
                if self.pointer:
                    self._driver.free_address(self.pointer, self.allocation_bytes)
                    self.pointer = 0
                if self.handle:
                    self._driver.release(self.handle)
                    self.handle = 0
            self._driver.release_context(self.device)
            self.context_handle = 0
        self._released = True

    def _release_or_retain(self):
        try:
            self._release()
        except Exception:
            # A constructor still re-raises its original error. A finalizer cannot
            # propagate safely: resurrect the owner instead of losing a live map.
            _RETAINED_FAILED_FABRIC_ALLOCATIONS.append(self)
            logger.exception(
                "FABRIC mapping cleanup failed; retaining it until process exit"
            )

    def __del__(self):
        self._release_or_retain()
