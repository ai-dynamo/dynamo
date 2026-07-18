# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA driver helpers shared by the GMS client and server."""

from __future__ import annotations

import os
import threading
from typing import Any

from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.snapshot_profile import SnapshotProfile
from gpu_memory_service.common.utils import fail

try:
    from cuda.bindings import driver as cuda
except ImportError:
    # Keep import-time collection working in CPU-only environments and let the
    # first real CUDA call fail with a targeted message instead.
    class _MissingCuda:
        def __getattr__(self, name):
            raise RuntimeError(
                "cuda-python is required for GPU Memory Service CUDA operations"
            )

    cuda = _MissingCuda()

try:
    from cuda.bindings import runtime as cuda_runtime
except ImportError:
    # Keep CPU-only import/collection working. Runtime calls fail with a
    # targeted message when the sharded-SSD backend is actually used.
    class _MissingCudaRuntime:
        def __getattr__(self, name):
            raise RuntimeError(
                "cuda-python is required for GPU Memory Service CUDA runtime operations"
            )

    cuda_runtime = _MissingCudaRuntime()

_cuda_runtime_calls_forbidden = False


def list_devices() -> list[int]:
    """Return list of CUDA device indices visible to this process via NVML."""
    import pynvml

    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
    finally:
        pynvml.nvmlShutdown()
    if count == 0:
        raise SystemExit("no nvidia devices found")
    return list(range(count))


def list_device_uuids() -> list[str]:
    """Return the full physical UUID of each GPU visible through NVML."""
    import pynvml

    pynvml.nvmlInit()
    try:
        uuids = []
        for device in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuids.append(uuid.decode() if isinstance(uuid, bytes) else str(uuid))
    finally:
        pynvml.nvmlShutdown()
    if not uuids:
        raise SystemExit("no nvidia devices found")
    return uuids


def device_memory_info(
    device: int,
    *,
    device_uuid: str | None = None,
) -> tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for a CUDA device via NVML."""
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = (
            pynvml.nvmlDeviceGetHandleByUUID(device_uuid)
            if device_uuid is not None
            else pynvml.nvmlDeviceGetHandleByIndex(device)
        )
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.free), int(info.total)
    finally:
        pynvml.nvmlShutdown()


def cuda_check_result(result: cuda.CUresult, name: str) -> None:
    if result != cuda.CUresult.CUDA_SUCCESS:
        err_result, err_str = cuda.cuGetErrorString(result)
        if err_result == cuda.CUresult.CUDA_SUCCESS and err_str:
            err_msg = err_str.decode() if isinstance(err_str, bytes) else str(err_str)
        else:
            err_msg = str(result)
        fail("fatal CUDA VMM error in %s: %s", name, err_msg)


def cuda_ensure_initialized() -> None:
    (result,) = cuda.cuInit(0)
    cuda_check_result(result, "cuInit")


def cuda_current_context() -> int:
    """Return the calling thread's current CUDA Driver context."""
    result, context = cuda.cuCtxGetCurrent()
    cuda_check_result(result, "cuCtxGetCurrent")
    return int(context)


def cumem_get_allocation_granularity(device: int) -> int:
    """Get VMM allocation granularity for a device.

    Args:
        device: CUDA device index

    Returns:
        Allocation granularity in bytes (typically 2 MiB)
    """
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = (
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )

    result, granularity = cuda.cuMemGetAllocationGranularity(
        prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    )
    cuda_check_result(result, "cuMemGetAllocationGranularity")
    return int(granularity)


def cumem_create_tolerate_oom(size: int, device: int) -> tuple[bool, int]:
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = (
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )

    result, handle = cuda.cuMemCreate(size, prop, 0)
    if result == cuda.CUresult.CUDA_SUCCESS:
        return True, int(handle)
    if result == cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
        return False, 0
    cuda_check_result(result, "cuMemCreate")
    return False, 0


def cumem_export_to_shareable_handle(handle: int) -> int:
    result, fd = cuda.cuMemExportToShareableHandle(
        handle,
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        0,
    )
    cuda_check_result(result, "cuMemExportToShareableHandle")
    return int(fd)


def align_to_granularity(size: int, granularity: int) -> int:
    """Align size up to VMM granularity.

    Args:
        size: Size in bytes
        granularity: Allocation granularity

    Returns:
        Aligned size
    """
    return ((size + granularity - 1) // granularity) * granularity


def cumem_import_from_shareable_handle_close_fd(fd: int) -> int:
    try:
        result, handle = cuda.cuMemImportFromShareableHandle(
            fd,
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
        cuda_check_result(result, "cuMemImportFromShareableHandle")
        return int(handle)
    finally:
        os.close(fd)


def cumem_address_reserve(size: int, granularity: int) -> int:
    result, va = cuda.cuMemAddressReserve(size, granularity, 0, 0)
    cuda_check_result(result, "cuMemAddressReserve")
    return int(va)


def cumem_address_free(va: int, size: int) -> None:
    (result,) = cuda.cuMemAddressFree(va, size)
    cuda_check_result(result, "cuMemAddressFree")


def cumem_map(va: int, size: int, handle: int) -> None:
    (result,) = cuda.cuMemMap(va, size, 0, handle, 0)
    cuda_check_result(result, "cuMemMap")


def cumem_set_access(va: int, size: int, device: int, access: GrantedLockType) -> None:
    access_desc = cuda.CUmemAccessDesc()
    access_desc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access_desc.location.id = device
    access_desc.flags = (
        cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
        if access == GrantedLockType.RO
        else cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    )
    (result,) = cuda.cuMemSetAccess(va, size, [access_desc], 1)
    cuda_check_result(result, "cuMemSetAccess")


def cumem_unmap(va: int, size: int) -> None:
    (result,) = cuda.cuMemUnmap(va, size)
    cuda_check_result(result, "cuMemUnmap")


def cumem_release(handle: int) -> None:
    (result,) = cuda.cuMemRelease(handle)
    cuda_check_result(result, "cuMemRelease")


def cuda_validate_pointer(va: int) -> None:
    result, _ = cuda.cuPointerGetAttribute(
        cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_POINTER, va
    )
    cuda_check_result(result, "cuPointerGetAttribute")


def cuda_synchronize() -> None:
    (result,) = cuda.cuCtxSynchronize()
    cuda_check_result(result, "cuCtxSynchronize")


def forbid_cuda_runtime_calls() -> None:
    """Fail closed if a GMS CUDA Runtime wrapper is reached."""
    global _cuda_runtime_calls_forbidden
    _cuda_runtime_calls_forbidden = True


def _ensure_cuda_runtime_calls_allowed(name: str) -> None:
    if _cuda_runtime_calls_forbidden:
        raise RuntimeError(
            f"GMS CUDA Runtime call {name} is forbidden in Driver-only loader mode"
        )


def cuda_runtime_check_result(result, name: str):
    if isinstance(result, tuple):
        code = result[0]
        payload = result[1:]
    else:
        code = result
        payload = ()
    if code == cuda_runtime.cudaError_t.cudaSuccess:
        if len(payload) == 1:
            return payload[0]
        return payload

    message_result = cuda_runtime.cudaGetErrorString(code)
    if isinstance(message_result, tuple):
        message = message_result[1] if len(message_result) > 1 else message_result[0]
    else:
        message = message_result
    if isinstance(message, bytes):
        err_msg = message.decode("utf-8", errors="replace")
    else:
        err_msg = str(message)
    raise RuntimeError(f"CUDA runtime error in {name}: {err_msg}")


def cuda_runtime_set_device(device: int) -> None:
    _ensure_cuda_runtime_calls_allowed("cudaSetDevice")
    cuda_runtime_check_result(
        cuda_runtime.cudaSetDevice(device),
        f"cudaSetDevice({device})",
    )


def cuda_host_register(ptr: int, size: int) -> None:
    _ensure_cuda_runtime_calls_allowed("cudaHostRegister")
    cuda_runtime_check_result(
        cuda_runtime.cudaHostRegister(ptr, size, 0),
        "cudaHostRegister",
    )


def cuda_host_unregister(ptr: int) -> None:
    _ensure_cuda_runtime_calls_allowed("cudaHostUnregister")
    cuda_runtime_check_result(
        cuda_runtime.cudaHostUnregister(ptr), "cudaHostUnregister"
    )


def cuda_stream_create_nonblocking():
    _ensure_cuda_runtime_calls_allowed("cudaStreamCreateWithFlags")
    flag = getattr(cuda_runtime, "cudaStreamNonBlocking", 1)
    return cuda_runtime_check_result(
        cuda_runtime.cudaStreamCreateWithFlags(flag),
        "cudaStreamCreateWithFlags",
    )


def cuda_stream_destroy(stream) -> None:
    _ensure_cuda_runtime_calls_allowed("cudaStreamDestroy")
    cuda_runtime_check_result(
        cuda_runtime.cudaStreamDestroy(stream), "cudaStreamDestroy"
    )


def cuda_stream_synchronize(stream) -> None:
    _ensure_cuda_runtime_calls_allowed("cudaStreamSynchronize")
    cuda_runtime_check_result(
        cuda_runtime.cudaStreamSynchronize(stream),
        "cudaStreamSynchronize",
    )


def cuda_event_create():
    _ensure_cuda_runtime_calls_allowed("cudaEventCreate")
    return cuda_runtime_check_result(
        cuda_runtime.cudaEventCreate(),
        "cudaEventCreate",
    )


def cuda_event_record(event, stream) -> None:
    _ensure_cuda_runtime_calls_allowed("cudaEventRecord")
    cuda_runtime_check_result(
        cuda_runtime.cudaEventRecord(event, stream),
        "cudaEventRecord",
    )


def cuda_event_elapsed_ns(start_event, end_event) -> int:
    _ensure_cuda_runtime_calls_allowed("cudaEventElapsedTime")
    milliseconds = cuda_runtime_check_result(
        cuda_runtime.cudaEventElapsedTime(start_event, end_event),
        "cudaEventElapsedTime",
    )
    return int(float(milliseconds) * 1_000_000)


def cuda_event_destroy(event) -> None:
    _ensure_cuda_runtime_calls_allowed("cudaEventDestroy")
    cuda_runtime_check_result(
        cuda_runtime.cudaEventDestroy(event),
        "cudaEventDestroy",
    )


def cuda_memcpy_h2d_async(
    dst_ptr: int,
    src_ptr: int,
    size: int,
    stream,
) -> None:
    _ensure_cuda_runtime_calls_allowed("cudaMemcpyAsync")
    cuda_runtime_check_result(
        cuda_runtime.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            size,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        ),
        "cudaMemcpyAsync",
    )


class RuntimeCudaTransferOperations:
    """Existing CUDA Runtime operations used by snapshot staging."""

    api = "runtime"

    def __init__(self, profile: SnapshotProfile | None = None) -> None:
        self._profile = profile or SnapshotProfile("loader", enabled=False)
        self._first_h2d_submitted = False
        self._first_h2d_lock = threading.Lock()

    @staticmethod
    def set_current_device(device: int) -> None:
        cuda_runtime_set_device(device)

    @staticmethod
    def host_register(ptr: int, size: int) -> None:
        cuda_host_register(ptr, size)

    @staticmethod
    def host_unregister(ptr: int) -> None:
        cuda_host_unregister(ptr)

    @staticmethod
    def stream_create_nonblocking():
        return cuda_stream_create_nonblocking()

    @staticmethod
    def stream_destroy(stream) -> None:
        cuda_stream_destroy(stream)

    @staticmethod
    def stream_synchronize(stream) -> None:
        cuda_stream_synchronize(stream)

    @staticmethod
    def event_create():
        return cuda_event_create()

    @staticmethod
    def event_record(event, stream) -> None:
        cuda_event_record(event, stream)

    @staticmethod
    def event_elapsed_ns(start_event, end_event) -> int:
        return cuda_event_elapsed_ns(start_event, end_event)

    @staticmethod
    def event_destroy(event) -> None:
        cuda_event_destroy(event)

    def memcpy_h2d_async(
        self,
        dst_ptr: int,
        src_ptr: int,
        size: int,
        stream,
    ) -> None:
        with self._first_h2d_lock:
            first = not self._first_h2d_submitted
            self._first_h2d_submitted = True
        span = (
            self._profile.phase(
                "first_h2d_submission",
                api="cudaMemcpyAsync",
                cuda_api=self.api,
                byte_count=size,
            )
            if first
            else None
        )
        if span is None:
            cuda_memcpy_h2d_async(dst_ptr, src_ptr, size, stream)
        else:
            with span:
                cuda_memcpy_h2d_async(dst_ptr, src_ptr, size, stream)

    @staticmethod
    def memcpy_d2h_async(
        dst_ptr: int,
        src_ptr: int,
        size: int,
        stream,
    ) -> None:
        cuda_memcpy_d2h_async(dst_ptr, src_ptr, size, stream)


RUNTIME_CUDA_TRANSFER_OPERATIONS = RuntimeCudaTransferOperations()


def cuda_driver_check_result(result: Any, name: str) -> None:
    """Raise a recoverable error for a failed CUDA Driver operation."""
    if result == cuda.CUresult.CUDA_SUCCESS:
        return
    err_result, err_str = cuda.cuGetErrorString(result)
    if err_result == cuda.CUresult.CUDA_SUCCESS and err_str:
        err_msg = err_str.decode() if isinstance(err_str, bytes) else str(err_str)
    else:
        err_msg = str(result)
    raise RuntimeError(f"CUDA Driver error in {name}: {err_msg}")


class DriverCudaProcess:
    """Process-scoped Driver initialization and retained primary contexts."""

    def __init__(self) -> None:
        self._initialized = False
        self._devices: dict[int, tuple[Any, Any]] = {}
        self._lock = threading.Lock()

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            (result,) = cuda.cuInit(0)
            cuda_driver_check_result(result, "cuInit")
            self._initialized = True

    def device_get(self, ordinal: int):
        if not self._initialized:
            raise RuntimeError("CUDA Driver process is not initialized")
        result, device = cuda.cuDeviceGet(ordinal)
        cuda_driver_check_result(result, f"cuDeviceGet({ordinal})")
        return device

    def primary_context_retain(self, ordinal: int, device) -> Any:
        with self._lock:
            if ordinal in self._devices:
                raise RuntimeError(
                    f"CUDA primary context for device {ordinal} was already retained"
                )
        result, context = cuda.cuDevicePrimaryCtxRetain(device)
        cuda_driver_check_result(
            result,
            f"cuDevicePrimaryCtxRetain({ordinal})",
        )
        with self._lock:
            self._devices[ordinal] = (device, context)
        return context

    def set_current(self, ordinal: int) -> int:
        with self._lock:
            try:
                _device, context = self._devices[ordinal]
            except KeyError as error:
                raise RuntimeError(
                    f"CUDA primary context for device {ordinal} is not retained"
                ) from error
        (result,) = cuda.cuCtxSetCurrent(context)
        cuda_driver_check_result(result, f"cuCtxSetCurrent(device={ordinal})")
        return int(context)

    def operations(
        self,
        ordinal: int,
        profile: SnapshotProfile,
    ) -> "DriverCudaTransferOperations":
        return DriverCudaTransferOperations(self, ordinal, profile)

    def close(self) -> None:
        with self._lock:
            retained = list(self._devices.items())
            self._devices.clear()
        first_error = None
        for ordinal, (device, _context) in retained:
            try:
                (result,) = cuda.cuDevicePrimaryCtxRelease(device)
                cuda_driver_check_result(
                    result,
                    f"cuDevicePrimaryCtxRelease({ordinal})",
                )
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


class DriverCudaTransferOperations:
    """CUDA Driver equivalents for all pinned snapshot transfer operations."""

    api = "driver"

    def __init__(
        self,
        process: DriverCudaProcess,
        device: int,
        profile: SnapshotProfile,
    ) -> None:
        self._process = process
        self._device = device
        self._profile = profile
        self._first_h2d_submitted = False
        self._first_h2d_lock = threading.Lock()

    def set_current_device(self, device: int) -> int:
        if device != self._device:
            raise RuntimeError(
                f"Driver transfer operations for device {self._device} "
                f"cannot select device {device}"
            )
        return self._process.set_current(device)

    @staticmethod
    def host_register(ptr: int, size: int) -> None:
        (result,) = cuda.cuMemHostRegister(ptr, size, 0)
        cuda_driver_check_result(result, "cuMemHostRegister")

    @staticmethod
    def host_unregister(ptr: int) -> None:
        (result,) = cuda.cuMemHostUnregister(ptr)
        cuda_driver_check_result(result, "cuMemHostUnregister")

    @staticmethod
    def stream_create_nonblocking():
        result, stream = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING)
        cuda_driver_check_result(result, "cuStreamCreate")
        return stream

    @staticmethod
    def stream_destroy(stream) -> None:
        (result,) = cuda.cuStreamDestroy(stream)
        cuda_driver_check_result(result, "cuStreamDestroy")

    @staticmethod
    def stream_synchronize(stream) -> None:
        (result,) = cuda.cuStreamSynchronize(stream)
        cuda_driver_check_result(result, "cuStreamSynchronize")

    @staticmethod
    def event_create():
        result, event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)
        cuda_driver_check_result(result, "cuEventCreate")
        return event

    @staticmethod
    def event_record(event, stream) -> None:
        (result,) = cuda.cuEventRecord(event, stream)
        cuda_driver_check_result(result, "cuEventRecord")

    @staticmethod
    def event_elapsed_ns(start_event, end_event) -> int:
        result, milliseconds = cuda.cuEventElapsedTime(start_event, end_event)
        cuda_driver_check_result(result, "cuEventElapsedTime")
        return int(float(milliseconds) * 1_000_000)

    @staticmethod
    def event_destroy(event) -> None:
        (result,) = cuda.cuEventDestroy(event)
        cuda_driver_check_result(result, "cuEventDestroy")

    def memcpy_h2d_async(
        self,
        dst_ptr: int,
        src_ptr: int,
        size: int,
        stream,
    ) -> None:
        with self._first_h2d_lock:
            first = not self._first_h2d_submitted
            self._first_h2d_submitted = True
        span = (
            self._profile.phase(
                "first_h2d_submission",
                api="cuMemcpyHtoDAsync",
                cuda_api=self.api,
                byte_count=size,
            )
            if first
            else None
        )
        if span is None:
            self._memcpy_h2d_async(dst_ptr, src_ptr, size, stream)
        else:
            with span:
                self._memcpy_h2d_async(dst_ptr, src_ptr, size, stream)

    @staticmethod
    def _memcpy_h2d_async(
        dst_ptr: int,
        src_ptr: int,
        size: int,
        stream,
    ) -> None:
        (result,) = cuda.cuMemcpyHtoDAsync(dst_ptr, src_ptr, size, stream)
        cuda_driver_check_result(result, "cuMemcpyHtoDAsync")

    @staticmethod
    def memcpy_d2h_async(
        dst_ptr: int,
        src_ptr: int,
        size: int,
        stream,
    ) -> None:
        (result,) = cuda.cuMemcpyDtoHAsync(dst_ptr, src_ptr, size, stream)
        cuda_driver_check_result(result, "cuMemcpyDtoHAsync")


def cuda_memcpy_d2h_async(
    dst_ptr: int,
    src_ptr: int,
    size: int,
    stream,
) -> None:
    _ensure_cuda_runtime_calls_allowed("cudaMemcpyAsync")
    cuda_runtime_check_result(
        cuda_runtime.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            size,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        ),
        "cudaMemcpyAsync",
    )
