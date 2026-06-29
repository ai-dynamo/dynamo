# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the daemon-side VmmAllocator.

The tests partition into two groups:

  - **No-CUDA tests** (always run): exercise the unsupported / lazy
    path. The allocator must:
      * report ``is_supported() == False`` cleanly when cuda-python
        isn't importable or the driver doesn't expose VMM
      * raise ``VmmUnsupportedError`` from ``allocate`` in that state
      * stay idempotent across ``release()`` of unknown ids

  - **CUDA-gated tests** (env knob ``GMS_TEST_VMM_REAL=1`` AND
    cuda-python importable AND the driver actually supports VMM):
    exercise the full cuMem* round trip.

The no-CUDA group uses ``monkeypatch`` to make the in-method import
of cuda-python fail or to inject a fake module."""

from __future__ import annotations

import os
import sys

import pytest
from gms_kv_ring.daemon.vmm_allocator import (
    VmmAllocation,
    VmmAllocator,
    VmmUnsupportedError,
    _round_up,
)

# ----------------------------------------------------------------------
# Small utility tests
# ----------------------------------------------------------------------


def test_round_up_zero():
    assert _round_up(0, 4096) == 0


def test_round_up_exact_multiple():
    assert _round_up(8192, 4096) == 8192


def test_round_up_below_multiple():
    assert _round_up(1, 4096) == 4096


def test_round_up_above_multiple():
    assert _round_up(4097, 4096) == 8192


# ----------------------------------------------------------------------
# No-CUDA: unsupported path
# ----------------------------------------------------------------------


def test_unsupported_when_cuda_python_missing(monkeypatch):
    """If ``from cuda import cuda`` raises ImportError, is_supported()
    returns False (not raises) and allocate() raises
    VmmUnsupportedError."""
    # Make the in-method import fail by replacing the ``cuda`` module
    # entry with an object that raises on attribute access. The simpler
    # approach: ensure ``cuda`` is not importable by adding a finder
    # that refuses it.
    real_cuda = sys.modules.pop("cuda", None)
    real_cuda_submod = sys.modules.pop("cuda.cuda", None)

    class _RefuseFinder:
        def find_module(self, name, path=None):
            return self if name in ("cuda", "cuda.cuda") else None

        def load_module(self, name):
            raise ImportError(f"Test refuses {name}")

        # PEP 451 path
        def find_spec(self, name, path, target=None):
            if name in ("cuda", "cuda.cuda"):
                raise ImportError(f"Test refuses {name}")
            return None

    sys.meta_path.insert(0, _RefuseFinder())
    try:
        a = VmmAllocator(device_index=0)
        assert a.is_supported() is False
        # Cached: a second call doesn't re-probe.
        assert a.is_supported() is False
        with pytest.raises(VmmUnsupportedError):
            a.allocate(4096)
    finally:
        sys.meta_path.pop(0)
        if real_cuda is not None:
            sys.modules["cuda"] = real_cuda
        if real_cuda_submod is not None:
            sys.modules["cuda.cuda"] = real_cuda_submod


def test_allocate_zero_size_raises_value_error():
    """Zero / negative size is rejected before the CUDA probe even
    runs, so this works on any host."""
    a = VmmAllocator(device_index=0)
    with pytest.raises(ValueError, match="size_bytes must be > 0"):
        a.allocate(0)
    with pytest.raises(ValueError, match="size_bytes must be > 0"):
        a.allocate(-1)


def test_release_unknown_pool_id_is_idempotent():
    """release() on an unknown pool_id returns False, doesn't raise.
    This holds regardless of CUDA availability."""
    a = VmmAllocator(device_index=0)
    assert a.release("not-a-real-pool") is False
    # Even called twice in a row — defensive against duplicate
    # release_kv_pool RPCs from a confused client.
    assert a.release("not-a-real-pool") is False


def test_shutdown_with_no_allocations_is_safe():
    """shutdown() on a freshly-constructed allocator is a no-op,
    not an error."""
    a = VmmAllocator(device_index=0)
    a.shutdown()  # must not raise
    assert a.known_pool_ids() == []


def test_unsupported_probe_handles_driver_failure(monkeypatch):
    """If cuda-python imports but cuMemGetAllocationGranularity returns
    a non-SUCCESS code, is_supported() returns False (not raises) and
    logs a warning."""

    # Build a minimal fake `cuda.cuda` module that exposes just what
    # _build_alloc_prop + _probe_support touch, and whose
    # cuMemGetAllocationGranularity returns ERROR_NOT_SUPPORTED.
    class _FakeCUresult:
        CUDA_SUCCESS = 0
        CUDA_ERROR_NOT_SUPPORTED = 801

    class _FakeAllocType:
        CU_MEM_ALLOCATION_TYPE_PINNED = 1

    class _FakeLocType:
        CU_MEM_LOCATION_TYPE_DEVICE = 1

    class _FakeHandleType:
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1

    class _FakeGranFlags:
        CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0

    class _FakeProp:
        def __init__(self):
            self.type = None
            self.location = None
            self.requestedHandleTypes = None

    class _FakeLoc:
        def __init__(self):
            self.type = None
            self.id = -1

    fake_cuda = type(sys)("cuda.cuda")
    fake_cuda.CUresult = _FakeCUresult
    fake_cuda.CUmemAllocationType = _FakeAllocType
    fake_cuda.CUmemLocationType = _FakeLocType
    fake_cuda.CUmemAllocationHandleType = _FakeHandleType
    fake_cuda.CUmemAllocationGranularity_flags = _FakeGranFlags
    fake_cuda.CUmemAllocationProp = _FakeProp
    fake_cuda.CUmemLocation = _FakeLoc

    def _gran(_prop, _flag):
        return (
            _FakeCUresult.CUDA_ERROR_NOT_SUPPORTED,
            0,
        )

    def _err_name(_err):
        return (_FakeCUresult.CUDA_SUCCESS, b"CUDA_ERROR_NOT_SUPPORTED")

    fake_cuda.cuMemGetAllocationGranularity = _gran
    fake_cuda.cuGetErrorName = _err_name

    fake_outer = type(sys)("cuda")
    fake_outer.cuda = fake_cuda

    monkeypatch.setitem(sys.modules, "cuda", fake_outer)
    monkeypatch.setitem(sys.modules, "cuda.cuda", fake_cuda)

    a = VmmAllocator(device_index=0)
    assert a.is_supported() is False
    # Subsequent calls don't re-probe (cached).
    assert a.is_supported() is False


def test_unsupported_probe_handles_zero_granularity(monkeypatch):
    """Defensive: if the driver returns SUCCESS but granularity=0,
    treat as unsupported (avoid div-by-zero / infinite loop later)."""

    class _OK:
        CUDA_SUCCESS = 0

    class _AT:
        CU_MEM_ALLOCATION_TYPE_PINNED = 1

    class _LT:
        CU_MEM_LOCATION_TYPE_DEVICE = 1

    class _HT:
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1

    class _GF:
        CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0

    fake_cuda = type(sys)("cuda.cuda")
    fake_cuda.CUresult = _OK
    fake_cuda.CUmemAllocationType = _AT
    fake_cuda.CUmemLocationType = _LT
    fake_cuda.CUmemAllocationHandleType = _HT
    fake_cuda.CUmemAllocationGranularity_flags = _GF
    fake_cuda.CUmemAllocationProp = type(
        "P",
        (),
        {"__init__": lambda s: setattr(s, "_x", None)},
    )
    fake_cuda.CUmemLocation = type(
        "L",
        (),
        {"__init__": lambda s: None},
    )
    fake_cuda.cuMemGetAllocationGranularity = lambda *_a: (
        _OK.CUDA_SUCCESS,
        0,
    )
    fake_outer = type(sys)("cuda")
    fake_outer.cuda = fake_cuda
    monkeypatch.setitem(sys.modules, "cuda", fake_outer)
    monkeypatch.setitem(sys.modules, "cuda.cuda", fake_cuda)

    a = VmmAllocator(device_index=0)
    assert a.is_supported() is False


# ----------------------------------------------------------------------
# CUDA-gated round trip
# ----------------------------------------------------------------------


_REAL_VMM_ENABLED = bool(os.environ.get("GMS_TEST_VMM_REAL"))


@pytest.mark.skipif(
    not _REAL_VMM_ENABLED,
    reason="Set GMS_TEST_VMM_REAL=1 and run on a CUDA host with cuda-python "
    "to exercise the real cuMem* round trip.",
)
def test_supports_query_real():
    """On a CUDA host, is_supported() must succeed and cache a
    granularity > 0."""
    a = VmmAllocator(device_index=0)
    assert a.is_supported() is True
    assert a._granularity is not None
    assert a._granularity > 0
    # Second call returns the cached result.
    assert a.is_supported() is True


@pytest.mark.skipif(
    not _REAL_VMM_ENABLED,
    reason="Set GMS_TEST_VMM_REAL=1 on a CUDA host to run.",
)
def test_alloc_release_round_trip_real():
    """End-to-end: allocate a small pool, verify the bookkeeping
    record is populated correctly, then release and confirm the FD
    is closed + the pool_id no longer tracked."""
    a = VmmAllocator(device_index=0)
    assert a.is_supported() is True

    # Request 4 MiB; it'll be rounded up to granularity.
    requested = 4 * 1024 * 1024
    alloc = a.allocate(requested)
    try:
        assert isinstance(alloc, VmmAllocation)
        assert alloc.size_aligned >= requested
        assert alloc.size_aligned % alloc.granularity == 0
        assert alloc.fd > 0
        assert alloc.va != 0
        assert alloc.device_index == 0
        assert alloc.pool_id in a.known_pool_ids()
    finally:
        ok = a.release(alloc.pool_id)
        assert ok is True
        # The pool is no longer tracked.
        assert alloc.pool_id not in a.known_pool_ids()
        # The exported FD is closed (further close should fail).
        with pytest.raises(OSError):
            os.close(alloc.fd)


@pytest.mark.skipif(
    not _REAL_VMM_ENABLED,
    reason="Set GMS_TEST_VMM_REAL=1 on a CUDA host to run.",
)
def test_shutdown_releases_all_pending_real():
    """shutdown() releases every outstanding allocation."""
    a = VmmAllocator(device_index=0)
    a1 = a.allocate(4 * 1024 * 1024)
    a2 = a.allocate(4 * 1024 * 1024)
    assert sorted(a.known_pool_ids()) == sorted([a1.pool_id, a2.pool_id])
    a.shutdown()
    assert a.known_pool_ids() == []
