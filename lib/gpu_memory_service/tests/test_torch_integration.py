# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch integration coverage for GMS-backed tensors and modules.

This module exercises tensor remap after unmap/remap cycles and module
materialization from committed GMS-backed weights.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from typing import cast

import pytest
from _deps import HAS_GMS, HAS_GPU, HAS_TORCH, HAS_XPU

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

if not HAS_GPU:
    pytest.skip(
        "CUDA or XPU GPU is required for torch GMS integration tests",
        allow_module_level=True,
    )

import torch

# Device-agnostic helper: prefer XPU when available, fall back to CUDA.
if HAS_XPU:
    _DEVICE = "xpu"

    def _synchronize():
        torch.xpu.synchronize()

    def _empty_cache():
        torch.xpu.empty_cache()

else:
    _DEVICE = "cuda"

    def _synchronize():
        torch.cuda.synchronize()

    def _empty_cache():
        torch.cuda.empty_cache()


from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    register_module_tensors,
)
from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm import _reset_vmm_singleton, get_vmm
from gpu_memory_service.server.rpc import GMSRPCServer

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
]

requires_cuda = pytest.mark.skipif(HAS_XPU, reason="fixed-VA windows are CUDA-only")

_SERVER_START_TIMEOUT_SECONDS = 5.0
_CHILD_TIMEOUT_SECONDS = 60
# In a fresh process: clear of the ranges the driver picks itself (around
# 0x10000000000) and of the mmap area just below 0x800000000000.
_WINDOW = (0x500000000000, 64 << 20)
_SERVER_STOP_TIMEOUT_SECONDS = 15.0  # XPU: SYCL synchronize may take longer
_POLL_INTERVAL_SECONDS = 0.01


class _TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 4, bias=False, device=_DEVICE)
        self.register_buffer(
            "scale",
            torch.linspace(0.5, 2.0, steps=4, device=_DEVICE, dtype=torch.float32),
        )
        self.extra = torch.arange(1, 5, device=_DEVICE, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = y + self.scale
        y = y * self.extra
        return torch.relu(y)


@pytest.fixture
def running_gms(tmp_path):
    socket_path = str(tmp_path / "gms.sock")
    server = GMSRPCServer(socket_path, device=0)
    loop: asyncio.AbstractEventLoop | None = None
    task: asyncio.Task[None] | None = None
    thread_error: BaseException | None = None

    def run() -> None:
        nonlocal loop, task, thread_error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(server.serve())
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            thread_error = exc
        finally:
            pending = [
                pending_task
                for pending_task in asyncio.all_tasks(loop)
                if not pending_task.done()
            ]
            for pending_task in pending:
                pending_task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
    while True:
        if thread_error is not None:
            raise thread_error
        if server._server is not None and os.path.exists(socket_path):
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"GMS socket did not appear at {socket_path}")
        time.sleep(_POLL_INTERVAL_SECONDS)

    try:
        yield socket_path
    finally:
        if loop is not None:

            def cancel() -> None:
                if server._server is not None:
                    server._server.close()
                if task is not None:
                    task.cancel()

            loop.call_soon_threadsafe(cancel)
        thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        if thread.is_alive():
            raise RuntimeError(f"GMS server thread failed to stop for {socket_path}")
        if thread_error is not None:
            raise thread_error
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        _reset_vmm_singleton()


def _make_gms_tensor(
    manager: GMSClientMemoryManager,
    tensor: torch.Tensor,
    *,
    tag: str,
) -> tuple[str, torch.Tensor]:
    storage_bytes = tensor.untyped_storage().nbytes()
    va = manager.create_mapping(size=storage_bytes, tag=tag)
    allocation_id = manager.mappings[va].allocation_id
    gms_tensor = _tensor_from_pointer(
        va,
        list(tensor.shape),
        list(tensor.stride()),
        tensor.dtype,
        tensor.device.index or 0,
    )
    gms_tensor.copy_(tensor)
    return allocation_id, gms_tensor


def _assert_exact_tensor_equal(expected: torch.Tensor, actual: torch.Tensor) -> None:
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_gms_tensor_matches_plain_torch_ops(running_gms):
    socket_path = running_gms
    baseline = torch.arange(64, device=_DEVICE, dtype=torch.float32).reshape(8, 8)
    rhs = torch.arange(32, device=_DEVICE, dtype=torch.float32).reshape(8, 4)

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    allocation_id, writer_tensor = _make_gms_tensor(writer, baseline, tag="weights")
    assert writer.commit()
    del writer_tensor
    writer.close()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    va = reader.create_mapping(allocation_id=allocation_id)
    gms_tensor = _tensor_from_pointer(
        va,
        list(baseline.shape),
        list(baseline.stride()),
        baseline.dtype,
        0,
    )

    _assert_exact_tensor_equal(
        torch.relu((baseline + 3.0) @ rhs), torch.relu((gms_tensor + 3.0) @ rhs)
    )
    _assert_exact_tensor_equal(
        baseline.transpose(0, 1).contiguous(), gms_tensor.transpose(0, 1).contiguous()
    )
    _assert_exact_tensor_equal(
        baseline[:, 2:6].sum(dim=1), gms_tensor[:, 2:6].sum(dim=1)
    )
    _assert_exact_tensor_equal(
        (baseline * 2.0 - 5.0).square(), (gms_tensor * 2.0 - 5.0).square()
    )

    reader.close()


def test_finalize_gms_write_prunes_unreferenced_allocations(running_gms):
    from gpu_memory_service.integrations.common.utils import finalize_gms_write

    socket_path = running_gms
    torch.manual_seed(11)
    baseline = _TinyModule().to(_DEVICE)
    gms_model = _TinyModule().to(_DEVICE)
    gms_model.load_state_dict(baseline.state_dict())
    inputs = torch.randn(3, 8, device=_DEVICE, dtype=torch.float32)
    expected = baseline(inputs).detach().clone()

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)

    baseline_weight = cast(torch.Tensor, baseline.linear.weight)
    baseline_scale = cast(torch.Tensor, baseline.scale)
    baseline_extra = cast(torch.Tensor, baseline.extra)

    _, gms_weight = _make_gms_tensor(writer, baseline_weight, tag="weights")
    gms_model.linear.weight = torch.nn.Parameter(
        gms_weight, requires_grad=baseline_weight.requires_grad
    )
    _, gms_scale = _make_gms_tensor(writer, baseline_scale, tag="weights")
    gms_model._buffers["scale"] = gms_scale
    _, gms_extra = _make_gms_tensor(writer, baseline_extra, tag="weights")
    gms_model.extra = gms_extra

    unreferenced_va = writer.create_mapping(size=1024 * 1024, tag="weights")
    unreferenced_allocation_id = writer.mappings[unreferenced_va].allocation_id

    committed_bytes = finalize_gms_write(writer, gms_model).committed_bytes
    assert committed_bytes == sum(m.aligned_size for m in writer.mappings.values())
    assert all(
        mapping.allocation_id != unreferenced_allocation_id
        for mapping in writer.mappings.values()
    )

    del gms_weight
    del gms_scale
    del gms_extra
    del gms_model
    writer.close()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    try:
        handles = reader.list_handles()
        assert len(handles) == 3
        assert all(info.allocation_id != unreferenced_allocation_id for info in handles)

        materialized = _TinyModule().to(_DEVICE)
        materialize_module_from_gms(reader, materialized, device_index=0)
        _assert_exact_tensor_equal(expected, materialized(inputs))
    finally:
        reader.close()


def test_finalize_gms_write_rebinds_nonparameter_tensors(running_gms):
    from gpu_memory_service.integrations.common.utils import finalize_gms_write

    socket_path = running_gms
    torch.manual_seed(13)
    baseline = _TinyModule().to(_DEVICE)
    gms_model = _TinyModule().to(_DEVICE)
    inputs = torch.randn(3, 8, device=_DEVICE, dtype=torch.float32)
    expected = baseline(inputs).detach().clone()

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)

    baseline_weight = cast(torch.Tensor, baseline.linear.weight)
    baseline_scale = cast(torch.Tensor, baseline.scale)
    baseline_extra = cast(torch.Tensor, baseline.extra)

    _, gms_weight = _make_gms_tensor(writer, baseline_weight, tag="weights")
    gms_model.linear.weight = torch.nn.Parameter(
        gms_weight, requires_grad=baseline_weight.requires_grad
    )
    _, gms_scale = _make_gms_tensor(writer, baseline_scale, tag="weights")
    gms_model._buffers["scale"] = gms_scale
    _, gms_extra = _make_gms_tensor(writer, baseline_extra, tag="weights")
    gms_model.extra = gms_extra
    del gms_weight, gms_scale, gms_extra

    weight_ptr = gms_model.linear.weight.data_ptr()

    finalize_gms_write(writer, gms_model)

    def _in_gms(tensor: torch.Tensor) -> bool:
        ptr = tensor.data_ptr()
        return any(
            va <= ptr < va + mapping.aligned_size
            for va, mapping in writer.mappings.items()
        )

    try:
        # Parameters keep their shared (now read-only) GMS binding.
        assert gms_model.linear.weight.data_ptr() == weight_ptr
        assert _in_gms(cast(torch.Tensor, gms_model.linear.weight))

        # The buffer and the tensor attr are rebound to private memory.
        assert not _in_gms(cast(torch.Tensor, gms_model.scale))
        assert not _in_gms(cast(torch.Tensor, gms_model.extra))

        # Values are preserved across the rebind.
        _assert_exact_tensor_equal(expected, gms_model(inputs))

        # The rebound copies are writable. Without the rebind these writes
        # would land on the PROT_READ weights mapping (Xid 31).
        cast(torch.Tensor, gms_model.scale).add_(1.0)
        cast(torch.Tensor, gms_model.extra).zero_()
        _synchronize()
    finally:
        del gms_model
        writer.close()


def test_live_gms_tensor_survives_unmap_and_remap(running_gms):
    socket_path = running_gms
    baseline = torch.arange(64, device=_DEVICE, dtype=torch.float32).reshape(8, 8)

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    allocation_id, writer_tensor = _make_gms_tensor(writer, baseline, tag="weights")
    del writer_tensor
    assert writer.commit()
    writer.close()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    va = reader.create_mapping(allocation_id=allocation_id)
    gms_tensor = _tensor_from_pointer(
        va,
        list(baseline.shape),
        list(baseline.stride()),
        baseline.dtype,
        0,
    )
    pointer_before = gms_tensor.data_ptr()
    expected = torch.relu((baseline + 7.0).square())

    reader.unmap_all_vas()
    reader.abort()
    reader.connect(RequestedLockType.RO)
    reader.remap_all_vas()

    assert gms_tensor.data_ptr() == pointer_before
    _assert_exact_tensor_equal(expected, torch.relu((gms_tensor + 7.0).square()))

    reader.close()


# Well under a second on a GPU; the bound is for a stalled lock or socket.
@pytest.mark.timeout(60)
def test_bulk_import_and_remap_map_each_allocation_at_its_own_address(
    running_gms, monkeypatch
):
    socket_path = running_gms
    # Several chunks, so an allocation mapped at another one's address shows.
    monkeypatch.setattr(GMSClientMemoryManager, "_BULK_CHUNK", 2)
    shape = (16,)

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    expected = {}
    for value in range(1, 6):
        source = torch.full(shape, float(value), device=_DEVICE)
        allocation_id, writer_tensor = _make_gms_tensor(writer, source, tag="weights")
        del writer_tensor
        expected[allocation_id] = source
    assert writer.commit()
    writer.close()

    reader = GMSClientMemoryManager(socket_path, device=0, bulk_import=True)
    reader.connect(RequestedLockType.RO)
    try:
        reader.import_allocations(expected)
        tensors = {
            allocation_id: _tensor_from_pointer(
                reader.create_mapping(allocation_id=allocation_id),
                list(shape),
                [1],
                torch.float32,
                0,
            )
            for allocation_id in expected
        }
        pointers = {a: t.data_ptr() for a, t in tensors.items()}
        assert len(set(pointers.values())) == len(expected)
        for allocation_id, source in expected.items():
            _assert_exact_tensor_equal(source, tensors[allocation_id])

        reader.unmap_all_vas()
        reader.abort()
        reader.connect(RequestedLockType.RO)
        reader.remap_all_vas()

        for allocation_id, source in expected.items():
            assert tensors[allocation_id].data_ptr() == pointers[allocation_id]
            _assert_exact_tensor_equal(source, tensors[allocation_id])
    finally:
        reader.close()


# A fresh process that maps the committed layout into the fixed-VA window and
# prints, per allocation, its address and the first values stored there.
_FIXED_VA_READER = """
import json, sys
import torch
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
from gpu_memory_service.common.locks import RequestedLockType

socket_path, window, ids, bulk = sys.argv[1:5]
ids = json.loads(ids)
manager = GMSClientMemoryManager(
    socket_path, device=0, bulk_import=bulk == "bulk", fixed_va_window=tuple(json.loads(window))
)
manager.connect(RequestedLockType.RO)
if bulk == "bulk":
    manager.import_allocations(reversed(ids))
seen = {}
for allocation_id in ids:
    va = manager.create_mapping(allocation_id=allocation_id)
    seen[allocation_id] = [va, _tensor_from_pointer(va, [4], [1], torch.float32, 0).tolist()]
manager.close()
print(json.dumps(seen))
"""


def _driver_chosen_window(size: int) -> tuple[int, int]:
    """A window where the driver just reserved and released ``size`` bytes.

    This process has launched kernels by now, after which drivers with HMM
    addressing honor a placement hint only for ranges they have handed out
    before (see the fixed-VA section of the README).
    """
    vmm = get_vmm()
    granularity = vmm.get_allocation_granularity(0)
    base = vmm.address_reserve(size, granularity)
    vmm.address_free(base, size)
    return base, size


@requires_cuda
@pytest.mark.timeout(3 * _CHILD_TIMEOUT_SECONDS)
def test_fixed_va_window_maps_each_allocation_at_one_address_in_two_processes(
    running_gms,
):
    socket_path = running_gms
    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    granularity = writer.granularity
    # 4 KiB, a granule plus 4 KiB, and 1 MiB: 1, 2 and 1 granules once aligned.
    values = {}
    for value, nbytes in ((1.0, 4096), (2.0, granularity + 4096), (3.0, 1 << 20)):
        source = torch.full((nbytes // 4,), value, device=_DEVICE)
        allocation_id, writer_tensor = _make_gms_tensor(writer, source, tag="weights")
        del writer_tensor
        values[allocation_id] = value
    assert writer.commit()
    writer.close()
    base = _WINDOW[0]
    expected = {
        allocation_id: [base + offset, [value] * 4]
        for (allocation_id, value), offset in zip(
            values.items(), (0, granularity, 3 * granularity)
        )
    }

    # Each reader reserves the window before it loads any CUDA module.
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, sys.path)),
        "CUDA_MODULE_LOADING": "LAZY",
    }
    for mode in ("one at a time", "bulk"):
        reader = subprocess.run(
            [
                sys.executable,
                "-c",
                _FIXED_VA_READER,
                socket_path,
                json.dumps(_WINDOW),
                json.dumps(list(values)),
                mode,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_SECONDS,
        )
        assert reader.returncode == 0, reader.stderr
        assert json.loads(reader.stdout.splitlines()[-1]) == expected, mode


@requires_cuda
@pytest.mark.timeout(_CHILD_TIMEOUT_SECONDS)
def test_fixed_va_window_is_released_and_its_ranges_reused(running_gms, tmp_path):
    socket_path = running_gms
    window = _driver_chosen_window(64 << 20)
    base, size = window
    manager = GMSClientMemoryManager(
        socket_path,
        device=0,
        tag="kv_cache",
        scratch_size=2 << 20,
        fixed_va_window=window,
    )
    try:
        top = manager.create_scratch_mapping(4 << 20)
        assert top == base + size - (4 << 20)
        below = manager.create_scratch_mapping(2 << 20)
        assert below == top - (2 << 20)
        _tensor_from_pointer(top, [1 << 20], [1], torch.float32, 0).fill_(1.0)
        _synchronize()
        assert manager.destroy_scratch_mapping(top)
        assert manager.create_scratch_mapping(4 << 20) == top
    finally:
        manager.close()

    # Closed, or dropped after a failed first connect, a manager leaves the
    # window free for the next one.
    GMSClientMemoryManager(socket_path, device=0, fixed_va_window=window).close()
    dropped = GMSClientMemoryManager(
        str(tmp_path / "no-server.sock"), device=0, fixed_va_window=window
    )
    with pytest.raises(ConnectionError):
        dropped.connect(RequestedLockType.RO, timeout_ms=0)
    GMSClientMemoryManager(socket_path, device=0, fixed_va_window=window).close()


@requires_cuda
@pytest.mark.timeout(_CHILD_TIMEOUT_SECONDS)
def test_fixed_va_window_that_is_taken_is_refused(running_gms):
    socket_path = running_gms
    window = _driver_chosen_window(64 << 20)
    base, size = window
    vmm = get_vmm()
    holder = GMSClientMemoryManager(socket_path, device=0, fixed_va_window=window)
    granularity = holder.granularity
    try:
        with pytest.raises(RuntimeError, match="is not available") as refused:
            GMSClientMemoryManager(socket_path, device=0, fixed_va_window=window)
    finally:
        holder.close()
    # The reservation the driver made elsewhere was returned.
    match = re.search(r"returned 0x([0-9a-f]+)", str(refused.value))
    assert match is not None
    stray = int(match.group(1), 16)
    assert stray != base
    assert vmm.address_reserve(size, granularity, addr=stray) == stray
    vmm.address_free(stray, size)

    last_granule = vmm.address_reserve(
        granularity, granularity, addr=base + size - granularity
    )
    try:
        assert last_granule == base + size - granularity
        with pytest.raises(RuntimeError, match="is not available"):
            GMSClientMemoryManager(socket_path, device=0, fixed_va_window=window)
    finally:
        vmm.address_free(last_granule, granularity)
    GMSClientMemoryManager(socket_path, device=0, fixed_va_window=window).close()


def test_materialized_module_from_gms_matches_plain_module_forward(running_gms):
    socket_path = running_gms
    torch.manual_seed(7)
    baseline = _TinyModule().to(_DEVICE)
    gms_model = _TinyModule().to(_DEVICE)
    gms_model.load_state_dict(baseline.state_dict())
    inputs = torch.randn(3, 8, device=_DEVICE, dtype=torch.float32)
    expected = baseline(inputs).detach().clone()

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)

    baseline_weight = cast(torch.Tensor, baseline.linear.weight)
    baseline_scale = cast(torch.Tensor, baseline.scale)
    baseline_extra = cast(torch.Tensor, baseline.extra)

    _, gms_weight = _make_gms_tensor(writer, baseline_weight, tag="weights")
    gms_model.linear.weight = torch.nn.Parameter(
        gms_weight, requires_grad=baseline_weight.requires_grad
    )
    _, gms_scale = _make_gms_tensor(writer, baseline_scale, tag="weights")
    gms_model._buffers["scale"] = gms_scale
    _, gms_extra = _make_gms_tensor(writer, baseline_extra, tag="weights")
    gms_model.extra = gms_extra

    register_module_tensors(writer, gms_model)
    _assert_exact_tensor_equal(expected, gms_model(inputs))
    assert writer.commit()
    del gms_weight
    del gms_scale
    del gms_extra
    del gms_model
    writer.close()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    materialized = _TinyModule().to(_DEVICE)
    materialize_module_from_gms(reader, materialized, device_index=0)

    _assert_exact_tensor_equal(expected, materialized(inputs))
    _assert_exact_tensor_equal(baseline_scale, cast(torch.Tensor, materialized.scale))
    _assert_exact_tensor_equal(baseline_extra, cast(torch.Tensor, materialized.extra))
    _assert_exact_tensor_equal(
        baseline_weight,
        cast(torch.Tensor, materialized.linear.weight),
    )

    reader.close()


def test_integration_helper_without_explicit_init_vmm(tmp_path):
    """Ensure GMSClientMemoryManager works without pre-seeding the VMM singleton.

    Integration paths (vLLM, SGLang, TRTLLM, gms-storage-client) construct
    GMSClientMemoryManager without calling init_vmm() first. The lazy
    auto-detection in get_vmm() must initialize transparently based on
    available hardware.
    """
    # Reset singleton to simulate a fresh process that never called init_vmm()
    _reset_vmm_singleton()

    from gpu_memory_service.common.vmm import (
        _detect_device_type,
        get_vmm,
        get_vmm_device_type,
    )

    # get_vmm() should lazily auto-detect and initialize without raising
    vmm = get_vmm()
    assert vmm is not None

    # device type should match what auto-detection would pick
    expected = _detect_device_type()
    assert get_vmm_device_type() == expected

    # Clean up
    _reset_vmm_singleton()
