# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch integration coverage for GMS-backed tensors and modules.

This module exercises tensor remap after unmap/remap cycles and module
materialization from committed GMS-backed weights.
"""

from __future__ import annotations

import asyncio
import os
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
from gpu_memory_service.client.torch import allocator as gms_allocator
from gpu_memory_service.client.torch.extensions import _allocator_ext
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    register_module_tensors,
)
from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
from gpu_memory_service.common import utils as common_utils
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm import _reset_vmm_singleton
from gpu_memory_service.integrations.common import patches as common_patches
from gpu_memory_service.integrations.common.utils import publish_gms_write, torch_device
from gpu_memory_service.integrations.vllm import model_loader
from gpu_memory_service.server.rpc import GMSRPCServer

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
]

_SERVER_START_TIMEOUT_SECONDS = 5.0
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


def _in_gms(manager: GMSClientMemoryManager, tensor: torch.Tensor) -> bool:
    ptr = tensor.data_ptr()
    return any(
        va <= ptr < va + mapping.aligned_size
        for va, mapping in manager.mappings.items()
    )


def test_vllm_writer_publishes_from_a_released_pool(running_gms, monkeypatch):
    """vLLM first writer: drop the weight pool, copy live non-Parameters out of
    GMS, publish, and keep the committed layout intact across a later free."""
    if _allocator_ext is None:
        pytest.skip("GMS allocator extension is not built")
    socket_path = running_gms
    # Reconnecting after abort re-derives the socket from the GPU UUID.
    monkeypatch.setattr(common_utils, "get_socket_path", lambda *_: socket_path)
    torch.manual_seed(17)
    baseline = _TinyModule().to(_DEVICE)
    inputs = torch.randn(3, 8, device=_DEVICE, dtype=torch.float32)
    expected = baseline(inputs).detach().clone()

    writer = gms_allocator.get_or_create_gms_client_memory_manager(
        socket_path, 0, RequestedLockType.RW, tag="weights"
    )
    try:
        with gms_allocator.gms_use_mem_pool("weights", torch.device(_DEVICE, 0)):
            model = _TinyModule()
            model.load_state_dict(baseline.state_dict())
            model.extra.copy_(baseline.extra)
            # A non-Parameter view of a weight, and a zero-size buffer.
            model.weight_rows = model.linear.weight.detach()[1:]
            model.register_buffer("empty", torch.empty(0, device=_DEVICE))
            # Held outside the module tree, like a kernel workspace.
            workspace = torch.zeros(256, device=_DEVICE)
            transient = torch.empty(32 << 20, dtype=torch.uint8, device=_DEVICE)
            del transient
        weight_ptr = model.linear.weight.data_ptr()
        allocated_bytes = writer.total_bytes

        register_module_tensors(writer, model)
        stats, copied_bytes, owners = model_loader._copy_out_and_release_pool(
            writer, model
        )

        assert stats.pruned_count == 1 and stats.pruned_bytes >= 32 << 20
        assert stats.committed_bytes == writer.total_bytes
        assert writer.total_bytes == allocated_bytes - stats.pruned_bytes
        assert model.linear.weight.data_ptr() == weight_ptr
        assert _in_gms(writer, model.linear.weight)
        for tensor in (model.scale, model.extra, model.weight_rows, workspace):
            assert not _in_gms(writer, tensor)
        # weight_rows is copied with its detached (4, 8) view base.
        assert copied_bytes == (4 + 4 + 4 * 8 + 256) * 4
        # scale, extra, and the weight storage behind weight_rows.
        assert len(owners) == 3

        publish_gms_write(writer)
        # Copies stay writable after the weights become read-only.
        cast(torch.Tensor, model.scale).mul_(1.0)
        workspace.add_(1.0)
        _synchronize()
        _assert_exact_tensor_equal(expected, model(inputs))

        # A weight storage freed after publication keeps its committed
        # mapping, so a later unmap/remap still matches the layout.
        mapping_count = len(writer.mappings)
        model.linear.weight = torch.nn.Parameter(torch.zeros(4, 8, device=_DEVICE))
        del owners
        _synchronize()
        _empty_cache()
        gms_allocator.raise_if_gms_allocator_failed(writer)
        assert len(writer.mappings) == mapping_count
        writer.unmap_all_vas()
        writer.abort()
        writer.connect(RequestedLockType.RO)
        writer.remap_all_vas()
        assert len(writer.mappings) == mapping_count
    finally:
        model = workspace = None
        writer.close()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    try:
        materialized = _TinyModule().to(_DEVICE)
        materialize_module_from_gms(reader, materialized, device_index=0)
        _assert_exact_tensor_equal(expected, materialized(inputs))
        _assert_exact_tensor_equal(
            cast(torch.Tensor, baseline.linear.weight)[1:],
            cast(torch.Tensor, materialized.weight_rows),
        )
    finally:
        materialized = None
        reader.close()


def test_vllm_writer_refuses_an_allocation_no_registered_tensor_is_in(running_gms):
    """A Parameter held outside the module tree is not copied out, so it keeps
    its allocation through the pool release, and nothing registers it. A
    reader would map fewer allocations than the layout has and fail to wake,
    so the writer fails before it publishes."""
    if _allocator_ext is None:
        pytest.skip("GMS allocator extension is not built")
    writer = gms_allocator.get_or_create_gms_client_memory_manager(
        running_gms, 0, RequestedLockType.RW, tag="weights"
    )
    try:
        with gms_allocator.gms_use_mem_pool("weights", torch.device(_DEVICE, 0)):
            model = _TinyModule()
            external = torch.nn.Parameter(
                torch.zeros(64 << 20, dtype=torch.uint8, device=_DEVICE),
                requires_grad=False,
            )
        referenced = register_module_tensors(writer, model)
        stats, _, owners = model_loader._copy_out_and_release_pool(writer, model)
        assert _in_gms(writer, external) and stats.pruned_count == 0
        assert len(writer.mappings) == 2 and len(referenced) == 1
        with pytest.raises(RuntimeError, match="keeps 1 allocation"):
            model_loader._raise_if_unreferenced_allocations(writer, referenced)
        del owners
    finally:
        model = external = None
        writer.close()


def test_vllm_writer_commits_no_allocation_a_reader_cannot_reference(
    running_gms, monkeypatch
):
    """A workspace held outside the module tree, in an allocation of its own,
    leaves GMS before publication. With that allocation still committed, a
    reader would map fewer allocations than the layout has and fail to wake."""
    if _allocator_ext is None:
        pytest.skip("GMS allocator extension is not built")
    socket_path = running_gms
    monkeypatch.setattr(common_utils, "get_socket_path", lambda *_: socket_path)
    # As in a GMS worker: empty_cache() does nothing while mappings are live.
    device_module = torch_device()
    monkeypatch.setattr(device_module, "empty_cache", device_module.empty_cache)
    monkeypatch.setattr(common_patches, "_empty_cache_patched", False)
    common_patches.patch_empty_cache()
    torch.manual_seed(17)
    baseline = _TinyModule().to(_DEVICE)

    writer = gms_allocator.get_or_create_gms_client_memory_manager(
        socket_path, 0, RequestedLockType.RW, tag="weights"
    )
    try:
        with gms_allocator.gms_use_mem_pool("weights", torch.device(_DEVICE, 0)):
            model = _TinyModule()
            model.load_state_dict(baseline.state_dict())
            workspace = torch.zeros(64 << 20, dtype=torch.uint8, device=_DEVICE)
        assert len(writer.mappings) == 2
        referenced = register_module_tensors(writer, model)
        stats, _, owners = model_loader._copy_out_and_release_pool(writer, model)
        model_loader._raise_if_unreferenced_allocations(writer, referenced)
        assert not _in_gms(writer, workspace)
        assert stats.pruned_count == 1 and stats.pruned_bytes >= 64 << 20
        publish_gms_write(writer)
        del owners
        committed = len(writer.list_handles())
        assert committed == len(writer.mappings) == 1
    finally:
        model = workspace = None
        writer.close()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    try:
        materialized = _TinyModule().to(_DEVICE)
        materialize_module_from_gms(reader, materialized, device_index=0)
        assert len(reader.mappings) == committed
        reader.unmap_all_vas()
        reader.abort()
        reader.connect(RequestedLockType.RO)
        reader.remap_all_vas()
        _assert_exact_tensor_equal(
            cast(torch.Tensor, baseline.linear.weight),
            cast(torch.Tensor, materialized.linear.weight),
        )
    finally:
        materialized = None
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
