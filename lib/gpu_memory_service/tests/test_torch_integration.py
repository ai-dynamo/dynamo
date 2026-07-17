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
from _deps import HAS_CUDA, HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

if not HAS_CUDA:
    pytest.skip(
        "CUDA is required for torch GMS integration tests", allow_module_level=True
    )

import torch
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    rebind_nonparameter_tensors,
    register_module_tensors,
)
from gpu_memory_service.client.torch.tensor import (
    _storage_from_pointer,
    _tensor_from_pointer,
    _tensor_from_storage,
)
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm import _reset_vmm_singleton
from gpu_memory_service.integrations.common.utils import prepare_gms_write
from gpu_memory_service.integrations.trtllm import model_loader as trtllm_model_loader
from gpu_memory_service.integrations.vllm import model_loader
from gpu_memory_service.server.rpc import GMSRPCServer

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_1,
]

_SERVER_START_TIMEOUT_SECONDS = 5.0
_SERVER_STOP_TIMEOUT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.01


class _TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 4, bias=False, device="cuda")
        self.register_buffer(
            "scale",
            torch.linspace(0.5, 2.0, steps=4, device="cuda", dtype=torch.float32),
        )
        self.extra = torch.arange(1, 5, device="cuda", dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = y + self.scale
        y = y * self.extra
        return torch.relu(y)


class _AliasedRuntimeTensor(torch.nn.Module):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("runtime", tensor, persistent=False)
        self.direct = tensor


@pytest.mark.timeout(120)
def test_trtllm_moves_complete_shared_parameter_storage_once(running_gms):
    backing = torch.arange(16, device="cuda", dtype=torch.float32)
    left = torch.nn.Parameter(backing[2:14:2], requires_grad=True)
    sibling = torch.nn.Parameter(backing[3:12:2], requires_grad=False)
    view = backing[1:15:3]
    empty = torch.empty(0, device="cuda")
    model = torch.nn.Module()
    model.register_parameter("left", left)
    model.register_parameter("left_alias", left)
    model.register_parameter("sibling", sibling)
    model.register_buffer("empty", empty)
    model.backing = backing
    model.view = view
    model.view_alias = view
    objects = (left, sibling, backing, view)
    expected = tuple(tensor.detach().clone() for tensor in objects)
    layouts = tuple((tensor.storage_offset(), tensor.stride()) for tensor in objects)
    old_storage = backing.untyped_storage()._cdata
    storage_nbytes = backing.untyped_storage().nbytes()

    manager = GMSClientMemoryManager(running_gms, device=0)
    manager.connect(RequestedLockType.RW)
    tensor = expected_tensor = None
    try:
        trtllm_model_loader._move_untracked_params(
            model,
            manager,
            torch.device("cuda", 0),
        )

        assert len(manager.mappings) == 1
        assert next(iter(manager.mappings.values())).size == storage_nbytes
        assert model.left is model.left_alias is left
        assert model.sibling is sibling
        assert model.empty is empty
        assert model.backing is backing
        assert model.view is model.view_alias is view
        assert model.left.requires_grad
        assert not model.sibling.requires_grad
        storage_ids = {tensor.untyped_storage()._cdata for tensor in objects}
        assert len(storage_ids) == 1
        assert old_storage not in storage_ids
        for tensor, expected_tensor, layout in zip(
            objects, expected, layouts, strict=True
        ):
            assert (tensor.storage_offset(), tensor.stride()) == layout
            torch.testing.assert_close(tensor, expected_tensor)
        model.backing[3] = 123
        assert model.sibling[0].item() == 123
    finally:
        del tensor, expected_tensor
        del model, left, sibling, backing, view, empty, objects, expected
        manager.close()


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
    baseline = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)
    rhs = torch.arange(32, device="cuda", dtype=torch.float32).reshape(8, 4)

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
    baseline = _TinyModule().cuda()
    gms_model = _TinyModule().cuda()
    gms_model.load_state_dict(baseline.state_dict())
    inputs = torch.randn(3, 8, device="cuda", dtype=torch.float32)
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

        materialized = _TinyModule().cuda()
        materialize_module_from_gms(reader, materialized, device_index=0)
        _assert_exact_tensor_equal(expected, materialized(inputs))
    finally:
        reader.close()


def test_finalize_gms_write_rebinds_nonparameter_tensors(running_gms):
    from gpu_memory_service.integrations.common.utils import finalize_gms_write

    socket_path = running_gms
    torch.manual_seed(13)
    baseline = _TinyModule().cuda()
    gms_model = _TinyModule().cuda()
    inputs = torch.randn(3, 8, device="cuda", dtype=torch.float32)
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
        torch.cuda.synchronize()
    finally:
        del gms_model
        writer.close()


@pytest.mark.timeout(120)
def test_deferred_gms_write_reconstructs_runtime_tensor_aliases(
    running_gms, monkeypatch
):
    monkeypatch.setattr(model_loader, "_pending_gms_client", None)
    monkeypatch.setattr(model_loader, "_last_imported_weights_bytes", 0)
    monkeypatch.setattr(model_loader, "_last_model_memory_usage_offset_bytes", 0)
    writer = GMSClientMemoryManager(running_gms, device=0)
    writer.connect(RequestedLockType.RW)
    model: _AliasedRuntimeTensor | None = None
    original: torch.Tensor | None = None

    try:
        original_values = torch.arange(8, device="cuda", dtype=torch.int32)
        _, original = _make_gms_tensor(writer, original_values, tag="weights")
        model = _AliasedRuntimeTensor(original)

        stats = prepare_gms_write(writer, model)
        rebound_bytes = rebind_nonparameter_tensors(writer, model)
        assert rebound_bytes == original.numel() * original.element_size()
        model_loader._store_pending_gms_write(writer, stats, rebound_bytes)
        assert model_loader.publish_pending_gms_write()
        assert model.runtime is model.direct
        model.runtime.fill_(17)
        _assert_exact_tensor_equal(torch.full_like(original, 17), model.direct)
    finally:
        if model_loader.has_pending_gms_write():
            model_loader.abort_pending_gms_write()
        model = None
        original = None
        writer.close()

    with GMSClientMemoryManager(running_gms, device=0) as manager:
        manager.connect(RequestedLockType.RO)
        placeholder = torch.empty_like(original_values, device="meta")
        reader = _AliasedRuntimeTensor(placeholder)
        materialize_module_from_gms(manager, reader, device_index=0)

        reconstructed = reader.runtime
        assert reconstructed is reader.direct
        assert reconstructed is not placeholder
        assert "runtime" in reader._non_persistent_buffers_set
        assert "runtime" not in reader.state_dict()
        _assert_exact_tensor_equal(original_values, reconstructed)


def test_live_gms_tensor_survives_unmap_and_remap(running_gms):
    socket_path = running_gms
    baseline = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)

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


@pytest.mark.timeout(120)
def test_shared_interior_storages_survive_import_and_remap(running_gms):
    with GMSClientMemoryManager(running_gms, device=0) as writer:
        writer.connect(RequestedLockType.RW)
        data_storage = view_storage = None
        data_tensor = view_tensor = None
        writer_model = None
        try:
            allocation_va = writer.create_mapping(size=4096, tag="weights")
            allocation_id = writer.mappings[allocation_va].allocation_id

            data_storage = _storage_from_pointer(allocation_va + 128, 32, 0)
            view_storage = _storage_from_pointer(allocation_va + 256, 48, 0)
            data_tensor = _tensor_from_storage(data_storage, [8], [1], torch.float32)
            view_tensor = _tensor_from_storage(view_storage, [12], [1], torch.float32)
            data_tensor.copy_(torch.arange(8, device="cuda", dtype=torch.float32))
            view_tensor.copy_(torch.arange(12, device="cuda", dtype=torch.float32) + 20)

            writer_model = torch.nn.Module()
            writer_model.register_parameter(
                "data_weight", torch.nn.Parameter(data_tensor, requires_grad=True)
            )
            writer_model.data_exact = writer_model.data_weight
            writer_model.data_view = writer_model.data_weight.data
            writer_model.register_parameter(
                "view_weight", torch.nn.Parameter(view_tensor, requires_grad=False)
            )
            writer_model.strided = writer_model.view_weight[2:10:2]
            writer_model.transposed = writer_model.view_weight.reshape(3, 4).T
            register_module_tensors(writer, writer_model)
            assert writer.commit()
        finally:
            del writer_model, data_tensor, view_tensor, data_storage, view_storage

    reader_model = torch.nn.Module()
    reader_model.register_parameter(
        "data_weight",
        torch.nn.Parameter(torch.zeros(8, device="cuda"), requires_grad=True),
    )
    reader_model.data_exact = reader_model.data_weight
    reader_model.data_view = reader_model.data_weight.data
    reader_model.register_parameter(
        "view_weight",
        torch.nn.Parameter(torch.zeros(12, device="cuda"), requires_grad=False),
    )
    reader_model.strided = reader_model.view_weight[2:10:2]
    reader_model.transposed = reader_model.view_weight.reshape(3, 4).T

    with GMSClientMemoryManager(running_gms, device=0) as reader:
        reader.connect(RequestedLockType.RO)
        materialize_module_from_gms(reader, reader_model, device_index=0)
        assert len(reader.mappings) == 1
        assert next(iter(reader.mappings.values())).allocation_id == allocation_id
        assert reader_model.data_weight is reader_model.data_exact
        assert reader_model.data_weight is not reader_model.data_view
        assert (
            reader_model.data_weight.untyped_storage()._cdata
            == reader_model.data_view.untyped_storage()._cdata
        )
        assert (
            reader_model.view_weight.untyped_storage()._cdata
            == reader_model.strided.untyped_storage()._cdata
            == reader_model.transposed.untyped_storage()._cdata
        )
        reader_va = next(iter(reader.mappings))
        assert reader_model.data_weight.data_ptr() == reader_va + 128
        assert reader_model.strided.storage_offset() == 2
        assert reader_model.transposed.stride() == (1, 4)

        reader.unmap_all_vas()
        reader.abort()
        reader.connect(RequestedLockType.RO)
        reader.remap_all_vas()
        torch.testing.assert_close(
            reader_model.data_view,
            torch.arange(8, device="cuda", dtype=torch.float32),
        )
        torch.testing.assert_close(
            reader_model.strided,
            torch.tensor([22, 24, 26, 28], device="cuda", dtype=torch.float32),
        )
        torch.testing.assert_close(
            reader_model.transposed,
            (torch.arange(12, device="cuda", dtype=torch.float32) + 20).reshape(3, 4).T,
        )
        del reader_model


def test_materialized_module_from_gms_matches_plain_module_forward(running_gms):
    socket_path = running_gms
    torch.manual_seed(7)
    baseline = _TinyModule().cuda()
    gms_model = _TinyModule().cuda()
    gms_model.load_state_dict(baseline.state_dict())
    inputs = torch.randn(3, 8, device="cuda", dtype=torch.float32)
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
    materialized = _TinyModule().cuda()
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
