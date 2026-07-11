# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch integration coverage for GMS-backed tensors and modules.

This module exercises tensor remap after unmap/remap cycles and module
materialization from committed GMS-backed weights.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import threading
import time
from types import SimpleNamespace
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
from gpu_memory_service.client.torch.allocator import (
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    rebind_nonparameter_tensors,
    register_module_tensors,
)
from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm import _reset_vmm_singleton
from gpu_memory_service.integrations.common.utils import prepare_gms_write
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


class _RoutedExpertsLike(torch.nn.Module):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_expert_map", tensor, persistent=False)

    @property
    def expert_map(self) -> torch.Tensor:
        return self._expert_map


class _AliasedRuntimeTensor(torch.nn.Module):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.indexer = _RoutedExpertsLike(tensor)
        self.indexer.__dict__["duplicate_buffer_alias"] = tensor
        self.indexer.tensor_list = [tensor]
        self.indexer.tensor_tuple = (tensor,)
        self.indexer.indexer_op = torch.nn.Module()
        self.indexer.indexer_op.expert_map = tensor
        self.helper = SimpleNamespace(expert_map=tensor)

    def aliases(self) -> tuple[torch.Tensor, ...]:
        return (
            self.indexer._expert_map,
            self.indexer.expert_map,
            self.indexer.duplicate_buffer_alias,
            self.indexer.tensor_list[0],
            self.indexer.tensor_tuple[0],
            self.indexer.indexer_op.expert_map,
            self.helper.expert_map,
        )


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


@pytest.mark.timeout(60)
def test_deferred_gms_write_preserves_runtime_tensor_aliases(running_gms, monkeypatch):
    socket_path = running_gms
    monkeypatch.setattr(
        "gpu_memory_service.common.utils.get_socket_path",
        lambda _device, _tag: socket_path,
    )
    monkeypatch.setattr(model_loader, "_pending_gms_client", None)
    monkeypatch.setattr(model_loader, "_pending_retained_gms_tensors", [])
    monkeypatch.setattr(model_loader, "_last_imported_weights_bytes", 0)
    monkeypatch.setattr(model_loader, "_last_model_memory_usage_offset_bytes", 0)
    model: _AliasedRuntimeTensor | None = None
    original: torch.Tensor | None = None
    retained_gms_tensors: list[torch.Tensor] = []
    writer = get_or_create_gms_client_memory_manager(
        socket_path,
        device=0,
        mode=RequestedLockType.RW,
        tag="weights",
    )

    try:
        original_values = torch.arange(8, device="cuda", dtype=torch.int32)
        with gms_use_mem_pool("weights", device=0):
            original = original_values.clone()
        gms_ptr = original.data_ptr()
        unrelated = torch.full((8,), -1, device="cuda", dtype=torch.int32)
        unrelated_helper = SimpleNamespace(tensor=unrelated)
        model = _AliasedRuntimeTensor(original)
        model.unrelated = unrelated_helper

        stats = prepare_gms_write(writer, model)
        rebound_bytes = rebind_nonparameter_tensors(
            writer,
            model,
            retain_gms_tensors=retained_gms_tensors,
        )
        assert rebound_bytes == original.numel() * original.element_size()
        holder = model.__dict__["_gms_rebound_tensor_owners"]
        assert all(alias is original for alias in model.aliases())
        assert original.data_ptr() != gms_ptr
        assert len(retained_gms_tensors) == len(holder.tensors) == 1
        assert retained_gms_tensors[0] is holder.tensors[0] is not original
        assert retained_gms_tensors[0].data_ptr() == gms_ptr
        assert model.unrelated is unrelated_helper
        assert model.unrelated.tensor is unrelated
        assert list(model.state_dict()) == []
        del holder

        repeated_retained: list[torch.Tensor] = []
        assert (
            rebind_nonparameter_tensors(
                writer,
                model,
                retain_gms_tensors=repeated_retained,
            )
            == 0
        )
        assert repeated_retained == []

        model_loader._store_pending_gms_write(
            writer,
            stats,
            rebound_bytes,
            retained_gms_tensors,
        )
        assert model_loader.publish_pending_gms_write()
        retained_gms_tensors.clear()

        writer.unmap_all_vas()
        writer.abort()
        gc.collect()
        torch.cuda.empty_cache()
        writer.connect(RequestedLockType.RO)
        writer.remap_all_vas()

        model.indexer.tensor_list[0].fill_(17)
        for alias in model.aliases():
            _assert_exact_tensor_equal(torch.full_like(original, 17), alias)

        reader_original = torch.empty_like(original, device="meta")
        reader = _AliasedRuntimeTensor(reader_original)
        reader_helper = reader.helper

        materialize_module_from_gms(writer, reader, device_index=0)

        assert all(alias is reader_original for alias in reader.aliases())
        assert reader.helper is reader_helper
        assert reader_original.is_cuda
        assert reader_original.data_ptr() != gms_ptr
        _assert_exact_tensor_equal(original_values, reader_original)
    finally:
        primary_failure = sys.exc_info()[0] is not None
        if model is not None:
            del model
        original = None
        retained_gms_tensors.clear()
        cleanup_error: BaseException | None = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as exc:
            cleanup_error = exc
        if model_loader.has_pending_gms_write():
            try:
                model_loader.abort_pending_gms_write()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        if get_gms_client_memory_manager("weights") is writer:
            try:
                writer.close()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
                try:
                    writer.close(best_effort=True)
                except BaseException:
                    pass
        if cleanup_error is not None and not primary_failure:
            raise cleanup_error


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
