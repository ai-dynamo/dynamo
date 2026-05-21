# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from gpu_memory_service.client.torch.module import (
        materialize_module_from_gms,
        move_tensor_attrs_out_of_gms,
        register_module_tensors,
    )
    from gpu_memory_service.client.torch.allocator import (
        evict_gms_client_memory_manager,
        get_or_create_gms_client_memory_manager,
        gms_use_mem_pool,
    )
    from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
    from gpu_memory_service.common.locks import RequestedLockType
    from gpu_memory_service.server.rpc import GMSRPCServer
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch", reason="torch is required")

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_1,
]

_SERVER_START_TIMEOUT_SECONDS = 5.0
_SERVER_STOP_TIMEOUT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.01


if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for torch GMS integration tests", allow_module_level=True
    )


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


class _TinyModuleWithEmptyTensors(torch.nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(4, 4, device=device, dtype=torch.float32)
        )
        self.empty_param = torch.nn.Parameter(
            torch.empty(2, 0, device=device, dtype=torch.int32),
            requires_grad=False,
        )
        self.register_buffer(
            "empty_buffer", torch.empty(0, device=device, dtype=torch.float16)
        )
        self.empty_attr = torch.empty(3, 0, device=device, dtype=torch.float32)


class _TinyWorkspaceModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(4, 4, device="cuda", dtype=torch.float32)
        )
        self.workspace = torch.zeros(8, device="cuda", dtype=torch.int32)
        self.cache = type("_Cache", (), {})()
        self.cache.workspace = self.workspace
        self.helper = type("_Helper", (), {})()
        self.helper.runtime_workspace = torch.arange(
            4, device="cuda", dtype=torch.int32
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
    evict_gms_client_memory_manager(writer)


def test_mutable_workspace_attrs_are_moved_out_of_gms(running_gms):
    socket_path = running_gms
    writer = get_or_create_gms_client_memory_manager(
        socket_path,
        0,
        RequestedLockType.RW,
        tag="weights",
    )

    with gms_use_mem_pool("weights", torch.device("cuda", 0)):
        model = _TinyWorkspaceModule()
        expected_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        model.weight.data.copy_(expected_weight)
        model.workspace.copy_(
            torch.arange(8, device="cuda", dtype=torch.int32),
        )

    moved = move_tensor_attrs_out_of_gms(writer, model, device_index=0)

    assert moved == ["workspace", "helper.runtime_workspace"]
    assert model.cache.workspace is model.workspace

    register_module_tensors(writer, model)
    assert writer.commit()
    del model
    writer.close()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    materialized = torch.nn.Module()
    materialized.weight = torch.nn.Parameter(
        torch.empty(4, 4, device="meta", dtype=torch.float32), requires_grad=False
    )
    materialize_module_from_gms(reader, materialized, device_index=0)

    _assert_exact_tensor_equal(
        expected_weight, cast(torch.Tensor, materialized.weight).cpu()
    )
    assert not hasattr(materialized, "workspace")
    assert not hasattr(materialized, "helper")
    reader.close()
    evict_gms_client_memory_manager(writer)


def test_zero_size_tensors_are_skipped_then_materialized(running_gms):
    socket_path = running_gms
    baseline = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    gms_model = _TinyModuleWithEmptyTensors()

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)

    _, gms_weight = _make_gms_tensor(writer, baseline, tag="weights")
    gms_model.weight = torch.nn.Parameter(gms_weight, requires_grad=False)

    # The empty CUDA tensors are regular registered/attr tensors, but carry no
    # payload and therefore have no GMS allocation to record.
    register_module_tensors(writer, gms_model)
    assert writer.commit()
    del gms_weight
    del gms_model
    writer.close()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    materialized = _TinyModuleWithEmptyTensors(device="meta")
    materialize_module_from_gms(reader, materialized, device_index=0)

    _assert_exact_tensor_equal(baseline, cast(torch.Tensor, materialized.weight))
    assert materialized.empty_param.is_cuda
    assert materialized.empty_buffer.is_cuda
    assert materialized.empty_attr.is_cuda
    assert tuple(materialized.empty_param.shape) == (2, 0)
    assert tuple(materialized.empty_buffer.shape) == (0,)
    assert tuple(materialized.empty_attr.shape) == (3, 0)
    assert materialized.empty_param.dtype is torch.int32
    assert materialized.empty_buffer.dtype is torch.float16
    assert materialized.empty_attr.dtype is torch.float32

    reader.close()


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
