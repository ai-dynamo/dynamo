# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM GMS worker's weights-admission deadline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

pytest.importorskip("vllm", reason="vLLM is required")

from gpu_memory_service.common.locks import RequestedLockType  # noqa: E402
from gpu_memory_service.common.vmm import VMMDeviceType  # noqa: E402
from gpu_memory_service.integrations.vllm import worker as gms_worker  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]

_RO_CONNECT_TIMEOUT_MS = 4200


@pytest.fixture
def init_device_calls(monkeypatch, tmp_path):
    """Run ``GMSWorker.init_device`` with every device collaborator replaced.

    Returns the keyword arguments the worker handed to the client-manager
    factory.
    """
    calls: list[dict[str, object]] = []

    def fake_factory(socket_path, device, mode=None, *, tag=None, timeout_ms=None):
        calls.append(
            {
                "socket_path": socket_path,
                "device": device,
                "mode": mode,
                "tag": tag,
                "timeout_ms": timeout_ms,
            }
        )
        return SimpleNamespace()

    monkeypatch.setattr(
        gms_worker, "get_or_create_gms_client_memory_manager", fake_factory
    )
    monkeypatch.setattr(gms_worker, "get_vmm_device_type", lambda: VMMDeviceType.CUDA)
    # Socket discovery queries NVML for a GPU UUID, which gpu_0 runners lack.
    monkeypatch.setattr(
        gms_worker,
        "get_socket_path",
        lambda device, tag: str(tmp_path / f"gms_{device}_{tag}.sock"),
    )
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(set_device=lambda _device: None),
    )
    # The parent's device init allocates on a real GPU; this test is only about
    # the arguments GMS resolves before it runs.
    monkeypatch.setattr(gms_worker._BaseWorker, "init_device", lambda self: None)

    def run(extra_config: dict) -> list[dict[str, object]]:
        worker = gms_worker.GMSWorker.__new__(gms_worker.GMSWorker)
        worker.local_rank = 0
        worker.parallel_config = SimpleNamespace(
            distributed_executor_backend="mp",
            data_parallel_backend="mp",
            nnodes_within_dp=1,
            data_parallel_rank_local=0,
            data_parallel_index=0,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        worker.vllm_config = SimpleNamespace(
            load_config=SimpleNamespace(model_loader_extra_config=extra_config)
        )
        worker.init_device()
        return calls

    return run


def test_init_device_forwards_the_resolved_ro_connect_timeout(init_device_calls):
    """A shadow's weights admission must carry the deadline, not drop it.

    ``init_device`` resolves ``gms_ro_connect_timeout_ms`` for its own use on
    the wake path. If it does not also hand the value to the factory, the very
    first weights handshake is unbounded, and a shadow refused admission parks
    there with no log and no traceback.
    """
    calls = init_device_calls(
        {
            "gms_read_only": True,
            "gms_ro_connect_timeout_ms": _RO_CONNECT_TIMEOUT_MS,
        }
    )

    assert len(calls) == 1
    assert calls[0]["tag"] == "weights"
    assert calls[0]["mode"] == RequestedLockType.RO
    assert calls[0]["timeout_ms"] == _RO_CONNECT_TIMEOUT_MS
