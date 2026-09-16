# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip("gpu_memory_service is required", allow_module_level=True)
if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)
pytest.importorskip("vllm")

from gpu_memory_service.integrations.vllm import worker as gms_worker  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def test_init_device_forwards_ro_connect_timeout(monkeypatch, tmp_path):
    manager_factory = Mock()
    monkeypatch.setattr(
        gms_worker, "get_or_create_gms_client_memory_manager", manager_factory
    )
    monkeypatch.setattr(gms_worker, "_get_dp_adjusted_local_rank", lambda *_: 0)
    monkeypatch.setattr(
        gms_worker, "get_vmm_device_type", lambda: SimpleNamespace(value="cuda")
    )
    monkeypatch.setattr(
        gms_worker, "get_socket_path", lambda *_: str(tmp_path / "weights.sock")
    )
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(set_device=lambda _: None),
    )
    monkeypatch.setattr(gms_worker._BaseWorker, "init_device", lambda _: None)

    worker = gms_worker.GMSWorker.__new__(gms_worker.GMSWorker)
    worker.local_rank = 0
    worker.parallel_config = None
    worker.vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(
            model_loader_extra_config={
                "gms_read_only": True,
                "gms_ro_connect_timeout_ms": 4200,
            }
        )
    )

    worker.init_device()

    assert manager_factory.call_args.kwargs["timeout_ms"] == 4200
