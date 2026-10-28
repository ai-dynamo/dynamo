# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GMS V1 loader options and the ModelExpress source identity."""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.cli import snapshot as per_device
from gpu_memory_service.snapshot.backends import mx
from gpu_memory_service.snapshot.transfer import (
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
)
from gpu_memory_service.v1.snapshot import loader, mx_source

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


# A pod's GPUs in PCI bus order, and in the order CUDA numbers them.
_BUS_ORDER = ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
_CUDA_ORDER = ["GPU-b", "GPU-a", "GPU-c", "GPU-d"]


def _load(argv: list[str]):
    """The loader's load_weights call on that pod."""
    identity = loader.device_identity
    with patch.object(loader, "init_vmm"), patch.object(
        loader, "load_weights"
    ) as load_weights, patch.object(
        identity, "gpu_uuid", _BUS_ORDER.__getitem__
    ), patch.object(
        identity, "cuda_ordinal", _CUDA_ORDER.index
    ):
        loader.main(["--checkpoint-dir", "/ckpt", *argv])
    return load_weights.call_args


def _backend_config(argv: list[str]) -> dict:
    return _load(argv).kwargs["backend_config"]


def test_device_n_loads_its_artifact_into_the_nth_gpu_in_bus_order(monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", "/run/gms")

    assert _load(["--device", "1"]).args == (
        "/ckpt/device-1",
        "/run/gms/gms_GPU-b_weights.sock",
        0,
    )


def test_confined_server_passes_its_gpu_instead_of_listing_gpus(monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", "/run/gms")
    args = loader.parse_args(["--checkpoint-dir", "/ckpt", "--device", "1"])
    args.gpu_uuid = "GPU-b"

    def unlisted(_device):
        raise AssertionError("a confined process cannot list the GPUs")

    identity = loader.device_identity
    with patch.object(loader, "init_vmm"), patch.object(
        loader, "load_weights"
    ) as load_weights, patch.object(identity, "gpu_uuid", unlisted), patch.object(
        identity, "cuda_ordinal", _CUDA_ORDER.index
    ):
        loader.load(args)

    assert load_weights.call_args.args[1] == "/run/gms/gms_GPU-b_weights.sock"


def test_backend_config_carries_storage_backend_settings():
    assert _backend_config(
        [
            "--sharded-ssd-roots",
            "/ssd0,/ssd1",
            "--sharded-ssd-queues-per-root",
            "4",
            "--posix-backend-param",
            "ios_pool_size=64",
        ]
    ) == {
        "sharded_ssd_roots": ["/ssd0", "/ssd1"],
        "sharded_ssd_queues_per_root": 4,
        "posix_backend_params": {"ios_pool_size": "64"},
    }


def test_mx_device_selects_peer_rank_and_listen_port():
    config = _backend_config(
        ["--transfer-backend", "mx", "--mx-source-key", "ckpt-1", "--device", "3"]
    )

    assert config["mx_rank"] == 3
    assert config["mx_source_key"] == "ckpt-1"
    assert config["mx_server_url"] is None
    assert config["mx_listen_port"] == 25558


def test_mx_needs_a_source_key():
    with pytest.raises(SystemExit):
        _backend_config(["--transfer-backend", "mx"])


def test_mx_source_serves_every_device_by_default(monkeypatch):
    launched = []

    class _Exited:
        def __init__(self, argv) -> None:
            launched.append(argv[3:])
            self.pid = len(launched)

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    class _Vmm:
        def list_devices(self) -> list[int]:
            return [0, 1, 2]

    monkeypatch.setattr(per_device, "init_vmm", lambda _device_type: None)
    monkeypatch.setattr(per_device, "get_vmm", _Vmm)
    monkeypatch.setattr(per_device.subprocess, "Popen", _Exited)

    mx_source.main(["--source-key", "ckpt-1"])

    assert launched == [
        ["--source-key", "ckpt-1", "--device", str(device)] for device in range(3)
    ]


@pytest.mark.skipif(
    importlib.util.find_spec("modelexpress") is None, reason="modelexpress required"
)
def test_source_key_is_the_whole_identity():
    identity = mx.source_identity("ckpt-1")

    assert identity.model_name == "ckpt-1"
    assert identity.tensor_parallel_size == 0
    assert dict(identity.extra_parameters) == {"memory": "gms-v1"}
    assert mx.source_identity("ckpt-1") == identity
    assert mx.source_identity("ckpt-2") != identity


class _FakeAgent:
    def __init__(self, port: int) -> None:
        self.port = port
        self.shut_down = False

    def shutdown(self) -> None:
        self.shut_down = True


@pytest.mark.skipif(
    importlib.util.find_spec("modelexpress") is None or not HAS_TORCH,
    reason="modelexpress and torch required",
)
def test_mx_restore_retries_the_next_source_on_its_listen_port(monkeypatch):
    import torch
    from modelexpress import p2p_pb2
    from modelexpress.adapter import StrategyFailed
    from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
    from modelexpress.metadata import client_factory

    agents: list[_FakeAgent] = []

    def start_agent(_rank, _device, port):
        agents.append(_FakeAgent(port))
        return agents[-1]

    tried = []

    def load_as_target(_self, result, ctx, _worker, _source_id, worker_id):
        tried.append((worker_id, ctx.nixl_manager.port))
        if worker_id == "peer-0":
            raise StrategyFailed("peer-0 broke mid-transfer", mutated=True)
        return result

    monkeypatch.setattr(mx, "_start_nixl_manager", start_agent)
    monkeypatch.setattr(
        mx,
        "byte_tensors",
        lambda _extents, _device: {
            mx.allocation_tensor_name(0): torch.zeros(64, dtype=torch.uint8)
        },
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(client_factory, "create_metadata_client", lambda **_: None)
    monkeypatch.setattr(RdmaStrategy, "is_available", lambda _self, _ctx: True)
    monkeypatch.setattr(
        RdmaStrategy,
        "_find_source_instances",
        lambda _self, _ctx: [
            p2p_pb2.SourceInstanceRef(mx_source_id="src", worker_id=f"peer-{i}")
            for i in range(2)
        ],
    )
    monkeypatch.setattr(
        RdmaStrategy,
        "_fetch_worker_metadata",
        lambda *_args: p2p_pb2.WorkerMetadata(),
    )
    monkeypatch.setattr(RdmaStrategy, "_accelerator_compatible", lambda *_args: True)
    monkeypatch.setattr(RdmaStrategy, "_load_as_target", load_as_target)

    backend = mx.MXTransferBackend(
        config=GMSSnapshotConfig(
            device=0,
            max_workers=1,
            backend_config={
                "mx_rank": 2,
                "mx_source_key": "ckpt-1",
                "mx_listen_port": 25557,
            },
        )
    )
    session = backend.start_restore([FileTransferSource("a0", "", 0, 64)])
    session.restore({"a0": GMSTransferTarget("a0", 0x1000, 0, 64)})
    backend.close()

    assert tried == [("peer-0", 25557), ("peer-1", 25557)]
    assert [agent.port for agent in agents] == [25557, 25557]
    assert all(agent.shut_down for agent in agents)
