# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
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

    def no_cuda(_device_type):
        raise AssertionError("listing devices must not start CUDA")

    monkeypatch.setattr(per_device, "init_vmm", no_cuda)
    monkeypatch.setattr(mx_source.device_identity, "node_gpus", lambda: [object()] * 3)
    monkeypatch.setattr(per_device.subprocess, "Popen", _Exited)

    mx_source.main(["--source-key", "ckpt-1"])

    assert launched == [
        ["--source-key", "ckpt-1", "--device", str(device)] for device in range(3)
    ]


def _wait_for(paths, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not all(path.exists() for path in paths):
        assert time.monotonic() < deadline, f"timed out waiting for {paths}"
        time.sleep(0.05)


@pytest.mark.timeout(60)
def test_sigterm_stops_each_device_process_and_waits_for_its_cleanup(tmp_path):
    # Stands in for one device's process, e.g. an MX source that must
    # unpublish before it exits. It quits on its own if orphaned.
    (tmp_path / "device_child.py").write_text(
        textwrap.dedent(
            """
            import os, pathlib, signal, sys, time
            out = pathlib.Path(sys.argv[1])
            device = sys.argv[sys.argv.index("--device") + 1]
            parent = os.getppid()

            def stop(*_):
                time.sleep(0.2)
                (out / f"cleaned-{device}").touch()
                sys.exit(0)

            signal.signal(signal.SIGTERM, stop)
            (out / f"ready-{device}").touch()
            while os.getppid() == parent:
                time.sleep(0.05)
            """
        )
    )
    python_path = [str(tmp_path), os.environ.get("PYTHONPATH", "")]
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from gpu_memory_service.cli.snapshot import run_per_device\n"
            "run_per_device('device_child', sys.argv[1:], [0, 1])",
            str(tmp_path),
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(python_path)},
    )
    try:
        _wait_for([tmp_path / "ready-0", tmp_path / "ready-1"], timeout=30)
        launcher.send_signal(signal.SIGTERM)
        assert launcher.wait(timeout=30) == 128 + signal.SIGTERM
        assert (tmp_path / "cleaned-0").exists()
        assert (tmp_path / "cleaned-1").exists()
    finally:
        launcher.kill()
        launcher.wait()


def test_mx_source_fails_without_gpus(monkeypatch):
    monkeypatch.setattr(mx_source.device_identity, "node_gpus", lambda: [])

    with pytest.raises(SystemExit, match="no GPU"):
        mx_source.main(["--source-key", "ckpt-1"])


@pytest.mark.skipif(
    importlib.util.find_spec("modelexpress") is None or not HAS_TORCH,
    reason="modelexpress and torch required",
)
@pytest.mark.parametrize("publish_error", [None, RuntimeError("port in use")])
def test_mx_source_stops_advertising_before_releasing_its_weights(
    monkeypatch, publish_error
):
    import threading

    import torch
    from modelexpress.load_strategy import base
    from modelexpress.metadata import publish

    events = []
    ctx = SimpleNamespace(
        mx_client=None,
        nixl_manager=None,
        tensors={},
        worker_rank=0,
        device_id=0,
        identity=None,
        worker_id="gms-test",
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    def register(_model, context, reuse_discovered):
        context.nixl_manager = SimpleNamespace(
            shutdown=lambda: events.append("shutdown agent")
        )

    def publish_and_ready(*_args, **_kwargs):
        events.append("publish")
        if publish_error is not None:
            raise publish_error

    vmm = SimpleNamespace(
        ensure_initialized=lambda: None,
        runtime_set_device=lambda _device: None,
        get_allocation_granularity=lambda _device: 2**21,
    )
    session = SimpleNamespace(close=lambda: events.append("close session"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(mx_source, "get_vmm", lambda: vmm)
    monkeypatch.setattr(mx_source, "_GMSClientSession", lambda *_args: session)
    monkeypatch.setattr(mx_source, "_verify_session_device", lambda *_args: None)
    monkeypatch.setattr(
        mx_source,
        "_list_allocations",
        lambda _session: [SimpleNamespace(aligned_size=2**21)],
    )
    monkeypatch.setattr(
        mx_source, "_map_export", lambda *_args: [SimpleNamespace(base=0x1000)]
    )
    monkeypatch.setattr(
        mx_source, "_release_mapping", lambda *_args: events.append("unmap")
    )
    monkeypatch.setattr(mx_source, "byte_tensors", lambda *_args: {})
    monkeypatch.setattr(mx_source, "mx_load_context", lambda *_args: ctx)
    monkeypatch.setattr(base, "register_tensors", register)
    monkeypatch.setattr(
        base, "unpublish_metadata", lambda _ctx: events.append("unpublish")
    )
    monkeypatch.setattr(publish, "publish_metadata_and_ready", publish_and_ready)
    stop = threading.Event()
    stop.set()

    if publish_error is None:
        mx_source.serve("/tmp/gms.sock", 0, 0, {}, stop)
    else:
        with pytest.raises(RuntimeError, match="port in use"):
            mx_source.serve("/tmp/gms.sock", 0, 0, {}, stop)

    assert events == [
        "publish",
        "unpublish",
        "shutdown agent",
        "unmap",
        "close session",
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
