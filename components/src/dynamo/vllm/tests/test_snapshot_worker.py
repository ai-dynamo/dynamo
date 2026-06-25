# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from dynamo.vllm.flashinfer_snapshot import FlashInferResourceReport

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def snapshot_worker_module(monkeypatch):
    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.rank = 3
            self.local_rank = 1
            self.calls = []
            self.base_sleep_exc = None
            self._sleep_saved_buffers = {}

        def sleep(self, level=1):
            self.calls.append(("base_sleep", level))
            if self.base_sleep_exc is not None:
                raise self.base_sleep_exc

    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    vllm_v1_module = types.ModuleType("vllm.v1")
    vllm_v1_module.__path__ = []
    vllm_worker_module = types.ModuleType("vllm.v1.worker")
    vllm_worker_module.__path__ = []
    gpu_worker_module = types.ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker_module.Worker = FakeWorker
    device_allocator_module = types.ModuleType("vllm.device_allocator")
    device_allocator_module.get_mem_allocator_instance = lambda: None

    monkeypatch.delitem(sys.modules, "dynamo.vllm.snapshot_worker", raising=False)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", vllm_worker_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker_module)
    monkeypatch.setitem(sys.modules, "vllm.device_allocator", device_allocator_module)

    module = importlib.import_module("dynamo.vllm.snapshot_worker")
    try:
        yield module
    finally:
        sys.modules.pop("dynamo.vllm.snapshot_worker", None)


def _empty_report(operation):
    return FlashInferResourceReport(operation=operation, resources=())


def test_snapshot_worker_sleep_pauses_sleeps_synchronizes(
    monkeypatch, snapshot_worker_module
):
    worker = snapshot_worker_module.SnapshotWorker()

    def pause_flashinfer_peer_resources(worker):
        worker.calls.append("pause")
        return _empty_report("pause")

    def synchronize_snapshot_device():
        worker.calls.append("synchronize")
        return "test"

    monkeypatch.setattr(
        snapshot_worker_module,
        "pause_flashinfer_peer_resources",
        pause_flashinfer_peer_resources,
    )
    monkeypatch.setattr(
        snapshot_worker_module,
        "_synchronize_snapshot_device",
        synchronize_snapshot_device,
    )

    worker.sleep(level=2)

    assert worker.calls == ["pause", ("base_sleep", 2), "synchronize"]


def test_snapshot_worker_installs_no_nccl_guard(
    monkeypatch, snapshot_worker_module
):
    calls = []
    monkeypatch.setenv(
        "DYN_VLLM_NO_NCCL_SNAPSHOT",
        "1",
    )
    monkeypatch.setattr(
        snapshot_worker_module,
        "install_no_nccl_snapshot_guard",
        lambda: calls.append("install"),
    )

    snapshot_worker_module.SnapshotWorker()

    assert calls == ["install"]


def test_snapshot_worker_installs_legacy_strict_guard(
    monkeypatch, snapshot_worker_module
):
    calls = []
    monkeypatch.setenv("DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES", "1")
    monkeypatch.setattr(
        snapshot_worker_module,
        "install_no_nccl_snapshot_guard",
        lambda: calls.append("no_nccl"),
    )
    monkeypatch.setattr(
        snapshot_worker_module,
        "install_strict_flashinfer_collective_guard",
        lambda: calls.append("strict"),
    )

    snapshot_worker_module.SnapshotWorker()

    assert calls == ["no_nccl", "strict"]


def test_snapshot_worker_installs_no_nccl_patch_before_base_init(
    monkeypatch, snapshot_worker_module
):
    calls = []
    monkeypatch.setenv("DYN_VLLM_NO_NCCL_SNAPSHOT", "1")

    base_worker = snapshot_worker_module.Worker
    original_init = base_worker.__init__

    def base_init(self, *args, **kwargs):
        calls.append("base_init")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(base_worker, "__init__", base_init)
    monkeypatch.setattr(
        snapshot_worker_module,
        "install_no_nccl_snapshot_guard",
        lambda: calls.append("guard"),
    )

    snapshot_worker_module.SnapshotWorker()

    assert calls == ["guard", "base_init"]


def test_snapshot_worker_child_process_installs_real_no_nccl_guard():
    repo_root = Path(__file__).resolve().parents[5]
    script = r"""
import json
import os
import sys
import types

vllm_module = types.ModuleType("vllm")
vllm_module.__path__ = []
vllm_v1_module = types.ModuleType("vllm.v1")
vllm_v1_module.__path__ = []
vllm_worker_module = types.ModuleType("vllm.v1.worker")
vllm_worker_module.__path__ = []
gpu_worker_module = types.ModuleType("vllm.v1.worker.gpu_worker")

class Worker:
    def __init__(self, *args, **kwargs):
        self.rank = 0
        self.local_rank = 0
        self._sleep_saved_buffers = {}

gpu_worker_module.Worker = Worker

device_allocator_module = types.ModuleType("vllm.device_allocator")
device_allocator_module.get_mem_allocator_instance = lambda: None

cuda_communicator_module = types.ModuleType(
    "vllm.distributed.device_communicators.cuda_communicator"
)

class CudaCommunicator:
    def all_reduce(self, tensor):
        return "unsafe-all-reduce"

    def send(self, tensor, dst=None):
        return "unsafe-send"

    def recv(self, tensor, src=None):
        return "unsafe-recv"

    def broadcast(self, tensor, src=0):
        return "unsafe-broadcast"

    def gather(self, tensor, dst=0):
        return "unsafe-gather"

cuda_communicator_module.CudaCommunicator = CudaCommunicator

torch_module = types.ModuleType("torch")
torch_module.__path__ = []
torch_distributed_module = types.ModuleType("torch.distributed")

def _noop_collective(*args, **kwargs):
    return None

torch_distributed_module.broadcast = _noop_collective
torch_module.distributed = torch_distributed_module

sys.modules.update(
    {
        "vllm": vllm_module,
        "vllm.v1": vllm_v1_module,
        "vllm.v1.worker": vllm_worker_module,
        "vllm.v1.worker.gpu_worker": gpu_worker_module,
        "vllm.device_allocator": device_allocator_module,
        "vllm.distributed.device_communicators.cuda_communicator": (
            cuda_communicator_module
        ),
        "torch": torch_module,
        "torch.distributed": torch_distributed_module,
    }
)

os.environ["DYN_VLLM_NO_NCCL_SNAPSHOT"] = "1"
from dynamo.vllm.snapshot_worker import SnapshotWorker

SnapshotWorker()
comm = CudaCommunicator()
comm.world_size = 2
comm.unique_name = "tp:0"
comm.rank = 0
comm.rank_in_group = 0
comm.use_flashinfer_allreduce = True
comm.fi_ar_comm = None

result = {
    "guard_installed": getattr(
        CudaCommunicator,
        "_dynamo_no_nccl_snapshot_guard_installed",
        False,
    ),
    "blocked": {},
}
for method_name in ("all_reduce", "send", "recv", "broadcast", "gather"):
    try:
        getattr(comm, method_name)(object())
    except RuntimeError as exc:
        result["blocked"][method_name] = str(exc)
    else:
        result["blocked"][method_name] = None

print(json.dumps(result, sort_keys=True))
"""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "components" / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
    )

    result = json.loads(completed.stdout)
    assert result["guard_installed"] is True
    assert set(result["blocked"]) == {
        "all_reduce",
        "send",
        "recv",
        "broadcast",
        "gather",
    }
    for method_name, message in result["blocked"].items():
        assert method_name in message
        assert "Dynamo no-NCCL vLLM snapshot mode blocked" in message


def test_snapshot_worker_sleep_rolls_back_when_base_sleep_fails(
    monkeypatch, snapshot_worker_module
):
    worker = snapshot_worker_module.SnapshotWorker()
    original_error = RuntimeError("base sleep failed")
    worker.base_sleep_exc = original_error

    def pause_flashinfer_peer_resources(worker):
        worker.calls.append("pause")
        return _empty_report("pause")

    def resume_flashinfer_peer_resources(worker):
        worker.calls.append("resume")
        return _empty_report("resume")

    monkeypatch.setattr(
        snapshot_worker_module,
        "pause_flashinfer_peer_resources",
        pause_flashinfer_peer_resources,
    )
    monkeypatch.setattr(
        snapshot_worker_module,
        "resume_flashinfer_peer_resources",
        resume_flashinfer_peer_resources,
    )
    monkeypatch.setattr(
        snapshot_worker_module,
        "_synchronize_snapshot_device",
        lambda: pytest.fail("unexpected synchronize after base sleep failure"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        worker.sleep(level=3)

    assert exc_info.value is original_error
    assert worker.calls == ["pause", ("base_sleep", 3), "resume"]


def test_snapshot_worker_sleep_rolls_back_when_synchronize_fails(
    monkeypatch, snapshot_worker_module
):
    worker = snapshot_worker_module.SnapshotWorker()
    original_error = RuntimeError("device synchronize failed")

    def pause_flashinfer_peer_resources(worker):
        worker.calls.append("pause")
        return _empty_report("pause")

    def resume_flashinfer_peer_resources(worker):
        worker.calls.append("resume")
        return _empty_report("resume")

    def synchronize_snapshot_device():
        worker.calls.append("synchronize")
        raise original_error

    monkeypatch.setattr(
        snapshot_worker_module,
        "pause_flashinfer_peer_resources",
        pause_flashinfer_peer_resources,
    )
    monkeypatch.setattr(
        snapshot_worker_module,
        "resume_flashinfer_peer_resources",
        resume_flashinfer_peer_resources,
    )
    monkeypatch.setattr(
        snapshot_worker_module,
        "_synchronize_snapshot_device",
        synchronize_snapshot_device,
    )

    with pytest.raises(RuntimeError) as exc_info:
        worker.sleep(level=4)

    assert exc_info.value is original_error
    assert worker.calls == ["pause", ("base_sleep", 4), "synchronize", "resume"]
