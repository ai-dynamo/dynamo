# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local retained GPU banks; native transfer qualification is separate."""

import base64
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from kvbm.vllm_integration import gpu_memory, gpu_mnnvl
from kvbm.vllm_integration.gpu_memory import (
    GPUCacheCopyHandler,
    GPUCacheWorker,
    GPUMultiRankCacheRegion,
)
from kvbm.vllm_integration.gpu_offloading import GPURetainedLoadStoreSpec
from vllm.v1.kv_offload.base import GPULoadStoreSpec

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.kvbm,
]


@pytest.fixture
def fake_banks(monkeypatch, tmp_path):
    state = SimpleNamespace(device=0, created=[])

    def identity(device):
        return gpu_mnnvl.FabricIdentity(f"GPU-{uuid.UUID(int=device + 1)}", "1" * 32, 1)

    @contextmanager
    def device_context(device):
        prior = state.device
        state.device = device if type(device) is int else device.index
        try:
            yield
        finally:
            state.device = prior

    class Bank:
        gpu_transport = "mnnvl"

        def __init__(self, engine_id, chunks, width, base_domain, *, _path):
            self.path = _path
            self.path.mkdir(mode=0o700)
            assert not (self.path.parent / "group.json").exists()
            self.shape, self.strides = (chunks, width), (width, 1)
            self.tensor = SimpleNamespace(device=SimpleNamespace(index=state.device))
            self._closed = False
            observed = identity(state.device)
            self.fabric_domain = observed.domain
            self.compatibility_domain = gpu_mnnvl.effective_domain(
                base_domain, self.fabric_domain
            )
            self.metadata = {
                "schema": "dynamo-gpu-fabric-v1",
                "gpu_transport": "mnnvl",
                "engine_id": engine_id,
                "owner_pid": os.getpid(),
                "owner_start_ticks": 17,
                "gpu_uuid": observed.gpu_uuid,
                "fabric_domain": self.fabric_domain,
                "base_domain": base_domain,
                "compatibility_domain": self.compatibility_domain,
                "shape": list(self.shape),
                "stride": list(self.strides),
                "logical_bytes": chunks * width,
                "allocation_bytes": chunks * width,
                "fabric_handle": base64.b64encode(b"x" * 64).decode(),
            }
            descriptor = self.path / "ipc.json"
            descriptor.write_text(json.dumps(self.metadata))
            descriptor.chmod(0o600)
            gpu_memory._OWNED_GPU_REGIONS[self.path] = self
            state.created.append(self)

        def cleanup(self):
            if not self._closed:
                (self.path / "ipc.json").unlink()
                self.path.rmdir()
                self.tensor = None
                self._closed = True

    monkeypatch.setattr(gpu_memory.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(gpu_memory.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(gpu_memory.torch.cuda, "device", device_context)
    monkeypatch.setattr(gpu_memory.torch.cuda, "current_device", lambda: state.device)
    monkeypatch.setattr(gpu_memory, "_process_start_ticks", lambda pid: 17)
    monkeypatch.setattr(gpu_memory, "_OWNED_GPU_REGIONS", {})
    monkeypatch.setattr(gpu_memory, "_RETAINED_FAILED_GPU_GROUPS", [])
    monkeypatch.setattr(gpu_mnnvl, "observe_fabric", identity)
    monkeypatch.setattr(gpu_memory, "GPUFabricCacheRegion", Bank)
    return SimpleNamespace(state=state, identity=identity, bank_type=Bank)


@pytest.fixture
def bank_group(fake_banks):
    group = GPUMultiRankCacheRegion(
        "rank-bank-test", 4, 24, "test-layout", gpu_rank_devices=[1, 0]
    )
    try:
        yield group
    finally:
        group.cleanup()


@pytest.mark.gpu_0
def test_rank_bank_group_commits_all_banks_and_explicit_device_order(bank_group):
    assert bank_group.shape == (4, 48)
    assert bank_group.strides == (48, 1)
    assert not hasattr(bank_group, "tensor")
    manifest = json.loads((bank_group.path / "group.json").read_bytes())
    assert manifest == bank_group.metadata
    assert [entry["device"] for entry in manifest["banks"]] == [1, 0]
    for rank, bank in enumerate(bank_group.banks):
        assert bank.path == bank_group.path / f"rank-{rank}"
        assert bank.shape == (4, 24)
        entry = manifest["banks"][rank]
        assert entry["rank"] == rank
        assert entry["world_size"] == 2
        assert entry["owner_pid"] == os.getpid()
        assert entry["owner_start_ticks"] == 17
        assert (
            entry["descriptor_sha256"]
            == hashlib.sha256((bank.path / "ipc.json").read_bytes()).hexdigest()
        )
    # A legitimate TP1 engine name ending in ':rank0' occupies another hash.
    assert gpu_memory.region_path("rank-bank-test:rank0") != bank_group.banks[0].path


@pytest.mark.gpu_0
@pytest.mark.parametrize("devices", [None, [], [0], [0, 0], [True, 1], [-1, 0], [0, 2]])
def test_rank_bank_group_rejects_invalid_catalog_before_allocation(fake_banks, devices):
    with pytest.raises(ValueError):
        GPUMultiRankCacheRegion("bad-catalog", 4, 24, gpu_rank_devices=devices)
    assert fake_banks.state.created == []


@pytest.mark.gpu_0
def test_worker_import_selects_rank_then_checks_actual_gpu(
    bank_group, fake_banks, monkeypatch
):
    monkeypatch.setattr(gpu_memory, "_OWNED_GPU_REGIONS", {})
    imported = []
    marker = object()

    def import_selected(data, engine_id, shape, *domains):
        imported.append((data, engine_id, shape, domains))
        return marker

    monkeypatch.setattr(gpu_memory, "import_fabric", import_selected)
    # rank1 deliberately maps to device0; rank is not a CUDA device ordinal.
    result = gpu_memory.open_region(
        "rank-bank-test",
        (4, 24),
        rank=1,
        world_size=2,
        gpu_transport="mnnvl",
        base_domain="test-layout",
        compatibility_domain=bank_group.compatibility_domain,
        fabric_domain=bank_group.fabric_domain,
    )
    assert result is marker
    assert len(imported) == 1
    assert imported[0][0]["gpu_uuid"] == fake_banks.identity(0).gpu_uuid
    assert imported[0][2] == (4, 24)
    with pytest.raises(ValueError, match="same physical GPU"):
        gpu_memory.open_region(
            "rank-bank-test",
            (4, 24),
            rank=0,
            world_size=2,
            gpu_transport="mnnvl",
            base_domain="test-layout",
            compatibility_domain=bank_group.compatibility_domain,
            fabric_domain=bank_group.fabric_domain,
        )
    assert len(imported) == 1


@pytest.mark.gpu_0
@pytest.mark.parametrize("change", ["rank", "width", "hash", "owner", "extra"])
def test_group_mismatch_rejects_before_any_handle_import(
    bank_group, monkeypatch, change
):
    manifest = json.loads((bank_group.path / "group.json").read_bytes())
    if change == "rank":
        manifest["banks"][1]["rank"] = 0
    elif change == "width":
        manifest["banks"][1]["shape"][1] += 1
    elif change == "hash":
        manifest["banks"][0]["descriptor_sha256"] = "0" * 64
    elif change == "owner":
        manifest["owner_start_ticks"] += 1
    else:
        manifest["arbitrary_path"] = "untrusted"
    (bank_group.path / "group.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(gpu_memory, "_OWNED_GPU_REGIONS", {})
    monkeypatch.setattr(
        gpu_memory,
        "import_fabric",
        lambda *args: pytest.fail("Invalid group must fail before handle import"),
    )
    with pytest.raises(ValueError):
        gpu_memory.open_region(
            "rank-bank-test",
            (4, 24),
            rank=1,
            world_size=2,
            gpu_transport="mnnvl",
            base_domain="test-layout",
            compatibility_domain=bank_group.compatibility_domain,
            fabric_domain=bank_group.fabric_domain,
        )


@pytest.mark.gpu_0
def test_partial_allocation_never_publishes_group_and_cleans_ready_bank(
    fake_banks, monkeypatch
):
    def allocate(*args, **kwargs):
        if fake_banks.state.created:
            raise RuntimeError("second bank allocation failed")
        return fake_banks.bank_type(*args, **kwargs)

    monkeypatch.setattr(gpu_memory, "GPUFabricCacheRegion", allocate)
    with pytest.raises(RuntimeError, match="second bank"):
        GPUMultiRankCacheRegion(
            "partial-allocation", 4, 24, "test-layout", gpu_rank_devices=[0, 1]
        )
    assert fake_banks.state.created[0]._closed
    assert not gpu_memory.region_path("partial-allocation").exists()


@pytest.mark.gpu_0
def test_failed_cleanup_retains_bank_owner_and_withdraws_group(bank_group, monkeypatch):
    bank = bank_group.banks[0]
    original_cleanup = bank.cleanup

    def fail_cleanup():
        raise RuntimeError("CUDA drain failed")

    monkeypatch.setattr(bank, "cleanup", fail_cleanup)
    with pytest.raises(RuntimeError, match="CUDA drain"):
        bank_group.cleanup()
    assert any(owner is bank_group for owner in gpu_memory._RETAINED_FAILED_GPU_GROUPS)
    assert bank.tensor is not None
    assert not (bank_group.path / "group.json").exists()
    monkeypatch.setattr(bank, "cleanup", original_cleanup)


@pytest.mark.gpu_0
def test_rank_local_copy_still_rejects_cross_gpu_payload():
    original = SimpleNamespace(
        is_cuda=True, dtype=torch.int8, device=torch.device("cuda", 0), shape=(4, 8)
    )
    cached = SimpleNamespace(
        is_cuda=True, dtype=torch.int8, device=torch.device("cuda", 1), shape=(4, 8)
    )
    with pytest.raises(ValueError, match="one GPU"):
        GPUCacheCopyHandler([original], [cached], 1, [], True)


@pytest.mark.gpu_0
def test_worker_passes_rank_identity_and_uses_only_local_row_stride(monkeypatch):
    opened, views, handlers = [], [], []

    class ByteTensor:
        def view(self, *args):
            return self

    def view(shape, strides, offset):
        views.append((shape, strides, offset))
        return object()

    def open_selected(*args, **kwargs):
        opened.append((args, kwargs))
        return SimpleNamespace(as_strided=view)

    def handler(*args):
        handlers.append(args)
        return object()

    monkeypatch.setattr(gpu_memory, "open_region", open_selected)
    monkeypatch.setattr(gpu_memory, "GPUCacheCopyHandler", handler)
    groups = [[SimpleNamespace(tensor_idx=0, page_size_bytes=3)]]
    caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(tensor=ByteTensor(), page_size_bytes=3),
            SimpleNamespace(tensor=ByteTensor(), page_size_bytes=5),
        ],
        group_data_refs=groups,
    )
    spec = SimpleNamespace(
        config=SimpleNamespace(
            engine_id="worker-mapping", parallel=SimpleNamespace(rank=1, world_size=2)
        ),
        gpu_transport="cuda_ipc",
        num_chunks=7,
        row_bytes=16,
        logical_row_bytes=32,
        blocks_per_chunk=2,
    )
    worker = GPUCacheWorker(spec, caches)
    worker._initialize()
    assert opened == [(("worker-mapping", (7, 16)), {"rank": 1, "world_size": 2})]
    assert views == [((7, 6), (16, 1), 0), ((7, 10), (16, 1), 6)]
    assert len(handlers) == 2
    assert all(arguments[3] is groups for arguments in handlers)
    assert [arguments[4] for arguments in handlers] == [True, False]
    spec.row_bytes = 18
    with pytest.raises(ValueError, match="does not cover"):
        GPUCacheWorker(spec, caches)._initialize()


@pytest.mark.gpu_0
def test_rank_descriptor_symlink_is_rejected_before_read(fake_banks, tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    target.chmod(0o600)
    link = tmp_path / "descriptor.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        gpu_memory._read_private_json(link)


@pytest.mark.gpu_2
@pytest.mark.timeout(60)
def test_two_rank_local_d2d_roundtrip_shared_slots_without_peer_copy():
    engine_id = f"two-rank-local-copy-{uuid.uuid4()}"
    group = GPUMultiRankCacheRegion(
        engine_id, 6, 48, gpu_rank_devices=[1, 0], gpu_transport="cuda_ipc"
    )
    workers = []
    try:
        for rank, device in enumerate((1, 0)):
            with torch.cuda.device(device):
                originals = [
                    torch.full(
                        (8, width),
                        11 + rank + index * 20,
                        dtype=torch.int8,
                        device="cuda",
                    )
                    for index, width in enumerate((8, 16))
                ]
                expected = [tensor.clone() for tensor in originals]
                caches = SimpleNamespace(
                    tensors=[
                        SimpleNamespace(tensor=tensor, page_size_bytes=tensor.shape[1])
                        for tensor in originals
                    ],
                    group_data_refs=[
                        [SimpleNamespace(tensor_idx=0, page_size_bytes=8)],
                        [SimpleNamespace(tensor_idx=1, page_size_bytes=16)],
                    ],
                )
                spec = SimpleNamespace(
                    config=SimpleNamespace(
                        engine_id=engine_id,
                        parallel=SimpleNamespace(rank=rank, world_size=2),
                    ),
                    gpu_transport="cuda_ipc",
                    num_chunks=6,
                    row_bytes=48,
                    blocks_per_chunk=2,
                )
                worker = GPUCacheWorker(spec, caches)
                workers.append((device, worker))
                source = GPULoadStoreSpec([2, 0, 3, 1], [2, 2], [0, 0])
                slots = GPURetainedLoadStoreSpec([4, 1])
                assert worker.submit_store(1, source, slots)
                worker.wait({1})
                assert [result.job_id for result in worker.get_finished()] == [1]
                if rank == 0:
                    with torch.cuda.device(0):
                        assert bool(torch.all(group.banks[1].tensor == 0).item())
                originals[0][4:6].zero_()
                originals[1][6:8].zero_()
                target = GPULoadStoreSpec([5, 4, 6, 7], [2, 2], [0, 0])
                assert worker.submit_load(2, slots, target)
                worker.wait({2})
                assert [result.job_id for result in worker.get_finished()] == [2]
                assert torch.equal(originals[0][5], expected[0][2])
                assert torch.equal(originals[0][4], expected[0][0])
                assert torch.equal(originals[1][6], expected[1][3])
                assert torch.equal(originals[1][7], expected[1][1])
                assert bool(torch.all(group.banks[rank].tensor[0] == 0).item())
    finally:
        for device, worker in workers:
            with torch.cuda.device(device):
                worker.shutdown()
        group.cleanup()
