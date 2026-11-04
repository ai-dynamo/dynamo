# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU retained-cache configuration, source isolation and real local D2D copies.

The CUDA tests run only on allocated GPUs. They compare GPU tensors directly;
only scalar verdicts return to the host. Remote KVCR/NIXL tests are separate.
"""

import base64
import os
import subprocess
import sys
import uuid
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from kvbm.vllm_integration import gpu_memory, gpu_mnnvl, gpu_offloading
from kvbm.vllm_integration.gpu_memory import GPUCacheRegion, GPUCacheWorker, import_ipc
from kvbm.vllm_integration.gpu_offloading import (
    GPUKVCRSecondaryTierManager,
    GPUOffloadingConnector,
    GPURetainedLoadStoreSpec,
)
from kvcr.types import BlockKey
from vllm.distributed.kv_events import BlockRemoved
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_offload.base import GPULoadStoreSpec, ReqContext, make_offload_key

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
]


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("native_failure", [False, True])
def test_gpu_kvcr_rank_registration_and_failed_start_preserve_bank_owners(
    monkeypatch, tp_size, native_failure
):
    spec = SimpleNamespace(
        config=SimpleNamespace(parallel=SimpleNamespace(world_size=tp_size)),
        gpu_transport="cuda_ipc",
        num_chunks=8,
        row_bytes=384,
        logical_row_bytes=384 * tp_size,
        base_domain="test-gpu-layout-v1",
        extra_config={"router_hint_transfer_domain": "test-gpu-layout-v1"},
    )
    banks = [
        SimpleNamespace(
            shape=(spec.num_chunks, spec.row_bytes),
            strides=(spec.row_bytes, 1),
            tensor=SimpleNamespace(
                device=SimpleNamespace(index=tp_size - rank - 1),
                data_ptr=lambda rank=rank: 4096 * (rank + 1),
            ),
        )
        for rank in range(tp_size)
    ]
    region = (
        banks[0]
        if tp_size == 1
        else SimpleNamespace(
            shape=(spec.num_chunks, spec.logical_row_bytes),
            strides=(spec.logical_row_bytes, 1),
            banks=banks,
        )
    )
    listener = MagicMock()
    listener.__enter__.return_value = listener
    listener.getsockname.return_value = ("127.0.0.1", 49152)
    socket_factory = MagicMock(return_value=listener)
    control = object()
    control_factory = MagicMock(return_value=control)
    client = object()
    forwarded = []

    def create_kvcr(config, bindings, backends):
        listener.__exit__.assert_called_once_with(None, None, None)
        assert config.nixl_listen_port == 49152
        assert bindings.framework_control is control
        forwarded.append(backends)
        if native_failure:
            raise RuntimeError("native startup failed after registration")
        return client

    monkeypatch.setattr(gpu_offloading.socket, "socket", socket_factory)
    monkeypatch.setattr(gpu_offloading, "ZmqPeerControlChannel", control_factory)
    monkeypatch.setattr(gpu_offloading, "KVCR", create_kvcr)
    retained = []
    monkeypatch.setattr(gpu_offloading, "_RETAINED_FAILED_GPU_MANAGERS", retained)
    expected = (
        pytest.raises(RuntimeError, match="native startup failed")
        if native_failure
        else nullcontext()
    )
    with expected:
        manager = GPUKVCRSecondaryTierManager(
            spec,
            region,
            {
                "compatibility_domain": spec.base_domain,
                "control_ports": [49153],
                "control_host": "0.0.0.0",
                "control_advertise_host": "gpu-cache.example",
                "router_capabilities": ["router_hint_gpu_direct"],
            },
        )

    socket_factory.assert_called_once_with()
    listener.bind.assert_called_once_with(("127.0.0.1", 0))
    listener.getsockname.assert_called_once_with()
    control_factory.assert_called_once_with("0.0.0.0", 49153, "gpu-cache.example")
    if native_failure:
        assert len(retained) == 1
        assert retained[0]._region is region
    else:
        assert manager._kvcr is client
        assert retained == []
    assert [
        (entry.label, entry.device_Id, entry.addr, entry.count, entry.size)
        for entry in forwarded[0].framework_regions
    ] == [
        (
            "" if tp_size == 1 else f":rank{rank}",
            tp_size - rank - 1,
            4096 * (rank + 1),
            spec.num_chunks,
            spec.row_bytes,
        )
        for rank in range(tp_size)
    ]


@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_gpu_inventory_has_its_own_medium(monkeypatch):
    event = BlockRemoved(block_hashes=[123], medium="CPU")
    monkeypatch.setattr(OffloadingConnector, "take_events", lambda self: iter([event]))
    connector = GPUOffloadingConnector.__new__(GPUOffloadingConnector)
    result = list(connector.take_events())
    assert result[0].medium == "gpu_direct"
    assert event.medium == "CPU"


@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_ipc_rejects_different_geometry_before_touching_cuda():
    with pytest.raises(ValueError, match="layout"):
        import_ipc({"schema": 1, "shape": [9, 384]}, (8, 384))


@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_uniprocess_worker_uses_owned_gpu_storage_without_self_ipc(monkeypatch):
    engine_id = f"uniprocess-test-{uuid.uuid4()}"
    tensor = object()
    owner = SimpleNamespace(
        gpu_transport="cuda_ipc",
        shape=(8, 384),
        tensor=tensor,
        _closed=False,
        consume_local_export=lambda: None,
    )
    monkeypatch.setitem(
        gpu_memory._OWNED_GPU_REGIONS, gpu_memory.region_path(engine_id), owner
    )
    assert gpu_memory.open_region(engine_id, (8, 384)) is tensor
    with pytest.raises(ValueError, match="layout"):
        gpu_memory.open_region(engine_id, (9, 384))
    owner._closed = True
    with pytest.raises(ValueError, match="closed"):
        gpu_memory.open_region(engine_id, (8, 384))


@pytest.mark.gpu_1
def test_local_cuda_owner_balances_unused_ipc_export_once(monkeypatch):
    engine_id = f"local-ipc-refcount-{uuid.uuid4()}"
    region = GPUCacheRegion(engine_id, 4, 256)
    released = []
    original = torch.storage.TypedStorage._release_ipc_counter

    def release(*args, **kwargs):
        original(*args, **kwargs)
        released.append((args, kwargs))

    monkeypatch.setattr(
        torch.storage.TypedStorage, "_release_ipc_counter", staticmethod(release)
    )
    try:
        alias = gpu_memory.open_region(engine_id, region.shape)
        assert alias is region.tensor
        assert gpu_memory.open_region(engine_id, region.shape) is region.tensor
        assert len(released) == 1
        assert not (region.path / "ipc.json").exists()
        region.tensor.fill_(29)
        assert bool(torch.all(region.tensor == 29).item())
        region.cleanup()
        assert region.tensor is None
        # A local worker alias retains its own storage after the closed native
        # endpoint drops the primary owner. No payload crosses host memory.
        alias.fill_(31)
        assert bool(torch.all(alias == 31).item())
    finally:
        region.cleanup()


@pytest.mark.gpu_1
@pytest.mark.timeout(120)
def test_cuda_ipc_owner_and_importer_share_payload_without_host_copy():
    engine_id = f"ipc-test-{uuid.uuid4()}"
    region = GPUCacheRegion(engine_id, 4, 256)
    try:
        # The child imports only IPC metadata and changes the GPU allocation.
        script = """import torch
from kvbm.vllm_integration.gpu_memory import open_region
region = open_region(__import__('sys').argv[1], (4,256))
region.fill_(37)
torch.cuda.synchronize()
del region
"""
        result = subprocess.run(
            [sys.executable, "-c", script, engine_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ),
        )
        assert result.returncode == 0, result.stderr
        assert "shared CUDA tensors released" not in result.stderr
        assert "could not unlink" not in result.stderr
        assert bool(torch.all(region.tensor == 37).item())
    finally:
        region.cleanup()


@pytest.mark.gpu_1
def test_mixed_bank_d2d_roundtrip_preserves_permuted_slots_and_guards():
    config = SimpleNamespace(
        engine_id=f"mixed-bank-{uuid.uuid4()}", parallel=SimpleNamespace(world_size=1)
    )
    spec = SimpleNamespace(
        config=config,
        num_chunks=8,
        row_bytes=384,
        blocks_per_chunk=1,
        gpu_transport="cuda_ipc",
    )
    region = GPUCacheRegion(config.engine_id, spec.num_chunks, spec.row_bytes)
    originals = [
        (torch.arange(6 * width, device="cuda", dtype=torch.int64) % 113)
        .to(torch.int8)
        .reshape(6, width)
        for width in (128, 256)
    ]
    expected = [tensor.clone() for tensor in originals]
    caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(tensor=t, page_size_bytes=t.shape[1]) for t in originals
        ],
        group_data_refs=[
            [SimpleNamespace(tensor_idx=0, page_size_bytes=128)],
            [SimpleNamespace(tensor_idx=1, page_size_bytes=256)],
        ],
    )
    worker = GPUCacheWorker(spec, caches)
    try:
        source = GPULoadStoreSpec([3, 1, 2, 0], [2, 2], [0, 0])
        slots = GPURetainedLoadStoreSpec([5, 1, 6, 4])
        assert worker.submit_store(1, source, slots)
        worker.wait({1})
        assert [result.job_id for result in worker.get_finished()] == [1]
        assert bool(torch.all(region.tensor[0] == 0).item())
        originals[0][4].zero_()
        originals[0][0].zero_()
        originals[1][5].zero_()
        originals[1][1].zero_()
        target = GPULoadStoreSpec([4, 0, 5, 1], [2, 2], [0, 0])
        assert worker.submit_load(2, slots, target)
        worker.wait({2})
        assert [result.job_id for result in worker.get_finished()] == [2]
        assert torch.equal(originals[0][4], expected[0][3])
        assert torch.equal(originals[0][0], expected[0][1])
        assert torch.equal(originals[1][5], expected[1][2])
        assert torch.equal(originals[1][1], expected[1][0])
        assert torch.equal(originals[0][2], expected[0][2])
        assert torch.equal(originals[1][3], expected[1][3])
        assert bool(torch.all(region.tensor[0] == 0).item())
        handler_refs = [weakref.ref(handler) for handler in worker.handlers]
        view_ref = weakref.ref(worker.handlers[0].dst_tensors[0])
        worker.shutdown()
        # Correct teardown must release CUDA views without a garbage-collection
        # pass; a retained bound-method cycle otherwise outlives IPC globals.
        assert all(reference() is None for reference in handler_refs)
        assert view_ref() is None
    finally:
        worker.shutdown()
        region.cleanup()


@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_local_fabric_descriptor_mismatch_fails_before_importing_any_gpu_handle(
    monkeypatch,
):
    identity = gpu_mnnvl.FabricIdentity(
        "GPU-00000000-0000-0000-0000-000000000001", "1" * 32, 0
    )
    base = "test-gpu-layout-v1"
    domain = gpu_mnnvl.effective_domain(base, identity.domain)
    monkeypatch.setattr(gpu_memory.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(gpu_mnnvl, "observe_fabric", lambda device: identity)
    monkeypatch.setattr(gpu_memory, "_process_start_ticks", lambda pid: 17)
    monkeypatch.setattr(
        gpu_mnnvl,
        "FabricAllocation",
        lambda *args, **kwargs: pytest.fail(
            "Invalid descriptor must not import a handle"
        ),
    )
    original = {
        "schema": "dynamo-gpu-fabric-v1",
        "gpu_transport": "mnnvl",
        "engine_id": "fixture",
        "owner_pid": os.getpid(),
        "owner_start_ticks": 17,
        "gpu_uuid": identity.gpu_uuid,
        "fabric_domain": identity.domain,
        "base_domain": base,
        "compatibility_domain": domain,
        "shape": [2, 8],
        "stride": [8, 1],
        "logical_bytes": 16,
        "allocation_bytes": 16,
        "fabric_handle": base64.b64encode(b"x" * 64).decode(),
    }
    for change in (
        {"schema": 1},
        {"gpu_transport": "cuda_ipc"},
        {"engine_id": "other"},
        {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000002"},
        {"fabric_domain": "mnnvl:" + "2" * 32 + ":0"},
        {"base_domain": "other-layout"},
        {"compatibility_domain": "other-domain"},
        {"shape": [True, 8]},
        {"stride": [8, True]},
        {"owner_start_ticks": 18},
        {"logical_bytes": True},
        {"allocation_bytes": True},
        {"fabric_handle": base64.b64encode(b"x" * 63).decode()},
        {"unexpected": "value"},
    ):
        with pytest.raises(ValueError):
            gpu_memory.import_fabric(
                {**original, **change}, "fixture", (2, 8), base, domain, identity.domain
            )


def _multipart_tier(tp_size=2):
    tier = GPUKVCRSecondaryTierManager.__new__(GPUKVCRSecondaryTierManager)
    tier._part_labels = tuple(f":rank{rank}" for rank in range(tp_size))
    tier._region = SimpleNamespace(shape=(4, 384 * tp_size))
    tier._key_adapter = gpu_offloading._VllmKeyAdapter()
    tier._finished_jobs = []
    tier._jobs_by_op = {}
    calls = []
    tier._kvcr = SimpleNamespace(
        config=SimpleNamespace(nixl_agent_name="multipart"),
        deliver=lambda blocks, **kwargs: calls.append((blocks, kwargs)) or 91,
    )
    return tier, calls


@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_gpu_key_adapter_preserves_group_keys_separately_from_router_hashes():
    adapter = gpu_offloading._VllmKeyAdapter()
    block_hash = BlockHash(bytes(range(32)))
    keys = [make_offload_key(block_hash, group) for group in (0, 7)]
    canonical = [adapter.decode_from_framework(key) for key in keys]
    assert canonical == [BlockKey(key) for key in keys]
    assert len(set(canonical)) == 2
    assert [adapter.encode_to_framework(key) for key in canonical] == keys
    router_key = maybe_convert_block_hash(block_hash)
    assert router_key not in canonical
    assert [adapter.encode_to_router(key) for key in canonical] == [router_key] * 2
    with pytest.raises(NotImplementedError, match="group identity"):
        adapter.decode_from_router(router_key)


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parametrize("tp_size", [2, 3, 4, 8])
def test_tp_delivers_every_rank_part_in_one_logical_operation(tp_size):
    tier, calls = _multipart_tier(tp_size)
    keys = [make_offload_key(value, 0) for value in (b"first", b"second")]
    tier.submit_load(
        SimpleNamespace(
            job_id=7, keys=keys, chunk_ids=[3, 1], req_context=ReqContext("load")
        )
    )
    assert len(calls) == 1
    blocks, kwargs = calls[0]
    assert kwargs == {"request_id": "load"}
    assert len(blocks) == 2
    for key, slot in zip(keys, [3, 1], strict=True):
        assert [(ref.label, ref.element_index) for ref in blocks[BlockKey(key)]] == [
            (f":rank{rank}", slot) for rank in range(tp_size)
        ]
    assert tier._jobs_by_op == {91: (7, 2, True, set())}
    assert tier._finished_jobs == []


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parametrize("slot", [-1, 4, True])
def test_tp2_descriptor_bounds_fail_before_native_transfer(slot):
    tier, calls = _multipart_tier()
    with pytest.raises(ValueError, match="outside"):
        tier._make_descriptors(slot)
    assert calls == []
