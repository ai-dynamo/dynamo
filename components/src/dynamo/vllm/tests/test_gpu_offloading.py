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
import threading
import uuid
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from kvbm.vllm_integration import gpu_memory, gpu_mnnvl, gpu_offloading
from kvbm.vllm_integration.gpu_memory import GPUCacheRegion, GPUCacheWorker, import_ipc
from kvbm.vllm_integration.gpu_offloading import (
    GPUKVCRSecondaryTierManager,
    GPUOffloadingConnector,
    GPUOffloadingSpec,
    GPUPrimaryManager,
    GPURetainedLoadStoreSpec,
    GPUTieringManager,
    _GPUFrameworkPinAdapter,
)
from kvcr.types import BlockKey
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    TransferJobStatus,
)
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.distributed.kv_events import BlockRemoved
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    _OffloadEventMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    Medium,
    ReqContext,
    make_offload_key,
)

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
]


def make_config(**changes):
    extra = {
        "gpu_bytes_to_use": 8 * 384,
        "offload_prompt_only": False,
        "self_describing_kv_events": True,
        "router_hint_transfer_domain": "test-gpu-layout-v1",
        "secondary_tiers": [{"type": "GPUKVCRSecondaryTierManager"}],
    }
    extra.update(changes)
    return SimpleNamespace(
        extra_config=extra,
        engine_id=f"gpu-cache-test-{uuid.uuid4()}",
        groups=(SimpleNamespace(tokens_per_block=64),),
        cache=SimpleNamespace(tokens_per_hash=64, blocks_per_chunk=1),
        parallel=SimpleNamespace(
            rank=0, world_size=1, tp_size=1, pp_size=1, pcp_size=1, dcp_size=1
        ),
        canonical_layout=False,
        replicated_layout=False,
        worker_kv_bytes_per_block=384,
        enable_kv_cache_events=True,
    )


@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_bytes_to_use": 0},
        {"gpu_bytes_to_use": True},
        {"gpu_bytes_to_use": 383},
        {"cpu_bytes_to_use": 4096},
        {"canonical_layout": True},
        {"router_hint_transfer_domain": ""},
        {"secondary_tiers": [{"type": "kvcr"}]},
        {"secondary_tiers": []},
        {"self_describing_kv_events": False},
    ],
)
def test_gpu_spec_rejects_cpu_or_uncertified_configuration(changes):
    with pytest.raises(ValueError):
        GPUOffloadingSpec(make_config(**changes))


@pytest.mark.gpu_0
def test_gpu_spec_rejects_uncertified_parallelism():
    config = make_config()
    config.parallel.world_size = 2
    with pytest.raises(ValueError, match="TP1"):
        GPUOffloadingSpec(config)


@pytest.mark.gpu_0
def test_gpu_inventory_has_its_own_medium(monkeypatch):
    event = BlockRemoved(block_hashes=[123], medium="CPU")
    monkeypatch.setattr(OffloadingConnector, "take_events", lambda self: iter([event]))
    connector = GPUOffloadingConnector.__new__(GPUOffloadingConnector)
    result = list(connector.take_events())
    assert result[0].medium == "gpu_direct"
    assert event.medium == "CPU"


@pytest.mark.gpu_0
def test_gpu_reset_does_not_release_native_owned_slots():
    manager = GPUTieringManager.__new__(GPUTieringManager)
    manager._gpu_lock = threading.RLock()
    manager._jobs = {}
    manager.primary_tier = SimpleNamespace(
        _num_allocated_chunks=2,
        _free_list=[],
        _num_evictable_cache_chunks=1,
    )
    manager.secondary_tiers = [SimpleNamespace(_hazard_ops=set())]
    with pytest.raises(RuntimeError, match="own its slots"):
        manager.reset_cache()


@pytest.mark.gpu_0
def test_failed_native_shutdown_preserves_primary_gpu_owner():
    manager = GPUTieringManager.__new__(GPUTieringManager)
    manager._gpu_lock = threading.RLock()
    manager._gpu_stop = threading.Event()
    manager._gpu_thread = SimpleNamespace(
        join=lambda **kwargs: None, is_alive=lambda: False
    )
    released = []
    manager.primary_tier = SimpleNamespace(shutdown=lambda: released.append(True))

    def failed_shutdown():
        raise RuntimeError("native transfer has not quiesced")

    manager.secondary_tiers = [
        SimpleNamespace(tier_type="gpu_kvcr", shutdown=failed_shutdown)
    ]
    with pytest.raises(RuntimeError, match="not quiesced"):
        manager.shutdown()
    assert released == []


@pytest.mark.gpu_0
def test_gpu_reset_revokes_published_keys_and_drops_unpublished_stores():
    spec = SimpleNamespace(
        num_chunks=4,
        extra_config={},
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True),
    )
    manager = GPUPrimaryManager(spec, SimpleNamespace())
    published = make_offload_key(b"published", 0)
    pending = make_offload_key(b"pending", 0)
    request = ReqContext("gpu-reset")
    manager.prepare_store([published], request)
    manager.complete_store([published], request)
    stored = list(manager.take_events())
    assert len(stored) == 1 and stored[0].keys == [published]
    assert not stored[0].removed
    manager.prepare_store([pending], request)
    manager.complete_store([pending], request)
    # Both are physically ready, but only one was published to the router.
    assert manager.lookup(pending, request) == LookupResult.HIT
    manager.reset_cache()
    manager.reset_cache()  # Must not drop the first queued revocation.
    events = list(manager.take_events())
    assert len(events) == 1
    assert events[0].removed and events[0].keys == [published]
    assert manager.lookup(published, ReqContext("after-reset")) == LookupResult.MISS
    assert manager.lookup(pending, ReqContext("after-reset")) == LookupResult.MISS
    manager.reset_cache()
    assert list(manager.take_events()) == []


@pytest.mark.gpu_0
def test_gpu_connector_reset_revokes_every_hash_in_a_published_chunk():
    events_config = SimpleNamespace(
        enable_kv_cache_events=True, self_describing_kv_events=True
    )
    spec = SimpleNamespace(
        num_chunks=4, extra_config={}, kv_events_config=events_config
    )
    manager = GPUPrimaryManager(spec, SimpleNamespace())
    tracker = OffloadingEventsTracker(events_config)
    hashes = tuple(value.to_bytes(8, "big") for value in (11, 12))
    key = make_offload_key(hashes[-1], 0)
    pending_key = make_offload_key(b"unpublished", 0)
    metadata = _OffloadEventMetadata(
        block_hashes=hashes,
        parent_block_hash=None,
        token_ids=tuple(range(32)),
        block_size=16,
        lora_id=None,
        lora_name=None,
        extra_keys=None,
        group_idx=0,
        kv_cache_spec=OffloadingEventGroupSpec("full_attention", None),
        active_residencies={(Medium.CPU, None)},
    )
    tracker._pending_event_metadata.update({key: metadata, pending_key: metadata})
    request = ReqContext("chunk-reset")
    manager.prepare_store([key], request)
    manager.complete_store([key], request)

    def reset_scheduler():
        manager.reset_cache()
        tracker.reset()

    connector = GPUOffloadingConnector.__new__(GPUOffloadingConnector)
    connector.connector_scheduler = SimpleNamespace(
        manager=SimpleNamespace(primary_tier=manager),
        _events_tracker=tracker,
        reset_cache=reset_scheduler,
        take_events=lambda: tracker.take_events(manager.take_events()),
    )
    stored = list(connector.take_events())
    assert len(stored) == 1 and len(stored[0].block_hashes) == 2
    assert connector.reset_cache() is True
    assert pending_key not in tracker._pending_event_metadata
    removed = list(connector.take_events())
    assert len(removed) == 1 and isinstance(removed[0], BlockRemoved)
    assert removed[0].medium == "gpu_direct"
    assert removed[0].block_hashes == stored[0].block_hashes
    assert tracker._pending_event_metadata == {}


@pytest.mark.gpu_0
def test_ipc_rejects_different_geometry_before_touching_cuda():
    with pytest.raises(ValueError, match="layout"):
        import_ipc({"schema": 1, "shape": [9, 384]}, (8, 384))


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
    config = make_config()
    spec = GPUOffloadingSpec(config)
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


def _mnnvl_tier(**changes):
    return {
        "type": "GPUKVCRSecondaryTierManager",
        "gpu_transport": "mnnvl",
        "ucx_tls": ["cuda_ipc", "tcp", "self"],
        "compatibility_domain": "test-gpu-layout-v1",
        **changes,
    }


@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_transport": "invalid"},
        {"gpu_transport": "mnnvl"},
        {
            "gpu_transport": "mnnvl",
            "secondary_tiers": [_mnnvl_tier(gpu_transport="cuda_ipc")],
        },
        {
            "gpu_transport": "mnnvl",
            "secondary_tiers": [_mnnvl_tier(compatibility_domain="other-layout")],
        },
        {
            "gpu_transport": "mnnvl",
            "router_hint_transfer_domain": "bad\x00layout",
            "secondary_tiers": [_mnnvl_tier(compatibility_domain="bad\x00layout")],
        },
        {"gpu_transport": "mnnvl", "secondary_tiers": [_mnnvl_tier(ucx_tls=None)]},
        {"gpu_transport": "mnnvl", "secondary_tiers": [_mnnvl_tier(ucx_tls=True)]},
        {
            "gpu_transport": "mnnvl",
            "secondary_tiers": [_mnnvl_tier(ucx_tls=["tcp", "cuda_copy", "self"])],
        },
    ],
)
def test_mnnvl_spec_requires_matching_explicit_gpu_only_transport(changes):
    with pytest.raises(ValueError):
        GPUOffloadingSpec(make_config(**changes))


@pytest.mark.gpu_0
def test_mnnvl_spec_resolves_actual_fabric_without_rewriting_original_model_config(
    monkeypatch,
):
    config = make_config(gpu_transport="mnnvl", secondary_tiers=[_mnnvl_tier()])
    spec = GPUOffloadingSpec(config)
    observed = SimpleNamespace(domain="mnnvl:" + "1" * 32 + ":0")
    helper = SimpleNamespace(
        observe_fabric=lambda: observed,
        effective_domain=lambda base, fabric: base + "@" + fabric,
    )
    monkeypatch.setattr(
        gpu_offloading, "importlib", SimpleNamespace(import_module=lambda name: helper)
    )
    assert not spec._transport_resolved
    spec._resolve_gpu_transport()
    expected = "test-gpu-layout-v1@" + observed.domain
    assert spec.fabric_domain == observed.domain
    assert spec.extra_config["router_hint_transfer_domain"] == expected
    assert spec.tiers[0]["compatibility_domain"] == expected
    assert config.extra_config["router_hint_transfer_domain"] == "test-gpu-layout-v1"
    assert (
        config.extra_config["secondary_tiers"][0]["compatibility_domain"]
        == "test-gpu-layout-v1"
    )
    assert spec.base_domain == "test-gpu-layout-v1"


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


def make_tp2_config(**changes):
    defaults = {
        "gpu_rank_devices": [0, 1],
        "secondary_tiers": [
            {
                "type": "GPUKVCRSecondaryTierManager",
                "compatibility_domain": "test-gpu-layout-v1",
            }
        ],
    }
    defaults.update(changes)
    config = make_config(**defaults)
    config.parallel.world_size = config.parallel.tp_size = 2
    return config


@pytest.mark.gpu_0
def test_tp2_budget_counts_all_rank_shards_and_isolates_raw_layout_domain():
    config = make_tp2_config()
    spec = GPUOffloadingSpec(config)
    assert spec.row_bytes == 384
    assert spec.logical_row_bytes == 768
    assert spec.num_chunks == 4
    spec._resolve_gpu_transport()
    assert spec.base_domain.endswith("|gpu-raw-tp-v1:tp=2")
    assert spec.extra_config["router_hint_transfer_domain"] == spec.base_domain
    assert spec.tiers[0]["compatibility_domain"] == spec.base_domain
    assert config.extra_config["router_hint_transfer_domain"] == "test-gpu-layout-v1"


@pytest.mark.gpu_0
@pytest.mark.parametrize("devices", [None, [], [0], [0, 0], [0, True], [-1, 0], (0, 1)])
def test_tp2_requires_an_explicit_unique_rank_device_mapping(devices):
    with pytest.raises(ValueError, match="gpu_rank_devices"):
        GPUOffloadingSpec(make_tp2_config(gpu_rank_devices=devices))


@pytest.mark.gpu_0
@pytest.mark.parametrize("field", ["pp_size", "pcp_size", "dcp_size"])
def test_tp2_rejects_other_parallel_dimensions(field):
    config = make_tp2_config()
    setattr(config.parallel, field, 2)
    with pytest.raises(ValueError, match="without PP/CP"):
        GPUOffloadingSpec(config)


@pytest.mark.gpu_0
def test_tp2_rejects_a_budget_that_only_fits_one_rank():
    with pytest.raises(ValueError, match="one full KV row"):
        GPUOffloadingSpec(make_tp2_config(gpu_bytes_to_use=767))


def _multipart_tier():
    tier = GPUKVCRSecondaryTierManager.__new__(GPUKVCRSecondaryTierManager)
    tier._part_labels = (":rank0", ":rank1")
    tier._region = SimpleNamespace(shape=(4, 768))
    tier._key_adapter = gpu_offloading._VllmKeyAdapter()
    tier._finished_jobs = []
    tier._jobs_by_op = {}
    calls = []
    tier._kvcr = SimpleNamespace(
        config=SimpleNamespace(nixl_agent_name="multipart"),
        deliver=lambda blocks, **kwargs: calls.append((blocks, kwargs)) or 91,
    )
    return tier, calls


@pytest.mark.gpu_0
def test_tp2_delivers_both_rank_parts_in_one_logical_operation():
    tier, calls = _multipart_tier()
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
            (":rank0", slot),
            (":rank1", slot),
        ]
    assert tier._jobs_by_op == {91: (7, 2, True, set())}
    assert tier._finished_jobs == []


@pytest.mark.gpu_0
@pytest.mark.parametrize("slot", [-1, 4, True])
def test_tp2_descriptor_bounds_fail_before_native_transfer(slot):
    tier, calls = _multipart_tier()
    with pytest.raises(ValueError, match="outside"):
        tier._make_descriptors(slot)
    assert calls == []


@pytest.mark.gpu_0
def test_tp2_source_pin_holds_one_primary_job_for_all_rank_parts():
    tier, _ = _multipart_tier()
    adapter = _GPUFrameworkPinAdapter(tier)
    key, missing = BlockKey(b"present"), BlockKey(b"missing")
    jobs, finished = [], []
    parent = SimpleNamespace(
        on_new_request=lambda ctx: None,
        lookup=lambda candidate, ctx: LookupResult.HIT
        if candidate == key
        else LookupResult.MISS,
        create_store_job=lambda keys, ctx: jobs.append(tuple(keys))
        or SimpleNamespace(job_id=17, keys=keys, chunk_ids=[3]),
        on_request_finished=lambda ctx: finished.append(ctx),
    )
    pin, descriptors = adapter._resolve_pin(parent, [key, missing])
    assert jobs == [(key,)] and len(finished) == 1
    assert descriptors[missing] is None
    assert [(ref.label, ref.element_index) for ref in descriptors[key]] == [
        (":rank0", 3),
        (":rank1", 3),
    ]
    assert adapter.has_active_pins()
    assert adapter.take_pin_job_results() == []
    adapter.release_pin(pin)
    adapter.release_pin(pin)
    results = adapter.take_pin_job_results()
    assert len(results) == 1 and results[0].job_id == 17
    assert not adapter.has_active_pins()


@pytest.mark.gpu_0
def test_tp2_pin_expansion_failure_releases_the_primary_job():
    tier, _ = _multipart_tier()
    adapter = _GPUFrameworkPinAdapter(tier)
    key = BlockKey(b"present")
    parent = SimpleNamespace(
        on_new_request=lambda ctx: None,
        lookup=lambda candidate, ctx: LookupResult.HIT,
        create_store_job=lambda keys, ctx: SimpleNamespace(
            job_id=17, keys=keys, chunk_ids=[4]
        ),
        on_request_finished=lambda ctx: None,
    )
    with pytest.raises(ValueError, match="outside"):
        adapter._resolve_pin(parent, [key])
    assert not adapter.has_active_pins()
    assert [result.job_id for result in adapter.take_pin_job_results()] == [17]


@pytest.mark.gpu_0
def test_gpu_adapter_preserves_the_typed_vllm_hint_envelope():
    tier, _ = _multipart_tier()
    submitted = []
    tier._kvcr.submit_hint = lambda **kwargs: submitted.append(kwargs)
    hint = KvHintsEnvelope(
        "0.1", "message", [KvHintAction("action", "kv.fetch_gpu", "1.0", {})]
    )
    tier.on_new_request(ReqContext("request", kv_hints=hint))
    assert submitted == [{"request_id": "request", "hints": hint}]
    assert submitted[0]["hints"] is hint


@pytest.mark.gpu_0
def test_tp2_retained_inventory_waits_for_both_worker_completions():
    primary = GPUPrimaryManager(
        SimpleNamespace(
            num_chunks=4,
            extra_config={},
            kv_events_config=SimpleNamespace(enable_kv_cache_events=True),
        ),
        SimpleNamespace(),
    )
    context = ReqContext("store")
    key = make_offload_key(b"both-ranks", 0)
    primary.prepare_store([key], context)
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    scheduler.manager = primary
    scheduler._jobs = {
        7: TransferJobStatus(req_id="store", pending_count=2, keys={key}, is_store=True)
    }
    scheduler._req_status = {
        "store": SimpleNamespace(
            req_context=context, transfer_jobs={7}, finished_signaled=False
        )
    }
    scheduler._stale_job_threshold = 0
    scheduler._block_id_to_pending_jobs = {}
    scheduler._chunks_being_loaded = set()
    scheduler._manager_locked = nullcontext
    output = SimpleNamespace(
        kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs={7: 1})
    )
    scheduler.update_connector_output(output)
    assert primary.lookup(key, context) is LookupResult.HIT_PENDING
    assert list(primary.take_events()) == []
    assert scheduler._jobs[7].pending_count == 1
    scheduler.update_connector_output(output)
    assert primary.lookup(key, context) is LookupResult.HIT
    assert [event.keys for event in primary.take_events()] == [[key]]
    assert scheduler._jobs == {}


@pytest.mark.gpu_0
@pytest.mark.parametrize("transport", ["cuda_ipc", "mnnvl"])
def test_tp2_discovery_and_storage_publish_the_same_domain(monkeypatch, transport):
    import json
    from unittest.mock import MagicMock

    from dynamo.llm import WorkerType
    from dynamo.vllm.kv_hints import publish_kv_hint_capabilities

    tier = {
        "type": "GPUKVCRSecondaryTierManager",
        "gpu_transport": transport,
        "compatibility_domain": "test-gpu-layout-v1",
        "router_capabilities": ["router_hint_gpu_direct"],
        "control_advertise_host": "worker-a",
        "control_ports": [24123],
    }
    if transport == "mnnvl":
        tier["ucx_tls"] = ["cuda_ipc", "tcp", "self"]
        observed = SimpleNamespace(domain="mnnvl:" + "1" * 32 + ":0")
        monkeypatch.setattr(gpu_mnnvl, "observe_fabric", lambda: observed)
    config = make_tp2_config(gpu_transport=transport, secondary_tiers=[tier])
    spec = GPUOffloadingSpec(config)
    spec._resolve_gpu_transport()
    runtime = MagicMock()
    publish_kv_hint_capabilities(
        runtime,
        SimpleNamespace(
            tensor_parallel_size=2,
            kv_transfer_config=SimpleNamespace(
                kv_connector="GPUOffloadingConnector",
                kv_connector_extra_config=config.extra_config,
            ),
        ),
        WorkerType.Decode,
    )
    fields = {
        call.args[0]: json.loads(call.args[1])
        for call in runtime.set_engine_specific.call_args_list
    }
    assert (
        fields["router_hint_transfer_domain"]
        == spec.extra_config["router_hint_transfer_domain"]
    )
    assert fields["router_hint_transfer_domain"] != "test-gpu-layout-v1"
    assert (
        spec.tiers[0]["compatibility_domain"] == fields["router_hint_transfer_domain"]
    )


@pytest.mark.gpu_0
def test_tp2_spec_rejects_multiple_data_parallel_replicas():
    config = make_tp2_config()
    config.parallel.data_parallel_size = 2
    with pytest.raises(ValueError, match="data parallel size one"):
        GPUOffloadingSpec(config)
