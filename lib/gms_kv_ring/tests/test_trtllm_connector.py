# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock-driven tests for the TRT-LLM connector.

Validates parity with the vLLM + SGLang connectors for:

  - Block-layout derivation from a TRT-LLM-shaped KV cache tensor
    `(num_blocks, num_layers, kv_factor, block_size)`.
  - PrefixIndex lookup → restore-queue building via
    `get_num_new_matched_tokens` → `update_state_after_alloc`.
  - `request_finished` records the prefix, queues evicts with
    correct generations (Race #3) and content hashes (P4d).
  - Metadata wire round-trip (v6 schema, hashes/staging optional).
  - Worker `wait_for_save` drains the evict queue through the
    BlockGdsConnector + issues batched P4d registration when
    enabled. Self-disables on skipped/error.
  - Worker `start_load_kv` drains the restore queue through
    BlockGdsConnector.restore_blocks_remap.

Runs WITHOUT tensorrt_llm installed (the connector falls back to
stub bases when the import fails). Validates all scheduler/worker
logic with mock LlmArgs + mock kv_cache_tensor + mock connector."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gms_kv_ring.common.prefix_hashes import prefix_block_hashes
from gpu_memory_service.integrations.trtllm.gms_connector import (
    GMSTrtllmConnectorMeta,
    GMSTrtllmKvCacheScheduler,
    GMSTrtllmKvCacheWorker,
    _build_block_layout_trtllm,
    _decode_meta,
    _encode_meta,
)

# ----------------------------------------------------------------------
# Mocks
# ----------------------------------------------------------------------


class _FakeTensor:
    """Stand-in for torch.Tensor exposing only the attributes
    `_build_block_layout_trtllm` reads."""

    def __init__(self, shape, base_ptr=0x10000000, elem_size=2):
        self.shape = tuple(int(x) for x in shape)
        self._base_ptr = int(base_ptr)
        self._elem_size = int(elem_size)

    def data_ptr(self):
        return self._base_ptr

    def element_size(self):
        return self._elem_size


def _make_llm_args(block_size=4):
    """Mock TRT-LLM llm_args exposing kv_cache_config.tokens_per_block."""
    return SimpleNamespace(
        kv_cache_config=SimpleNamespace(tokens_per_block=block_size),
    )


class _FakeRequest:
    """LlmRequest stand-in matching the connector's reads."""

    def __init__(self, req_id, tokens, cache_salt_id=None, context_pos=0):
        self.request_id = req_id
        self._tokens = list(tokens)
        self.prompt_token_ids = list(tokens)
        self.cache_salt_id = cache_salt_id
        self.context_current_position = int(context_pos)

    def get_tokens(self, _beam):
        return self._tokens


# ----------------------------------------------------------------------
# Block layout
# ----------------------------------------------------------------------


def test_block_layout_default_shape():
    """`(num_blocks, num_layers, kv_factor, block_size)` produces
    `num_layers × kv_factor` layer entries with the right strides."""
    # 8 blocks × 4 layers × 2 (K/V) × (16 tokens × 8 heads × 64) = ...
    # block_size_axis: 16*8*64 = 8192 elements per block per layer per K/V
    tensor = _FakeTensor(shape=(8, 4, 2, 16 * 8 * 64), elem_size=2)
    layers, layout, n_layers_total, block_bytes_per_block = _build_block_layout_trtllm(
        tensor
    )
    # 4 layers × 2 kv_factor = 8 "layer" indices in the daemon's view.
    assert n_layers_total == 8
    assert len(layers) == 8
    # Per-block bytes = 4 × 2 × 8192 × 2 = 131072 bytes
    expected_kv_size = 8192 * 2
    expected_per_block = expected_kv_size * 2 * 4
    assert block_bytes_per_block == expected_per_block
    # Layout for block 0: 8 entries, each `(layer, offset, size)`.
    ranges_0 = layout(0)
    assert len(ranges_0) == 8
    # First entry is (layer=0, offset=0, size=expected_kv_size)
    assert ranges_0[0] == (0, 0, expected_kv_size)
    # Last entry: (layer=7, offset=(3*2+1)*kv_size, size=kv_size)
    assert ranges_0[7] == (7, (3 * 2 + 1) * expected_kv_size, expected_kv_size)
    # Block 1: offset shifted by block_bytes_per_block
    ranges_1 = layout(1)
    assert ranges_1[0] == (0, expected_per_block, expected_kv_size)


def test_block_layout_rejects_low_dim_tensor():
    with pytest.raises(ValueError, match="at least 4D"):
        _build_block_layout_trtllm(_FakeTensor(shape=(8, 4)))


def test_block_layout_rejects_non_tensor():
    with pytest.raises(TypeError, match="torch.Tensor"):
        _build_block_layout_trtllm("not a tensor")


# ----------------------------------------------------------------------
# Wire format (encode/decode round-trip + v6 hashes/staging)
# ----------------------------------------------------------------------


def test_meta_round_trip_no_hashes():
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[("r1", [10, 11, 12], [1, 2, 3], [])],
        restore_queue=[("r2", [(20, 30, 5), (21, 31, 7)])],
    )
    blob = _encode_meta(meta)
    decoded = _decode_meta(blob)
    assert decoded.evict_queue == [("r1", [10, 11, 12], [1, 2, 3], [])]
    assert decoded.restore_queue == [("r2", [(20, 30, 5), (21, 31, 7)])]


def test_meta_round_trip_with_hashes():
    import hashlib

    h1 = hashlib.sha256(b"block-0").digest()
    h2 = hashlib.sha256(b"block-1").digest()
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[("r1", [10, 11], [4, 5], [h1, h2])],
        restore_queue=[],
    )
    blob = _encode_meta(meta)
    # When hashes are present, the JSON includes hashes_per_evict.
    import json as _json

    obj = _json.loads(blob.decode("utf-8"))
    assert "hashes_per_evict" in obj
    decoded = _decode_meta(blob)
    assert decoded.evict_queue[0][3] == [h1, h2]


def test_meta_round_trip_with_staging_restore():
    import hashlib

    h = hashlib.sha256(b"staged").digest()
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[],
        restore_queue=[],
        staging_restore_queue=[("r-stage", [(h, 42, 7)])],
    )
    blob = _encode_meta(meta)
    import json as _json

    obj = _json.loads(blob.decode("utf-8"))
    assert obj["v"] == 6
    assert obj["restore_staging"] == [["r-stage", [[h.hex(), 42, 7]]]]
    decoded = _decode_meta(blob)
    assert decoded.staging_restore_queue == [("r-stage", [(h, 42, 7)])]


# ----------------------------------------------------------------------
# Scheduler: get_num_new_matched_tokens + update_state_after_alloc
# ----------------------------------------------------------------------


def _make_scheduler(monkeypatch, **env):
    """Build a scheduler with epoch thread disabled (no daemon socket)
    and any extra env overrides applied."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    sched = GMSTrtllmKvCacheScheduler(_make_llm_args(block_size=4))
    return sched


def test_scheduler_returns_zero_on_cold_index(monkeypatch):
    sched = _make_scheduler(monkeypatch)
    req = _FakeRequest("r1", [1, 2, 3, 4, 5, 6, 7, 8])
    extra, async_ = sched.get_num_new_matched_tokens(req, 0)
    assert extra == 0
    assert async_ is False


def test_scheduler_uses_staging_hits_when_no_local_match(monkeypatch):
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    sched = _make_scheduler(monkeypatch)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    hashes = prefix_block_hashes(tokens, 4, None)
    sched._staging_scan = lambda _hashes: {
        hashes[0]: {"generation": 3},
        hashes[1]: {"generation": 4},
    }
    req = _FakeRequest("r-stage", tokens)
    extra, async_ = sched.get_num_new_matched_tokens(req, 0)
    assert (extra, async_) == (8, False)
    sched.update_state_after_alloc(req, [200, 201])
    meta = sched.build_connector_meta(SimpleNamespace())
    assert meta.restore_queue == []
    assert meta.staging_restore_queue == [
        ("r-stage", [(hashes[0], 200, 3), (hashes[1], 201, 4)]),
    ]


def test_scheduler_record_then_lookup_hits(monkeypatch):
    sched = _make_scheduler(monkeypatch)
    # Record a prefix.
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    req1 = _FakeRequest("r1", tokens)
    sched.request_finished(req1, [100, 101])
    # The prefix is now in the PrefixIndex; a fresh request with the
    # same tokens should report the cache hit.
    req2 = _FakeRequest("r2", tokens)
    extra, _ = sched.get_num_new_matched_tokens(req2, 0)
    assert extra == 8  # 2 blocks × 4 tokens
    # update_state_after_alloc binds dst block ids and emits restore
    # queue entries.
    sched.update_state_after_alloc(req2, [200, 201])
    meta = sched.build_connector_meta(SimpleNamespace())
    assert len(meta.restore_queue) == 1
    rid, triples = meta.restore_queue[0]
    assert rid == "r2"
    assert len(triples) == 2
    # Each triple is (src, dst, gen). src ∈ {100, 101}, dst ∈ {200, 201}.
    srcs = {t[0] for t in triples}
    dsts = {t[1] for t in triples}
    assert srcs == {100, 101}
    assert dsts == {200, 201}


def test_scheduler_race3_generations_threaded_through(monkeypatch):
    sched = _make_scheduler(monkeypatch)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    # First spill: assigns gen=1 per block.
    req1 = _FakeRequest("r1", tokens)
    sched.request_finished(req1, [100, 101])
    meta1 = sched.build_connector_meta(SimpleNamespace())
    assert meta1.evict_queue[0][2] == [1, 1]
    # Re-record the SAME prefix to slots {100, 101} again — gens bump to 2.
    req2 = _FakeRequest("r2", tokens)
    sched.request_finished(req2, [100, 101])
    meta2 = sched.build_connector_meta(SimpleNamespace())
    assert meta2.evict_queue[0][2] == [2, 2]
    # And a future cache-hit lookup carries the bumped generations
    # forward.
    req3 = _FakeRequest("r3", tokens)
    sched.get_num_new_matched_tokens(req3, 0)
    sched.update_state_after_alloc(req3, [200, 201])
    meta3 = sched.build_connector_meta(SimpleNamespace())
    triples = meta3.restore_queue[0][1]
    # expected_gen on each triple is the LAST recorded gen (=2).
    for src, dst, gen in triples:
        assert gen == 2


def test_scheduler_cross_node_p4d_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GMS_KVR_CROSS_NODE", raising=False)
    sched = _make_scheduler(monkeypatch)
    req = _FakeRequest("r1", [1, 2, 3, 4, 5, 6, 7, 8])
    sched.request_finished(req, [100, 101])
    meta = sched.build_connector_meta(SimpleNamespace())
    # hashes list is empty — no cross-node registration.
    assert meta.evict_queue[0][3] == []


def test_scheduler_cross_node_p4d_enabled_attaches_hashes(monkeypatch):
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    sched = _make_scheduler(monkeypatch)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    req = _FakeRequest("r1", tokens)
    sched.request_finished(req, [100, 101])
    meta = sched.build_connector_meta(SimpleNamespace())
    hashes = meta.evict_queue[0][3]
    # Two full blocks at block_size=4 → 2 hashes.
    expected = prefix_block_hashes(tokens, 4, None)
    assert hashes[:2] == expected
    # The hashes match what vLLM/SGLang would produce for the same
    # prefix — cross-engine compatibility.


def test_scheduler_request_finished_no_tokens_no_record(monkeypatch):
    sched = _make_scheduler(monkeypatch)

    class _NoTokensReq:
        request_id = "x"

        def get_tokens(self, *_):
            raise AttributeError

    req = _NoTokensReq()
    # Should NOT crash even though token retrieval fails.
    assert sched.request_finished(req, [1, 2]) is False


# ----------------------------------------------------------------------
# Worker: register_kv_caches + wait_for_save + start_load_kv
# ----------------------------------------------------------------------


class _FakeBlockConnector:
    def __init__(self, available=True):
        self._available = available
        self.evict_calls: list = []
        self.host_evict_calls: list = []
        self.restore_calls: list = []
        self.host_restore_calls: list = []

    def is_available(self):
        return self._available

    def evict_blocks_to_storage(self, block_ids, generations=None):
        from gms_kv_ring.engines.gds_block_connector import EvictResult

        bids = list(block_ids)
        self.evict_calls.append((bids, dict(generations or {})))
        return EvictResult(succeeded=len(bids), failed=0, failed_ids=[])

    def evict_blocks_to_host(self, block_ids, generations=None):
        from gms_kv_ring.engines.gds_block_connector import EvictResult

        bids = list(block_ids)
        self.host_evict_calls.append(bids)
        return EvictResult(succeeded=len(bids), failed=0, failed_ids=[])

    def restore_blocks_remap(self, triples):
        triples_l = list(triples)
        self.restore_calls.append(triples_l)
        out = {}
        for t in triples_l:
            if len(t) == 3:
                _src, dst, _gen = t
            else:
                _src, dst = t
            out[dst] = True
        return out

    def restore_blocks_remap_from_host(self, src_engine_id, triples):
        triples_l = list(triples)
        self.host_restore_calls.append((str(src_engine_id), triples_l))
        out = {}
        for t in triples_l:
            if len(t) == 3:
                _src, dst, _gen = t
            else:
                _src, dst = t
            out[dst] = True
        return out


class _FakeDaemonClient:
    def __init__(self, skipped=False, raise_on_call=False):
        self._skipped = skipped
        self._raise = raise_on_call
        self.batch_calls: list = []

    def register_content_addresses_batch(self, items):
        if self._raise:
            raise RuntimeError("daemon error")
        self.batch_calls.append([dict(it) for it in items])
        total = sum(sum(r[2] for r in it["ranges"]) for it in items)
        return (total, self._skipped)


class _FakeHandle:
    def __init__(self, client):
        self._client = client
        self.staging_restore_calls: list = []
        self.staging_restore_ok = True

    def restore_staging_blocks_sync(self, triples):
        self.staging_restore_calls.append(list(triples))
        return self.staging_restore_ok


class _FakeStream:
    def synchronize(self):
        pass


def _install_worker_mock(
    worker, *, available=True, daemon_skipped=False, daemon_raise=False
):
    """Plug a FakeBlockConnector + FakeHandle into the worker, bypassing
    real install_for_trtllm (which would try to open a UDS socket)."""
    worker._gds_conn = _FakeBlockConnector(available=available)
    worker._handle = _FakeHandle(
        _FakeDaemonClient(
            skipped=daemon_skipped,
            raise_on_call=daemon_raise,
        )
    )
    # Minimal block layout: per-block returns one (layer, offset, size).
    worker._block_layout = lambda bid: [(0, int(bid) * 64, 64)]
    worker._n_layers = 1
    worker._block_bytes = 64
    return worker


def test_worker_wait_for_save_drains_evict_queue(monkeypatch):
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    worker = GMSTrtllmKvCacheWorker(_make_llm_args(block_size=4))
    _install_worker_mock(worker)
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[("r1", [10, 11], [3, 4], [])],
        restore_queue=[],
    )
    worker.bind_connector_meta(meta)
    worker.wait_for_save(_FakeStream())
    assert len(worker._gds_conn.evict_calls) == 1
    bids, gens_map = worker._gds_conn.evict_calls[0]
    assert sorted(bids) == [10, 11]
    assert gens_map == {10: 3, 11: 4}
    # No cross-node calls (hashes empty).
    assert worker._handle._client.batch_calls == []
    # `r1` reported as save-complete.
    saves, _loads = worker.get_finished(["r1"], [])
    assert saves == ["r1"]


def test_worker_wait_for_save_cross_node_emits_batch(monkeypatch):
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    worker = GMSTrtllmKvCacheWorker(_make_llm_args(block_size=4))
    _install_worker_mock(worker)
    import hashlib

    h1 = hashlib.sha256(b"a").digest()
    h2 = hashlib.sha256(b"b").digest()
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[("r1", [10, 11], [1, 1], [h1, h2])],
        restore_queue=[],
    )
    worker.bind_connector_meta(meta)
    worker.wait_for_save(_FakeStream())
    assert len(worker._handle._client.batch_calls) == 1
    items = worker._handle._client.batch_calls[0]
    assert len(items) == 2
    assert items[0]["content_hash"] == h1
    assert items[1]["content_hash"] == h2


def test_worker_cross_node_self_disables_on_skipped(monkeypatch):
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    worker = GMSTrtllmKvCacheWorker(_make_llm_args(block_size=4))
    _install_worker_mock(worker, daemon_skipped=True)
    assert worker._cross_node_xfer_enabled is True
    import hashlib

    h = hashlib.sha256(b"x").digest()
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[("r1", [10], [1], [h])],
        restore_queue=[],
    )
    worker.bind_connector_meta(meta)
    worker.wait_for_save(_FakeStream())
    # Was True, now False because daemon returned skipped=True.
    assert worker._cross_node_xfer_enabled is False


def test_worker_cross_node_self_disables_on_rpc_error(monkeypatch):
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    worker = GMSTrtllmKvCacheWorker(_make_llm_args(block_size=4))
    _install_worker_mock(worker, daemon_raise=True)
    assert worker._cross_node_xfer_enabled is True
    import hashlib

    h = hashlib.sha256(b"x").digest()
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[("r1", [10], [1], [h])],
        restore_queue=[],
    )
    worker.bind_connector_meta(meta)
    worker.wait_for_save(_FakeStream())
    # Disabled after the exception.
    assert worker._cross_node_xfer_enabled is False
    # And the underlying spill still succeeded.
    assert len(worker._gds_conn.evict_calls) == 1


def test_worker_start_load_kv_drains_restore_queue(monkeypatch):
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    worker = GMSTrtllmKvCacheWorker(_make_llm_args(block_size=4))
    _install_worker_mock(worker)
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[],
        restore_queue=[("r1", [(10, 20, 1), (11, 21, 2)])],
    )
    worker.bind_connector_meta(meta)
    worker.start_load_kv(_FakeStream())
    assert len(worker._gds_conn.restore_calls) == 1
    triples = worker._gds_conn.restore_calls[0]
    assert sorted(triples) == [(10, 20, 1), (11, 21, 2)]
    # get_finished reports the load complete.
    _saves, loads = worker.get_finished([], ["r1"])
    assert loads == ["r1"]


def test_worker_start_load_kv_drains_staging_restore_queue(monkeypatch):
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    worker = GMSTrtllmKvCacheWorker(_make_llm_args(block_size=4))
    _install_worker_mock(worker)
    import hashlib

    h = hashlib.sha256(b"staged-trt").digest()
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[],
        restore_queue=[],
        staging_restore_queue=[("r-stage", [(h, 30, 5)])],
    )
    worker.bind_connector_meta(meta)
    worker.start_load_kv(_FakeStream())
    assert worker._handle.staging_restore_calls == [[(h, 30, 5)]]
    _saves, loads = worker.get_finished([], ["r-stage"])
    assert loads == ["r-stage"]


def test_worker_no_op_when_gds_unavailable(monkeypatch):
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    worker = GMSTrtllmKvCacheWorker(_make_llm_args(block_size=4))
    _install_worker_mock(worker, available=False)
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[("r1", [10], [1], [])],
        restore_queue=[("r2", [(11, 21, 1)])],
    )
    worker.bind_connector_meta(meta)
    worker.start_load_kv(_FakeStream())
    worker.wait_for_save(_FakeStream())
    # No connector calls when unavailable.
    assert worker._gds_conn.evict_calls == []
    assert worker._gds_conn.restore_calls == []
    # But save-side req_id is still reported as completed (so
    # TRT-LLM can deallocate).
    saves, _ = worker.get_finished(["r1"], [])
    assert saves == ["r1"]


def test_scheduler_shadow_lookup_uses_source_engine(monkeypatch):
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    monkeypatch.setenv("GMS_TRTLLM_ENGINE_ID", "shadow")
    monkeypatch.setenv("GMS_TRTLLM_SOURCE_ENGINE_ID", "primary")
    sched = GMSTrtllmKvCacheScheduler(_make_llm_args(block_size=4))
    req = _FakeRequest("r1", [1, 2, 3, 4])
    sched._prefix_index.record(req.prompt_token_ids, [17], 4, None, "primary")

    matched, is_async = sched.get_num_new_matched_tokens(req, 0)
    sched.update_state_after_alloc(req, [99])

    assert matched == 4
    assert is_async is False
    assert sched._sched_restore_queue == [("r1", [(17, 99, 1)])]


def test_worker_host_tier_restore_and_evict_when_gds_unavailable(monkeypatch):
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    monkeypatch.setenv("GMS_TRTLLM_ENGINE_ID", "shadow")
    monkeypatch.setenv("GMS_TRTLLM_SOURCE_ENGINE_ID", "primary")
    monkeypatch.setenv("GMS_KVR_HOST_TIER_FALLBACK", "1")
    worker = GMSTrtllmKvCacheWorker(_make_llm_args(block_size=4))
    _install_worker_mock(worker, available=False)
    meta = GMSTrtllmConnectorMeta(
        evict_queue=[("save", [10], [1], [])],
        restore_queue=[("load", [(17, 99, 1)])],
    )
    worker.bind_connector_meta(meta)

    worker.start_load_kv(_FakeStream())
    worker.wait_for_save(_FakeStream())

    assert worker._gds_conn.restore_calls == []
    assert worker._gds_conn.host_restore_calls == [("primary", [(17, 99, 1)])]
    assert worker._gds_conn.evict_calls == []
    assert worker._gds_conn.host_evict_calls == [[10]]
    saves, loads = worker.get_finished(["save"], ["load"])
    assert saves == ["save"]
    assert loads == ["load"]


# ----------------------------------------------------------------------
# TRT-LLM V2 lease reclaim integration
# ----------------------------------------------------------------------


class _FakeV2Allocator:
    def __init__(self, free=0):
        self._free = int(free)

    @property
    def num_free_slots(self):
        return self._free


class _FakeV2Controller:
    def __init__(self, evictable_by_group):
        self.evictable_by_group = dict(evictable_by_group)

    def num_evictable_pages(self, pg_idx):
        return self.evictable_by_group.get(int(pg_idx), 0)


class _FakeV2Storage:
    def __init__(self, allocator, controller, num_pool_groups=2):
        self._allocator = allocator
        self._controller = controller
        self.num_pool_groups = int(num_pool_groups)
        self._levels = [SimpleNamespace(controller=controller)]
        self.force_evict_calls = []

    def force_evict(self, level, goals):
        goals = [int(g) for g in goals]
        self.force_evict_calls.append((int(level), goals))
        reclaimed = goals[1]
        self._allocator._free += reclaimed
        self._controller.evictable_by_group[1] -= reclaimed


def test_trtllm_v2_reclaim_uses_eviction_controller(monkeypatch):
    from gpu_memory_service.integrations.trtllm import install_kv_leases_v2

    old_state = dict(install_kv_leases_v2._ALLOC_STATE)
    install_kv_leases_v2._ALLOC_STATE.clear()
    try:
        allocator = _FakeV2Allocator(free=4)
        controller = _FakeV2Controller({1: 3})
        storage = _FakeV2Storage(allocator, controller)
        manager = SimpleNamespace(_storage=storage)
        install_kv_leases_v2._ALLOC_STATE[id(allocator)] = {
            "allocator": allocator,
            "manager": manager,
            "pool_group_index": 1,
        }

        reclaimed = install_kv_leases_v2.reclaim_cached_kv_v2_for_allocator(
            id(allocator), 8
        )

        assert reclaimed == 3
        assert storage.force_evict_calls == [(0, [0, 3])]
        assert allocator.num_free_slots == 7
        assert controller.num_evictable_pages(1) == 0
    finally:
        install_kv_leases_v2._ALLOC_STATE.clear()
        install_kv_leases_v2._ALLOC_STATE.update(old_state)


def test_trtllm_v2_register_manager_binds_pool_group(monkeypatch):
    from gpu_memory_service.integrations.trtllm import install_kv_leases_v2

    old_state = dict(install_kv_leases_v2._ALLOC_STATE)
    install_kv_leases_v2._ALLOC_STATE.clear()
    started = []
    monkeypatch.setattr(
        install_kv_leases_v2,
        "_start_reclaim_watcher",
        lambda allocator_id, st: started.append((allocator_id, dict(st))),
    )
    try:
        allocator = _FakeV2Allocator()
        pool_groups = [
            SimpleNamespace(_slot_allocator=_FakeV2Allocator()),
            SimpleNamespace(_slot_allocator=allocator),
        ]
        storage = SimpleNamespace(
            _levels=[SimpleNamespace(storage=SimpleNamespace(_pool_groups=pool_groups))]
        )
        manager = SimpleNamespace(_storage=storage)
        install_kv_leases_v2._ALLOC_STATE[id(allocator)] = {
            "allocator": allocator,
            "namespace_suffix": "v2-gpu-pool-1",
        }

        install_kv_leases_v2._register_manager_for_reclaim(manager)

        state = install_kv_leases_v2._ALLOC_STATE[id(allocator)]
        assert state["manager"] is manager
        assert state["pool_group_index"] == 1
        assert started and started[0][0] == id(allocator)
    finally:
        install_kv_leases_v2._ALLOC_STATE.clear()
        install_kv_leases_v2._ALLOC_STATE.update(old_state)


def test_trtllm_v2_unregister_tolerates_cleared_module_state(monkeypatch):
    from gpu_memory_service.integrations.trtllm import install_kv_leases_v2

    manager = object()
    monkeypatch.setattr(install_kv_leases_v2, "_ALLOC_STATE", None)

    install_kv_leases_v2._unregister_manager_for_reclaim(manager)


def test_trtllm_v2_reclaim_watcher_uses_pool_namespace(monkeypatch):
    from gpu_memory_service.integrations.common import transition_reclaim
    from gpu_memory_service.integrations.trtllm import install_kv_leases_v2

    class _Watcher:
        def stop(self):
            pass

    calls = []

    def _fake_start(engine, reclaim, *, namespace_suffix, **kwargs):
        calls.append((engine, namespace_suffix, reclaim(5), kwargs))
        return _Watcher()

    monkeypatch.setenv("GMS_TRTLLM_V2_TRANSITION_RECLAIM", "1")
    monkeypatch.setattr(
        transition_reclaim, "start_kv_transition_reclaim_watcher", _fake_start
    )
    monkeypatch.setattr(
        install_kv_leases_v2,
        "reclaim_cached_kv_v2_for_allocator",
        lambda allocator_id, target: int(target) - 1,
    )
    state = {"namespace_suffix": "v2-gpu-pool-0:slots16:sizes64"}

    install_kv_leases_v2._start_reclaim_watcher(123, state)

    assert calls == [("trtllm", "v2-gpu-pool-0:slots16:sizes64", 4, {})]
    assert "reclaim_watcher" in state


def test_trtllm_scheduler_reclaim_delegates_to_v2(monkeypatch):
    from gpu_memory_service.integrations.trtllm import install_kv_leases_v2

    monkeypatch.setattr(install_kv_leases_v2, "reclaim_cached_kv_v2", lambda n: 7)
    sched = _make_scheduler(monkeypatch)

    assert sched.reclaim_cached_kv(99) == 7
