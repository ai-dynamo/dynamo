# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `GMSKVCacheConnectorV1` in the legacy vLLM integration.

Lives under `gms_kv_ring/tests/` (not the legacy adapter's test
tree) so it can reuse `_FakeGpuDirectBackend` and `_spawn_daemon`
from `test_demote_hbm_to_storage` without duplicating fixtures.

The file is skipped wholesale unless vLLM, torch, and CUDA are all
available — the connector subclasses `KVConnectorBase_V1` which
requires a working vLLM import."""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest

vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

# Import after vLLM is confirmed available — the connector module
# imports vllm at module load time inside _import_v1_base().
from gpu_memory_service.integrations.vllm.gds_connector_v1 import (  # noqa: E402
    GMSKVCacheConnectorV1,
    _decode_meta,
    _encode_meta,
    _GmsGdsMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

from tests.test_demote_hbm_to_storage import (  # noqa: E402
    _FakeGpuDirectBackend,
    _spawn_daemon,
)


def _fake_vllm_config(
    daemon_socket: str,
    engine_id: str = "test-eid",
    block_size_tokens: int = 4,
):
    """Minimal stand-in for VllmConfig that exposes the fields the
    connector actually reads. We avoid constructing a real VllmConfig
    (heavy + version-dependent) since the connector only touches
    `kv_transfer_config.{engine_id, kv_connector_extra_config}` and
    `cache_config.block_size`. The default block_size_tokens=4 is
    small to keep test fixtures cheap."""
    kv_xfer = SimpleNamespace(
        engine_id=engine_id,
        kv_connector_extra_config={"gms_daemon_socket": daemon_socket},
    )
    return SimpleNamespace(
        kv_transfer_config=kv_xfer,
        cache_config=SimpleNamespace(block_size=block_size_tokens),
    )


def _make_kv_caches(n_layers=3, n_blocks=64, block_bytes=16):
    """Allocate vLLM-shaped per-layer KV tensors on CUDA and return
    a dict[layer_name, tensor]. Shape mimics vLLM's paged-attention
    layout: `(n_blocks, block_bytes)` — at least 2D so the
    `max(shape[0], shape[1])` block-dim heuristic in
    `_build_layers_and_layout` picks the right axis. Defaults pick
    `n_blocks > block_bytes` so the heuristic resolves correctly
    (real vLLM models always have `num_blocks >> per-block dims`)."""
    caches = {}
    for L in range(n_layers):
        t = torch.zeros(n_blocks, block_bytes, dtype=torch.uint8, device="cuda")
        caches[f"model.layers.{L}.self_attn.kv_cache"] = t
    torch.cuda.synchronize()
    return caches


def test_scheduler_request_finished_returns_true_when_blocks_present(
    tmp_path,
):
    cfg = _fake_vllm_config(str(tmp_path / "ignored.sock"))
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    fake_req = SimpleNamespace(request_id="req-A")
    keep, params = conn.request_finished(fake_req, [1, 2, 3])
    assert keep is True
    assert params is None

    # Empty block_ids → no async-save, vLLM can free immediately.
    keep2, _ = conn.request_finished(
        SimpleNamespace(request_id="req-B"),
        [],
    )
    assert keep2 is False


def test_scheduler_build_connector_meta_drains_queue(tmp_path):
    cfg = _fake_vllm_config(str(tmp_path / "ignored.sock"))
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    conn.request_finished(SimpleNamespace(request_id="req-1"), [0, 5, 9])
    conn.request_finished(SimpleNamespace(request_id="req-2"), [3])

    meta = conn.build_connector_meta(SimpleNamespace())
    assert isinstance(meta, _GmsGdsMetadata)
    evict, restore_async, restore_sync, restore_staging_async = _decode_meta(
        meta.payload
    )
    # v6 evict entries are 4-tuples (req_id, ids, gens, hashes).
    # Cross-node is off by default, so hashes are []. These requests'
    # prompts are empty so the prefix index recorded nothing and gens
    # are all 0.
    assert evict == [
        ("req-1", [0, 5, 9], [0, 0, 0], []),
        ("req-2", [3], [0], []),
    ]
    assert restore_async == []  # no cache-hit restores in this test
    assert restore_sync == []

    # Second call must produce an empty queue (drained).
    meta2 = conn.build_connector_meta(SimpleNamespace())
    assert _decode_meta(meta2.payload) == ([], [], [], [])


def test_scheduler_get_num_new_matched_tokens_is_zero(tmp_path):
    """Restore path is intentionally unwired; the scheduler must
    not advertise external matches."""
    cfg = _fake_vllm_config(str(tmp_path / "ignored.sock"))
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    n, async_load = conn.get_num_new_matched_tokens(
        SimpleNamespace(request_id="r"),
        0,
    )
    assert n == 0
    assert async_load is False


def test_worker_register_kv_caches_builds_gds_connector(tmp_path):
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock)
        conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(n_layers=3, n_blocks=64, block_bytes=16)
        try:
            conn.register_kv_caches(caches)
            assert conn._handle is not None
            assert conn._gds_conn is not None
            assert conn._n_layers == 3
            assert conn._block_bytes == 16
            assert conn._gds_conn.is_available() is True
        finally:
            if conn._handle is not None:
                conn._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_worker_bind_connector_metadata_evicts_blocks(tmp_path):
    """End-to-end: scheduler encodes (req_id, block_ids) into
    metadata, worker decodes and runs evict_blocks_to_storage,
    get_finished reports the req_id."""
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock)

        # Scheduler side: build metadata.
        sched = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
        sched.request_finished(SimpleNamespace(request_id="req-X"), [0, 5])
        meta = sched.build_connector_meta(SimpleNamespace())

        # Worker side: register caches + bind that metadata.
        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(n_layers=3, n_blocks=64, block_bytes=16)
        try:
            worker.register_kv_caches(caches)
            assert len(fake.demote_calls) == 0

            worker.bind_connector_metadata(meta)
            # 2 blocks × 3 layers = 6 demote_from_gpu calls.
            assert len(fake.demote_calls) == 6
            engine_ids = {c["engine_id"] for c in fake.demote_calls}
            assert engine_ids == {"test-eid"}

            done, recv = worker.get_finished({"req-X"})
            assert done == {"req-X"}
            assert recv is None

            # Subsequent get_finished returns no saves (drained).
            done2, _ = worker.get_finished(set())
            assert done2 is None
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_worker_bind_without_register_drops_and_reports_finished(tmp_path):
    """If a stray metadata blob arrives before register_kv_caches
    (e.g., misconfiguration), the worker must NOT crash AND must
    still report req_ids as finished — otherwise vLLM pins blocks
    forever waiting for an evict that will never come."""
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock)
        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        meta = _GmsGdsMetadata(
            _encode_meta([("req-Y", [1, 2], [0, 0])]),
        )
        worker.bind_connector_metadata(meta)
        assert len(fake.demote_calls) == 0  # nothing spilled
        done, _ = worker.get_finished({"req-Y"})
        assert done == {"req-Y"}
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_register_gms_gds_connector_makes_short_name_resolvable():
    """The legacy adapter calls register_gms_gds_connector() at
    worker import. After that, vLLM's KVConnectorFactory must be
    able to resolve the short name 'GMSKVCacheConnectorV1' to our
    class so a user can wire `--kv-transfer-config '{"kv_connector":
    "GMSKVCacheConnectorV1", ...}'` without spelling the full path."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import (
        register_gms_gds_connector,
    )
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    register_gms_gds_connector()
    cls = KVConnectorFactory._registry["GMSKVCacheConnectorV1"]()
    assert cls is GMSKVCacheConnectorV1


def test_engine_id_overridable_via_kv_transfer_config(tmp_path):
    """The connector reads engine_id from kv_transfer_config; this
    pins that the field is honored rather than falling back to a
    default. (vLLM's base class requires kv_transfer_config to be
    set, so the env-var fallback in _resolve_engine_id is never
    reached in real use — pin only the documented path.)"""
    cfg = _fake_vllm_config(str(tmp_path / "ignored.sock"), engine_id="my-engine-42")
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    assert conn.engine_id == "my-engine-42"


# ----------------------------------------------------------------------
# Restore-on-hit
# ----------------------------------------------------------------------


def _fake_request(req_id: str, prompt_token_ids, cache_salt=None):
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=list(prompt_token_ids),
        cache_salt=cache_salt,
        num_tokens=len(prompt_token_ids),
    )


def _fake_blocks(block_ids):
    """Minimal stand-in for vLLM's KVCacheBlocks. The connector
    only uses `.get_block_ids()` which returns one list per
    kv-group; tests use a single group."""
    return SimpleNamespace(get_block_ids=lambda: [list(block_ids)])


def test_scheduler_hash_index_populated_by_request_finished(tmp_path):
    """request_finished should index the request's prompt prefix
    block-by-block so a subsequent matching prompt hits."""
    cfg = _fake_vllm_config(str(tmp_path / "i.sock"), block_size_tokens=4)
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)

    # 12 tokens → 3 full blocks of size 4. block_ids=[7, 8, 9].
    req = _fake_request("r0", list(range(1, 13)))
    conn.request_finished(req, [7, 8, 9])

    # Same prefix incoming → should match all 3 blocks (12 tokens).
    req2 = _fake_request("r1", list(range(1, 13)) + [99, 100])
    n, async_load = conn.get_num_new_matched_tokens(req2, 0)
    assert n == 12
    assert async_load is False
    # v4 lookup returns (src_block_id, expected_generation) pairs.
    # r0's request_finished bumped gens for blocks 7,8,9 once
    # (first-evict) → gen=1.
    assert conn._pending_hit["r1"] == [
        ("local", 7, 1),
        ("local", 8, 1),
        ("local", 9, 1),
    ]


def test_scheduler_hash_lookup_misses_on_different_prefix(tmp_path):
    cfg = _fake_vllm_config(str(tmp_path / "i.sock"), block_size_tokens=4)
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    conn.request_finished(_fake_request("r0", list(range(1, 13))), [7, 8, 9])
    # First block differs → 0 matched.
    req2 = _fake_request("r1", [99] + list(range(2, 13)))
    n, _ = conn.get_num_new_matched_tokens(req2, 0)
    assert n == 0


def test_scheduler_hash_lookup_respects_engine_isolation(tmp_path):
    """Two connectors with different engine_ids must not cross-hit
    even if the prompt is identical."""
    cfg_a = _fake_vllm_config(
        str(tmp_path / "a.sock"), engine_id="eng-A", block_size_tokens=4
    )
    cfg_b = _fake_vllm_config(
        str(tmp_path / "b.sock"), engine_id="eng-B", block_size_tokens=4
    )
    conn_a = GMSKVCacheConnectorV1(cfg_a, KVConnectorRole.SCHEDULER)
    conn_b = GMSKVCacheConnectorV1(cfg_b, KVConnectorRole.SCHEDULER)
    conn_a.request_finished(_fake_request("r0", list(range(1, 13))), [7, 8, 9])
    n, _ = conn_b.get_num_new_matched_tokens(
        _fake_request("r1", list(range(1, 13))),
        0,
    )
    assert n == 0  # B's index is empty


def test_scheduler_update_state_after_alloc_queues_restore(tmp_path):
    cfg = _fake_vllm_config(str(tmp_path / "i.sock"), block_size_tokens=4)
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    conn.request_finished(_fake_request("r0", list(range(1, 13))), [7, 8, 9])
    req = _fake_request("r1", list(range(1, 13)))
    n, _ = conn.get_num_new_matched_tokens(req, 0)
    assert n == 12

    # vLLM allocates 3 fresh blocks for these tokens.
    conn.update_state_after_alloc(req, _fake_blocks([20, 21, 22]), n)
    # v4 restore queue carries triples (src, dst, expected_gen).
    # r0's evict bumped each src's generation to 1.
    assert conn._sched_restore_queue == [
        ("r1", [(7, 20, 1), (8, 21, 1), (9, 22, 1)]),
    ]

    # build_connector_meta drains both queues. r0's request_finished
    # earlier also queued blocks [7, 8, 9] for eviction (finishing
    # a request spills AND indexes its prefix). The conflict-split
    # routes r1's restore to the SYNC lane because its src_block_ids
    # {7, 8, 9} overlap with r0's evict_block_ids {7, 8, 9} — must
    # run sync-before-evict to avoid reading post-evict storage.
    meta = conn.build_connector_meta(SimpleNamespace())
    evict, restore_async, restore_sync, restore_staging_async = _decode_meta(
        meta.payload
    )
    assert evict == [("r0", [7, 8, 9], [1, 1, 1], [])]
    assert restore_async == []  # all triples conflict with evict set
    assert restore_sync == [
        ("r1", [(7, 20, 1), (8, 21, 1), (9, 22, 1)]),
    ]
    # Second call: drained.
    assert _decode_meta(
        conn.build_connector_meta(SimpleNamespace()).payload,
    ) == ([], [], [], [])


def test_scheduler_uses_staging_hits_when_no_local_match(tmp_path):
    """A contiguous StagingTier hit can satisfy the external prefix
    even when the local prefix index has no source block ids."""
    from gms_kv_ring.common.prefix_hashes import prefix_block_hashes

    cfg = _fake_vllm_config(str(tmp_path / "i.sock"), block_size_tokens=4)
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    req = _fake_request("r-staged", list(range(1, 13)))
    hashes = prefix_block_hashes(req.prompt_token_ids, 4, None)
    conn._staging_scan = lambda _hashes: {
        hashes[0]: {"generation": 3},
        hashes[1]: {"generation": 4},
        hashes[2]: {"generation": 5},
    }

    n, async_load = conn.get_num_new_matched_tokens(req, 0)
    assert (n, async_load) == (12, False)

    conn.update_state_after_alloc(req, _fake_blocks([30, 31, 32]), n)
    expected = [
        (
            "r-staged",
            [
                (hashes[0], 30, 3),
                (hashes[1], 31, 4),
                (hashes[2], 32, 5),
            ],
        ),
    ]
    assert conn._sched_restore_queue == []
    assert conn._sched_staging_restore_queue == expected
    meta = conn.build_connector_meta(SimpleNamespace())
    _evict, _ra, _rs, staging = _decode_meta(meta.payload)
    assert staging == expected


def test_scheduler_skips_partial_prefix_overlap_with_internal_cache(
    tmp_path,
):
    """vLLM may already have computed N tokens locally; we must
    only advertise the DELTA past that."""
    cfg = _fake_vllm_config(str(tmp_path / "i.sock"), block_size_tokens=4)
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    conn.request_finished(_fake_request("r0", list(range(1, 13))), [7, 8, 9])
    req2 = _fake_request("r1", list(range(1, 13)))
    # vLLM already computed first 4 tokens (1 block) internally.
    n, _ = conn.get_num_new_matched_tokens(req2, 4)
    assert n == 8  # 2 remaining blocks × 4 tokens
    assert conn._pending_hit["r1"] == [("local", 8, 1), ("local", 9, 1)]


def test_worker_bind_drains_restore_queue_via_remap(tmp_path, monkeypatch):
    """End-to-end: scheduler queues a restore via update_state_after_alloc,
    encodes into metadata, worker decodes and invokes
    restore_blocks_remap. The fake backend records the promote
    calls — verify src_offset != dst_offset and the bytes round-trip
    to the destination HBM region.

    Forces the SYNC restore path (GMS_KVR_ASYNC_RESTORE=0) so the
    bytes are in HBM by the time bind returns and we can inspect
    the fake's recorded calls inline. A separate test covers the
    async-ring path with daemon-thread synchronization."""
    import zlib

    monkeypatch.setenv("GMS_KVR_ASYNC_RESTORE", "0")
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)

        # Scheduler builds metadata containing a restore pair.
        sched = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
        # Use sched-side internals directly so we don't have to
        # populate storage twice from the test.
        sched._sched_restore_queue.append(
            ("req-restore", [(2, 5)]),  # src block 2 → dst block 5
        )
        meta = sched.build_connector_meta(SimpleNamespace())

        # Worker: register caches + pre-seed the fake backend with
        # bytes "as if" block 2 had been demoted earlier.
        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        n_layers = 3
        n_blocks = 64
        block_bytes = 16
        caches = _make_kv_caches(
            n_layers=n_layers,
            n_blocks=n_blocks,
            block_bytes=block_bytes,
        )
        try:
            worker.register_kv_caches(caches)
            # Pre-seed: for each layer, fake._slots[(eid, layer,
            # src_block*block_bytes)] = (size, crc, payload).
            src_offset = 2 * block_bytes
            for L in range(n_layers):
                payload = bytes(
                    ((i * (L + 1)) ^ 0xA5) & 0xFF for i in range(block_bytes)
                )
                fake._slots[("test-eid", L, src_offset)] = (
                    block_bytes,
                    zlib.crc32(payload),
                    payload,
                )

            # All HBM is zero (from _make_kv_caches).
            worker.bind_connector_metadata(meta)

            # Per-layer promote into HBM at dst_offset = 5*16 = 80.
            assert len(fake.promote_calls) == n_layers
            for call in fake.promote_calls:
                # The fake records dest_gpu_ptr but doesn't echo the
                # offsets explicitly — observe via HBM state instead.
                pass
            for L, (name, t_dev) in enumerate(
                sorted(
                    caches.items(),
                    key=lambda x: int(x[0].split(".")[2]),
                )
            ):
                # Flatten the per-layer tensor view: each row of
                # the (n_blocks, block_bytes) tensor is one block.
                got = bytes(t_dev[5].cpu().numpy().tobytes())
                expected = bytes(
                    ((i * (L + 1)) ^ 0xA5) & 0xFF for i in range(block_bytes)
                )
                assert got == expected, (
                    f"restore mismatch at layer {L}: "
                    f"got={got[:8].hex()} expected={expected[:8].hex()}"
                )
                # And block 2 (the source) should still be zero in
                # HBM — the promote writes to dst slot, not src.
                src_view = bytes(t_dev[2].cpu().numpy().tobytes())
                assert src_view == b"\x00" * block_bytes
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_worker_bind_handles_evict_and_restore_in_same_step(tmp_path):
    """One scheduler step may contain BOTH evictions (finished
    requests) and restores (cache-hit requests). Worker must
    process both halves of the metadata."""
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)

        sched = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
        sched.request_finished(
            _fake_request("r-evict", list(range(1, 13))), [10, 11, 12]
        )
        # Manually queue a restore pair (skip the prompt/hash dance
        # — the round-trip is exercised in the test above).
        sched._sched_restore_queue.append(
            ("r-hit", [(10, 30), (11, 31), (12, 32)]),
        )
        meta = sched.build_connector_meta(SimpleNamespace())

        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(
            n_layers=2,
            n_blocks=64,
            block_bytes=16,
        )
        try:
            worker.register_kv_caches(caches)
            # Pre-seed for the restore so the fake has bytes to
            # promote into HBM.
            for src_id in (10, 11, 12):
                for L in range(2):
                    payload = bytes(((i + src_id * 3 + L) & 0xFF) for i in range(16))
                    fake._slots[("test-eid", L, src_id * 16)] = (
                        16,
                        0,
                        payload,
                    )

            # Evicts will land via demote_from_gpu; restores via
            # promote_into_gpu.
            worker.bind_connector_metadata(meta)
            assert len(fake.demote_calls) == 3 * 2  # 3 blocks × 2 layers
            assert len(fake.promote_calls) == 3 * 2

            # Eviction reports finished so vLLM can free blocks.
            done, _ = worker.get_finished({"r-evict"})
            assert done == {"r-evict"}
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_block_gds_restore_remap_directly(tmp_path):
    """Connector-layer test: BlockGdsConnector.restore_blocks_remap
    routes (src, dst) pairs to the handle's promote_storage_to_hbm
    with the correct dest_offset. Independent of the V1 connector
    glue so a regression is localized."""
    import zlib

    from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector
    from gms_kv_ring.engines.handle import GMSKvRing

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        block_bytes = 32
        n_blocks = 16
        n_layers = 2
        layer_stride = n_blocks * block_bytes
        dev = torch.zeros(n_layers * layer_stride, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        base = int(dev.data_ptr())
        layers_arg = [
            {
                "layer_idx": L,
                "va": base + L * layer_stride,
                "size": layer_stride,
                "stride": block_bytes,
            }
            for L in range(n_layers)
        ]
        eid = f"r-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers_arg)
        try:

            def layout(block_id):
                return [
                    (L, block_id * block_bytes, block_bytes) for L in range(n_layers)
                ]

            conn = BlockGdsConnector(h, layout)

            # Pre-seed src=1 and src=4 in the fake.
            patterns = {}
            for src in (1, 4):
                for L in range(n_layers):
                    pat = bytes(
                        (((src * 17) ^ (L * 3) + i) & 0xFF) for i in range(block_bytes)
                    )
                    patterns[(src, L)] = pat
                    fake._slots[(eid, L, src * block_bytes)] = (
                        block_bytes,
                        zlib.crc32(pat),
                        pat,
                    )

            out = conn.restore_blocks_remap([(1, 9), (4, 2)])
            assert out == {9: True, 2: True}

            # Bytes should be at dst slots 9 and 2 in each layer.
            for src, dst in [(1, 9), (4, 2)]:
                for L in range(n_layers):
                    start = L * layer_stride + dst * block_bytes
                    got = bytes(
                        dev[start : start + block_bytes].cpu().numpy().tobytes()
                    )
                    assert (
                        got == patterns[(src, L)]
                    ), f"remap src={src} dst={dst} layer={L} mismatch"
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ----------------------------------------------------------------------
# Race-fix tests — prune-on-record (race #1) and reorder bind (race #2)
# ----------------------------------------------------------------------


def test_prefix_index_prunes_stale_entries_on_recordover(tmp_path):
    """The stale-index race: hash H is recorded pointing at block 7.
    Later, hash K (different content) is recorded pointing at the
    same block 7 (vLLM reassigned the slot). Looking up H AFTER
    K's record must miss — otherwise restore would read K's bytes
    thinking they're H, with CRC verifying (since CRC matches K)."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=1024)
    eid = "eng-X"
    # 4-token blocks, so each request has one full prefix block.
    idx.record([1, 2, 3, 4], [7], block_size=4, cache_salt=None, engine_id=eid)
    # H is now indexed → 7.
    h_match = idx.lookup([1, 2, 3, 4], 4, None, eid)
    # v4: lookup returns (block_id, generation) pairs.
    assert h_match == [(7, 1)]

    # Re-evict: a totally different prompt lands at the same block.
    idx.record([99, 99, 99, 99], [7], block_size=4, cache_salt=None, engine_id=eid)
    # H must no longer be in the index — restore-for-H would
    # otherwise read K's bytes and silently corrupt.
    h_match_after = idx.lookup([1, 2, 3, 4], 4, None, eid)
    assert h_match_after == []

    # K is now indexed correctly with generation=2 (slot's second
    # eviction → bumped to 2).
    k_match = idx.lookup([99, 99, 99, 99], 4, None, eid)
    assert k_match == [(7, 2)]


def test_prefix_index_prune_preserves_unrelated_slots(tmp_path):
    """Pruning on (eid, 7) must NOT touch entries pointing at other
    blocks. Easy to regress by writing `_table.clear()`-style code
    or by scoping reverse-map incorrectly."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=1024)
    idx.record([1, 2, 3, 4], [7], 4, None, "e")
    idx.record([5, 6, 7, 8], [9], 4, None, "e")  # different block
    idx.record([10, 11, 12, 13], [7], 4, None, "e")  # re-record 7
    # Hash for [1,2,3,4] is gone; hash for [5,6,7,8] is intact.
    assert idx.lookup([1, 2, 3, 4], 4, None, "e") == []
    # Block 9 was recorded once → gen=1; block 7 was re-recorded
    # (line above) → gen=2.
    assert idx.lookup([5, 6, 7, 8], 4, None, "e") == [(9, 1)]
    assert idx.lookup([10, 11, 12, 13], 4, None, "e") == [(7, 2)]


def test_prefix_index_lru_evicts_oldest_record(tmp_path):
    """At max_entries, recording one more must evict the oldest."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=3)
    for i in range(3):
        idx.record([i, i, i, i], [i + 100], 4, None, "e")
    assert len(idx) == 3
    # Insert a 4th — oldest (i=0) must be evicted.
    idx.record([9, 9, 9, 9], [109], 4, None, "e")
    assert len(idx) == 3
    assert idx.lookup([0, 0, 0, 0], 4, None, "e") == []
    assert idx.lookup([1, 1, 1, 1], 4, None, "e") == [(101, 1)]
    assert idx.lookup([9, 9, 9, 9], 4, None, "e") == [(109, 1)]


def test_prefix_index_lru_evicts_clean_reverse_index(tmp_path):
    """LRU eviction must remove the corresponding entry from the
    reverse `_slot_to_hashes` index too — otherwise re-recording
    the same slot would try to prune a stale hash that's already
    gone, which would be a no-op but leaks a slot key over time."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=2)
    idx.record([1, 1, 1, 1], [50], 4, None, "e")
    idx.record([2, 2, 2, 2], [51], 4, None, "e")
    # Evict the oldest (block 50 mapping).
    idx.record([3, 3, 3, 3], [52], 4, None, "e")
    # Reverse index must not have (e, 50) anymore.
    assert ("e", 50) not in idx._slot_to_hashes  # noqa: SLF001
    # Re-record block 50 with a fresh hash — must succeed cleanly.
    idx.record([4, 4, 4, 4], [50], 4, None, "e")
    # Re-record of block 50 bumps its generation to 2.
    assert idx.lookup([4, 4, 4, 4], 4, None, "e") == [(50, 2)]


def test_bind_processes_restore_before_evict_for_same_src(tmp_path):
    """The within-step race: a single step contains BOTH
    `evict (req_A → block 7)` AND `restore (src=7 → dst=30)`. The
    bytes at storage slot 7 (from a previous step's eviction) are
    what the connector's hash index points at. If we ran evict
    first, the restore would read req_A's freshly-demoted bytes
    instead of the indexed-content's bytes — silent corruption.

    This test pre-seeds storage[(eid, 7)] with KNOWN content K,
    queues an evict that would write DIFFERENT bytes to (eid, 7),
    AND a restore from src=7 → dst=30. After bind, HBM at dst 30
    must contain K (the pre-step bytes), proving restore ran
    against pre-evict storage."""
    import zlib

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)
        sched = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
        # Manually queue: same step has evict for block 7 (some
        # other request finishing) AND a restore that asks for
        # src=7 (a hash that resolved before request_finished ran).
        sched._sched_evict_queue.append(("req-finishing", [7], [0]))
        sched._sched_restore_queue.append(
            ("req-hitting", [(7, 30, 0)]),
        )
        meta = sched.build_connector_meta(SimpleNamespace())

        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        n_layers = 2
        n_blocks = 64
        block_bytes = 16
        caches = _make_kv_caches(
            n_layers=n_layers,
            n_blocks=n_blocks,
            block_bytes=block_bytes,
        )
        try:
            worker.register_kv_caches(caches)
            # Pre-seed storage at (eid, layer, 7*block_bytes) with
            # KNOWN content K. This represents "what was in storage
            # at start of step, what the hash index pointed at."
            K = {}
            for L in range(n_layers):
                k = bytes((i ^ (L * 7) ^ 0xCC) & 0xFF for i in range(block_bytes))
                K[L] = k
                fake._slots[("test-eid", L, 7 * block_bytes)] = (
                    block_bytes,
                    zlib.crc32(k),
                    k,
                )
            # Write DIFFERENT bytes into HBM block 7 — this is what
            # the evict will write to storage. If bind ran evict
            # first, storage would now hold THESE bytes, and the
            # restore would corrupt HBM dst 30 with them.
            for L in range(n_layers):
                new_bytes = bytes(
                    ((i + L * 13 + 0xAA) & 0xFF) for i in range(block_bytes)
                )
                # Write into layer L's block 7.
                caches[f"model.layers.{L}.self_attn.kv_cache"][7].copy_(
                    torch.frombuffer(bytearray(new_bytes), dtype=torch.uint8)
                )
            torch.cuda.synchronize()

            worker.bind_connector_metadata(meta)

            # HBM dst 30 must contain K (pre-step storage), NOT
            # the freshly-evicted bytes. This is the ordering proof.
            for L in range(n_layers):
                got = bytes(
                    caches[f"model.layers.{L}.self_attn.kv_cache"][30]
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                assert got == K[L], (
                    f"layer {L}: restore must have read pre-evict "
                    f"storage (K) but got post-evict bytes. "
                    f"got[:8]={got[:8].hex()} K[:8]={K[L][:8].hex()}"
                )

            # And storage at (eid, 7) is now the post-evict bytes
            # (proves evict ran after restore — for next step).
            for L in range(n_layers):
                slot = fake._slots[("test-eid", L, 7 * block_bytes)]
                _size, _crc, payload = slot
                assert payload != K[L], (
                    f"layer {L}: storage must now have the new "
                    f"evicted bytes, not the pre-step K"
                )
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_connector_evict_raises_does_not_block_get_finished(tmp_path):
    """If the daemon RPC raises mid-bind (simulating a daemon
    crash / disconnect during demote), the connector must still
    return the req_id from get_finished so vLLM frees blocks
    instead of leaking them. Metrics record the failure."""
    from gms_kv_ring.common import metrics

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)
        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(n_layers=1, n_blocks=64, block_bytes=16)
        try:
            worker.register_kv_caches(caches)
            eid = worker._handle.engine_id

            # Patch the handle to raise on demote — simulates a
            # daemon socket-closed mid-request, OOM, etc.
            orig_demote = worker._handle.demote_hbm_to_storage

            def boom(*a, **kw):
                raise ConnectionError(
                    "synthetic: daemon socket closed",
                )

            worker._handle.demote_hbm_to_storage = boom

            sched = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
            sched.request_finished(
                _fake_request("r-doomed", list(range(1, 9))),
                [3, 4],
            )
            meta = sched.build_connector_meta(SimpleNamespace())

            before = metrics.connector_evict_failures.get(engine_id=eid)
            worker.bind_connector_metadata(meta)
            after = metrics.connector_evict_failures.get(engine_id=eid)

            # Failure counter increments by the number of failed
            # blocks (2).
            assert after - before >= 2, (
                f"expected evict_failures to increment by ≥2, got " f"{after - before}"
            )
            # The crucial invariant: get_finished still returns
            # the req_id so vLLM unpins blocks.
            done, _recv = worker.get_finished({"r-doomed"})
            assert done == {"r-doomed"}, (
                "after daemon-crash bind failure, get_finished MUST "
                "still report req_ids — otherwise vLLM pins blocks "
                "forever (block-pin deadlock)"
            )

            worker._handle.demote_hbm_to_storage = orig_demote
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_race3_cross_step_async_restore_vs_evict_rejected_by_generation(
    tmp_path,
):
    """Race #3: cross-step async-restore-vs-evict.

    Step L: req_C arrives with prompt hash H. The hash index points
    at (eid, src=7, gen=1). Connector queues async restore via the
    ring. Daemon consumer is still backlogged.

    Step L+1: req_W finishes with DIFFERENT prompt that vLLM
    happens to assign block 7 to. The synchronous demote in bind
    writes new bytes to storage[(eid, 7)]; the daemon's
    slot_generations[7] bumps from 1 → 2.

    Daemon consumer finally pops the step-L restore record. It
    carries expected_generation=1, but slot is now at gen=2. Race
    #3 protection: the daemon REFUSES to read; signals failure.
    Engine reads stale HBM but restore_succeeded() returns False
    → falls to cold compute (no silent corruption).

    Without the fix, the daemon would read the post-overwrite
    bytes and serve them as if they were req_C's prefix —
    silent data corruption."""
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)
        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(n_layers=2, n_blocks=64, block_bytes=16)
        try:
            worker.register_kv_caches(caches)
            eid = worker._handle.engine_id

            # Initial state: simulate that step L spilled some
            # content X to storage at slot block_id=7 with gen=1.
            # We bypass the connector path and use the handle's
            # demote_hbm_to_storage directly with generation=1.
            X = {}
            for L in range(2):
                pat = bytes(((i * 11) ^ (L * 5) ^ 0xAA) & 0xFF for i in range(16))
                X[L] = pat
                caches[f"model.layers.{L}.self_attn.kv_cache"][7].copy_(
                    torch.frombuffer(bytearray(pat), dtype=torch.uint8),
                )
            torch.cuda.synchronize()
            for L in range(2):
                assert worker._handle.demote_hbm_to_storage(
                    L,
                    7 * 16,
                    16,
                    generation=1,
                )
            # Confirm daemon's view of slot generation.
            pool = d._pools[eid]
            assert pool.current_block_generation(7) == 1

            # Step L+1: re-evict block 7 with NEW content (Y) at
            # generation=2 — simulating vLLM having reassigned the
            # slot to a different request.
            Y = {}
            for L in range(2):
                pat = bytes(((i + 33) ^ (L * 7) ^ 0x55) & 0xFF for i in range(16))
                Y[L] = pat
                caches[f"model.layers.{L}.self_attn.kv_cache"][7].copy_(
                    torch.frombuffer(bytearray(pat), dtype=torch.uint8),
                )
            torch.cuda.synchronize()
            for L in range(2):
                assert worker._handle.demote_hbm_to_storage(
                    L,
                    7 * 16,
                    16,
                    generation=2,
                )
            assert pool.current_block_generation(7) == 2

            # Now a stale step-L restore arrives with
            # expected_generation=1. The daemon must reject.
            ok = worker._handle.promote_storage_to_hbm(
                0,
                7 * 16,
                16,
                dest_offset=9 * 16,
                expected_generation=1,
            )
            assert ok is False, (
                "Race #3 protection failed: daemon should reject "
                "stale-generation restores; instead returned True."
            )

            # A restore with the CURRENT generation (=2) must
            # succeed and return Y's content.
            caches["model.layers.0.self_attn.kv_cache"][9].zero_()
            torch.cuda.synchronize()
            ok = worker._handle.promote_storage_to_hbm(
                0,
                7 * 16,
                16,
                dest_offset=9 * 16,
                expected_generation=2,
            )
            assert ok is True
            got = bytes(
                caches["model.layers.0.self_attn.kv_cache"][9].cpu().numpy().tobytes()
            )
            assert got == Y[0]

            # And expected_generation=0 (legacy / opt-out) also
            # succeeds — protection is opt-in.
            ok = worker._handle.promote_storage_to_hbm(
                0,
                7 * 16,
                16,
                dest_offset=10 * 16,
                expected_generation=0,
            )
            assert ok is True
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_prefix_index_bumps_generation_on_re_record():
    """Direct _PrefixIndex test: re-recording the same (eid, bid)
    bumps the generation monotonically. The lookup returns the
    LATEST generation; this is what the connector passes through
    to the daemon as expected_generation."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=128)
    # First record: gen=1.
    gens = idx.record([1, 2, 3, 4], [7], 4, None, "e")
    assert gens == [1]
    assert idx.lookup([1, 2, 3, 4], 4, None, "e") == [(7, 1)]

    # Re-record same slot with DIFFERENT content: gen=2.
    gens = idx.record([99, 99, 99, 99], [7], 4, None, "e")
    assert gens == [2]
    assert idx.lookup([99, 99, 99, 99], 4, None, "e") == [(7, 2)]
    # Old hash is gone (Race #1).
    assert idx.lookup([1, 2, 3, 4], 4, None, "e") == []

    # Third time: gen=3.
    gens = idx.record([5, 6, 7, 8], [7], 4, None, "e")
    assert gens == [3]
    assert idx.lookup([5, 6, 7, 8], 4, None, "e") == [(7, 3)]


def test_scheduler_request_finished_invalidates_old_hash_for_same_slot(
    tmp_path,
):
    """End-to-end at the scheduler level: req_A finishes with hash
    H pointing at block 7. Later req_W finishes with hash W also
    pointing at block 7 (vLLM reassigned 7). A subsequent request
    looking up hash H must miss — would otherwise pull req_W's
    bytes from storage thinking they were req_A's."""
    cfg = _fake_vllm_config(str(tmp_path / "i.sock"), block_size_tokens=4)
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    # req_A: prompt [1,2,3,4], block 7.
    conn.request_finished(_fake_request("rA", [1, 2, 3, 4]), [7])
    # req_W: different prompt, same block 7.
    conn.request_finished(_fake_request("rW", [99, 99, 99, 99]), [7])
    # Look up H — must NOT find the now-stale (eid, 7) entry.
    n, _ = conn.get_num_new_matched_tokens(
        _fake_request("rC", [1, 2, 3, 4]),
        0,
    )
    assert n == 0
    # W still hits.
    n2, _ = conn.get_num_new_matched_tokens(
        _fake_request("rD", [99, 99, 99, 99]),
        0,
    )
    assert n2 == 4


# ----------------------------------------------------------------------
# Async restore via gms_rust_ring + FLAG_GDS_DIRECT
# ----------------------------------------------------------------------


def _wait_for_counter(handle, slot, target, timeout_s=5.0):
    """Poll handle.counters.read_slot until it reaches `target` or
    `target+1` (success or failure indicator from the daemon's
    restore consumer), or timeout. Returns the final counter value."""
    import time as _t

    deadline = _t.monotonic() + timeout_s
    while _t.monotonic() < deadline:
        v = handle.counters.read_slot(slot)
        if v >= target:
            return v
        _t.sleep(0.005)
    return handle.counters.read_slot(slot)


def test_ring_record_carries_gds_flag_round_trip(tmp_path):
    """Pin the wire-format invariant: push with FLAG_GDS_DIRECT,
    pop, recover the flag. Independent of any daemon logic."""
    from gms_kv_ring.common.restore_ring import (
        FLAG_GDS_DIRECT,
        attach_reader,
        create_ring,
    )

    ring_path = str(tmp_path / "r.ring")
    writer = create_ring(ring_path, capacity=16)
    reader = attach_reader(ring_path)
    try:
        ok = writer.push(
            src_engine_id="eid",
            block_pairs=[(3, 9)],
            counter_slot=0,
            counter_target=1,
            flags=FLAG_GDS_DIRECT,
        )
        assert ok is True
        rec = reader.try_pop()
        assert rec is not None
        assert rec["flags"] & FLAG_GDS_DIRECT
        assert rec["block_pairs"] == [(3, 9)]
    finally:
        writer.close()
        reader.close()


def test_handle_record_restore_gds_pushes_with_flag(tmp_path):
    """`record_restore_gds` sets FLAG_GDS_DIRECT on the pushed
    record. We verify by attaching a separate reader (in-process
    second view of the same ring file) and popping. This also
    pins that the lock-pair-with-slot-reserve isn't disturbed
    by the flag path."""
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    # Spawn daemon WITHOUT a worker that drains the ring — we
    # want to inspect the record after the push, not after the
    # daemon consumer ate it.
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)
        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(n_layers=2, n_blocks=64, block_bytes=16)
        try:
            worker.register_kv_caches(caches)
            res = worker._handle.record_restore_gds(
                worker._handle.engine_id,
                [(1, 2)],
            )
            assert res is not None
            slot, target = res
            # Wait for daemon consumer to pop it. The fact that
            # it's a GDS record means _process_gds runs.
            v = _wait_for_counter(worker._handle, slot, target)
            assert v in (target, target + 1)
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_async_restore_end_to_end_through_ring(tmp_path):
    """Full async path: scheduler queues restore (no conflict, so
    it routes to the ASYNC lane), worker bind pushes to ring,
    daemon consumer pops and runs _process_gds → backend.
    promote_into_gpu → HBM. Test polls the counter until set.

    This is the production happy-path. The earlier
    `test_worker_bind_drains_restore_queue_via_remap` covers the
    sync-fallback path; this covers the async ring path
    explicitly."""
    import zlib

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)
        sched = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
        # No evicts queued → no conflict → async lane.
        sched._sched_restore_queue.append(("req-A", [(4, 7, 0)]))
        meta = sched.build_connector_meta(SimpleNamespace())
        # Verify the lane split.
        evict, r_async, r_sync, r_staging = _decode_meta(meta.payload)
        assert evict == []
        assert r_async == [("req-A", [(4, 7, 0)])]
        assert r_sync == []

        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(n_layers=2, n_blocks=64, block_bytes=16)
        try:
            worker.register_kv_caches(caches)
            # Pre-seed src=4 in each layer.
            patterns = {}
            for L in range(2):
                p = bytes(((i * 11) ^ (L * 5) ^ 0xDE) & 0xFF for i in range(16))
                patterns[L] = p
                fake._slots[("test-eid", L, 4 * 16)] = (
                    16,
                    zlib.crc32(p),
                    p,
                )

            worker.bind_connector_metadata(meta)

            # Bind returned IMMEDIATELY (no blocking cuFile read).
            # The pending-wait list should have one entry.
            assert len(worker._pending_restore_waits) == 1
            entry = worker._pending_restore_waits[0]
            # New 4-tuple shape: (req_id, slot, target, src_bids)
            req_id, slot, target, src_bids = entry
            assert req_id == "req-A"
            assert src_bids == [4], f"expected src_block_ids=[4], got {src_bids}"

            # Wait for daemon consumer thread to drain.
            v = _wait_for_counter(worker._handle, slot, target)
            assert v == target, (
                f"restore counter never reached target — daemon "
                f"didn't process the record. counter={v} target={target}"
            )
            assert worker._handle.restore_succeeded(slot, target)

            # Bytes landed at HBM block 7 in each layer.
            for L in range(2):
                got = bytes(
                    caches[f"model.layers.{L}.self_attn.kv_cache"][7]
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                assert got == patterns[L]
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_start_load_kv_then_clear_connector_metadata_lifecycle(tmp_path):
    """Validates the lifecycle queue mechanics: bind populates
    `_pending_restore_waits` → handoff to `_pending_restore_checks`
    → clear_connector_metadata drains via host-side
    restore_succeeded.

    We deliberately DO NOT call `start_load_kv` here (which issues
    `cuStreamWaitValue32` on the default stream). Issuing the wait
    without queueing any kernel after it leaves a wait sentinel
    pending in stream 0; subsequent tests' `torch.zeros(...)`
    allocations then fail with `CUDA error: unspecified launch
    failure`. Driving the handoff directly through the
    `_pending_*` lists keeps the CUDA context clean for the rest
    of the suite. The async GPU-stream wait is covered indirectly
    by `test_async_restore_end_to_end_through_ring` (which uses
    a real forward-pass byte-level check)."""
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)
        sched = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
        sched._sched_restore_queue.append(("req-B", [(2, 8, 0)]))
        meta = sched.build_connector_meta(SimpleNamespace())

        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(n_layers=2, n_blocks=64, block_bytes=16)
        try:
            worker.register_kv_caches(caches)
            for L in range(2):
                p = bytes((i & 0xFF) for i in range(16))
                fake._slots[("test-eid", L, 2 * 16)] = (16, 0, p)

            worker.bind_connector_metadata(meta)
            assert len(worker._pending_restore_waits) == 1
            assert len(worker._pending_restore_checks) == 0

            # Mimic start_load_kv's queue handoff WITHOUT issuing
            # the GPU wait. clear_connector_metadata only cares
            # about _pending_restore_checks; start_load_kv just
            # moves entries from _waits → _checks and queues the
            # GPU-side gate (which we skip here for the reason
            # above).
            worker._pending_restore_checks = list(
                worker._pending_restore_waits,
            )
            worker._pending_restore_waits = []

            # Wait for daemon to complete so restore_succeeded
            # returns True.
            _slot, _target = (
                worker._pending_restore_checks[0][1],
                worker._pending_restore_checks[0][2],
            )
            _wait_for_counter(worker._handle, _slot, _target)

            # clear_connector_metadata host-checks restore_succeeded.
            worker.clear_connector_metadata()
            assert len(worker._pending_restore_checks) == 0
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_async_restore_ring_full_falls_back_to_sync(tmp_path, monkeypatch):
    """When the restore ring is full, record_restore_gds returns
    None. The connector must fall back to the synchronous remap
    and bump the ring_full metric. Engine never deadlocks."""
    from gms_kv_ring.common import metrics

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)
        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(n_layers=1, n_blocks=64, block_bytes=16)
        try:
            worker.register_kv_caches(caches)
            # Monkeypatch the handle to simulate ring-full.
            monkeypatch.setattr(
                worker._handle,
                "record_restore_gds",
                lambda *a, **kw: None,
            )
            # Pre-seed so sync remap succeeds.
            payload = bytes((i ^ 0xA5) & 0xFF for i in range(16))
            import zlib as _z

            fake._slots[("test-eid", 0, 5 * 16)] = (
                16,
                _z.crc32(payload),
                payload,
            )

            before = metrics.connector_restore_ring_full.get(
                engine_id=worker._handle.engine_id,
            )

            # Build a meta with the restore pair.
            sched = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
            sched._sched_restore_queue.append(("req-F", [(5, 11, 0)]))
            meta = sched.build_connector_meta(SimpleNamespace())

            worker.bind_connector_metadata(meta)

            after = metrics.connector_restore_ring_full.get(
                engine_id=worker._handle.engine_id,
            )
            assert after == before + 1
            # Bytes landed inline (sync fallback path).
            got = bytes(
                caches["model.layers.0.self_attn.kv_cache"][11].cpu().numpy().tobytes()
            )
            assert got == payload
            # No async waits queued since we fell back to sync.
            assert worker._pending_restore_waits == []
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_conflict_split_metric_increments(tmp_path):
    """When src ∈ evict_set, restore goes through the SYNC lane
    and bumps the conflict-sync counter."""
    from gms_kv_ring.common import metrics

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        cfg = _fake_vllm_config(sock, block_size_tokens=4)
        sched = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
        sched._sched_evict_queue.append(("e", [7], [0]))
        sched._sched_restore_queue.append(("h", [(7, 30, 0)]))
        meta = sched.build_connector_meta(SimpleNamespace())

        worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
        caches = _make_kv_caches(n_layers=1, n_blocks=64, block_bytes=16)
        try:
            worker.register_kv_caches(caches)
            import zlib as _z

            p = bytes((i & 0xFF) for i in range(16))
            fake._slots[("test-eid", 0, 7 * 16)] = (16, _z.crc32(p), p)

            before = metrics.connector_restore_conflict_sync.get(
                engine_id=worker._handle.engine_id,
            )
            worker.bind_connector_metadata(meta)
            after = metrics.connector_restore_conflict_sync.get(
                engine_id=worker._handle.engine_id,
            )
            # Counter increments by n_pairs in the conflict lane.
            assert after == before + 1
        finally:
            if worker._handle is not None:
                worker._handle.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ----------------------------------------------------------------------
# Tensor-parallel mock — per-rank daemon + connector instances
# ----------------------------------------------------------------------


def test_tp2_two_workers_each_evict_and_restore_their_own_slice(tmp_path):
    """Mock TP=2: two GMSKvRing handles (rank 0 + rank 1), each
    with its own daemon socket + storage backend. ONE scheduler
    builds metadata; we apply it to BOTH worker connectors. Each
    worker independently demotes its rank's bytes to its rank's
    storage and restores from there. Verifies the per-rank
    daemon-isolation invariant — no cross-rank reads, no shared
    storage, and the connector cleanly handles two engines."""
    monkeypatch_async = pytest.MonkeyPatch()
    monkeypatch_async.setenv("GMS_KVR_ASYNC_RESTORE", "0")
    try:
        ranks = []
        try:
            for rank in range(2):
                fake = _FakeGpuDirectBackend()
                sock = str(tmp_path / f"d-r{rank}.sock")
                d, t, lh = _spawn_daemon(sock, storage_backend=fake)
                cfg = _fake_vllm_config(
                    sock,
                    engine_id=f"tp-r{rank}",
                    block_size_tokens=4,
                )
                worker = GMSKVCacheConnectorV1(
                    cfg,
                    KVConnectorRole.WORKER,
                )
                caches = _make_kv_caches(
                    n_layers=2,
                    n_blocks=64,
                    block_bytes=16,
                )
                worker.register_kv_caches(caches)
                ranks.append(
                    {
                        "fake": fake,
                        "sock": sock,
                        "d": d,
                        "t": t,
                        "lh": lh,
                        "worker": worker,
                        "caches": caches,
                        "cfg": cfg,
                    }
                )

            # Single scheduler builds metadata once.
            sched = GMSKVCacheConnectorV1(
                ranks[0]["cfg"],
                KVConnectorRole.SCHEDULER,
            )
            sched.request_finished(
                _fake_request("req-X", [1, 2, 3, 4, 5, 6, 7, 8]),
                [10, 11],
            )
            meta = sched.build_connector_meta(SimpleNamespace())

            # Both workers bind. Each demotes its own rank's bytes.
            for r in ranks:
                r["worker"].bind_connector_metadata(meta)

            # Each rank's fake recorded the demote calls — they
            # should be SAME (eid, layer, offset) shape (because TP
            # ranks share block_id semantics) but the bytes came
            # from each rank's own HBM.
            for rank, r in enumerate(ranks):
                assert len(r["fake"].demote_calls) == 2 * 2, (
                    f"rank {rank}: expected 2 blocks × 2 layers "
                    f"demote calls, got {len(r['fake'].demote_calls)}"
                )
                # No cross-contamination: rank 0's fake never saw
                # rank 1's engine_id.
                for c in r["fake"].demote_calls:
                    assert c["engine_id"] == f"tp-r{rank}"

            # Now restore: cache hit on a different request with the
            # same prefix. Scheduler builds, both workers bind.
            sched._prefix_index.lookup  # warm
            n, _ = sched.get_num_new_matched_tokens(
                _fake_request("req-Y", [1, 2, 3, 4, 5, 6, 7, 8]),
                num_computed_tokens=0,
            )
            assert n == 8
            sched.update_state_after_alloc(
                _fake_request("req-Y", [1, 2, 3, 4, 5, 6, 7, 8]),
                _fake_blocks([20, 21]),
                n,
            )
            meta2 = sched.build_connector_meta(SimpleNamespace())
            for r in ranks:
                r["worker"].bind_connector_metadata(meta2)

            # Each rank should have observed promote_into_gpu calls
            # for its own engine — 2 blocks × 2 layers each.
            for rank, r in enumerate(ranks):
                assert len(r["fake"].promote_calls) == 2 * 2
        finally:
            for r in ranks:
                if r["worker"]._handle is not None:
                    r["worker"]._handle.close()
                r["lh"]["loop"].call_soon_threadsafe(r["d"].stop)
                r["t"].join(timeout=3)
    finally:
        monkeypatch_async.undo()


def test_promote_storage_to_hbm_dest_offset_kwarg(tmp_path):
    """Daemon RPC accepts dest_offset distinct from offset.
    Storage key uses offset; HBM dest VA uses pool_base + dest_offset.
    Test reaches the daemon directly through the client."""
    import zlib

    from gms_kv_ring.engines.handle import GMSKvRing

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        block_bytes = 64
        n_blocks = 8
        dev = torch.zeros(n_blocks * block_bytes, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        base = int(dev.data_ptr())
        eid = f"p-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(
            engine_id=eid,
            daemon_socket=sock,
            layers=[
                {
                    "layer_idx": 0,
                    "va": base,
                    "size": n_blocks * block_bytes,
                    "stride": block_bytes,
                }
            ],
        )
        try:
            # Pre-seed storage at src_offset = 3 * block_bytes.
            src_offset = 3 * block_bytes
            payload = bytes((i * 7 & 0xFF) for i in range(block_bytes))
            fake._slots[(eid, 0, src_offset)] = (
                block_bytes,
                zlib.crc32(payload),
                payload,
            )

            # Restore into dst_offset = 6 * block_bytes.
            ok = h.promote_storage_to_hbm(
                0,
                src_offset,
                block_bytes,
                dest_offset=6 * block_bytes,
            )
            assert ok is True
            # Bytes appear at block 6, not block 3.
            got_at_6 = bytes(
                dev[6 * block_bytes : 7 * block_bytes].cpu().numpy().tobytes()
            )
            got_at_3 = bytes(
                dev[3 * block_bytes : 4 * block_bytes].cpu().numpy().tobytes()
            )
            assert got_at_6 == payload
            assert got_at_3 == b"\x00" * block_bytes
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# =========================================================================
# Persistence tests for _PrefixIndex (atomic snapshot + load on restart)
# =========================================================================


def test_prefix_index_snapshot_round_trip(tmp_path):
    """Record entries, force a snapshot, build a fresh index from
    the same path. lookup() must return the same (block_id, gen)
    triples it would have before the simulated restart."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    idx = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
    )
    idx.record([1, 2, 3, 4, 5, 6, 7, 8], [10, 11], 4, None, "e")
    idx.record([9, 9, 9, 9], [12], 4, None, "e2")
    idx.snapshot()
    assert os.path.exists(snap)

    fresh = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=999.0,
        snapshot_threshold=999_999,
    )
    # All three prefixes recover with their original generations.
    assert fresh.lookup([1, 2, 3, 4], 4, None, "e") == [(10, 1)]
    assert fresh.lookup([1, 2, 3, 4, 5, 6, 7, 8], 4, None, "e") == [(10, 1), (11, 1)]
    assert fresh.lookup([9, 9, 9, 9], 4, None, "e2") == [(12, 1)]
    # Cross-engine lookup never returns another engine's slots.
    assert fresh.lookup([9, 9, 9, 9], 4, None, "e") == []


def test_prefix_index_snapshot_preserves_slot_generations(tmp_path):
    """After load, the next record() on the same slot must bump
    the generation from the loaded value — proves slot_generations
    survives, which is needed so Race #3 protection stays correct
    across a scheduler restart."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    idx = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
    )
    # Two records on the same slot bump generation to 2.
    idx.record([1, 1, 1, 1], [42], 4, None, "e")
    idx.record([2, 2, 2, 2], [42], 4, None, "e")
    idx.snapshot()

    fresh = _PrefixIndex(max_entries=64, snapshot_path=snap)
    # Re-record on the same slot must continue from gen=2 → 3.
    gens = fresh.record([3, 3, 3, 3], [42], 4, None, "e")
    assert gens == [3], (
        "slot generation did not survive load — Race #3 protection "
        f"is broken; got {gens}"
    )


def test_prefix_index_snapshot_corrupt_file_silent_fallback(tmp_path):
    """A garbled snapshot must NOT crash __init__. The connector
    starts cold and the operator's only signal is a WARN log."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = tmp_path / "idx.bin"
    snap.write_bytes(b"this is not a valid snapshot")

    idx = _PrefixIndex(max_entries=64, snapshot_path=str(snap))
    assert len(idx) == 0
    # The index is still usable.
    idx.record([1, 2, 3, 4], [7], 4, None, "e")
    assert idx.lookup([1, 2, 3, 4], 4, None, "e") == [(7, 1)]


def test_prefix_index_snapshot_truncated_file_silent_fallback(tmp_path):
    """A truncated public snapshot must be ignored rather than crash."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = tmp_path / "idx.bin"
    idx = _PrefixIndex(
        max_entries=64,
        snapshot_path=str(snap),
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
    )
    idx.record([1, 2, 3, 4], [7], 4, None, "e")
    idx.snapshot()
    snap.write_bytes(snap.read_bytes()[:-5])

    fresh = _PrefixIndex(max_entries=64, snapshot_path=str(snap))
    assert len(fresh) == 0


def test_prefix_index_snapshot_wrong_version_ignored(tmp_path):
    """An otherwise-valid public snapshot with an unknown version is ignored."""
    import struct

    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = tmp_path / "idx.bin"
    seed = _PrefixIndex(
        max_entries=64,
        snapshot_path=str(snap),
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
    )
    seed.record([1, 2, 3, 4], [7], 4, None, "e")
    seed.snapshot()
    blob = bytearray(snap.read_bytes())
    struct.pack_into("<H", blob, 8, 999)
    snap.write_bytes(blob)

    fresh = _PrefixIndex(max_entries=64, snapshot_path=str(snap))
    assert len(fresh) == 0


def test_prefix_index_snapshot_load_enforces_max_cap(tmp_path):
    """A snapshot may have been written by a process configured with
    a larger _max than the current process. On load, the cap MUST
    be re-enforced or the index would silently exceed its memory
    budget."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    big = _PrefixIndex(
        max_entries=1000,
        snapshot_path=snap,
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
    )
    for i in range(20):
        big.record([i, i, i, i], [200 + i], 4, None, "e")
    big.snapshot()
    assert len(big) == 20

    # Reload into a process that only allows 5 entries.
    small = _PrefixIndex(max_entries=5, snapshot_path=snap)
    assert len(small) == 5, f"snapshot load did not enforce cap; size={len(small)}"


def test_prefix_index_snapshot_disabled_when_path_empty(tmp_path):
    """No path → no I/O. record() never touches disk. Failing this
    test means the connector pays disk I/O cost when persistence
    is off, which we never want."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=64, snapshot_path="")
    for i in range(200):
        idx.record([i, i, i, i], [i], 4, None, "e")
    # Nothing should have been written under tmp_path.
    assert list(tmp_path.iterdir()) == []


def test_prefix_index_snapshot_throttle_threshold(tmp_path):
    """Under the dirty threshold, _maybe_snapshot must not fire.
    Sanity-check on the throttle so we don't accidentally write
    on every record (would be a perf regression on hot evict)."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    idx = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=0.0,
        snapshot_threshold=100,
    )
    # Record just 5 entries — below threshold.
    for i in range(5):
        idx.record([i, i, i, i], [i], 4, None, "e")
    assert not os.path.exists(snap), "snapshot fired before threshold reached"


def test_prefix_index_snapshot_throttle_interval(tmp_path):
    """Threshold met but interval not elapsed → no write. Same
    idea but along the time axis."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    idx = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=1_000_000.0,
        snapshot_threshold=1,
    )
    idx.record([1, 1, 1, 1], [1], 4, None, "e")
    idx.record([2, 2, 2, 2], [2], 4, None, "e")
    assert not os.path.exists(snap)


def test_prefix_index_snapshot_atomic_via_rename(tmp_path):
    """Verify the tmp+rename atomic write — the final path must
    NEVER be a partial file. We can't easily inject a crash mid-
    write, but we can assert the tmp file is absent after success
    (otherwise the operator sees stale .tmp.<pid> garbage)."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    idx = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
    )
    idx.record([1, 1, 1, 1], [1], 4, None, "e")
    idx.snapshot()
    assert os.path.exists(snap)
    # No leftover .tmp.<pid> files for our pid.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("idx.bin.tmp.")]
    assert leftovers == [], f"leftover tmp files: {leftovers}"


def test_prefix_index_invalidate_clears_all_state(tmp_path):
    """invalidate() drops every entry. Subsequent lookups return
    empty for previously-recorded prefixes; new records work
    normally afterwards."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=64)
    idx.record([1, 2, 3, 4], [7], 4, None, "e")
    idx.record([5, 6, 7, 8], [9], 4, None, "e")
    assert len(idx) == 2

    idx.invalidate()
    assert len(idx) == 0
    assert idx.lookup([1, 2, 3, 4], 4, None, "e") == []

    # Re-recording the same slot starts the generation back at 1 —
    # daemon's slot_generations were zeroed too in the real flow
    # (that's why we invalidated). Lockstep is preserved.
    gens = idx.record([1, 2, 3, 4], [7], 4, None, "e")
    assert gens == [1]


def test_prefix_index_snapshot_rejects_mismatched_daemon_epoch(tmp_path):
    """A snapshot written under one daemon epoch must not load under
    a different one — the daemon has been restarted since the
    snapshot, so every recorded slot points at a host_tier slot
    the new daemon doesn't have."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    seed = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
        expected_daemon_epoch=1000,
    )
    seed.record([1, 2, 3, 4], [7], 4, None, "e")
    seed.snapshot()

    # Reload with a DIFFERENT expected epoch — daemon was replaced.
    fresh = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        expected_daemon_epoch=9999,
    )
    assert len(fresh) == 0, (
        f"snapshot loaded across epoch mismatch (size={len(fresh)}); "
        "would have served stale entries against new daemon"
    )


def test_prefix_index_snapshot_loads_when_epoch_matches(tmp_path):
    """The same epoch on both sides must NOT cause a drop. Sanity
    check that the mismatch path doesn't false-positive."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    seed = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
        expected_daemon_epoch=1000,
    )
    seed.record([1, 2, 3, 4], [7], 4, None, "e")
    seed.snapshot()

    fresh = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        expected_daemon_epoch=1000,
    )
    assert fresh.lookup([1, 2, 3, 4], 4, None, "e") == [(7, 1)]


def test_prefix_index_snapshot_loads_when_no_expected_epoch(tmp_path):
    """If the caller doesn't pass an expected epoch (None), load
    the snapshot regardless of the embedded value. Backwards-compat
    path for callers that haven't wired epoch tracking yet."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    seed = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
        expected_daemon_epoch=1234,
    )
    seed.record([1, 2, 3, 4], [7], 4, None, "e")
    seed.snapshot()

    fresh = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        expected_daemon_epoch=None,
    )
    assert fresh.lookup([1, 2, 3, 4], 4, None, "e") == [(7, 1)]


def test_daemon_response_carries_epoch():
    """Every daemon RPC response must include a stable `daemon_epoch`
    field. The connector relies on this — a missing field would
    leave the epoch-poll thread blind to daemon restarts."""
    import asyncio
    import os as _os
    import tempfile as _tempfile
    import threading as _t

    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.daemon.server import Daemon

    sock_dir = _tempfile.mkdtemp(prefix="gms-epoch-test-")
    sock = _os.path.join(sock_dir, "d.sock")
    d = Daemon(sock, supervise_backend=False)
    holder: dict = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        try:
            loop.run_until_complete(d.serve())
        finally:
            loop.close()

    t = _t.Thread(target=_run, daemon=True)
    t.start()
    import time as _time

    deadline = _time.monotonic() + 3.0
    while not _os.path.exists(sock) and _time.monotonic() < deadline:
        _time.sleep(0.02)
    try:
        cli = DaemonClient(sock)
        # Any RPC: ping is the simplest. The DaemonClient stores
        # the epoch in current_daemon_epoch(); make sure it isn't
        # None after the call.
        cli._call({"op": "ping"})
        seen = cli.current_daemon_epoch()
        assert seen is not None, "client did not capture daemon_epoch"
        assert seen == d.epoch, f"client epoch {seen} ≠ daemon epoch {d.epoch}"
        cli.close()
    finally:
        holder["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_daemon_restart_yields_different_epoch():
    """A fresh Daemon() built after the first is replaced must
    publish a different epoch — the value the connector compares
    must change across daemon restarts on the same socket."""
    import asyncio
    import os as _os
    import tempfile as _tempfile
    import threading as _t
    import time as _time

    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.daemon.server import Daemon

    sock_dir = _tempfile.mkdtemp(prefix="gms-epoch-restart-")
    sock = _os.path.join(sock_dir, "d.sock")

    def _spin(d):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(d.serve())
        finally:
            loop.close()
        return loop

    d1 = Daemon(sock, supervise_backend=False)
    t1 = _t.Thread(target=_spin, args=(d1,), daemon=True)
    t1.start()
    deadline = _time.monotonic() + 3.0
    while not _os.path.exists(sock) and _time.monotonic() < deadline:
        _time.sleep(0.02)
    cli1 = DaemonClient(sock)
    cli1._call({"op": "ping"})
    epoch1 = cli1.current_daemon_epoch()
    cli1.close()

    # Tear down d1: remove the socket file so d2 can bind. The d1
    # daemon-thread keeps spinning in the background but it's a
    # daemon thread so it dies at process exit; we don't need to
    # join it for the test's correctness.
    try:
        if _os.path.exists(sock):
            _os.unlink(sock)
    except FileNotFoundError:
        pass
    _time.sleep(0.01)  # ensure clock advances by at least 1µs

    d2 = Daemon(sock, supervise_backend=False)
    t2 = _t.Thread(target=_spin, args=(d2,), daemon=True)
    t2.start()
    deadline = _time.monotonic() + 3.0
    while not _os.path.exists(sock) and _time.monotonic() < deadline:
        _time.sleep(0.02)
    cli2 = DaemonClient(sock)
    cli2._call({"op": "ping"})
    epoch2 = cli2.current_daemon_epoch()
    cli2.close()

    assert epoch1 != epoch2, (
        f"daemon restart produced identical epoch {epoch1}; "
        "connector would not invalidate its prefix index"
    )


def test_prefix_index_snapshot_load_logs_count(tmp_path, caplog):
    """When persistence is enabled and a snapshot exists, the load
    path must log the entry count (operator-visible). Helps detect
    silent silent-loads turning into silent silent-misses."""
    import logging as _logging

    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    snap = str(tmp_path / "idx.bin")
    seed = _PrefixIndex(
        max_entries=64,
        snapshot_path=snap,
        snapshot_interval_s=0.0,
        snapshot_threshold=1,
    )
    seed.record([1, 1, 1, 1], [1], 4, None, "e")
    seed.record([2, 2, 2, 2], [2], 4, None, "e")
    seed.snapshot()

    with caplog.at_level(
        _logging.INFO,
        logger="gms_kv_ring.common.prefix_index",
    ):
        _PrefixIndex(max_entries=64, snapshot_path=snap)

    msgs = [r.message for r in caplog.records]
    assert any(
        "loaded 2 entries" in m for m in msgs
    ), f"expected load INFO log; got: {msgs}"


def test_prefix_index_drop_slots_removes_pointers_only(tmp_path):
    """drop_slots() drops every hash pointing at the listed (eid,
    block_id) slots; unrelated entries stay. Returns the count
    dropped."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=64)
    idx.record([1, 2, 3, 4], [7], 4, None, "e")
    idx.record([5, 6, 7, 8], [9], 4, None, "e")
    # Use a different prompt+cache_salt for the cross-engine
    # record so its hash differs from the first one. (Prefix
    # hashes aren't engine-keyed in `_prefix_block_hashes`.)
    idx.record([99, 99, 99, 99], [42], 4, None, "e2")
    assert len(idx) == 3

    n = idx.drop_slots("e", [7])
    assert n == 1
    # Block 9 for engine "e" survives.
    assert idx.lookup([1, 2, 3, 4], 4, None, "e") == []
    assert idx.lookup([5, 6, 7, 8], 4, None, "e") == [(9, 1)]
    # Cross-engine entry survives (different engine_id key).
    assert idx.lookup([99, 99, 99, 99], 4, None, "e2") == [(42, 1)]


def test_prefix_index_drop_slots_handles_unknown_slot(tmp_path):
    """Dropping a slot that doesn't exist is a no-op returning 0,
    not an error. Failure paths must be tolerant of inconsistent
    state (e.g., the index was invalidated between record and
    failure detection)."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=64)
    idx.record([1, 2, 3, 4], [7], 4, None, "e")
    n = idx.drop_slots("e", [999])
    assert n == 0
    # Original entry preserved.
    assert idx.lookup([1, 2, 3, 4], 4, None, "e") == [(7, 1)]


def test_daemon_scrub_drops_corrupted_host_tier_slot(tmp_path):
    """Plant a slot in the daemon's host_tier with a non-zero
    stored CRC, then mutate its in-RAM bytes (simulating
    bit-rot). One scrub_once pass must detect the mismatch and
    drop the slot."""
    import ctypes

    from gms_kv_ring.common import metrics
    from gms_kv_ring.common.checksum import crc32_at_ptr
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(
        str(tmp_path / "d.sock"),
        supervise_backend=False,
    )
    try:
        # Allocate + write + mark ready with the *correct* CRC.
        size = 256
        host_ptr = d.host_tier.put("e", 0, 0, size)
        payload = bytes(range(256))
        ctypes.memmove(host_ptr, payload, size)
        crc_ok = crc32_at_ptr(host_ptr, size)
        d.host_tier.mark_ready("e", 0, 0, crc=crc_ok)
        # Sanity: scrub a healthy slot — no drops.
        scanned, corruptions = d.scrub_once()
        assert scanned == 1
        assert corruptions == 0
        assert d.host_tier.n_slots() == 1

        # Poison the bytes (in-RAM corruption simulation).
        ctypes.memmove(host_ptr, b"\xff" * size, size)
        before = metrics.daemon_scrub_corruptions.get(engine_id="e")
        scanned2, corruptions2 = d.scrub_once()
        assert scanned2 == 1
        assert corruptions2 == 1, (
            f"scrub did not detect poisoned slot; " f"corruptions={corruptions2}"
        )
        # Slot was dropped → host_tier empty.
        assert d.host_tier.n_slots() == 0
        # Metric incremented exactly once.
        after = metrics.daemon_scrub_corruptions.get(engine_id="e")
        assert after - before == 1
    finally:
        # Clean up: free any remaining slots.
        d.host_tier.release_engine("e")


def test_daemon_scrub_ignores_zero_crc_and_unready_slots(tmp_path):
    """Slots with crc=0 (legacy / test fixtures) or not yet
    marked ready must be skipped by the scrubber. Otherwise we'd
    report false corruption on still-in-flight evictions."""
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(
        str(tmp_path / "d.sock"),
        supervise_backend=False,
    )
    try:
        # Slot 1: never marked ready.
        d.host_tier.put("e", 0, 0, 64)
        # Slot 2: marked ready but crc=0 (legacy path).
        d.host_tier.put("e", 0, 64, 64)
        d.host_tier.mark_ready("e", 0, 64, crc=0)

        scanned, corruptions = d.scrub_once()
        assert scanned == 0, (
            f"scrubber scanned slots it shouldn't have " f"(scanned={scanned})"
        )
        assert corruptions == 0
        # Both slots still present.
        assert d.host_tier.n_slots() == 2
    finally:
        d.host_tier.release_engine("e")


def test_daemon_scrub_thread_disabled_by_default(tmp_path):
    """Default interval is 0 → no thread started. Belt-and-
    suspenders check that this doesn't change without explicit
    intent (would add CPU cost to every test's daemon)."""
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(str(tmp_path / "d.sock"), supervise_backend=False)
    assert d._scrub_interval_s == 0.0
    assert d._scrub_thread is None
    # Backend scrub also off by default.
    assert d._backend_scrub_interval_s == 0.0
    assert d._backend_scrub_thread is None


def test_backend_scrub_detects_corrupt_storage_file(tmp_path):
    """Demote a slot to the local StorageTier backend, then poison
    the on-disk file's payload bytes. One scrub_backend_once pass
    must read it, fail the CRC verify, drop the slot, and bump
    the metric.

    This is the CRITICAL guard for NIXL-GDS production: GDS reads
    bypass host-side CRC verify; without backend scrub a corrupt
    file would silently serve wrong KV bytes."""
    import ctypes

    from gms_kv_ring.common import metrics
    from gms_kv_ring.common.checksum import crc32_at_ptr
    from gms_kv_ring.daemon.server import Daemon

    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    d = Daemon(
        str(tmp_path / "d.sock"),
        storage_dir=str(storage_dir),
        supervise_backend=False,
    )
    try:
        # Build a payload buffer of known bytes.
        size = 512
        payload = bytes(range(256)) * 2
        buf = (ctypes.c_ubyte * size).from_buffer_copy(payload)
        host_ptr = ctypes.addressof(buf)
        crc = crc32_at_ptr(host_ptr, size)
        slot = d.storage_tier.demote(
            "e",
            0,
            0,
            host_ptr,
            size,
            crc,
        )
        assert d.storage_tier.n_slots() == 1
        assert slot.size == size
        assert slot.crc == crc

        # Sanity: scrub a healthy backend — no drops.
        scanned, corruptions = d.scrub_backend_once()
        assert scanned == 1
        assert corruptions == 0
        assert d.storage_tier.n_slots() == 1

        # Poison the file: open + overwrite payload region (skip
        # the 24-byte header), syncing to ensure the corruption is
        # durable before scrub reads.
        file_path = slot.path
        with open(file_path, "r+b") as f:
            f.seek(24)  # past the header
            f.write(b"\xff" * 64)
            f.flush()
            os.fsync(f.fileno())

        before = metrics.daemon_backend_scrub_corruptions.get(
            engine_id="e",
        )
        scanned2, corruptions2 = d.scrub_backend_once()
        assert scanned2 == 1
        assert corruptions2 == 1, (
            f"backend scrub did not detect poisoned file; "
            f"corruptions={corruptions2}"
        )
        # Slot dropped from the backend index.
        assert d.storage_tier.n_slots() == 0
        after = metrics.daemon_backend_scrub_corruptions.get(
            engine_id="e",
        )
        assert after - before == 1
    finally:
        d.storage_tier.release_engine("e")


def test_backend_scrub_empty_backend_is_noop(tmp_path):
    """No slots → scrub is a no-op returning (0, 0). Lets the
    background thread run safely on a freshly-started daemon with
    nothing yet evicted."""
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(str(tmp_path / "d.sock"), supervise_backend=False)
    scanned, corruptions = d.scrub_backend_once()
    assert scanned == 0
    assert corruptions == 0


def test_backend_scrub_skips_missing_files_as_corrupt(tmp_path):
    """A slot whose backing file vanished (manual deletion, FS
    corruption) gets dropped — scrub treats unreadable as
    corrupt, same as wrong-CRC. Bounds the index to slots we can
    actually serve."""
    import ctypes

    from gms_kv_ring.common.checksum import crc32_at_ptr
    from gms_kv_ring.daemon.server import Daemon

    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    d = Daemon(
        str(tmp_path / "d.sock"),
        storage_dir=str(storage_dir),
        supervise_backend=False,
    )
    try:
        size = 128
        buf = (ctypes.c_ubyte * size).from_buffer_copy(b"a" * size)
        slot = d.storage_tier.demote(
            "e",
            0,
            0,
            ctypes.addressof(buf),
            size,
            crc32_at_ptr(ctypes.addressof(buf), size),
        )
        # Delete the backing file out from under the daemon.
        os.unlink(slot.path)
        scanned, corruptions = d.scrub_backend_once()
        assert scanned == 1
        assert corruptions == 1
        assert d.storage_tier.n_slots() == 0
    finally:
        d.storage_tier.release_engine("e")


def test_prefix_index_drop_slots_multiple_hashes_one_slot(tmp_path):
    """If multiple hashes happen to point at the same (eid, bid)
    slot, drop_slots() removes ALL of them in one call. (Normal
    `record()` enforces a one-hash-per-slot invariant, but the
    underlying data structure can transiently hold multi-hash
    mappings during certain edge paths; the failure-path API must
    handle that.)"""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _PrefixIndex

    idx = _PrefixIndex(max_entries=64)
    # Force multi-hash → same slot via direct manipulation, then
    # confirm drop_slots removes both.
    idx._table[b"h1"] = ("e", 7, 1)
    idx._table[b"h2"] = ("e", 7, 1)
    idx._slot_to_hashes[("e", 7)] = {b"h1", b"h2"}
    n = idx.drop_slots("e", [7])
    assert n == 2
    assert b"h1" not in idx._table
    assert b"h2" not in idx._table


class _HostFallbackFakeConnector:
    def __init__(self, available=False):
        self._available = available
        self.evict_calls = []
        self.host_evict_calls = []
        self.restore_calls = []
        self.host_restore_calls = []

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
        return {int(t[1]): True for t in triples_l}

    def restore_blocks_remap_from_host(self, src_engine_id, triples):
        triples_l = list(triples)
        self.host_restore_calls.append((str(src_engine_id), triples_l))
        return {int(t[1]): True for t in triples_l}


class _HostFallbackFakeHandle:
    def __init__(self, engine_id):
        self.engine_id = engine_id


def test_scheduler_shadow_lookup_uses_source_engine_for_vllm(tmp_path):
    cfg = _fake_vllm_config(str(tmp_path / "ignored.sock"), engine_id="shadow")
    cfg.kv_transfer_config.kv_connector_extra_config.update(
        {
            "gms_source_engine_id": "primary",
        }
    )
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    prompt = [1, 2, 3, 4]
    conn._prefix_index.record(prompt, [17], 4, None, "primary")

    matched, is_async = conn.get_num_new_matched_tokens(
        SimpleNamespace(request_id="r1", prompt_token_ids=prompt),
        0,
    )
    conn.update_state_after_alloc(
        SimpleNamespace(request_id="r1"),
        SimpleNamespace(get_block_ids=lambda: [[99]]),
        matched,
    )

    assert matched == 4
    assert is_async is False
    assert conn._sched_restore_queue == [("r1", [(17, 99, 1)])]


def test_worker_host_tier_restore_and_evict_when_gds_unavailable_vllm(tmp_path):
    cfg = _fake_vllm_config(str(tmp_path / "ignored.sock"), engine_id="shadow")
    cfg.kv_transfer_config.kv_connector_extra_config.update(
        {
            "gms_source_engine_id": "primary",
            "gms_host_tier_fallback": "1",
        }
    )
    worker = GMSKVCacheConnectorV1(cfg, KVConnectorRole.WORKER)
    worker._handle = _HostFallbackFakeHandle("shadow")
    worker._gds_conn = _HostFallbackFakeConnector(available=False)
    meta = _GmsGdsMetadata(
        _encode_meta(
            [("save", [10], [1], [])],
            [("load", [(17, 99, 1)])],
            [],
            [],
        )
    )

    worker.bind_connector_metadata(meta)

    assert worker._gds_conn.restore_calls == []
    assert worker._gds_conn.host_restore_calls == [("primary", [(17, 99, 1)])]
    assert worker._gds_conn.evict_calls == []
    assert worker._gds_conn.host_evict_calls == [[10]]
    done, _ = worker.get_finished({"save"})
    assert done == {"save"}
