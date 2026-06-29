# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P4d: tests for `register_content_addresses_batch` RPC and the
metadata-schema v6 encoding/decoding that carries content hashes
from the scheduler-side connector to the worker-side bind.

Covers:
  - Daemon batch RPC populates `_content_hash_index` and emits one
    `Stored` event per item (transport enabled).
  - Daemon batch RPC short-circuits with `skipped=True` when the
    transport is disabled (no staging_send_buffer configured).
  - `_encode_meta` / `_decode_meta` round-trip for v6 with hashes;
    backward-compat with 3-tuple inputs (no hashes → no field in
    JSON, decoder defaults to []).
  - Connector scheduler-side: GMS_KVR_CROSS_NODE controls whether
    hashes get stashed in `_sched_evict_queue`; default is OFF.

The pure-Python paths (encode/decode + daemon RPC) need NO vLLM /
CUDA and run as plain unit tests. The connector scheduler test is
gated behind `vllm + torch + CUDA` like the existing legacy file."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import os
import tempfile
import threading
import time

import pytest
from gms_kv_ring.daemon.client import DaemonClient
from gms_kv_ring.daemon.server import Daemon
from gms_kv_ring.daemon.staging_tier import _BytearrayAllocator


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


# Per-process unique port counter. NIXL listen sockets don't release
# cleanly between Daemon instances (the agent's metadata listener
# holds the port even after object teardown), so each test gets a
# fresh port to avoid "Address already in use".
_PORT_BASE = 0x9000 + (os.getpid() % 100) * 100
_port_lock = threading.Lock()
_next_port = [_PORT_BASE]


def _next_port_n() -> int:
    with _port_lock:
        p = _next_port[0]
        _next_port[0] += 1
    return p


def _spawn(
    *,
    transport_enabled: bool,
    base: str = "gms-p4d-",
):
    """Spawn a daemon; transport_enabled controls whether
    staging_send_buffer is allocated (which is the daemon's gating
    field for cross-node send-side support)."""
    tmpdir = tempfile.mkdtemp(prefix=base)
    sock = os.path.join(tmpdir, "d.sock")
    kw: dict = dict(
        listen_socket=sock,
        storage_dir=tmpdir,
        supervise_backend=False,
        staging_capacity_bytes=1 << 20,
        staging_allocator=_BytearrayAllocator(),
    )
    if transport_enabled:
        # Use an ephemeral port — these tests don't actually exercise
        # the NIXL transport, just the presence of the send-buffer
        # field that the RPC handler checks. Test uses a high random
        # port to avoid collisions.
        kw["transport_listen_port"] = _next_port_n()
        kw["transport_agent_name"] = f"p4d-{os.getpid()}-{time.time_ns()}"
        kw["staging_receive_buffer_bytes"] = 1 << 20
        kw["staging_send_buffer_bytes"] = 1 << 20
        # NIXL agent init can fail in environments without the
        # library; let those tests be skipped rather than fail at
        # daemon construction.
        try:
            d = Daemon(**kw)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"NIXL transport init failed: {exc}")
            return None
    else:
        d = Daemon(**kw)

    lh: dict = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        lh["loop"] = loop
        try:
            loop.run_until_complete(d.serve())
        finally:
            loop.close()

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not os.path.exists(sock):
        time.sleep(0.02)
    assert os.path.exists(sock), "daemon socket never appeared"
    return d, sock, th, lh


def _stop(d, th, lh):
    if "loop" in lh:
        lh["loop"].call_soon_threadsafe(d.stop)
    th.join(timeout=3)


def test_restore_ring_carries_source_staging_flag(tmp_path):
    """The restore ring preserves FLAG_SOURCE_STAGING and opaque
    staging handle ids in the existing src_block field."""
    from gms_kv_ring.common.restore_ring import (
        FLAG_SOURCE_STAGING,
        attach_reader,
        create_ring,
    )

    ring_path = str(tmp_path / "restore.ring")
    writer = create_ring(ring_path, capacity=8)
    reader = attach_reader(ring_path)
    try:
        assert writer.push(
            src_engine_id="engine",
            block_pairs=[(123, 9)],
            counter_slot=1,
            counter_target=2,
            flags=FLAG_SOURCE_STAGING,
        )
        rec = reader.try_pop()
        assert rec is not None
        assert rec["flags"] & FLAG_SOURCE_STAGING
        assert rec["block_pairs"] == [(123, 9)]
    finally:
        writer.close()
        reader.close()


def test_host_tier_respill_clears_ready_and_lru_quota(monkeypatch):
    """Reusing a slot for a new spill must hide stale bytes until mark_ready."""
    from gms_kv_ring.daemon import host_tier as host_mod

    next_ptr = [0x1000]
    freed: list[int] = []

    def fake_alloc(size):
        ptr = next_ptr[0]
        next_ptr[0] += max(1, int(size)) + 0x100
        return ptr

    monkeypatch.setattr(host_mod, "_alloc_host", fake_alloc)
    monkeypatch.setattr(host_mod, "_free_host", lambda ptr: freed.append(ptr))

    ht = host_mod.HostTier()
    ptr0 = ht.put("eng", 0, 0, 16)
    assert ht.mark_ready("eng", 0, 0, crc=123)
    assert ht.get("eng", 0, 0) is not None

    assert ht.put("eng", 0, 0, 16) == ptr0
    assert ht.get("eng", 0, 0) is None
    assert ht.mark_ready("eng", 0, 0, crc=456)

    ht.put("eng", 0, 16, 16)
    assert ht.mark_ready("eng", 0, 16, crc=789)
    evicted = ht.evict_lru_until_under(16)
    assert evicted == [("eng", 0, 0)]
    assert ht.total_bytes() == 16
    assert ptr0 in freed


def test_batch_registration_skips_unsealed_items_with_transport(monkeypatch):
    """sealed=False entries are removed instead of advertised to peers."""
    d, sock, th, lh = _spawn(transport_enabled=False)
    try:
        d.transport = object()
        h_ready = _hash(b"ready")
        h_unsealed = _hash(b"unsealed")
        with DaemonClient(sock) as client:
            total, skipped = client.register_content_addresses_batch(
                [
                    {
                        "content_hash": h_ready,
                        "engine_id": "eng",
                        "ranges": [(0, 0, 16)],
                        "generation": 7,
                    },
                    {
                        "content_hash": h_unsealed,
                        "engine_id": "eng",
                        "ranges": [(0, 16, 16)],
                        "sealed": False,
                    },
                ]
            )
        assert skipped is False
        assert total == 16
        assert d._content_hash_index[h_ready]["generation"] == 7
        assert h_unsealed not in d._content_hash_index
    finally:
        _stop(d, th, lh)


def test_get_bootstrap_info_returns_all_host_ranges(monkeypatch):
    """Host fallback descriptors must preserve every layer range."""
    from gms_kv_ring.daemon import host_tier as host_mod

    next_ptr = [0x2000]
    monkeypatch.setattr(
        host_mod,
        "_alloc_host",
        lambda size: next_ptr.__setitem__(0, next_ptr[0] + 0x1000) or next_ptr[0],
    )
    monkeypatch.setattr(host_mod, "_free_host", lambda _ptr: None)

    class FakeAgent:
        def get_agent_metadata(self):
            return b"fake-metadata"

    class FakeTransport:
        _agent = FakeAgent()

        def __init__(self):
            self.registered = []

        def agent_name(self):
            return "fake-agent"

        def listen_port(self):
            return 4444

        def register_buffer(self, ptr, size, label=""):
            self.registered.append((int(ptr), int(size), str(label)))

        def close(self):
            return None

    d, sock, th, lh = _spawn(transport_enabled=False)
    try:
        d.transport = FakeTransport()
        d.staging_send_buffer = object()
        h = _hash(b"multi-range")
        p0 = d.host_tier.put("eng", 0, 0, 16)
        p1 = d.host_tier.put("eng", 1, 32, 24)
        assert d.host_tier.mark_ready("eng", 0, 0, crc=1)
        assert d.host_tier.mark_ready("eng", 1, 32, crc=2)

        with DaemonClient(sock) as client:
            total = client.register_content_address(
                h,
                "eng",
                [(0, 0, 16), (1, 32, 24)],
                generation=11,
            )
            info = client.get_bootstrap_info([h])

        assert total == 40
        desc = info["descriptors"][0]
        assert desc["tier"] == "host"
        assert desc["size"] == 40
        assert desc["generation"] == 11
        assert desc["sealed"] is True
        assert desc["ptr"] == p0
        assert desc["ranges"] == [
            {"ptr": p0, "size": 16, "tier": "host", "layer": 0, "offset": 0},
            {"ptr": p1, "size": 24, "tier": "host", "layer": 1, "offset": 32},
        ]
        assert d.transport.registered == [
            (p0, 16, f"host:{h.hex()[:8]}:0"),
            (p1, 24, f"host:{h.hex()[:8]}:1"),
        ]
    finally:
        _stop(d, th, lh)


# ----------------------------------------------------------------------
# Daemon-side: batch RPC behavior
# ----------------------------------------------------------------------


def test_batch_rpc_skipped_when_transport_disabled():
    """Without staging_send_buffer the daemon has no cross-node
    transport — the RPC must short-circuit with skipped=True so the
    connector can self-disable future calls."""
    d, sock, th, lh = _spawn(transport_enabled=False)
    try:
        with DaemonClient(sock) as client:
            total, skipped = client.register_content_addresses_batch(
                [
                    {
                        "content_hash": _hash(b"a"),
                        "engine_id": "eng",
                        "ranges": [(0, 0, 16)],
                    },
                ]
            )
            assert skipped is True
            assert total == 0
            # Index must remain empty
            assert len(d._content_hash_index) == 0
    finally:
        _stop(d, th, lh)


def test_batch_rpc_populates_index_when_transport_enabled():
    """Happy path: each item lands in _content_hash_index and the
    publisher emits one Stored event per item."""
    from gms_kv_ring.daemon.placement_publisher import LoggingPlacementPublisher

    out = _spawn(transport_enabled=False)
    if out is None:
        return
    d, sock, th, lh = out
    try:
        d.transport = object()
        d.placement_publisher = LoggingPlacementPublisher(
            daemon_id="test-daemon",
            daemon_epoch=0,
        )
        items = [
            {
                "content_hash": _hash(b"alpha"),
                "engine_id": "eng-1",
                "ranges": [(0, 0, 16), (1, 0, 16)],
            },
            {
                "content_hash": _hash(b"beta"),
                "engine_id": "eng-1",
                "ranges": [(0, 16, 16), (1, 16, 16)],
            },
        ]
        with DaemonClient(sock) as client:
            total, skipped = client.register_content_addresses_batch(items)
            assert skipped is False
            # 2 items × 2 layers × 16 bytes = 64
            assert total == 64
        # Index has both hashes
        assert _hash(b"alpha") in d._content_hash_index
        assert _hash(b"beta") in d._content_hash_index
        assert d._content_hash_index[_hash(b"alpha")] == {
            "engine_id": "eng-1",
            "ranges": [(0, 0, 16), (1, 0, 16)],
        }
        # Publisher saw two Stored events
        stats = d.placement_publisher.stats()
        assert stats["stored"] == 2
    finally:
        _stop(d, th, lh)


def test_record_restore_staging_uses_destination_block_order():
    """GMSKvRing.record_restore_staging maps metadata entries
    (hash, dst, generation) into ring pairs (handle, dst)."""
    from gms_kv_ring.common.restore_ring import FLAG_SOURCE_STAGING
    from gms_kv_ring.engines.handle import GMSKvRing

    h1 = _hash(b"staged-order")

    class FakeClient:
        def __init__(self):
            self.items = None
            self.released = []

        def register_staging_restore_handles(self, items):
            self.items = items
            return [101]

        def release_staging_restore_handles(self, handles):
            self.released.extend(handles)
            return len(handles)

    class FakeCounters:
        def try_reserve_completed_slot(self):
            return (7, 8)

    class FakeRestoreWriter:
        def __init__(self):
            self.record = None

        def push(self, **kwargs):
            self.record = kwargs
            return True

    handle = object.__new__(GMSKvRing)
    handle.engine_id = "engine"
    handle._client = FakeClient()
    handle.counters = FakeCounters()
    handle.restore_writer = FakeRestoreWriter()
    handle._restore_lock = threading.Lock()

    assert handle.record_restore_staging([(h1, 30, 3)]) == (7, 8)
    assert handle._client.items == [{"content_hash": h1, "generation": 3}]
    assert handle.restore_writer.record["src_engine_id"] == "engine"
    assert handle.restore_writer.record["block_pairs"] == [(101, 30)]
    assert handle.restore_writer.record["flags"] == FLAG_SOURCE_STAGING
    assert handle._client.released == []


def test_read_bootstrap_into_staging_client_payload_shape():
    """DaemonClient serializes router placement descriptors for the
    destination-daemon READ path."""
    h1 = _hash(b"router-read-client")
    client = object.__new__(DaemonClient)
    seen = {}

    def fake_ok(body):
        seen.update(body)
        return {
            "ok": True,
            "accepted": 1,
            "already_ready": 0,
            "coalesced": 0,
            "failed": 0,
            "skipped": 0,
            "bytes_read": 16,
        }

    client._ok = fake_ok
    result = client.read_bootstrap_into_staging(
        "source-agent",
        "010203",
        [h1],
        [
            {
                "remote_ptr": 0xCAFE,
                "size": 16,
                "tier": "host",
                "ranges": [{"remote_ptr": 0xCAFE, "size": 16, "tier": "host"}],
            }
        ],
        timeout_s=4.0,
        batch_size=8,
    )

    assert result["accepted"] == 1
    assert result["bytes_read"] == 16
    assert seen == {
        "op": "read_bootstrap_into_staging",
        "source_nixl_name": "source-agent",
        "source_agent_metadata_hex": "010203",
        "hashes": [h1.hex()],
        "descriptors": [
            {
                "remote_ptr": 0xCAFE,
                "size": 16,
                "tier": "host",
                "ranges": [{"remote_ptr": 0xCAFE, "size": 16, "tier": "host"}],
            }
        ],
        "timeout_s": 4.0,
        "batch_size": 8,
    }


def test_read_bootstrap_into_staging_commits_vectored_read():
    """Destination daemon READs router descriptors into staging so all
    inference engines can consume them through the existing staging path."""
    from gms_kv_ring.daemon.staging_receive_buffer import StagingReceiveBuffer

    part0 = b"left".ljust(16, b"L")
    part1 = b"right".ljust(24, b"R")
    payload = part0 + part1
    # Production router keys are prefix hashes, not hashes of the KV bytes.
    content_hash = b"\xab" * 32

    src0 = (ctypes.c_ubyte * len(part0))()
    src1 = (ctypes.c_ubyte * len(part1))()
    src0_ptr = ctypes.addressof(src0)
    src1_ptr = ctypes.addressof(src1)
    ctypes.memmove(src0_ptr, part0, len(part0))
    ctypes.memmove(src1_ptr, part1, len(part1))

    class FakeTransport:
        def __init__(self):
            self.peers = []
            self.reads = []

        def add_peer_from_metadata(self, nixl_name, metadata):
            self.peers.append((nixl_name, metadata))

        def read_batch(self, source_nixl_name, items, timeout_s=30.0):
            self.reads.append((source_nixl_name, list(items), timeout_s))
            for local_ptr, size, remote_ptr in items:
                ctypes.memmove(int(local_ptr), int(remote_ptr), int(size))

    d = Daemon(
        listen_socket="/tmp/gms-read-bootstrap-test.sock",
        storage_dir=tempfile.mkdtemp(prefix="gms-read-bootstrap-"),
        supervise_backend=False,
        staging_capacity_bytes=1 << 20,
        staging_allocator=_BytearrayAllocator(),
    )
    d.transport = FakeTransport()
    d.staging_receive_buffer = StagingReceiveBuffer(1 << 20)

    resp = d._dispatch(
        {
            "op": "read_bootstrap_into_staging",
            "source_nixl_name": "source-agent",
            "source_agent_metadata_hex": "010203",
            "hashes": [content_hash.hex()],
            "descriptors": [
                {
                    "remote_ptr": src0_ptr,
                    "size": len(payload),
                    "tier": "host",
                    "ranges": [
                        {"remote_ptr": src0_ptr, "size": len(part0), "tier": "host"},
                        {"ptr": src1_ptr, "size": len(part1), "tier": "host"},
                    ],
                }
            ],
            "timeout_s": 5.0,
        }
    )

    assert resp["ok"] is True
    assert resp["accepted"] == 1
    assert resp["failed"] == 0
    assert resp["bytes_read"] == len(payload)
    assert d.transport.peers == [("source-agent", b"\x01\x02\x03")]
    assert len(d.transport.reads) == 1

    hit = d.staging_tier.scan([content_hash])[content_hash]
    assert hit.bytes_size == len(payload)
    assert d.staging_tier._alloc.read(hit.bytes_ptr, hit.bytes_size) == payload


def test_restore_staging_ranges_client_payload_shape():
    """DaemonClient.restore_staging_ranges serializes explicit
    destination ranges without block-id assumptions."""
    h1 = _hash(b"staged-ranges-client")
    client = object.__new__(DaemonClient)
    seen = {}

    def fake_ok(body):
        seen.update(body)
        return {"ok": True, "success": True}

    client._ok = fake_ok
    ok = client.restore_staging_ranges(
        "engine",
        [(h1, 4, [(0, 64, 16), (1, 128, 16)])],
    )

    assert ok is True
    assert seen == {
        "op": "restore_staging_ranges",
        "engine_id": "engine",
        "items": [
            {
                "content_hash": h1.hex(),
                "generation": 4,
                "ranges": [
                    {"layer": 0, "offset": 64, "size": 16},
                    {"layer": 1, "offset": 128, "size": 16},
                ],
            },
        ],
    }


def test_restore_staging_ranges_sync_delegates_to_client():
    """GMSKvRing exposes the range restore RPC to engine adapters."""
    from gms_kv_ring.engines.handle import GMSKvRing

    h1 = _hash(b"staged-ranges-handle")

    class FakeClient:
        def __init__(self):
            self.calls = []

        def restore_staging_ranges(self, engine_id, hits):
            self.calls.append((engine_id, hits))
            return True

    handle = object.__new__(GMSKvRing)
    handle.engine_id = "engine"
    handle._client = FakeClient()

    hits = [(h1, 5, [(0, 0, 32)])]
    assert handle.restore_staging_ranges_sync(hits) is True
    assert handle._client.calls == [("engine", hits)]


def test_staging_restore_handle_register_and_release():
    """Daemon allocates one-shot u32 handles for staging-source
    restore records and can release unused handles."""
    d, sock, th, lh = _spawn(transport_enabled=False)
    try:
        h1 = _hash(b"restore-handle-0")
        h2 = _hash(b"restore-handle-1")
        client = DaemonClient(sock)
        try:
            handles = client.register_staging_restore_handles(
                [
                    {"content_hash": h1, "generation": 1},
                    {"content_hash": h2, "generation": 2},
                ]
            )
            assert len(handles) == 2
            assert all(isinstance(h, int) and h > 0 for h in handles)
            assert handles[0] != handles[1]
            released = client.release_staging_restore_handles(handles)
            assert released == 2
        finally:
            client.close()
    finally:
        _stop(d, th, lh)


# ----------------------------------------------------------------------
# Schema v6: encode/decode round-trip with hashes
# ----------------------------------------------------------------------


def test_v6_round_trip_with_hashes():
    """When evict tuples carry hashes, _encode_meta writes the
    `hashes_per_evict` field; _decode_meta restores them."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import (
        _decode_meta,
        _encode_meta,
    )

    h1 = _hash(b"block-0")
    h2 = _hash(b"block-1")
    payload = _encode_meta(
        [("req-A", [10, 11], [3, 4], [h1, h2])],
    )
    evict, _ra, _rs, _rstg = _decode_meta(payload)
    assert evict == [("req-A", [10, 11], [3, 4], [h1, h2])]


def test_v6_round_trip_without_hashes_omits_field():
    """If no entry has hashes, the encoder omits the optional field
    entirely (keeps default-case payload identical to v4)."""
    import json

    from gpu_memory_service.integrations.vllm.gds_connector_v1 import (
        _decode_meta,
        _encode_meta,
    )

    # 3-tuple input — encoder must still produce v6 output and
    # decode-back into 4-tuples with empty hashes.
    payload = _encode_meta([("req-B", [1, 2], [0, 0])])
    obj = json.loads(payload.decode("utf-8"))
    assert obj["v"] == 6
    assert (
        "hashes_per_evict" not in obj
    ), "encoder should not emit hashes_per_evict when all empty"
    evict, _ra, _rs, _rstg = _decode_meta(payload)
    assert evict == [("req-B", [1, 2], [0, 0], [])]


def test_v6_decoder_accepts_v4_payload():
    """A v4 payload (no hashes_per_evict field) decodes into v6
    shape with empty hashes lists — rolling-upgrade scenario."""
    import json

    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _decode_meta

    v4 = json.dumps(
        {
            "v": 4,
            "evict": [("req-Z", [5, 6], [1, 2])],
            "restore_async": [],
            "restore_sync": [],
        }
    ).encode("utf-8")
    evict, _ra, _rs, _rstg = _decode_meta(v4)
    assert evict == [("req-Z", [5, 6], [1, 2], [])]


def test_v6_round_trip_with_staging_restore():
    """Staging restore records carry content hashes and generations
    separately from local src_block restore triples."""
    from gpu_memory_service.integrations.vllm.gds_connector_v1 import (
        _decode_meta,
        _encode_meta,
    )

    h1 = _hash(b"staged-0")
    h2 = _hash(b"staged-1")
    payload = _encode_meta(
        [],
        restore_staging_async=[
            ("req-S", [(h1, 30, 1), (h2, 31, 2)]),
        ],
    )
    evict, local_async, local_sync, staging = _decode_meta(payload)
    assert evict == []
    assert local_async == []
    assert local_sync == []
    assert staging == [("req-S", [(h1, 30, 1), (h2, 31, 2)])]


def test_v6_decoder_tolerates_malformed_hash_hex():
    """A malformed hex string in hashes_per_evict yields empty bytes
    rather than crashing the bind path."""
    import json

    from gpu_memory_service.integrations.vllm.gds_connector_v1 import _decode_meta

    payload = json.dumps(
        {
            "v": 5,
            "evict": [("req-B", [1], [0])],
            "hashes_per_evict": [["NOT_HEX"]],
            "restore_async": [],
            "restore_sync": [],
        }
    ).encode("utf-8")
    evict, _ra, _rs, _rstg = _decode_meta(payload)
    assert evict[0][3] == [b""]


# ----------------------------------------------------------------------
# Connector scheduler-side: env-var-controlled hash stashing
# ----------------------------------------------------------------------

# These need vLLM + torch + CUDA because the connector imports vLLM
# at module load. Skipped on dev machines without the full stack.
vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _make_cfg(sock: str, block_size_tokens: int = 4):
    from types import SimpleNamespace

    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="test-eid",
            kv_connector_extra_config={"gms_daemon_socket": sock},
        ),
        cache_config=SimpleNamespace(block_size=block_size_tokens),
    )


def test_scheduler_skips_hashes_when_cross_node_disabled(
    monkeypatch,
    tmp_path,
):
    """Default behavior: GMS_KVR_CROSS_NODE unset → evict tuples have
    empty hashes, and the encoded payload omits `hashes_per_evict`."""
    monkeypatch.delenv("GMS_KVR_CROSS_NODE", raising=False)
    from types import SimpleNamespace

    from gpu_memory_service.integrations.vllm.gds_connector_v1 import (
        GMSKVCacheConnectorV1,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    cfg = _make_cfg(str(tmp_path / "irrelevant.sock"))
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    assert conn._cross_node_register is False
    # 8 tokens = 2 full prefix blocks at block_size=4
    req = SimpleNamespace(
        request_id="r0",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        cache_salt=None,
    )
    conn.request_finished(req, [10, 11])
    # 4th element is empty
    assert conn._sched_evict_queue[0][3] == []


def test_scheduler_stashes_hashes_when_cross_node_enabled(
    monkeypatch,
    tmp_path,
):
    """With GMS_KVR_CROSS_NODE=1, per-block hashes flow through into
    the evict tuple AND into the encoded metadata payload."""
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    from types import SimpleNamespace

    from gpu_memory_service.integrations.vllm.gds_connector_v1 import (
        GMSKVCacheConnectorV1,
        _decode_meta,
        _prefix_block_hashes,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    cfg = _make_cfg(str(tmp_path / "irrelevant.sock"))
    conn = GMSKVCacheConnectorV1(cfg, KVConnectorRole.SCHEDULER)
    assert conn._cross_node_register is True
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    req = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=prompt,
        cache_salt=None,
    )
    # 3rd block_id beyond the indexed prefix gets empty hash padding.
    conn.request_finished(req, [10, 11, 12])
    expected = _prefix_block_hashes(prompt, 4, None)
    hashes = conn._sched_evict_queue[0][3]
    assert len(hashes) == 3
    assert hashes[0] == expected[0]
    assert hashes[1] == expected[1]
    assert hashes[2] == b""  # block 12 is past the full-prefix range

    # Round-trip through the metadata payload.
    meta = conn.build_connector_meta(SimpleNamespace())
    evict, _ra, _rs, _rstg = _decode_meta(meta.payload)
    assert evict[0][3] == [expected[0], expected[1], b""]
