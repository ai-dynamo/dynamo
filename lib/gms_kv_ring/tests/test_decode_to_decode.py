# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode-to-decode KV transfer — a GMS-unique feature.

Background
==========

In Dynamo today, cross-worker KV transfer is **prefill-to-decode only**: a
prefill worker computes KV for a request's prompt, then hands the KV off
to a decode worker that runs autoregressive generation. Once a decode
worker has its KV in HBM, it stays private to that worker for the rest of
the request's lifetime. There is no way for one decode worker to source
KV from another running decode worker.

GMS removes that restriction. Because the daemon owns the KV memory
(via VMM-IPC and/or host_tier) and exposes it over NIXL, any GMS daemon
can serve any other GMS daemon. The engine role attached to a daemon
(prefill vs decode) doesn't matter for the wire transfer. This means:

* A new request landing on decode worker B can sip KV from decode worker
  A's local cache for any overlapping prefix.
* Bidirectional concurrent decode↔decode is supported by construction.
* Conversational workloads with shared system prompts see near-zero
  prefill cost on prefix hits across running workers.

These tests exercise the decode-to-decode shape on both the
engine-direct (default) and the `fetch_remote` (fallback) paths.
Engine roles are simulated — we don't need real inference engines to
prove the wire pattern is symmetric.

Gated behind ``GMS_KVR_TEST_NIXL_TRANSPORT=1`` + a working UCX runtime.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("nixl")
if not os.environ.get("GMS_KVR_TEST_NIXL_TRANSPORT"):
    pytest.skip(
        "set GMS_KVR_TEST_NIXL_TRANSPORT=1 + working UCX to run",
        allow_module_level=True,
    )

from gms_kv_ring.daemon.client import DaemonClient  # noqa: E402
from gms_kv_ring.tests._p5_helpers import inject_block as _inject_block  # noqa: E402
from gms_kv_ring.tests._p5_helpers import (  # noqa: E402
    next_port_pair as _next_port_pair,
)
from gms_kv_ring.tests._p5_helpers import (  # noqa: E402
    spawn_daemon as _spawn_daemon_full,
)
from gms_kv_ring.tests._p5_helpers import stop_daemon as _stop_daemon  # noqa: E402
from gms_kv_ring.tests._p5_helpers import (  # noqa: E402
    wait_for_arrival as _wait_for_arrival,
)


def _spawn_daemon(uds, port, agent_name, *, cap=64 << 20, buf=32 << 20):
    return _spawn_daemon_full(
        uds,
        port,
        agent_name,
        cap=cap,
        send_buf=buf,
        recv_buf=buf,
        storage_dir_prefix="gms-d2d-",
    )


# ======================================================================
# Engine-direct decode-to-decode
# ======================================================================


def test_decode_to_decode_engine_direct():
    """A decode worker reads KV blocks directly from a running decode
    peer's daemon via NIXL READ. Verifies the engine-direct path
    (default) supports decode-worker → decode-worker transfer.

    Scenario:
        * Decode-A is decoding request X. As it produces new KV blocks,
          it spills them to daemon-A and registers bootstrap handles.
        * Decode-B receives a new request Y whose prompt overlaps with
          X's recent context. The router (test code, simulating dynamo
          PrefillRouter behavior) attaches a SharedPlacementInfo
          pointing at decode-A as the source.
        * Decode-B's connector (test code) NIXL-reads the overlap
          blocks from daemon-A's registered memory directly.

    Wire pattern is identical to a prefill→decode handoff — the
    daemon doesn't care which engine role is upstream.
    """
    from nixl._api import nixl_agent, nixl_agent_config

    n_blocks = 32
    block_size = 16 * 1024
    p_a, p_b = _next_port_pair()
    p_b_engine = p_b + 100
    uds_a = tempfile.mkdtemp(prefix="gms-d2d-decA-") + "/d.sock"
    uds_b = tempfile.mkdtemp(prefix="gms-d2d-decB-") + "/d.sock"
    name_a = f"decode-a-{p_a}"
    name_b = f"decode-b-{p_b}"
    name_b_engine = f"engine-on-decode-b-{p_b_engine}"
    da, ta, ha = _spawn_daemon(uds_a, p_a, name_a)
    db, tb, hb = _spawn_daemon(uds_b, p_b, name_b)
    try:
        if da.transport is None or db.transport is None:
            pytest.skip("NIXL UCX transport unavailable")

        # Decode-A (the source) produces KV blocks and registers them
        # with its daemon's NIXL agent. In production this happens
        # incrementally as the engine emits each layer's KV.
        layer_bytes = n_blocks * block_size
        # Daemon-A allocates a registered buffer; we fill it with
        # distinct per-block content.
        src_buf = (ctypes.c_ubyte * layer_bytes)()
        src_ptr = ctypes.addressof(src_buf)
        bootstrap_items = []
        layer_hashes = []
        payloads = []
        for i in range(n_blocks):
            seed = f"A-block-{i}".encode()
            payload = (seed.ljust(64, b"x") * (block_size // 64 + 1))[:block_size]
            ctypes.memmove(src_ptr + i * block_size, payload, block_size)
            content_hash = hashlib.sha256(payload).digest()
            bootstrap_items.append(
                {
                    "content_hash": content_hash,
                    "ptr": src_ptr + i * block_size,
                    "size": block_size,
                }
            )
            layer_hashes.append(content_hash)
            payloads.append(payload)
        with DaemonClient(uds_a) as sc:
            n_registered = sc.register_bootstrap_handle(bootstrap_items)
        assert n_registered == n_blocks

        # Decode-B (the destination) needs the bytes. Engine on B
        # creates its own NIXL agent (in production this is the
        # engine's own NIXL agent, owned by its connector).
        cfg_b_engine = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=True,
            listen_port=p_b_engine,
            backends=["UCX"],
        )
        engine_b = nixl_agent(name_b_engine, cfg_b_engine)
        dst_buf = (ctypes.c_ubyte * layer_bytes)()
        dst_ptr = ctypes.addressof(dst_buf)
        engine_b.register_memory(
            [(dst_ptr, layer_bytes, 0, "decode-b-engine-dst")],
            mem_type="DRAM",
        )

        # Production flow: router attaches SharedPlacementInfo to
        # request Y. We simulate that lookup directly.
        with DaemonClient(uds_a) as dc_src:
            info = dc_src.get_bootstrap_info(layer_hashes)
        assert info["nixl_agent_name"] == name_a
        src_agent_md = bytes.fromhex(info["agent_metadata_b64"])
        engine_b.add_remote_agent(src_agent_md)

        # Decode-B's connector NIXL-reads each block from daemon-A's
        # registered memory. Concurrent reads (one per block in
        # production this would be one per layer).
        handles = []
        for i in range(n_blocks):
            desc = info["descriptors"][i]
            assert desc is not None and desc["tier"] == "hbm"
            local = engine_b.get_xfer_descs(
                [(dst_ptr + i * block_size, block_size, 0)],
                mem_type="DRAM",
            )
            remote = engine_b.get_xfer_descs(
                [(int(desc["ptr"]), int(desc["size"]), 0)],
                mem_type="DRAM",
            )
            h = engine_b.initialize_xfer(
                "READ",
                local,
                remote,
                name_a,
                notif_msg=f"d2d-{i}".encode(),
            )
            engine_b.transfer(h)
            handles.append(h)
        deadline = time.monotonic() + 30.0
        pending = list(handles)
        while pending:
            assert time.monotonic() < deadline, "engine-direct READ timed out"
            next_pending = []
            for h in pending:
                st = engine_b.check_xfer_state(h)
                if st in ("DONE", "ERR"):
                    if st == "ERR":
                        raise RuntimeError("engine-direct READ ERR")
                    engine_b.release_xfer_handle(h)
                else:
                    next_pending.append(h)
            pending = next_pending
            if pending:
                time.sleep(0.0005)

        # Verify byte-identical arrival.
        for i, expected in enumerate(payloads):
            actual = bytes(
                (ctypes.c_ubyte * block_size).from_address(
                    dst_ptr + i * block_size,
                )
            )
            assert actual == expected, f"block {i} byte mismatch"

        # Decode-B's connector notifies its local daemon
        # (for the indexer / Stored event).
        with DaemonClient(uds_b) as dc_local:
            published = dc_local.notify_kv_arrived(
                [{"content_hash": h, "size": block_size} for h in layer_hashes]
            )
        # publish_stored may be 0 if placement_publisher isn't enabled
        # in this test config; just assert no error.
        assert published >= 0
    finally:
        _stop_daemon(da, ta, ha)
        _stop_daemon(db, tb, hb)


def test_decode_to_decode_engine_direct_bidirectional():
    """Both decode workers act as source and destination
    simultaneously — A pulls from B while B pulls from A. Proves the
    wire path is symmetric: no source/destination role is baked in."""
    from nixl._api import nixl_agent, nixl_agent_config

    n_blocks = 16
    block_size = 16 * 1024
    p_a, p_b = _next_port_pair()
    p_a_engine = p_a + 200
    p_b_engine = p_b + 200
    uds_a = tempfile.mkdtemp(prefix="gms-d2d-bidi-A-") + "/d.sock"
    uds_b = tempfile.mkdtemp(prefix="gms-d2d-bidi-B-") + "/d.sock"
    name_a = f"decode-a-{p_a}"
    name_b = f"decode-b-{p_b}"
    name_a_engine = f"engine-on-decode-a-{p_a_engine}"
    name_b_engine = f"engine-on-decode-b-{p_b_engine}"
    da, ta, ha = _spawn_daemon(uds_a, p_a, name_a)
    db, tb, hb = _spawn_daemon(uds_b, p_b, name_b)
    try:
        if da.transport is None or db.transport is None:
            pytest.skip("NIXL UCX transport unavailable")

        # Each daemon registers its own pool of "decoded KV" blocks.
        layer_bytes = n_blocks * block_size
        # Daemon-A's blocks (decoded by engine A)
        a_src_buf = (ctypes.c_ubyte * layer_bytes)()
        a_src_ptr = ctypes.addressof(a_src_buf)
        a_items = []
        a_hashes = []
        a_payloads = []
        for i in range(n_blocks):
            payload = (f"A-{i}".encode().ljust(64, b"a") * (block_size // 64 + 1))[
                :block_size
            ]
            ctypes.memmove(a_src_ptr + i * block_size, payload, block_size)
            h = hashlib.sha256(payload).digest()
            a_items.append(
                {
                    "content_hash": h,
                    "ptr": a_src_ptr + i * block_size,
                    "size": block_size,
                }
            )
            a_hashes.append(h)
            a_payloads.append(payload)
        # Daemon-B's blocks (decoded by engine B)
        b_src_buf = (ctypes.c_ubyte * layer_bytes)()
        b_src_ptr = ctypes.addressof(b_src_buf)
        b_items = []
        b_hashes = []
        b_payloads = []
        for i in range(n_blocks):
            payload = (f"B-{i}".encode().ljust(64, b"b") * (block_size // 64 + 1))[
                :block_size
            ]
            ctypes.memmove(b_src_ptr + i * block_size, payload, block_size)
            h = hashlib.sha256(payload).digest()
            b_items.append(
                {
                    "content_hash": h,
                    "ptr": b_src_ptr + i * block_size,
                    "size": block_size,
                }
            )
            b_hashes.append(h)
            b_payloads.append(payload)

        with DaemonClient(uds_a) as ca:
            ca.register_bootstrap_handle(a_items)
        with DaemonClient(uds_b) as cb:
            cb.register_bootstrap_handle(b_items)

        # Each engine creates its own NIXL agent + destination buffer
        # for the blocks coming FROM the peer.
        cfg_a = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=True,
            listen_port=p_a_engine,
            backends=["UCX"],
        )
        cfg_b = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=True,
            listen_port=p_b_engine,
            backends=["UCX"],
        )
        engine_a = nixl_agent(name_a_engine, cfg_a)
        engine_b = nixl_agent(name_b_engine, cfg_b)
        a_dst_buf = (ctypes.c_ubyte * layer_bytes)()  # holds bytes from B
        b_dst_buf = (ctypes.c_ubyte * layer_bytes)()  # holds bytes from A
        a_dst_ptr = ctypes.addressof(a_dst_buf)
        b_dst_ptr = ctypes.addressof(b_dst_buf)
        engine_a.register_memory(
            [(a_dst_ptr, layer_bytes, 0, "engine-a-dst")],
            mem_type="DRAM",
        )
        engine_b.register_memory(
            [(b_dst_ptr, layer_bytes, 0, "engine-b-dst")],
            mem_type="DRAM",
        )

        # Each engine adds the OTHER daemon as remote peer.
        with DaemonClient(uds_a) as ca:
            info_a = ca.get_bootstrap_info(a_hashes)
        with DaemonClient(uds_b) as cb:
            info_b = cb.get_bootstrap_info(b_hashes)
        engine_b.add_remote_agent(bytes.fromhex(info_a["agent_metadata_b64"]))
        engine_a.add_remote_agent(bytes.fromhex(info_b["agent_metadata_b64"]))

        # Build per-direction xfers.
        def _build_reads(engine, dst_ptr_base, info, src_agent_name, tag):
            handles = []
            for i in range(n_blocks):
                desc = info["descriptors"][i]
                local = engine.get_xfer_descs(
                    [(dst_ptr_base + i * block_size, block_size, 0)],
                    mem_type="DRAM",
                )
                remote = engine.get_xfer_descs(
                    [(int(desc["ptr"]), int(desc["size"]), 0)],
                    mem_type="DRAM",
                )
                h = engine.initialize_xfer(
                    "READ",
                    local,
                    remote,
                    src_agent_name,
                    notif_msg=f"{tag}-{i}".encode(),
                )
                engine.transfer(h)
                handles.append(h)
            return handles

        # Fire both directions concurrently — bidirectional decode↔decode.
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_b_reads_a = ex.submit(
                _build_reads,
                engine_b,
                b_dst_ptr,
                info_a,
                name_a,
                "ba",
            )
            f_a_reads_b = ex.submit(
                _build_reads,
                engine_a,
                a_dst_ptr,
                info_b,
                name_b,
                "ab",
            )
            handles_b = f_b_reads_a.result()
            handles_a = f_a_reads_b.result()

        # Wait both directions DONE.
        def _drain(engine, handles, label):
            deadline = time.monotonic() + 30.0
            pending = list(handles)
            while pending:
                assert time.monotonic() < deadline, f"{label}: READ timed out"
                next_pending = []
                for h in pending:
                    st = engine.check_xfer_state(h)
                    if st in ("DONE", "ERR"):
                        if st == "ERR":
                            raise RuntimeError(f"{label}: READ ERR")
                        engine.release_xfer_handle(h)
                    else:
                        next_pending.append(h)
                pending = next_pending
                if pending:
                    time.sleep(0.0005)

        with ThreadPoolExecutor(max_workers=2) as ex:
            ex.submit(_drain, engine_a, handles_a, "a←b").result()
            ex.submit(_drain, engine_b, handles_b, "b←a").result()

        # Verify both directions byte-identical.
        for i, expected in enumerate(a_payloads):
            actual = bytes(
                (ctypes.c_ubyte * block_size).from_address(b_dst_ptr + i * block_size),
            )
            assert actual == expected, f"a→b block {i} byte mismatch"
        for i, expected in enumerate(b_payloads):
            actual = bytes(
                (ctypes.c_ubyte * block_size).from_address(a_dst_ptr + i * block_size),
            )
            assert actual == expected, f"b→a block {i} byte mismatch"
    finally:
        _stop_daemon(da, ta, ha)
        _stop_daemon(db, tb, hb)


# ======================================================================
# Pattern C decode-to-decode (fallback path)
# ======================================================================


def test_decode_to_decode_pattern_c_mutual_exchange():
    """Same decode↔decode shape via the fetch_remote (Pattern C)
    fallback path. Use case: source's KV is in host_tier (evicted from
    HBM) or NIXL backend can't read it directly — daemon orchestrates
    the transfer.

    Verifies the fallback path is also symmetric between decode
    workers."""
    n_blocks = 8
    block_size = 16 * 1024
    p_a, p_b = _next_port_pair()
    uds_a = tempfile.mkdtemp(prefix="gms-d2d-pcA-") + "/d.sock"
    uds_b = tempfile.mkdtemp(prefix="gms-d2d-pcB-") + "/d.sock"
    name_a = f"decode-a-{p_a}"
    name_b = f"decode-b-{p_b}"
    da, ta, ha = _spawn_daemon(uds_a, p_a, name_a)
    db, tb, hb = _spawn_daemon(uds_b, p_b, name_b)
    try:
        if da.transport is None or db.transport is None:
            pytest.skip("NIXL UCX transport unavailable")
        da.transport.add_peer_inproc(db.transport)
        db.transport.add_peer_inproc(da.transport)

        # Each daemon hosts a distinct pool of decode-produced KV.
        a_hashes = []
        a_payloads = []
        with DaemonClient(uds_a) as ca:
            for i in range(n_blocks):
                payload = (f"A-{i}".encode().ljust(64, b"a") * (block_size // 64 + 1))[
                    :block_size
                ]
                content_hash = hashlib.sha256(payload).digest()
                _inject_block(
                    da, "decode-A", layer=0, offset=i * block_size, data=payload
                )
                ca.register_content_address(
                    content_hash=content_hash,
                    engine_id="decode-A",
                    ranges=[(0, i * block_size, block_size)],
                )
                a_hashes.append(content_hash.hex())
                a_payloads.append(payload)

        b_hashes = []
        b_payloads = []
        with DaemonClient(uds_b) as cb:
            for i in range(n_blocks):
                payload = (f"B-{i}".encode().ljust(64, b"b") * (block_size // 64 + 1))[
                    :block_size
                ]
                content_hash = hashlib.sha256(payload).digest()
                _inject_block(
                    db, "decode-B", layer=0, offset=i * block_size, data=payload
                )
                cb.register_content_address(
                    content_hash=content_hash,
                    engine_id="decode-B",
                    ranges=[(0, i * block_size, block_size)],
                )
                b_hashes.append(content_hash.hex())
                b_payloads.append(payload)

        # Fire both fetch_remote calls in parallel.
        def _fetch_into_b():
            with DaemonClient(uds_b) as cb:
                return cb.fetch_remote(
                    source_uds_path=uds_a,
                    source_nixl_name=name_a,
                    source_ip="127.0.0.1",
                    source_port=p_a,
                    hashes=[bytes.fromhex(h) for h in a_hashes],
                    bytes_per_hash=block_size,
                    timeout_s=30.0,
                    batch_size=n_blocks,
                )

        def _fetch_into_a():
            with DaemonClient(uds_a) as ca:
                return ca.fetch_remote(
                    source_uds_path=uds_b,
                    source_nixl_name=name_b,
                    source_ip="127.0.0.1",
                    source_port=p_b,
                    hashes=[bytes.fromhex(h) for h in b_hashes],
                    bytes_per_hash=block_size,
                    timeout_s=30.0,
                    batch_size=n_blocks,
                )

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_b = ex.submit(_fetch_into_b)
            f_a = ex.submit(_fetch_into_a)
            r_b = f_b.result()
            r_a = f_a.result()
        assert r_b["failed"] == 0, r_b
        assert r_a["failed"] == 0, r_a

        assert _wait_for_arrival(db, a_hashes, sla=15.0) == n_blocks
        assert _wait_for_arrival(da, b_hashes, sla=15.0) == n_blocks

        # Byte-identical both directions.
        for hex_hash, expected in zip(a_hashes, a_payloads):
            key = bytes.fromhex(hex_hash)
            slot = db.staging_tier._slots[key]
            actual = db.staging_tier._alloc.read(slot.bytes_ptr, slot.bytes_size)
            assert actual == expected, f"a→b: {hex_hash[:16]} mismatch"
        for hex_hash, expected in zip(b_hashes, b_payloads):
            key = bytes.fromhex(hex_hash)
            slot = da.staging_tier._slots[key]
            actual = da.staging_tier._alloc.read(slot.bytes_ptr, slot.bytes_size)
            assert actual == expected, f"b→a: {hex_hash[:16]} mismatch"
    finally:
        _stop_daemon(da, ta, ha)
        _stop_daemon(db, tb, hb)


def test_decode_to_decode_role_symmetry_invariants():
    """Documentation-as-code: assert the architectural invariants
    that make decode-to-decode possible. These are properties of the
    GMS daemon API, not behavior tests — they fail loudly if a future
    refactor accidentally bakes in a prefill/decode role assumption."""

    # 1. register_bootstrap_handle takes no engine-role parameter.
    import inspect

    from gms_kv_ring.daemon.client import DaemonClient

    sig = inspect.signature(DaemonClient.register_bootstrap_handle)
    assert "role" not in sig.parameters
    assert "is_prefill" not in sig.parameters
    assert "is_decode" not in sig.parameters

    # 2. fetch_remote takes no role parameter.
    sig = inspect.signature(DaemonClient.fetch_remote)
    assert "role" not in sig.parameters
    assert "is_prefill_source" not in sig.parameters

    # 3. get_bootstrap_info returns descriptors with tier hints (hbm /
    #    host / disk) but no role hints.
    sig = inspect.signature(DaemonClient.get_bootstrap_info)
    assert "role_filter" not in sig.parameters
