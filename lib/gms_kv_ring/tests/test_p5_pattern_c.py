# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
"""Pattern C end-to-end: router decides + delegates to dest daemon
via a single `fetch_remote` RPC. Bytes flow daemon→daemon over NIXL
with zero router-side per-block orchestration.

This is the architecturally-cleaner alternative to the Rust
orchestrator's `issue_transfer` (Pattern A). The dest daemon plays
the role of router for the wire transfer: opens a UDS to the source
daemon, drives `staging_reserve` (local) + `transfer_block` (source).

Verifies:

  * One RPC from caller to dest daemon kicks off N transfers
  * Bytes arrive byte-identical at the dest's staging tier
  * Engine connector code stays unmodified (none of this code lives
    in vLLM/SGLang/TRT-LLM connectors)

Gated behind ``GMS_KVR_TEST_NIXL_TRANSPORT=1`` and a working UCX.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time

import pytest

pytest.importorskip("nixl")
if not os.environ.get("GMS_KVR_TEST_NIXL_TRANSPORT"):
    pytest.skip(
        "set GMS_KVR_TEST_NIXL_TRANSPORT=1 + working UCX to run",
        allow_module_level=True,
    )

from gms_kv_ring.daemon.client import DaemonClient
from gms_kv_ring.tests._p5_helpers import inject_block as _inject
from gms_kv_ring.tests._p5_helpers import next_port_pair as _next_port_pair
from gms_kv_ring.tests._p5_helpers import spawn_daemon as _spawn_full
from gms_kv_ring.tests._p5_helpers import stop_daemon as _stop
from gms_kv_ring.tests._p5_helpers import wait_for_arrival as _wait_for_arrival


def _spawn(uds, nixl_port, agent_name, *, cap=64 << 20, buf=32 << 20):
    return _spawn_full(
        uds,
        nixl_port,
        agent_name,
        cap=cap,
        send_buf=buf,
        recv_buf=buf,
        storage_dir_prefix="gms-pc-",
    )


def test_pattern_c_single_fetch_remote_rpc_moves_n_blocks():
    """One RPC to the dest daemon orchestrates N transfers. The
    caller (representing the engine connector after reading
    SharedPlacementInfo) supplies source daemon + hashes; dest does
    everything else."""
    n_blocks = 64
    block_size = 4 * 1024
    p_src, p_dst = _next_port_pair()
    uds_src = tempfile.mkdtemp(prefix="gms-pc-src-") + "/d.sock"
    uds_dst = tempfile.mkdtemp(prefix="gms-pc-dst-") + "/d.sock"
    name_src = f"src-{p_src}"
    name_dst = f"dst-{p_dst}"
    ds, ts, lhs = _spawn(uds_src, p_src, name_src)
    dd, td, lhd = _spawn(uds_dst, p_dst, name_dst)
    try:
        if ds.transport is None or dd.transport is None:
            pytest.skip("NIXL UCX transport unavailable on this host")
        # NIXL peer wiring (same as Pattern A tests). In production
        # this is done once at deployment time via transport_add_peer.
        ds.transport.add_peer_inproc(dd.transport)

        # Plant payloads + register them on the source.
        engine_id = "pattern-c-source"
        payloads = []
        hex_hashes = []
        with DaemonClient(uds_src) as sc:
            for i in range(n_blocks):
                payload = (str(i).encode().ljust(64, b"x") * (block_size // 64 + 1))[
                    :block_size
                ]
                content_hash = hashlib.sha256(payload).digest()
                _inject(ds, engine_id, layer=0, offset=i * block_size, data=payload)
                sc.register_content_address(
                    content_hash=content_hash,
                    engine_id=engine_id,
                    ranges=[(0, i * block_size, block_size)],
                )
                payloads.append(payload)
                hex_hashes.append(content_hash.hex())

        # The Pattern C call: ONE RPC to dest daemon kicks off
        # everything. No router-side staging_reserve. No router-side
        # transfer_block. The dest daemon orchestrates.
        with DaemonClient(uds_dst) as dc:
            t0 = time.monotonic()
            result = dc.fetch_remote(
                source_uds_path=uds_src,
                source_nixl_name=name_src,
                source_ip="127.0.0.1",
                source_port=p_src,
                hashes=[bytes.fromhex(h) for h in hex_hashes],
                bytes_per_hash=block_size,
                timeout_s=30.0,
            )
            elapsed = time.monotonic() - t0

        assert result["failed"] == 0, result
        assert (
            result["accepted"] + result["already_ready"] + result["coalesced"]
            == n_blocks
        ), result

        ready = _wait_for_arrival(dd, hex_hashes, sla=30.0)
        assert ready == n_blocks, f"only {ready}/{n_blocks} arrived"

        # Byte-identical check on every block.
        for hex_hash, expected in zip(hex_hashes, payloads):
            key = bytes.fromhex(hex_hash)
            slot = dd.staging_tier._slots[key]
            actual = dd.staging_tier._alloc.read(slot.bytes_ptr, slot.bytes_size)
            assert actual == expected, f"byte mismatch on {hex_hash[:16]}..."

        print(
            f"\n[Pattern C] {n_blocks} blocks × {block_size}B in {elapsed*1000:.0f}ms"
            f" (single fetch_remote RPC, dest-orchestrated, byte-identical)"
        )
    finally:
        _stop(ds, ts, lhs)
        _stop(dd, td, lhd)


def test_pattern_c_incremental_arrival_via_batch_size():
    """Pipelining: with `batch_size=K`, the dest sees blocks arrive
    in batches (one per NIXL xfer) rather than all at once. This is
    the Dynamo-style per-stage pipelining shape: decode can start
    processing layer N's KV while layer N+1 is still in flight.

    Verifies that at least one intermediate observation shows
    `0 < ready_count < N`, meaning blocks didn't all flip to READY
    simultaneously."""
    n_blocks = 128
    block_size = 16 * 1024
    cap = max(n_blocks * block_size * 2, 64 << 20)
    buf = max(n_blocks * block_size * 2, 32 << 20)
    p_src, p_dst = _next_port_pair()
    uds_src = tempfile.mkdtemp(prefix="gms-pc-src-") + "/d.sock"
    uds_dst = tempfile.mkdtemp(prefix="gms-pc-dst-") + "/d.sock"
    name_src = f"src-{p_src}"
    name_dst = f"dst-{p_dst}"
    ds, ts, lhs = _spawn(uds_src, p_src, name_src, cap=cap, buf=buf)
    dd, td, lhd = _spawn(uds_dst, p_dst, name_dst, cap=cap, buf=buf)
    try:
        if ds.transport is None or dd.transport is None:
            pytest.skip("NIXL UCX transport unavailable on this host")
        ds.transport.add_peer_inproc(dd.transport)

        engine_id = "pattern-c-pipeline"
        payloads = []
        hex_hashes = []
        with DaemonClient(uds_src) as sc:
            for i in range(n_blocks):
                payload = (str(i).encode().ljust(64, b"x") * (block_size // 64 + 1))[
                    :block_size
                ]
                content_hash = hashlib.sha256(payload).digest()
                _inject(ds, engine_id, layer=0, offset=i * block_size, data=payload)
                sc.register_content_address(
                    content_hash=content_hash,
                    engine_id=engine_id,
                    ranges=[(0, i * block_size, block_size)],
                )
                payloads.append(payload)
                hex_hashes.append(content_hash.hex())

        # batch_size=8 → 16 batches. Each batch = one NIXL xfer + one
        # multi-record notif → 8 PlacementEvents per batch on dest.
        # Decode-side observer sees ready-count climb in steps of 8.
        import threading

        result_holder: dict = {}
        dc_handle = DaemonClient(uds_dst)

        def _do_fetch():
            try:
                result_holder["result"] = dc_handle.fetch_remote(
                    source_uds_path=uds_src,
                    source_nixl_name=name_src,
                    source_ip="127.0.0.1",
                    source_port=p_src,
                    hashes=[bytes.fromhex(h) for h in hex_hashes],
                    bytes_per_hash=block_size,
                    timeout_s=60.0,
                    batch_size=8,
                )
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = exc

        ft = threading.Thread(target=_do_fetch, daemon=True)
        ft.start()

        # Poll dest staging tier rapidly while transfer is running.
        needed = {bytes.fromhex(h) for h in hex_hashes}
        samples: list[int] = []
        sample_deadline = time.monotonic() + 30.0
        while time.monotonic() < sample_deadline:
            ready = sum(
                1
                for k in needed
                if k in dd.staging_tier._slots
                and dd.staging_tier._slots[k].state.value == "ready"
            )
            samples.append(ready)
            if ready == n_blocks:
                break
            time.sleep(0.002)
        ft.join(timeout=15.0)
        dc_handle.close()
        assert "error" not in result_holder, result_holder["error"]

        # Final state: all ready, byte-identical.
        final_ready = _wait_for_arrival(dd, hex_hashes, sla=15.0)
        assert final_ready == n_blocks, f"final: only {final_ready}/{n_blocks}"
        for hex_hash, expected in zip(hex_hashes, payloads):
            key = bytes.fromhex(hex_hash)
            slot = dd.staging_tier._slots[key]
            actual = dd.staging_tier._alloc.read(slot.bytes_ptr, slot.bytes_size)
            assert actual == expected, f"byte mismatch on {hex_hash[:16]}..."

        # Pipelining contract: ≥2 distinct nonzero ready-counts seen
        # during the transfer. With one big xfer, we'd jump from 0 to
        # N in a single sample.
        distinct = sorted({s for s in samples if 0 < s < n_blocks})
        assert len(distinct) >= 2, (
            f"no incremental pipelining observed; distinct partial "
            f"ready-counts: {distinct}"
        )
        print(
            f"\n[Pattern C pipelining] {n_blocks} blocks × {block_size}B, "
            f"batch_size=8 → {len(distinct)} distinct partial ready-counts; "
            f"first partial sample at count={distinct[0]}"
        )
    finally:
        _stop(ds, ts, lhs)
        _stop(dd, td, lhd)


@pytest.mark.parametrize("hash_mode", ["sha256", "blake2b_256"])
def test_pattern_c_hash_mode(hash_mode, monkeypatch):
    """All supported collision-resistant modes move identical bytes."""
    monkeypatch.setenv("GMS_HASH_MODE", hash_mode)
    from gms_kv_ring.daemon.staging_tier import hash_fn_for_mode

    hash_fn = hash_fn_for_mode(hash_mode)
    n_blocks = 16
    block_size = 4 * 1024
    p_src, p_dst = _next_port_pair()
    uds_src = tempfile.mkdtemp(prefix="gms-pc-src-") + "/d.sock"
    uds_dst = tempfile.mkdtemp(prefix="gms-pc-dst-") + "/d.sock"
    name_src = f"src-{p_src}"
    name_dst = f"dst-{p_dst}"
    ds, ts, lhs = _spawn(uds_src, p_src, name_src)
    dd, td, lhd = _spawn(uds_dst, p_dst, name_dst)
    try:
        if ds.transport is None or dd.transport is None:
            pytest.skip("NIXL UCX transport unavailable")
        ds.transport.add_peer_inproc(dd.transport)

        engine_id = f"pattern-c-hash-{hash_mode}"
        payloads = []
        hex_hashes = []
        with DaemonClient(uds_src) as sc:
            for i in range(n_blocks):
                payload = (str(i).encode().ljust(64, b"x") * (block_size // 64 + 1))[
                    :block_size
                ]
                # IMPORTANT: caller computes content_hash using the
                # SAME mode the daemon will verify with.
                content_hash = hash_fn(payload)
                _inject(ds, engine_id, layer=0, offset=i * block_size, data=payload)
                sc.register_content_address(
                    content_hash=content_hash,
                    engine_id=engine_id,
                    ranges=[(0, i * block_size, block_size)],
                )
                payloads.append(payload)
                hex_hashes.append(content_hash.hex())

        with DaemonClient(uds_dst) as dc:
            t0 = time.monotonic()
            result = dc.fetch_remote(
                source_uds_path=uds_src,
                source_nixl_name=name_src,
                source_ip="127.0.0.1",
                source_port=p_src,
                hashes=[bytes.fromhex(h) for h in hex_hashes],
                bytes_per_hash=block_size,
                timeout_s=30.0,
            )
            elapsed = time.monotonic() - t0

        assert result["failed"] == 0, (hash_mode, result)
        ready = _wait_for_arrival(dd, hex_hashes, sla=30.0)
        assert ready == n_blocks, (hash_mode, f"only {ready}/{n_blocks}")
        # Byte-identical regardless of hash mode.
        for hex_hash, expected in zip(hex_hashes, payloads):
            key = bytes.fromhex(hex_hash)
            slot = dd.staging_tier._slots[key]
            actual = dd.staging_tier._alloc.read(slot.bytes_ptr, slot.bytes_size)
            assert (
                actual == expected
            ), f"[{hash_mode}] byte mismatch on {hex_hash[:16]}..."
        print(
            f"\n[Pattern C hash_mode={hash_mode}] {n_blocks} blocks × "
            f"{block_size}B in {elapsed*1000:.0f}ms, byte-identical"
        )
    finally:
        _stop(ds, ts, lhs)
        _stop(dd, td, lhd)


def test_pattern_c_high_concurrency_1024_blocks():
    """Same as above but at N=1024 to prove Pattern C scales like
    Pattern A's async path (the daemon-side per-peer rate limit
    applies regardless of who triggered the transfer)."""
    n_blocks = 1024
    block_size = 4 * 1024
    cap = max(n_blocks * block_size * 2, 64 << 20)
    buf = max(64 * block_size * 4, 32 << 20)
    p_src, p_dst = _next_port_pair()
    uds_src = tempfile.mkdtemp(prefix="gms-pc-src-") + "/d.sock"
    uds_dst = tempfile.mkdtemp(prefix="gms-pc-dst-") + "/d.sock"
    name_src = f"src-{p_src}"
    name_dst = f"dst-{p_dst}"
    ds, ts, lhs = _spawn(uds_src, p_src, name_src, cap=cap, buf=buf)
    dd, td, lhd = _spawn(uds_dst, p_dst, name_dst, cap=cap, buf=buf)
    try:
        if ds.transport is None or dd.transport is None:
            pytest.skip("NIXL UCX transport unavailable on this host")
        ds.transport.add_peer_inproc(dd.transport)

        engine_id = "pattern-c-1024"
        payloads = []
        hex_hashes = []
        with DaemonClient(uds_src) as sc:
            for i in range(n_blocks):
                payload = (str(i).encode().ljust(64, b"x") * (block_size // 64 + 1))[
                    :block_size
                ]
                content_hash = hashlib.sha256(payload).digest()
                _inject(ds, engine_id, layer=0, offset=i * block_size, data=payload)
                sc.register_content_address(
                    content_hash=content_hash,
                    engine_id=engine_id,
                    ranges=[(0, i * block_size, block_size)],
                )
                payloads.append(payload)
                hex_hashes.append(content_hash.hex())

        with DaemonClient(uds_dst) as dc:
            t0 = time.monotonic()
            result = dc.fetch_remote(
                source_uds_path=uds_src,
                source_nixl_name=name_src,
                source_ip="127.0.0.1",
                source_port=p_src,
                hashes=[bytes.fromhex(h) for h in hex_hashes],
                bytes_per_hash=block_size,
                timeout_s=60.0,
            )
            t_submit = time.monotonic() - t0

        assert result["failed"] == 0, result
        ready = _wait_for_arrival(dd, hex_hashes, sla=60.0)
        assert ready == n_blocks, f"only {ready}/{n_blocks} arrived"

        for hex_hash, expected in zip(hex_hashes, payloads):
            key = bytes.fromhex(hex_hash)
            slot = dd.staging_tier._slots[key]
            actual = dd.staging_tier._alloc.read(slot.bytes_ptr, slot.bytes_size)
            assert actual == expected, f"byte mismatch on {hex_hash[:16]}..."

        print(
            f"\n[Pattern C, N=1024] submit {t_submit*1000:.0f}ms, "
            f"all bytes byte-identical via ONE fetch_remote RPC"
        )
    finally:
        _stop(ds, ts, lhs)
        _stop(dd, td, lhd)
