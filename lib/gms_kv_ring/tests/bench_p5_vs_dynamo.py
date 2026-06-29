# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apples-to-apples benchmark mimicking Dynamo's actual KV-transfer
pattern: one NIXL operation per layer, layers issued concurrently
(production decode-side issues all `wait_for_layer_load` calls in
parallel; transfers pipeline behind compute).

Two configurations measured:

  * **small**: 32 layers × 4 blocks/layer × 16 KB = 2 MB per request
    (representative of a 2k-token prompt on a 32-layer model with
    16-token blocks).
  * **large**: 32 layers × 32 blocks/layer × 16 KB = 16 MB per request
    (representative of 16k-token prompt).

Baseline = Dynamo engine-direct shape: 32 concurrent NIXL xfers, each
carrying one layer's worth of descriptors. This matches what vLLM's
`KvConnectorBase_V1.save_kv_layer` + decode's `wait_for_layer_load`
produce on the wire.

GMS Pattern C: `fetch_remote(batch_size=blocks_per_layer)` → daemon
issues N parallel batched-NIXL xfers internally (same 32 concurrent
xfers underneath). Hash mode varied.

Each repetition uses a fresh pair of daemons / agents to eliminate
state-accumulation noise.
"""

from __future__ import annotations

import ctypes
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


from gms_kv_ring.tests._p5_helpers import (  # noqa: E402
    next_port_pair as _next_port_pair,
)
from gms_kv_ring.tests._p5_helpers import (  # noqa: E402
    spawn_daemon as _spawn_daemon_full,
)
from gms_kv_ring.tests._p5_helpers import stop_daemon as _stop_daemon  # noqa: E402

# ----------------------------------------------------------------------
# Dynamo baseline: per-layer concurrent NIXL xfers
# ----------------------------------------------------------------------


def _dynamo_baseline_one_run(n_layers: int, blocks_per_layer: int, block_size: int):
    """One end-to-end measurement of Dynamo-equivalent KV transfer.
    Spawns fresh NIXL agents, registers source + destination buffers,
    issues `n_layers` concurrent xfers, waits for all DONE.

    Wire pattern matches what `save_kv_layer`/`wait_for_layer_load`
    produce: one NIXL operation per layer, layers in parallel."""
    from nixl._api import nixl_agent, nixl_agent_config

    pa, pb = _next_port_pair()
    name_a = f"bench-a-{pa}"
    name_b = f"bench-b-{pb}"
    cfg_a = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=pa,
        backends=["UCX"],
    )
    cfg_b = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=True,
        listen_port=pb,
        backends=["UCX"],
    )
    agent_a = nixl_agent(name_a, cfg_a)
    agent_b = nixl_agent(name_b, cfg_b)

    layer_bytes = blocks_per_layer * block_size
    total_bytes = n_layers * layer_bytes
    src_buf = (ctypes.c_ubyte * total_bytes)()
    dst_buf = (ctypes.c_ubyte * total_bytes)()
    src_ptr = ctypes.addressof(src_buf)
    dst_ptr = ctypes.addressof(dst_buf)

    agent_a.register_memory(
        [(src_ptr, total_bytes, 0, "bench-src")],
        mem_type="DRAM",
    )
    agent_b.register_memory(
        [(dst_ptr, total_bytes, 0, "bench-dst")],
        mem_type="DRAM",
    )
    agent_a.add_remote_agent(agent_b.get_agent_metadata())

    # Fill src with distinct per-block content.
    for L in range(n_layers):
        for i in range(blocks_per_layer):
            seed = f"{L}-{i}".encode()
            payload = (seed.ljust(64, b"x") * (block_size // 64 + 1))[:block_size]
            ctypes.memmove(
                src_ptr + (L * blocks_per_layer + i) * block_size,
                payload,
                block_size,
            )

    # Build per-layer descriptor lists (one xfer per layer).
    per_layer_descs = []
    for L in range(n_layers):
        base_off = L * layer_bytes
        local_descs = agent_a.get_xfer_descs(
            [
                (src_ptr + base_off + i * block_size, block_size, 0)
                for i in range(blocks_per_layer)
            ],
            mem_type="DRAM",
        )
        remote_descs = agent_a.get_xfer_descs(
            [
                (dst_ptr + base_off + i * block_size, block_size, 0)
                for i in range(blocks_per_layer)
            ],
            mem_type="DRAM",
        )
        per_layer_descs.append((local_descs, remote_descs))

    # Measured region: issue all xfers concurrently, wait for all DONE.
    ctypes.memset(dst_ptr, 0, total_bytes)
    t0 = time.perf_counter()
    handles = []
    for L in range(n_layers):
        local, remote = per_layer_descs[L]
        h = agent_a.initialize_xfer(
            "WRITE",
            local,
            remote,
            name_b,
            notif_msg=f"L{L}".encode(),
        )
        agent_a.transfer(h)
        handles.append(h)
    # Poll all handles concurrently.
    deadline = time.perf_counter() + 60.0
    pending = list(handles)
    while pending:
        if time.perf_counter() > deadline:
            raise TimeoutError("baseline xfer timed out")
        next_pending = []
        for h in pending:
            st = agent_a.check_xfer_state(h)
            if st in ("DONE", "ERR"):
                if st == "ERR":
                    raise RuntimeError("baseline NIXL ERR")
                agent_a.release_xfer_handle(h)
            else:
                next_pending.append(h)
        pending = next_pending
        if pending:
            time.sleep(0.0001)
    elapsed = time.perf_counter() - t0
    # Verify a sample
    sample = bytes((ctypes.c_ubyte * 16).from_address(dst_ptr))
    assert sample.startswith(b"0-0"), sample[:16]
    return elapsed


# ----------------------------------------------------------------------
# GMS Pattern C: fetch_remote with batch_size = blocks_per_layer
# ----------------------------------------------------------------------


def _spawn_daemon(uds, port, agent_name, *, cap, buf):
    return _spawn_daemon_full(
        uds,
        port,
        agent_name,
        cap=cap,
        send_buf=buf,
        recv_buf=buf,
        storage_dir_prefix="gms-bench-",
    )


# ----------------------------------------------------------------------
# Test driver
# ----------------------------------------------------------------------


def _fmt_ms(s: float) -> str:
    return f"{s * 1000:.2f} ms" if s >= 1e-3 else f"{s * 1_000_000:.0f} µs"


def _stats(samples: list[float]) -> dict:
    samples_sorted = sorted(samples)
    n = len(samples_sorted)
    return {
        "median": samples_sorted[n // 2],
        "p99": samples_sorted[max(0, int(n * 0.99) - 1)],
        "min": samples_sorted[0],
        "max": samples_sorted[-1],
        "mean": sum(samples_sorted) / n,
    }


def _engine_direct_one_run(
    n_layers: int,
    blocks_per_layer: int,
    block_size: int,
):
    """Default engine-direct path with GMS-managed memory:

      1. Daemon A allocates + NIXL-registers its 'engine HBM'.
      2. Daemon A.register_bootstrap_handle(hashes → regions).
      3. (Test code, representing decode engine on B side) creates
         its own NIXL agent, fetches bootstrap_info from daemon A,
         adds A as remote peer, issues N concurrent NIXL READs
         (one per layer).
      4. After reads complete, test calls notify_kv_arrived on
         daemon B (off critical path — for the indexer).

    The daemon is NEVER on the wire path. Latency should match the
    pure-Dynamo baseline within local-UDS overhead (~50-100 µs).
    """
    from gms_kv_ring.daemon.client import DaemonClient
    from nixl._api import nixl_agent, nixl_agent_config

    total_blocks = n_layers * blocks_per_layer
    layer_bytes = blocks_per_layer * block_size
    total_bytes = total_blocks * block_size
    cap = max(total_bytes * 2, 64 << 20)
    bufsize = max(total_bytes * 2, 128 << 20)
    pa, pb = _next_port_pair()
    p_engine = pa + 50  # test 'engine' agent on the dest side
    uds_a = tempfile.mkdtemp(prefix="gms-bench-eda-") + "/d.sock"
    uds_b = tempfile.mkdtemp(prefix="gms-bench-edb-") + "/d.sock"
    name_a = f"gms-bench-eda-{pa}"
    name_b = f"gms-bench-edb-{pb}"
    name_engine = f"engine-{p_engine}"
    da, ta, ha = _spawn_daemon(uds_a, pa, name_a, cap=cap, buf=bufsize)
    db, tb, hb = _spawn_daemon(uds_b, pb, name_b, cap=cap, buf=bufsize)
    if da.transport is None or db.transport is None:
        _stop_daemon(da, ta, ha)
        _stop_daemon(db, tb, hb)
        pytest.skip("NIXL UCX transport unavailable")
    try:
        # Pre-setup on source: allocate one big buffer, fill with
        # per-block content, NIXL-register, register bootstrap handles.
        src_buf = (ctypes.c_ubyte * total_bytes)()
        src_ptr = ctypes.addressof(src_buf)
        for L in range(n_layers):
            for i in range(blocks_per_layer):
                seed = f"{L}-{i}".encode()
                payload = (seed.ljust(64, b"x") * (block_size // 64 + 1))[:block_size]
                off = (L * blocks_per_layer + i) * block_size
                ctypes.memmove(src_ptr + off, payload, block_size)
        # Daemon NIXL-registers and stores bootstrap handles per layer.
        # We register one handle PER LAYER (matching the per-layer
        # NIXL-read pattern below).
        with DaemonClient(uds_a) as sc:
            layer_items = []
            for L in range(n_layers):
                # Use the layer-content hash as the key.
                seed = f"L{L}".encode()
                content_hash = hashlib.sha256(seed).digest()
                layer_items.append(
                    {
                        "content_hash": content_hash,
                        "ptr": src_ptr + L * layer_bytes,
                        "size": layer_bytes,
                    }
                )
            sc.register_bootstrap_handle(layer_items)

        # Set up the "engine on dest side" — a fresh NIXL agent +
        # destination buffer + remote agent registration.
        cfg_engine = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=True,
            listen_port=p_engine,
            backends=["UCX"],
        )
        engine = nixl_agent(name_engine, cfg_engine)
        dst_buf = (ctypes.c_ubyte * total_bytes)()
        dst_ptr = ctypes.addressof(dst_buf)
        engine.register_memory(
            [(dst_ptr, total_bytes, 0, "engine-dst")],
            mem_type="DRAM",
        )

        # PRE-MEASURED SETUP — these are ONE-TIME per (engine, peer)
        # in production. Pre-opening DaemonClient (persistent UDS),
        # pre-fetching bootstrap_info (info travels in-band via
        # SharedPlacementInfo on the request — no per-request RPC),
        # and pre-adding the remote NIXL agent (one-time NIXL
        # metadata exchange per peer pair).
        layer_hashes = [
            hashlib.sha256(f"L{L}".encode()).digest() for L in range(n_layers)
        ]
        # Pre-fetch bootstrap_info (in production this travels in-band
        # via SharedPlacementInfo, NOT a per-request RPC).
        with DaemonClient(uds_a) as dc_a:
            info = dc_a.get_bootstrap_info(layer_hashes)
        src_agent_name = info["nixl_agent_name"]
        src_agent_md = bytes.fromhex(info["agent_metadata_b64"])
        descriptors = info["descriptors"]
        # Pre-add remote NIXL agent (one-time per peer in production).
        engine.add_remote_agent(src_agent_md)

        # MEASURED REGION — per-request critical path only.
        ctypes.memset(dst_ptr, 0, total_bytes)
        t0 = time.perf_counter()

        # N concurrent NIXL READs, one per layer
        handles = []
        for L in range(n_layers):
            desc = descriptors[L]
            assert desc is not None, f"missing bootstrap descriptor for L{L}"
            local_descs = engine.get_xfer_descs(
                [(dst_ptr + L * layer_bytes, layer_bytes, 0)],
                mem_type="DRAM",
            )
            remote_descs = engine.get_xfer_descs(
                [(int(desc["ptr"]), int(desc["size"]), 0)],
                mem_type="DRAM",
            )
            handle = engine.initialize_xfer(
                "READ",
                local_descs,
                remote_descs,
                src_agent_name,
                notif_msg=f"L{L}".encode(),
            )
            engine.transfer(handle)
            handles.append(handle)
        # Poll all handles
        deadline = time.perf_counter() + 60.0
        pending = list(handles)
        while pending:
            if time.perf_counter() > deadline:
                raise TimeoutError("engine-direct READ timed out")
            next_pending = []
            for h in pending:
                st = engine.check_xfer_state(h)
                if st in ("DONE", "ERR"):
                    if st == "ERR":
                        raise RuntimeError("engine-direct READ ERR")
                    engine.release_xfer_handle(h)
                else:
                    next_pending.append(h)
            pending = next_pending
            if pending:
                time.sleep(0.0001)
        elapsed = time.perf_counter() - t0

        # Verify a sample
        sample = bytes((ctypes.c_ubyte * 16).from_address(dst_ptr))
        assert sample.startswith(b"0-0"), sample[:16]
        return elapsed
    finally:
        _stop_daemon(da, ta, ha)
        _stop_daemon(db, tb, hb)


def _gms_pattern_c_amortized(
    n_layers: int,
    blocks_per_layer: int,
    block_size: int,
    n_reps: int,
    hash_mode: str,
) -> list[float]:
    """Amortized GMS-initiated (Pattern C / fetch_remote) measurement.

    Setup ONCE outside the timed region:
      - daemons + NIXL agents spawned
      - add_peer_inproc completed (NIXL metadata exchange done)
      - DaemonClient to dest pre-opened (persistent UDS)
      - 1 warmup fetch_remote (establishes dest's source-pool, drains
        first-time alloc + JIT effects)

    Then measure `n_reps` consecutive fetch_remote calls against the
    same daemon pair. Each rep uses unique content_hashes so no slot
    collisions. Reports per-rep samples.
    """
    os.environ["GMS_HASH_MODE"] = hash_mode
    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.daemon.staging_tier import hash_fn_for_mode

    total_blocks_per_rep = n_layers * blocks_per_layer
    block_bytes_per_rep = total_blocks_per_rep * block_size
    # Daemon capacity must hold ALL reps' worth of slots, since
    # they accumulate (different content_hashes per rep).
    cap = max(block_bytes_per_rep * (n_reps + 2) * 2, 256 << 20)
    bufsize = max(block_bytes_per_rep * 4, 128 << 20)
    pa, pb = _next_port_pair()
    uds_a = tempfile.mkdtemp(prefix="gms-amort-a-") + "/d.sock"
    uds_b = tempfile.mkdtemp(prefix="gms-amort-b-") + "/d.sock"
    name_a = f"gms-amort-a-{pa}"
    name_b = f"gms-amort-b-{pb}"
    da, ta, ha = _spawn_daemon(uds_a, pa, name_a, cap=cap, buf=bufsize)
    db, tb, hb = _spawn_daemon(uds_b, pb, name_b, cap=cap, buf=bufsize)
    if da.transport is None or db.transport is None:
        _stop_daemon(da, ta, ha)
        _stop_daemon(db, tb, hb)
        pytest.skip("NIXL UCX transport unavailable")
    try:
        da.transport.add_peer_inproc(db.transport)
        hash_fn = hash_fn_for_mode(hash_mode)

        # Pre-populate ALL reps' host_tier slots + register content
        # addresses. (Done outside the timed region; in production
        # this is the engine's existing prefill→spill path.)
        all_hashes_per_rep = []
        with DaemonClient(uds_a) as sc:
            for r in range(n_reps + 1):  # +1 for warmup
                rep_hashes = []
                for L in range(n_layers):
                    for i in range(blocks_per_layer):
                        seed = f"r{r}-L{L}-i{i}".encode()
                        payload = (seed.ljust(64, b"x") * (block_size // 64 + 1))[
                            :block_size
                        ]
                        content_hash = hash_fn(payload)
                        ptr = da.host_tier.put(
                            "bench",
                            layer=r * 1000 + L,
                            offset=i * block_size,
                            size=block_size,
                        )
                        ctypes.memmove(ptr, payload, block_size)
                        da.host_tier.mark_ready(
                            "bench",
                            r * 1000 + L,
                            i * block_size,
                            crc=0,
                        )
                        sc.register_content_address(
                            content_hash=content_hash,
                            engine_id="bench",
                            ranges=[(r * 1000 + L, i * block_size, block_size)],
                        )
                        rep_hashes.append(content_hash.hex())
                all_hashes_per_rep.append(rep_hashes)

        # Pre-open persistent DaemonClient to dest. In production this
        # is the engine's long-lived UDS to its local GMS daemon.
        dc = DaemonClient(uds_b)
        try:
            # Warmup: one full fetch_remote round-trip. This:
            # - opens the dest daemon's source-client pool to A (the
            #   _SourceClientPool ctor — 4 UDS connects + pings)
            # - drains first-time NIXL state setup
            # - warms Python JITs and any internal caches
            warmup_hashes = all_hashes_per_rep[0]
            dc.fetch_remote(
                source_uds_path=uds_a,
                source_nixl_name=name_a,
                source_ip="127.0.0.1",
                source_port=pa,
                hashes=[bytes.fromhex(h) for h in warmup_hashes],
                bytes_per_hash=block_size,
                timeout_s=30.0,
                batch_size=blocks_per_layer,
            )
            # Drain warmup arrivals
            needed = {bytes.fromhex(h) for h in warmup_hashes}
            deadline = time.perf_counter() + 30.0
            while True:
                ready = sum(
                    1
                    for k in needed
                    if k in db.staging_tier._slots
                    and db.staging_tier._slots[k].state.value == "ready"
                )
                if ready == total_blocks_per_rep:
                    break
                if time.perf_counter() > deadline:
                    raise TimeoutError("warmup fetch_remote stalled")
                time.sleep(0.0001)

            # Measured region: per-rep fetch_remote against warm state.
            samples = []
            for r in range(1, n_reps + 1):
                hex_hashes = all_hashes_per_rep[r]
                t0 = time.perf_counter()
                result = dc.fetch_remote(
                    source_uds_path=uds_a,
                    source_nixl_name=name_a,
                    source_ip="127.0.0.1",
                    source_port=pa,
                    hashes=[bytes.fromhex(h) for h in hex_hashes],
                    bytes_per_hash=block_size,
                    timeout_s=60.0,
                    batch_size=blocks_per_layer,
                )
                needed_r = {bytes.fromhex(h) for h in hex_hashes}
                deadline = time.perf_counter() + 60.0
                while True:
                    ready = sum(
                        1
                        for k in needed_r
                        if k in db.staging_tier._slots
                        and db.staging_tier._slots[k].state.value == "ready"
                    )
                    if ready == total_blocks_per_rep:
                        break
                    if time.perf_counter() > deadline:
                        raise TimeoutError(
                            f"rep {r}: only {ready}/{total_blocks_per_rep}"
                        )
                    time.sleep(0.0001)
                elapsed = time.perf_counter() - t0
                samples.append(elapsed)
                assert result["failed"] == 0, result
            return samples
        finally:
            dc.close()
    finally:
        _stop_daemon(da, ta, ha)
        _stop_daemon(db, tb, hb)


def _run_config(
    label: str, n_layers: int, blocks_per_layer: int, block_size: int, n_reps: int
):
    total_bytes = n_layers * blocks_per_layer * block_size
    print(
        f"\n\n=== {label}: {n_layers} layers × {blocks_per_layer} blocks/layer × "
        f"{block_size // 1024} KB = {total_bytes // (1024 * 1024)} MB per request, "
        f"{n_reps} reps (fresh daemons per rep), UCX TCP ==="
    )

    # Baseline
    print("\n[baseline] Dynamo-equivalent (N concurrent NIXL xfers, 1 per layer)...")
    bs_samples = []
    for r in range(n_reps):
        try:
            bs_samples.append(
                _dynamo_baseline_one_run(n_layers, blocks_per_layer, block_size)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  rep {r}: FAILED ({exc})")
    bs = _stats(bs_samples) if bs_samples else None
    if bs:
        print(
            f"  median={_fmt_ms(bs['median'])}  p99={_fmt_ms(bs['p99'])}  min={_fmt_ms(bs['min'])}  (n={len(bs_samples)})"
        )

    # Engine-direct path (default for GMS — same NIXL wire as Dynamo)
    print(
        "\n[engine-direct] GMS engine-direct (daemon-managed memory, engine NIXL-reads)..."
    )
    ed_samples = []
    for r in range(n_reps):
        try:
            ed_samples.append(
                _engine_direct_one_run(n_layers, blocks_per_layer, block_size)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  rep {r}: FAILED ({exc})")
    ed = _stats(ed_samples) if ed_samples else None
    if ed:
        print(
            f"  median={_fmt_ms(ed['median'])}  p99={_fmt_ms(ed['p99'])}  min={_fmt_ms(ed['min'])}  (n={len(ed_samples)})"
        )

    # Pattern C variants (fallback path) — AMORTIZED: daemon spawn,
    # NIXL peering, UDS pool, and warmup all done ONCE outside the
    # timed region. Measures steady-state per-request cost.
    modes = ["sha256", "blake2b_256"]
    results: list = []
    for mode in modes:
        samples = []
        try:
            samples = _gms_pattern_c_amortized(
                n_layers,
                blocks_per_layer,
                block_size,
                n_reps,
                mode,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{mode}] FAILED ({exc})")
        if samples:
            s = _stats(samples)
            results.append((f"GMS Pattern C ({mode})", s, len(samples), n_reps))
            print(
                f"\n[{mode}] median={_fmt_ms(s['median'])}  p99={_fmt_ms(s['p99'])}  "
                f"min={_fmt_ms(s['min'])}  (n={len(samples)}/{n_reps})"
            )
        else:
            print(f"\n[{mode}] ALL REPS FAILED")

    # Final table
    print(f"\n\n========== RESULTS: {label} ==========")
    print(
        f"{'Path':<38} {'median':>10} {'p99':>10} {'min':>10} "
        f"{'vs base (median)':>20}  {'n':>5}"
    )
    print("-" * 100)
    if bs:
        bs_median = bs["median"]
        print(
            f"{'Dynamo (per-layer NIXL × N)':<38} {_fmt_ms(bs['median']):>10} "
            f"{_fmt_ms(bs['p99']):>10} {_fmt_ms(bs['min']):>10} "
            f"{'—':>20}  {len(bs_samples):>5}/{n_reps}"
        )
        if ed:
            ratio = ed["median"] / bs_median
            delta_ms = (ed["median"] - bs_median) * 1000
            note = (
                f"{ratio:.2f}×  ({delta_ms:+.1f} ms)"
                if abs(ratio - 1.0) > 0.01
                else "1.00× (parity)"
            )
            print(
                f"{'GMS engine-direct (default)':<38} {_fmt_ms(ed['median']):>10} "
                f"{_fmt_ms(ed['p99']):>10} {_fmt_ms(ed['min']):>10} "
                f"{note:>20}  {len(ed_samples):>5}/{n_reps}"
            )
        for name, s, n_ok, _ in results:
            ratio = s["median"] / bs_median
            delta_ms = (s["median"] - bs_median) * 1000
            note = f"{ratio:.2f}×  (+{delta_ms:.1f} ms)"
            print(
                f"{name:<38} {_fmt_ms(s['median']):>10} {_fmt_ms(s['p99']):>10} "
                f"{_fmt_ms(s['min']):>10} {note:>20}  {n_ok:>5}/{n_reps}"
            )


def test_bench_dynamo_pattern(capsys):
    """Run both small + large configurations."""
    # Small: 32 layers × 4 blocks × 16 KB = 2 MB
    _run_config(
        "small (2 MB per req)",
        n_layers=32,
        blocks_per_layer=4,
        block_size=16 * 1024,
        n_reps=5,
    )
    # Large: 32 layers × 32 blocks × 16 KB = 16 MB
    _run_config(
        "large (16 MB per req)",
        n_layers=32,
        blocks_per_layer=32,
        block_size=16 * 1024,
        n_reps=5,
    )
    # Just make sure the test ran something
    captured = capsys.readouterr()
    assert "RESULTS" in captured.out
