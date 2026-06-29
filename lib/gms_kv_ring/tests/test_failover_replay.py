# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failover replay test — the only supported multi-engine pattern.

Under the single-active-engine model (see memory:
project_single_active_engine.md), at most one engine actively
drives a daemon at any moment. The only way two engines interact
is FAILOVER: engine A dies, engine B starts up and takes over,
typically reattaching with the SAME engine_id to resume from A's
KV cache state on durable storage.

What this test pins:

  1. Engine A attaches, evicts content to storage (GDS-direct
     path), reads back its own bytes successfully.
  2. Engine A's handle is closed (simulating crash + cleanup).
  3. Engine B starts up, attaches with the SAME engine_id, AND
     can successfully restore A's bytes from storage tier.
  4. Bonus: B can also evict + restore its own new content,
     proving the pool is functional post-replay.

Important: the daemon WIPES the per-engine `EnginePool` (and
therefore the in-memory `slot_generations` dict) on each
re-attach, consistent with the existing re-attach safety
invariant (see test_reattach_engine_id_wipes_prior_host_tier_slots).
What survives across the reattach is the STORAGE TIER content
— files keyed by `(engine_id, layer, offset)`. The in-memory
prefix-hash index also doesn't persist (it lives on the
scheduler process, which dies with A).

Practical implication for the failover path:
  - B can recover A's KV bytes by reading them with
    `expected_generation=0` (opt-out of Race #3 check).
  - B can't trust hashes from A's prefix index without rebuilding
    them (or persisting them — a future enhancement).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


from tests.test_demote_hbm_to_storage import (  # noqa: E402
    _FakeGpuDirectBackend,
    _spawn_daemon,
)


def _open_handle(eid: str, sock: str, layers: list[dict]):
    """Open a GMSKvRing handle for `eid` against `sock`."""
    from gms_kv_ring.engines.handle import GMSKvRing

    return GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)


def _make_layers(dev: torch.Tensor, n_layers: int, block_bytes: int):
    """Build LayerDesc-shaped dicts for a contiguous device tensor."""
    base = int(dev.data_ptr())
    n_blocks = dev.numel() // (n_layers * block_bytes)
    layer_stride = n_blocks * block_bytes
    return [
        {
            "layer_idx": L,
            "va": base + L * layer_stride,
            "size": layer_stride,
            "stride": block_bytes,
        }
        for L in range(n_layers)
    ], n_blocks


def test_failover_engine_a_dies_engine_b_resumes_with_same_engine_id(tmp_path):
    """Full failover cycle.

    Engine A attaches with engine_id="failover-test", spills
    block 5 with generation=1, closes. Engine B comes up with
    the SAME engine_id (continuation), attaches, restores from
    block 5 → block 11 (cross-block-id remap). Daemon's
    slot_generations[5]=1 survived A's close; B's restore
    succeeds.
    """
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "failover.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    eid = "failover-test"

    try:
        # --- Engine A: attach, evict, close ---
        n_layers = 2
        block_bytes = 32
        n_blocks = 16
        dev_a = torch.zeros(
            n_layers * n_blocks * block_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        # Stamp known content X at block 5 of each layer.
        layer_stride = n_blocks * block_bytes
        X = {}
        for L in range(n_layers):
            start = L * layer_stride + 5 * block_bytes
            pat = bytes(((i * 7) ^ (L * 13) ^ 0xC3) & 0xFF for i in range(block_bytes))
            X[L] = pat
            dev_a[start : start + block_bytes].copy_(
                torch.frombuffer(bytearray(pat), dtype=torch.uint8),
            )
        torch.cuda.synchronize()
        layers_a, _ = _make_layers(dev_a, n_layers, block_bytes)

        h_a = _open_handle(eid, sock, layers_a)
        try:
            # A spills block 5 with generation=1 across all layers.
            for L in range(n_layers):
                ok = h_a.demote_hbm_to_storage(
                    L,
                    5 * block_bytes,
                    block_bytes,
                    generation=1,
                )
                assert ok, f"A's demote failed at layer {L}"
            # Daemon confirms generation.
            pool = d._pools[eid]
            assert pool.current_block_generation(5) == 1
        finally:
            h_a.close()

        # A's handle is gone. Daemon still has the storage AND
        # the generation tracking — both keyed by engine_id.

        # --- Engine B: same engine_id, fresh device buffer ---
        # B allocates its OWN HBM (different VA than A). We use a
        # different tensor to make sure B's restore writes go to
        # B's pool, not A's stale pool.
        dev_b = torch.zeros(
            n_layers * n_blocks * block_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        torch.cuda.synchronize()
        layers_b, _ = _make_layers(dev_b, n_layers, block_bytes)

        h_b = _open_handle(eid, sock, layers_b)
        try:
            # B's pool is fresh — generation tracking was wiped on
            # re-attach (consistent with the engine-respawn safety
            # invariant in test_reattach_engine_id_wipes_prior_host_tier_slots).
            # That's the safe-by-default behavior: B can't trust
            # A's leftover generations. But storage TIER survives
            # (it's keyed by engine_id only, not by pool epoch).
            #
            # So B's restores need to either: (a) use
            # expected_generation=0 (opt-out, accepts any), or (b)
            # rebuild a hash index from storage if it wants Race
            # #3 protection on the recovered content.
            # We test the realistic failover pattern: B restores
            # without expected_generation, accepting whatever's
            # in storage.

            # Restore block 5 (A's spill) into B's block 11.
            for L in range(n_layers):
                ok = h_b.promote_storage_to_hbm(
                    L,
                    5 * block_bytes,
                    block_bytes,
                    dest_offset=11 * block_bytes,
                    expected_generation=0,  # opt-out (failover)
                )
                assert ok, f"B's restore failed at layer {L}"

            # Verify byte correctness: B's HBM at block 11
            # matches A's original content X.
            for L in range(n_layers):
                start = L * layer_stride + 11 * block_bytes
                got = bytes(dev_b[start : start + block_bytes].cpu().numpy().tobytes())
                assert got == X[L], (
                    f"failover: layer {L} restored bytes don't " f"match A's spill"
                )

            # Bonus: B can also evict + restore its own new content.
            # Stamp Y into B's block 7, evict with gen=1.
            Y = {}
            for L in range(n_layers):
                start = L * layer_stride + 7 * block_bytes
                pat = bytes(((i + L * 5 + 33) & 0xFF) for i in range(block_bytes))
                Y[L] = pat
                dev_b[start : start + block_bytes].copy_(
                    torch.frombuffer(bytearray(pat), dtype=torch.uint8),
                )
            torch.cuda.synchronize()
            for L in range(n_layers):
                ok = h_b.demote_hbm_to_storage(
                    L,
                    7 * block_bytes,
                    block_bytes,
                    generation=1,
                )
                assert ok
            # B has its OWN EnginePool (the daemon wipes the pool
            # on re-attach). Look up the current one to verify B's
            # generation tracking works.
            pool_b = d._pools[eid]
            assert pool_b.current_block_generation(7) == 1
        finally:
            h_b.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_failover_fresh_engine_id_does_not_see_prior_engines_storage(
    tmp_path,
):
    """Negative test: if engine B attaches with a DIFFERENT
    engine_id than A, B's restore queries against A's storage
    keys must miss (storage is per-engine_id-namespaced)."""
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "failover.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        block_bytes = 32
        n_blocks = 8
        n_layers = 1
        dev_a = torch.zeros(
            n_layers * n_blocks * block_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        pat = bytes(((i * 7) & 0xFF) for i in range(block_bytes))
        dev_a[0:block_bytes].copy_(
            torch.frombuffer(bytearray(pat), dtype=torch.uint8),
        )
        torch.cuda.synchronize()
        layers_a, _ = _make_layers(dev_a, n_layers, block_bytes)

        h_a = _open_handle("engine-A", sock, layers_a)
        try:
            assert h_a.demote_hbm_to_storage(0, 0, block_bytes, generation=1)
        finally:
            h_a.close()

        # B attaches with a DIFFERENT engine_id.
        dev_b = torch.zeros(
            n_layers * n_blocks * block_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        torch.cuda.synchronize()
        layers_b, _ = _make_layers(dev_b, n_layers, block_bytes)
        h_b = _open_handle("engine-B-different", sock, layers_b)
        try:
            # B's promote against "engine-A"'s slot must fail —
            # B's pool doesn't have a slot at offset 0; the
            # backend's _slots dict is keyed by (engine_id, ...).
            # B's engine_id doesn't match where A's bytes live.
            ok = h_b.promote_storage_to_hbm(
                0,
                0,
                block_bytes,
                expected_generation=0,
            )
            assert ok is False, (
                "Engine B with a fresh engine_id must NOT see "
                "engine A's storage. Per-engine_id isolation is "
                "load-bearing for the failover model."
            )
        finally:
            h_b.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)
