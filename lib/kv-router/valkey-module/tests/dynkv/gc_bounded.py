# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
import time
from pathlib import Path

from .support import (
    ADMISSION_RESERVED,
    LEGACY_LEASED_REGISTRATION_VERSION,
    REMOVE,
    STORE,
    WIRE_VERSION,
    _apply_owned,
    _command,
    _drain_gc,
    _event,
    _find_free_port,
    _gc,
    _gc_current,
    _gc_stats,
    _leased_registration_payload,
    _match,
    _memory_usage,
    _rank_generation,
    _register_worker,
    _register_worker_ranks_leased,
    _registration_generation,
    _release,
    _replace_rank_if_generation,
    _reserve,
    _reserve_request,
    _start,
    _stats,
    _stop,
    _unregister_worker,
)



def run_bounded_gc(server: str, module: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        port = _find_free_port()
        process = _start(server, module, directory, port, appendonly=True)
        key = b"bounded-gc-index"
        try:
            # Registration v3 is an owner/rank-set CAS. Exact ambiguous
            # retries succeed, but an older rank set cannot roll back a newer
            # one and a token from before epoch GC cannot recreate its owner.
            cas_key = b"bounded-gc-registration-cas"
            cas_worker, cas_owner = 1190, 119_001
            stale_rank_generation = _rank_generation(port, cas_key, cas_worker, 0)
            initial_token = _registration_generation(port, cas_key, cas_worker)
            initial_registration = _leased_registration_payload(
                cas_owner, 120_000, [0], initial_token
            )
            assert (
                _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    cas_key,
                    struct.pack("!Q", cas_worker),
                    initial_registration,
                )
                == b"OK"
            )
            assert (
                _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    cas_key,
                    struct.pack("!Q", cas_worker),
                    initial_registration,
                )
                == b"OK"
            )
            update_token = _registration_generation(port, cas_key, cas_worker)
            expanded_registration = _leased_registration_payload(
                cas_owner, 120_000, [0, 1], update_token
            )
            assert (
                _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    cas_key,
                    struct.pack("!Q", cas_worker),
                    expanded_registration,
                )
                == b"OK"
            )
            assert (
                _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    cas_key,
                    struct.pack("!Q", cas_worker),
                    expanded_registration,
                )
                == b"OK"
            )
            try:
                _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    cas_key,
                    struct.pack("!Q", cas_worker),
                    initial_registration,
                )
            except RuntimeError as error:
                assert "DYNKV_STALE_REGISTRATION_GENERATION" in str(error)
            else:
                raise AssertionError("an old rank set must not roll back v3 state")
            assert (
                _apply_owned(
                    port,
                    cas_key,
                    cas_owner,
                    _event(
                        STORE,
                        cas_worker,
                        0,
                        1,
                        blocks=[(11901, 119)],
                    ),
                )
                == b"OK"
            )
            assert _unregister_worker(port, cas_key, cas_worker, cas_owner) == b"OK"
            post_end_rank_generation = _rank_generation(
                port, cas_key, cas_worker, 0
            )
            _drain_gc(port, cas_key)
            assert _stats(port, cas_key)[:2] == (0, 0)
            try:
                _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    cas_key,
                    struct.pack("!Q", cas_worker),
                    expanded_registration,
                )
            except RuntimeError as error:
                assert "DYNKV_STALE_REGISTRATION_GENERATION" in str(error)
            else:
                raise AssertionError("a pre-GC owner token must not resurrect")
            try:
                _replace_rank_if_generation(
                    port,
                    cas_key,
                    cas_worker,
                    0,
                    stale_rank_generation,
                    [_event(STORE, cas_worker, 0, 1, blocks=[(11901, 119)])],
                )
            except RuntimeError as error:
                assert "DYNKV_STALE_GENERATION" in str(error)
            else:
                raise AssertionError("epoch GC must not create recovery CAS ABA")
            try:
                _replace_rank_if_generation(
                    port,
                    cas_key,
                    cas_worker,
                    0,
                    post_end_rank_generation,
                    [_event(STORE, cas_worker, 0, 1, blocks=[(11901, 119)])],
                )
            except RuntimeError as error:
                assert "DYNKV_STALE_GENERATION" in str(error)
            else:
                raise AssertionError("post-unregister recovery token must stale after GC")
            successor_owner = 119_002
            assert (
                _register_worker_ranks_leased(
                    port, cas_key, cas_worker, successor_owner, 120_000, [0]
                )
                == b"OK"
            )
            try:
                _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    cas_key,
                    struct.pack("!Q", cas_worker),
                    expanded_registration,
                )
            except RuntimeError as error:
                assert "DYNKV_STALE_REGISTRATION_GENERATION" in str(error)
            else:
                raise AssertionError("old registration must not replace successor")

            # High-rate KV generations from another active worker do not
            # invalidate an absent worker's separate lifecycle token. Direct
            # STORE/REMOVE churn prunes ownership and radix nodes immediately.
            pending_worker = 1191
            pending_token = _registration_generation(port, cas_key, pending_worker)
            for offset in range(64):
                external_hash = 11910 + offset
                assert (
                    _apply_owned(
                        port,
                        cas_key,
                        successor_owner,
                        _event(
                            STORE,
                            cas_worker,
                            0,
                            offset * 2 + 1,
                            blocks=[(external_hash, 500 + offset)],
                        ),
                    )
                    == b"OK"
                )
                assert (
                    _apply_owned(
                        port,
                        cas_key,
                        successor_owner,
                        _event(
                            REMOVE,
                            cas_worker,
                            0,
                            offset * 2 + 2,
                            blocks=[(external_hash, 0)],
                        ),
                    )
                    == b"OK"
                )
                assert _stats(port, cas_key)[0] == 0
            pending_registration = _leased_registration_payload(
                119_003, 120_000, [0], pending_token
            )
            assert (
                _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    cas_key,
                    struct.pack("!Q", pending_worker),
                    pending_registration,
                )
                == b"OK"
            )

            # Compacting a multi-block chain repeatedly swap-moves the last
            # worker node into the removed slot. Its dictionary position must
            # be replaced, not insert-only, or every other owner is stranded.
            multiblock_key = b"bounded-gc-active-multiblock-compaction"
            multiblock_worker, multiblock_owner = 1196, 119_008
            assert (
                _register_worker_ranks_leased(
                    port,
                    multiblock_key,
                    multiblock_worker,
                    multiblock_owner,
                    120_000,
                    [0],
                )
                == b"OK"
            )
            event_id = 1
            for block_count in (2, 16):
                blocks = [
                    (11960 + block_count * 100 + offset, 900 + offset)
                    for offset in range(block_count)
                ]
                assert (
                    _apply_owned(
                        port,
                        multiblock_key,
                        multiblock_owner,
                        _event(
                            STORE,
                            multiblock_worker,
                            0,
                            event_id,
                            blocks=blocks,
                        ),
                    )
                    == b"OK"
                )
                event_id += 1
                assert (
                    _apply_owned(
                        port,
                        multiblock_key,
                        multiblock_owner,
                        _event(
                            REMOVE,
                            multiblock_worker,
                            0,
                            event_id,
                            blocks=blocks,
                        ),
                    )
                    == b"OK"
                )
                event_id += 1
                assert _stats(port, multiblock_key)[0] == 0
                multiblock_gc = _gc_stats(port, multiblock_key)
                assert multiblock_gc[5:7] == (0, 0), multiblock_gc
                request = bytearray(
                    struct.pack("!BI", WIRE_VERSION, len(blocks))
                )
                for _, local_hash in blocks:
                    request.extend(struct.pack("!Q", local_hash))
                assert _match(
                    _command(
                        port, b"DYNKV.MATCH", multiblock_key, bytes(request)
                    )
                ) == {}

            scale_key = b"bounded-gc-registration-scale"
            scale_workers = list(range(1800, 1928))
            shared_tokens = [
                _registration_generation(port, scale_key, worker)
                for worker in scale_workers
            ]
            assert len(set(shared_tokens)) == 1
            for offset, worker in enumerate(scale_workers):
                assert (
                    _command(
                        port,
                        b"DYNKV.REGISTER_WORKER_RANKS",
                        scale_key,
                        struct.pack("!Q", worker),
                        _leased_registration_payload(
                            180_000 + offset,
                            120_000,
                            [0],
                            shared_tokens[offset],
                        ),
                    )
                    == b"OK"
                )
            installed_tokens = {
                _registration_generation(port, scale_key, worker)
                for worker in scale_workers
            }
            assert len(installed_tokens) == len(scale_workers)
            for offset, worker in enumerate(scale_workers):
                assert (
                    _unregister_worker(port, scale_key, worker, 180_000 + offset)
                    == b"OK"
                )
            _drain_gc(port, scale_key, budget=1024)
            assert _stats(port, scale_key)[:2] == (0, 0)

            # Expired crash owners are ended and subsequently compacted by
            # bounded GC without requiring admission traffic or a successor.
            crash_key = b"bounded-gc-crash-expiry"
            assert (
                _register_worker_ranks_leased(
                    port, crash_key, 1192, 119_004, 60, [0]
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    crash_key,
                    119_004,
                    _event(STORE, 1192, 0, 1, blocks=[(11921, 600)]),
                )
                == b"OK"
            )
            time.sleep(0.09)
            _drain_gc(port, crash_key, budget=16)
            assert _stats(port, crash_key)[:2] == (0, 0)

            crash_churn_key = b"bounded-gc-crash-churn"
            for offset in range(32):
                assert (
                    _register_worker_ranks_leased(
                        port,
                        crash_churn_key,
                        1700 + offset,
                        170_000 + offset,
                        60,
                        [0],
                    )
                    == b"OK"
                )
                assert (
                    _apply_owned(
                        port,
                        crash_churn_key,
                        170_000 + offset,
                        _event(
                            STORE,
                            1700 + offset,
                            0,
                            1,
                            blocks=[(17000 + offset, 700 + offset)],
                        ),
                    )
                    == b"OK"
                )
            time.sleep(0.09)
            _drain_gc(port, crash_churn_key, budget=256)
            assert _stats(port, crash_churn_key)[:2] == (0, 0)

            # Version-2 leased registration and accepted legacy registration
            # NOOPs are retained and explicitly counted for rolling safety.
            legacy_v2_key = b"bounded-gc-legacy-v2"
            legacy_v2_payload = _leased_registration_payload(
                119_005,
                120_000,
                [0],
                version=LEGACY_LEASED_REGISTRATION_VERSION,
            )
            assert (
                _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    legacy_v2_key,
                    struct.pack("!Q", 1193),
                    legacy_v2_payload,
                )
                == b"OK"
            )
            assert _unregister_worker(port, legacy_v2_key, 1193, 119_005) == b"OK"
            for _ in range(8):
                _gc_current(port, legacy_v2_key, 32)
            assert _stats(port, legacy_v2_key)[1] == 1
            assert _gc_stats(port, legacy_v2_key)[2:5] == (1, 0, 1)

            legacy_noop_key = b"bounded-gc-legacy-noop"
            assert (
                _register_worker_ranks_leased(
                    port, legacy_noop_key, 1194, 119_006, 120_000, [0]
                )
                == b"OK"
            )
            assert _register_worker(port, legacy_noop_key, 1194, 0) == b"NOOP"
            assert _unregister_worker(port, legacy_noop_key, 1194, 119_006) == b"OK"
            for _ in range(8):
                _gc_current(port, legacy_noop_key, 32)
            assert _stats(port, legacy_noop_key)[1] == 1
            assert _gc_stats(port, legacy_noop_key)[2:5] == (1, 0, 1)

            # Two owner-fenced ranks share the same parent/leaf topology. GC
            # may remove one owner's records but must preserve shared nodes.
            first, second = 1200, 1201
            first_owner, second_owner = 120_001, 120_002
            for worker, owner in ((first, first_owner), (second, second_owner)):
                assert (
                    _register_worker_ranks_leased(
                        port, key, worker, owner, 120_000, [0]
                    )
                    == b"OK"
                )
                assert (
                    _apply_owned(
                        port,
                        key,
                        owner,
                        _event(
                            STORE,
                            worker,
                            0,
                            1,
                            blocks=[(12001, 120), (12002, 121)],
                        ),
                    )
                    == b"OK"
                )
            assert _unregister_worker(port, key, first, first_owner) == b"OK"
            watermark = _gc_stats(port, key)[0]
            # A one-item budget bounds both inspected and reclaimed work.
            for _ in range(64):
                result = _gc(port, key, watermark, 1)
                assert result[0] <= 1 and result[1] <= 1
                if _stats(port, key)[1] == 1:
                    break
            assert _stats(port, key)[:2] == (2, 1)
            request = struct.pack("!BIQQ", WIRE_VERSION, 2, 120, 121)
            assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {
                (second, 0): (2, 12002)
            }
            try:
                _apply_owned(
                    port,
                    key,
                    first_owner,
                    _event(STORE, first, 0, 2, blocks=[(12003, 122)]),
                )
            except RuntimeError as error:
                assert "DYNKV_STALE_WORKER_OWNER" in str(error)
            else:
                raise AssertionError("GC must not weaken the expired owner fence")

            # The retention watermark is inclusive: a generation immediately
            # below the tombstone cannot remove any part of that rank.
            retained_worker, retained_owner = 1205, 120_007
            assert (
                _register_worker_ranks_leased(
                    port, key, retained_worker, retained_owner, 120_000, [0]
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    retained_owner,
                    _event(
                        STORE,
                        retained_worker,
                        0,
                        1,
                        blocks=[(12051, 128)],
                    ),
                )
                == b"OK"
            )
            assert (
                _unregister_worker(port, key, retained_worker, retained_owner)
                == b"OK"
            )
            retained_generation = _gc_stats(port, key)[0]
            try:
                _gc(port, key, retained_generation + 1, 1)
            except RuntimeError as error:
                assert "DYNKV_GC_FUTURE_WATERMARK" in str(error)
            else:
                raise AssertionError("GC must reject a future generation watermark")
            for _ in range(12):
                _gc(port, key, retained_generation - 1, 32)
            assert _stats(port, key)[1] == 2
            for _ in range(32):
                _gc(port, key, retained_generation, 32)
                if _stats(port, key)[1] == 1:
                    break
            assert _stats(port, key)[1] == 1

            # A successor claimed after bounded unregister cleanup. Neither
            # its rank nor its ownership can be reclaimed by an old watermark.
            successor_worker = 1202
            predecessor_owner, successor_owner = 120_003, 120_004
            assert (
                _register_worker_ranks_leased(
                    port,
                    key,
                    successor_worker,
                    predecessor_owner,
                    120_000,
                    [0],
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    predecessor_owner,
                    _event(
                        STORE,
                        successor_worker,
                        0,
                        1,
                        blocks=[(12021, 123)],
                    ),
                )
                == b"OK"
            )
            assert (
                _unregister_worker(
                    port, key, successor_worker, predecessor_owner
                )
                == b"OK"
            )
            old_watermark = _gc_stats(port, key)[0]
            for _ in range(64):
                try:
                    registered = _register_worker_ranks_leased(
                        port,
                        key,
                        successor_worker,
                        successor_owner,
                        120_000,
                        [0],
                    )
                except RuntimeError as error:
                    assert "DYNKV_WORKER_CLEANUP_PENDING" in str(error)
                    _gc_current(port, key, 64)
                    continue
                assert registered == b"OK"
                break
            else:
                raise AssertionError("successor cleanup did not converge")
            for _ in range(16):
                _gc(port, key, old_watermark, 16)
            assert (
                _apply_owned(
                    port,
                    key,
                    successor_owner,
                    _event(
                        STORE,
                        successor_worker,
                        0,
                        1,
                        blocks=[(12022, 124)],
                    ),
                )
                == b"OK"
            )
            try:
                _apply_owned(
                    port,
                    key,
                    predecessor_owner,
                    _event(
                        STORE,
                        successor_worker,
                        0,
                        2,
                        blocks=[(12023, 125)],
                    ),
                )
            except RuntimeError as error:
                assert "DYNKV_STALE_WORKER_OWNER" in str(error)
            else:
                raise AssertionError("successor must preserve owner fencing")

            # Admission authority is reclaimed only after its reservation is
            # released and the corresponding worker lease ends.
            admission_worker, admission_owner = 1203, 120_005
            assert (
                _register_worker_ranks_leased(
                    port, key, admission_worker, admission_owner, 120_000, [0]
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    admission_owner,
                    _event(
                        STORE,
                        admission_worker,
                        0,
                        1,
                        blocks=[(12031, 126)],
                    ),
                )
                == b"OK"
            )
            raw, reservation = _reserve(
                port,
                key,
                _reserve_request(
                    b"gc",
                    120_100,
                    120_101,
                    60_000,
                    [126],
                    [(admission_worker, 0, 1)],
                ),
            )
            assert raw[1] == ADMISSION_RESERVED and reservation is not None
            assert (
                _release(
                    port,
                    key,
                    b"gc",
                    120_100,
                    120_101,
                    reservation["expires"],
                )
                == 1
            )
            assert (
                _unregister_worker(port, key, admission_worker, admission_owner)
                == b"OK"
            )

            # One legacy event permanently taints this incarnation. Bounded
            # GC reports and retains it for rolling-upgrade compatibility.
            legacy_worker, legacy_owner = 1204, 120_006
            assert (
                _register_worker_ranks_leased(
                    port, key, legacy_worker, legacy_owner, 120_000, [0]
                )
                == b"OK"
            )
            assert (
                _command(
                    port,
                    b"DYNKV.APPLY",
                    key,
                    _event(
                        STORE,
                        legacy_worker,
                        0,
                        1,
                        blocks=[(12041, 127)],
                    ),
                )
                == b"OK"
            )
            assert _unregister_worker(port, key, legacy_worker, legacy_owner) == b"OK"

            assert _unregister_worker(port, key, second, second_owner) == b"OK"
            assert (
                _unregister_worker(
                    port, key, successor_worker, successor_owner
                )
                == b"OK"
            )
            _drain_gc(port, key)
            gc_stats = _gc_stats(port, key)
            assert gc_stats[1] == 0 and gc_stats[3] == 0
            assert gc_stats[2] == 1 and gc_stats[4] == 1
            assert _stats(port, key)[:2] == (1, 1)

            # Repeated clean direct-owner churn reaches a state plateau rather
            # than growing ranks, epochs, owners, or nodes monotonically.
            plateau_samples = []
            for offset in range(64):
                worker = 1300 + offset
                owner = 130_000 + offset
                assert (
                    _register_worker_ranks_leased(
                        port, key, worker, owner, 120_000, [0]
                    )
                    == b"OK"
                )
                assert (
                    _apply_owned(
                        port,
                        key,
                        owner,
                        _event(
                            STORE,
                            worker,
                            0,
                            1,
                            blocks=[(13000 + offset, 200 + offset)],
                        ),
                    )
                    == b"OK"
                )
                assert _unregister_worker(port, key, worker, owner) == b"OK"
                _drain_gc(port, key)
                assert _stats(port, key)[:2] == (1, 1)
                if offset in (31, 63):
                    plateau_samples.append(_memory_usage(port, key))
            assert len(plateau_samples) == 2
            assert plateau_samples[1] <= plateau_samples[0] + 1024, plateau_samples

            # AOF restart preserves provenance, watermark fences, and already
            # compacted topology. The retained legacy state remains explicit.
            assert (
                _command(port, b"BGREWRITEAOF")
                == b"Background append only file rewriting started"
            )
            time.sleep(0.2)
            incremental_key = b"bounded-gc-incremental-aof"
            assert (
                _register_worker_ranks_leased(
                    port, incremental_key, 1500, 150_000, 120_000, [0]
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    incremental_key,
                    150_000,
                    _event(STORE, 1500, 0, 1, blocks=[(15001, 401)]),
                )
                == b"OK"
            )
            assert (
                _unregister_worker(port, incremental_key, 1500, 150_000)
                == b"OK"
            )
            _drain_gc(port, incremental_key, budget=2)
            assert _stats(port, incremental_key)[:2] == (0, 0)
        finally:
            _stop(process, port)

        process = _start(server, module, directory, port, appendonly=True)
        try:
            assert _stats(port, key)[:2] == (1, 1)
            assert _stats(port, b"bounded-gc-incremental-aof")[:2] == (0, 0)
            assert _gc_stats(port, key)[2:5] == (1, 0, 1)
            assert _match(
                _command(
                    port,
                    b"DYNKV.MATCH",
                    key,
                    struct.pack("!BIQ", WIRE_VERSION, 1, 127),
                )
            ) == {}
            # Internal apply commands are never client-callable.
            try:
                _command(port, b"DYNKV.GC_APPLY", key, b"invalid")
            except RuntimeError as error:
                assert "INTERNAL_ONLY" in str(error)
            else:
                raise AssertionError("internal GC plan must reject clients")
        finally:
            _stop(process, port)
