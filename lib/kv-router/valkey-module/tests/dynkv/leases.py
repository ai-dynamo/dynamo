# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
import time
from pathlib import Path

from .support import (
    ADMISSION_NO_CAPACITY,
    ADMISSION_VERSION,
    CLEAR,
    STORE,
    WIRE_VERSION,
    _admission_stats,
    _apply_owned,
    _command,
    _drain_gc,
    _event,
    _find_free_port,
    _gc_current,
    _leased_registration_payload,
    _lifecycle_stats,
    _match,
    _rank_generation,
    _register_worker,
    _register_worker_ranks_leased,
    _renew_worker_lease,
    _reserve,
    _reserve_request,
    _start,
    _stats,
    _stop,
    _unregister_worker,
)



def run_worker_owner_leases(server: str, module: str) -> None:
    """Exercise owner fencing, crash expiry, reclamation, and RDB restore."""
    with tempfile.TemporaryDirectory(prefix="dynkv-worker-lease-") as path:
        directory = Path(path)
        port = _find_free_port()
        key = b"worker-owner-lease-index"
        process = _start(server, module, directory, port, appendonly=False)
        try:
            invalid_key = b"invalid-worker-owner-lease-index"
            invalid_leased_payloads = (
                _leased_registration_payload(0, 1_000, [0]),
                _leased_registration_payload(1, 0, [0]),
                _leased_registration_payload(1, 1_000, [0, 0]),
                _leased_registration_payload(1, 1_000, []),
            )
            for payload in invalid_leased_payloads:
                try:
                    _command(
                        port,
                        b"DYNKV.REGISTER_WORKER_RANKS",
                        invalid_key,
                        struct.pack("!Q", 939),
                        payload,
                    )
                except RuntimeError as error:
                    assert "DYNKV_INVALID_REGISTRATION" in str(error)
                else:
                    raise AssertionError("invalid leased rank set must fail")
                assert int(_command(port, b"EXISTS", invalid_key)) == 0

            # Rank and worker-wide retirement are permanent fences, unlike an
            # owner unregister/expiry. One fenced member rejects the complete
            # leased set before an unseen sibling is created.
            fence_key = b"worker-owner-retirement-fence-index"
            assert _register_worker(port, fence_key, 938, 9) == b"OK"
            assert (
                _command(
                    port,
                    b"DYNKV.REMOVE_WORKER",
                    fence_key,
                    struct.pack("!Q", 938),
                    struct.pack("!I", 9),
                )
                == b"OK"
            )
            before_rank_fence = _stats(port, fence_key)
            try:
                _register_worker_ranks_leased(
                    port, fence_key, 938, 9_001, 1_000, [8, 9]
                )
            except RuntimeError as error:
                assert "DYNKV_WORKER_RETIRED" in str(error)
            else:
                raise AssertionError("retired rank must reject leased registration")
            assert _stats(port, fence_key) == before_rank_fence
            assert (
                _command(
                    port,
                    b"DYNKV.REMOVE_WORKER_ALL",
                    fence_key,
                    struct.pack("!Q", 937),
                )
                == b"OK"
            )
            before_worker_fence = _stats(port, fence_key)
            try:
                _register_worker_ranks_leased(
                    port, fence_key, 937, 9_002, 1_000, [0, 1]
                )
            except RuntimeError as error:
                assert "DYNKV_WORKER_RETIRED" in str(error)
            else:
                raise AssertionError(
                    "worker retirement must reject leased registration"
                )
            assert _stats(port, fence_key) == before_worker_fence

            worker = 940
            owner = 10_001
            rival = 10_002
            assert (
                _register_worker_ranks_leased(port, key, worker, owner, 5_000, [0, 1])
                == b"OK"
            )
            assert _lifecycle_stats(port, key) == (1, 0, 0)
            try:
                _apply_owned(
                    port,
                    key,
                    owner,
                    _event(STORE, worker, 2, 1, blocks=[(9400, 40)]),
                )
            except RuntimeError as error:
                assert "DYNKV_UNREGISTERED_WORKER_RANK" in str(error)
            else:
                raise AssertionError("an owner must not publish an unregistered rank")
            assert (
                _apply_owned(
                    port,
                    key,
                    owner,
                    _event(STORE, worker, 0, 1, blocks=[(9401, 41)]),
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    owner,
                    _event(STORE, worker, 1, 1, blocks=[(9401, 41)]),
                )
                == b"OK"
            )
            request = struct.pack("!BIQ", WIRE_VERSION, 1, 41)
            assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {
                (worker, 0): (1, 9401),
                (worker, 1): (1, 9401),
            }

            # Same-owner registration is an exact rank-set reconciliation,
            # not an additive update. Omitted ranks lose membership and their
            # cached owners, then can be added back atomically.
            assert (
                _register_worker_ranks_leased(port, key, worker, owner, 5_000, [0])
                == b"OK"
            )
            assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {
                (worker, 0): (1, 9401)
            }
            assert _lifecycle_stats(port, key) == (1, 1, 0)
            try:
                _apply_owned(
                    port,
                    key,
                    owner,
                    _event(STORE, worker, 1, 2, blocks=[(9402, 42)]),
                )
            except RuntimeError as error:
                assert "DYNKV_UNREGISTERED_WORKER_RANK" in str(error)
            else:
                raise AssertionError("an omitted rank must not publish owned events")
            assert (
                _register_worker_ranks_leased(port, key, worker, owner, 5_000, [0, 1])
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    owner,
                    _event(STORE, worker, 1, 1, blocks=[(9401, 41)]),
                )
                == b"OK"
            )
            assert _lifecycle_stats(port, key) == (1, 0, 0)

            # Existing single-rank publishers can repeat their own rank, but
            # cannot add membership behind the owner lease.
            assert _register_worker(port, key, worker, 0) == b"NOOP"
            try:
                _register_worker(port, key, worker, 2)
            except RuntimeError as error:
                assert "DYNKV_WORKER_OWNED" in str(error)
            else:
                raise AssertionError("legacy registration must not bypass an owner")

            for stale_call in (
                lambda: _register_worker_ranks_leased(
                    port, key, worker, rival, 5_000, [0, 1]
                ),
                lambda: _renew_worker_lease(port, key, worker, rival, 5_000),
                lambda: _unregister_worker(port, key, worker, rival),
                lambda: _apply_owned(
                    port,
                    key,
                    rival,
                    _event(STORE, worker, 0, 2, blocks=[(9402, 42)]),
                ),
            ):
                try:
                    stale_call()
                except RuntimeError as error:
                    assert "DYNKV_WORKER_OWNED" in str(
                        error
                    ) or "DYNKV_STALE_WORKER_OWNER" in str(error)
                else:
                    raise AssertionError("a competing owner must be fenced")

            assert _renew_worker_lease(port, key, worker, owner, 5_000) == b"OK"
            assert (
                _apply_owned(port, key, owner, _event(CLEAR, worker, 0, 100)) == b"OK"
            )
            assert _unregister_worker(port, key, worker, owner) == b"OK"
            assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {}
            assert _lifecycle_stats(port, key) == (0, 0, 1)
            # Unregister is not retirement: a new process using the same
            # discovery worker ID can claim it after bounded cleanup.
            successor = 10_003
            for _ in range(64):
                try:
                    registered = _register_worker_ranks_leased(
                        port, key, worker, successor, 10_000, [0, 1]
                    )
                except RuntimeError as error:
                    assert "DYNKV_WORKER_CLEANUP_PENDING" in str(error)
                    _gc_current(port, key, 64)
                    continue
                assert registered == b"OK"
                break
            else:
                raise AssertionError("bounded unregister cleanup did not converge")
            assert _lifecycle_stats(port, key) == (1, 0, 0)
            assert (
                _apply_owned(
                    port,
                    key,
                    successor,
                    _event(STORE, worker, 0, 1, blocks=[(9401, 41)]),
                )
                == b"OK"
            )
            assert (
                _apply_owned(port, key, successor, _event(CLEAR, worker, 0, 2)) == b"OK"
            )
            assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {}
            try:
                _apply_owned(
                    port,
                    key,
                    owner,
                    _event(STORE, worker, 0, 2, blocks=[(9402, 42)]),
                )
            except RuntimeError as error:
                assert "DYNKV_STALE_WORKER_OWNER" in str(error)
            else:
                raise AssertionError("the prior owner must stay fenced after reclaim")

            # Simulate a crash: no heartbeat/unregister follows. MATCH filters
            # the absolute expiry immediately; the next admission mutation
            # deterministically reclaims membership and the live reservation.
            crashed_worker = 941
            crashed_owner = 20_001
            assert (
                _register_worker_ranks_leased(
                    port, key, crashed_worker, crashed_owner, 80, [3]
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    crashed_owner,
                    _event(STORE, crashed_worker, 3, 50, blocks=[(9411, 51)]),
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port, key, crashed_owner, _event(CLEAR, crashed_worker, 3, 100)
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    crashed_owner,
                    _event(STORE, crashed_worker, 3, 101, blocks=[(9411, 51)]),
                )
                == b"OK"
            )
            _, crashed_reservation = _reserve(
                port,
                key,
                _reserve_request(
                    b"crash-expiry", 1, 1, 60_000, [], [(crashed_worker, 3, 1)]
                ),
            )
            assert crashed_reservation is not None
            time.sleep(0.12)
            crash_request = struct.pack("!BIQ", WIRE_VERSION, 1, 51)
            assert _match(_command(port, b"DYNKV.MATCH", key, crash_request)) == {}
            # Read-only lifecycle diagnostics use the same absolute-expiry
            # semantics as MATCH even before a write lazily commits cleanup.
            assert _lifecycle_stats(port, key) == (1, 0, 1)
            assert _admission_stats(port, key) == 1
            expired_raw, expired = _reserve(
                port,
                key,
                _reserve_request(
                    b"crash-expiry", 2, 2, 60_000, [], [(crashed_worker, 3, 1)]
                ),
            )
            assert expired is None
            assert expired_raw == bytes((ADMISSION_VERSION, ADMISSION_NO_CAPACITY))
            assert _admission_stats(port, key) == 0
            assert _rank_generation(port, key, crashed_worker, 3) > 0
            active_leases, retained_ranks, _ = _lifecycle_stats(port, key)
            assert active_leases == 1  # successor for worker 940
            assert retained_ranks == 1

            crash_successor = 20_002
            assert (
                _register_worker_ranks_leased(
                    port, key, crashed_worker, crash_successor, 10_000, [3]
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    crash_successor,
                    _event(STORE, crashed_worker, 3, 1, blocks=[(9411, 51)]),
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port, key, crash_successor, _event(CLEAR, crashed_worker, 3, 2)
                )
                == b"OK"
            )
            assert _match(_command(port, b"DYNKV.MATCH", key, crash_request)) == {}
            try:
                _renew_worker_lease(port, key, crashed_worker, crashed_owner, 5_000)
            except RuntimeError as error:
                assert "DYNKV_STALE_WORKER_OWNER" in str(error)
            else:
                raise AssertionError("crashed owner heartbeat must be fenced")

            # Persist one live lease and one lease that expires while the
            # server is down. Both owner metadata and the expiry heap rebuild
            # through the native RDB callback.
            durable_worker = 942
            durable_owner = 30_001
            assert (
                _register_worker_ranks_leased(
                    port, key, durable_worker, durable_owner, 10_000, [4]
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    durable_owner,
                    _event(STORE, durable_worker, 4, 1, blocks=[(9421, 61)]),
                )
                == b"OK"
            )
            offline_worker = 943
            offline_owner = 40_001
            assert (
                _register_worker_ranks_leased(
                    port, key, offline_worker, offline_owner, 100, [5]
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    offline_owner,
                    _event(STORE, offline_worker, 5, 50, blocks=[(9431, 71)]),
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port, key, offline_owner, _event(CLEAR, offline_worker, 5, 100)
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    offline_owner,
                    _event(STORE, offline_worker, 5, 101, blocks=[(9431, 71)]),
                )
                == b"OK"
            )
            assert _command(port, b"SAVE") == b"OK"

            # Internal absolute-time transitions must not be client callable.
            internal_calls = (
                (b"DYNKV.WORKER_LEASE_APPLY", key, b"not-internal"),
                (
                    b"DYNKV.APPLY_OWNED_AT",
                    key,
                    struct.pack("!Q", durable_owner),
                    struct.pack("!Q", 1),
                    _event(STORE, durable_worker, 4, 2, blocks=[(9422, 62)]),
                ),
            )
            for call in internal_calls:
                try:
                    _command(port, *call)
                except RuntimeError as error:
                    assert "INTERNAL_ONLY" in str(error)
                else:
                    raise AssertionError("internal lifecycle apply must reject clients")
        finally:
            _stop(process, port)

        time.sleep(0.15)
        process = _start(server, module, directory, port, appendonly=False)
        try:
            durable_request = struct.pack("!BIQ", WIRE_VERSION, 1, 61)
            assert _match(_command(port, b"DYNKV.MATCH", key, durable_request)) == {
                (942, 4): (1, 9421)
            }
            assert _renew_worker_lease(port, key, 942, 30_001, 10_000) == b"OK"
            offline_request = struct.pack("!BIQ", WIRE_VERSION, 1, 71)
            assert _match(_command(port, b"DYNKV.MATCH", key, offline_request)) == {}
            try:
                _register_worker_ranks_leased(
                    port, key, 943, 40_002, 10_000, [5]
                )
            except RuntimeError as error:
                assert "DYNKV_WORKER_CLEANUP_PENDING" in str(error)
            else:
                raise AssertionError("successor must not race expired-state cleanup")
            _drain_gc(port, key, budget=16)
            assert (
                _register_worker_ranks_leased(port, key, 943, 40_002, 10_000, [5])
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    40_002,
                    _event(STORE, 943, 5, 1, blocks=[(9431, 71)]),
                )
                == b"OK"
            )
            assert _apply_owned(port, key, 40_002, _event(CLEAR, 943, 5, 2)) == b"OK"
            assert _match(_command(port, b"DYNKV.MATCH", key, offline_request)) == {}
            try:
                _apply_owned(
                    port,
                    key,
                    40_001,
                    _event(STORE, 943, 5, 2, blocks=[(9432, 72)]),
                )
            except RuntimeError as error:
                assert "DYNKV_STALE_WORKER_OWNER" in str(error)
            else:
                raise AssertionError("pre-restart owner must remain fenced")
        finally:
            _stop(process, port)

    # Separate incremental-AOF path: no rewrite and no RDB SAVE. Public
    # owner commands append only their deterministic internal apply records,
    # which must be accepted during AOF loading and rebuild the lease heap.
    with tempfile.TemporaryDirectory(prefix="dynkv-worker-lease-aof-") as path:
        directory = Path(path)
        port = _find_free_port()
        key = b"worker-owner-lease-incremental-aof-index"
        process = _start(server, module, directory, port)
        try:
            assert (
                _register_worker_ranks_leased(port, key, 944, 44_001, 120_000, [6])
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    44_001,
                    _event(STORE, 944, 6, 1, blocks=[(9441, 74)]),
                )
                == b"OK"
            )
            assert (
                _register_worker_ranks_leased(port, key, 945, 45_001, 120_000, [7])
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    key,
                    45_001,
                    _event(STORE, 945, 7, 50, blocks=[(9451, 75)]),
                )
                == b"OK"
            )
            assert _apply_owned(port, key, 45_001, _event(CLEAR, 945, 7, 100)) == b"OK"
            assert _unregister_worker(port, key, 945, 45_001) == b"OK"
        finally:
            _stop(process, port)

        process = _start(server, module, directory, port)
        try:
            request = struct.pack("!BIQ", WIRE_VERSION, 1, 74)
            assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {
                (944, 6): (1, 9441)
            }
            # Worker 944 remains live; worker 945's AOF-replayed unregister
            # remains fenced in bounded-cleanup state.
            assert _lifecycle_stats(port, key) == (1, 0, 1)
            try:
                _register_worker_ranks_leased(port, key, 944, 44_002, 120_000, [6])
            except RuntimeError as error:
                assert "DYNKV_WORKER_OWNED" in str(error)
            else:
                raise AssertionError("incremental AOF must retain owner fencing")
            for _ in range(64):
                try:
                    registered = _register_worker_ranks_leased(
                        port, key, 945, 45_002, 120_000, [7]
                    )
                except RuntimeError as error:
                    assert "DYNKV_WORKER_CLEANUP_PENDING" in str(error)
                    _gc_current(port, key, 64)
                    continue
                assert registered == b"OK"
                break
            else:
                raise AssertionError("AOF-restored unregister cleanup did not converge")
            assert (
                _apply_owned(
                    port,
                    key,
                    45_002,
                    _event(STORE, 945, 7, 1, blocks=[(9451, 75)]),
                )
                == b"OK"
            )
            assert _apply_owned(port, key, 45_002, _event(CLEAR, 945, 7, 2)) == b"OK"
            request = struct.pack("!BIQ", WIRE_VERSION, 1, 75)
            assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {}
        finally:
            _stop(process, port)
