# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
import time
from pathlib import Path

from .fences import _wait_for_replica
from .support import (
    ADMISSION_VERSION,
    STORE,
    WIRE_VERSION,
    _admission_stats,
    _apply_owned,
    _command,
    _command_and_wait,
    _event,
    _find_free_port,
    _integer_array_command_and_wait,
    _leased_registration_payload,
    _lifecycle_stats,
    _match,
    _parse_reserve,
    _rank_generation,
    _register_worker,
    _register_worker_ranks,
    _register_worker_ranks_leased,
    _registration_generation,
    _registration_ranks_payload,
    _release_request,
    _replace_snapshot,
    _reserve_request,
    _start,
    _stats,
    _stop,
    _worker_lease_control_payload,
)



def run_admission_replication(server: str, module: str) -> None:
    with tempfile.TemporaryDirectory(prefix="dynkv-admission-replication-") as path:
        directory = Path(path)
        primary_port = _find_free_port()
        replica_port = _find_free_port()
        primary = _start(
            server, module, directory / "primary", primary_port, strict_replication=True
        )
        replica = _start(
            server,
            module,
            directory / "replica",
            replica_port,
            primary_port=primary_port,
        )
        try:
            _wait_for_replica(replica_port)

            # Three ranks replicate as one verbatim module command. The
            # replica's mutation counter is one, not one per rank, proving the
            # batch was applied as a single persisted transition.
            batch_key = b"registration-replication-index"
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.REGISTER_WORKER_RANKS",
                batch_key,
                struct.pack("!Q", 909),
                _registration_ranks_payload([0, 1, 2]),
            )
            assert mutation == b"OK" and acknowledged == b"1"
            assert _stats(primary_port, batch_key) == (0, 3, 1)
            assert _stats(replica_port, batch_key) == (0, 3, 1)
            assert (
                _register_worker_ranks(primary_port, batch_key, 909, [2, 1, 0])
                == b"NOOP"
            )
            assert _stats(replica_port, batch_key) == (0, 3, 1)

            # Worker-owned registration, heartbeat, owner-aware events, and
            # graceful unregister all replicate through deterministic internal
            # apply records carrying the primary's absolute time.
            lease_key = b"worker-lease-replication-index"
            lease_worker = 908
            lease_owner = 90_001
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.REGISTER_WORKER_RANKS",
                lease_key,
                struct.pack("!Q", lease_worker),
                _leased_registration_payload(
                    lease_owner,
                    5_000,
                    [0, 1],
                    _registration_generation(primary_port, lease_key, lease_worker),
                ),
            )
            assert mutation == b"OK" and acknowledged == b"1"
            assert _lifecycle_stats(primary_port, lease_key) == (1, 0, 0)
            assert _lifecycle_stats(replica_port, lease_key) == (1, 0, 0)
            lease_event = _event(STORE, lease_worker, 0, 1, blocks=[(9081, 88)])
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.APPLY_OWNED",
                lease_key,
                struct.pack("!Q", lease_owner),
                lease_event,
            )
            assert mutation == b"OK" and acknowledged == b"1"
            lease_match = struct.pack("!BIQ", WIRE_VERSION, 1, 88)
            assert _match(
                _command(primary_port, b"DYNKV.MATCH", lease_key, lease_match)
            ) == {(lease_worker, 0): (1, 9081)}
            assert _match(
                _command(primary_port, b"DYNKV.MATCH_PRIMARY", lease_key, lease_match)
            ) == {(lease_worker, 0): (1, 9081)}
            assert _match(
                _command(replica_port, b"DYNKV.MATCH", lease_key, lease_match)
            ) == {(lease_worker, 0): (1, 9081)}
            try:
                _command(replica_port, b"DYNKV.MATCH_PRIMARY", lease_key, lease_match)
            except RuntimeError as error:
                assert "DYNKV_NOT_PRIMARY" in str(error)
            else:
                raise AssertionError("MATCH_PRIMARY must reject replica reads")
            # Ordinary MATCH remains replica-readable for state verification.
            assert _match(
                _command(replica_port, b"DYNKV.MATCH", lease_key, lease_match)
            ) == {(lease_worker, 0): (1, 9081)}
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.RENEW_WORKER_LEASE",
                lease_key,
                _worker_lease_control_payload(lease_worker, lease_owner, 5_000),
            )
            assert mutation == b"OK" and acknowledged == b"1"
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.UNREGISTER_WORKER",
                lease_key,
                _worker_lease_control_payload(lease_worker, lease_owner),
            )
            assert mutation == b"OK" and acknowledged == b"1"
            assert _lifecycle_stats(primary_port, lease_key) == (0, 0, 1)
            assert _lifecycle_stats(replica_port, lease_key) == (0, 0, 1)
            assert (
                _match(_command(replica_port, b"DYNKV.MATCH", lease_key, lease_match))
                == {}
            )

            # A lease that expires without unregister is reclaimed atomically
            # by a successor owner on both primary and replica.
            crash_worker = 907
            crash_owner = 90_101
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.REGISTER_WORKER_RANKS",
                lease_key,
                struct.pack("!Q", crash_worker),
                _leased_registration_payload(
                    crash_owner,
                    80,
                    [3],
                    _registration_generation(primary_port, lease_key, crash_worker),
                ),
            )
            assert mutation == b"OK" and acknowledged == b"1"
            crash_event = _event(STORE, crash_worker, 3, 1, blocks=[(9071, 87)])
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.APPLY_OWNED",
                lease_key,
                struct.pack("!Q", crash_owner),
                crash_event,
            )
            assert mutation == b"OK" and acknowledged == b"1"
            time.sleep(0.12)
            successor_owner = 90_102
            try:
                _register_worker_ranks_leased(
                    primary_port,
                    lease_key,
                    crash_worker,
                    successor_owner,
                    5_000,
                    [3],
                )
            except RuntimeError as error:
                assert "DYNKV_WORKER_CLEANUP_PENDING" in str(error)
            else:
                raise AssertionError("successor must wait for replicated cleanup")
            for _ in range(12):
                mutation, acknowledged = _integer_array_command_and_wait(
                    primary_port,
                    b"DYNKV.GC",
                    lease_key,
                    b"CURRENT",
                    struct.pack("!I", 16),
                )
                assert mutation[0] <= 16 and mutation[1] <= mutation[0]
                assert acknowledged == b"1"
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.REGISTER_WORKER_RANKS",
                lease_key,
                struct.pack("!Q", crash_worker),
                _leased_registration_payload(
                    successor_owner,
                    5_000,
                    [3],
                    _registration_generation(primary_port, lease_key, crash_worker),
                ),
            )
            assert mutation == b"OK" and acknowledged == b"1"
            assert _rank_generation(
                primary_port, lease_key, crash_worker, 3
            ) == _rank_generation(replica_port, lease_key, crash_worker, 3)
            try:
                _apply_owned(primary_port, lease_key, crash_owner, crash_event)
            except RuntimeError as error:
                assert "DYNKV_STALE_WORKER_OWNER" in str(error)
            else:
                raise AssertionError("expired replicated owner must remain fenced")

            key = b"admission-replication-index"
            for worker in (910, 911):
                mutation, acknowledged = _command_and_wait(
                    primary_port,
                    b"DYNKV.REGISTER_WORKER",
                    key,
                    struct.pack("!Q", worker),
                    struct.pack("!I", 0),
                )
                assert mutation == b"OK" and acknowledged == b"1"
                mutation, acknowledged = _command_and_wait(
                    primary_port,
                    b"DYNKV.APPLY",
                    key,
                    _event(STORE, worker, 0, 1, blocks=[(9101, 91)]),
                )
                assert mutation == b"OK" and acknowledged == b"1"
            payload = _reserve_request(
                b"prefill", 1, 2, 60_000, [91], [(910, 0, 1), (911, 0, 1)]
            )
            mutation, acknowledged = _command_and_wait(
                primary_port, b"DYNKV.SELECT_RESERVE", key, payload
            )
            reservation = _parse_reserve(mutation)
            assert reservation is not None and acknowledged == b"1"
            assert _admission_stats(primary_port, key) == 1
            assert _admission_stats(replica_port, key) == 1

            late_port = _find_free_port()
            late = _start(
                server, module, directory / "late", late_port, primary_port=primary_port
            )
            try:
                _wait_for_replica(late_port)
                assert _admission_stats(late_port, key) == 1
                assert _stats(late_port, batch_key) == (0, 3, 1)
                assert _lifecycle_stats(late_port, lease_key)[0] == 1
            finally:
                _stop(late, late_port)

            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.RELEASE",
                key,
                _release_request(b"prefill", 1, 2, reservation["expires"]),
            )
            assert mutation == bytes((ADMISSION_VERSION, 1)) and acknowledged == b"1"
            assert _admission_stats(primary_port, key) == 0
            assert _admission_stats(replica_port, key) == 0

            # A successful reserve that reaps an expired predecessor must
            # replicate the cleanup before the new reservation. Otherwise the
            # replica rejects the reserve's active-at-grant fence and retains
            # the expired identity indefinitely.
            expiring_payload = _reserve_request(
                b"expiry-replace", 20, 21, 30, [91], [(910, 0, 1)]
            )
            mutation, acknowledged = _command_and_wait(
                primary_port, b"DYNKV.SELECT_RESERVE", key, expiring_payload
            )
            assert _parse_reserve(mutation) is not None and acknowledged == b"1"
            time.sleep(0.05)
            successor_payload = _reserve_request(
                b"expiry-replace", 22, 23, 60_000, [91], [(910, 0, 1)]
            )
            mutation, acknowledged = _command_and_wait(
                primary_port, b"DYNKV.SELECT_RESERVE", key, successor_payload
            )
            successor = _parse_reserve(mutation)
            assert successor is not None and acknowledged == b"1"
            assert _stats(primary_port, key) == _stats(replica_port, key)
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.RELEASE",
                key,
                _release_request(
                    b"expiry-replace", 22, 23, successor["expires"]
                ),
            )
            assert mutation == bytes((ADMISSION_VERSION, 1)) and acknowledged == b"1"
            assert _admission_stats(primary_port, key) == 0
            assert _admission_stats(replica_port, key) == 0

            replace_payload = _reserve_request(
                b"replace", 3, 4, 60_000, [91], [(911, 0, 1)]
            )
            mutation, acknowledged = _command_and_wait(
                primary_port, b"DYNKV.SELECT_RESERVE", key, replace_payload
            )
            assert _parse_reserve(mutation) is not None and acknowledged == b"1"
            replace_generation = _rank_generation(primary_port, key, 911, 0)
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.REPLACE_RANK_IF_GENERATION",
                key,
                struct.pack("!Q", 911),
                struct.pack("!I", 0),
                struct.pack("!Q", replace_generation),
                _replace_snapshot([_event(STORE, 911, 0, 1, blocks=[(9101, 91)])]),
            )
            assert len(mutation) == 8 and acknowledged == b"1"
            assert _admission_stats(primary_port, key) == 0
            assert _admission_stats(replica_port, key) == 0

            # Lifecycle retirement is replicated verbatim, including lease
            # revocation and the worker-wide admission fence.
            retire_payload = _reserve_request(
                b"retire", 5, 6, 60_000, [91], [(910, 0, 1)]
            )
            mutation, acknowledged = _command_and_wait(
                primary_port, b"DYNKV.SELECT_RESERVE", key, retire_payload
            )
            assert _parse_reserve(mutation) is not None and acknowledged == b"1"
            assert _admission_stats(primary_port, key) == 1
            assert _admission_stats(replica_port, key) == 1
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.REMOVE_WORKER_ALL",
                key,
                struct.pack("!Q", 910),
            )
            assert mutation == b"OK" and acknowledged == b"1"
            assert _admission_stats(primary_port, key) == 0
            assert _admission_stats(replica_port, key) == 0
            try:
                _register_worker(primary_port, key, 910, 7)
            except RuntimeError as error:
                assert "DYNKV_WORKER_RETIRED" in str(error)
            else:
                raise AssertionError(
                    "replicated all-ranks retirement must preserve the fence"
                )
        finally:
            _stop(replica, replica_port)
            _stop(primary, primary_port)
