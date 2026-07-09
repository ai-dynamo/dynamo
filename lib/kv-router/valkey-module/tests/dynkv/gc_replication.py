# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
import time
from pathlib import Path

from .fences import _wait_for_replica
from .support import (
    STORE,
    _admission_stats,
    _command_and_wait,
    _event,
    _find_free_port,
    _gc,
    _gc_stats,
    _integer_array_command_and_wait,
    _leased_registration_payload,
    _registration_generation,
    _reserve,
    _reserve_request,
    _start,
    _stats,
    _stop,
    _worker_lease_control_payload,
)



def run_gc_replication(server: str, module: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        primary_port = _find_free_port()
        replica_port = _find_free_port()
        primary = _start(
            server,
            module,
            directory / "primary",
            primary_port,
            strict_replication=True,
        )
        replica = _start(
            server,
            module,
            directory / "replica",
            replica_port,
            primary_port=primary_port,
        )
        key = b"gc-replication-index"
        try:
            _wait_for_replica(replica_port)
            for offset in range(8):
                worker = 1400 + offset
                owner = 140_000 + offset
                mutation, acknowledged = _command_and_wait(
                    primary_port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    key,
                    struct.pack("!Q", worker),
                    _leased_registration_payload(
                        owner,
                        120_000,
                        [0],
                        _registration_generation(primary_port, key, worker),
                    ),
                )
                assert mutation == b"OK" and acknowledged == b"1"
                mutation, acknowledged = _command_and_wait(
                    primary_port,
                    b"DYNKV.APPLY_OWNED",
                    key,
                    struct.pack("!Q", owner),
                    _event(
                        STORE,
                        worker,
                        0,
                        1,
                        blocks=[(14000 + offset, 300 + offset)],
                    ),
                )
                assert mutation == b"OK" and acknowledged == b"1"
                mutation, acknowledged = _command_and_wait(
                    primary_port,
                    b"DYNKV.UNREGISTER_WORKER",
                    key,
                    _worker_lease_control_payload(worker, owner),
                )
                assert mutation == b"OK" and acknowledged == b"1"

            watermark = _gc_stats(primary_port, key)[0]
            _gc(primary_port, key, watermark, 3)
            for _ in range(100):
                if _stats(replica_port, key) == _stats(primary_port, key):
                    break
                time.sleep(0.02)

            # Full-sync while a two-rank epoch is pending. The late replica
            # can rebuild arrays in a different order; semantic identities
            # plus O(1) swaps must still accept every later exact chunk.
            partial_key = b"gc-replication-partial-lease"
            partial_worker, partial_owner = 1450, 145_000
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.REGISTER_WORKER_RANKS",
                partial_key,
                struct.pack("!Q", partial_worker),
                _leased_registration_payload(
                    partial_owner,
                    80,
                    [0, 1],
                    _registration_generation(
                        primary_port, partial_key, partial_worker
                    ),
                ),
            )
            assert mutation == b"OK" and acknowledged == b"1"
            for rank in (0, 1):
                mutation, acknowledged = _command_and_wait(
                    primary_port,
                    b"DYNKV.APPLY_OWNED",
                    partial_key,
                    struct.pack("!Q", partial_owner),
                    _event(
                        STORE,
                        partial_worker,
                        rank,
                        1,
                        blocks=[
                            (14500 + rank * 10, 400 + rank * 10),
                            (14501 + rank * 10, 401 + rank * 10),
                        ],
                    ),
                )
                assert mutation == b"OK" and acknowledged == b"1"
                _, reservation = _reserve(
                    primary_port,
                    partial_key,
                    _reserve_request(
                        f"partial-{rank}".encode(),
                        145_100 + rank,
                        145_200 + rank,
                        60_000,
                        [],
                        [(partial_worker, rank, 1)],
                    ),
                )
                assert reservation is not None
            time.sleep(0.11)
            for _ in range(256):
                result, acknowledged = _integer_array_command_and_wait(
                    primary_port,
                    b"DYNKV.GC",
                    partial_key,
                    b"CURRENT",
                    struct.pack("!I", 1),
                )
                if result[1] != 0:
                    assert int(acknowledged) >= 1
                if result[6] != 0:
                    break
            else:
                raise AssertionError("replicated cleanup BEGIN did not commit")

            # Join during partial compaction. Full-sync/RDB must rebuild child
            # and rank/epoch reference counts so later exact plans still apply.
            late_port = _find_free_port()
            late = _start(
                server,
                module,
                directory / "late",
                late_port,
                primary_port=primary_port,
            )
            try:
                _wait_for_replica(late_port)
                assert _stats(late_port, key) == _stats(primary_port, key)
                assert _gc_stats(late_port, key)[:7] == _gc_stats(
                    primary_port, key
                )[:7]
                assert _stats(late_port, partial_key) == _stats(
                    primary_port, partial_key
                )
                assert _admission_stats(late_port, partial_key) == 2
                assert _gc_stats(late_port, partial_key)[:7] == _gc_stats(
                    primary_port, partial_key
                )[:7]
                idle = 0
                for _ in range(512):
                    result, acknowledged = _integer_array_command_and_wait(
                        primary_port,
                        b"DYNKV.GC",
                        partial_key,
                        b"CURRENT",
                        struct.pack("!I", 1),
                    )
                    if result[1] != 0:
                        assert int(acknowledged) >= 1
                    stats = _gc_stats(primary_port, partial_key)
                    if (
                        stats[1] == 0
                        and stats[3] == 0
                        and stats[6] == 0
                        and _admission_stats(primary_port, partial_key) == 0
                    ):
                        break
                    idle = idle + 1 if result[1] == 0 else 0
                    assert idle < 24, (result, stats)
                else:
                    raise AssertionError("replicated partial cleanup did not drain")
                for _ in range(64):
                    _gc(primary_port, key, watermark, 3)
                    if _stats(primary_port, key)[:2] == (0, 0):
                        break
                assert _stats(primary_port, key)[:2] == (0, 0)
                for _ in range(100):
                    if (
                        _stats(replica_port, key) == _stats(primary_port, key)
                        and _stats(late_port, key) == _stats(primary_port, key)
                    ):
                        break
                    time.sleep(0.02)
                assert _stats(replica_port, key) == _stats(primary_port, key)
                assert _stats(late_port, key) == _stats(primary_port, key)
                assert _gc_stats(primary_port, key)[:7] == _gc_stats(
                    replica_port, key
                )[:7]
                assert _gc_stats(primary_port, key)[:7] == _gc_stats(
                    late_port, key
                )[:7]
                for _ in range(100):
                    if (
                        _stats(replica_port, partial_key)
                        == _stats(primary_port, partial_key)
                        and _stats(late_port, partial_key)
                        == _stats(primary_port, partial_key)
                    ):
                        break
                    time.sleep(0.02)
                assert _stats(primary_port, partial_key)[:2] == (0, 0)
                assert _stats(replica_port, partial_key) == _stats(
                    primary_port, partial_key
                )
                assert _stats(late_port, partial_key) == _stats(
                    primary_port, partial_key
                )
                assert _gc_stats(primary_port, partial_key)[:7] == _gc_stats(
                    replica_port, partial_key
                )[:7]
                assert _gc_stats(primary_port, partial_key)[:7] == _gc_stats(
                    late_port, partial_key
                )[:7]
            finally:
                _stop(late, late_port)
        finally:
            _stop(replica, replica_port)
            _stop(primary, primary_port)
