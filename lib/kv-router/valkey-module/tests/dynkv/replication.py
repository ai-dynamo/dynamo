# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
from pathlib import Path

from .fences import _assert_full_worker_two, _wait_for_replica
from .support import (
    CLEAR,
    REMOVE,
    STORE,
    _command,
    _command_and_wait,
    _event,
    _find_free_port,
    _rank_generation,
    _replace_snapshot,
    _start,
    _stop,
)



def run_replication(server: str, module: str) -> None:
    """Exercise native module command replication and synchronous WAIT."""
    with tempfile.TemporaryDirectory(prefix="dynkv-replication-") as path:
        directory = Path(path)
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
        try:
            _wait_for_replica(replica_port)
            root = [(101, 11), (102, 12)]
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.APPLY",
                b"router-index",
                _event(STORE, 2, 0, 1, blocks=root),
            )
            assert mutation == b"OK"
            assert acknowledged == b"1"
            _assert_full_worker_two(primary_port)
            _assert_full_worker_two(replica_port)
            assert (
                _command(
                    primary_port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(STORE, 2, 0, 1, blocks=root),
                )
                == b"NOOP"
            )
            _assert_full_worker_two(replica_port)

            # A client that lost its APPLY/WAIT response may retry and see
            # NOOP even though the original primary mutation was ambiguous.
            # The replicated barrier gives that retry a new connection-local
            # offset for WAIT, proving all preceding module mutations reached
            # the replica without replicating every ordinary NOOP.
            barrier, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.BARRIER",
                b"router-index",
            )
            assert barrier == b"OK"
            assert acknowledged == b"1"
            _assert_full_worker_two(replica_port)

            # Conditional rank replacement is a normal replicated mutation.
            # The replica must retain the same fence generation and snapshot.
            fence_key = b"replicated-generation-fence"
            fence_root = _event(STORE, 44, 1, 1, blocks=[(801, 81)])
            mutation, acknowledged = _command_and_wait(
                primary_port, b"DYNKV.APPLY", fence_key, fence_root
            )
            assert mutation == b"OK"
            assert acknowledged == b"1"
            assert _rank_generation(primary_port, fence_key, 44, 1) == 1
            assert _rank_generation(replica_port, fence_key, 44, 1) == 1
            snapshot = _replace_snapshot([fence_root])
            mutation, acknowledged = _command_and_wait(
                primary_port,
                b"DYNKV.REPLACE_RANK_IF_GENERATION",
                fence_key,
                struct.pack("!Q", 44),
                struct.pack("!I", 1),
                struct.pack("!Q", 1),
                snapshot,
            )
            assert struct.unpack("!Q", mutation)[0] == 2
            assert acknowledged == b"1"
            assert _rank_generation(primary_port, fence_key, 44, 1) == 2
            assert _rank_generation(replica_port, fence_key, 44, 1) == 2

            # A late replica receives the native type through an RDB full sync,
            # not the preceding command stream. Its worker epoch/ticket must
            # survive that path too.
            late_replica_port = _find_free_port()
            late_replica = _start(
                server,
                module,
                directory / "late-replica",
                late_replica_port,
                primary_port=primary_port,
            )
            try:
                _wait_for_replica(late_replica_port)
                assert _rank_generation(late_replica_port, fence_key, 44, 1) == 2
            finally:
                _stop(late_replica, late_replica_port)

            # Invalid payloads must neither mutate nor replicate. In
            # particular, a trailing byte on clear/remove used to be accepted
            # after it had already changed the primary index.
            for malformed in (
                _event(CLEAR, 2, 0, 2) + b"trailing",
                _event(REMOVE, 2, 0, 2, blocks=[(102, 0)]) + b"trailing",
                _event(99, 2, 0, 2),
            ):
                try:
                    _command(primary_port, b"DYNKV.APPLY", b"router-index", malformed)
                except RuntimeError as error:
                    assert "DYNKV_INVALID_EVENT" in str(error)
                else:
                    raise AssertionError("malformed event must fail")
                _assert_full_worker_two(primary_port)
                _assert_full_worker_two(replica_port)
        finally:
            _stop(replica, replica_port)
            _stop(primary, primary_port)
