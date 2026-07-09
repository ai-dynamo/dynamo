# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
from pathlib import Path

from .support import (
    MAX_REGISTRATION_RANKS,
    REGISTRATION_VERSION,
    _command,
    _find_free_port,
    _register_worker,
    _register_worker_ranks,
    _registration_ranks_payload,
    _start,
    _stats,
    _stop,
)



def run_multi_rank_registration(server: str, module: str) -> None:
    """Exercise batch validation, fencing, idempotence, AOF, and compatibility."""
    with tempfile.TemporaryDirectory(prefix="dynkv-registration-") as path:
        directory = Path(path)
        port = _find_free_port()
        key = b"multi-rank-registration-index"
        worker = 930
        process = _start(server, module, directory, port)
        try:
            # Invalid input must be rejected before even materializing the
            # module key. This covers every fixed-width boundary plus duplicate
            # ranks, which are invalid rather than silently normalized.
            malformed_payloads = (
                b"",
                struct.pack("!BI", REGISTRATION_VERSION + 1, 1) + struct.pack("!I", 0),
                struct.pack("!BI", REGISTRATION_VERSION, 0),
                struct.pack("!BI", REGISTRATION_VERSION, 2) + struct.pack("!I", 0),
                _registration_ranks_payload([0]) + b"trailing",
                _registration_ranks_payload([1, 1]),
                struct.pack("!BI", REGISTRATION_VERSION, MAX_REGISTRATION_RANKS + 1),
            )
            for payload in malformed_payloads:
                try:
                    _command(
                        port,
                        b"DYNKV.REGISTER_WORKER_RANKS",
                        key,
                        struct.pack("!Q", worker),
                        payload,
                    )
                except RuntimeError as error:
                    assert "DYNKV_INVALID_REGISTRATION" in str(error)
                else:
                    raise AssertionError("malformed rank registration must fail")
                assert int(_command(port, b"EXISTS", key)) == 0

            # One batch creates all three ranks but is one logical mutation.
            assert _register_worker_ranks(port, key, worker, [9, 1, 5]) == b"OK"
            assert _stats(port, key) == (0, 3, 1)

            # Existing single-rank clients remain compatible with batch-created
            # ranks, and a single-created rank participates in later batches.
            assert _register_worker(port, key, worker, 1) == b"NOOP"
            assert _register_worker(port, key, worker, 7) == b"OK"
            assert _stats(port, key) == (0, 4, 2)
            assert _register_worker_ranks(port, key, worker, [9, 7, 1, 5]) == b"NOOP"
            assert _stats(port, key) == (0, 4, 2)

            # A partially overlapping batch registers only the missing rank,
            # still with one mutation-count increment for the whole command.
            assert _register_worker_ranks(port, key, worker, [5, 6, 7]) == b"OK"
            assert _stats(port, key) == (0, 5, 3)

            # Put a retired rank last in canonical rank order. The missing rank
            # before it must not leak into the index when preflight finds the
            # retirement fence.
            assert _register_worker(port, key, worker, 20) == b"OK"
            assert (
                _command(
                    port,
                    b"DYNKV.REMOVE_WORKER",
                    key,
                    struct.pack("!Q", worker),
                    struct.pack("!I", 20),
                )
                == b"OK"
            )
            before_retired_batch = _stats(port, key)
            assert before_retired_batch == (0, 6, 5)
            try:
                _register_worker_ranks(port, key, worker, [10, 5, 20])
            except RuntimeError as error:
                assert "DYNKV_WORKER_RETIRED" in str(error)
            else:
                raise AssertionError("one retired rank must reject the whole batch")
            assert _stats(port, key) == before_retired_batch
            assert _register_worker(port, key, worker, 5) == b"NOOP"

            # Reset explicitly revives the retired rank but leaves it
            # unregistered. The next batch atomically registers it and the
            # previously absent rank.
            assert (
                _command(
                    port,
                    b"DYNKV.RESET_WORKER",
                    key,
                    struct.pack("!Q", worker),
                    struct.pack("!I", 20),
                )
                == b"OK"
            )
            assert _register_worker_ranks(port, key, worker, [10, 5, 20]) == b"OK"
            assert _stats(port, key) == (0, 7, 7)
            for dp_rank in (5, 10, 20):
                assert _register_worker(port, key, worker, dp_rank) == b"NOOP"

            # REMOVE_WORKER_ALL fences ranks that do not exist yet. A batch
            # containing only unseen ranks must not materialize any of them.
            all_retired_worker = 931
            assert _register_worker(port, key, all_retired_worker, 0) == b"OK"
            assert (
                _command(
                    port,
                    b"DYNKV.REMOVE_WORKER_ALL",
                    key,
                    struct.pack("!Q", all_retired_worker),
                )
                == b"OK"
            )
            before_worker_fence = _stats(port, key)
            assert before_worker_fence == (0, 8, 9)
            try:
                _register_worker_ranks(port, key, all_retired_worker, [1, 2])
            except RuntimeError as error:
                assert "DYNKV_WORKER_RETIRED" in str(error)
            else:
                raise AssertionError("worker-wide retirement must reject the batch")
            assert _stats(port, key) == before_worker_fence
        finally:
            # SHUTDOWN NOSAVE still flushes the append-only command stream;
            # the restart below therefore exercises replay of the batch command.
            _stop(process, port)

        process = _start(server, module, directory, port)
        try:
            assert _stats(port, key) == (0, 8, 9)
            assert _register_worker_ranks(port, key, worker, [5, 10, 20]) == b"NOOP"
            try:
                _register_worker_ranks(port, key, 931, [1, 2])
            except RuntimeError as error:
                assert "DYNKV_WORKER_RETIRED" in str(error)
            else:
                raise AssertionError("AOF replay must preserve the worker-wide fence")
        finally:
            _stop(process, port)
