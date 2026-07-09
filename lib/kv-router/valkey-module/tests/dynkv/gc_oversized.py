# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
import time
from pathlib import Path

from .support import (
    _command_with_timeout,
    _find_free_port,
    _integer_array_command_with_timeout,
    _large_store_event,
    _register_worker_ranks_leased,
    _start,
    _stop,
    _unregister_worker,
)



def run_oversized_chunked_gc(server: str, module: str) -> None:
    """A legal epoch larger than the maximum GC budget must still advance."""
    with tempfile.TemporaryDirectory(prefix="dynkv-oversized-gc-") as path:
        directory = Path(path)
        port = _find_free_port()
        key = b"oversized-chunked-gc"
        worker, owner = 2200, 220_001
        process = _start(server, module, directory, port, appendonly=False)
        try:
            assert (
                _register_worker_ranks_leased(
                    port, key, worker, owner, 120_000, [0]
                )
                == b"OK"
            )
            # The old all-at-once cost was block_count + worker/rank overhead,
            # strictly above DYNKV_MAX_GC_ITEMS for this one legal STORE.
            event = _large_store_event(worker, 0, 1, 1_048_576)
            assert (
                _command_with_timeout(
                    port,
                    60,
                    b"DYNKV.APPLY_OWNED",
                    key,
                    struct.pack("!Q", owner),
                    event,
                )
                == b"OK"
            )
            del event
            unregister_started = time.monotonic()
            assert _unregister_worker(port, key, worker, owner) == b"OK"
            assert time.monotonic() - unregister_started < 2

            # Unregister only fences and begins cleanup, so the next small tick does
            # real bounded work instead of rescanning a million live nodes.
            result = _integer_array_command_with_timeout(
                port,
                60,
                b"DYNKV.GC",
                key,
                b"CURRENT",
                struct.pack("!I", 17),
            )
            assert result[0] <= 17 and result[1] <= 17
            assert result[2] == 17, result
            try:
                _register_worker_ranks_leased(
                    port, key, worker, 220_002, 10_000, [0]
                )
            except RuntimeError as error:
                assert "DYNKV_WORKER_CLEANUP_PENDING" in str(error)
            else:
                raise AssertionError("successor raced oversized partial cleanup")
        finally:
            _stop(process, port)
