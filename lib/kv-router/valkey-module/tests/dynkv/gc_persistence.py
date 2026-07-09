# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
import time
from pathlib import Path

from .fences import _wait_for_aof_rewrite
from .support import (
    ADMISSION_NO_CAPACITY,
    ADMISSION_VERSION,
    STORE,
    _admission_stats,
    _advance_to_partial_lease_cleanup,
    _apply_owned,
    _command,
    _drain_gc,
    _event,
    _find_free_port,
    _gc_current,
    _rank_generation,
    _register_worker,
    _register_worker_ranks_leased,
    _registration_ranks_payload,
    _renew_worker_lease,
    _replace_rank_if_generation,
    _reserve,
    _reserve_request,
    _start,
    _stop,
    _unregister_worker,
)



def run_chunked_gc_persistence(server: str, module: str) -> None:
    """Resume semantic-marker cleanup through RDB and AOF rewrite/replay."""

    def expect_pending(call) -> None:
        try:
            call()
        except RuntimeError as error:
            assert "DYNKV_WORKER_CLEANUP_PENDING" in str(error)
        else:
            raise AssertionError("mutation must be fenced during partial cleanup")

    with tempfile.TemporaryDirectory(prefix="dynkv-chunked-gc-rdb-") as path:
        directory = Path(path)
        port = _find_free_port()
        key = b"chunked-gc-rdb"
        worker, owner = 2100, 210_001
        process = _start(server, module, directory, port, appendonly=False)
        try:
            assert (
                _register_worker_ranks_leased(
                    port, key, worker, owner, 60, [0, 1]
                )
                == b"OK"
            )
            for rank in (0, 1):
                assert (
                    _apply_owned(
                        port,
                        key,
                        owner,
                        _event(
                            STORE,
                            worker,
                            rank,
                            1,
                            blocks=[
                                (21000 + rank * 10, 800 + rank * 10),
                                (21001 + rank * 10, 801 + rank * 10),
                            ],
                        ),
                    )
                    == b"OK"
                )
            time.sleep(0.09)
            _advance_to_partial_lease_cleanup(port, key)

            expect_pending(
                lambda: _register_worker_ranks_leased(
                    port, key, worker, 210_002, 10_000, [0, 1]
                )
            )
            expect_pending(
                lambda: _command(
                    port,
                    b"DYNKV.APPLY",
                    key,
                    _event(STORE, worker, 0, 2, blocks=[(21099, 899)]),
                )
            )
            expect_pending(
                lambda: _command(
                    port,
                    b"DYNKV.RESET_WORKER",
                    key,
                    struct.pack("!Q", worker),
                    struct.pack("!I", 0),
                )
            )
            expect_pending(
                lambda: _command(
                    port,
                    b"DYNKV.REMOVE_WORKER",
                    key,
                    struct.pack("!Q", worker),
                    struct.pack("!I", 0),
                )
            )
            expect_pending(
                lambda: _command(
                    port,
                    b"DYNKV.REMOVE_WORKER_ALL",
                    key,
                    struct.pack("!Q", worker),
                )
            )
            expect_pending(lambda: _register_worker(port, key, worker, 0))
            expect_pending(
                lambda: _command(
                    port,
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    key,
                    struct.pack("!Q", worker),
                    _registration_ranks_payload([0, 1]),
                )
            )
            expect_pending(
                lambda: _replace_rank_if_generation(
                    port,
                    key,
                    worker,
                    0,
                    _rank_generation(port, key, worker, 0),
                    [_event(STORE, worker, 0, 1, blocks=[(21000, 800)])],
                )
            )
            for control in (
                lambda: _renew_worker_lease(port, key, worker, owner, 10_000),
                lambda: _unregister_worker(port, key, worker, owner),
            ):
                try:
                    control()
                except RuntimeError as error:
                    assert "DYNKV_STALE_WORKER_OWNER" in str(error)
                else:
                    raise AssertionError("old owner control must be fenced")
            assert _command(port, b"SAVE") == b"OK"
        finally:
            _stop(process, port)

        process = _start(server, module, directory, port, appendonly=False)
        try:
            expect_pending(
                lambda: _register_worker_ranks_leased(
                    port, key, worker, 210_002, 10_000, [0, 1]
                )
            )
            _drain_gc(port, key, budget=1)
            assert (
                _register_worker_ranks_leased(
                    port, key, worker, 210_002, 10_000, [0, 1]
                )
                == b"OK"
            )

            # Existing reservation replay must never dispatch to an expired
            # owner while a large epoch is waiting for chunked cleanup.
            stale_key = b"chunked-gc-stale-reservation"
            stale_worker, stale_owner = 2101, 210_003
            assert (
                _register_worker_ranks_leased(
                    port, stale_key, stale_worker, stale_owner, 60, [0]
                )
                == b"OK"
            )
            blocks = [(21100 + i, 1000 + i) for i in range(300)]
            assert (
                _apply_owned(
                    port,
                    stale_key,
                    stale_owner,
                    _event(STORE, stale_worker, 0, 1, blocks=blocks),
                )
                == b"OK"
            )
            reserve_payload = _reserve_request(
                b"stale-owner", 210_100, 210_101, 60_000, [], [(stale_worker, 0, 1)]
            )
            _, reservation = _reserve(port, stale_key, reserve_payload)
            assert reservation is not None
            time.sleep(0.09)
            replay, reservation = _reserve(port, stale_key, reserve_payload)
            assert replay == bytes((ADMISSION_VERSION, ADMISSION_NO_CAPACITY))
            assert reservation is None and _admission_stats(port, stale_key) == 0
        finally:
            _stop(process, port)

    with tempfile.TemporaryDirectory(prefix="dynkv-chunked-gc-aof-") as path:
        directory = Path(path)
        port = _find_free_port()
        key = b"chunked-gc-aof"
        worker, owner = 2102, 210_004
        process = _start(server, module, directory, port, appendonly=True)
        try:
            assert (
                _register_worker_ranks_leased(
                    port, key, worker, owner, 60, [0, 1]
                )
                == b"OK"
            )
            for rank in (0, 1):
                assert (
                    _apply_owned(
                        port,
                        key,
                        owner,
                        _event(
                            STORE,
                            worker,
                            rank,
                            1,
                            blocks=[
                                (21200 + rank * 10 + offset, 1200 + rank * 10 + offset)
                                for offset in range(3)
                            ],
                        ),
                    )
                    == b"OK"
                )
            time.sleep(0.09)
            _advance_to_partial_lease_cleanup(port, key)
            _wait_for_aof_rewrite(port)
            for _ in range(128):
                result = _gc_current(port, key, 1)
                if result[2] != 0:
                    break
            else:
                raise AssertionError("incremental owner chunk did not commit")
        finally:
            _stop(process, port)

        process = _start(server, module, directory, port, appendonly=True)
        try:
            expect_pending(
                lambda: _register_worker_ranks_leased(
                    port, key, worker, 210_005, 10_000, [0, 1]
                )
            )
            _drain_gc(port, key, budget=1)
            assert (
                _register_worker_ranks_leased(
                    port, key, worker, 210_005, 10_000, [0, 1]
                )
                == b"OK"
            )
        finally:
            _stop(process, port)
