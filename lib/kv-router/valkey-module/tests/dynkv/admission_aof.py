# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import tempfile
from pathlib import Path

from .fences import _wait_for_aof_rewrite
from .support import (
    STORE,
    WIRE_VERSION,
    _admission_stats,
    _admission_workers,
    _apply_owned,
    _command,
    _event,
    _find_free_port,
    _match,
    _register_worker_ranks_leased,
    _renew_worker_lease,
    _reserve,
    _reserve_request,
    _start,
    _stats,
    _stop,
)



def run_admission_aof_rewrite(server: str, module: str) -> None:
    """Exercise the admission snapshot through the module AOF rewrite path."""
    with tempfile.TemporaryDirectory(prefix="dynkv-admission-aof-") as path:
        directory = Path(path)
        port = _find_free_port()
        process = _start(server, module, directory, port)
        try:
            assert (
                _command(port, b"CONFIG", b"SET", b"aof-use-rdb-preamble", b"no")
                == b"OK"
            )
            key = b"admission-aof-index"
            _admission_workers(port, key)
            payload = _reserve_request(
                b"prefill", 42, 43, 60_000, [81, 82], [(810, 0, 1)]
            )
            response, reservation = _reserve(port, key, payload)
            assert reservation is not None
            lease_key = b"worker-lease-aof-index"
            assert (
                _register_worker_ranks_leased(
                    port, lease_key, 809, 80_901, 120_000, [2]
                )
                == b"OK"
            )
            assert (
                _apply_owned(
                    port,
                    lease_key,
                    80_901,
                    _event(STORE, 809, 2, 1, blocks=[(8091, 89)]),
                )
                == b"OK"
            )
            lease_stats_before_rewrite = _stats(port, lease_key)
            _wait_for_aof_rewrite(port)
        finally:
            _stop(process, port)

        process = _start(server, module, directory, port)
        try:
            assert _admission_stats(port, key) == 1
            replay, replay_reservation = _reserve(port, key, payload)
            assert replay == response
            assert replay_reservation == reservation
            try:
                _reserve(
                    port,
                    key,
                    _reserve_request(
                        b"prefill", 44, 45, 60_000, [81, 82], [(810, 0, 2)]
                    ),
                )
            except RuntimeError as error:
                assert "DYNKV_INVALID_CANDIDATE" in str(error)
            else:
                raise AssertionError("AOF reload must retain admission capacity")
            lease_request = struct.pack("!BIQ", WIRE_VERSION, 1, 89)
            assert _match(_command(port, b"DYNKV.MATCH", lease_key, lease_request)) == {
                (809, 2): (1, 8091)
            }
            assert _stats(port, lease_key) == lease_stats_before_rewrite
            assert _renew_worker_lease(port, lease_key, 809, 80_901, 120_000) == b"OK"
            try:
                _register_worker_ranks_leased(
                    port, lease_key, 809, 80_902, 120_000, [2]
                )
            except RuntimeError as error:
                assert "DYNKV_WORKER_OWNED" in str(error)
            else:
                raise AssertionError("AOF reload must retain the worker owner")
        finally:
            _stop(process, port)
