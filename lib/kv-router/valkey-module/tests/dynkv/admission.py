# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path

from .admission_expiry import _exercise_admission_expiry_heap
from .admission_helpers import _exercise_admission
from .support import (
    _admission_stats,
    _command,
    _find_free_port,
    _renew,
    _reserve,
    _reserve_request,
    _start,
    _stop,
)



def run_admission(server: str, module: str) -> None:
    with tempfile.TemporaryDirectory(prefix="dynkv-admission-") as path:
        directory = Path(path)
        port = _find_free_port()
        process = _start(server, module, directory, port, appendonly=False)
        try:
            (
                persistent_payload,
                persistent_raw,
                persistent,
                persistent_initial_expires,
            ) = _exercise_admission(port)
            _exercise_admission_expiry_heap(port)
            assert _command(port, b"SAVE") == b"OK"
        finally:
            _stop(process, port)

        process = _start(server, module, directory, port, appendonly=False)
        try:
            assert _admission_stats(port, b"admission-index") == 1
            renewed_retry_raw, renewed_retry = _renew(
                port,
                b"admission-index",
                b"prefill",
                persistent["client"],
                persistent["request"],
                persistent_initial_expires,
                120_000,
            )
            assert renewed_retry_raw == persistent_raw
            assert renewed_retry == persistent
            replay_raw, replay = _reserve(port, b"admission-index", persistent_payload)
            assert replay_raw == persistent_raw
            assert replay == persistent
            # `_exercise_admission` initialized then released decode/810 with
            # capacity 2. Its idle domain-scoped configuration is durable.
            try:
                _reserve(
                    port,
                    b"admission-index",
                    _reserve_request(
                        b"decode", 9010, 9011, 60_000, [81, 82], [(810, 0, 1)]
                    ),
                )
            except RuntimeError as error:
                assert "DYNKV_INVALID_CANDIDATE" in str(error)
            else:
                raise AssertionError("RDB reload must retain domain-scoped capacity")
        finally:
            _stop(process, port)
