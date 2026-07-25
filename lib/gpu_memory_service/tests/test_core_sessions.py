# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading

import pytest
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.core.server.sessions import GMSSessionManager

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _acquire_in_threads(
    sessions: GMSSessionManager,
    requested: RequestedLockType,
    count: int,
) -> tuple[queue.Queue, list[threading.Thread]]:
    results = queue.Queue()
    threads = [
        threading.Thread(
            target=lambda: results.put(sessions.acquire(requested)),
            daemon=True,
        )
        for _ in range(count)
    ]
    for thread in threads:
        thread.start()
    return results, threads


def test_rw_or_ro_follows_writer_commit_and_abort_epochs() -> None:
    sessions = GMSSessionManager(lambda _reason, _replacing: None)
    writer = sessions.acquire(RequestedLockType.RW)
    assert writer is not None

    committed_results, committed_threads = _acquire_in_threads(
        sessions,
        RequestedLockType.RW_OR_RO,
        2,
    )
    sessions.commit(writer)

    committed_readers = [committed_results.get(timeout=1) for _ in range(2)]
    assert all(
        reader is not None and reader.mode == GrantedLockType.RO
        for reader in committed_readers
    )
    assert sessions.snapshot().ro_session_count == 3
    for reader in committed_readers:
        sessions.close(reader)
    sessions.close(writer)
    for thread in committed_threads:
        thread.join(timeout=1)
        assert not thread.is_alive()

    writer = sessions.acquire(RequestedLockType.RW)
    assert writer is not None
    aborted_results, aborted_threads = _acquire_in_threads(
        sessions,
        RequestedLockType.RW_OR_RO,
        2,
    )
    sessions.close(writer)

    replacement = aborted_results.get(timeout=1)
    assert replacement is not None
    assert replacement.mode == GrantedLockType.RW
    assert aborted_results.empty()

    sessions.close(replacement)
    next_replacement = aborted_results.get(timeout=1)
    assert next_replacement is not None
    assert next_replacement.mode == GrantedLockType.RW
    sessions.close(next_replacement)
    for thread in aborted_threads:
        thread.join(timeout=1)
        assert not thread.is_alive()
