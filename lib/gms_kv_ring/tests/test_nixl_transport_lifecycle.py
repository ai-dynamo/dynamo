# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for the NIXL transport wrapper.

These do not instantiate pynixl. They exercise shutdown bookkeeping with a
fake agent so the behavior is covered in ordinary unit-test runs.
"""

from __future__ import annotations

import threading

from gms_kv_ring.daemon.transport.nixl_transport import (
    NixlTransport,
    PeerHandle,
    _AsyncSend,
)


class _FakeThread:
    def __init__(self) -> None:
        self.join_calls: list[float | None] = []

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class _FakeAgent:
    def __init__(self) -> None:
        self.released: list[object] = []
        self.removed: list[str] = []

    def release_xfer_handle(self, handle):
        self.released.append(handle)

    def remove_remote_agent(self, name: str):
        self.removed.append(name)


def test_close_joins_completion_thread_and_releases_pending_async_handles():
    transport = object.__new__(NixlTransport)
    transport._closed = False
    transport._stop = threading.Event()
    transport._completion_thread = _FakeThread()
    transport._notif_thread = _FakeThread()
    transport._async_lock = threading.Lock()
    transport._lock = threading.Lock()
    agent = _FakeAgent()
    transport._agent = agent
    transport._peers = {
        "peer-a": PeerHandle("peer-a", "127.0.0.1", 48101),
    }

    callback_results: list[tuple[bool, str]] = []
    handle = object()
    transport._async_pending = {
        id(handle): _AsyncSend(
            handle=handle,
            peer_name="peer-a",
            reservation_id="rid-1",
            deadline_monotonic=0.0,
            on_complete=lambda ok, err: callback_results.append((ok, err)),
        ),
    }

    transport.close()

    assert transport._closed is True
    assert transport._stop.is_set()
    assert transport._completion_thread.join_calls == [2.0]
    assert transport._notif_thread.join_calls == [2.0]
    assert transport._async_pending == {}
    assert agent.released == [handle]
    assert callback_results == [(False, "transport closed")]
    assert agent.removed == ["peer-a"]
    assert transport._peers == {}
    assert transport._agent is None

    transport.close()
    assert agent.released == [handle]
    assert agent.removed == ["peer-a"]
