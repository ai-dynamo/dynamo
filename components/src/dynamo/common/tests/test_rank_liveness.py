# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
import uuid

import pytest

from dynamo.common import rank_liveness as rl

pytestmark = pytest.mark.pre_merge


def _endpoint() -> str:
    return f"inproc://gms-rank-liveness-{uuid.uuid4().hex}"


def _wait(event: threading.Event, timeout: float = 1.0) -> None:
    assert event.wait(timeout), "liveness callback did not fire"


def test_endpoint_overrides_support_replica_scoping(monkeypatch):
    monkeypatch.setenv(
        "DYN_GMS_RANK_LIVENESS_BIND_ADDR",
        "tcp://127.0.0.1:31001",
    )
    monkeypatch.setenv(
        "DYN_GMS_RANK_LIVENESS_CONNECT_ADDR",
        "tcp://{leader_host}:31001",
    )

    assert rl.leader_bind_addr() == "tcp://127.0.0.1:31001"
    assert rl.leader_connect_addr("replica-a") == "tcp://replica-a:31001"


def test_fire_is_exactly_once_even_if_callback_raises():
    calls: list[tuple[int, str]] = []

    def callback(rank: int, reason: str) -> None:
        calls.append((rank, reason))
        raise RuntimeError("expected test error")

    monitor = rl.RankLivenessMonitor(callback)

    assert monitor._fire(1, "first") is True
    assert monitor._fire(2, "second") is False
    assert calls == [(1, "first")]


def test_expected_rank_that_never_registers_fires_startup_timeout():
    endpoint = _endpoint()
    fired = threading.Event()
    calls: list[tuple[int, str]] = []
    monitor = rl.RankLivenessMonitor(
        lambda rank, reason: (calls.append((rank, reason)), fired.set()),
        bind_addr=endpoint,
        timeout_ms_override=100,
        expected_ranks={1},
        startup_grace_ms_override=40,
    )

    monitor.start()
    try:
        _wait(fired)
        assert calls == [(1, "startup-timeout")]
    finally:
        monitor.stop()


def test_legacy_monitor_does_not_require_unseen_ranks():
    fired = threading.Event()
    monitor = rl.RankLivenessMonitor(
        lambda _rank, _reason: fired.set(),
        bind_addr=_endpoint(),
        timeout_ms_override=40,
    )

    monitor.start()
    try:
        time.sleep(0.12)
        assert not fired.is_set()
    finally:
        monitor.stop()


def test_registered_rank_silence_fires_liveness_timeout():
    endpoint = _endpoint()
    fired = threading.Event()
    calls: list[tuple[int, str]] = []
    monitor = rl.RankLivenessMonitor(
        lambda rank, reason: (calls.append((rank, reason)), fired.set()),
        bind_addr=endpoint,
        timeout_ms_override=80,
        expected_ranks={1},
        startup_grace_ms_override=500,
    )
    client = rl.RankLivenessClient(
        "unused",
        1,
        interval_ms=20,
        connect_addr=endpoint,
    )

    monitor.start()
    time.sleep(0.03)
    client.start()
    try:
        time.sleep(0.12)
        client.stop()
        _wait(fired)
        assert calls == [(1, "liveness-timeout")]
    finally:
        client.stop()
        monitor.stop()


def test_unexpected_multipart_identity_does_not_arm_monitor():
    import zmq

    endpoint = _endpoint()
    fired = threading.Event()
    calls: list[tuple[int, str]] = []
    monitor = rl.RankLivenessMonitor(
        lambda rank, reason: (calls.append((rank, reason)), fired.set()),
        bind_addr=endpoint,
        timeout_ms_override=100,
        expected_ranks={1},
        startup_grace_ms_override=80,
    )
    socket = zmq.Context.instance().socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, b"rank-9")
    socket.setsockopt(zmq.LINGER, 0)

    monitor.start()
    time.sleep(0.03)
    socket.connect(endpoint)
    try:
        socket.send(b"hb")
        _wait(fired)
        assert calls == [(1, "startup-timeout")]
    finally:
        socket.close(0)
        monitor.stop()
