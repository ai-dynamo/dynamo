# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from dynamo.common import rank_liveness as rl

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _endpoint() -> str:
    return f"inproc://gms-rank-liveness-{uuid.uuid4().hex}"


def test_default_endpoints_are_scoped_by_tp_cohort(monkeypatch):
    for name in (
        "DYN_GMS_RANK_LIVENESS_PORT",
        "DYN_GMS_RANK_LIVENESS_BIND_ADDR",
        "DYN_GMS_RANK_LIVENESS_CONNECT_ADDR",
    ):
        monkeypatch.delenv(name, raising=False)

    primary = "10.0.0.1:29500"
    shadow = "10.0.0.1:29501"
    primary_port = rl.liveness_port(primary)
    shadow_port = rl.liveness_port(shadow)

    assert primary_port != shadow_port
    assert rl.leader_bind_addr(primary) == f"tcp://*:{primary_port}"
    assert rl.leader_connect_addr("leader", primary) == (f"tcp://leader:{primary_port}")


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


def test_wait_for_ranks_is_a_bounded_registration_barrier():
    endpoint = _endpoint()
    monitor = rl.RankLivenessMonitor(
        lambda _rank, _reason: None,
        bind_addr=endpoint,
        timeout_ms_override=200,
    )
    client = rl.RankLivenessClient(
        "unused",
        1,
        interval_ms=20,
        connect_addr=endpoint,
    )

    monitor.start()
    try:
        assert monitor.wait_for_ranks({1}, timeout=0.02) is False
        client.start()
        assert monitor.wait_for_ranks({1}, timeout=1.0) is True
    finally:
        client.stop()
        monitor.stop()


def test_wait_for_ranks_rejects_stale_seen_state_after_monitor_stops():
    monitor = rl.RankLivenessMonitor(
        lambda _rank, _reason: None,
        bind_addr=_endpoint(),
        timeout_ms_override=200,
    )
    monitor._seen_ranks.add(1)

    monitor._stop.set()

    assert monitor.wait_for_ranks({1}, timeout=0.0) is False


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
        assert socket.poll(40, zmq.POLLIN) == 0
        _wait(fired)
        assert calls == [(1, "startup-timeout")]
    finally:
        socket.close(0)
        monitor.stop()


def test_worker_detects_leader_loss_after_acknowledgement():
    endpoint = _endpoint()
    fired = threading.Event()
    calls: list[tuple[int, str]] = []
    monitor = rl.RankLivenessMonitor(
        lambda _rank, _reason: None,
        bind_addr=endpoint,
        timeout_ms_override=80,
    )
    client = rl.RankLivenessClient(
        "unused",
        1,
        interval_ms=20,
        connect_addr=endpoint,
        on_leader_lost=lambda rank, reason: (calls.append((rank, reason)), fired.set()),
        timeout_ms_override=80,
        startup_grace_ms_override=500,
    )

    monitor.start()
    client.start()
    try:
        time.sleep(0.12)
        monitor.stop()
        _wait(fired)
        assert calls == [(0, "liveness-timeout")]
    finally:
        client.stop()
        monitor.stop()


def test_worker_detects_leader_that_never_appears():
    fired = threading.Event()
    calls: list[tuple[int, str]] = []
    client = rl.RankLivenessClient(
        "unused",
        1,
        interval_ms=20,
        connect_addr=_endpoint(),
        on_leader_lost=lambda rank, reason: (calls.append((rank, reason)), fired.set()),
        timeout_ms_override=80,
        startup_grace_ms_override=40,
    )

    client.start()
    try:
        _wait(fired)
        assert calls == [(0, "startup-timeout")]
    finally:
        client.stop()


def test_noisy_peer_cannot_starve_registered_rank_timeout(monkeypatch):
    import zmq

    clock = [0.0]
    calls = []

    class BusySocket:
        received = 0
        closed = False

        def setsockopt(self, *_args):
            pass

        def bind(self, _addr):
            pass

        def recv_multipart(self, **_kwargs):
            self.received += 1
            # Old unbounded draining fails promptly rather than hanging this test.
            assert self.received <= 3 * rl._MAX_HEARTBEATS_PER_POLL
            clock[0] += 0.001
            # Rank 1 disappears after registering; rank 2 never stops sending.
            return [b"rank-1" if self.received == 1 else b"rank-2", b"hb"]

        def send_multipart(self, *_args, **_kwargs):
            pass

        def close(self, _linger):
            self.closed = True

    sock = BusySocket()

    class ReadyPoller:
        def register(self, *_args):
            pass

        def poll(self, _timeout):
            return [(sock, zmq.POLLIN)]

    monkeypatch.setattr(
        zmq,
        "Context",
        SimpleNamespace(instance=lambda: SimpleNamespace(socket=lambda _: sock)),
    )
    monkeypatch.setattr(zmq, "Poller", ReadyPoller)
    monkeypatch.setattr(rl.time, "monotonic", lambda: clock[0])
    monitor = rl.RankLivenessMonitor(
        lambda rank, reason: calls.append((rank, reason)),
        bind_addr=_endpoint(),
        timeout_ms_override=100,
        expected_ranks={1, 2},
        startup_grace_ms_override=1000,
    )

    monitor._run()

    assert calls == [(1, "liveness-timeout")]
    assert sock.received == 2 * rl._MAX_HEARTBEATS_PER_POLL
    assert sock.closed
