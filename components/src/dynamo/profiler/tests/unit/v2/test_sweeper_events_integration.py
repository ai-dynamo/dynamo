# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real integration test for DEP #15073's Rust binding: round-trips an
actual message through SweeperEventPublisher -> Dynamo's event plane ->
SweeperEventSubscriber, using a process-local (`mem` discovery backend)
DistributedRuntime -- no external NATS/etcd/Kubernetes needed, matching the
pattern the real event_plane/mod.rs tests use
(`DistributedConfig::process_local()` on the Rust side).

Skips cleanly (does not fail) if dynamo._core hasn't been built with the new
binding yet -- this is meant to run only after `maturin develop`, unlike
test_sweeper_event_plane.py which needs no compiled binding at all.

Uses a publish-retry loop rather than a single publish-then-recv, because the
real ZMQ direct-mode transport requires the subscriber to discover and
connect to the publisher first; a message published before that connection
completes is lost. This mirrors the retry pattern in the real Rust tests
(e.g. `direct_zmq_endpoint_scopes_are_isolated` in event_plane/mod.rs) rather
than inventing a new synchronization strategy.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time

import pytest

dynamo_core = pytest.importorskip(
    "dynamo._core",
    reason="requires `maturin develop` to have built the SweeperEventPublisher/"
    "SweeperEventSubscriber binding from DEP #15073",
)

DistributedRuntime = dynamo_core.DistributedRuntime
SweeperEventPublisher = dynamo_core.SweeperEventPublisher
SweeperEventSubscriber = dynamo_core.SweeperEventSubscriber

# Requires SweeperEventPublisher/SweeperEventSubscriber specifically -- an
# older dynamo._core built before this DEP landed would otherwise fail with
# AttributeError deep inside a test instead of a clean skip.
pytestmark = pytest.mark.skipif(
    not hasattr(dynamo_core, "SweeperEventPublisher"),
    reason="dynamo._core predates DEP #15073's SweeperEventPublisher binding",
)


def _make_process_local_runtime() -> "DistributedRuntime":
    loop = asyncio.new_event_loop()
    return DistributedRuntime(loop, "mem", "tcp")


def test_publisher_subscriber_round_trip_over_real_event_plane():
    drt = _make_process_local_runtime()
    endpoint = drt.endpoint("sweeper-integration-test.worker.events")

    publisher = SweeperEventPublisher(endpoint)
    subscriber = SweeperEventSubscriber(endpoint)

    subject = "sweeper.integration-run-1.round_completed"
    payload = b'{"round_no": 1, "cumulative_candidates": 1}'

    result_q: "queue.Queue[bytes | None]" = queue.Queue()

    def _recv():
        result_q.put(subscriber.recv(subject))

    recv_thread = threading.Thread(target=_recv, daemon=True)
    recv_thread.start()

    received = None
    deadline = time.monotonic() + 10.0
    try:
        while time.monotonic() < deadline:
            publisher.publish_subject(subject, payload)
            try:
                received = result_q.get(timeout=0.2)
                break
            except queue.Empty:
                continue
    finally:
        publisher.close()

    assert received == payload, (
        "subscriber never received the published payload within 10s -- "
        "either discovery/connection didn't complete, or the wire format "
        "doesn't round-trip"
    )


def test_multiple_subjects_are_independent():
    """A subscriber listening on one subject must not see events published
    to a different subject on the same endpoint -- exercises the DEP's
    'one subject per event type' design, not just that publish/recv work at
    all."""
    drt = _make_process_local_runtime()
    endpoint = drt.endpoint("sweeper-integration-test.worker.events-multi")

    publisher = SweeperEventPublisher(endpoint)
    subscriber_a = SweeperEventSubscriber(endpoint)
    subscriber_b = SweeperEventSubscriber(endpoint)

    subject_a = "sweeper.integration-run-2.round_completed"
    subject_b = "sweeper.integration-run-2.run_completed"
    payload_a = b'{"round_no": 1}'
    payload_b = b'{"outcome": "succeeded"}'

    result_a: "queue.Queue[bytes | None]" = queue.Queue()
    result_b: "queue.Queue[bytes | None]" = queue.Queue()

    threading.Thread(
        target=lambda: result_a.put(subscriber_a.recv(subject_a)), daemon=True
    ).start()
    threading.Thread(
        target=lambda: result_b.put(subscriber_b.recv(subject_b)), daemon=True
    ).start()

    received_a = received_b = None
    deadline = time.monotonic() + 10.0
    try:
        while time.monotonic() < deadline and (received_a is None or received_b is None):
            publisher.publish_subject(subject_a, payload_a)
            publisher.publish_subject(subject_b, payload_b)
            if received_a is None:
                try:
                    received_a = result_a.get(timeout=0.1)
                except queue.Empty:
                    pass
            if received_b is None:
                try:
                    received_b = result_b.get(timeout=0.1)
                except queue.Empty:
                    pass
    finally:
        publisher.close()

    assert received_a == payload_a
    assert received_b == payload_b
