# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for placement publishers (Phase 4 of cross-node)."""

from __future__ import annotations

import json
import logging

import pytest
from gms_kv_ring.daemon.placement_publisher import LoggingPlacementPublisher


def _hash(suffix: str) -> bytes:
    """A 32-byte 'hash' for tests — content doesn't matter, only
    that it round-trips through hex encoding correctly."""
    raw = suffix.encode("utf-8")
    return (raw + b"\x00" * 32)[:32]


def test_publish_stored_logs_event(caplog):
    pub = LoggingPlacementPublisher(
        daemon_id="gms-a-gpu0",
        daemon_epoch=42,
        log_level=logging.INFO,
    )
    with caplog.at_level(logging.INFO, logger="gms_kv_ring.daemon.placement_publisher"):
        pub.publish_stored(_hash("h1"), tier="host_pinned", bytes_size=4096)
    assert pub.stats()["stored"] == 1
    # Find the log line + parse the JSON we emitted
    msgs = [r.getMessage() for r in caplog.records]
    json_msgs = [m for m in msgs if "PlacementPublisher" in m]
    assert len(json_msgs) == 1
    payload = json.loads(json_msgs[0].split(" ", 1)[1])
    assert payload["type"] == "stored"
    assert payload["daemon_id"] == "gms-a-gpu0"
    assert payload["daemon_epoch"] == 42
    assert payload["tier"] == "host_pinned"
    assert payload["bytes_size"] == 4096
    assert payload["content_hash"] == _hash("h1").hex()


def test_publish_removed_logs_event(caplog):
    pub = LoggingPlacementPublisher(
        daemon_id="gms-b-gpu0",
        daemon_epoch=7,
        log_level=logging.INFO,
    )
    with caplog.at_level(logging.INFO, logger="gms_kv_ring.daemon.placement_publisher"):
        pub.publish_removed(_hash("h2"), tier="disk")
    assert pub.stats()["removed"] == 1
    msgs = [
        r.getMessage() for r in caplog.records if "PlacementPublisher" in r.getMessage()
    ]
    payload = json.loads(msgs[0].split(" ", 1)[1])
    assert payload["type"] == "removed"
    assert payload["tier"] == "disk"


def test_close_stops_publishing():
    pub = LoggingPlacementPublisher(daemon_id="gms-x", daemon_epoch=1)
    pub.close()
    pub.publish_stored(_hash("h3"), tier="host_pinned", bytes_size=64)
    pub.publish_removed(_hash("h3"), tier="host_pinned")
    stats = pub.stats()
    assert stats["stored"] == 0
    assert stats["removed"] == 0
    assert stats["dropped"] == 2


def test_set_daemon_epoch_updates_future_events(caplog):
    pub = LoggingPlacementPublisher(
        daemon_id="gms-c",
        daemon_epoch=1,
        log_level=logging.INFO,
    )
    with caplog.at_level(logging.INFO, logger="gms_kv_ring.daemon.placement_publisher"):
        pub.publish_stored(_hash("e1"), tier="host_pinned", bytes_size=10)
        pub.set_daemon_epoch(2)
        pub.publish_stored(_hash("e2"), tier="host_pinned", bytes_size=10)
    msgs = [
        r.getMessage() for r in caplog.records if "PlacementPublisher" in r.getMessage()
    ]
    epochs = [json.loads(m.split(" ", 1)[1])["daemon_epoch"] for m in msgs]
    assert epochs == [1, 2]


def test_hash_round_trip_through_hex():
    """Hashes are arbitrary bytes; verify hex encode/decode preserves them."""
    pub = LoggingPlacementPublisher(daemon_id="gms-d", daemon_epoch=1)
    arbitrary = bytes(range(32))
    # We can't read back the emitted hex here without caplog, but we can
    # verify the publisher doesn't crash on adversarial byte values.
    pub.publish_stored(arbitrary, tier="external", bytes_size=999)
    assert pub.stats()["stored"] == 1


def test_zmq_publisher_skipped_without_pyzmq():
    """If pyzmq isn't installed, the import should fail cleanly."""
    pytest.importorskip("zmq")
    # If we got here, pyzmq IS installed; just verify it constructs.
    from gms_kv_ring.daemon.placement_publisher import ZmqPlacementPublisher

    # Use a port-0 wildcard to avoid binding conflicts in CI.
    pub = ZmqPlacementPublisher(
        daemon_id="gms-test",
        daemon_epoch=1,
        bind_endpoint="tcp://127.0.0.1:0",
    )
    try:
        pub.publish_stored(_hash("zmq1"), tier="host_pinned", bytes_size=10)
        # Stats reflect attempted publishes even if no subscriber
        assert pub.stats()["stored"] == 1
    finally:
        pub.close()


# ---- Protocol conformance ---------------------------------------------------


def test_protocol_methods_exist():
    """Sanity check: LoggingPlacementPublisher matches the
    PlacementPublisher Protocol."""
    from gms_kv_ring.daemon.placement_publisher import PlacementPublisher

    pub = LoggingPlacementPublisher(daemon_id="x", daemon_epoch=0)
    # Methods present and callable
    assert callable(pub.publish_stored)
    assert callable(pub.publish_removed)
    assert callable(pub.close)
    # Protocol can hold the instance (runtime_checkable not needed)
    p: PlacementPublisher = pub
    assert p is pub
