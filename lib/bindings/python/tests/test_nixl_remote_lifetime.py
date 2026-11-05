# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Remote ownership with only native NIXL calls replaced."""

import gc
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo import nixl_connect

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]


@pytest.fixture
def connection(monkeypatch):
    native = MagicMock()
    peers = set()

    def add(metadata):
        name = metadata.decode("utf-8")
        peers.add(name)
        return metadata

    native.add_remote_agent.side_effect = add
    native.remove_remote_agent.side_effect = peers.discard
    monkeypatch.setattr(
        nixl_connect, "nixl_api", SimpleNamespace(nixl_agent=lambda name: native)
    )
    connection = nixl_connect.Connection(nixl_connect.Connector(), 1)
    return connection, native, peers


def test_last_release_cannot_retire_an_incoming_remote(connection):
    connection, native, peers = connection
    old_holder = [nixl_connect.Remote(connection, b"peer")]
    loaded = threading.Event()
    release_attempted = threading.Event()
    released = threading.Event()
    native_add = native.add_remote_agent.side_effect

    def add_while_retiring(metadata):
        name = native_add(metadata)
        # loadRemoteMD releases the GIL. Its new metadata can be installed before
        # the caller regains Python execution and increments its reference count.
        loaded.set()
        assert release_attempted.wait(3), "retiring thread did not start"
        # The fixed lifecycle lock intentionally blocks retirement until this
        # call returns; bound the gate so the corrected code cannot deadlock.
        released.wait(0.1)
        return name

    def retire_old():
        assert loaded.wait(3), "native load did not start"
        release_attempted.set()
        try:
            # Exercise the actual destructor on another thread.
            old_holder.clear()
        finally:
            released.set()

    native.add_remote_agent.side_effect = add_while_retiring
    newer = None
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            retirement = executor.submit(retire_old)
            replacement = executor.submit(nixl_connect.Remote, connection, b"peer")
            newer = replacement.result(timeout=3)
            retirement.result(timeout=3)
        assert connection._remote_refs == {"peer": 1}
        assert peers == {"peer"}
        native.remove_remote_agent.assert_not_called()
    finally:
        if newer is not None:
            newer.__exit__(None, None, None)
        old_holder.clear()
    native.remove_remote_agent.assert_called_once_with("peer")


def test_shared_remote_is_removed_once_after_last_reference(connection):
    connection, native, peers = connection
    first = nixl_connect.Remote(connection, b"peer")
    second = nixl_connect.Remote(connection, b"peer")
    other = nixl_connect.Remote(connection, b"other")
    first.__exit__(None, None, None)
    first.__exit__(None, None, None)
    assert connection._remote_refs == {"peer": 1, "other": 1}
    native.remove_remote_agent.assert_not_called()
    second.__exit__(None, None, None)
    assert peers == {"other"}
    other.__exit__(None, None, None)
    assert connection._remote_refs == {}
    assert peers == set()
    assert native.remove_remote_agent.call_count == 2


def test_last_remove_finishes_before_a_new_remote_is_registered(
    connection, monkeypatch
):
    connection, native, peers = connection
    old = nixl_connect.Remote(connection, b"peer")
    decremented = threading.Event()
    incoming_attempted = threading.Event()
    incoming_done = threading.Event()
    release_ref = connection.release_remote_ref

    def release_before_native_remove(name):
        last = release_ref(name)
        if last:
            decremented.set()
            assert incoming_attempted.wait(3), "replacement did not start"
            # Bound the scheduling gate: the fixed lock keeps the replacement
            # out until the pending native removal has finished.
            incoming_done.wait(0.1)
        return last

    def incoming():
        assert decremented.wait(3), "last reference was not released"
        incoming_attempted.set()
        remote = nixl_connect.Remote(connection, b"peer")
        incoming_done.set()
        return remote

    monkeypatch.setattr(connection, "release_remote_ref", release_before_native_remove)
    newer = None
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            retirement = executor.submit(old.__exit__, None, None, None)
            replacement = executor.submit(incoming)
            retirement.result(timeout=3)
            newer = replacement.result(timeout=3)
        assert connection._remote_refs == {"peer": 1}
        assert peers == {"peer"}
        native.remove_remote_agent.assert_called_once_with("peer")
    finally:
        if newer is not None:
            newer.__exit__(None, None, None)
        old.__exit__(None, None, None)


def test_failed_native_add_does_not_release_an_existing_remote(connection, monkeypatch):
    connection, native, peers = connection
    old = nixl_connect.Remote(connection, b"peer")
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    def fail_add(_):
        raise RuntimeError("native load failed")

    native.add_remote_agent.side_effect = fail_add
    try:
        with pytest.raises(RuntimeError, match="native load failed"):
            nixl_connect.Remote(connection, b"peer")
        gc.collect()
        assert unraisable == []
        assert connection._remote_refs == {"peer": 1}
        assert peers == {"peer"}
        native.remove_remote_agent.assert_not_called()
    finally:
        old.__exit__(None, None, None)


def test_failed_native_remove_does_not_release_a_later_owner(connection):
    connection, native, peers = connection
    old = nixl_connect.Remote(connection, b"peer")
    native.remove_remote_agent.side_effect = RuntimeError("native removal failed")
    with pytest.raises(RuntimeError, match="native removal failed"):
        old.__exit__(None, None, None)
    assert connection._remote_refs == {}
    native.remove_remote_agent.side_effect = peers.discard
    # A failure must release the lifecycle lock for a different thread too.
    with ThreadPoolExecutor(max_workers=1) as executor:
        newer = executor.submit(nixl_connect.Remote, connection, b"peer").result(
            timeout=3
        )
    try:
        old.__exit__(None, None, None)
        assert connection._remote_refs == {"peer": 1}
        assert peers == {"peer"}
        assert native.remove_remote_agent.call_count == 1
    finally:
        newer.__exit__(None, None, None)
    assert peers == set()
    assert native.remove_remote_agent.call_count == 2
