# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real passive operations with only the native NIXL agent mocked."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from dynamo import nixl_connect

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]


@pytest.fixture
def passive_operations(monkeypatch):
    agent = MagicMock()
    agent.register_memory.side_effect = lambda *args: object()
    agent.notifs = {}
    agent.update_notifs.side_effect = lambda: agent.notifs
    monkeypatch.setattr(
        nixl_connect, "nixl_api", SimpleNamespace(nixl_agent=lambda name: agent)
    )
    connection = nixl_connect.Connection(nixl_connect.Connector(), 1)
    operations = []

    def make_operation():
        descriptor = nixl_connect.Descriptor(torch.zeros(4, dtype=torch.uint8))
        operation = nixl_connect.ReadableOperation(connection, descriptor)
        operations.append(operation)
        return operation

    yield nixl_connect, agent, connection, make_operation
    for operation in operations:
        operation.__exit__(None, None, None)


def test_completed_notifications_do_not_accumulate(passive_operations):
    module, agent, _, make_operation = passive_operations
    for _ in range(64):
        operation = make_operation()
        agent.notifs.setdefault("remote", []).append(
            operation._notification_key.encode("utf-8")
        )
        assert operation.status is module.OperationStatus.COMPLETE
    assert agent.notifs == {"remote": []}


def test_consumption_preserves_other_notifications(passive_operations):
    module, agent, _, make_operation = passive_operations
    first, second, third, pending = [make_operation() for _ in range(4)]
    first_key, second_key, third_key = [
        operation._notification_key.encode("utf-8")
        for operation in (first, second, third)
    ]
    prefix_collision = first_key + b"-unrelated"
    agent.notifs = {
        "remote-a": [prefix_collision, second_key, first_key, b"other-a"],
        "remote-b": [third_key, b"other-b"],
    }
    assert pending.status is module.OperationStatus.INITIALIZED
    assert first.status is module.OperationStatus.COMPLETE
    assert agent.notifs == {
        "remote-a": [prefix_collision, second_key, b"other-a"],
        "remote-b": [third_key, b"other-b"],
    }
    assert second.status is module.OperationStatus.COMPLETE
    assert third.status is module.OperationStatus.COMPLETE
    assert pending.status is module.OperationStatus.INITIALIZED
    assert agent.notifs == {
        "remote-a": [prefix_collision, b"other-a"],
        "remote-b": [b"other-b"],
    }
    calls = agent.update_notifs.call_count
    assert first.status is module.OperationStatus.COMPLETE
    assert agent.update_notifs.call_count == calls


@pytest.mark.parametrize("same_operation", [False, True])
def test_concurrent_status_does_not_restore_consumed_notifications(
    passive_operations, same_operation
):
    module, agent, connection, make_operation = passive_operations
    first = make_operation()
    second = first if same_operation else make_operation()
    agent.notifs = {
        "remote": list(
            dict.fromkeys(
                operation._notification_key.encode("utf-8")
                for operation in (first, second)
            )
        )
    }
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    second_reached = threading.Event()
    call_lock = threading.Lock()
    native_calls = 0

    class ObservedLock:
        def __init__(self):
            self.inner = threading.Lock()
            self.attempts = 0

        def __enter__(self):
            with call_lock:
                self.attempts += 1
                if self.attempts == 2:
                    second_reached.set()
            self.inner.acquire()

        def __exit__(self, *args):
            self.inner.release()

    connection._notifications_lock = ObservedLock()

    def update_notifs():
        nonlocal native_calls
        # NIXL converts the retained Python map to C++ containers, releases the
        # GIL while polling, then returns a new map. Overlapping snapshots can
        # restore notifications consumed after the snapshot was captured.
        snapshot = {remote: list(values) for remote, values in agent.notifs.items()}
        with call_lock:
            index = native_calls
            native_calls += 1
        assert index < 2
        entered[index].set()
        if index == 1:
            second_reached.set()
        assert release[index].wait(3), "notification polling gate timed out"
        agent.notifs = snapshot
        return snapshot

    agent.update_notifs.side_effect = update_notifs
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_result = executor.submit(lambda: first.status)
        try:
            assert entered[0].wait(3)
            second_result = executor.submit(lambda: second.status)
            assert second_reached.wait(3)
            overlapping_native_polls = entered[1].is_set()
            release[0].set()
            assert first_result.result(timeout=3) is module.OperationStatus.COMPLETE
            release[1].set()
            assert second_result.result(timeout=3) is module.OperationStatus.COMPLETE
        finally:
            for event in release:
                event.set()

    assert not overlapping_native_polls
    assert native_calls == (1 if same_operation else 2)
    assert first.status is module.OperationStatus.COMPLETE
    assert second.status is module.OperationStatus.COMPLETE
    assert agent.notifs == {"remote": []}


def test_notification_poll_error_does_not_prevent_retry(passive_operations):
    module, agent, _, make_operation = passive_operations
    operation = make_operation()
    notifications = {"remote": [operation._notification_key.encode("utf-8")]}
    agent.update_notifs.side_effect = [
        RuntimeError("native polling failed"),
        notifications,
    ]
    with pytest.raises(RuntimeError, match="native polling failed"):
        _ = operation.status
    assert operation.status is module.OperationStatus.COMPLETE
    assert notifications == {"remote": []}
