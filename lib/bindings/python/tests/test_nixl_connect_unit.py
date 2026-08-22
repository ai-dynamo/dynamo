# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.nixl_connect

Tests the ERRORED state handling in ActiveOperation._wait_for_completion_() added
to prevent decode workers from silently consuming bad data when a prefill worker
disappears mid-transfer (issue #7319).

NIXL and CUDA are mocked so these tests run on CPU-only machines.
"""

import sys
from unittest.mock import MagicMock, call, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge]


def _make_nixl_mocks():
    """Create minimal mocks for nixl._api and nixl._bindings."""
    nixl_api_mock = MagicMock()
    nixl_bindings_mock = MagicMock()

    # nixl_agent mock (returned by nixl_api.nixl_agent(...))
    agent_instance = MagicMock()
    agent_instance.get_agent_metadata.return_value = b"mock-metadata"
    agent_instance.add_remote_agent.return_value = b"mock-remote-agent"
    agent_instance.get_xfer_descs.return_value = MagicMock()
    agent_instance.initialize_xfer.return_value = MagicMock()
    agent_instance.register_memory.return_value = MagicMock()
    nixl_api_mock.nixl_agent.return_value = agent_instance
    nixl_api_mock.nixl_xfer_handle = MagicMock

    return nixl_api_mock, nixl_bindings_mock, agent_instance


@pytest.fixture
def nixl_mocks():
    nixl_api_mock, nixl_bindings_mock, agent_instance = _make_nixl_mocks()

    # Patch cupy import too since nixl_connect tries to import it
    cupy_mock = MagicMock()
    cupy_mock.cuda = MagicMock()
    cupy_mock.cuda.is_available = MagicMock(return_value=False)
    cupy_mock.ndarray = type("ndarray", (), {})

    with (
        patch.dict(
            sys.modules,
            {
                "nixl": MagicMock(),
                "nixl._api": nixl_api_mock,
                "nixl._bindings": nixl_bindings_mock,
                "cupy": cupy_mock,
                "cupy_backends": MagicMock(),
                "cupy_backends.cuda": MagicMock(),
                "cupy_backends.cuda.api": MagicMock(),
                "cupy_backends.cuda.api.runtime": MagicMock(),
            },
        ),
    ):
        yield nixl_api_mock, nixl_bindings_mock, agent_instance


@pytest.fixture
def testable_active_op(nixl_mocks):
    """Factory fixture: returns a function that creates a _TestableActiveOp with a given status sequence.

    The subclass short-circuits ActiveOperation.__init__ to avoid NIXL hardware
    calls, while preserving the real _wait_for_completion_() logic under test.
    """
    from dynamo.nixl_connect import ActiveOperation, OperationStatus

    class _TestableActiveOp(ActiveOperation):
        def __init__(self, status_sequence):
            self._status = OperationStatus.INITIALIZED
            self._status_sequence = iter(status_sequence)
            self._remote = MagicMock()
            self._remote.name = "mock-prefill-worker"
            self._xfer_hndl = MagicMock()
            self._connection = MagicMock()
            self._local_desc_list = MagicMock()
            self._local_desc_tlist = []
            self._remote_desc_tlist = []
            self._local_device_kind = MagicMock()
            self._remote_device_kind = MagicMock()
            self._notification_key = "test-key"
            self._operation_kind = MagicMock()

        @property
        def status(self):
            try:
                self._status = next(self._status_sequence)
            except StopIteration:
                pass
            return self._status

        def cancel(self):
            pass

        async def wait_for_completion(self):
            await self._wait_for_completion_()

        def _release(self):
            pass

    return _TestableActiveOp


@pytest.mark.asyncio
async def test_wait_for_completion_raises_on_errored_status(testable_active_op):
    """ActiveOperation._wait_for_completion_ must raise RuntimeError when ERRORED.

    Before fix: silently returned, leaving caller unaware the transfer failed.
    After fix: raises RuntimeError so the caller can handle the failure (e.g.,
    convert it to a retryable RequestError instead of propagating a segfault).

    This is the core decode-side fix for issue #7319.
    """
    from dynamo.nixl_connect import OperationStatus

    # Simulate: INITIALIZED -> IN_PROGRESS -> ERRORED (remote agent disappeared)
    op = testable_active_op(
        [
            OperationStatus.INITIALIZED,
            OperationStatus.IN_PROGRESS,
            OperationStatus.ERRORED,
        ]
    )

    with pytest.raises(RuntimeError, match=r"ERRORED|errored|error"):
        await op.wait_for_completion()


@pytest.mark.asyncio
async def test_connector_can_disable_nixl_progress_thread(nixl_mocks):
    """Actively polled callers can avoid an otherwise redundant agent thread."""
    import dynamo.nixl_connect as nixl_connect

    nixl_api, _, agent = nixl_mocks
    config = MagicMock()
    nixl_api.nixl_agent_config.return_value = config

    # Other test modules may have imported dynamo.nixl_connect during
    # collection, before the fixture patched sys.modules. Patch the module's
    # binding directly so this assertion is collection-order independent.
    with patch.object(nixl_connect, "nixl_api", nixl_api):
        connection = await nixl_connect.Connector(
            "actively-polled", enable_progress_thread=False
        )._create_connection()

    nixl_api.nixl_agent_config.assert_called_once_with(enable_prog_thread=False)
    nixl_api.nixl_agent.assert_called_once_with("actively-polled-1", config)
    assert connection._nixl is agent


def test_connector_rejects_non_boolean_progress_thread_option(nixl_mocks):
    from dynamo.nixl_connect import Connector

    with pytest.raises(TypeError, match="enable_progress_thread"):
        Connector("consumer", enable_progress_thread=0)


@pytest.mark.asyncio
async def test_remote_agent_is_loaded_once_while_operations_overlap(nixl_mocks):
    """Overlapping reads from one producer must share its loaded metadata."""
    from dynamo.nixl_connect import Connector, Remote

    connector = Connector("consumer")
    connection = await connector._create_connection()
    agent = connection._nixl
    agent.add_remote_agent.return_value = b"mock-remote-agent"

    first = Remote(connection, b"producer-metadata")
    second = Remote(connection, b"producer-metadata")

    agent.add_remote_agent.assert_called_once_with(b"producer-metadata")
    first._release()
    agent.remove_remote_agent.assert_not_called()
    second._release()
    agent.remove_remote_agent.assert_not_called()


@pytest.mark.asyncio
async def test_remote_agent_remains_loaded_after_last_operation(nixl_mocks):
    """A connection reuses producer metadata across sequential operations."""
    from dynamo.nixl_connect import Connector, Remote

    connector = Connector("consumer")
    connection = await connector._create_connection()
    agent = connection._nixl
    agent.add_remote_agent.return_value = b"mock-remote-agent"

    first = Remote(connection, b"producer-metadata")
    first._release()
    second = Remote(connection, b"producer-metadata")
    second._release()

    agent.add_remote_agent.assert_called_once_with(b"producer-metadata")
    agent.remove_remote_agent.assert_not_called()


@pytest.mark.asyncio
async def test_remote_agent_uses_stable_name_when_metadata_changes(nixl_mocks):
    """Fresh metadata snapshots from one producer must not reload its agent."""
    from dynamo.nixl_connect import Connector, Remote

    connector = Connector("consumer")
    connection = await connector._create_connection()
    agent = connection._nixl
    agent.add_remote_agent.return_value = b"producer-agent"

    first = Remote(connection, b"producer-metadata-1", expected_name="producer-agent")
    first._release()
    second = Remote(connection, b"producer-metadata-2", expected_name="producer-agent")
    second._release()

    agent.add_remote_agent.assert_called_once_with(b"producer-metadata-1")
    agent.remove_remote_agent.assert_not_called()


@pytest.mark.asyncio
async def test_remote_agent_merges_distinct_partial_metadata(nixl_mocks):
    """New memory registrations are merged without replaying duplicates."""
    from dynamo.nixl_connect import Connector, Remote

    connector = Connector("consumer")
    connection = await connector._create_connection()
    agent = connection._nixl
    agent.add_remote_agent.return_value = b"producer-agent"

    for full_metadata, partial_metadata in (
        (b"full-1", b"partial-1"),
        (b"full-2", b"partial-2"),
        (b"full-3", b"partial-2"),
    ):
        remote = Remote(
            connection,
            full_metadata,
            expected_name="producer-agent",
            partial_nixl_metadata=partial_metadata,
        )
        remote._release()

    assert agent.add_remote_agent.call_args_list == [
        call(b"full-1"),
        call(b"partial-2"),
        call(b"partial-2"),
    ]
    agent.remove_remote_agent.assert_not_called()


@pytest.mark.asyncio
async def test_connection_retains_notifications_for_their_operations(nixl_mocks):
    """Polling one lease must not discard another lease's completion."""
    from dynamo.nixl_connect import Connector

    connection = await Connector("producer")._create_connection()
    agent = connection._nixl
    agent.update_notifs.return_value = {"consumer": [b"first", b"second"]}

    assert connection.consume_notification("first")
    assert connection.consume_notification("second")
    agent.update_notifs.assert_called_once_with()


@pytest.mark.asyncio
async def test_operation_does_not_release_pre_registered_descriptor(nixl_mocks):
    """Pool-owned registrations outlive individual readable operations."""
    from dynamo.nixl_connect import Connector, Descriptor, ReadableOperation

    connection = await Connector("producer")._create_connection()
    descriptor = Descriptor(b"pool-slot")
    descriptor.register_with_connector(connection)
    operation = ReadableOperation(connection, descriptor)

    operation.__exit__(None, None, None)

    assert descriptor.is_registered
    connection._nixl.deregister_memory.assert_not_called()
    descriptor.deregister_with_connector(connection)
    connection._nixl.deregister_memory.assert_called_once()
