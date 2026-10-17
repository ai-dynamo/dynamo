# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.nixl_connect

Tests metadata import ordering, remote reference cleanup, and ERRORED transfer
handling when a prefill worker disappears mid-transfer (issue #7319).

NIXL and CUDA are mocked so these tests run on CPU-only machines.
"""

import base64
import zlib
from unittest.mock import MagicMock, call, patch

import pytest

torch = pytest.importorskip("torch", reason="nixl_connect requires PyTorch")

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
    from dynamo import nixl_connect

    nixl_api_mock, nixl_bindings_mock, agent_instance = _make_nixl_mocks()

    # Patch dependencies in place so module and class identities stay stable.
    cupy_mock = MagicMock()
    cupy_mock.cuda = MagicMock()
    cupy_mock.cuda.is_available = MagicMock(return_value=False)
    cupy_mock.ndarray = type("ndarray", (), {})

    with (
        patch.object(nixl_connect, "nixl_api", nixl_api_mock),
        patch.object(nixl_connect, "nixl_bindings", nixl_bindings_mock),
        patch.object(nixl_connect, "array_module", cupy_mock),
    ):
        yield nixl_api_mock, nixl_bindings_mock, agent_instance


@pytest.fixture
def connection(nixl_mocks):
    from dynamo import nixl_connect

    return nixl_connect.Connection(nixl_connect.Connector(), 1)


@pytest.mark.gpu_0
@pytest.mark.parametrize("encoding", ["bytes", "b64", "hex"])
def test_remote_imports_legacy_metadata(connection, encoding):
    from dynamo.nixl_connect import Remote

    metadata = b"legacy metadata"
    if encoding == "b64":
        encoded = "b64:" + base64.b64encode(zlib.compress(metadata)).decode()
    elif encoding == "hex":
        encoded = zlib.compress(metadata).hex()
    else:
        encoded = metadata
    with Remote(connection, encoded):
        connection._nixl.add_remote_agent.assert_called_once_with(metadata)
        assert connection._remote_refs == {"mock-remote-agent": 1}
    assert connection._remote_refs == {}
    connection._nixl.remove_remote_agent.assert_called_once_with("mock-remote-agent")


@pytest.mark.gpu_0
def test_remote_imports_connection_before_buffer(connection):
    from dynamo.nixl_connect import Remote

    def encode(value):
        return "b64:" + base64.b64encode(zlib.compress(value)).decode()

    with Remote(
        connection,
        encode(b"buffer"),
        nixl_connection_metadata=encode(b"connection"),
    ):
        assert connection._nixl.add_remote_agent.call_args_list == [
            call(b"connection"),
            call(b"buffer"),
        ]
        assert connection._remote_refs == {"mock-remote-agent": 1}
    assert connection._remote_refs == {}
    connection._nixl.remove_remote_agent.assert_called_once_with("mock-remote-agent")


@pytest.mark.gpu_0
@pytest.mark.parametrize("failed_import", [1, 2])
def test_read_import_failure_balances_python_remote_refs(connection, failed_import):
    """Check Python reference accounting, not native state after a failed import.

    NIXL is mocked here. A real failed import can invalidate shared native state
    even when the Python reference counts remain correct.
    """
    from dynamo.nixl_connect import (
        Descriptor,
        OperationKind,
        RdmaMetadata,
        ReadOperation,
        Remote,
    )

    with Remote(connection, b"existing read"):
        connection._nixl.add_remote_agent.reset_mock()
        connection._nixl.add_remote_agent.side_effect = (
            [RuntimeError("import failed")]
            if failed_import == 1
            else [b"mock-remote-agent", RuntimeError("import failed")]
        )
        metadata = "b64:" + base64.b64encode(zlib.compress(b"buffer")).decode()
        with pytest.raises(RuntimeError, match="import failed"):
            ReadOperation(
                connection,
                RdmaMetadata(
                    nixl_metadata=metadata,
                    operation_kind=int(OperationKind.READ),
                ),
                Descriptor(torch.empty(16, dtype=torch.uint8)),
                nixl_connection_metadata=b"connection",
            )
        assert connection._remote_refs == {"mock-remote-agent": 1}
        connection._nixl.remove_remote_agent.assert_not_called()
        connection._nixl.initialize_xfer.assert_not_called()
    assert connection._remote_refs == {}
    connection._nixl.remove_remote_agent.assert_called_once_with("mock-remote-agent")


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
