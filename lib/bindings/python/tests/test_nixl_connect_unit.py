# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.nixl_connect

Tests the ERRORED state handling in ActiveOperation._wait_for_completion_() added
to prevent decode workers from silently consuming bad data when a prefill worker
disappears mid-transfer (issue #7319).

NIXL and CUDA are mocked so these tests run on CPU-only machines.
"""

import sys
from unittest.mock import MagicMock, patch

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
