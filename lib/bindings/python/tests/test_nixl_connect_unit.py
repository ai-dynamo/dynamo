# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.nixl_connect

Tests transfer-error propagation and descriptor registration ownership across
operation cleanup and reuse. Native NIXL calls are replaced; ownership tests use
real Python operations and CPU tensor descriptors.

NIXL and CUDA are mocked so these tests run on CPU-only machines.
"""

import gc
import sys
import zlib
from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


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


@pytest.fixture
def operation_factory(nixl_mocks, monkeypatch):
    """Real Python operations/descriptors with only native NIXL calls mocked."""
    from dynamo import nixl_connect

    api, _, agent = nixl_mocks
    monkeypatch.setattr(nixl_connect, "nixl_api", api)
    agent.register_memory.side_effect = lambda *args: object()
    agent.transfer.return_value = "DONE"
    agent.check_xfer_state.return_value = "DONE"
    connection = nixl_connect.Connection(nixl_connect.Connector(), 1)

    def make_descriptors(count):
        return [
            nixl_connect.Descriptor(torch.zeros(4, dtype=torch.uint8))
            for _ in range(count)
        ]

    def make_operation(kind, descriptors):
        local = descriptors[0] if len(descriptors) == 1 else descriptors
        if kind == "readable":
            return nixl_connect.ReadableOperation(connection, local)
        if kind == "writable":
            return nixl_connect.WritableOperation(connection, local)
        metadata = nixl_connect.RdmaMetadata(
            descriptors=[
                nixl_connect.SerializedDescriptor(ptr=1, size=4, device="cpu")
                for _ in descriptors
            ],
            operation_kind=1 if kind == "read" else 2,
            notification_key="release-test",
            nixl_metadata=zlib.compress(b"mock-native-metadata").hex(),
        )
        if kind == "read":
            return nixl_connect.ReadOperation(connection, metadata, local)
        return nixl_connect.WriteOperation(connection, local, metadata)

    return make_descriptors, make_operation, connection, agent


@pytest.mark.parametrize("kind", ["read", "write", "readable", "writable"])
@pytest.mark.parametrize("descriptor_count", [1, 2])
def test_old_operation_destruction_preserves_reused_registration(
    operation_factory, kind, descriptor_count
):
    descriptors, operation, _, agent = operation_factory
    local = descriptors(descriptor_count)
    old = operation(kind, local)
    old.__exit__(None, None, None)
    current = operation(kind, local)
    registrations = [d._nixl_hndl for d in local]
    try:
        del old
        gc.collect()
        assert all(d.is_registered for d in local)
        assert [d._nixl_hndl for d in local] == registrations
        assert agent.deregister_memory.call_count == descriptor_count
    finally:
        current.__exit__(None, None, None)


def test_release_does_not_claim_later_registration(operation_factory):
    descriptors, operation, connection, _ = operation_factory
    local = descriptors(1)
    old = operation("readable", local)
    local[0].deregister_with_connector(connection)
    current = operation("readable", local)
    registration = local[0]._nixl_hndl
    try:
        old.__exit__(None, None, None)
        assert local[0].is_registered
        assert local[0]._nixl_hndl is registration
    finally:
        current.__exit__(None, None, None)


@pytest.mark.parametrize("kind", ["read", "write", "readable", "writable"])
@pytest.mark.parametrize("duplicate_first", [False, True])
def test_release_attempts_all_descriptors_and_does_not_retry_after_failure(
    operation_factory, kind, duplicate_first
):
    descriptors, operation, _, agent = operation_factory
    local = descriptors(2)
    old = operation(kind, [local[0], *local] if duplicate_first else local)
    failed_registration = local[0]._nixl_hndl

    def fail_first(registration):
        if registration is failed_registration:
            raise RuntimeError("deregistration failed")

    agent.deregister_memory.side_effect = fail_first
    with pytest.raises(RuntimeError, match="deregistration failed"):
        old.__exit__(None, None, None)
    assert agent.deregister_memory.call_count == 2
    assert local[0].is_registered
    assert not local[1].is_registered

    agent.deregister_memory.side_effect = None
    current = operation(kind, local)
    registrations = [d._nixl_hndl for d in local]
    try:
        del old
        gc.collect()
        assert all(d.is_registered for d in local)
        assert [d._nixl_hndl for d in local] == registrations
        assert agent.deregister_memory.call_count == 2
    finally:
        current.__exit__(None, None, None)


def test_release_preserves_preregistered_duplicate_descriptor_behavior(
    operation_factory,
):
    descriptors, operation, connection, agent = operation_factory
    descriptor = descriptors(1)[0]
    descriptor.register_with_connector(connection)
    op = operation("readable", [descriptor, descriptor])
    op.__exit__(None, None, None)
    op.__exit__(None, None, None)
    assert not descriptor.is_registered
    assert agent.register_memory.call_count == 1
    assert agent.deregister_memory.call_count == 1
