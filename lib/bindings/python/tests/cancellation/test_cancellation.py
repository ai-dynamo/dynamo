# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from dynamo.runtime import Context

pytestmark = pytest.mark.pre_merge


class MockServer:
    """
    Test request handler that simulates a generate method with cancellation support
    """

    def __init__(self):
        self.context_is_stopped = False
        self.context_is_killed = False

    async def generate(self, request, context):
        print("################## generate called ######################")

        self.context_is_stopped = False
        self.context_is_killed = False

        method_name = request
        assert hasattr(
            self, method_name
        ), f"Method '{method_name}' not found on {self.__class__.__name__}"
        method = getattr(self, method_name)
        async for response in method(request, context):
            yield response

    async def _generate_until_context_cancelled(self, request, context):
        """
        Generate method that yields numbers 0-999 every 0.1 seconds
        Checks for context.is_stopped() / context.is_killed() before each yield and raises
        CancelledError if stopped / killed
        """
        for i in range(1000):
            print(f"Processing iteration {i}")

            # Check if context is stopped
            if context.is_stopped():
                print(f"Context stopped at iteration {i}")
                self.context_is_stopped = True
                self.context_is_killed = context.is_killed()
                raise asyncio.CancelledError

            # Check if context is killed
            if context.is_killed():
                print(f"Context killed at iteration {i}")
                self.context_is_stopped = context.is_stopped()
                self.context_is_killed = True
                raise asyncio.CancelledError

            await asyncio.sleep(0.1)

            print(f"Sending iteration {i}")
            yield i

        assert (
            False
        ), "Test failed: generate_until_cancelled did not raise CancelledError"

    async def _generate_until_asyncio_cancelled(self, request, context):
        """
        Generate method that yields numbers 0-999 every 0.1 seconds
        """
        i = 0
        try:
            for i in range(1000):
                print(f"Processing iteration {i}")
                await asyncio.sleep(0.1)
                print(f"Sending iteration {i}")
                yield i
        except asyncio.CancelledError:
            print(f"Cancelled at iteration {i}")
            self.context_is_stopped = context.is_stopped()
            self.context_is_killed = context.is_killed()
            raise

        assert (
            False
        ), "Test failed: generate_until_cancelled did not raise CancelledError"

    async def _generate_and_cancel_context(self, request, context):
        """
        Generate method that yields numbers 0-1, and then cancel the context
        """
        for i in range(2):
            print(f"Processing iteration {i}")
            await asyncio.sleep(0.1)
            print(f"Sending iteration {i}")
            yield i

        context.stop_generating()

        self.context_is_stopped = context.is_stopped()
        self.context_is_killed = context.is_killed()

    async def _generate_and_raise_cancelled(self, request, context):
        """
        Generate method that yields numbers 0-1, and then raise asyncio.CancelledError
        """
        for i in range(2):
            print(f"Processing iteration {i}")
            await asyncio.sleep(0.1)
            print(f"Sending iteration {i}")
            yield i

        raise asyncio.CancelledError


@pytest.fixture
def namespace():
    """Namespace for this test file"""
    return "cancellation-unit-test"


@pytest.fixture
async def server(runtime, namespace):
    """Start a test server in the background"""

    handler = MockServer()

    async def init_server():
        """Initialize the test server component and serve the generate endpoint"""
        component = runtime.namespace(namespace).component("backend")
        endpoint = component.endpoint("generate")
        print("Started test server instance")

        # Serve the endpoint - this will block until shutdown
        await endpoint.serve_endpoint(handler.generate)

    # Start server in background task
    server_task = asyncio.create_task(init_server())

    # Give server time to start up
    await asyncio.sleep(0.5)

    yield server_task, handler

    # Cleanup - cancel server task
    if not server_task.done():
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


@pytest.fixture
async def client(runtime, namespace):
    """Create a client connected to the test server"""
    # Create client
    endpoint = runtime.namespace(namespace).component("backend").endpoint("generate")
    client = await endpoint.client()
    await client.wait_for_instances()

    return client


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
async def test_client_context_cancel(temp_file_store, server, client):
    _, handler = server
    context = Context()
    stream = await client.generate("_generate_until_context_cancelled", context=context)

    iteration_count = 0
    async for annotated in stream:
        number = annotated.data()
        print(f"Received iteration: {number}")

        # Verify received valid number
        assert number == iteration_count

        # Break after receiving 2 responses
        if iteration_count >= 2:
            print("Cancelling after 2 responses...")
            context.stop_generating()
            break

        iteration_count += 1

    # Give server a moment to process the cancellation
    await asyncio.sleep(0.2)

    # Verify server detected the cancellation
    assert handler.context_is_stopped
    assert not handler.context_is_killed

    # TODO: Test with _generate_until_asyncio_cancelled server handler


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
async def test_client_loop_break(temp_file_store, server, client):
    _, handler = server
    stream = await client.generate("_generate_until_context_cancelled")

    iteration_count = 0
    async for annotated in stream:
        number = annotated.data()
        print(f"Received iteration: {number}")

        # Verify received valid number
        assert number == iteration_count

        # Break after receiving 2 responses
        if iteration_count >= 2:
            print("Cancelling after 2 responses...")
            break

        iteration_count += 1

    # Give server a moment to process the cancellation
    await asyncio.sleep(0.2)

    # TODO: Implicit cancellation is not yet implemented, so the server context will not
    #       show any cancellation.
    assert not handler.context_is_stopped
    assert not handler.context_is_killed

    # TODO: Test with _generate_until_asyncio_cancelled server handler


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
async def test_server_context_cancel(temp_file_store, server, client):
    _, handler = server
    stream = await client.generate("_generate_and_cancel_context")

    iteration_count = 0
    try:
        async for annotated in stream:
            number = annotated.data()
            print(f"Received iteration: {number}")
            assert number == iteration_count
            iteration_count += 1
        assert False, "Stream completed without cancellation"
    except ValueError as e:
        # Verify the expected cancellation exception is received
        # TODO: Should this be a asyncio.CancelledError?
        assert str(e).startswith("Stream ended before generation completed")

    # Verify server context cancellation status
    assert handler.context_is_stopped
    assert not handler.context_is_killed


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
async def test_server_raise_cancelled(temp_file_store, server, client):
    _, handler = server
    stream = await client.generate("_generate_and_raise_cancelled")

    iteration_count = 0
    try:
        async for annotated in stream:
            number = annotated.data()
            print(f"Received iteration: {number}")
            assert number == iteration_count
            iteration_count += 1
        assert False, "Stream completed without cancellation"
    except ValueError as e:
        # Verify the expected cancellation exception is received
        # TODO: Should this be a asyncio.CancelledError?
        assert (
            str(e)
            == "a python exception was caught while processing the async generator: CancelledError: "
        )

    # Verify server context cancellation status
    # TODO: Server to gracefully stop the stream?
    assert not handler.context_is_stopped
    assert not handler.context_is_killed


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
async def test_client_context_already_cancelled(temp_file_store, server, client):
    _, handler = server
    context = Context()
    context.stop_generating()
    # TODO: (DIS-830) The outgoing call should raise if context is cancelled
    stream = await client.generate("_generate_until_context_cancelled", context=context)

    async for _ in stream:
        raise AssertionError(
            "Request should be cancelled before any responses are generated"
        )

    # Give server a moment to update status
    await asyncio.sleep(0.2)

    # Verify server context cancellation status
    assert handler.context_is_stopped
    assert not handler.context_is_killed


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
async def test_client_context_cancel_before_await_request(
    temp_file_store, server, client
):
    _, handler = server
    context = Context()
    request = client.generate("_generate_until_context_cancelled", context=context)
    context.stop_generating()
    # TODO: (DIS-830) The outgoing call should raise if context is cancelled
    stream = await request

    async for _ in stream:
        raise AssertionError(
            "Request should be cancelled before any responses are generated"
        )

    # Give server a moment to update status
    await asyncio.sleep(0.2)

    # Verify server context cancellation status
    assert handler.context_is_stopped
    assert not handler.context_is_killed
