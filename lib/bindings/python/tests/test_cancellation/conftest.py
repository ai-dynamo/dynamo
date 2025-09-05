# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import random
import string

import pytest

from dynamo._core import DistributedRuntime


class MockServer:
    """
    Test request handler that simulates a generate method with cancellation support
    """

    def __init__(self):
        self.context_is_stopped = False
        self.context_is_killed = False

    async def generate(self, request, context):
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


def random_string(length=10):
    """Generate a random string for namespace isolation"""
    # Start with a letter to satisfy Prometheus naming requirements
    first_char = random.choice(string.ascii_lowercase)
    remaining_chars = string.ascii_lowercase + string.digits
    rest = "".join(random.choices(remaining_chars, k=length - 1))
    return first_char + rest


@pytest.fixture
async def runtime():
    """Create a DistributedRuntime for testing"""
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, True)
    yield runtime
    runtime.shutdown()


@pytest.fixture
def namespace():
    """Generate a random namespace for test isolation"""
    return random_string()


@pytest.fixture
async def server(runtime, namespace):
    """Start a test server in the background"""

    handler = MockServer()

    async def init_server():
        """Initialize the test server component and serve the generate endpoint"""
        component = runtime.namespace(namespace).component("backend")
        await component.create_service()

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
