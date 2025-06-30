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

"""
Tests for startup and shutdown hooks in Dynamo services.

This file contains unit tests for the basic hook functionality and
tests for async init hooks in web workers with a frontend.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

from dynamo.sdk.cli.serve_dynamo import (
    add_fastapi_routes,
    run_shutdown_hooks,
    run_startup_hooks,
)
from dynamo.sdk.core.protocol.interface import DynamoTransport
from dynamo.sdk.lib.decorators import async_on_start, on_shutdown


class MockServiceWithHooks:
    """Mock service class with startup and shutdown hooks for testing."""

    def __init__(self):
        self.sync_hook_called = False
        self.async_hook_called = False
        self.another_hook_called = False
        self.shutdown_hook_called = False
        self.hook_execution_order = []

        # Add counters to track how many times each hook is called
        self.sync_hook_count = 0
        self.async_hook_count = 0
        self.another_hook_count = 0

        # Add FastAPI apps for HTTP server testing
        self.app = FastAPI()
        self.system_app = FastAPI()

    @async_on_start
    async def async_startup_hook(self):
        """Async startup hook that should be called during startup."""
        await asyncio.sleep(0.1)  # Simulate some async work
        self.async_hook_called = True
        self.async_hook_count += 1
        if hasattr(self, "hook_execution_order"):
            self.hook_execution_order.append("async_startup_hook")

    @async_on_start
    def sync_startup_hook(self):
        """Sync startup hook that should be called during startup."""
        self.sync_hook_called = True
        self.sync_hook_count += 1
        if hasattr(self, "hook_execution_order"):
            self.hook_execution_order.append("sync_startup_hook")

    @async_on_start
    async def another_startup_hook(self):
        """Another async startup hook."""
        await asyncio.sleep(0.1)  # Simulate some async work
        self.another_hook_called = True
        self.another_hook_count += 1
        if hasattr(self, "hook_execution_order"):
            self.hook_execution_order.append("another_startup_hook")

    @on_shutdown
    def shutdown_hook(self):
        """Shutdown hook that should be called during shutdown."""
        self.shutdown_hook_called = True
        if hasattr(self, "hook_execution_order"):
            self.hook_execution_order.append("shutdown_hook")

    def regular_method(self):
        """Regular method that should not be called during startup."""
        raise Exception("This method should not be called")

    def test_endpoint_method(self):
        """A test endpoint method that can be bound to the class instance."""
        return {"status": "ok"}

    def get_dynamo_endpoints(self):
        """Return mock endpoints with HTTP transport for testing."""
        # Create a mock endpoint with HTTP transport
        mock_endpoint = MagicMock()
        mock_endpoint.transports = [DynamoTransport.HTTP, DynamoTransport.DEFAULT]
        # Set the func attribute to a real method that can be bound to the class instance
        mock_endpoint.func = MockServiceWithHooks.test_endpoint_method
        mock_endpoint.request_type = dict
        return {"test_endpoint": mock_endpoint}

    def inner(self):
        """Return self as the inner service instance."""
        return self


@pytest.mark.asyncio
async def test_run_startup_hooks():
    """Test that startup hooks are properly called."""
    # Create an instance of our mock service
    service = MockServiceWithHooks()

    # Run the startup hooks
    await run_startup_hooks(service)

    # Verify that all startup hooks were called
    assert service.sync_hook_called, "Sync startup hook was not called"
    assert service.async_hook_called, "Async startup hook was not called"
    assert service.another_hook_called, "Another startup hook was not called"

    # Verify that shutdown hooks were not called
    assert (
        not service.shutdown_hook_called
    ), "Shutdown hook was incorrectly called during startup"


def test_run_shutdown_hooks():
    """Test that shutdown hooks are properly called."""
    # Create an instance of our mock service
    service = MockServiceWithHooks()

    # Run the shutdown hooks
    run_shutdown_hooks(service)

    # Verify that shutdown hooks were called
    assert service.shutdown_hook_called, "Shutdown hook was not called"

    # Verify that startup hooks were not called during shutdown
    assert (
        not service.sync_hook_called
    ), "Startup hook was incorrectly called during shutdown"
    assert (
        not service.async_hook_called
    ), "Startup hook was incorrectly called during shutdown"
    assert (
        not service.another_hook_called
    ), "Startup hook was incorrectly called during shutdown"


@pytest.mark.asyncio
async def test_run_startup_hooks_with_exception():
    """Test that exceptions in startup hooks are properly propagated."""

    # Create a mock service with a failing startup hook
    class MockServiceWithFailingHook:
        @async_on_start
        async def failing_hook(self):
            raise ValueError("Hook failure")

    service = MockServiceWithFailingHook()

    # Run the startup hooks and expect an exception
    with pytest.raises(ValueError, match="Hook failure"):
        await run_startup_hooks(service)


def test_run_shutdown_hooks_with_exception():
    """Test that exceptions in shutdown hooks are properly propagated."""

    # Create a mock service with a failing shutdown hook
    class MockServiceWithFailingShutdownHook:
        @on_shutdown
        def failing_hook(self):
            raise ValueError("Shutdown hook failure")

    service = MockServiceWithFailingShutdownHook()

    # Run the shutdown hooks and expect an exception
    with pytest.raises(ValueError, match="Shutdown hook failure"):
        run_shutdown_hooks(service)


@pytest.mark.asyncio
async def test_hooks_called_once_with_http_server():
    """Test that hooks are called exactly once when using both Dynamo worker and HTTP server."""

    # Create our mock service with HTTP endpoints
    service = MockServiceWithHooks()

    # First simulate the dyn_worker creating the instance and running startup hooks
    await run_startup_hooks(service)

    # Now simulate the web_worker using the same instance
    # This should not run the startup hooks again
    added_routes = add_fastapi_routes(service.app, service, service)

    # Verify that startup hooks were called exactly once
    assert (
        service.sync_hook_count == 1
    ), "Sync startup hook should be called exactly once"
    assert (
        service.async_hook_count == 1
    ), "Async startup hook should be called exactly once"
    assert (
        service.another_hook_count == 1
    ), "Another startup hook should be called exactly once"
    assert len(added_routes) > 0, "HTTP routes should be added"
