# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test runner for handler_base error handling.
Run with: python test_handler_base.py

This script mocks heavy dependencies before importing handler_base to test error handling.

MOCK LIFECYCLE EXPLANATION:
1. Module-level mocking: Required to successfully import handler_base
   which has dependencies on tensorrt_llm, torch, etc.
2. Import handler_base: Imports work because of the mocks
3. Immediate cleanup: Removes mocks from sys.modules to prevent
   interference with pytest's test collection (prevents "tensorrt_llm.__spec__ is not set")
4. setup_method: Re-establishes mocks before each test runs
5. teardown_method: Cleans up after each test

This dual approach allows us to import handler_base (which needs mocks) while
preventing our mocks from breaking pytest's collection of other test files.
"""
# type: ignore  # This file uses dynamic mocking which confuses mypy

import asyncio
import atexit
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Add both the current directory and the components/src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Add current directory
sys.path.insert(0, os.path.join(current_dir, "../../.."))  # Add components/src

# Save original sys.modules state before mocking
original_modules = {}
modules_to_mock = [
    "torch",
    "tensorrt_llm",
    "tensorrt_llm.executor",
    "tensorrt_llm.executor.result",
    "tensorrt_llm.executor.utils",
    "tensorrt_llm.llmapi",
    "tensorrt_llm.llmapi.llm",
    "dynamo._core",
    "dynamo.logits_processing",
    "dynamo.logits_processing.examples",
    "dynamo.nixl_connect",
    "dynamo.runtime",
    "dynamo.runtime.logging",
    "dynamo.trtllm.engine",
    "dynamo.trtllm.logits_processing",
    "dynamo.trtllm.logits_processing.adapter",
    "dynamo.trtllm.multimodal_processor",
    "dynamo.trtllm.publisher",
    "dynamo.trtllm.utils",
    "dynamo.trtllm.utils.disagg_utils",
]

for module_name in modules_to_mock:
    if module_name in sys.modules:
        original_modules[module_name] = sys.modules[module_name]

# Mock all heavy dependencies BEFORE importing handler_base
# WHY WE NEED THIS: handler_base.py imports tensorrt_llm, torch, etc.
# Without these mocks, the import on line 114-119 would fail because these
# packages aren't installed in the test environment.
# This is DIFFERENT from the mocking in setup_method - this enables the import,
# while setup_method re-establishes mocks for test execution after cleanup.
sys.modules["torch"] = MagicMock()
sys.modules["tensorrt_llm"] = MagicMock()
sys.modules["tensorrt_llm.executor"] = MagicMock()
sys.modules["tensorrt_llm.executor.result"] = MagicMock()
sys.modules["tensorrt_llm.executor.utils"] = MagicMock()
sys.modules["tensorrt_llm.llmapi"] = MagicMock()
sys.modules["tensorrt_llm.llmapi.llm"] = MagicMock()


# Create RequestError exception class
class RequestError(Exception):
    """Mock RequestError from tensorrt_llm.executor.utils"""

    pass


sys.modules["tensorrt_llm.executor.utils"].RequestError = RequestError  # type: ignore[attr-defined]

# Mock other dynamo modules - but NOT dynamo.trtllm.request_handlers since we want to import the real one
sys.modules["dynamo._core"] = MagicMock()
sys.modules["dynamo.logits_processing"] = MagicMock()
sys.modules["dynamo.logits_processing.examples"] = MagicMock()
sys.modules["dynamo.nixl_connect"] = MagicMock()
sys.modules["dynamo.runtime"] = MagicMock()
sys.modules["dynamo.runtime.logging"] = MagicMock()
sys.modules["dynamo.trtllm.engine"] = MagicMock()
sys.modules["dynamo.trtllm.logits_processing"] = MagicMock()
sys.modules["dynamo.trtllm.logits_processing.adapter"] = MagicMock()
sys.modules["dynamo.trtllm.multimodal_processor"] = MagicMock()
sys.modules["dynamo.trtllm.publisher"] = MagicMock()
sys.modules["dynamo.trtllm.utils"] = MagicMock()
sys.modules["dynamo.trtllm.utils.disagg_utils"] = MagicMock()


# Mock Context class
class Context:
    """Mock Context from dynamo._core"""

    def __init__(self, request_id: str) -> None:
        self._id = request_id
        self._cancelled: asyncio.Future[None] = asyncio.Future()
        self._cancelled.set_result(None)
        # For async_killed_or_stopped - a future that never completes
        self._killed_or_stopped: asyncio.Future[None] = asyncio.Future()

    def id(self) -> str:
        return self._id

    def cancelled(self) -> asyncio.Future[None]:
        return self._cancelled

    async def async_killed_or_stopped(self) -> None:
        # The hanging behavior ensures the cancellation monitor stays "dormant"
        # and doesn't interfere with our test scenarios.
        await self._killed_or_stopped


sys.modules["dynamo._core"].Context = Context  # type: ignore[attr-defined]

import pytest  # noqa: E402

# Import handler_base directly from its location
from request_handlers.handler_base import (  # noqa: E402
    DisaggregationMode,
    DisaggregationStrategy,
    HandlerBase,
    RequestHandlerConfig,
)


def cleanup_modules():
    """Restore original sys.modules state."""
    # Remove mocked modules
    for module_name in modules_to_mock:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Restore original modules if they existed
    for module_name, original_module in original_modules.items():
        sys.modules[module_name] = original_module


# Register cleanup to run at exit
atexit.register(cleanup_modules)

# IMPORTANT: Clean up immediately after imports are done
# WHY WE CLEAN UP HERE: When pytest collects tests, it imports all test files.
# If we leave tensorrt_llm mocked in sys.modules, when pytest tries to check
# if tensorrt_llm is available for test_trtllm_unit.py (via conftest.py),
# it finds our MagicMock which doesn't have __spec__, causing:
# "ValueError: tensorrt_llm.__spec__ is not set"
# By cleaning up here, we prevent our mocks from interfering with pytest collection.
# The mocks will be re-established when tests actually run via setup_method.
cleanup_modules()


class TestHandlerBase:
    """Tests for HandlerBase error handling"""

    def setup_method(self):
        """Re-establish mocks before each test method runs.

        WHY WE NEED THIS: After cleanup_modules() removed all mocks to prevent
        pytest collection issues, we need to put them back when tests actually run.
        The HandlerBase code that was imported earlier expects these modules to be
        mocked when it executes during the test.
        """
        # Put mocks back for test execution
        sys.modules["torch"] = MagicMock()
        sys.modules["tensorrt_llm"] = MagicMock()
        sys.modules["tensorrt_llm.executor"] = MagicMock()
        sys.modules["tensorrt_llm.executor.result"] = MagicMock()
        sys.modules["tensorrt_llm.executor.utils"] = MagicMock()
        sys.modules["tensorrt_llm.llmapi"] = MagicMock()
        sys.modules["tensorrt_llm.llmapi.llm"] = MagicMock()

        # Re-create RequestError
        class RequestError(Exception):
            pass

        sys.modules["tensorrt_llm.executor.utils"].RequestError = RequestError

        # Re-mock dynamo modules
        sys.modules["dynamo._core"] = MagicMock()
        sys.modules["dynamo.logits_processing"] = MagicMock()
        sys.modules["dynamo.logits_processing.examples"] = MagicMock()
        sys.modules["dynamo.nixl_connect"] = MagicMock()
        sys.modules["dynamo.runtime"] = MagicMock()
        sys.modules["dynamo.runtime.logging"] = MagicMock()
        sys.modules["dynamo.trtllm.engine"] = MagicMock()
        sys.modules["dynamo.trtllm.logits_processing"] = MagicMock()
        sys.modules["dynamo.trtllm.logits_processing.adapter"] = MagicMock()
        sys.modules["dynamo.trtllm.multimodal_processor"] = MagicMock()
        sys.modules["dynamo.trtllm.publisher"] = MagicMock()
        sys.modules["dynamo.trtllm.utils"] = MagicMock()
        sys.modules["dynamo.trtllm.utils.disagg_utils"] = MagicMock()

        # Re-establish Context if needed
        sys.modules["dynamo._core"].Context = Context

    def teardown_method(self):
        """Clean up mocks after each test method.

        WHY WE NEED THIS: Ensures clean state between tests and prevents
        any lingering mocked modules from affecting subsequent tests or
        pytest operations.
        """
        cleanup_modules()

    def create_mock_config(self, with_runtime=True):
        """Helper to create a mock RequestHandlerConfig"""
        mock_engine = MagicMock()
        mock_engine.cleanup = AsyncMock()
        mock_engine.llm = MagicMock()  # Add llm attribute

        runtime = None
        if with_runtime:
            runtime = MagicMock()
            runtime.shutdown = MagicMock()

        mock_component = MagicMock()
        mock_component.rank = 0

        config = RequestHandlerConfig(
            component=mock_component,
            engine=mock_engine,
            default_sampling_params=MagicMock(),
            publisher=None,
            runtime=runtime,
            disaggregation_mode=DisaggregationMode.AGGREGATED,
            disaggregation_strategy=DisaggregationStrategy.PREFILL_FIRST,
            next_client=None,
            next_router_client=None,
            encode_client=None,
            multimodal_processor=None,
            connector=None,
        )
        return config

    def create_mock_generation_result(self, exception_to_raise=None):
        """Helper to create a mock generation result"""
        mock_gen_result = MagicMock()

        async def mock_generator():
            # Create a mock result that matches what handler_base expects
            mock_res = MagicMock()
            mock_res.finished = False
            mock_output = MagicMock()
            mock_output.token_ids = [1, 2, 3]
            mock_output.finish_reason = None
            mock_output.stop_reason = None
            mock_res.outputs = [mock_output]
            yield mock_res

            if exception_to_raise:
                raise exception_to_raise

        # __aiter__ should be a method that returns the async generator
        # The lambda needs to accept self (passed by MagicMock) but ignore it
        mock_gen_result.__aiter__ = lambda self: mock_generator()
        return mock_gen_result

    def get_test_request(self):
        """Helper to get a standard test request"""
        return {
            "prompt": "test",
            "sampling_options": {},
            "stop_conditions": {"max_tokens": 10},
            "trace": {"service_name": "test"},
            "tokens": [1, 2, 3],  # Mock tokens
        }

    @pytest.mark.asyncio
    async def test_request_error_no_shutdown(self):
        """Test that RequestError doesn't trigger shutdown"""
        # Setup
        config = self.create_mock_config(with_runtime=True)
        mock_engine = config.engine
        mock_runtime = config.runtime

        handler = HandlerBase(config)
        mock_context = Context("test-request-123")

        # Mock engine to raise RequestError after yielding
        mock_gen_result = self.create_mock_generation_result(
            exception_to_raise=RequestError("Invalid request parameters")
        )
        mock_engine.llm.generate_async = lambda *args, **kwargs: mock_gen_result

        # Run test
        request = self.get_test_request()
        responses = []
        async for response in handler.generate_locally(request, mock_context):
            responses.append(response)

        # Verify
        assert len(responses) == 2, f"Expected 2 responses, got {len(responses)}"
        assert responses[0]["token_ids"] == [1, 2, 3]
        assert responses[1]["finish_reason"] == "error"

        # Critical: NO shutdown should be called
        mock_runtime.shutdown.assert_not_called()
        mock_engine.cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_generic_exception_triggers_shutdown(self):
        """Test that generic exceptions trigger graceful shutdown"""
        # Setup
        config = self.create_mock_config(with_runtime=True)
        mock_engine = config.engine
        mock_runtime = config.runtime

        handler = HandlerBase(config)
        mock_context = Context("test-request-456")

        # Mock engine to raise RuntimeError
        mock_gen_result = self.create_mock_generation_result(
            exception_to_raise=RuntimeError("Engine CUDA out of memory")
        )
        mock_engine.llm.generate_async = lambda *args, **kwargs: mock_gen_result

        # Run test with mocked os._exit
        with patch("os._exit") as mock_exit:
            request = self.get_test_request()
            responses = []
            async for response in handler.generate_locally(request, mock_context):
                responses.append(response)

            # Verify error response was sent
            assert len(responses) == 2
            assert responses[1]["finish_reason"] == "error"

            # Critical: Shutdown SHOULD be called
            mock_runtime.shutdown.assert_called_once()
            mock_engine.cleanup.assert_called_once()
            mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_cancelled_error_no_shutdown(self):
        """Test that CancelledError doesn't trigger shutdown"""
        # Setup
        config = self.create_mock_config(with_runtime=True)
        mock_engine = config.engine
        mock_runtime = config.runtime

        handler = HandlerBase(config)
        mock_context = Context("test-request-789")

        # Mock engine to raise CancelledError
        mock_gen_result = self.create_mock_generation_result(
            exception_to_raise=asyncio.CancelledError("Client disconnected")
        )
        mock_engine.llm.generate_async = lambda *args, **kwargs: mock_gen_result

        # Run test
        request = self.get_test_request()
        responses = []
        async for response in handler.generate_locally(request, mock_context):
            responses.append(response)

        # Should only have the first response (no error response)
        assert len(responses) == 1
        assert responses[0]["token_ids"] == [1, 2, 3]

        # Critical: NO shutdown should be called
        mock_runtime.shutdown.assert_not_called()
        mock_engine.cleanup.assert_not_called()


if __name__ == "__main__":
    try:
        # Allow running with python test_handler_base.py
        pytest.main([__file__, "-v"])
    finally:
        # Ensure cleanup happens even if tests fail
        cleanup_modules()
