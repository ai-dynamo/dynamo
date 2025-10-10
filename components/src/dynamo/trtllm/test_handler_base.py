#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test runner for handler_base error handling.
Run with: python test_handler_base.py

This script mocks heavy dependencies before importing handler_base to test error handling.
"""
# type: ignore  # This file uses dynamic mocking which confuses mypy

import sys
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Any, AsyncGenerator

# Add both the current directory and the components/src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Add current directory
sys.path.insert(0, os.path.join(current_dir, "../../.."))  # Add components/src

# Mock all heavy dependencies BEFORE importing handler_base
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

    def id(self) -> str:
        return self._id

    def cancelled(self) -> asyncio.Future[None]:
        return self._cancelled


sys.modules["dynamo._core"].Context = Context  # type: ignore[attr-defined]

# Import handler_base directly from its location
from request_handlers.handler_base import (
    HandlerBase,
    RequestHandlerConfig,
    DisaggregationMode,
    DisaggregationStrategy,
)

import pytest


class TestHandlerBase:
    """Tests for HandlerBase error handling"""

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

        async def mock_generator(self):
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

        mock_gen_result.__aiter__ = mock_generator
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
        assert "Invalid request" in responses[1]["error"]

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
            assert "service restarting" in responses[1]["error"].lower()

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
    # Allow running with python test_handler_base.py
    pytest.main([__file__, "-v"])
