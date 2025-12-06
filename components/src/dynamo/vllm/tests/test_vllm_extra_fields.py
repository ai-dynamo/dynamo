# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for extra_fields handling in vLLM handlers."""

import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Filter Pydantic deprecation warning before importing handlers
warnings.filterwarnings(
    "ignore",
    message=".*json_encoders.*is deprecated.*",
    category=DeprecationWarning,
)

from dynamo.vllm.handlers import (  # noqa: E402
    DecodeWorkerHandler,
    PrefillWorkerHandler,
    _request_contains_timing_metrics,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class TestShouldIncludeTimingMetrics:
    """Tests for _request_contains_timing_metrics helper function."""

    def test_returns_true_with_multiple_extra_fields(self):
        """Timing metrics should be included when explicitly requested."""
        request = {"extra_fields": ["worker_id", "timing_metrics", "other_field"]}
        assert _request_contains_timing_metrics(request) is True

    def test_returns_false_when_extra_fields_is_none(self):
        """Timing metrics should not be included when extra_fields is None."""
        request = {"extra_fields": None}
        assert _request_contains_timing_metrics(request) is False

    def test_returns_false_when_extra_fields_missing(self):
        """Timing metrics should not be included when extra_fields key is absent."""
        request: dict[str, list[str]] = {}
        assert _request_contains_timing_metrics(request) is False


def make_mock_request_output(
    token_ids: list[int],
    finish_reason: str | None = None,
    prompt_token_ids: list[int] | None = None,
):
    """Create a mock vLLM RequestOutput."""
    output = MagicMock()
    output.token_ids = token_ids
    output.finish_reason = finish_reason
    output.stop_reason = None

    request_output = MagicMock()
    request_output.outputs = [output]
    request_output.prompt_token_ids = prompt_token_ids or [1, 2, 3]
    request_output.num_cached_tokens = 0
    request_output.kv_transfer_params = None
    return request_output


def create_mock_handler(handler_class: type):
    """Create a handler with mocked dependencies."""
    runtime = MagicMock()
    component = MagicMock()
    engine = MagicMock()
    default_sampling_params: dict[str, str] = {}

    with patch("dynamo.vllm.handlers.VllmEngineMonitor"):
        with patch("dynamo.vllm.handlers.ImageLoader"):
            handler = handler_class(
                runtime=runtime,
                component=component,
                engine=engine,
                default_sampling_params=default_sampling_params,
                model_max_len=4096,
            )
    return handler


def create_mock_context(request_id: str = "test-request-123"):
    """Create a mock context that doesn't trigger abort."""
    context = MagicMock()
    context.id.return_value = request_id
    # Make async_killed_or_stopped hang forever (never abort)
    context.async_killed_or_stopped = AsyncMock(side_effect=asyncio.CancelledError)
    return context


class TestDecodeWorkerHandlerTiming:
    """E2E tests for timing metrics in DecodeWorkerHandler."""

    @pytest.mark.asyncio
    async def test_no_timing_metrics_when_not_requested(self):
        """When timing_metrics not requested, no timing data in output."""
        handler = create_mock_handler(DecodeWorkerHandler)
        context = create_mock_context()

        final_output = make_mock_request_output([100], finish_reason="stop")

        async def mock_generate(*args, **kwargs):
            yield final_output

        handler.engine_client.generate = mock_generate

        request = {
            "token_ids": [1, 2, 3],
            "sampling_options": {},
            "stop_conditions": {},
        }

        results = []
        async for output in handler.generate(request, context):
            results.append(output)

        final = results[-1]
        assert (
            final.get("disaggregated_params") is None
            or final.get("disaggregated_params", {}).get("timing_metrics") is None
        )

    @pytest.mark.asyncio
    async def test_disaggregated_mode_preserves_frontend_timestamp(self):
        """In disaggregated mode, frontend's request_received_seconds is preserved."""
        handler = create_mock_handler(DecodeWorkerHandler)
        context = create_mock_context()

        final_output = make_mock_request_output([100], finish_reason="stop")

        async def mock_generate(*args, **kwargs):
            yield final_output

        handler.engine_client.generate = mock_generate

        request = {
            "token_ids": [1, 2, 3],
            "sampling_options": {},
            "stop_conditions": {},
            "extra_fields": ["timing_metrics"],
            "request_received_seconds": 1000.0,
            "prefill_result": {
                "disaggregated_params": {
                    "timing_metrics": {
                        "request_received_seconds": 999.0,
                        "prefill_start_seconds": 1001.0,
                        "prefill_end_seconds": 1002.0,
                    }
                }
            },
        }

        results = []
        async for output in handler.generate(request, context):
            results.append(output)

        timing = results[-1]["disaggregated_params"]["timing_metrics"]

        # Frontend's timestamp must be preserved
        assert timing["request_received_seconds"] == 1000.0
        # Prefill timing should be merged
        assert timing["prefill_start_seconds"] == 1001.0
        assert timing["prefill_end_seconds"] == 1002.0


class TestPrefillWorkerHandlerTiming:
    """E2E tests for timing metrics in PrefillWorkerHandler."""

    @pytest.mark.asyncio
    async def test_timing_metrics_included_in_prefill_output(self):
        """When timing_metrics requested, prefill output contains timing data."""
        handler = create_mock_handler(PrefillWorkerHandler)
        context = create_mock_context()

        prefill_output = make_mock_request_output([100])
        prefill_output.kv_transfer_params = {"some": "params"}

        async def mock_generate(*args, **kwargs):
            yield prefill_output

        handler.engine_client.generate = mock_generate

        request = {
            "token_ids": [1, 2, 3],
            "sampling_options": {},
            "stop_conditions": {},
            "extra_fields": ["timing_metrics"],
            "request_received_seconds": 1000.0,
        }

        results = []
        async for output in handler.generate(request, context):
            results.append(output)

        timing = results[-1]["disaggregated_params"]["timing_metrics"]

        assert timing["request_received_seconds"] == 1000.0
        assert "prefill_start_seconds" in timing
        assert "prefill_end_seconds" in timing
