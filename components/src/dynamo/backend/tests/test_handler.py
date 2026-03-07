# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Handler base class.

Tests cover the reusable utilities provided by Handler:
- process_generation_output() — token delta extraction
- get_trace_header() — W3C traceparent header creation
- cleanup() — KV publisher and temp dir cleanup
- add_temp_dir() — temp dir tracking
- _cancellation_monitor() — async cancellation context manager
"""

import asyncio
import tempfile
from typing import Any, AsyncGenerator, Dict
from unittest.mock import MagicMock

import pytest

from dynamo.backend.handler import Handler
from dynamo.common import Context

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


# -- Concrete subclass for testing (Handler is abstract) ------------------


class _ConcreteHandler(Handler):
    """Minimal concrete Handler for testing base class methods."""

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"token_ids": [1]}


def _make_handler(**kwargs) -> _ConcreteHandler:
    return _ConcreteHandler(**kwargs)


# -- Helpers --------------------------------------------------------------


def _mock_output(token_ids, logprobs=None, finish_reason=None, stop_reason=None):
    """Create a mock engine output object."""
    out = MagicMock()
    out.token_ids = token_ids
    out.logprobs = logprobs
    out.finish_reason = finish_reason
    out.stop_reason = stop_reason
    return out


def _mock_context(trace_id=None, span_id=None):
    """Create a mock Context with trace fields."""
    ctx = MagicMock()
    ctx.trace_id = trace_id
    ctx.span_id = span_id
    # async_killed_or_stopped blocks until stopped — tests cancel the task
    never_done = asyncio.Future()
    ctx.async_killed_or_stopped = MagicMock(return_value=never_done)
    ctx.is_stopped = MagicMock(return_value=False)
    ctx.is_killed = MagicMock(return_value=False)
    return ctx


# =========================================================================
# process_generation_output
# =========================================================================


class TestProcessGenerationOutput:
    """Tests for Handler.process_generation_output()."""

    def test_extracts_token_delta(self):
        """Should return only new tokens since num_output_tokens_so_far."""
        output = _mock_output(token_ids=[10, 20, 30, 40])
        result, new_total = Handler.process_generation_output(output, 2)

        assert result["token_ids"] == [30, 40]
        assert new_total == 4

    def test_includes_finish_reason_when_present(self):
        """Should include finish_reason in result when output has one."""
        output = _mock_output(token_ids=[10], finish_reason="stop")
        result, _ = Handler.process_generation_output(output, 0)

        assert result["finish_reason"] == "stop"

    def test_excludes_finish_reason_when_absent(self):
        """Should not include finish_reason when output has None."""
        output = _mock_output(token_ids=[10], finish_reason=None)
        result, _ = Handler.process_generation_output(output, 0)

        assert "finish_reason" not in result

    def test_includes_stop_reason_when_present(self):
        """Should include stop_reason in result when output has one."""
        output = _mock_output(token_ids=[10], stop_reason="<|endoftext|>")
        result, _ = Handler.process_generation_output(output, 0)

        assert result["stop_reason"] == "<|endoftext|>"

    def test_returns_updated_token_count(self):
        """Second element of return tuple is the new total token count."""
        output = _mock_output(token_ids=[1, 2, 3, 4, 5])
        _, new_total = Handler.process_generation_output(output, 3)

        assert new_total == 5


# =========================================================================
# get_trace_header
# =========================================================================


class TestGetTraceHeader:
    """Tests for Handler.get_trace_header()."""

    def test_returns_traceparent_when_both_ids_present(self):
        """Should return W3C traceparent header dict."""
        handler = _make_handler()
        ctx = _mock_context(trace_id="abc123", span_id="def456")

        header = handler.get_trace_header(ctx)

        assert header == {"traceparent": "00-abc123-def456-01"}

    def test_returns_none_when_trace_id_missing(self):
        """Should return None if trace_id is not set."""
        handler = _make_handler()
        ctx = _mock_context(trace_id=None, span_id="def456")

        assert handler.get_trace_header(ctx) is None

    def test_returns_none_when_span_id_missing(self):
        """Should return None if span_id is not set."""
        handler = _make_handler()
        ctx = _mock_context(trace_id="abc123", span_id=None)

        assert handler.get_trace_header(ctx) is None

    def test_returns_none_when_both_missing(self):
        """Should return None if neither trace field is set."""
        handler = _make_handler()
        ctx = _mock_context(trace_id=None, span_id=None)

        assert handler.get_trace_header(ctx) is None


# =========================================================================
# cleanup
# =========================================================================


class TestCleanup:
    """Tests for Handler.cleanup()."""

    def test_shuts_down_kv_publishers(self):
        """Should call shutdown() on each KV publisher."""
        handler = _make_handler()
        pub1 = MagicMock()
        pub2 = MagicMock()
        handler.kv_publishers = [pub1, pub2]

        handler.cleanup()

        pub1.shutdown.assert_called_once()
        pub2.shutdown.assert_called_once()

    def test_cleans_up_temp_dirs(self):
        """Should call cleanup() on tracked temp directories."""
        handler = _make_handler()
        td = MagicMock(spec=tempfile.TemporaryDirectory)
        handler._temp_dirs = [td]

        handler.cleanup()

        td.cleanup.assert_called_once()
        assert handler._temp_dirs == []

    def test_handles_publisher_error_gracefully(self):
        """Should not raise if a publisher shutdown fails."""
        handler = _make_handler()
        bad_pub = MagicMock()
        bad_pub.shutdown.side_effect = RuntimeError("boom")
        handler.kv_publishers = [bad_pub]

        handler.cleanup()  # should not raise

    def test_handles_none_publishers(self):
        """Should not raise when kv_publishers is None."""
        handler = _make_handler()
        handler.kv_publishers = None

        handler.cleanup()  # should not raise


# =========================================================================
# add_temp_dir
# =========================================================================


class TestAddTempDir:
    """Tests for Handler.add_temp_dir()."""

    def test_adds_temp_dir(self):
        """Should append non-None temp dir to tracking list."""
        handler = _make_handler()
        td = MagicMock(spec=tempfile.TemporaryDirectory)

        handler.add_temp_dir(td)

        assert td in handler._temp_dirs

    def test_ignores_none(self):
        """Should not append None."""
        handler = _make_handler()

        handler.add_temp_dir(None)

        assert len(handler._temp_dirs) == 0


# =========================================================================
# _cancellation_monitor
# =========================================================================


class TestCancellationMonitor:
    """Tests for Handler._cancellation_monitor()."""

    @pytest.mark.asyncio
    async def test_yields_a_task(self):
        """Context manager should yield an asyncio.Task."""
        handler = _make_handler()
        ctx = _mock_context()

        async with handler._cancellation_monitor(ctx) as task:
            assert isinstance(task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_cancels_task_on_exit(self):
        """Task should be cancelled when context manager exits."""
        handler = _make_handler()
        ctx = _mock_context()

        async with handler._cancellation_monitor(ctx) as task:
            assert not task.done()

        # After exiting, task should be cancelled
        assert task.done()
