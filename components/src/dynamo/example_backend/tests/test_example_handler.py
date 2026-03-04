"""Example tests for the ExampleHandler."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from dynamo.example_backend.handlers import ExampleHandler, _FALLBACK_REPLY_IDS

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def create_mock_context(request_id: str = "test-id") -> MagicMock:
    """Create a mock Context matching the dynamo Context interface."""
    ctx = MagicMock()
    ctx.id = MagicMock(return_value=request_id)
    ctx.is_stopped = MagicMock(return_value=False)
    ctx.is_killed = MagicMock(return_value=False)
    return ctx


class TestExampleHandlerTokenDelay:
    """Tests for the token_delay feature of ExampleHandler."""

    @pytest.mark.asyncio
    async def test_no_delay_when_zero(self):
        """With token_delay=0.0, generation should complete near-instantly."""
        handler = ExampleHandler(token_delay=0.0)
        ctx = create_mock_context()

        start = time.monotonic()
        results = [r async for r in handler.generate({}, ctx)]
        elapsed = time.monotonic() - start

        assert len(results) > 0
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_delay_adds_latency(self):
        """With token_delay > 0, total time should be at least n_tokens * delay."""
        delay = 0.05
        handler = ExampleHandler(token_delay=delay)
        ctx = create_mock_context()

        start = time.monotonic()
        results = [r async for r in handler.generate({}, ctx)]
        elapsed = time.monotonic() - start

        n_tokens = len(results)
        expected_min = n_tokens * delay
        assert elapsed >= expected_min * 0.9, (
            f"Expected >= {expected_min * 0.9:.3f}s for {n_tokens} tokens "
            f"at {delay}s delay, got {elapsed:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_delay_scales_with_value(self):
        """A larger delay should produce proportionally longer generation time."""
        ctx = create_mock_context()

        # Fast
        handler_fast = ExampleHandler(token_delay=0.02)
        start = time.monotonic()
        results_fast = [r async for r in handler_fast.generate({}, ctx)]
        time_fast = time.monotonic() - start

        # Slow (3x delay)
        handler_slow = ExampleHandler(token_delay=0.06)
        ctx_slow = create_mock_context()
        start = time.monotonic()
        results_slow = [r async for r in handler_slow.generate({}, ctx_slow)]
        time_slow = time.monotonic() - start

        assert len(results_fast) == len(results_slow)
        # Slow should take roughly 3x longer (allow some tolerance)
        assert time_slow > time_fast * 1.5, (
            f"3x delay should be notably slower: fast={time_fast:.3f}s, slow={time_slow:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_yields_correct_token_format(self):
        """Each yielded dict should have token_ids, last should have finish_reason."""
        handler = ExampleHandler(token_delay=0.0)
        ctx = create_mock_context()

        results = [r async for r in handler.generate({}, ctx)]

        assert len(results) >= 1
        for r in results:
            assert "token_ids" in r
            assert isinstance(r["token_ids"], list)
            assert len(r["token_ids"]) == 1

        assert results[-1].get("finish_reason") == "stop"

        # Non-last results should not have finish_reason
        for r in results[:-1]:
            assert "finish_reason" not in r

    @pytest.mark.asyncio
    async def test_cancellation_stops_generation_during_delay(self):
        """If context is stopped mid-generation, handler should stop yielding."""
        delay = 0.1
        handler = ExampleHandler(token_delay=delay)
        ctx = create_mock_context()

        results = []
        call_count = 0

        # Stop after first token
        def check_stopped():
            return call_count > 0

        ctx.is_stopped = MagicMock(side_effect=check_stopped)

        async for r in handler.generate({}, ctx):
            results.append(r)
            call_count += 1

        # Should have yielded only 1 token before stopping
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_default_delay_is_zero(self):
        """ExampleHandler should default to token_delay=0.0."""
        handler = ExampleHandler()
        assert handler.token_delay == 0.0

    @pytest.mark.asyncio
    async def test_fallback_reply_ids_when_no_model(self):
        """With empty model, should use fallback reply IDs."""
        handler = ExampleHandler(token_delay=0.0)
        ctx = create_mock_context()

        results = [r async for r in handler.generate({}, ctx)]

        all_token_ids = [r["token_ids"][0] for r in results]
        assert all_token_ids == _FALLBACK_REPLY_IDS
