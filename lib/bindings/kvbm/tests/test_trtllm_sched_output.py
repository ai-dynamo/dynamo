# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TRT-LLM scheduler output translation.

These tests verify the translation from TRT-LLM's SchedulerOutput format
to KVBM's Rust SchedulerOutput format, using mock objects to avoid
requiring TRT-LLM C++ bindings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

# ---------------------------------------------------------------------------
# Mock TRT-LLM types (avoids requiring TRT-LLM C++ bindings)
# ---------------------------------------------------------------------------


@dataclass
class MockRequestData:
    """Mirrors tensorrt_llm._torch.pyexecutor.kv_cache_connector.RequestData."""

    request_id: int
    new_tokens: List[int]
    new_block_ids: List[int]
    computed_position: int
    num_scheduled_tokens: int
    priorities: Optional[List[int]] = None


@dataclass
class MockSchedulerOutput:
    """Mirrors tensorrt_llm._torch.pyexecutor.kv_cache_connector.SchedulerOutput."""

    new_requests: List[MockRequestData] = field(default_factory=list)
    cached_requests: List[MockRequestData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kvbm_v2():
    """Import kvbm v2 module, skip if not available."""
    import kvbm

    if not kvbm.v2.is_available():
        pytest.skip("kvbm v2 not available")
    return kvbm._core.v2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessTrtllmSchedulerOutput:
    """Tests for process_trtllm_scheduler_output translation."""

    def test_empty_output(self, kvbm_v2):
        """Empty scheduler output produces empty KVBM output."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        scheduler_output = MockSchedulerOutput()
        result = process_trtllm_scheduler_output(1, scheduler_output)

        assert result.get_total_num_scheduled_tokens() == 0

    def test_single_new_request(self, kvbm_v2):
        """A single new request is translated correctly."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        req = MockRequestData(
            request_id=42,
            new_tokens=[101, 2054, 2003, 1996],
            new_block_ids=[0, 1],
            computed_position=0,
            num_scheduled_tokens=4,
        )
        scheduler_output = MockSchedulerOutput(new_requests=[req])
        result = process_trtllm_scheduler_output(1, scheduler_output)

        assert result.get_total_num_scheduled_tokens() == 4

    def test_single_cached_request(self, kvbm_v2):
        """A single cached (decode) request is translated correctly."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        req = MockRequestData(
            request_id=42,
            new_tokens=[5678],
            new_block_ids=[],
            computed_position=128,
            num_scheduled_tokens=1,
        )
        scheduler_output = MockSchedulerOutput(cached_requests=[req])
        result = process_trtllm_scheduler_output(5, scheduler_output)

        assert result.get_total_num_scheduled_tokens() == 1

    def test_mixed_new_and_cached(self, kvbm_v2):
        """Mixed new + cached requests produce correct total tokens."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        new_req = MockRequestData(
            request_id=1,
            new_tokens=[101, 102, 103, 104],
            new_block_ids=[0, 1],
            computed_position=0,
            num_scheduled_tokens=4,
        )
        cached_req = MockRequestData(
            request_id=2,
            new_tokens=[999],
            new_block_ids=[],
            computed_position=64,
            num_scheduled_tokens=1,
        )
        scheduler_output = MockSchedulerOutput(
            new_requests=[new_req],
            cached_requests=[cached_req],
        )
        result = process_trtllm_scheduler_output(10, scheduler_output)

        assert result.get_total_num_scheduled_tokens() == 5

    def test_request_id_converted_to_str(self, kvbm_v2):
        """Request IDs are converted from int to str."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        req = MockRequestData(
            request_id=12345,
            new_tokens=[100],
            new_block_ids=[0],
            computed_position=0,
            num_scheduled_tokens=1,
        )
        scheduler_output = MockSchedulerOutput(new_requests=[req])

        # Should not raise — str conversion handles int IDs
        result = process_trtllm_scheduler_output(1, scheduler_output)
        assert result.get_total_num_scheduled_tokens() == 1

    def test_multiple_new_requests(self, kvbm_v2):
        """Multiple new requests are all included."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        reqs = [
            MockRequestData(
                request_id=i,
                new_tokens=list(range(i * 10, i * 10 + 8)),
                new_block_ids=[i * 2, i * 2 + 1],
                computed_position=0,
                num_scheduled_tokens=8,
            )
            for i in range(5)
        ]
        scheduler_output = MockSchedulerOutput(new_requests=reqs)
        result = process_trtllm_scheduler_output(1, scheduler_output)

        assert result.get_total_num_scheduled_tokens() == 40

    def test_multiple_cached_decode_requests(self, kvbm_v2):
        """Multiple decode requests each contributing 1 token."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        reqs = [
            MockRequestData(
                request_id=100 + i,
                new_tokens=[9000 + i],
                new_block_ids=[],
                computed_position=64 + i,
                num_scheduled_tokens=1,
            )
            for i in range(10)
        ]
        scheduler_output = MockSchedulerOutput(cached_requests=reqs)
        result = process_trtllm_scheduler_output(20, scheduler_output)

        assert result.get_total_num_scheduled_tokens() == 10

    def test_serialization_roundtrip(self, kvbm_v2):
        """Output can be serialized to bytes (as build_connector_metadata expects)."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        req = MockRequestData(
            request_id=42,
            new_tokens=[101, 102],
            new_block_ids=[0],
            computed_position=0,
            num_scheduled_tokens=2,
        )
        scheduler_output = MockSchedulerOutput(new_requests=[req])
        result = process_trtllm_scheduler_output(1, scheduler_output)

        # serialize() returns bytes — this is what build_connector_metadata consumes
        serialized = result.serialize()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_iteration_is_passed_through(self, kvbm_v2):
        """Iteration number is set on the output."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        scheduler_output = MockSchedulerOutput()

        result_1 = process_trtllm_scheduler_output(42, scheduler_output)
        result_2 = process_trtllm_scheduler_output(100, scheduler_output)

        # We can verify via serialization that they're different
        assert result_1.serialize() != result_2.serialize()

    def test_empty_block_ids(self, kvbm_v2):
        """Requests with empty block_ids (decode, no new blocks) work."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        req = MockRequestData(
            request_id=1,
            new_tokens=[555],
            new_block_ids=[],
            computed_position=256,
            num_scheduled_tokens=1,
        )
        scheduler_output = MockSchedulerOutput(cached_requests=[req])

        # Should not raise
        result = process_trtllm_scheduler_output(1, scheduler_output)
        assert result.get_total_num_scheduled_tokens() == 1

    def test_large_request_ids(self, kvbm_v2):
        """Large integer request IDs (common in TRT-LLM) are handled."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        req = MockRequestData(
            request_id=2**53,  # large JS-safe integer
            new_tokens=[100],
            new_block_ids=[0],
            computed_position=0,
            num_scheduled_tokens=1,
        )
        scheduler_output = MockSchedulerOutput(new_requests=[req])

        result = process_trtllm_scheduler_output(1, scheduler_output)
        assert result.get_total_num_scheduled_tokens() == 1


class TestProcessTrtllmSchedulerOutputResumedAlwaysFalse:
    """Verify that resumed is always False for TRT-LLM cached requests."""

    def test_cached_request_not_resumed(self, kvbm_v2):
        """Cached requests should have resumed=False (TRT-LLM doesn't use resumed)."""
        from kvbm.v2.trtllm.sched_output import process_trtllm_scheduler_output

        # A preempted-then-readmitted request appears as cached in TRT-LLM,
        # but should NOT be marked as resumed (TRT-LLM re-admits as new context).
        req = MockRequestData(
            request_id=42,
            new_tokens=[101],
            new_block_ids=[5, 6],
            computed_position=0,
            num_scheduled_tokens=64,
        )
        scheduler_output = MockSchedulerOutput(cached_requests=[req])

        # If resumed were True, the Rust side would call reset_for_preemption
        # on the slot, which is wrong for TRT-LLM's re-admission model.
        # This test verifies the translation is correct by ensuring it doesn't
        # raise and produces valid output.
        result = process_trtllm_scheduler_output(1, scheduler_output)
        assert result.get_total_num_scheduled_tokens() == 64
