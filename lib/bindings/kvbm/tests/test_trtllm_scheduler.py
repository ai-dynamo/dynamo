# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TrtllmConnectorScheduler.

These tests mock both the Rust ConnectorLeader and TRT-LLM's LlmRequest
to verify the scheduler's adaptation logic without requiring either
C++ binding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock TRT-LLM types
# ---------------------------------------------------------------------------


class MockLlmRequest:
    """Minimal mock of tensorrt_llm.bindings.internal.batch_manager.LlmRequest."""

    def __init__(
        self,
        request_id: int,
        tokens: List[int],
        max_new_tokens: int = 128,
    ):
        self.request_id = request_id
        self._tokens = tokens
        self.max_new_tokens = max_new_tokens
        self.is_generation_only_request = False

    def get_tokens(self, beam_idx: int) -> List[int]:
        return list(self._tokens)


@dataclass
class MockRequestData:
    request_id: int
    new_tokens: List[int]
    new_block_ids: List[int]
    computed_position: int
    num_scheduled_tokens: int
    priorities: Optional[List[int]] = None


@dataclass
class MockSchedulerOutput:
    new_requests: List[MockRequestData] = field(default_factory=list)
    cached_requests: List[MockRequestData] = field(default_factory=list)


@dataclass
class MockKVConnectorOutput:
    finished_sending: List[int] = field(default_factory=list)
    finished_recving: List[int] = field(default_factory=list)
    invalid_block_ids: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kvbm_v2():
    import kvbm

    if not kvbm.v2.is_available():
        pytest.skip("kvbm v2 not available")
    return kvbm._core.v2


@pytest.fixture
def mock_connector():
    """Create a mock Rust ConnectorLeader."""
    connector = MagicMock()
    connector.has_slot.return_value = False
    connector.get_num_new_matched_tokens.return_value = (0, False)
    connector.request_finished.return_value = False
    connector.build_connector_metadata.return_value = b"\x00"
    connector.get_slot_total_tokens.return_value = 0
    return connector


@pytest.fixture
def scheduler(kvbm_v2, mock_connector):
    """
    Create a TrtllmConnectorScheduler with mocked internals.

    Bypasses the actual KvbmRuntime construction (which requires Velo/tokio)
    by directly injecting the mock connector after construction.
    """
    from kvbm.v2.trtllm.scheduler import TrtllmConnectorScheduler

    mock_runtime = MagicMock()
    mock_runtime.instance_id.return_value = b"\x00" * 16

    # Patch both KvbmRuntime and ConnectorLeader at the kvbm.v2.connector.base
    # module level, where _ConnectorLeader and _KvbmRuntime are resolved.
    with patch("kvbm.v2.connector.base._KvbmRuntime") as MockRuntime, patch(
        "kvbm.v2.connector.base._ConnectorLeader", return_value=mock_connector
    ):
        MockRuntime.build_leader.return_value = mock_runtime

        # Also patch the module-level KvbmRuntime in scheduler.py
        with patch("kvbm.v2.trtllm.scheduler.KvbmRuntime", MockRuntime):
            mock_llm_args = MagicMock()
            mock_llm_args.kv_cache_config.tokens_per_block = 64
            sched = TrtllmConnectorScheduler(mock_llm_args)

    return sched


# ---------------------------------------------------------------------------
# Tests: _create_slot
# ---------------------------------------------------------------------------


class TestCreateSlot:
    def test_new_slot_created(self, scheduler, mock_connector):
        """First call for a request creates a slot."""
        request = MockLlmRequest(42, [101, 102, 103])

        scheduler._create_slot(request)

        mock_connector.create_slot.assert_called_once()
        kv_request = mock_connector.create_slot.call_args[0][0]
        assert kv_request.request_id == "42"
        assert "42" in scheduler._inflight_requests

    def test_existing_slot_syncs_tokens(self, scheduler, mock_connector):
        """Second call for same request syncs tokens, doesn't create new slot."""
        mock_connector.has_slot.return_value = True
        mock_connector.get_slot_total_tokens.return_value = 3

        request = MockLlmRequest(42, [101, 102, 103, 104, 105])
        scheduler._create_slot(request)

        mock_connector.create_slot.assert_not_called()
        # Should extend by 2 tokens (5 current - 3 in slot)
        mock_connector.extend_slot_tokens.assert_called_once_with("42", [104, 105])

    def test_existing_slot_no_new_tokens(self, scheduler, mock_connector):
        """Sync with same token count is a no-op."""
        mock_connector.has_slot.return_value = True
        mock_connector.get_slot_total_tokens.return_value = 3

        request = MockLlmRequest(42, [101, 102, 103])
        scheduler._create_slot(request)

        mock_connector.create_slot.assert_not_called()
        mock_connector.extend_slot_tokens.assert_not_called()

    def test_request_id_int_to_str(self, scheduler, mock_connector):
        """Integer request IDs are converted to strings for Rust."""
        request = MockLlmRequest(99999, [100])

        scheduler._create_slot(request)

        mock_connector.has_slot.assert_called_with("99999")

    def test_inflight_request_stored(self, scheduler, mock_connector):
        """LlmRequest reference is stored for potential decode offload use."""
        request = MockLlmRequest(7, [100, 200])

        scheduler._create_slot(request)

        assert scheduler._inflight_requests["7"] is request


# ---------------------------------------------------------------------------
# Tests: get_num_new_matched_tokens
# ---------------------------------------------------------------------------


class TestGetNumNewMatchedTokens:
    def test_no_match(self, scheduler, mock_connector):
        """No match returns (0, False)."""
        mock_connector.get_num_new_matched_tokens.return_value = (None, False)

        request = MockLlmRequest(1, [100, 200])
        num_tokens, async_flag = scheduler.get_num_new_matched_tokens(request, 0)

        assert num_tokens == 0
        assert async_flag is False

    def test_match_found_sync(self, scheduler, mock_connector):
        """Match found with sync loading."""
        mock_connector.get_num_new_matched_tokens.return_value = (64, False)

        request = MockLlmRequest(1, [100, 200])
        num_tokens, async_flag = scheduler.get_num_new_matched_tokens(request, 0)

        assert num_tokens == 64
        assert async_flag is False

    def test_match_found_async(self, scheduler, mock_connector):
        """Match found with async loading."""
        mock_connector.get_num_new_matched_tokens.return_value = (128, True)

        request = MockLlmRequest(1, [100, 200])
        num_tokens, async_flag = scheduler.get_num_new_matched_tokens(request, 0)

        assert num_tokens == 128
        assert async_flag is True

    def test_creates_slot_on_first_call(self, scheduler, mock_connector):
        """Slot is created automatically on first call."""
        request = MockLlmRequest(1, [100, 200])
        scheduler.get_num_new_matched_tokens(request, 0)

        mock_connector.create_slot.assert_called_once()

    def test_request_id_passed_as_str(self, scheduler, mock_connector):
        """Request ID is passed as str to Rust."""
        request = MockLlmRequest(42, [100])
        scheduler.get_num_new_matched_tokens(request, 64)

        mock_connector.get_num_new_matched_tokens.assert_called_with("42", 64)

    def test_none_result_becomes_zero(self, scheduler, mock_connector):
        """Rust's Optional[int] None is converted to 0 for TRT-LLM."""
        mock_connector.get_num_new_matched_tokens.return_value = (None, False)

        request = MockLlmRequest(1, [100])
        num_tokens, _ = scheduler.get_num_new_matched_tokens(request, 0)

        assert num_tokens == 0


# ---------------------------------------------------------------------------
# Tests: update_state_after_alloc
# ---------------------------------------------------------------------------


class TestUpdateStateAfterAlloc:
    def test_delegates_to_rust(self, scheduler, mock_connector):
        """Arguments are forwarded to Rust with str request ID."""
        request = MockLlmRequest(42, [100])
        block_ids = [0, 1, 2, 3]

        scheduler.update_state_after_alloc(request, block_ids, 128)

        mock_connector.update_state_after_alloc.assert_called_once_with(
            "42", [0, 1, 2, 3], 128
        )

    def test_zero_external_tokens(self, scheduler, mock_connector):
        """Zero external tokens (no match) is forwarded correctly."""
        request = MockLlmRequest(42, [100])

        scheduler.update_state_after_alloc(request, [0], 0)

        mock_connector.update_state_after_alloc.assert_called_once_with("42", [0], 0)


# ---------------------------------------------------------------------------
# Tests: request_finished
# ---------------------------------------------------------------------------


class TestRequestFinished:
    def test_returns_false_no_delay(self, scheduler, mock_connector):
        """No inflight offload → returns False (free blocks immediately)."""
        mock_connector.request_finished.return_value = False

        # Must create slot first so it's in _inflight_requests
        request = MockLlmRequest(42, [100])
        scheduler._create_slot(request)

        result = scheduler.request_finished(request, [0, 1])

        assert result is False
        mock_connector.request_finished.assert_called_with("42")

    def test_returns_true_delay(self, scheduler, mock_connector):
        """Inflight offload → returns True (delay block freeing)."""
        mock_connector.request_finished.return_value = True

        request = MockLlmRequest(42, [100])
        scheduler._create_slot(request)

        result = scheduler.request_finished(request, [0, 1])

        assert result is True

    def test_removes_from_inflight(self, scheduler, mock_connector):
        """Request is removed from _inflight_requests."""
        request = MockLlmRequest(42, [100])
        scheduler._create_slot(request)
        assert "42" in scheduler._inflight_requests

        scheduler.request_finished(request, [0])

        assert "42" not in scheduler._inflight_requests


# ---------------------------------------------------------------------------
# Tests: update_connector_output
# ---------------------------------------------------------------------------


class TestUpdateConnectorOutput:
    def test_int_ids_converted_to_str_sets(self, scheduler, mock_connector):
        """Integer IDs from TRT-LLM are converted to str sets for Rust."""
        output = MockKVConnectorOutput(
            finished_sending=[1, 2, 3],
            finished_recving=[4, 5],
        )

        scheduler.update_connector_output(output)

        mock_connector.update_connector_output.assert_called_once_with(
            {"1", "2", "3"},
            {"4", "5"},
        )

    def test_empty_output(self, scheduler, mock_connector):
        """Empty lists produce empty sets."""
        output = MockKVConnectorOutput()

        scheduler.update_connector_output(output)

        mock_connector.update_connector_output.assert_called_once_with(set(), set())


# ---------------------------------------------------------------------------
# Tests: build_connector_meta
# ---------------------------------------------------------------------------


class TestBuildConnectorMeta:
    def test_increments_iteration(self, scheduler, mock_connector):
        """Each call increments the iteration counter."""
        scheduler_output = MockSchedulerOutput()

        scheduler.build_connector_meta(scheduler_output)
        assert scheduler._iteration == 1

        scheduler.build_connector_meta(scheduler_output)
        assert scheduler._iteration == 2

    def test_returns_bytes(self, scheduler, mock_connector):
        """Returns serialized bytes from Rust."""
        mock_connector.build_connector_metadata.return_value = b"metadata"
        scheduler_output = MockSchedulerOutput()

        result = scheduler.build_connector_meta(scheduler_output)

        assert isinstance(result, bytes)

    def test_no_decode_offload_sync(self, scheduler, mock_connector):
        """Decode offload is disabled — no token extension calls."""
        scheduler.enable_decode_offload = False

        request = MockLlmRequest(42, [100, 200, 300])
        scheduler._create_slot(request)

        scheduler_output = MockSchedulerOutput(
            cached_requests=[
                MockRequestData(42, [400], [], 3, 1),
            ]
        )
        scheduler.build_connector_meta(scheduler_output)

        # extend_slot_tokens should NOT be called when decode offload is off
        mock_connector.extend_slot_tokens.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: decode offload (token extension from scheduler output)
# ---------------------------------------------------------------------------


class TestDecodeOffload:
    def test_extends_tokens_from_cached_request_new_tokens(
        self, scheduler, mock_connector
    ):
        """When decode offload is enabled, new_tokens from cached requests
        are applied to the Rust slot via extend_slot_tokens."""
        scheduler.enable_decode_offload = True

        scheduler_output = MockSchedulerOutput(
            cached_requests=[
                MockRequestData(42, [500, 501], [], 130, 1),
            ]
        )
        scheduler.build_connector_meta(scheduler_output)

        mock_connector.extend_slot_tokens.assert_called_once_with("42", [500, 501])

    def test_no_extension_when_new_tokens_empty(self, scheduler, mock_connector):
        """No extension when cached request has empty new_tokens."""
        scheduler.enable_decode_offload = True

        scheduler_output = MockSchedulerOutput(
            cached_requests=[
                MockRequestData(42, [], [], 128, 1),
            ]
        )
        scheduler.build_connector_meta(scheduler_output)

        mock_connector.extend_slot_tokens.assert_not_called()

    def test_extends_multiple_cached_requests(self, scheduler, mock_connector):
        """Multiple cached requests each get their tokens extended."""
        scheduler.enable_decode_offload = True

        scheduler_output = MockSchedulerOutput(
            cached_requests=[
                MockRequestData(42, [500], [], 128, 1),
                MockRequestData(43, [600, 601], [], 64, 1),
            ]
        )
        scheduler.build_connector_meta(scheduler_output)

        calls = mock_connector.extend_slot_tokens.call_args_list
        assert len(calls) == 2
        assert calls[0] == (("42", [500]),)
        assert calls[1] == (("43", [600, 601]),)

    def test_no_extension_for_new_requests(self, scheduler, mock_connector):
        """new_requests don't trigger token extension (only cached)."""
        scheduler.enable_decode_offload = True

        scheduler_output = MockSchedulerOutput(
            new_requests=[
                MockRequestData(42, [100, 200, 300], [0, 1], 0, 3),
            ]
        )
        scheduler.build_connector_meta(scheduler_output)

        mock_connector.extend_slot_tokens.assert_not_called()

    def test_disabled_by_default(self, scheduler, mock_connector):
        """Decode offload is disabled by default."""
        # Don't set enable_decode_offload — should be False from base class
        scheduler_output = MockSchedulerOutput(
            cached_requests=[
                MockRequestData(42, [500], [], 128, 1),
            ]
        )
        scheduler.build_connector_meta(scheduler_output)

        mock_connector.extend_slot_tokens.assert_not_called()

    def test_request_id_converted_to_str(self, scheduler, mock_connector):
        """Integer request IDs are converted to str for Rust."""
        scheduler.enable_decode_offload = True

        scheduler_output = MockSchedulerOutput(
            cached_requests=[
                MockRequestData(99999, [700], [], 64, 1),
            ]
        )
        scheduler.build_connector_meta(scheduler_output)

        mock_connector.extend_slot_tokens.assert_called_once_with("99999", [700])

    def test_extension_happens_before_metadata_build(self, scheduler, mock_connector):
        """Token extension happens before build_connector_metadata is called."""
        scheduler.enable_decode_offload = True
        call_order = []

        def track_extend(*args):
            call_order.append("extend")

        def track_build(*args):
            call_order.append("build")
            return b"\x00"

        mock_connector.extend_slot_tokens.side_effect = track_extend
        mock_connector.build_connector_metadata.side_effect = track_build

        scheduler_output = MockSchedulerOutput(
            cached_requests=[
                MockRequestData(42, [500], [], 128, 1),
            ]
        )
        scheduler.build_connector_meta(scheduler_output)

        assert call_order == ["extend", "build"]
