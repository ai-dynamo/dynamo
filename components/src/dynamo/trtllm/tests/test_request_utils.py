# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for request_utils.py

Tests the pure utility functions for request handling.
"""

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from dynamo.trtllm.request_handlers.request_utils import RequestUtils


class TestNormalizeRequestFormat:
    """Tests for RequestUtils.normalize_request_format"""

    def test_moves_max_tokens_to_stop_conditions(self):
        """Test that max_tokens is moved to stop_conditions."""
        request: dict[str, Any] = {"max_tokens": 100}

        RequestUtils.normalize_request_format(request)

        assert "max_tokens" not in request
        assert request["stop_conditions"]["max_tokens"] == 100

    def test_moves_temperature_to_sampling_options(self):
        """Test that temperature is moved to sampling_options."""
        request: dict[str, Any] = {"temperature": 0.7}

        RequestUtils.normalize_request_format(request)

        assert "temperature" not in request
        assert request["sampling_options"]["temperature"] == 0.7

    def test_creates_stop_conditions_if_missing(self):
        """Test that stop_conditions is created if not present."""
        request: dict[str, Any] = {}

        RequestUtils.normalize_request_format(request)

        assert "stop_conditions" in request
        assert isinstance(request["stop_conditions"], dict)

    def test_creates_sampling_options_if_missing(self):
        """Test that sampling_options is created if not present."""
        request: dict[str, Any] = {}

        RequestUtils.normalize_request_format(request)

        assert "sampling_options" in request
        assert isinstance(request["sampling_options"], dict)

    def test_does_not_overwrite_existing_stop_conditions_max_tokens(self):
        """Test that existing max_tokens in stop_conditions is not overwritten."""
        request: dict[str, Any] = {
            "max_tokens": 100,
            "stop_conditions": {"max_tokens": 50},
        }

        RequestUtils.normalize_request_format(request)

        # Original max_tokens in stop_conditions should remain
        # Top-level max_tokens is NOT moved (because nested already exists)
        assert request["stop_conditions"]["max_tokens"] == 50
        # Note: top-level max_tokens is left as-is when not moved
        assert request.get("max_tokens") == 100

    def test_does_not_overwrite_existing_sampling_options_temperature(self):
        """Test that existing temperature in sampling_options is not overwritten."""
        request: dict[str, Any] = {
            "temperature": 0.7,
            "sampling_options": {"temperature": 0.5},
        }

        RequestUtils.normalize_request_format(request)

        # Original temperature in sampling_options should remain
        # Top-level temperature is NOT moved (because nested already exists)
        assert request["sampling_options"]["temperature"] == 0.5
        # Note: top-level temperature is left as-is when not moved
        assert request.get("temperature") == 0.7

    def test_handles_complete_request(self):
        """Test normalization of a complete request with multiple fields."""
        request: dict[str, Any] = {
            "max_tokens": 100,
            "temperature": 0.7,
            "prompt": "Hello",
            "token_ids": [1, 2, 3],
        }

        RequestUtils.normalize_request_format(request)

        assert request == {
            "prompt": "Hello",
            "token_ids": [1, 2, 3],
            "stop_conditions": {"max_tokens": 100},
            "sampling_options": {"temperature": 0.7},
        }

    def test_preserves_other_fields(self):
        """Test that other fields are not affected."""
        request: dict[str, Any] = {
            "prompt": "Hello",
            "custom_field": "value",
            "max_tokens": 100,
        }

        RequestUtils.normalize_request_format(request)

        assert request["prompt"] == "Hello"
        assert request["custom_field"] == "value"


class TestExtractLogprobs:
    """Tests for RequestUtils.extract_logprobs"""

    @dataclass
    class MockLogprobInfo:
        """Mock for TRTLLM LogProb info."""

        logprob: float
        rank: int = 0
        decoded_token: Optional[str] = None

    @dataclass
    class MockOutput:
        """Mock for TRTLLM CompletionOutput."""

        logprobs: Optional[list]
        token_ids: list

    def test_returns_none_when_logprobs_is_none(self):
        """Test that None is returned when output has no logprobs."""
        output = self.MockOutput(logprobs=None, token_ids=[1, 2, 3])

        log_probs, top_logprobs = RequestUtils.extract_logprobs(output, 0)

        assert log_probs is None
        assert top_logprobs is None

    def test_returns_none_when_no_new_tokens(self):
        """Test that None is returned when no new tokens."""
        output = self.MockOutput(logprobs=[0.5, 0.6], token_ids=[1, 2])

        log_probs, top_logprobs = RequestUtils.extract_logprobs(output, 2)

        assert log_probs is None
        assert top_logprobs is None

    def test_handles_float_logprobs(self):
        """Test handling of simple float logprobs (edge case)."""
        output = self.MockOutput(logprobs=[-0.5, -0.6, -0.7], token_ids=[1, 2, 3])

        log_probs, top_logprobs = RequestUtils.extract_logprobs(output, 1)

        assert log_probs == [-0.6, -0.7]
        assert top_logprobs is None

    def test_extracts_logprobs_for_new_tokens(self):
        """Test extraction of logprobs for new tokens."""
        logprob_info = self.MockLogprobInfo(logprob=-0.5, rank=1)
        output = self.MockOutput(
            logprobs=[{10: logprob_info}, {20: logprob_info}, {30: logprob_info}],
            token_ids=[10, 20, 30],
        )

        log_probs, top_logprobs = RequestUtils.extract_logprobs(output, 1)

        assert len(log_probs) == 2
        assert log_probs[0] == -0.5
        assert len(top_logprobs) == 2

    def test_builds_top_logprobs_structure(self):
        """Test that top_logprobs has correct structure."""
        logprob_info_1 = self.MockLogprobInfo(logprob=-0.5, rank=1, decoded_token="a")
        logprob_info_2 = self.MockLogprobInfo(logprob=-1.0, rank=2, decoded_token="b")

        output = self.MockOutput(
            logprobs=[{10: logprob_info_1, 11: logprob_info_2}],
            token_ids=[10],
        )

        log_probs, top_logprobs = RequestUtils.extract_logprobs(output, 0)

        assert len(top_logprobs) == 1
        assert len(top_logprobs[0]) == 2

        # Check structure of top_logprobs entry
        entry = top_logprobs[0][0]
        assert "rank" in entry
        assert "token_id" in entry
        assert "token" in entry
        assert "logprob" in entry

    def test_uses_fallback_when_token_not_in_logprobs(self):
        """Test fallback when actual token not in logprobs dict."""
        logprob_info = self.MockLogprobInfo(logprob=-0.5, rank=1)

        # Token 10 is generated but only token 99 is in logprobs dict
        output = self.MockOutput(
            logprobs=[{99: logprob_info}],
            token_ids=[10],
        )

        log_probs, top_logprobs = RequestUtils.extract_logprobs(output, 0)

        # Should use the first (only) logprob as fallback
        assert log_probs == [-0.5]

    def test_skips_none_logprobs_entries(self):
        """Test that None entries in logprobs are skipped."""
        logprob_info = self.MockLogprobInfo(logprob=-0.5, rank=1)

        output = self.MockOutput(
            logprobs=[None, {20: logprob_info}],
            token_ids=[10, 20],
        )

        log_probs, top_logprobs = RequestUtils.extract_logprobs(output, 0)

        # Should only have one entry (None was skipped)
        assert len(log_probs) == 1
        assert log_probs[0] == -0.5
