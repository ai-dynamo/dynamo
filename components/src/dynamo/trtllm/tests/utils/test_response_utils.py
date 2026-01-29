# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for ResponseUtils."""

from dataclasses import dataclass
from typing import Any

import pytest

from dynamo.trtllm.utils.response_utils import ResponseUtils

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
]


# Mock classes to simulate TRTLLM output structures
@dataclass
class MockLogProb:
    """Mock TRTLLM LogProb object."""

    logprob: float
    rank: int = 0
    decoded_token: str | None = None


@dataclass
class MockCompletionOutput:
    """Mock TRTLLM CompletionOutput object."""

    logprobs: Any
    token_ids: list[int] | None = None


class TestExtractLogprobs:
    """Tests for ResponseUtils.extract_logprobs"""

    def test_returns_none_when_logprobs_is_none(self):
        """Should return (None, None) when output.logprobs is None."""
        output = MockCompletionOutput(logprobs=None)
        log_probs, top_logprobs = ResponseUtils.extract_logprobs(output, 0)
        assert log_probs is None
        assert top_logprobs is None

    def test_returns_none_when_no_new_tokens(self):
        """Should return (None, None) when all tokens have been processed."""
        output = MockCompletionOutput(logprobs=[-1.0, -2.0, -3.0])
        log_probs, top_logprobs = ResponseUtils.extract_logprobs(output, 3)
        assert log_probs is None
        assert top_logprobs is None

    def test_handles_list_of_floats(self):
        """Should handle edge case where logprobs is a simple list of floats."""
        output = MockCompletionOutput(logprobs=[-1.0, -2.0, -3.0, -4.0])
        log_probs, top_logprobs = ResponseUtils.extract_logprobs(output, 2)
        assert log_probs == [-3.0, -4.0]
        assert top_logprobs is None

    def test_handles_token_logprobs_dict(self):
        """Should extract logprobs from TokenLogprobs format."""
        # Simulate TRTLLM TokenLogprobs: dict mapping token_id -> LogProb
        token_logprobs = [
            {100: MockLogProb(logprob=-1.5, rank=1)},
            {101: MockLogProb(logprob=-2.5, rank=1)},
        ]
        output = MockCompletionOutput(
            logprobs=token_logprobs,
            token_ids=[100, 101],
        )
        log_probs, top_logprobs = ResponseUtils.extract_logprobs(output, 0)
        assert log_probs == [-1.5, -2.5]
        assert len(top_logprobs) == 2
        assert top_logprobs[0][0]["token_id"] == 100
        assert top_logprobs[0][0]["logprob"] == -1.5

    def test_handles_partial_extraction(self):
        """Should only extract logprobs for new tokens."""
        token_logprobs = [
            {100: MockLogProb(logprob=-1.0, rank=1)},
            {101: MockLogProb(logprob=-2.0, rank=1)},
            {102: MockLogProb(logprob=-3.0, rank=1)},
        ]
        output = MockCompletionOutput(
            logprobs=token_logprobs,
            token_ids=[100, 101, 102],
        )
        log_probs, top_logprobs = ResponseUtils.extract_logprobs(output, 1)
        assert log_probs == [-2.0, -3.0]
        assert len(top_logprobs) == 2

    def test_handles_top_logprobs_with_multiple_candidates(self):
        """Should extract all top logprob candidates."""
        token_logprobs = [
            {
                100: MockLogProb(logprob=-1.0, rank=1, decoded_token="hello"),
                200: MockLogProb(logprob=-2.0, rank=2, decoded_token="hi"),
                300: MockLogProb(logprob=-3.0, rank=3, decoded_token="hey"),
            },
        ]
        output = MockCompletionOutput(
            logprobs=token_logprobs,
            token_ids=[100],
        )
        log_probs, top_logprobs = ResponseUtils.extract_logprobs(output, 0)
        assert log_probs == [-1.0]
        assert len(top_logprobs) == 1
        assert len(top_logprobs[0]) == 3
        # Check all candidates are present
        token_ids = {entry["token_id"] for entry in top_logprobs[0]}
        assert token_ids == {100, 200, 300}

    def test_fallback_when_selected_token_not_in_dict(self):
        """Should use first logprob if selected token not found."""
        token_logprobs = [
            {200: MockLogProb(logprob=-2.0, rank=1)},  # token 100 not in dict
        ]
        output = MockCompletionOutput(
            logprobs=token_logprobs,
            token_ids=[100],  # actual token is 100
        )
        log_probs, _ = ResponseUtils.extract_logprobs(output, 0)
        assert log_probs == [-2.0]  # falls back to first entry

    def test_skips_none_entries_in_logprobs(self):
        """Should skip None entries in the logprobs list."""
        token_logprobs = [
            {100: MockLogProb(logprob=-1.0, rank=1)},
            None,  # Should be skipped
            {102: MockLogProb(logprob=-3.0, rank=1)},
        ]
        output = MockCompletionOutput(
            logprobs=token_logprobs,
            token_ids=[100, 101, 102],
        )
        log_probs, top_logprobs = ResponseUtils.extract_logprobs(output, 0)
        assert log_probs == [-1.0, -3.0]
        assert len(top_logprobs) == 2

    def test_handles_logprob_without_rank_attribute(self):
        """Should default rank to 0 if not present."""

        @dataclass
        class LogProbNoRank:
            logprob: float

        token_logprobs = [{100: LogProbNoRank(logprob=-1.5)}]
        output = MockCompletionOutput(
            logprobs=token_logprobs,
            token_ids=[100],
        )
        _, top_logprobs = ResponseUtils.extract_logprobs(output, 0)
        assert top_logprobs[0][0]["rank"] == 0

    def test_handles_logprob_without_decoded_token(self):
        """Should default decoded_token to None if not present."""

        @dataclass
        class LogProbNoToken:
            logprob: float
            rank: int = 1

        token_logprobs = [{100: LogProbNoToken(logprob=-1.5)}]
        output = MockCompletionOutput(
            logprobs=token_logprobs,
            token_ids=[100],
        )
        _, top_logprobs = ResponseUtils.extract_logprobs(output, 0)
        assert top_logprobs[0][0]["token"] is None

    def test_returns_none_for_empty_results(self):
        """Should return None instead of empty lists."""
        # All entries are None
        token_logprobs = [None, None, None]
        output = MockCompletionOutput(
            logprobs=token_logprobs,
            token_ids=[100, 101, 102],
        )
        log_probs, top_logprobs = ResponseUtils.extract_logprobs(output, 0)
        assert log_probs is None
        assert top_logprobs is None
