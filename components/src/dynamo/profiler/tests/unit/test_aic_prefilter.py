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

"""Unit tests for AIC-based THOROUGH pre-filter."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from dynamo.profiler.utils.aic_prefilter import (
    prefilter_decode_candidates,
    prefilter_prefill_candidates,
)


@dataclass
class _FakeCandidate:
    tp: int
    pp: int
    dp: int
    moe_tp: int
    moe_ep: int
    num_gpus: int


def _make_candidates(n: int) -> list[_FakeCandidate]:
    return [
        _FakeCandidate(tp=i + 1, pp=1, dp=1, moe_tp=1, moe_ep=1, num_gpus=i + 1)
        for i in range(n)
    ]


class TestPrefilterPrefill:
    def test_noop_when_top_n_none(self):
        candidates = _make_candidates(5)
        result = prefilter_prefill_candidates(
            candidates, None, "model", "sys", "backend", 128, 128
        )
        assert result == candidates

    def test_noop_when_candidates_within_top_n(self):
        candidates = _make_candidates(3)
        result = prefilter_prefill_candidates(
            candidates, 5, "model", "sys", "backend", 128, 128
        )
        assert result == candidates

    @patch("dynamo.profiler.utils.aic_prefilter.TaskRunner")
    @patch("dynamo.profiler.utils.aic_prefilter.TaskConfig")
    def test_selects_top_n_by_ttft(self, mock_tc_cls, mock_runner_cls):
        candidates = _make_candidates(5)
        runner_instance = MagicMock()
        mock_runner_cls.return_value = runner_instance
        runner_instance.run.side_effect = [
            {"ttft": 50.0},
            {"ttft": 10.0},
            {"ttft": 30.0},
            {"ttft": 20.0},
            {"ttft": 40.0},
        ]

        result = prefilter_prefill_candidates(
            candidates, 3, "model", "sys", "backend", 128, 128
        )

        assert len(result) == 3
        assert result[0] is candidates[1]  # ttft=10
        assert result[1] is candidates[3]  # ttft=20
        assert result[2] is candidates[2]  # ttft=30

    @patch(
        "dynamo.profiler.utils.aic_prefilter.TaskRunner",
        side_effect=Exception("AIC unavailable"),
    )
    def test_fallback_on_aic_error(self, _mock):
        candidates = _make_candidates(5)
        result = prefilter_prefill_candidates(
            candidates, 3, "model", "sys", "backend", 128, 128
        )
        assert len(result) == 5


class TestPrefilterDecode:
    @patch("dynamo.profiler.utils.aic_prefilter.TaskRunner")
    @patch("dynamo.profiler.utils.aic_prefilter.TaskConfig")
    def test_selects_top_n_by_throughput_descending(self, mock_tc_cls, mock_runner_cls):
        candidates = _make_candidates(5)
        runner_instance = MagicMock()
        mock_runner_cls.return_value = runner_instance
        runner_instance.run.side_effect = [
            {"thpt_per_gpu": 100.0},
            {"thpt_per_gpu": 500.0},
            {"thpt_per_gpu": 300.0},
            {"thpt_per_gpu": 200.0},
            {"thpt_per_gpu": 400.0},
        ]

        result = prefilter_decode_candidates(
            candidates, 3, "model", "sys", "backend", 128, 128
        )

        assert len(result) == 3
        assert result[0] is candidates[1]  # thpt=500
        assert result[1] is candidates[4]  # thpt=400
        assert result[2] is candidates[2]  # thpt=300

    @patch(
        "dynamo.profiler.utils.aic_prefilter.TaskRunner",
        side_effect=Exception("AIC unavailable"),
    )
    def test_fallback_on_aic_error(self, _mock):
        candidates = _make_candidates(5)
        result = prefilter_decode_candidates(
            candidates, 3, "model", "sys", "backend", 128, 128
        )
        assert len(result) == 5

    def test_noop_when_top_n_zero(self):
        candidates = _make_candidates(5)
        result = prefilter_decode_candidates(
            candidates, 0, "model", "sys", "backend", 128, 128
        )
        assert result == candidates
