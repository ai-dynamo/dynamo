# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared DP-rank helpers."""

from __future__ import annotations

import pytest

from dynamo.common.backend.dp_rank import forced_dp_rank, validate_global_dp_rank

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_forced_dp_rank_returns_none_when_routing_absent():
    assert forced_dp_rank({"token_ids": [1]}) is None
    assert forced_dp_rank({"token_ids": [1], "routing": {}}) is None


def test_forced_dp_rank_coerces_to_int():
    # Router payloads come in JSON, so numeric-ish strings can appear.
    assert forced_dp_rank({"token_ids": [], "routing": {"dp_rank": "2"}}) == 2
    assert forced_dp_rank({"token_ids": [], "routing": {"dp_rank": 3}}) == 3


def test_validate_global_dp_rank_passes_in_range():
    # dp_start=2, dp_size=4 → valid global ranks are [2, 6).
    assert validate_global_dp_rank(2, 2, 4, "test") == 2
    assert validate_global_dp_rank(5, 2, 4, "test") == 5


def test_validate_global_dp_rank_returns_none_out_of_range(caplog):
    with caplog.at_level("WARNING"):
        assert validate_global_dp_rank(7, 2, 4, "test") is None
    assert "outside [2, 6)" in caplog.text


def test_validate_global_dp_rank_short_circuits_when_dp_size_one():
    # Single-rank workers can't honour a forced rank; the helper returns
    # None without warning so the engine falls back to its internal LB.
    assert validate_global_dp_rank(0, 0, 1, "test") is None
    assert validate_global_dp_rank(None, 0, 4, "test") is None
