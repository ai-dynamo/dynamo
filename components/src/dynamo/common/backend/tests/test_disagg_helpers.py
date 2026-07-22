# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared disagg request helpers."""

from __future__ import annotations

import pytest

from dynamo.common.backend.disagg import (
    enforce_prefill_max_tokens,
    extract_prefill_result,
    require_prefill_result,
)
from dynamo.common.constants import DisaggregationMode

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_enforce_prefill_max_tokens_clamps_to_one():
    request = {
        "token_ids": [1, 2, 3],
        "stop_conditions": {"max_tokens": 64},
    }
    enforce_prefill_max_tokens(request)
    assert request["stop_conditions"]["max_tokens"] == 1


def test_enforce_prefill_max_tokens_creates_missing_section():
    # Some test fixtures omit `stop_conditions` entirely. The helper must
    # create it rather than KeyError-ing on the lookup, otherwise engines
    # would have to guard every call site.
    request = {"token_ids": [1, 2, 3]}
    enforce_prefill_max_tokens(request)
    assert request["stop_conditions"]["max_tokens"] == 1


def test_extract_prefill_result_returns_none_when_absent():
    assert extract_prefill_result({"token_ids": []}) is None


def test_extract_prefill_result_passes_through_value():
    payload = {"disaggregated_params": {"handle": "abc"}}
    out = extract_prefill_result({"token_ids": [], "prefill_result": payload})
    assert out == payload


def test_require_prefill_result_returns_value_when_present():
    payload = {"disaggregated_params": {"handle": "abc"}}
    out = require_prefill_result(
        {"token_ids": [], "prefill_result": payload},
        DisaggregationMode.DECODE,
    )
    assert out == payload


def test_require_prefill_result_raises_when_absent():
    # Decode workers can't proceed without the prefill peer's payload —
    # surface the misconfiguration eagerly. This pre-condition lives in
    # the shared helper so every backend's decode path can rely on it.
    with pytest.raises(ValueError, match="prefill_result"):
        require_prefill_result({"token_ids": []}, DisaggregationMode.DECODE)
