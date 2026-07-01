# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GeneratePayload.validate token-count logic (RL TITO)."""

from unittest.mock import MagicMock

import pytest

from tests.utils.payloads import GeneratePayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _resp(token_ids, finish_reason="stop", request_id="r1"):
    r = MagicMock()
    r.json.return_value = {
        "request_id": request_id,
        "choices": [
            {"index": 0, "token_ids": token_ids, "finish_reason": finish_reason}
        ],
    }
    return r


def _payload(sampling_params, expected_finish_reason=None):
    return GeneratePayload(
        body={"token_ids": [1, 2], "sampling_params": sampling_params},
        expected_response=[],
        expected_log=[],
        expected_finish_reason=expected_finish_reason,
    )


def test_bounded_count_allows_early_stop_within_range():
    # ignore_eos=false -> bounded; a count within [min_tokens, max_tokens] passes.
    p = _payload({"max_tokens": 8, "min_tokens": 2}, expected_finish_reason="stop")
    p.validate(_resp([10, 11, 12], "stop"), "")


def test_bounded_count_rejects_below_min_tokens():
    p = _payload({"max_tokens": 8, "min_tokens": 4})
    with pytest.raises(AssertionError):
        p.validate(_resp([10, 11], "stop"), "")  # 2 < min_tokens 4


def test_ignore_eos_with_stop_uses_bounded_not_exact():
    # ignore_eos=true BUT an explicit stop_token_ids -> bounded, so a short
    # count passes (ignore_eos only suppresses EOS, not explicit stops).
    p = _payload(
        {"max_tokens": 8, "ignore_eos": True, "stop_token_ids": [13]},
        expected_finish_reason="stop",
    )
    p.validate(_resp([10, 13], "stop"), "")


def test_ignore_eos_no_stop_requires_exact_count():
    # ignore_eos=true, no stop -> exact count; a non-max count must fail.
    p = _payload({"max_tokens": 8, "ignore_eos": True}, expected_finish_reason="length")
    with pytest.raises(AssertionError):
        p.validate(_resp([1, 2, 3], "length"), "")  # 3 != 8
