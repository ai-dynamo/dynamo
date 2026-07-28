# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def normalize():
    from dynamo.openai_backend.worker import _normalize_usage_reasoning_tokens

    return _normalize_usage_reasoning_tokens


def test_top_level_reasoning_tokens_move_into_details(normalize):
    payload = {
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 300,
            "total_tokens": 315,
            "reasoning_tokens": 300,
        }
    }

    normalize(payload)

    assert "reasoning_tokens" not in payload["usage"]
    assert payload["usage"]["completion_tokens_details"] == {"reasoning_tokens": 300}


def test_existing_nested_count_is_not_clobbered(normalize):
    payload = {
        "usage": {
            "completion_tokens": 10,
            "reasoning_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 4},
        }
    }

    normalize(payload)

    assert "reasoning_tokens" not in payload["usage"]
    assert payload["usage"]["completion_tokens_details"] == {"reasoning_tokens": 4}


def test_existing_details_object_gains_the_count(normalize):
    payload = {
        "usage": {
            "completion_tokens": 10,
            "reasoning_tokens": 7,
            "completion_tokens_details": {"accepted_prediction_tokens": 2},
        }
    }

    normalize(payload)

    assert payload["usage"]["completion_tokens_details"] == {
        "accepted_prediction_tokens": 2,
        "reasoning_tokens": 7,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"usage": None},
        {"usage": {"completion_tokens": 10}},
        {"usage": {"reasoning_tokens": "300"}},
        {"usage": {"reasoning_tokens": True}},
    ],
)
def test_payloads_without_a_usable_count_pass_through(normalize, payload):
    import copy

    original = copy.deepcopy(payload)

    normalize(payload)

    assert payload == original
