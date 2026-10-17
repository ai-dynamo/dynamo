# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared enforcement semantics, including native SGLang batches."""

from copy import deepcopy

import pytest

from dynamo.common.output_token_limit import cap_output_tokens, cap_sglang_output_tokens
from dynamo.llm.exceptions import InvalidArgument

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize(
    "requested,expected",
    [
        (None, 131072),
        (1, 1),
        (512, 512),
        (131072, 131072),
        (131073, 131072),
        (1000000, 131072),
    ],
)
def test_output_ceiling(requested, expected):
    assert cap_output_tokens(requested, 131072) == expected


@pytest.mark.parametrize("requested", [None, 16, 1000000])
def test_unconfigured_ceiling_preserves_backend_behavior(requested):
    assert cap_output_tokens(requested, None) == requested


@pytest.mark.parametrize(
    "requested,minimum", [(1000000, 131073), (512, 513), (None, 131073)]
)
def test_incompatible_minimum_is_a_client_error(requested, minimum):
    with pytest.raises(InvalidArgument, match="effective maximum output tokens"):
        cap_output_tokens(requested, 131072, minimum)


def test_minimum_at_ceiling_is_allowed():
    assert cap_output_tokens(1000000, 131072, 131072) == 131072


def test_native_batch_is_capped_without_mutating_request():
    params = [
        {"max_new_tokens": 1000000, "ignore_eos": True},
        {"max_new_tokens": 32},
        {},
    ]
    original = deepcopy(params)
    result = cap_sglang_output_tokens(params, 131072)
    assert [item["max_new_tokens"] for item in result] == [131072, 32, 131072]
    assert result[0]["ignore_eos"] is True
    assert params == original


def test_native_batch_rejects_incompatible_member():
    with pytest.raises(InvalidArgument):
        cap_sglang_output_tokens([{}, {"min_new_tokens": 131073}], 131072)


def test_native_omission_is_bounded():
    assert cap_sglang_output_tokens(None, 131072) == {"max_new_tokens": 131072}


def test_unconfigured_native_request_is_unchanged():
    params = {"max_new_tokens": None}
    assert cap_sglang_output_tokens(params, None) is params
    assert cap_sglang_output_tokens(None, None) is None


def test_tokenizer_free_sglang_floor_cannot_bypass_ceiling():
    with pytest.raises(InvalidArgument):
        cap_sglang_output_tokens({"ignore_eos": True}, 131072, min_tokens=131073)
