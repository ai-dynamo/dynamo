# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.trtllm.utils.request_utils import (
    apply_stop_conditions_to_sampling_params,
    request_cache_salt,
    stored_event_cache_salt,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    ("request_body", "expected"),
    [
        (
            {
                "routing": {"cache_salt": "tenant-routing"},
                "extra_args": {"nvext": {"cache_salt": "tenant-nvext"}},
            },
            "tenant-routing",
        ),
        (
            {
                "routing": {"cache_salt": ""},
                "extra_args": {"nvext": {"cache_salt": "tenant-nvext"}},
            },
            "tenant-nvext",
        ),
        (
            {
                "routing": {"cache_salt": ""},
                "extra_args": {"nvext": {"cache_salt": ""}},
            },
            None,
        ),
        ({}, None),
    ],
)
def test_request_cache_salt_precedence_and_empty_fallback(
    request_body, expected
) -> None:
    assert request_cache_salt(request_body) == expected


def test_stored_event_cache_salt_uses_per_block_schema() -> None:
    data = {
        "blocks": [
            {"cache_salt": "tenant-a"},
            {"cache_salt": "tenant-a"},
        ]
    }

    assert stored_event_cache_salt(data) == "tenant-a"


def test_stored_event_cache_salt_supports_parent_fallback() -> None:
    assert stored_event_cache_salt({"cache_salt": "tenant-a", "blocks": []}) == (
        "tenant-a"
    )


def test_stored_event_cache_salt_allows_unsalted_blocks() -> None:
    assert stored_event_cache_salt({"blocks": [{}, {}]}) is None


def test_stored_event_cache_salt_rejects_conflicting_blocks() -> None:
    data = {
        "blocks": [
            {"cache_salt": "tenant-a"},
            {"cache_salt": "tenant-b"},
        ]
    }

    with pytest.raises(ValueError, match="conflicting cache_salt"):
        stored_event_cache_salt(data)


def test_visible_stop_tokens_disable_engine_stopping() -> None:
    sampling_params = SimpleNamespace(
        ignore_eos=False,
        min_tokens=None,
        stop_token_ids=[200, 300],
    )

    apply_stop_conditions_to_sampling_params(
        sampling_params,
        {
            "stop_token_ids_visible": [200],
            "stop_token_ids_hidden": [100, 200],
        },
    )

    assert sampling_params.ignore_eos is True
    assert set(sampling_params.stop_token_ids) == {100, 300}


def test_hidden_stop_tokens_keep_engine_stopping_enabled() -> None:
    sampling_params = SimpleNamespace(
        ignore_eos=False,
        min_tokens=None,
        stop_token_ids=[300],
    )

    apply_stop_conditions_to_sampling_params(
        sampling_params,
        {"stop_token_ids_hidden": [100], "min_tokens": 2},
    )

    assert sampling_params.ignore_eos is False
    assert sampling_params.min_tokens == 2
    assert set(sampling_params.stop_token_ids) == {100, 300}


def test_no_stop_trim_leaves_stopping_to_frontend() -> None:
    sampling_params = SimpleNamespace(
        ignore_eos=False,
        min_tokens=None,
        stop_token_ids=[200, 300],
    )

    apply_stop_conditions_to_sampling_params(
        sampling_params,
        {
            "stop_token_ids_hidden": [100],
            "stop_token_ids": [200],
            "min_tokens": 2,
        },
        no_stop_trim=True,
    )

    # TRT-LLM maps ignore_eos to end_id=-1 and an empty executor stop-word list.
    assert sampling_params.ignore_eos is True
    assert sampling_params.min_tokens == 2


@pytest.mark.parametrize("hidden", [[], [163585]])
def test_k3_visible_eos_stops_in_engine_and_retains_generated_token(hidden):
    params = SimpleNamespace(ignore_eos=False, stop_token_ids=None, pad_id=None)
    apply_stop_conditions_to_sampling_params(
        params, {"stop_token_ids_visible": [163586], "stop_token_ids_hidden": hidden}
    )
    assert params.ignore_eos is False
    assert params.end_id == -1
    assert params.stop_token_ids == sorted([163586, *hidden])
    assert params.include_stop_str_in_output is True
    assert params.pad_id == 163839


@pytest.mark.parametrize(
    "constraint",
    [
        {"ignore_eos": True},
        {"min_tokens": 2},
        {"stop": ["done"]},
        {"stop_token_ids": [42]},
        {"stop_token_ids_hidden": [42]},
    ],
)
def test_k3_constrained_stopping_retains_frontend_fallback(constraint):
    params = SimpleNamespace(ignore_eos=False, stop_token_ids=None, pad_id=7)
    apply_stop_conditions_to_sampling_params(
        params, {"stop_token_ids_visible": [163586], **constraint}
    )
    assert params.ignore_eos is True
    assert params.pad_id == 7
    assert not hasattr(params, "include_stop_str_in_output")


def test_k3_no_stop_trim_retains_frontend_stopping():
    params = SimpleNamespace(ignore_eos=False, stop_token_ids=None, pad_id=None)
    apply_stop_conditions_to_sampling_params(
        params, {"stop_token_ids_visible": [163586]}, no_stop_trim=True
    )
    assert params.ignore_eos is True
    assert not hasattr(params, "include_stop_str_in_output")


@pytest.mark.parametrize(
    "field,value",
    [
        ("bad", [1]),
        ("min_tokens", 2),
        ("use_beam_search", True),
        ("stop", ["done"]),
        ("stop_token_ids", [42]),
        ("ignore_eos", True),
    ],
)
def test_k3_existing_sampling_constraints_keep_fallback(field, value):
    params = SimpleNamespace(ignore_eos=False, stop_token_ids=None, pad_id=None)
    setattr(params, field, value)
    apply_stop_conditions_to_sampling_params(
        params, {"stop_token_ids_visible": [163586]}
    )
    assert params.ignore_eos is True
    assert not hasattr(params, "include_stop_str_in_output")
