# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.deepinfer.sampling import build_sampling_kwargs
from dynamo.llm.exceptions import InvalidArgument

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def build(request, **kwargs):
    return build_sampling_kwargs(request, **kwargs)


def test_maps_dynamo_request():
    kwargs = build(
        {
            "token_ids": [5, 6, 7],
            "sampling_options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 20,
                "seed": 123,
            },
            "stop_conditions": {
                "max_tokens": 17,
                "ignore_eos": True,
                "stop_token_ids_hidden": [1, 2],
            },
            "output_options": {"logprobs": 3},
        }
    )

    assert kwargs == {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "seq_seed": 123,
        "max_tokens": 17,
        "ignore_eos": True,
        "top_logprobs": 3,
    }


@pytest.mark.parametrize("top_k", [None, 0, -1])
def test_disabled_top_k_becomes_minus_one(top_k):
    kwargs = build({"token_ids": [1], "sampling_options": {"top_k": top_k}})

    assert kwargs["top_k"] == -1


def test_unset_fields_are_left_to_deepinfer_defaults():
    kwargs = build({"token_ids": [1]})

    assert kwargs == {"top_k": -1}


def test_max_tokens_defaults_to_remaining_context():
    kwargs = build({"token_ids": [1] * 10}, model_max_len=64)

    assert kwargs["max_tokens"] == 54


def test_max_tokens_never_drops_below_one():
    kwargs = build({"token_ids": [1] * 10}, model_max_len=10)

    assert kwargs["max_tokens"] == 1


def test_neutral_values_are_accepted():
    build(
        {
            "token_ids": [1],
            "sampling_options": {
                "frequency_penalty": 0.0,
                "presence_penalty": 0,
                "repetition_penalty": 1.0,
                "min_p": 0.0,
                "n": 1,
                "logit_bias": {},
            },
            "stop_conditions": {"min_tokens": 0},
        }
    )


@pytest.mark.parametrize(
    "option",
    [
        {"frequency_penalty": 0.5},
        {"presence_penalty": -0.5},
        {"repetition_penalty": 1.2},
        {"min_p": 0.1},
        {"n": 2},
        {"logit_bias": {"5": 1.0}},
        {"guided_decoding": {"regex": "a+"}},
    ],
)
def test_unsupported_sampling_options_are_rejected(option):
    with pytest.raises(InvalidArgument, match=next(iter(option))):
        build({"token_ids": [1], "sampling_options": option})


def test_min_tokens_is_rejected():
    with pytest.raises(InvalidArgument, match="min_tokens"):
        build({"token_ids": [1], "stop_conditions": {"min_tokens": 4}})


def test_stop_token_ids_are_left_to_the_dynamo_decoder():
    # The frontend adds the model's EOS ids (from generation_config.json, which
    # may differ from config.json) to every request, and its decoder enforces
    # them, so the backend must accept whatever it is sent.
    kwargs = build(
        {
            "token_ids": [1],
            "stop_conditions": {
                "max_tokens": 4,
                "stop_token_ids_hidden": [2, 9, 128009],
                "stop_token_ids_visible": [77],
                "stop_token_ids": [5],
            },
        }
    )

    assert kwargs["max_tokens"] == 4
    assert not any("stop" in key for key in kwargs)


def test_prompt_logprobs_are_rejected():
    with pytest.raises(InvalidArgument, match="prompt_logprobs"):
        build({"token_ids": [1], "output_options": {"prompt_logprobs": 1}})


def test_logprobs_above_limit_are_rejected():
    with pytest.raises(InvalidArgument, match="logprobs"):
        build({"token_ids": [1], "output_options": {"logprobs": 1001}})
