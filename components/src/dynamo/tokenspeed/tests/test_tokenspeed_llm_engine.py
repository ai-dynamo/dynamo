# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.llm.exceptions import InvalidArgument
from dynamo.tokenspeed.llm_engine import (
    _validate_single_choice_sampling,
    build_sampling_params,
    convert_output_to_chunk,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def wire_output(output_ids, finish_reason=None, finish_details=None, **fields):
    """One chunk as tokenspeed_engine.Frontend.next_outputs delivers it."""
    chunk = {
        "output_ids": output_ids,
        "finish_reason": finish_reason,
        "finish_details": finish_details,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "output_token_logprobs": [],
        "output_token_cumulative_probs": [],
        "spec_verify_ct": None,
        "accept_draft_tokens": None,
        "input_token_logprobs": None,
        "input_top_logprobs": None,
        "output_top_logprobs": None,
    }
    chunk.update(fields)
    return chunk


def test_build_sampling_params_maps_dynamo_request():
    params = build_sampling_params(
        {
            "token_ids": [1, 2, 3],
            "sampling_options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.1,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.4,
                "repetition_penalty": 1.1,
                "seed": 123,
                "n": 1,
                "guided_decoding": {
                    "choice": ["yes", "no"],
                },
            },
            "stop_conditions": {
                "max_tokens": 17,
                "min_tokens": 2,
                "ignore_eos": True,
                "stop_token_ids_hidden": [7, 8],
                "stop_token_ids": [8, 9],
            },
        },
        model_max_len=100,
    )

    assert params == {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.1,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.4,
        "repetition_penalty": 1.1,
        "seed": 123,
        "n": 1,
        "max_new_tokens": 17,
        "min_new_tokens": 2,
        "ignore_eos": True,
        "stop_token_ids": [7, 8, 9],
        "regex": "(yes|no)",
    }


@pytest.mark.parametrize("logit_bias", [{"18": -10.0, "42": 5.5}, {18: -10, 42: 5.5}])
def test_build_sampling_params_forwards_logit_bias_with_string_keys(logit_bias):
    params = build_sampling_params(
        {"token_ids": [1], "sampling_options": {"logit_bias": logit_bias}}
    )

    # The wire types logit_bias as dict[str, float].
    assert params["logit_bias"] == {"18": -10.0, "42": 5.5}


def test_build_sampling_params_maps_json_guided_decoding():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {"guided_decoding": {"json": {"type": "object"}}},
        },
        model_max_len=10,
    )

    assert json.loads(params["json_schema"]) == {"type": "object"}


def test_build_sampling_params_maps_grammar_guided_decoding():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {"guided_decoding": {"grammar": 'root ::= "x"'}},
        },
        model_max_len=10,
    )

    assert params["ebnf"] == 'root ::= "x"'


def test_build_sampling_params_maps_structural_tag_guided_decoding():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {
                "guided_decoding": {
                    "structural_tag": {"begin": "<a>", "schema": {"type": "object"}},
                },
            },
        },
        model_max_len=10,
    )

    assert json.loads(params["structural_tag"]) == {
        "begin": "<a>",
        "schema": {"type": "object"},
    }


def test_build_sampling_params_rejects_multiple_guided_constraints():
    with pytest.raises(InvalidArgument, match="one constraint"):
        build_sampling_params(
            {
                "token_ids": [1],
                "sampling_options": {
                    "guided_decoding": {
                        "json": {"type": "object"},
                        "choice": ["yes", "no"],
                    },
                },
            },
            model_max_len=10,
        )


def test_build_sampling_params_uses_dynamic_max_tokens():
    params = build_sampling_params({"token_ids": [1, 2, 3]}, model_max_len=10)

    assert params["max_new_tokens"] == 7


def test_convert_streaming_output_carries_only_new_tokens():
    assert convert_output_to_chunk(wire_output([11, 12])) == {
        "index": 0,
        "token_ids": [11, 12],
    }


@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_convert_output_to_chunk_maps_finish_reason_and_usage(finish_reason):
    chunk = convert_output_to_chunk(
        wire_output(
            [13],
            finish_reason,
            {"type": finish_reason},
            prompt_tokens=3,
            completion_tokens=2,
        )
    )

    assert chunk == {
        "index": 0,
        "token_ids": [13],
        "finish_reason": finish_reason,
        "completion_usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    }


def test_convert_output_reports_cached_prompt_tokens():
    chunk = convert_output_to_chunk(
        wire_output(
            [],
            "length",
            {"type": "length"},
            prompt_tokens=64,
            completion_tokens=1,
            cached_tokens=48,
        )
    )

    assert chunk["completion_usage"]["prompt_tokens_details"] == {"cached_tokens": 48}


def test_convert_output_maps_requested_abort_to_cancelled():
    chunk = convert_output_to_chunk(
        wire_output(
            [],
            "abort",
            {"type": "abort", "message": "AbortReq from client", "err_type": 522},
        ),
        cancelled=True,
    )

    assert chunk["finish_reason"] == "cancelled"


@pytest.mark.parametrize(
    "message,err_type",
    [
        ("PD/EPD remote transfer failed or timed out", 522),
        ("remote peer disconnected", 521),
        ("NaN in logits", 523),
        ("AbortReq from client", 522),
    ],
)
def test_convert_output_raises_engine_initiated_abort(message, err_type):
    # An abort Dynamo did not ask for is a failure, which migration may retry,
    # not a client cancellation.
    with pytest.raises(RuntimeError, match=message):
        convert_output_to_chunk(
            wire_output(
                [], "abort", {"type": "abort", "message": message, "err_type": err_type}
            )
        )


@pytest.mark.parametrize("cancelled", [False, True])
def test_convert_output_raises_client_error_refusal(cancelled):
    details = {
        "type": "abort",
        "message": "stop strings are not supported on the msgpack wire",
        "err_type": 522,
        "status_code": 400,
    }
    with pytest.raises(InvalidArgument, match="stop strings"):
        convert_output_to_chunk(wire_output([], "abort", details), cancelled=cancelled)


def test_validate_single_choice_sampling_rejects_n_greater_than_one():
    with pytest.raises(InvalidArgument, match="n=2"):
        _validate_single_choice_sampling(
            {"token_ids": [1], "sampling_options": {"n": 2}}
        )
