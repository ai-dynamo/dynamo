# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.llm.exceptions import InvalidArgument
from dynamo.tokenspeed.llm_engine import (
    _completion_delta_output,
    _validate_single_choice_sampling,
    build_sampling_params,
    convert_output_to_chunk,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


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

    assert params["temperature"] == 0.2
    assert params["top_p"] == 0.9
    assert params["top_k"] == 20
    assert params["min_p"] == 0.1
    assert params["frequency_penalty"] == 0.3
    assert params["presence_penalty"] == 0.4
    assert params["repetition_penalty"] == 1.1
    assert params["seed"] == 123
    assert params["n"] == 1
    assert params["max_new_tokens"] == 17
    assert params["min_new_tokens"] == 2
    assert params["ignore_eos"] is True
    assert params["stop_token_ids"] == [7, 8, 9]
    assert params["regex"] == "(yes|no)"


def test_build_sampling_params_maps_json_guided_decoding():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {
                "guided_decoding": {
                    "json": {"type": "object"},
                },
            },
        },
        model_max_len=10,
    )

    assert json.loads(params["json_schema"]) == {"type": "object"}


def test_build_sampling_params_maps_grammar_guided_decoding():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {
                "guided_decoding": {
                    "grammar": 'root ::= "x"',
                },
            },
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


def test_convert_output_to_chunk_maps_finish_reason_and_usage():
    class FinishReason:
        def to_json(self):
            return {"type": "length", "length": 2}

    chunk = convert_output_to_chunk(
        {
            "index": 1,
            "output_ids": [11, 12],
            "meta_info": {
                "finish_reason": FinishReason(),
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "cached_tokens": 1,
            },
        }
    )

    assert chunk == {
        "index": 1,
        "token_ids": [11, 12],
        "finish_reason": "length",
        "completion_usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    }


def test_convert_output_to_chunk_normalizes_abort_finish_reason():
    chunk = convert_output_to_chunk(
        {
            "output_ids": [],
            "meta_info": {
                "finish_reason": "abort_request",
                "prompt_tokens": 1,
                "completion_tokens": 0,
            },
        }
    )

    assert chunk["finish_reason"] == "cancelled"


def test_validate_single_choice_sampling_rejects_n_greater_than_one():
    with pytest.raises(InvalidArgument, match="n=2"):
        _validate_single_choice_sampling(
            {"token_ids": [1], "sampling_options": {"n": 2}}
        )


def test_completion_delta_output_strips_first_chunk_prompt_echo():
    out = {
        "output_ids": [10, 11, 12, 99],
        "meta_info": {"completion_tokens": 1},
    }

    delta_out, emitted = _completion_delta_output(out, 0)

    assert emitted == 1
    assert delta_out["output_ids"] == [99]
    assert out["output_ids"] == [10, 11, 12, 99]


def test_completion_delta_output_preserves_later_token_delta():
    out = {
        "output_ids": [100],
        "meta_info": {"completion_tokens": 2},
    }

    delta_out, emitted = _completion_delta_output(out, 1)

    assert emitted == 2
    assert delta_out["output_ids"] == [100]
