# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt logprob text uses the installed native vLLM decoding algorithm."""

import copy
import json
import math

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


class PromptTokenizer:
    # Mimic SentencePiece's leading-space marker without loading a model.
    _vllm_space_marker_cache = "▁"

    def decode(self, token_ids):
        pieces = {10: "a", 11: "�", 12: "�", 13: " other", 20: "word", 30: "<s>"}
        if token_ids[-2:] == [11, 12]:
            return "".join(pieces[token] for token in token_ids[:-2]) + "é"
        return "".join(pieces[token] for token in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        return ["▁word" if token == 20 else str(token) for token in token_ids]


def test_prompt_logprob_text_preserves_native_spaces_specials_and_utf8():
    from vllm.logprobs import Logprob

    from dynamo.vllm.handlers import _serialize_prompt_logprobs

    raw = [
        None,
        {10: Logprob(-0.1, rank=1)},
        {11: Logprob(-0.2, rank=4), 13: Logprob(-0.05, rank=1)},
        {12: Logprob(-0.3, rank=2), 13: Logprob(-0.15, rank=1)},
        {20: Logprob(-0.4, rank=1)},
        {30: Logprob(-0.5, rank=1)},
    ]
    result = _serialize_prompt_logprobs(raw, PromptTokenizer())
    assert result == [
        None,
        {"10": {"logprob": -0.1, "rank": 1, "decoded_token": "a"}},
        {
            "11": {"logprob": -0.2, "rank": 4, "decoded_token": ""},
            "13": {"logprob": -0.05, "rank": 1, "decoded_token": " other"},
        },
        {
            "12": {"logprob": -0.3, "rank": 2, "decoded_token": "é"},
            "13": {"logprob": -0.15, "rank": 1, "decoded_token": " other"},
        },
        {"20": {"logprob": -0.4, "rank": 1, "decoded_token": " word"}},
        {"30": {"logprob": -0.5, "rank": 1, "decoded_token": "<s>"}},
    ]
    # Reconstruction must not mutate the engine's objects or change ranking.
    assert raw[2][11].decoded_token is None
    assert raw[2][11].rank == 4


def test_prompt_logprob_text_preserves_engine_supplied_empty_and_nonempty_values():
    from vllm.logprobs import Logprob

    from dynamo.vllm.handlers import _serialize_prompt_logprobs

    class NoDecodeTokenizer:
        def decode(self, token_ids):
            raise AssertionError("Already decoded payload must not be decoded again")

    raw = [
        None,
        {10: Logprob(-0.1, rank=1, decoded_token="")},
        {20: Logprob(-0.2, rank=2, decoded_token="engine spelling")},
    ]
    result = _serialize_prompt_logprobs(raw, NoDecodeTokenizer())
    assert result[1]["10"]["decoded_token"] == ""
    assert result[2]["20"]["decoded_token"] == "engine spelling"


@pytest.mark.parametrize("value", [-math.inf, -1e30, -10000.0, -9999.0, -0.25, 0.0])
def test_prompt_logprob_normalization_matches_native_serving(value):
    from vllm.entrypoints.generate.base.serving import clamp_prompt_logprobs
    from vllm.logprobs import Logprob

    from dynamo.vllm.handlers import _serialize_prompt_logprobs

    raw = [None, {7: Logprob(value, rank=2, decoded_token="token")}]
    native = clamp_prompt_logprobs(copy.deepcopy(raw))
    result = _serialize_prompt_logprobs(raw)
    assert result == [
        None,
        {
            "7": {
                "logprob": native[1][7].logprob,
                "rank": native[1][7].rank,
                "decoded_token": native[1][7].decoded_token,
            }
        },
    ]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    # The native helper mutates its objects; Dynamo must not mutate engine output.
    assert raw[1][7].logprob == value


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_prompt_logprob_invalid_nonfinite_values_fail_without_invented_probability(
    value,
):
    from vllm.logprobs import Logprob

    from dynamo.vllm.handlers import _serialize_prompt_logprobs

    raw = [None, {7: Logprob(value, rank=2, decoded_token="private-payload")}]
    with pytest.raises(ValueError, match="^Invalid non-finite prompt logprob$"):
        _serialize_prompt_logprobs(raw)
