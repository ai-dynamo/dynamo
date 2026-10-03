# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare generated-probability boundaries with installed native vLLM.

No model or server is started. Native response projection is exercised directly;
this is not an HTTP or mixed-release conformance test.
"""

import json
import math
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize("value", [-math.inf, -1e30, -10000.0, -9999.0, -0.25, 0.0])
@pytest.mark.parametrize("return_ids", [False, True])
@pytest.mark.parametrize("top_count", [0, 2])
def test_generated_logprobs_match_native_after_json_transport(
    value, return_ids, top_count
):
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.logprobs import Logprob

    from dynamo.frontend.vllm_logprobs import chat_logprob_content
    from dynamo.vllm.handlers import BaseWorkerHandler

    raw = {
        7: Logprob(value, rank=1, decoded_token="sampled"),
        8: Logprob(value, rank=2, decoded_token="alternative"),
    }
    output = SimpleNamespace(token_ids=[7], logprobs=[raw])
    selected, alternatives = BaseWorkerHandler._extract_logprobs(output, 0)
    wire = json.loads(
        json.dumps(
            {"selected": selected, "alternatives": alternatives}, allow_nan=False
        )
    )
    expected_wire = -9999.0 if value == -math.inf else value
    assert wire["selected"] == [expected_wire]
    assert [entry["logprob"] for entry in wire["alternatives"][0]] == [
        expected_wire,
        expected_wire,
    ]
    restored = {
        entry["token_id"]: Logprob(
            entry["logprob"], rank=entry["rank"], decoded_token=entry["token"]
        )
        for entry in wire["alternatives"][0]
    }
    actual = chat_logprob_content(
        SimpleNamespace(token_ids=[7], logprobs=[restored]),
        {"top_logprobs": top_count, "return_tokens_as_token_ids": return_ids},
        None,
    )
    # These projection methods require only the configured display flag; bypass
    # engine initialization so the unit comparison never loads a model.
    native = OpenAIServingChat.__new__(OpenAIServingChat)
    native.return_tokens_as_token_ids = return_ids
    expected = native._create_chat_logprobs(
        [7], [raw], None, num_output_top_logprobs=top_count
    ).model_dump()["content"]
    assert actual == expected
    assert raw[7].logprob == value
    assert raw[8].logprob == value


@pytest.mark.parametrize("value", [math.nan, math.inf])
@pytest.mark.parametrize("invalid_token", [7, 8])
def test_malformed_generated_probability_fails_without_payload(value, invalid_token):
    from vllm.logprobs import Logprob

    from dynamo.vllm.handlers import BaseWorkerHandler

    raw = {
        7: Logprob(-0.25, rank=1, decoded_token="private-sampled"),
        8: Logprob(-0.5, rank=2, decoded_token="private-alternative"),
    }
    raw[invalid_token].logprob = value
    output = SimpleNamespace(token_ids=[7], logprobs=[raw])
    with pytest.raises(ValueError, match="^Invalid non-finite generated logprob$"):
        BaseWorkerHandler._extract_logprobs(output, 0)
