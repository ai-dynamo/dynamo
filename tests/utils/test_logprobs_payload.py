# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from tests.utils.payloads import ChatPayloadWithLogprobs, CompletionPayloadWithLogprobs

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize("text", ["é", "Aé"])
def test_completion_top_logprobs_token_maps(text):
    # Match the Rust streaming_and_aggregated schema test, including a sampled
    # token outside top-k: requesting logprobs=1 can return two entries.
    payload = CompletionPayloadWithLogprobs(
        body={"logprobs": 1}, expected_response=[text], expected_log=[]
    )
    response = Mock()
    response.json.return_value = {
        "choices": [
            {
                "text": text,
                "logprobs": {
                    "tokens": list(text),
                    "token_logprobs": [-0.5] * len(text),
                    "text_offset": [],
                    "top_logprobs": [{t: -0.5, "other": -0.25} for t in text],
                },
            }
        ]
    }
    assert payload.process_response(response) == text


@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize("shape", ["valid", "null", "empty", "no_alternatives"])
def test_chat_top_logprobs_requires_alternatives(count, shape):
    """Positive requests need alternatives; zero still permits sampled tokens only."""
    payload = ChatPayloadWithLogprobs(
        body={"logprobs": True, "top_logprobs": count},
        expected_response=["A"],
        expected_log=[],
    )
    token = {"token": "A", "logprob": -0.5, "bytes": [65]}
    logprobs = {"content": [{**token, "top_logprobs": [token]}]}
    if shape == "null":
        logprobs = None
    elif shape == "empty":
        logprobs["content"] = []
    elif shape == "no_alternatives":
        logprobs["content"][0]["top_logprobs"] = []
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "A"}, "logprobs": logprobs}]
    }
    if count > 0 and shape != "valid":
        with pytest.raises(AssertionError, match="Expected .*logprobs"):
            payload.process_response(response)
    else:
        assert payload.process_response(response) == "A"


@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize(
    "shape",
    ["valid", "null", "empty", "missing", "no_alternatives", "empty_map", "short"],
)
def test_completion_top_logprobs_requires_alternatives(count, shape):
    """Require alternatives for every generated token only for positive counts."""
    payload = CompletionPayloadWithLogprobs(
        body={"logprobs": count}, expected_response=["AB"], expected_log=[]
    )
    logprobs = {
        "tokens": ["A", "B"],
        "token_logprobs": [-0.5, -0.25],
        "top_logprobs": [{"A": -0.5}, {"B": -0.25}],
    }
    if shape == "null":
        logprobs = None
    elif shape == "empty":
        logprobs["token_logprobs"] = []
    elif shape == "missing":
        del logprobs["top_logprobs"]
    elif shape == "no_alternatives":
        logprobs["top_logprobs"] = []
    elif shape == "empty_map":
        logprobs["top_logprobs"][1] = {}
    elif shape == "short":
        logprobs["top_logprobs"].pop()
    response = Mock()
    response.json.return_value = {"choices": [{"text": "AB", "logprobs": logprobs}]}
    if count > 0 and shape != "valid":
        with pytest.raises(AssertionError, match="Expected"):
            payload.process_response(response)
    else:
        assert payload.process_response(response) == "AB"
