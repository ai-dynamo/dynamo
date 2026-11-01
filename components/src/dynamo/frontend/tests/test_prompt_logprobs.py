# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend.utils import PromptLogprobsAccumulator

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize("count", [None, 0])
@pytest.mark.parametrize("include_nvext", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_prompt_logprobs_wait_for_terminal_chunk(count, include_nvext, stream):
    request = {"stream": stream}
    if count is not None:
        request["prompt_logprobs"] = count
    if include_nvext:
        request["nvext"] = {"extra_fields": ["prompt_logprobs"]}
    accumulator = PromptLogprobsAccumulator(request)
    assert accumulator.needs_payload is (
        (count is not None and not stream) or include_nvext
    )
    payload = [None, {"17": {"logprob": -0.25, "rank": 1}}]
    accumulator.update({"engine_data": {"prompt_logprobs": payload}})
    assert not accumulator.needs_payload

    partial = {"choices": [{"finish_reason": None}]}
    accumulator.attach(partial)
    assert "prompt_logprobs" not in partial
    assert "nvext" not in partial

    accumulator.update({"token_ids": [], "finish_reason": "stop"})
    terminal = {
        "choices": [{"finish_reason": "stop"}],
        "nvext": {"stop_reason": "END"},
    }
    accumulator.attach(terminal)
    if count is not None and not stream:
        assert terminal["prompt_logprobs"] is payload
    else:
        assert "prompt_logprobs" not in terminal
    if include_nvext:
        assert terminal["nvext"]["prompt_logprobs"] is payload
    else:
        assert "prompt_logprobs" not in terminal["nvext"]
    assert terminal["nvext"]["stop_reason"] == "END"
    assert not accumulator.needs_payload

    accumulator.update({"engine_data": {"prompt_logprobs": payload}})
    another_choice = {"choices": [{"index": 1, "finish_reason": "stop"}]}
    accumulator.attach(another_choice)
    assert "prompt_logprobs" not in another_choice
    assert "nvext" not in another_choice


@pytest.mark.parametrize("engine_response", [{}, {"engine_data": {}}])
def test_absent_prompt_logprobs_remain_optional(engine_response):
    accumulator = PromptLogprobsAccumulator({"prompt_logprobs": 0})
    accumulator.update(engine_response)
    chunk = {"choices": [{"finish_reason": "stop"}]}
    accumulator.attach(chunk)
    assert "prompt_logprobs" not in chunk


@pytest.mark.parametrize("payload", [None, "invalid", {}])
def test_invalid_prompt_logprobs_are_rejected_only_when_requested(payload):
    response = {"engine_data": {"prompt_logprobs": payload}}
    PromptLogprobsAccumulator({}).update(response)
    PromptLogprobsAccumulator({"stream": True, "prompt_logprobs": 0}).update(response)
    with pytest.raises(ValueError, match="invalid prompt_logprobs payload"):
        PromptLogprobsAccumulator({"prompt_logprobs": 0}).update(response)


@pytest.mark.parametrize(
    "payload",
    [
        ["invalid"],
        [{"-1": {"logprob": -0.25}}],
        [{"4294967296": {"logprob": -0.25}}],
        [{"17": None}],
        [{"17": {"logprob": "invalid"}}],
        [{"17": {"logprob": True}}],
        [{"17": {"logprob": -0.25, "rank": -1}}],
        [{"17": {"logprob": -0.25, "decoded_token": 17}}],
    ],
)
def test_invalid_nvext_prompt_logprobs_entries_are_rejected(payload):
    accumulator = PromptLogprobsAccumulator(
        {
            "stream": True,
            "prompt_logprobs": 0,
            "nvext": {"extra_fields": ["prompt_logprobs"]},
        }
    )
    with pytest.raises(ValueError, match="invalid prompt_logprobs payload"):
        accumulator.update({"engine_data": {"prompt_logprobs": payload}})
