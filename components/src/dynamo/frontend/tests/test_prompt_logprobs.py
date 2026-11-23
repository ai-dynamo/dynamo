# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend.utils import PromptLogprobsAccumulator

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize("count", [None, 0])
@pytest.mark.parametrize("include_nvext", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_prompt_logprobs_are_owed_until_the_worker_sends_them(
    count, include_nvext, stream
):
    request = {"stream": stream}
    if count is not None:
        request["prompt_logprobs"] = count
    if include_nvext:
        request["nvext"] = {"extra_fields": ["prompt_logprobs"]}
    accumulator = PromptLogprobsAccumulator(request)
    requested = (count is not None and not stream) or include_nvext
    assert accumulator.needs_payload is requested

    accumulator.update({"token_ids": [1], "engine_data": {}})
    assert accumulator.needs_payload is requested

    payload = [None, {"17": {"logprob": -0.25, "rank": 1}}]
    accumulator.update({"engine_data": {"prompt_logprobs": payload}})
    assert not accumulator.needs_payload


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
