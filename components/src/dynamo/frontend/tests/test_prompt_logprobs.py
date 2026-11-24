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


def test_repeated_prompt_logprobs_are_dropped():
    """A worker repeats them on each choice's last frame; Dynamo gets one copy."""
    accumulator = PromptLogprobsAccumulator(
        {"stream": True, "nvext": {"extra_fields": ["prompt_logprobs"]}}
    )
    payload = [None, {"17": {"logprob": -0.25}}]
    first = {"engine_data": {"prompt_logprobs": payload, "routed_experts": [1]}}
    second = {"engine_data": {"prompt_logprobs": payload, "routed_experts": [2]}}

    accumulator.update(first)
    accumulator.update(second)

    assert first["engine_data"]["prompt_logprobs"] == payload
    assert second["engine_data"] == {"routed_experts": [2]}
