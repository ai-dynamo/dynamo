# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from tests.utils.payloads import CompletionPayloadWithLogprobs

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
