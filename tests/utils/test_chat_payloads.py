# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock, patch

import pytest

from tests.utils.payload_builder import (
    chat_payload_with_logprobs,
    streaming_chat_payload_with_logprobs,
)
from tests.utils.payloads import GuidedDecodingChatPayload

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "case", [None, "shifted", "coalesced", "prompt", "prompt_usage", "done"]
)
def test_streaming_logprobs_require_complete_token_metadata(case):
    payload = streaming_chat_payload_with_logprobs(
        prompt_logprobs=1, top_logprobs=1, expected_response=[]
    )
    payload.min_token_chunks = 2
    token = {"token": "token_id:17", "logprob": -9999.0, "bytes": [49, 55]}
    logprob = {**token, "token_id": 17, "top_logprobs": [token]}
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "logprobs": {"content": [logprob]},
                }
            ],
            "nvext": {"completion_token_ids": [17]},
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hello"},
                    "logprobs": {"content": [logprob]},
                    "finish_reason": "stop",
                }
            ],
            "nvext": {
                "completion_token_ids": [17],
                "prompt_token_ids": [4, 5],
                "prompt_logprobs": [None, {"5": {"logprob": -9999.0}}],
            },
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        },
    ]
    if case in ("shifted", "coalesced"):
        chunks[1]["choices"][0]["logprobs"]["content"].extend(
            chunks[0]["choices"][0].pop("logprobs")["content"]
        )
        if case == "coalesced":
            chunks[1]["nvext"]["completion_token_ids"].extend(
                chunks.pop(0)["nvext"]["completion_token_ids"]
            )
    elif case == "prompt":
        chunks[1]["nvext"]["prompt_logprobs"][1] = {"6": {"logprob": -0.2}}
    elif case == "prompt_usage":
        payload.body.pop("prompt_logprobs")
        payload.body["nvext"]["extra_fields"].remove("prompt_logprobs")
        chunks[1]["nvext"].pop("prompt_logprobs")
        chunks[2]["usage"].update(prompt_tokens=1, total_tokens=3)
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    if case != "done":
        lines.append("data: [DONE]")
    response = Mock()
    response.iter_lines.return_value = iter(lines)
    if case is None:
        assert payload.process_response(response) == "hello"
    else:
        expected = {
            "shifted": "Output logprobs do not match this chunk",
            "coalesced": "Expected at least 2 token-bearing chunks",
            "prompt": "Missing prompt token logprob",
            "prompt_usage": "Incorrect prompt usage",
            "done": r"Stream ended without \[DONE\]",
        }
        with pytest.raises(AssertionError, match=expected[case]):
            payload.process_response(response)
    response.close.assert_called_once()


def test_streaming_deadline_includes_keepalive_events():
    payload = streaming_chat_payload_with_logprobs()
    response = Mock()
    response.iter_lines.return_value = iter([": keepalive"])
    with patch("tests.utils.payloads.time.monotonic", side_effect=[0, payload.timeout]):
        with pytest.raises(AssertionError, match="deadline"):
            payload.process_response(response)
    response.close.assert_called_once()


def test_requested_logprobs_cannot_be_empty():
    payload = chat_payload_with_logprobs(expected_response=[])
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "hello"}, "logprobs": None}]
    }
    with pytest.raises(AssertionError, match="requested output logprobs"):
        payload.process_response(response)


def test_guided_json_checks_boolean_type_and_value():
    payload = GuidedDecodingChatPayload(
        body={},
        expected_log=[],
        expected_response=[],
        expected_json={"ok": True},
        expected_finish_reason="stop",
        needs_token_ids=True,
    )
    response = Mock()
    response.json.return_value = {
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "nvext": {"prompt_token_ids": [4, 5], "completion_token_ids": [17]},
    }
    payload.validate(response, '{"ok": true}')
    with pytest.raises(AssertionError, match="Expected JSON"):
        payload.validate(response, '{"ok": 1}')
    response.json.return_value["nvext"]["completion_token_ids"] = []
    with pytest.raises(AssertionError, match="completion_token_ids count"):
        payload.validate(response, '{"ok": true}')
