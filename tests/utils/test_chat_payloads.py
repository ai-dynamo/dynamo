# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock, patch

import pytest

from tests.utils.payload_builder import chat_payload_with_logprobs
from tests.utils.payloads import DisaggregatedChatPayload, GuidedDecodingChatPayload

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "field,value,error",
    [
        (None, None, None),
        ("finish_reason", "stop", "Expected finish reason"),
        ("completion_tokens", 7, "Expected 8 completion tokens"),
        ("completion_token_ids", [17] * 7, "completion_token_ids count"),
        ("prompt_token_ids", [4], "prompt_token_ids count"),
        ("total_tokens", 11, "Inconsistent total token usage"),
    ],
)
def test_disaggregated_token_accounting(field, value, error):
    payload = DisaggregatedChatPayload(
        body={},
        expected_response=[],
        expected_log=[],
        expected_finish_reason="length",
        expected_completion_tokens=8,
    )
    result = {
        "choices": [{"message": {"content": "hello"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 8, "total_tokens": 10},
        "nvext": {
            "completion_token_ids": [17] * 8,
            "prompt_token_ids": [4, 5],
            "worker_id": {"prefill_worker_id": 1, "decode_worker_id": 2},
        },
    }
    if field == "finish_reason":
        result["choices"][0][field] = value
    elif field in result["usage"]:
        result["usage"][field] = value
    elif field is not None:
        result["nvext"][field] = value
    response = Mock()
    response.json.return_value = result
    if error is None:
        assert payload.process_response(response) == "hello"
    else:
        with pytest.raises(AssertionError, match=error):
            payload.process_response(response)


@pytest.mark.parametrize("case", [None, "shifted", "coalesced", "prompt", "done"])
def test_streaming_logprobs_require_complete_token_metadata(case):
    payload = chat_payload_with_logprobs(
        stream=True, prompt_logprobs=1, top_logprobs=1, expected_response=[]
    )
    payload.min_token_chunks = 2
    token = {"token": "token_id:17", "logprob": -0.1, "bytes": [49, 55]}
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
                "prompt_logprobs": [None, {"5": {"logprob": -0.2}}],
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
            "done": r"Stream ended without \[DONE\]",
        }
        with pytest.raises(AssertionError, match=expected[case]):
            payload.process_response(response)
    response.close.assert_called_once()


def test_streaming_deadline_includes_keepalive_events():
    payload = chat_payload_with_logprobs(stream=True)
    response = Mock()
    response.iter_lines.return_value = iter([": keepalive"])
    with patch("tests.utils.payloads.time.monotonic", side_effect=[0, payload.timeout]):
        with pytest.raises(AssertionError, match="deadline"):
            payload.process_response(response)
    response.close.assert_called_once()


@pytest.mark.parametrize("logprobs", [None, {"content": []}])
def test_requested_logprobs_cannot_be_empty(logprobs):
    payload = chat_payload_with_logprobs(expected_response=[])
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "hello"}, "logprobs": logprobs}]
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
