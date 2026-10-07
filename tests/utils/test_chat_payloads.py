# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
from unittest.mock import Mock, patch

import pytest
import requests

from tests.utils.engine_metrics import VllmMetricsChecker
from tests.utils.payload_builder import (
    chat_payload_with_logprobs,
    streaming_chat_payload_with_logprobs,
)
from tests.utils.payloads import GuidedDecodingChatPayload, HttpCancellationPayload

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "case",
    [None, "human_readable", "shifted", "coalesced", "prompt", "done"],
)
def test_streaming_logprobs_require_complete_token_metadata(case):
    payload = streaming_chat_payload_with_logprobs(
        prompt_logprobs=1,
        top_logprobs=3,
        expected_response=[],
        extra_body=(
            {"return_tokens_as_token_ids": False} if case == "human_readable" else None
        ),
    )
    payload.min_token_chunks = 2
    token = {
        "token": "欧元" if case == "human_readable" else "token_id:17",
        "logprob": -9999,
        "bytes": None,
    }
    logprob = {**token, "token_id": 17, "top_logprobs": [token, token]}
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
                    "delta": {"content": "欧元" if case == "human_readable" else "hello"},
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
    if case == "human_readable":
        assert payload.body["return_tokens_as_token_ids"] is False
    elif case in ("shifted", "coalesced"):
        chunks[1]["choices"][0]["logprobs"]["content"].extend(
            chunks[0]["choices"][0].pop("logprobs")["content"]
        )
        if case == "coalesced":
            chunks[1]["nvext"]["completion_token_ids"].extend(
                chunks.pop(0)["nvext"]["completion_token_ids"]
            )
    elif case == "prompt":
        chunks[1]["nvext"]["prompt_logprobs"][1] = {"6": {"logprob": -0.2}}
    lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}" for chunk in chunks]
    if case != "done":
        lines.append("data: [DONE]")
    response = requests.Response()
    response.status_code = 200
    response.encoding = "ISO-8859-1"
    response.raw = io.BytesIO(("\n\n".join(lines) + "\n\n").encode("utf-8"))
    response.close = Mock(wraps=response.close)
    if case in (None, "human_readable"):
        assert payload.process_response(response) == (
            "欧元" if case == "human_readable" else "hello"
        )
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
    payload = streaming_chat_payload_with_logprobs()
    response = Mock()
    response.iter_lines.return_value = iter([": keepalive"])
    with patch("tests.utils.payloads.time.monotonic", side_effect=[0, payload.timeout]):
        with pytest.raises(AssertionError, match="deadline"):
            payload.process_response(response)
    response.close.assert_called_once()


def test_requested_logprobs_allow_unknown_bytes_but_require_finite_values():
    payload = chat_payload_with_logprobs(expected_response=[])
    response = Mock()
    logprob = {"token": "€", "bytes": None, "logprob": -9999, "top_logprobs": []}
    response.json.return_value = {
        "choices": [
            {"message": {"content": "hello"}, "logprobs": {"content": [logprob]}}
        ]
    }
    assert payload.process_response(response) == "hello"
    for invalid in (float("nan"), 0.1):
        logprob["logprob"] = invalid
        with pytest.raises(AssertionError, match="Invalid logprob"):
            payload.process_response(response)
    response.json.return_value["choices"][0]["logprobs"] = None
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


@pytest.mark.parametrize("generated_tokens", [3, 8])
def test_http_cancellation_rejects_completed_generation(generated_tokens):
    response = requests.Response()
    response.status_code = 200
    response.raw = io.BytesIO(
        b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
        b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    response.close = Mock(wraps=response.close)
    metrics = VllmMetricsChecker(url="unused", settle_timeout=0)

    def scrape():
        has_started = response.raw.closed or response.raw.tell() > 0
        running = int(has_started and not response.raw.closed)
        progress = 100 + (generated_tokens if has_started else 0)
        return (
            f"vllm:num_requests_running {running}\n"
            "vllm:num_requests_waiting 0\n"
            f"vllm:iteration_tokens_total_count {progress}\n"
        )

    payload = HttpCancellationPayload(
        body={"stream": True, "ignore_eos": True, "max_tokens": 8},
        expected_response=[],
        expected_log=[],
        metrics=metrics,
    )
    with patch.object(metrics, "scrape", side_effect=scrape):
        payload.before_request()
        if generated_tokens == 8:
            with pytest.raises(AssertionError, match="completed generation"):
                payload.process_response(response)
        else:
            assert payload.process_response(response) == "hello"
    response.close.assert_called_once()
