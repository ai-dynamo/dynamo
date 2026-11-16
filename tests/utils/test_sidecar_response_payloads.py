# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest

from tests.utils.engine_metrics import EngineMetrics
from tests.utils.payload_builder import (
    disaggregated_token_count_payload,
    http_cancellation_payloads,
    streaming_chat_payload_with_logprobs,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "prompt_tokens,top_logprobs,with_candidates,error",
    [
        (2, 0, False, None),
        (1, 0, False, "Incorrect prompt usage"),
        (2, 2, True, None),
        (2, 2, False, "Expected top_logprobs alternatives"),
    ],
)
def test_streaming_token_metadata_without_prompt_logprobs(
    prompt_tokens, top_logprobs, with_candidates, error
):
    payload = streaming_chat_payload_with_logprobs(
        expected_response=[],
        top_logprobs=top_logprobs,
        extra_body={"nvext": {"extra_fields": ["prompt_token_ids"]}},
    )
    token = {
        "token_id": 17,
        "token": "token_id:17",
        "logprob": -0.5,
        "bytes": None,
        "top_logprobs": [],
    }
    if with_candidates:
        token["top_logprobs"] = [
            {"token": "token_id:17", "logprob": -0.5, "bytes": None},
            {"token": "token_id:18", "logprob": -1.0, "bytes": None},
        ]
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": "hello"},
                "logprobs": {"content": [token]},
                "finish_reason": "stop",
            }
        ],
        "nvext": {"completion_token_ids": [17], "prompt_token_ids": [4, 5]},
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 1,
            "total_tokens": prompt_tokens + 1,
        },
    }
    response = Mock()
    response.iter_lines.return_value = iter(
        [f"data: {json.dumps(chunk)}", "data: [DONE]"]
    )
    if error is None:
        assert payload.process_response(response) == "hello"
    else:
        with pytest.raises(AssertionError, match=error):
            payload.process_response(response)
    response.close.assert_called_once()


@pytest.mark.parametrize("with_metrics", [False, True])
def test_disconnect_closes_before_terminal_and_preserves_cleanup_checks(with_metrics):
    metrics = Mock(spec=EngineMetrics) if with_metrics else None
    payload, _ = http_cancellation_payloads(metrics, max_tokens=2048)
    chunk = {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
    response = Mock()

    def lines(**kwargs):
        yield b"data: " + json.dumps(chunk).encode()
        pytest.fail("Cancellation consumed the rest of the stream")

    response.iter_lines.side_effect = lines
    payload.before_request()
    assert payload.process_response(response) == "hello"
    response.close.assert_called_once()
    if metrics is not None:
        metrics.wait_for_scheduler.assert_any_call(is_active=True)
        metrics.assert_cancelled.assert_called_once_with(
            before=metrics.progress.return_value,
            completion_progress=metrics.completion_progress.return_value,
        )


@pytest.mark.parametrize("first_id", [17, 99])
def test_handoff_logprobs_match_completion_ids(first_id):
    payload = disaggregated_token_count_payload(logprobs=True)
    result = {
        "choices": [
            {
                "message": {"content": "hello"},
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {
                            "token_id": token,
                            "token": "",
                            "logprob": -0.5,
                            "bytes": None,
                            "top_logprobs": [],
                        }
                        for token in [first_id, *range(18, 25)]
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 8, "total_tokens": 10},
        "nvext": {
            "prompt_token_ids": [4, 5],
            "completion_token_ids": list(range(17, 25)),
            "worker_id": {"prefill_worker_id": 1, "decode_worker_id": 2},
        },
    }
    response = Mock()
    response.json.return_value = result
    if first_id == 17:
        payload.process_response(response)
    else:
        with pytest.raises(AssertionError, match="Handoff logprobs"):
            payload.process_response(response)
