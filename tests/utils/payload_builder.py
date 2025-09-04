# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from tests.utils.client import send_request
from tests.utils.payloads import ChatPayload, CompletionPayload, MetricsPayload

# Common default text prompt used across tests
TEXT_PROMPT = "Tell me a short joke about AI."


def chat_payload_default(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 150,
    temperature: float = 0.1,
    stream: bool = False,
) -> ChatPayload:
    return ChatPayload(
        body={
            "messages": [
                {
                    "role": "user",
                    "content": TEXT_PROMPT,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response or ["AI"],
    )


def completion_payload_default(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 150,
    temperature: float = 0.1,
    stream: bool = False,
) -> CompletionPayload:
    return CompletionPayload(
        body={
            "prompt": TEXT_PROMPT,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response or ["AI"],
    )


def metric_payload_default(
    min_num_requests: int,
    repeat_count: int = 1,
    expected_log: Optional[List[str]] = None,
) -> MetricsPayload:
    return MetricsPayload(
        body={},
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=[],
        min_num_requests=min_num_requests,
    )


def chat_payload(
    content: Union[str, List[Dict[str, Any]]],
    repeat_count: int = 1,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 300,
    temperature: Optional[float] = None,
    stream: bool = False,
) -> ChatPayload:
    body: Dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if temperature is not None:
        body["temperature"] = temperature

    return ChatPayload(
        body=body,
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response or [],
    )


def completion_payload(
    prompt: str,
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 150,
    temperature: float = 0.1,
    stream: bool = False,
) -> CompletionPayload:
    return CompletionPayload(
        body={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response or [],
    )


# Build small request-based health checks for chat and completions
# these should only be used as a last resort. Generally want to use an actual health check


def make_chat_health_check(port: int, model: str):
    def _check_chat_endpoint(remaining_timeout: float = 30.0) -> bool:
        payload = chat_payload_default(
            repeat_count=1,
            expected_response=[],
            max_tokens=8,
            temperature=0.0,
            stream=False,
        ).with_model(model)
        payload.port = port
        try:
            resp = send_request(
                payload.url(),
                payload.body,
                timeout=min(max(1.0, remaining_timeout), 5.0),
                method=payload.method,
                log_level=10,
            )
            # Validate structure only; expected_response is empty
            _ = payload.response_handler(resp)
            return True
        except Exception:
            return False

    return _check_chat_endpoint


def make_completions_health_check(port: int, model: str):
    def _check_completions_endpoint(remaining_timeout: float = 30.0) -> bool:
        payload = completion_payload_default(
            repeat_count=1,
            expected_response=[],
            max_tokens=8,
            temperature=0.0,
            stream=False,
        ).with_model(model)
        payload.port = port
        try:
            resp = send_request(
                payload.url(),
                payload.body,
                timeout=min(max(1.0, remaining_timeout), 5.0),
                method=payload.method,
                log_level=10,
            )
            out = payload.response_handler(resp)
            if not out:
                raise ValueError("")
            return True
        except Exception:
            return False

    return _check_completions_endpoint
