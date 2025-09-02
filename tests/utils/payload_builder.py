# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

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
