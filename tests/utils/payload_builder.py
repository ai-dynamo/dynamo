# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from tests.utils.client import send_request
from tests.utils.payloads import (
    ChatPayload,
    CompletionPayload,
    EmbeddingPayload,
    MetricsPayload,
)

# Common default text prompt used across tests
TEXT_PROMPT = "Tell me a knock knock joke about AI."


def chat_payload_default(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
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
        # Accept any of these keywords in the response (case-insensitive)
        expected_response=expected_response
        or ["AI", "knock", "joke", "think", "artificial", "intelligence"],
    )


def completion_payload_default(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
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
        # Accept any of these keywords in the response (case-insensitive)
        expected_response=expected_response
        or ["AI", "knock", "joke", "think", "artificial", "intelligence"],
    )


def multimodal_payload_default(
    image_url: str = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    text: str = "Describe the image",
    repeat_count: int = 1,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 160,
    temperature: Optional[float] = None,
    stream: bool = False,
) -> ChatPayload:
    """Create a multimodal chat payload with image and text content.

    Args:
        image_url: URL of the image to include in the request
        text: Text prompt to accompany the image
        repeat_count: Number of times to repeat the request
        expected_response: List of strings expected in the response
        expected_log: List of regex patterns expected in logs
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (optional)
        stream: Whether to stream the response

    Returns:
        ChatPayload configured for multimodal requests
    """
    return chat_payload(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ],
        repeat_count=repeat_count,
        expected_response=expected_response or ["image"],
        expected_log=expected_log or [],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )


def metric_payload_default(
    min_num_requests: int,
    repeat_count: int = 1,
    expected_log: Optional[List[str]] = None,
    backend: Optional[str] = None,
    port: int = 8081,
) -> MetricsPayload:
    return MetricsPayload(
        body={},
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=[],
        min_num_requests=min_num_requests,
        backend=backend,
        port=port,
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


def embedding_payload_default(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
) -> EmbeddingPayload:
    return EmbeddingPayload(
        body={
            "input": ["The sky is blue.", "Machine learning is fascinating."],
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response
        or ["Generated 2 embeddings with dimension"],
    )


def embedding_payload(
    input_text: Union[str, List[str]],
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
) -> EmbeddingPayload:
    # Normalize input to list for consistent processing
    if isinstance(input_text, str):
        input_list = [input_text]
        expected_count = 1
    else:
        input_list = input_text
        expected_count = len(input_text)

    return EmbeddingPayload(
        body={
            "input": input_list,
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response
        or [f"Generated {expected_count} embeddings with dimension"],
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
