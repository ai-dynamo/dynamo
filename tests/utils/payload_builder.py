# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from tests.utils.client import send_request
from tests.utils.constants import DefaultPort
from tests.utils.payloads import (
    CachedTokensChatPayload,
    ChatPayload,
    ChatPayloadWithLogprobs,
    CompletionPayload,
    CompletionPayloadWithLogprobs,
    EmbeddingPayload,
    MetricsPayload,
)

# Common default text prompt used across tests
TEXT_PROMPT = "Tell me a knock knock joke about AI."

# Longer prompt for prefix caching tests - needs to be > 64 tokens (typical block size)
# to ensure at least one full block gets cached
LONG_PROMPT_FOR_CACHING = """In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, \
lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the \
shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled \
curiosity and courage, who has stumbled upon an ancient map hinting at the city's location. The map suggests that \
Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey \
will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. \
Your Task: Character Background: Develop a detailed background for your character. Describe their motivations \
for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends."""


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


def cached_tokens_chat_payload(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
    min_cached_tokens: int = 64,
) -> CachedTokensChatPayload:
    """Create a chat payload that validates cached tokens in usage field.

    This is useful for testing KV router cache-aware routing where repeated
    identical prompts should result in cached tokens being reported.

    Uses a longer prompt (~196 tokens) to ensure at least one full block (64 tokens)
    gets cached. vLLM only caches complete blocks, so short prompts won't trigger
    the cached_tokens field in the response.

    Args:
        repeat_count: Number of times to repeat the request (>1 needed to see caching)
        expected_response: List of expected strings in response
        expected_log: List of expected log patterns
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        min_cached_tokens: Minimum cached tokens expected after first request (default: 64, one block)

    Returns:
        CachedTokensChatPayload configured for testing prefix caching
    """
    return CachedTokensChatPayload(
        body={
            "messages": [
                {
                    "role": "user",
                    "content": LONG_PROMPT_FOR_CACHING,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response
        or ["Aeloria", "Eldoria", "explorer", "ancient", "character", "background"],
        min_cached_tokens=min_cached_tokens,
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
    port: int = DefaultPort.SYSTEM1.value,
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
    logprobs: bool = False,
    top_logprobs: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
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
        "logprobs": logprobs,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if logprobs is not None:
        body["logprobs"] = logprobs
    if top_logprobs is not None:
        body["top_logprobs"] = top_logprobs

    if top_logprobs is not None:
        body["top_logprobs"] = top_logprobs

    if extra_body:
        body.update(extra_body)

    if logprobs:
        return ChatPayloadWithLogprobs(
            body=body,
            repeat_count=repeat_count,
            expected_log=expected_log or [],
            expected_response=expected_response or [],
        )
    else:
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
    logprobs: Optional[int] = None,
) -> CompletionPayload:
    body: Dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
        return CompletionPayloadWithLogprobs(
            body=body,
            repeat_count=repeat_count,
            expected_log=expected_log or [],
            expected_response=expected_response or [],
        )
    else:
        return CompletionPayload(
            body=body,
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


def chat_payload_with_logprobs(
    content: Union[str, List[Dict[str, Any]]] = TEXT_PROMPT,
    repeat_count: int = 1,
    expected_response: Optional[List[str]] = None,
    max_tokens: int = 50,
    temperature: float = 0.0,
    top_logprobs: int = 3,
) -> ChatPayloadWithLogprobs:
    """
    Create a chat payload that requests and validates logprobs in the response.

    Args:
        content: Message content (text or structured content list)
        repeat_count: Number of times to repeat the request
        expected_response: List of strings expected in the response text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_logprobs: Number of top logprobs to return per token

    Returns:
        ChatPayloadWithLogprobs that validates logprobs in response
    """
    body: Dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }

    return ChatPayloadWithLogprobs(
        body=body,
        repeat_count=repeat_count,
        expected_log=[],
        expected_response=expected_response or ["AI", "knock", "joke"],
    )


def completion_payload_with_logprobs(
    prompt: str = TEXT_PROMPT,
    repeat_count: int = 1,
    expected_response: Optional[List[str]] = None,
    max_tokens: int = 50,
    temperature: float = 0.0,
    logprobs: int = 5,
) -> CompletionPayloadWithLogprobs:
    """
    Create a completion payload that requests and validates logprobs in the response.

    Args:
        prompt: Text prompt
        repeat_count: Number of times to repeat the request
        expected_response: List of strings expected in the response text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        logprobs: Number of logprobs to return per token

    Returns:
        CompletionPayloadWithLogprobs that validates logprobs in response
    """
    body: Dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": logprobs,
    }

    return CompletionPayloadWithLogprobs(
        body=body,
        repeat_count=repeat_count,
        expected_log=[],
        expected_response=expected_response or ["AI", "knock", "joke"],
    )
