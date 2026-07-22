# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from tests.utils.client import send_request
from tests.utils.constants import DefaultPort
from tests.utils.payloads import (
    AnthropicMessagesPayload,
    AnthropicMessagesStreamPayload,
    CachedTokensChatPayload,
    ChatPayload,
    ChatPayloadWithLogprobs,
    ClearKVBlocksPayload,
    CompletionPayload,
    CompletionPayloadWithLogprobs,
    ElasticEPScalePayload,
    EmbeddingPayload,
    GuidedDecodingChatPayload,
    ImagesPayload,
    KvEventMetricsPayload,
    LMCacheMetricsPayload,
    MetricsPayload,
    ResponsesPayload,
    ResponsesStreamPayload,
    RouterNvextChatPayload,
    SGLangMetricsPayload,
    TRTLLMMetricsPayload,
    VLLMMetricsPayload,
)
from tests.utils.router_nvext import RouterNvextExpectation, router_nvext_extra_body

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

# Deliberately distinct from other cache-test prompts and long enough to span
# multiple vLLM cache blocks.
CLEAR_KV_BLOCKS_PROMPT = """This is the unified vLLM block-clearing verification prompt, identified by the unique \
phrase cobalt-orchid-riverstone. Imagine a research station built beside a quiet polar observatory where engineers \
catalog unusual signals from distant stars. Describe how the team prepares its instruments, checks redundant clocks, \
records atmospheric conditions, and compares each observation with the previous night. Include the roles of the lead \
astronomer, systems engineer, data archivist, and safety coordinator. Explain why repeatable procedures matter when a \
faint signal could be caused by weather, hardware drift, software timing, or a genuine astronomical event. Then give a \
brief account of the team's morning review, including how they preserve raw measurements, annotate anomalies, and plan \
the next observation window. Keep the answer factual and concise while retaining the cobalt-orchid-riverstone marker."""


def chat_payload_default(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    stream: bool = False,
    extra_body: Optional[Dict[str, Any]] = None,
    router_nvext_expectation: RouterNvextExpectation | None = None,
) -> ChatPayload:
    body = {
        "messages": [
            {
                "role": "user",
                "content": TEXT_PROMPT,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if extra_body:
        body.update(extra_body)

    common_args = {
        "body": body,
        "repeat_count": repeat_count,
        "expected_log": expected_log or [],
        # Accept any of these keywords in the response (case-insensitive)
        "expected_response": expected_response
        or ["AI", "knock", "joke", "think", "artificial", "intelligence"],
    }
    if router_nvext_expectation:
        return RouterNvextChatPayload(
            **common_args,
            router_nvext_expectation=router_nvext_expectation,
        )
    return ChatPayload(
        **common_args,
    )


def cached_tokens_chat_payload(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
    min_cached_tokens: int = 64,
    extra_body: Optional[Dict[str, Any]] = None,
    router_nvext_expectation: RouterNvextExpectation | None = None,
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
    body = {
        "messages": [
            {
                "role": "user",
                "content": LONG_PROMPT_FOR_CACHING,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if extra_body:
        body.update(extra_body)

    return CachedTokensChatPayload(
        body=body,
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response
        or ["Aeloria", "Eldoria", "explorer", "ancient", "character", "background"],
        min_cached_tokens=min_cached_tokens,
        router_nvext_expectation=router_nvext_expectation,
    )


def clear_kv_blocks_payload(
    max_tokens: int = 16,
    timeout: int = 60,
) -> ClearKVBlocksPayload:
    """Create an admin-then-infer payload for unified vLLM cache clearing."""
    return ClearKVBlocksPayload(
        body={
            "messages": [{"role": "user", "content": CLEAR_KV_BLOCKS_PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=timeout,
    )


def router_selection_chat_payload_default(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    stream: bool = False,
) -> ChatPayload:
    return chat_payload_default(
        repeat_count=repeat_count,
        expected_response=expected_response,
        expected_log=expected_log,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        extra_body=router_nvext_extra_body(["worker_id"]),
        router_nvext_expectation=RouterNvextExpectation(worker_id=True),
    )


def router_cached_tokens_chat_payload(
    repeat_count: int = 3,
    expected_response: Optional[List[str]] = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
    min_cached_tokens: int = 64,
) -> CachedTokensChatPayload:
    return cached_tokens_chat_payload(
        repeat_count=repeat_count,
        expected_response=expected_response,
        max_tokens=max_tokens,
        temperature=temperature,
        min_cached_tokens=min_cached_tokens,
    )


def guided_decoding_chat_payload_default(
    repeat_count: int = 1,
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> GuidedDecodingChatPayload:
    """Create a json_schema guided decoding chat payload."""
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    return GuidedDecodingChatPayload(
        body={
            "messages": [
                {"role": "user", "content": "What is 2+2? Return only JSON."},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer_schema", "schema": schema},
            },
        },
        repeat_count=repeat_count,
        expected_log=[],
        expected_response=[],
        required_keys=["answer"],
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
    check_lifecycle_gauges: bool = False,
) -> MetricsPayload:
    """Create a metrics payload for the specified backend.

    Args:
        min_num_requests: Minimum number of requests expected in metrics
        repeat_count: Number of times to repeat the request
        expected_log: Expected log messages
        backend: Backend type ('vllm', 'sglang', 'trtllm', 'lmcache')
        port: Port to use for metrics endpoint
        check_lifecycle_gauges: Assert the unified-only lifecycle gauges
            (``cleanup_time_seconds``, ``drain_time_seconds``,
            ``kv_cache_hit_rate``) are registered. Default False because
            legacy entry points don't emit them.

    Returns:
        Backend-specific MetricsPayload subclass based on backend parameter
    """
    common_args: dict[str, Any] = {
        "body": {},
        "repeat_count": repeat_count,
        "expected_log": expected_log or [],
        "expected_response": [],
        "min_num_requests": min_num_requests,
        "port": port,
        "check_lifecycle_gauges": check_lifecycle_gauges,
    }

    # Return backend-specific payload class
    if backend == "vllm":
        return VLLMMetricsPayload(**common_args)
    elif backend == "sglang":
        return SGLangMetricsPayload(**common_args)
    elif backend == "trtllm":
        return TRTLLMMetricsPayload(**common_args)
    elif backend == "lmcache":
        return LMCacheMetricsPayload(**common_args)
    else:
        # Default to base MetricsPayload for unknown backends
        return MetricsPayload(**common_args)


def kv_events_metrics_payload(
    *,
    event_type: str = "stored",
    min_received: int = 1,
    min_accepted: int = 1,
    port: int = DefaultPort.SYSTEM1.value,
    system_ports: Optional[List[int]] = None,
    settle_seconds: float = 0.5,
) -> KvEventMetricsPayload:
    return KvEventMetricsPayload(
        body={},
        expected_response=[],
        expected_log=[],
        port=port,
        system_ports=system_ports or [],
        event_type=event_type,
        min_received=min_received,
        min_accepted=min_accepted,
        settle_seconds=settle_seconds,
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
    expected_num_choices: Optional[int] = None,
    max_attempts: int = 1,
    timeout: int = 60,
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
            expected_num_choices=expected_num_choices,
            max_attempts=max_attempts,
            timeout=timeout,
        )
    else:
        return ChatPayload(
            body=body,
            repeat_count=repeat_count,
            expected_log=expected_log or [],
            expected_response=expected_response or [],
            expected_num_choices=expected_num_choices,
            max_attempts=max_attempts,
            timeout=timeout,
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
    extra_body: Optional[Dict[str, Any]] = None,
) -> EmbeddingPayload:
    body: Dict[str, Any] = {
        "input": ["The sky is blue.", "Machine learning is fascinating."],
    }
    if extra_body:
        body.update(extra_body)
    return EmbeddingPayload(
        body=body,
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
    extra_body: Optional[Dict[str, Any]] = None,
) -> EmbeddingPayload:
    # Normalize input to list for consistent processing
    if isinstance(input_text, str):
        input_list = [input_text]
        expected_count = 1
    else:
        input_list = input_text
        expected_count = len(input_text)

    body: Dict[str, Any] = {
        "input": input_list,
    }
    if extra_body:
        body.update(extra_body)

    return EmbeddingPayload(
        body=body,
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


def images_payload_default(
    repeat_count: int = 1,
    timeout: int = 60,
) -> ImagesPayload:
    """Default image-generation request for the raw-media (DiffusionEngine)
    path. The sample diffusion engine returns a fixed 1x1 PNG whose base64
    begins with the PNG signature ``iVBOR`` — the validation anchor."""
    return ImagesPayload(
        body={
            "prompt": "a red balloon over green hills",
            "n": 1,
            "response_format": "b64_json",
        },
        expected_response=["iVBOR"],
        expected_log=[],
        repeat_count=repeat_count,
        timeout=timeout,
    )


def make_images_health_check(port: int, model: str):
    def _check_images_endpoint(remaining_timeout: float = 30.0) -> bool:
        payload = images_payload_default(repeat_count=1).with_model(model)
        payload.expected_response = []
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

    return _check_images_endpoint


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


def responses_payload_default(
    repeat_count: int = 1,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> ResponsesPayload:
    """Create a default Responses API payload (non-streaming).

    For full compliance testing, use the OpenResponses bun CLI instead.
    """
    return ResponsesPayload(
        body={
            "input": TEXT_PROMPT,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response
        or ["AI", "knock", "joke", "think", "artificial", "intelligence"],
    )


def responses_stream_payload_default(
    repeat_count: int = 1,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> ResponsesStreamPayload:
    """Create a default Responses API streaming payload.

    For full compliance testing, use the OpenResponses bun CLI instead.
    """
    return ResponsesStreamPayload(
        body={
            "input": TEXT_PROMPT,
            "stream": True,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response
        or ["AI", "knock", "joke", "think", "artificial", "intelligence"],
    )


def anthropic_messages_payload_default(
    repeat_count: int = 1,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> AnthropicMessagesPayload:
    """Create a default Anthropic Messages API payload (non-streaming)."""
    return AnthropicMessagesPayload(
        body={
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": TEXT_PROMPT,
                }
            ],
            "temperature": temperature,
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response
        or ["AI", "knock", "joke", "think", "artificial", "intelligence"],
    )


def anthropic_messages_stream_payload_default(
    repeat_count: int = 1,
    expected_response: Optional[List[str]] = None,
    expected_log: Optional[List[str]] = None,
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> AnthropicMessagesStreamPayload:
    """Create a default Anthropic Messages API streaming payload."""
    return AnthropicMessagesStreamPayload(
        body={
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": TEXT_PROMPT,
                }
            ],
            "stream": True,
            "temperature": temperature,
        },
        repeat_count=repeat_count,
        expected_log=expected_log or [],
        expected_response=expected_response
        or ["AI", "knock", "joke", "think", "artificial", "intelligence"],
    )


def elastic_ep_scale_payload(
    new_data_parallel_size: int,
    content: str = TEXT_PROMPT,
    max_tokens: int = 64,
    expected_response: Optional[List[str]] = None,
    system_port: int = DefaultPort.SYSTEM1.value,
    timeout: int = 300,
) -> ElasticEPScalePayload:
    """Scale the live data-parallel size, then verify a chat still completes."""
    return ElasticEPScalePayload(
        body={
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        new_data_parallel_size=new_data_parallel_size,
        system_port=system_port,
        expected_response=expected_response
        or ["AI", "knock", "joke", "think", "artificial", "intelligence"],
        timeout=timeout,
    )
