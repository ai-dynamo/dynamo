# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Request validation and wire helpers for the multimodal DAG example."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PUBLIC_MODEL_NAME = "multimodal-dag"
DEFAULT_BACKEND_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

ORCHESTRATOR_ENDPOINT = "multimodal_dag.orchestrator.generate"
VISION_ENCODER_ENDPOINT = "multimodal_dag.vision_encoder.generate"
CLASSIFIER_ENDPOINT = "multimodal_dag.classifier.generate"
VLLM_ENDPOINT = "multimodal_dag.vllm.generate"

_ALLOWED_REQUEST_FIELDS = frozenset(
    {
        "id",
        "request_id",
        "model",
        "messages",
        "stream",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "seed",
        "user",
    }
)
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class ValidatedChatRequest:
    """The supported OpenAI request subset plus Qwen processor input."""

    processor_messages: list[dict[str, Any]]
    image_url: str
    max_tokens: int
    temperature: float | None
    top_p: float | None
    stop: str | list[str] | None
    seed: int | None


@dataclass(frozen=True)
class VllmResult:
    """Collected token-mode output from the internal vLLM worker."""

    text: str
    finish_reason: str
    usage: dict[str, Any] | None


def _parse_content(
    *,
    role: str,
    content: Any,
) -> tuple[str | list[dict[str, Any]], list[str], int]:
    if isinstance(content, str):
        if not content:
            raise ValueError("message text must not be empty")
        return content, [], 1
    if not isinstance(content, list) or not content:
        raise ValueError("message content must be text or a non-empty content list")

    normalized: list[dict[str, Any]] = []
    image_urls: list[str] = []
    text_count = 0
    for part in content:
        if not isinstance(part, Mapping):
            raise ValueError("message content parts must be objects")
        part_type = part.get("type")
        if part_type == "text":
            if set(part) != {"type", "text"}:
                raise ValueError("text content parts support only type and text")
            text = part.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("text content parts must contain non-empty text")
            normalized.append({"type": "text", "text": text})
            text_count += 1
            continue
        if part_type == "image_url":
            if set(part) != {"type", "image_url"}:
                raise ValueError(
                    "image_url content parts support only type and image_url"
                )
            if role != "user":
                raise ValueError("image_url content is supported only in user messages")
            image_url = part.get("image_url")
            if not isinstance(image_url, Mapping):
                raise ValueError("image_url content must contain an image_url object")
            if set(image_url) != {"url"}:
                raise ValueError("image_url objects support only url")
            url = image_url.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError("image_url.url must be a non-empty string")
            normalized.append({"type": "image", "image": url})
            image_urls.append(url)
            continue
        raise ValueError(f"unsupported message content type: {part_type!r}")
    return normalized, image_urls, text_count


def validate_chat_request(request: Mapping[str, Any]) -> ValidatedChatRequest:
    """Validate the intentionally small public request surface."""

    if not isinstance(request, Mapping):
        raise TypeError("chat request must be an object")

    unsupported = sorted(
        field_name
        for field_name, value in request.items()
        if field_name not in _ALLOWED_REQUEST_FIELDS and value is not None
    )
    if unsupported:
        raise ValueError(f"unsupported request fields: {unsupported}")
    if request.get("model") not in (None, PUBLIC_MODEL_NAME):
        raise ValueError(f"model must be {PUBLIC_MODEL_NAME!r}")
    if request.get("stream") not in (None, False):
        raise ValueError("this example supports only stream=false")
    if request.get("n") not in (None, 1):
        raise ValueError("this example supports only n=1")

    seed = request.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")

    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    processor_messages: list[dict[str, Any]] = []
    image_urls: list[str] = []
    text_count = 0
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("messages must contain objects")
        unsupported_message_fields = sorted(
            field_name
            for field_name, value in message.items()
            if field_name not in {"role", "content"} and value is not None
        )
        if unsupported_message_fields:
            raise ValueError(
                f"unsupported message fields: {unsupported_message_fields}"
            )
        role = message.get("role")
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"unsupported message role: {role!r}")
        content, message_image_urls, message_text_count = _parse_content(
            role=role,
            content=message.get("content"),
        )
        processor_messages.append({"role": role, "content": content})
        image_urls.extend(message_image_urls)
        text_count += message_text_count

    if text_count == 0:
        raise ValueError("the request must contain at least one text content part")
    if len(image_urls) != 1:
        raise ValueError(
            f"the request must contain exactly one image_url; got {len(image_urls)}"
        )

    max_tokens = request.get("max_tokens")
    max_completion_tokens = request.get("max_completion_tokens")
    if (
        max_tokens is not None
        and max_completion_tokens is not None
        and max_tokens != max_completion_tokens
    ):
        raise ValueError(
            "max_tokens and max_completion_tokens must match when both are set"
        )
    max_tokens = (
        max_completion_tokens if max_completion_tokens is not None else max_tokens
    )
    if max_tokens is None:
        max_tokens = 64
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise ValueError("max_tokens must be a positive integer")

    temperature = request.get("temperature")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0.0 <= temperature <= 2.0
    ):
        raise ValueError("temperature must be between 0.0 and 2.0")

    top_p = request.get("top_p")
    if top_p is not None and (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not 0.0 <= top_p <= 1.0
    ):
        raise ValueError("top_p must be between 0.0 and 1.0")

    stop = request.get("stop")
    if isinstance(stop, str):
        if not stop:
            raise ValueError("stop must not be empty")
    elif stop is not None and not (
        isinstance(stop, list)
        and 1 <= len(stop) <= 4
        and all(isinstance(item, str) and item for item in stop)
    ):
        raise ValueError("stop must be a string or a list of one to four strings")

    return ValidatedChatRequest(
        processor_messages=processor_messages,
        image_url=image_urls[0],
        max_tokens=max_tokens,
        temperature=float(temperature) if temperature is not None else None,
        top_p=float(top_p) if top_p is not None else None,
        stop=stop,
        seed=seed,
    )


def build_vllm_request(
    *,
    model: str,
    token_ids: Sequence[int],
    external_mm_data: Mapping[str, Any],
    request: ValidatedChatRequest,
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Build Dynamo's token-in/token-out request for the vLLM worker."""

    return {
        "model": model,
        "token_ids": list(token_ids),
        "external_mm_data": dict(external_mm_data),
        "stop_conditions": {
            "max_tokens": request.max_tokens,
            "stop": request.stop,
            "stop_token_ids": None,
            "min_tokens": 0,
            "ignore_eos": False,
        },
        "sampling_options": {
            "n": 1,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
        },
        "output_options": {
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        },
        "eos_token_ids": list(eos_token_ids),
    }


def apply_stop(text: str, stop: str | list[str] | None) -> tuple[str, bool]:
    """Trim decoded text at the earliest requested stop string."""

    if stop is None:
        return text, False
    stop_strings = [stop] if isinstance(stop, str) else stop
    positions = [
        position for value in stop_strings if (position := text.find(value)) >= 0
    ]
    if not positions:
        return text, False
    return text[: min(positions)], True


def chat_chunk(
    *,
    request_id: str,
    content: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    classifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one OpenAI chat-completion stream chunk."""

    delta: dict[str, Any] = {}
    if content is not None:
        delta = {"role": "assistant", "content": content}
    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": PUBLIC_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    if classifier is not None:
        chunk["nvext"] = {"classifier": classifier}
    return chunk
