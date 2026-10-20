# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the standalone rolling-upgrade HTTP client."""

import json
import math


class ContractError(Exception):
    """An external response violated the compatibility contract."""


def check(condition, message):
    """Enforce a response contract even when Python optimization is enabled."""
    if not condition:
        raise ContractError(message)


def validate_embedding(body, count, dimensions):
    """Require finite float vectors with the requested shape and indices."""
    check(body["object"] == "list", body)
    check(len(body["data"]) == count, body)
    for index, item in enumerate(body["data"]):
        check(item["index"] == index, item)
        check(item["object"] == "embedding", item)
        vector = item["embedding"]
        # Do not decode strings here: float requests must return JSON arrays.
        check(
            isinstance(vector, list),
            f"Expected float array, got {type(vector).__name__}",
        )
        check(len(vector) == dimensions, len(vector))
        check(
            all(type(x) in (int, float) and math.isfinite(x) for x in vector),
            "Embedding vector must contain finite numbers",
        )


def validate_chat(body, max_tokens, stop=None):
    """Validate unary chat content, token limits, and derived stop semantics."""
    check("error" not in body, body)
    check(len(body["choices"]) == 1, body)
    choice = body["choices"][0]
    check(choice["index"] == 0, choice)
    check(choice["finish_reason"] in ("stop", "length"), choice)
    message = choice["message"]
    check(message["role"] == "assistant", message)
    check(isinstance(message["content"], str), body)
    if max_tokens > 1 and stop is None:
        check(message["content"].strip(), body)
    check(0 <= body["usage"]["completion_tokens"] <= max_tokens, body)
    if stop is not None:
        check(message["content"] == "", body)  # Stop is a prefix of the baseline.
        check(choice["finish_reason"] == "stop", body)


def validate_stream(lines):
    """Reject stream errors, malformed ordering, and incomplete termination."""
    content, finished, done = [], False, False
    for line in lines:
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            check(line.strip() != "event: error", line)
            continue
        check(line.startswith("data:"), f"Unexpected SSE line: {line}")
        check(not done, "Data after [DONE]")
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            continue
        chunk = json.loads(data)
        check("error" not in chunk, chunk)
        for choice in chunk["choices"]:
            check(choice["index"] == 0, choice)
            text = choice.get("delta", {}).get("content")
            if text:
                check(isinstance(text, str), "Stream content must be a string")
                check(not finished, "Content after finish_reason")
                content.append(text)
            if choice.get("finish_reason") is not None:
                check(choice["finish_reason"] in ("stop", "length"), choice)
                finished = True
    check(done and finished, "Incomplete SSE response")
    check("".join(content).strip(), "Empty streamed content")
