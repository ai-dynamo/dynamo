# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response contracts shared by deployment and component compatibility tests."""

import json
import math


def validate_embedding(body, count, dimensions):
    """Require finite float vectors with the requested shape and indices."""
    assert body["object"] == "list", body
    assert len(body["data"]) == count, body
    for index, item in enumerate(body["data"]):
        assert item["index"] == index, item
        assert item["object"] == "embedding", item
        vector = item["embedding"]
        assert isinstance(
            vector, list
        ), f"Expected float array, got {type(vector).__name__}"
        assert len(vector) == dimensions, len(vector)
        assert all(
            (type(x) in (int, float) and math.isfinite(x) for x in vector)
        ), "Embedding vector must contain finite numbers"


def validate_chat(body, max_tokens, stop=None):
    """Validate unary chat content, token limits, and derived stop semantics."""
    assert "error" not in body, body
    assert len(body["choices"]) == 1, body
    choice = body["choices"][0]
    assert choice["index"] == 0, choice
    assert choice["finish_reason"] in ("stop", "length"), choice
    message = choice["message"]
    assert message["role"] == "assistant", message
    content = message["content"]
    if stop is None:
        assert isinstance(content, str), body
        if max_tokens > 1:
            assert content.strip(), body
    tokens = body["usage"]["completion_tokens"]
    assert type(tokens) is int and 0 <= tokens <= max_tokens, body
    if stop is not None:
        assert content is None or isinstance(content, str), body
        assert stop not in (content or ""), body
        assert choice["finish_reason"] == "stop", body
        assert not message.get("refusal") and (not message.get("tool_calls")), body
        assert not message.get("function_call"), body


def validate_stop_response(body, baseline, stop):
    """Require a stopped response to end before the baseline's stop sequence."""
    content = body["choices"][0]["message"]["content"] or ""
    original = baseline["choices"][0]["message"]["content"]
    stop_index = original.index(stop)
    assert content.strip(), "Interior stop suppressed preceding text"
    assert original.startswith(content), "Stopped output diverged from baseline"
    assert len(content) <= stop_index, "Output continued past the stop position"
    assert (
        body["usage"]["completion_tokens"] < baseline["usage"]["completion_tokens"]
    ), "Stop did not reduce generated tokens"


def validate_stream(lines):
    """Reject stream errors, malformed ordering, and incomplete termination."""
    content, finished, done = ([], False, False)
    for line in lines:
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line.partition(":")[2].removeprefix(" ")
            assert event != "error", line
            continue
        assert line.startswith("data:"), f"Unexpected SSE line: {line}"
        assert not done, "Data after [DONE]"
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            continue
        chunk = json.loads(data)
        assert "error" not in chunk, chunk
        choices = chunk["choices"]
        assert isinstance(choices, list) and len(choices) <= 1, chunk
        for choice in choices:
            assert choice["index"] == 0, choice
            text = choice.get("delta", {}).get("content")
            if text is not None:
                assert isinstance(text, str), "Stream content must be a string"
            if text:
                assert not finished, "Content after finish_reason"
                content.append(text)
            if choice.get("finish_reason") is not None:
                assert not finished, "Duplicate finish_reason"
                assert choice["finish_reason"] in ("stop", "length"), choice
                finished = True
    assert done and finished, "Incomplete SSE response"
    assert "".join(content).strip(), "Empty streamed content"
