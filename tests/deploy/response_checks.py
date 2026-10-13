# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response contracts shared by deployment and component compatibility tests."""

import base64
import json
import math
import struct

from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk


def validate_embedding(
    body, count, dimensions, expected_model, encoding="float"
) -> CreateEmbeddingResponse:
    if encoding == "base64":
        items = []
        for item in body["data"]:
            encoded = item["embedding"]
            assert isinstance(encoded, str), "Expected base64 embedding"
            raw = base64.b64decode(encoded, validate=True)
            assert len(raw) == dimensions * 4, "Unexpected float32 byte count"
            vector = list(struct.unpack(f"<{dimensions}f", raw))
            items.append({**item, "embedding": vector})
        # Preserve the original wire response for failure artifacts.
        body = {**body, "data": items}
    result = CreateEmbeddingResponse.model_validate(body, strict=True)
    assert result.model == expected_model, result
    assert len(result.data) == count, result
    assert result.usage.prompt_tokens > 0, result.usage
    assert result.usage.total_tokens == result.usage.prompt_tokens, result.usage
    for index, item in enumerate(result.data):
        assert item.index == index, item
        assert len(item.embedding) == dimensions, len(item.embedding)
        assert all(math.isfinite(x) for x in item.embedding), "Nonfinite embedding"
    return result


def validate_chat(body, max_tokens=None, stop=None) -> ChatCompletion:
    result = ChatCompletion.model_validate(body, strict=True)
    assert "error" not in body, body
    assert len(result.choices) == 1, result
    choice = result.choices[0]
    assert choice.index == 0, choice
    assert choice.finish_reason in ("stop", "length"), choice
    message = choice.message
    content = message.content
    if stop is None:
        assert content is not None, message
        if max_tokens is not None and max_tokens > 1:
            assert content.strip(), message
    if max_tokens is not None:
        assert result.usage is not None, result
        assert 0 <= result.usage.completion_tokens <= max_tokens, result.usage
    if stop is not None:
        assert stop not in (content or ""), message
        assert choice.finish_reason == "stop", choice
        assert not message.refusal and not message.tool_calls, message
        assert not message.function_call, message
    return result


def validate_stop_response(body: ChatCompletion, baseline: ChatCompletion, stop: str):
    content = body.choices[0].message.content or ""
    original = baseline.choices[0].message.content
    assert original is not None, baseline
    expected = original[: original.index(stop)]
    assert content.strip(), "Interior stop suppressed preceding text"
    assert (
        content.rstrip() == expected.rstrip()
    ), "Stopped output lost or changed prefix"
    assert body.usage is not None and baseline.usage is not None
    assert (
        body.usage.completion_tokens < baseline.usage.completion_tokens
    ), "Stop did not reduce generated tokens"


def validate_stream(lines, expected_model):
    content, finished, done = ([], False, False)
    response_id = None
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
        body = json.loads(data)
        assert "error" not in body, body
        chunk = ChatCompletionChunk.model_validate(body, strict=True)
        assert chunk.model == expected_model, chunk
        if response_id is None:
            response_id = chunk.id
        assert chunk.id == response_id, "Response ID changed during stream"
        assert len(chunk.choices) <= 1, chunk
        for choice in chunk.choices:
            assert choice.index == 0, choice
            if choice.delta.role is not None:
                assert choice.delta.role == "assistant", choice.delta
            text = choice.delta.content
            if text:
                assert not finished, "Content after finish_reason"
                content.append(text)
            if choice.finish_reason is not None:
                assert not finished, "Duplicate finish_reason"
                assert choice.finish_reason in ("stop", "length"), choice
                finished = True
    assert done and finished, "Incomplete SSE response"
    assert "".join(content).strip(), "Empty streamed content"
