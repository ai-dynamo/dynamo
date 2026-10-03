# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import uuid

from dynamo.runtime import Client, Context


def assert_native_completion(
    outputs: list[dict],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "length",
) -> None:
    """Check streamed token accounting and the single terminal response."""
    assert outputs, "Sidecar produced no response"
    assert sum(len(output["token_ids"]) for output in outputs) == completion_tokens
    assert sum(output.get("finish_reason") is not None for output in outputs) == 1
    terminal = outputs[-1]
    assert terminal["finish_reason"] == finish_reason, terminal
    usage = terminal["completion_usage"]
    assert usage["prompt_tokens"] == prompt_tokens, usage
    assert usage["completion_tokens"] == completion_tokens, usage
    assert usage["total_tokens"] == prompt_tokens + completion_tokens, usage


def _assert_logprob(value: float) -> None:
    assert math.isfinite(value) and -9999.0 < value <= 0.0, value


async def assert_native_logprobs(
    *, backend: str, model: str, client: Client, worker_id: int
) -> None:
    """Check engine logprobs before the frontend converts them to OpenAI format."""
    payload = {
        "model": model,
        "token_ids": [11] * 128,
        "stop_conditions": {"max_tokens": 8, "ignore_eos": True},
        "sampling_options": {"temperature": 0.0},
        "output_options": {"logprobs": 2, "prompt_logprobs": 2},
    }
    context = Context(f"logprobs-{uuid.uuid4()}")

    async def collect() -> list[dict]:
        stream = await client.direct(
            payload, worker_id, annotated=False, context=context
        )
        return [output async for output in stream]

    try:
        outputs = await asyncio.wait_for(collect(), timeout=30)
    finally:
        context.stop_generating()

    assert_native_completion(outputs, prompt_tokens=128, completion_tokens=8)
    assert sum(bool(output["token_ids"]) for output in outputs) > 1
    for output in outputs:
        tokens = output["token_ids"]
        selected = output["log_probs"]
        candidates = output["top_logprobs"]
        assert len(selected) == len(candidates) == len(tokens), output
        if backend == "vllm":
            assert isinstance(output["text"], str), output
        for token, logprob, row in zip(tokens, selected, candidates):
            _assert_logprob(logprob)
            if backend == "vllm":
                assert len(row) == 3, row
                for entry in row[:2]:
                    assert entry["token_id"] == token, row
                    assert entry["logprob"] == logprob, row
                    assert entry["rank"] == 1, row
                assert row[2]["rank"] == 2, row
                assert row[1]["token_id"] != row[2]["token_id"], row
            else:
                assert len(row) == 2, row
                for entry in row:
                    if entry["token_id"] == token:
                        assert entry["logprob"] == logprob, row
                # Greedy ties may put the selected token outside the native top two.
                assert row[0]["logprob"] == logprob, row
                assert [entry["rank"] for entry in row] == [1, 2], row
                assert row[0]["token_id"] != row[1]["token_id"], row
            for entry in row:
                assert entry["rank"] > 0, row
                _assert_logprob(entry["logprob"])
        if output.get("finish_reason") is None:
            assert output.get("engine_data") is None, output

    prompt = outputs[-1]["engine_data"]["prompt_logprobs"]
    assert len(prompt) == len(payload["token_ids"]), prompt
    assert prompt[0] is None, prompt[0]
    for token, entries in zip(payload["token_ids"][1:], prompt[1:]):
        assert 2 <= len(entries) <= 3, entries
        assert str(token) in entries, entries
        if backend == "vllm":
            assert {1, 2}.issubset(entry["rank"] for entry in entries.values()), entries
        for candidate, entry in entries.items():
            assert 0 <= int(candidate) <= 2**32 - 1, candidate
            if backend == "vllm":
                assert entry["rank"] > 0, entry
            _assert_logprob(entry["logprob"])
