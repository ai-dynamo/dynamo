# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert the real frontend's unary/SSE output, then exercise concurrent serving."""

import argparse
import asyncio
import json
import time

import httpx

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Read the weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


async def chat(client, messages, *, stream, tools=False, max_tokens=256):
    body = {
        "model": "deepinfer-mock",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        body["tools"] = [WEATHER]
    if not stream:
        response = await client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        result = response.json()
        return (
            result["choices"][0]["message"],
            result["choices"][0]["finish_reason"],
            result["usage"],
        )

    body["stream_options"] = {"include_usage": True}
    message = {"content": "", "reasoning_content": "", "tool_calls": []}
    calls = {}
    finish = usage = None
    done = False
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            assert "error" not in event, event
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice["delta"]
                for field in ("content", "reasoning_content"):
                    message[field] += delta.get(field) or ""
                for call in delta.get("tool_calls") or []:
                    index = call["index"]
                    if index not in calls:
                        assert call.get("id") and call["function"].get("name"), call
                        calls[index] = {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    for field in ("name", "arguments"):
                        calls[index]["function"][field] += (
                            call.get("function", {}).get(field) or ""
                        )
    assert done and finish and usage, (done, finish, usage)
    assert sorted(calls) == list(range(len(calls))), calls
    message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return message, finish, usage


def check_clean(message, usage):
    for field in ("content", "reasoning_content"):
        value = message.get(field) or ""
        assert not any(
            marker in value for marker in ("<think>", "</think>", "DSML")
        ), message
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0, usage
    assert (
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    ), usage


async def check_case(client, case, stream):
    messages = [{"role": "user", "content": f"DEEPINFER_{case}"}]
    tools = case in ("TOOL", "PARALLEL")
    message, finish, usage = await chat(client, messages, stream=stream, tools=tools)
    check_clean(message, usage)
    if tools:
        assert finish == "tool_calls", (finish, message)
        calls = message["tool_calls"]
        cities = ["Paris", "Tokyo"] if case == "PARALLEL" else ["Paris"]
        assert len(calls) == len(cities) and len({c["id"] for c in calls}) == len(
            cities
        ), calls
        for call, city in zip(calls, cities):
            assert call["function"]["name"] == "get_weather", call
            assert json.loads(call["function"]["arguments"]) == {"city": city}, call
        assert message["reasoning_content"] == "I need the weather tool.", message
        assert not message.get("content"), message
        if case == "TOOL":
            messages += [
                {"role": "assistant", **message},
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": "DEEPINFER_TOOL_RESULT: Paris is 21 C",
                },
            ]
            followup, finish, usage = await chat(
                client, messages, stream=stream, tools=True
            )
            check_clean(followup, usage)
            assert (
                finish == "stop" and followup["content"] == "It is 21 C in Paris."
            ), followup
    elif case == "REASONING":
        assert finish == "stop" and message["content"] == "The answer is 42.", message
        assert (
            message["reasoning_content"] == "I will compute six times seven."
        ), message
    else:
        assert (
            finish == "stop" and message["content"] == "Hello from the DeepInfer mock."
        ), message
    return 2 if case == "TOOL" else 1


async def main(base_url, concurrency):
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        for _ in range(120):
            try:
                health = await client.get("/health")
                response = await client.get("/v1/models")
                if (
                    health.is_success
                    and response.is_success
                    and any(
                        m["id"] == "deepinfer-mock" for m in response.json()["data"]
                    )
                ):
                    break
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("DeepInfer model did not register within 60 seconds")
        count = 0
        started = time.perf_counter()
        for stream in (False, True):
            for case in ("PLAIN", "REASONING", "TOOL", "PARALLEL"):
                count += await check_case(client, case, stream)
            message, finish, usage = await chat(
                client,
                [{"role": "user", "content": "hello"}],
                stream=stream,
                max_tokens=1,
            )
            assert finish == "length" and usage["completion_tokens"] == 1, (
                message,
                finish,
                usage,
            )
            count += 1
        completed = await asyncio.gather(
            *(
                check_case(
                    client,
                    ("PLAIN", "REASONING", "TOOL", "PARALLEL")[i % 4],
                    bool(i % 2),
                )
                for i in range(concurrency)
            )
        )
        count += sum(completed)
        print(
            f"PASS: {count} HTTP requests; unary + SSE, reasoning, tools, parallel calls, tool results, length cap; concurrency={concurrency}; {time.perf_counter() - started:.2f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    asyncio.run(main(args.base_url, args.concurrency))
