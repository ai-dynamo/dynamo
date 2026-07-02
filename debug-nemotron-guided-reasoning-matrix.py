# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the six-case Nemotron guided-reasoning deployment matrix."""

import argparse
import json
import urllib.request

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
            "additionalProperties": False,
        },
    },
}
CASES = [
    (
        "required_tool",
        "required",
        "What is the weather in San Francisco? You must use get_weather.",
    ),
    (
        "auto_tool",
        "auto",
        "What is the weather in San Francisco? Use get_weather to answer.",
    ),
    (
        "auto_direct",
        "auto",
        "What is 37 multiplied by 19? Answer directly and do not use a tool.",
    ),
]
TAGS = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<|tool_call|>",
    "<|analysis|>",
    "<|final|>",
)


def post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=240)


def collect(url, payload, stream, expected_reasoning_field):
    response = post(url, payload)
    reasoning_content, reasoning, content, finish = "", "", "", None
    calls = {}
    if not stream:
        body = json.load(response)
        choice = body["choices"][0]
        message = choice["message"]
        reasoning_content = message.get("reasoning_content") or ""
        reasoning = message.get("reasoning") or ""
        content = message.get("content") or ""
        finish = choice.get("finish_reason")
        for index, call in enumerate(message.get("tool_calls") or []):
            calls[index] = call
    else:
        saw_expected_reasoning = False
        tool_emission_started = False
        terminal_finish_count = 0
        done_count = 0
        for raw in response:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                done_count += 1
                continue
            if not line.startswith("data: "):
                continue
            assert done_count == 0, ("data event emitted after [DONE]", line)
            event = json.loads(line[6:])
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                reasoning_content_delta = delta.get("reasoning_content") or ""
                reasoning_delta = delta.get("reasoning") or ""
                if reasoning_content_delta or reasoning_delta:
                    assert not tool_emission_started, (
                        "reasoning emitted after tool delta",
                        delta,
                    )
                expected_reasoning_delta = (
                    reasoning_delta
                    if expected_reasoning_field == "reasoning"
                    else reasoning_content_delta
                )
                if expected_reasoning_delta:
                    saw_expected_reasoning = True
                reasoning_content += reasoning_content_delta
                reasoning += reasoning_delta
                content += delta.get("content") or ""
                if choice.get("finish_reason") is not None:
                    terminal_finish_count += 1
                    finish = choice["finish_reason"]
                for part in delta.get("tool_calls") or []:
                    assert saw_expected_reasoning, (
                        "tool delta emitted before reasoning",
                        part,
                    )
                    tool_emission_started = True
                    assert set(part) <= {"index", "id", "type", "function"}, part
                    index = part.get("index", 0)
                    call = calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    call["id"] += part.get("id") or ""
                    call["type"] += part.get("type") or ""
                    function = part.get("function") or {}
                    assert set(function) <= {"name", "arguments"}, function
                    call["function"]["name"] += function.get("name") or ""
                    call["function"]["arguments"] += function.get("arguments") or ""
        assert terminal_finish_count == 1, terminal_finish_count
        assert done_count == 1, done_count
    return (
        reasoning_content,
        reasoning,
        content,
        [calls[index] for index in sorted(calls)],
        finish,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    results = []
    for name, tool_choice, prompt in CASES:
        for stream in (False, True):
            expected_reasoning_field = (
                "reasoning"
                if args.label.startswith("native-vllm")
                else "reasoning_content"
            )
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [TOOL],
                "tool_choice": tool_choice,
                "parallel_tool_calls": False,
                "temperature": 0,
                "max_tokens": 1024,
                "stream": stream,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            reasoning_content, native_reasoning, content, calls, finish = collect(
                args.url, payload, stream, expected_reasoning_field
            )
            if args.label.startswith("native-vllm"):
                reasoning = native_reasoning
                assert reasoning_content == "", (name, stream, reasoning_content)
            else:
                reasoning = reasoning_content
                assert native_reasoning == "", (name, stream, native_reasoning)

            combined = (
                reasoning_content + native_reasoning + content + json.dumps(calls)
            )
            leaked = [tag for tag in TAGS if tag in combined]
            native_vllm_stream_delimiter = (
                args.label.startswith("native-vllm")
                and name == "required_tool"
                and stream
                and content.strip() == "<tool_call>"
                and leaked == ["<tool_call>"]
            )
            if name.endswith("tool"):
                assert reasoning.strip(), (name, stream, "missing reasoning")
                assert len(calls) == 1, (name, stream, calls)
                call = calls[0]
                assert set(call) == {"id", "type", "function"}, call
                assert isinstance(call["id"], str) and call["id"], call
                assert call["type"] == "function", call
                assert set(call["function"]) == {"name", "arguments"}, call
                assert call["function"]["name"] == "get_weather", call
                raw_arguments = call["function"]["arguments"]
                assert isinstance(raw_arguments, str), call
                call_args = json.loads(raw_arguments)
                assert set(call_args) == {"location"}, call_args
                assert isinstance(call_args["location"], str), call_args
                assert "San Francisco" in call_args["location"], call_args

                if args.label.startswith("native-sglang") and name == "auto_tool":
                    assert content in {"", "\n", "\n\n"}, (name, stream, repr(content))
                elif native_vllm_stream_delimiter:
                    pass
                else:
                    assert content == "", (name, stream, repr(content))
                native_vllm_length = (
                    args.label.startswith("native-vllm")
                    and name == "required_tool"
                    and not stream
                    and finish == "length"
                )
                assert finish == "tool_calls" or native_vllm_length, (
                    name,
                    stream,
                    finish,
                )
            else:
                assert reasoning.strip(), (name, stream, "missing reasoning")
                assert not calls, (name, stream, calls)
                assert content.strip(), (name, stream, repr(content))
                assert finish == "stop", (name, stream, finish)
            assert not leaked or native_vllm_stream_delimiter, (name, stream, leaked)
            results.append(
                {
                    "endpoint": args.label,
                    "case": name,
                    "stream": stream,
                    "reasoning_chars": len(reasoning),
                    "content": content,
                    "tool_calls": len(calls),
                    "tool_name": calls[0]["function"]["name"] if calls else None,
                    "finish_reason": finish,
                    "tag_leaks": leaked,
                }
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
