#!/usr/bin/env python3
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
"""Check Flint chat, streaming, and tool round trips through Dynamo's frontend."""

import argparse
import json
import os

from openai import OpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_tool(name: str, arguments: str) -> None:
    require(name == "get_weather", f"Unexpected tool: {name!r}")
    parsed = json.loads(arguments)
    require(
        isinstance(parsed, dict), f"Tool arguments are not an object: {arguments!r}"
    )
    require(set(parsed) == {"city"}, f"Unexpected tool argument keys: {arguments!r}")
    city = parsed["city"]
    require(
        isinstance(city, str) and bool(city.strip()),
        f"Tool arguments do not match the schema: {arguments!r}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="RecursiveSI/flint")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "unused"),
        timeout=args.timeout,
        max_retries=0,
    )
    options = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"top_k": 20},
    }

    response = client.chat.completions.create(
        **options,
        messages=[{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
    )
    choice = response.choices[0]
    require(choice.finish_reason == "stop", f"Chat ended: {choice.finish_reason}")
    require(bool(choice.message.content), "Chat returned no final content")
    print("PASS chat")

    text = ""
    finish_reason = None
    with client.chat.completions.create(
        **options,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        stream=True,
    ) as stream:
        for chunk in stream:
            for part in chunk.choices:
                text += part.delta.content or ""
                finish_reason = part.finish_reason or finish_reason
    require(finish_reason == "stop", f"Streaming chat ended: {finish_reason}")
    require(bool(text.strip()), "Streaming chat returned no final content")
    print("PASS streaming chat")

    messages = [
        {"role": "user", "content": "Use get_weather to check the weather in Paris."}
    ]
    for tool_choice in ("auto", "required"):
        response = client.chat.completions.create(
            **options,
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
            parallel_tool_calls=False,
        )
        choice = response.choices[0]
        require(
            choice.finish_reason == "tool_calls",
            f"{tool_choice} tool call was not returned",
        )
        calls = choice.message.tool_calls or []
        require(len(calls) == 1, f"Expected one tool call, got {len(calls)}")
        call = calls[0]
        require(bool(call.id), f"{tool_choice} tool call has no ID")
        check_tool(call.function.name, call.function.arguments)
        print(f"PASS {tool_choice} tool call")
    # Preserve reasoning_content and the exact JSON argument string in history.
    messages.append(choice.message.model_dump(exclude_none=True))
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": '{"city":"Paris","weather":"sunny","temperature_c":21}',
        }
    )
    response = client.chat.completions.create(
        **options, messages=messages, tools=TOOLS, tool_choice="none"
    )
    choice = response.choices[0]
    require(choice.finish_reason == "stop", "Tool follow-up did not finish")
    require(bool(choice.message.content), "Tool follow-up returned no final content")
    require(not choice.message.tool_calls, "tool_choice=none returned a tool call")
    print("PASS multi-turn tool follow-up")

    name = ""
    arguments = ""
    call_id = ""
    finish_reason = None
    with client.chat.completions.create(
        **options,
        messages=messages[:1],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        parallel_tool_calls=False,
        stream=True,
    ) as stream:
        for chunk in stream:
            for part in chunk.choices:
                finish_reason = part.finish_reason or finish_reason
                for delta in part.delta.tool_calls or []:
                    require(
                        delta.index == 0, "Received an unexpected parallel tool call"
                    )
                    call_id += delta.id or ""
                    if delta.function:
                        name += delta.function.name or ""
                        arguments += delta.function.arguments or ""
    require(finish_reason == "tool_calls", "Streaming named tool call did not finish")
    require(bool(call_id), "Streaming tool call has no ID")
    check_tool(name, arguments)
    print("PASS streaming named tool call")


if __name__ == "__main__":
    main()
