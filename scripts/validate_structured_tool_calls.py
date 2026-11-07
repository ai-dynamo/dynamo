# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify tools + auto + structured output against a running Dynamo endpoint.

Requires openai with streaming helpers and jsonschema. For example:
python scripts/validate_structured_tool_calls.py --base-url http://localhost:8000/v1 --model MODEL
"""

import argparse
import json
import os

import jsonschema
from openai import OpenAI

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temperature_c": {"type": "integer"}},
    "required": ["city", "temperature_c"],
    "additionalProperties": False,
}
FUNCTION = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "enum": ["Seattle"]}},
        "required": ["city"],
        "additionalProperties": False,
    },
    "strict": True,
}
WEATHER = {"city": "Seattle", "temperature_c": 17}
PROMPT = (
    "Call get_weather for Seattle before answering. Do not invent a weather result."
)
DIRECT_PROMPT = 'Do not call tools. Return this known record: {"city":"Seattle","temperature_c":17}.'


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_answer(content: str) -> None:
    answer = json.loads(content)
    jsonschema.validate(answer, SCHEMA)
    check(answer == WEATHER, f"Unexpected weather record: {answer}")


def chat_round_trip(
    client: OpenAI, model: str, stream: bool, response_format: dict
) -> None:
    def complete(messages: list):
        options = {
            "model": model,
            "messages": messages,
            "tools": [{"type": "function", "function": FUNCTION}],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "response_format": response_format,
            "max_tokens": 2048,
            "temperature": 0,
        }
        if stream:
            with client.chat.completions.stream(**options) as events:
                return events.get_final_completion().choices[0]
        return client.chat.completions.create(**options).choices[0]

    messages = [{"role": "user", "content": PROMPT}]
    choice = complete(messages)
    check(choice.finish_reason == "tool_calls", "Expected finish_reason=tool_calls")
    calls = choice.message.tool_calls or []
    check(len(calls) == 1, "Expected one get_weather call")
    call = calls[0]
    check(call.function.name == "get_weather", "Unexpected tool name")
    check(json.loads(call.function.arguments) == {"city": "Seattle"}, "Wrong arguments")
    messages.extend(
        [
            {"role": "assistant", "tool_calls": [call.model_dump()]},
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(WEATHER)},
        ]
    )
    final = complete(messages)
    check(
        final.finish_reason == "stop", "Expected a final answer after the tool result"
    )
    check(not final.message.tool_calls, "Expected final text without tool calls")
    check_answer(final.message.content)

    direct = complete([{"role": "user", "content": DIRECT_PROMPT}])
    check(direct.finish_reason == "stop", "Expected a direct structured answer")
    check(not direct.message.tool_calls, "Expected a direct answer without tool calls")
    check_answer(direct.message.content)


def responses_round_trip(
    client: OpenAI, model: str, stream: bool, response_format: dict
) -> None:
    def complete(items: list):
        options = {
            "model": model,
            "input": items,
            "tools": [{"type": "function", **FUNCTION}],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "text": {"format": response_format},
            "max_output_tokens": 2048,
            "temperature": 0,
        }
        if stream:
            with client.responses.stream(**options) as events:
                return events.get_final_response()
        return client.responses.create(**options)

    items = [{"role": "user", "content": PROMPT}]
    response = complete(items)
    calls = [item for item in response.output if item.type == "function_call"]
    check(len(calls) == 1, "Expected one Responses function_call")
    call = calls[0]
    check(call.name == "get_weather", "Unexpected tool name")
    check(json.loads(call.arguments) == {"city": "Seattle"}, "Wrong arguments")
    items.extend(item.model_dump(exclude_none=True) for item in response.output)
    items.append(
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(WEATHER),
        }
    )
    final = complete(items)
    check(final.status == "completed", "Expected a completed final response")
    check(
        not any(item.type == "function_call" for item in final.output),
        "Expected final text",
    )
    check_answer(final.output_text)

    direct = complete([{"role": "user", "content": DIRECT_PROMPT}])
    check(direct.status == "completed", "Expected a completed direct response")
    check(
        not any(item.type == "function_call" for item in direct.output),
        "Expected a direct answer without tool calls",
    )
    check_answer(direct.output_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    with OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=180,
        max_retries=0,
    ) as client:
        for stream in [False, True]:
            for kind in ["json_schema", "json_object"]:
                schema_format = {"name": "weather", "schema": SCHEMA, "strict": True}
                chat_format = {"type": kind}
                responses_format = {"type": kind}
                if kind == "json_schema":
                    chat_format["json_schema"] = schema_format
                    responses_format.update(schema_format)
                chat_round_trip(client, args.model, stream, chat_format)
                responses_round_trip(client, args.model, stream, responses_format)
                print(f"PASS Chat Completions and Responses: {kind}, stream={stream}")


if __name__ == "__main__":
    main()
