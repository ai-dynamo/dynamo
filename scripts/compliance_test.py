#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
OpenResponses Compliance Test Suite
====================================
Replicates the 6 tests from https://www.openresponses.org/compliance

These tests validate a Responses API endpoint against the OpenAI spec.
Each test is annotated with the expected response structure.

Usage:
    python scripts/compliance_test.py [--base-url URL] [--model MODEL] [--timeout SECS]

Defaults:
    --base-url  http://localhost:8000/v1
    --model     Qwen/Qwen3-0.6B
    --timeout   30
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Expected Response Schema (from OpenAI Responses API spec)
# ---------------------------------------------------------------------------
#
# A valid ResponseResource must have:
#   {
#     "id": str,              # e.g. "resp_abc123..."
#     "object": "response",   # always "response"
#     "created_at": int,      # unix timestamp
#     "status": str,          # "completed" | "failed" | "in_progress" | "incomplete"
#     "model": str,           # model name
#     "output": [             # array of output items
#       {
#         "type": "message",          # or "function_call"
#         "id": str,                  # e.g. "msg_abc123..."
#         "status": str,              # "completed"
#         "role": "assistant",        # for message type
#         "content": [                # array of content parts
#           {
#             "type": "output_text",
#             "text": str,
#             "annotations": []
#           }
#         ]
#       }
#     ],
#     // optional fields: error, usage, tools, tool_choice, instructions,
#     //   previous_response_id, metadata, temperature, top_p, ...
#   }
#
# A function_call output item:
#   {
#     "type": "function_call",
#     "id": str,              # e.g. "fc_abc123..."
#     "call_id": str,         # e.g. "call_abc123..."
#     "name": str,            # function name
#     "arguments": str,       # JSON string of arguments
#     "status": "completed"
#   }
#
# Streaming SSE events follow this sequence:
#   response.created -> response.in_progress ->
#   response.output_item.added -> response.content_part.added ->
#   N x response.output_text.delta ->
#   response.output_text.done -> response.content_part.done ->
#   response.output_item.done -> response.completed -> [DONE]
#
# Each SSE event has:
#   {
#     "type": str,              # event type (e.g. "response.created")
#     "sequence_number": int,   # monotonically increasing
#     ...event-specific fields
#   }

VALID_SSE_EVENT_TYPES = {
    "response.created",
    "response.queued",
    "response.in_progress",
    "response.completed",
    "response.failed",
    "response.incomplete",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.refusal.delta",
    "response.refusal.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.reasoning.delta",
    "response.reasoning.done",
    "response.reasoning_summary.delta",
    "response.reasoning_summary.done",
    "response.output_text_annotation.added",
    "error",
}

# Required top-level lifecycle events
REQUIRED_LIFECYCLE_EVENTS = {
    "response.created",
    "response.in_progress",
    "response.completed",
}

# Base64 encoded 32x32 red heart PNG (same image used by OpenResponses)
IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAABmklEQVR42tyWAaTyUBzFew/eG4AHz"
    "+MBSAHKBiJRGFKwIgQQJKLUIioBIhCAiCAAEizAQIAECaASqFFJq84nudjnaqvuPnxzgP9xfrq593"
    "8csPn7PwHTKSoViCIEAYEAMhmoKsU2mUCWEQqB5xEMIp/HaGQG2G6RSuH9HQ7H34rFrtPbdz4jl6P"
    "bwmEsl3QA1mt4vcRKk8dz9eg6IpF7tt9fzGY0gCgafFRFo5Blc5vLhf3eCOj1yNhM5GRMVK0aATxPZ"
    "oz09YXjkQDmczJgquGQAPp9WwCNBgG027YACgUC6HRsAZRKBDAY2AJoNv/ZnwzA6WScznG3p4UAymXG"
    "AEkyXrTFAh8fLAGqagQAyGaZpYsi7bHTNPz8MEj//LxuFPo+UBS8vb0KaLXubrRa7aX0RMLCykwmn0"
    "z3+XA4WACcTpCkh9MFAZpmuVXo+mO/w+/HZvNgbblcUCxaSo/Hyck80Yu6XXDcvfVZr79cvMZjuN2U9"
    "O9vKAqjZrfbIZ0mV4TUi9Xqz6jddNy//7+e3n8Fhf/Llo2kxi8AQyGRoDkmAhAAAAAASUVORK5CYII="
)


# ---------------------------------------------------------------------------
# Test Result
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    test_id: str
    name: str
    passed: bool
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    response: Any = None
    stream_event_count: int = 0


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_response_schema(data: dict) -> list[str]:
    """Validate required fields in a ResponseResource.

    Expected top-level fields:
      - id: str (e.g. "resp_...")
      - object: "response"
      - created_at: int (unix timestamp)
      - status: str
      - model: str
      - output: list
    """
    errors = []

    for field_name in ("id", "object", "status", "model"):
        if field_name not in data or not isinstance(data[field_name], str):
            errors.append(f"Missing or invalid required field: {field_name}")

    if data.get("object") != "response":
        errors.append(f"Expected object='response', got '{data.get('object')}'")

    if "created_at" not in data or not isinstance(data["created_at"], (int, float)):
        errors.append("Missing or invalid created_at (expected integer)")

    if "output" not in data or not isinstance(data["output"], list):
        errors.append("Missing or invalid output (expected array)")

    return errors


def validate_output_items(data: dict) -> list[str]:
    """Validate each output item has required fields.

    For type="message":
      - id: str
      - type: "message"
      - status: str (usually "completed")
      - role: str (usually "assistant")
      - content: list of content parts, each with:
          - type: "output_text"
          - text: str
          - annotations: list

    For type="function_call":
      - id: str
      - type: "function_call"
      - call_id: str
      - name: str (function name)
      - arguments: str (JSON-encoded arguments)
      - status: str (usually "completed")
    """
    errors = []
    for i, item in enumerate(data.get("output", [])):
        if "type" not in item:
            errors.append(f"output[{i}]: missing 'type' field")
        if "id" not in item:
            errors.append(f"output[{i}]: missing 'id' field")

        item_type = item.get("type")
        if item_type == "message":
            if "content" not in item or not isinstance(item["content"], list):
                errors.append(f"output[{i}]: message missing 'content' array")
            if "role" not in item:
                errors.append(f"output[{i}]: message missing 'role'")
            if "status" not in item:
                errors.append(f"output[{i}]: message missing 'status'")
        elif item_type == "function_call":
            for fc_field in ("call_id", "name", "arguments", "status"):
                if fc_field not in item:
                    errors.append(
                        f"output[{i}]: function_call missing '{fc_field}'"
                    )
    return errors


def validate_has_output(data: dict) -> list[str]:
    """Check that at least one output item exists."""
    if not data.get("output"):
        return ["Response has no output items"]
    return []


def validate_completed_status(data: dict) -> list[str]:
    """Check that status is 'completed'."""
    status = data.get("status")
    if status != "completed":
        return [f"Expected status='completed', got '{status}'"]
    return []


def validate_has_output_type(data: dict, output_type: str) -> list[str]:
    """Check that at least one output item has the given type.

    For tool calling test: output_type='function_call'
    Expected function_call item structure:
      {
        "type": "function_call",
        "id": "fc_...",
        "call_id": "call_...",
        "name": "get_weather",
        "arguments": "{\"location\": \"San Francisco, CA\"}",
        "status": "completed"
      }
    """
    items = data.get("output", [])
    has_type = any(item.get("type") == output_type for item in items)
    if not has_type:
        types_found = [item.get("type", "unknown") for item in items]
        return [
            f"Expected output item of type '{output_type}' but found: {types_found}"
        ]
    return []


# ---------------------------------------------------------------------------
# SSE Stream Parsing
# ---------------------------------------------------------------------------


def parse_sse_stream(raw_text: str) -> tuple[dict | None, list[dict], list[str]]:
    """Parse SSE text into (final_response, events, errors).

    SSE format:
        event: response.created
        data: {"type":"response.created","sequence_number":0,...}

        event: response.output_text.delta
        data: {"type":"response.output_text.delta","sequence_number":5,...,"delta":"Hello"}

        ...
        data: [DONE]

    Returns:
        final_response: The response object from the response.completed event
        events: List of all parsed event dicts
        errors: List of parsing/validation errors
    """
    events = []
    errors = []
    final_response = None
    current_event = ""
    current_data = ""

    for line in raw_text.split("\n"):
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            current_data = line[5:].strip()
        elif line.strip() == "" and current_data:
            if current_data == "[DONE]":
                current_event = ""
                current_data = ""
                continue
            try:
                parsed = json.loads(current_data)
                events.append(parsed)

                evt_type = parsed.get("type", "unknown")
                if evt_type not in VALID_SSE_EVENT_TYPES:
                    errors.append(f"Unknown SSE event type: {evt_type}")
                if "sequence_number" not in parsed:
                    errors.append(f"{evt_type}: missing sequence_number")

                if current_event in ("response.completed", "response.failed"):
                    if "response" in parsed:
                        final_response = parsed["response"]
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in SSE data: {e}")

            current_event = ""
            current_data = ""

    return final_response, events, errors


def validate_streaming_events(events: list[dict]) -> list[str]:
    """Validate that required lifecycle events are present."""
    errors = []
    event_types = {e.get("type") for e in events}

    for required in REQUIRED_LIFECYCLE_EVENTS:
        if required not in event_types:
            errors.append(f"Missing required SSE event: {required}")

    return errors


# ---------------------------------------------------------------------------
# Test Definitions
# ---------------------------------------------------------------------------
# Each test matches the exact payload from OpenResponses compliance-tests.ts


def get_test_definitions(model: str) -> list[dict]:
    """Return the 6 compliance test definitions with exact payloads.

    Tests:
      1. Basic Text Response - simple message -> text output
      2. Streaming Response  - SSE event sequence validation
      3. System Prompt       - system role + user message
      4. Tool Calling        - function tool -> function_call output
      5. Image Input         - multimodal image content
      6. Multi-turn          - conversation history with assistant turns
    """
    return [
        {
            "id": "1-basic-response",
            "name": "Basic Text Response",
            "description": (
                "Simple user message, validates ResponseResource schema.\n"
                "  Expected: status=completed, output has a message with text content."
            ),
            "streaming": False,
            "payload": {
                "model": model,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Say hello in exactly 3 words.",
                    }
                ],
            },
            "validators": ["schema", "output_items", "has_output", "completed_status"],
        },
        {
            "id": "2-streaming-response",
            "name": "Streaming Response",
            "description": (
                "Validates SSE streaming events and final response.\n"
                "  Expected SSE sequence:\n"
                "    response.created -> response.in_progress ->\n"
                "    response.output_item.added -> response.content_part.added ->\n"
                "    N x response.output_text.delta ->\n"
                "    response.output_text.done -> response.content_part.done ->\n"
                "    response.output_item.done -> response.completed -> [DONE]\n"
                "  Each event must have 'type' and 'sequence_number' fields."
            ),
            "streaming": True,
            "payload": {
                "model": model,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Count from 1 to 5.",
                    }
                ],
                "stream": True,
            },
            "validators": [
                "streaming_events",
                "streaming_schema",
                "streaming_completed_status",
            ],
        },
        {
            "id": "3-system-prompt",
            "name": "System Prompt",
            "description": (
                "Include system role message in input.\n"
                "  Expected: system message is used as instructions, response "
                "reflects the persona."
            ),
            "streaming": False,
            "payload": {
                "model": model,
                "input": [
                    {
                        "type": "message",
                        "role": "system",
                        "content": "You are a pirate. Always respond in pirate speak.",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Say hello.",
                    },
                ],
            },
            "validators": ["schema", "output_items", "has_output", "completed_status"],
        },
        {
            "id": "4-tool-calling",
            "name": "Tool Calling",
            "description": (
                "Define a function tool and verify function_call output.\n"
                "  Expected output item:\n"
                '    type: "function_call"\n'
                '    name: "get_weather"\n'
                '    arguments: JSON string with "location" key\n'
                '    call_id: non-empty string\n'
                '    status: "completed"\n'
                "\n"
                "  Note: Some models (e.g. Qwen3) emit tool calls as\n"
                "  <tool_call>JSON</tool_call> text. Dynamo parses these\n"
                "  and converts them to structured function_call output items.\n"
                "  For best results, launch with --dyn-tool-call-parser."
            ),
            "streaming": False,
            "payload": {
                "model": model,
                "max_output_tokens": 500,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "What's the weather like in San Francisco?",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                }
                            },
                            "required": ["location"],
                        },
                    }
                ],
            },
            "validators": [
                "schema",
                "output_items",
                "has_output",
                "has_function_call",
            ],
        },
        {
            "id": "5-image-input",
            "name": "Image Input",
            "description": (
                "Send image URL in user content (multimodal).\n"
                "  Input uses content array with input_text + input_image.\n"
                "  The image is a 32x32 red heart on white background (base64 PNG).\n"
                "  Expected: model describes the image in its response.\n"
                "  Requires a vision-language model (e.g. Qwen2.5-VL-7B-Instruct)."
            ),
            "streaming": False,
            "payload": {
                "model": model,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "What do you see in this image? Answer in one sentence.",
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{IMAGE_BASE64}",
                                "detail": "auto",
                            },
                        ],
                    }
                ],
            },
            "validators": ["schema", "output_items", "has_output", "completed_status"],
        },
        {
            "id": "6-multi-turn",
            "name": "Multi-turn Conversation",
            "description": (
                "Send assistant + user messages as conversation history.\n"
                "  Input provides a 3-turn conversation:\n"
                '    user: "My name is Alice."\n'
                '    assistant: "Hello Alice! ..."\n'
                '    user: "What is my name?"\n'
                "  Expected: model recalls 'Alice' from history."
            ),
            "streaming": False,
            "payload": {
                "model": model,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "My name is Alice.",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "Hello Alice! Nice to meet you. How can I help you today?",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "What is my name?",
                    },
                ],
            },
            "validators": ["schema", "output_items", "has_output", "completed_status"],
        },
    ]


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------


def run_test(
    base_url: str, test_def: dict, timeout: int
) -> TestResult:
    """Run a single compliance test and return the result."""
    test_id = test_def["id"]
    name = test_def["name"]
    streaming = test_def.get("streaming", False)
    payload = test_def["payload"]
    validators = test_def["validators"]

    start = time.monotonic()

    try:
        if streaming:
            return _run_streaming_test(
                base_url, test_id, name, payload, validators, timeout
            )
        else:
            return _run_non_streaming_test(
                base_url, test_id, name, payload, validators, timeout
            )
    except requests.exceptions.ConnectionError:
        duration_ms = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            name=name,
            passed=False,
            duration_ms=duration_ms,
            errors=["Connection refused -- is the server running?"],
        )
    except requests.exceptions.Timeout:
        duration_ms = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            name=name,
            passed=False,
            duration_ms=duration_ms,
            errors=[f"Request timed out after {timeout}s"],
        )
    except Exception as e:
        duration_ms = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            name=name,
            passed=False,
            duration_ms=duration_ms,
            errors=[f"Unexpected error: {e}"],
        )


def _run_non_streaming_test(
    base_url: str,
    test_id: str,
    name: str,
    payload: dict,
    validators: list[str],
    timeout: int,
) -> TestResult:
    start = time.monotonic()
    resp = requests.post(
        f"{base_url}/responses",
        json=payload,
        timeout=timeout,
    )
    duration_ms = (time.monotonic() - start) * 1000

    if resp.status_code != 200:
        return TestResult(
            test_id=test_id,
            name=name,
            passed=False,
            duration_ms=duration_ms,
            errors=[f"HTTP {resp.status_code}: {resp.text[:200]}"],
            response=resp.text[:500],
        )

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return TestResult(
            test_id=test_id,
            name=name,
            passed=False,
            duration_ms=duration_ms,
            errors=["Response is not valid JSON"],
            response=resp.text[:500],
        )

    errors = _run_validators(data, validators)

    return TestResult(
        test_id=test_id,
        name=name,
        passed=len(errors) == 0,
        duration_ms=duration_ms,
        errors=errors,
        response=data,
    )


def _run_streaming_test(
    base_url: str,
    test_id: str,
    name: str,
    payload: dict,
    validators: list[str],
    timeout: int,
) -> TestResult:
    start = time.monotonic()

    # Stream the response to capture SSE events
    resp = requests.post(
        f"{base_url}/responses",
        json=payload,
        timeout=timeout,
        stream=True,
    )
    raw_text = resp.text
    duration_ms = (time.monotonic() - start) * 1000

    if resp.status_code != 200:
        return TestResult(
            test_id=test_id,
            name=name,
            passed=False,
            duration_ms=duration_ms,
            errors=[f"HTTP {resp.status_code}: {raw_text[:200]}"],
        )

    final_response, events, parse_errors = parse_sse_stream(raw_text)

    errors: list[str] = []

    for v in validators:
        if v == "streaming_events":
            if not events:
                errors.append("No streaming events received")
        elif v == "streaming_schema":
            errors.extend(parse_errors)
            errors.extend(validate_streaming_events(events))
        elif v == "streaming_completed_status":
            if final_response is None:
                errors.append("No response.completed event found")
            else:
                errors.extend(validate_completed_status(final_response))

    return TestResult(
        test_id=test_id,
        name=name,
        passed=len(errors) == 0,
        duration_ms=duration_ms,
        errors=errors,
        response=final_response,
        stream_event_count=len(events),
    )


def _run_validators(data: dict, validators: list[str]) -> list[str]:
    """Run the named validators against a non-streaming response."""
    errors: list[str] = []
    for v in validators:
        if v == "schema":
            errors.extend(validate_response_schema(data))
        elif v == "output_items":
            errors.extend(validate_output_items(data))
        elif v == "has_output":
            errors.extend(validate_has_output(data))
        elif v == "completed_status":
            errors.extend(validate_completed_status(data))
        elif v == "has_function_call":
            errors.extend(validate_has_output_type(data, "function_call"))
    return errors


# ---------------------------------------------------------------------------
# CLI + Output
# ---------------------------------------------------------------------------

# ANSI colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
DIM = "\033[2m"
NC = "\033[0m"


def print_result(result: TestResult, verbose: bool = False) -> None:
    status = f"{GREEN}PASS{NC}" if result.passed else f"{RED}FAIL{NC}"
    print(f"  [{status}] {result.name} {DIM}({result.duration_ms:.0f}ms){NC}")

    if result.stream_event_count > 0:
        print(f"         {DIM}SSE events: {result.stream_event_count}{NC}")

    if result.errors:
        for err in result.errors:
            print(f"         {RED}- {err}{NC}")

    if not result.passed and verbose and result.response:
        print(f"         {YELLOW}Response:{NC}")
        try:
            formatted = json.dumps(result.response, indent=2)
            for line in formatted.split("\n")[:20]:
                print(f"           {line}")
        except (TypeError, ValueError):
            print(f"           {str(result.response)[:300]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenResponses Compliance Test Suite"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="Base URL for the Responses API (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model name to use (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print full response on failures",
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5, 6],
        help="Run only specific test(s) by number",
    )
    args = parser.parse_args()

    print(f"\n{CYAN}{'=' * 50}{NC}")
    print(f"{CYAN}  OpenResponses Compliance Tests{NC}")
    print(f"{CYAN}  Base URL: {args.base_url}{NC}")
    print(f"{CYAN}  Model:    {args.model}{NC}")
    print(f"{CYAN}{'=' * 50}{NC}\n")

    tests = get_test_definitions(args.model)

    # Filter tests if --test specified
    if args.test:
        tests = [t for t in tests if int(t["id"][0]) in args.test]

    results: list[TestResult] = []
    for test_def in tests:
        print(f"  {YELLOW}[{test_def['id']}] {test_def['name']}{NC}")
        if args.verbose:
            for line in test_def["description"].split("\n"):
                print(f"    {DIM}{line}{NC}")

        result = run_test(args.base_url, test_def, args.timeout)
        results.append(result)
        print_result(result, verbose=args.verbose)
        print()

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    print(f"{CYAN}{'=' * 50}{NC}")
    print(
        f"  Results: {GREEN}{passed} passed{NC}, "
        f"{RED}{failed} failed{NC} / {total} total"
    )
    print(f"{CYAN}{'=' * 50}{NC}\n")

    if failed > 0:
        print(f"{RED}Some compliance tests failed.{NC}")
        return 1
    else:
        print(f"{GREEN}All compliance tests passed.{NC}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
