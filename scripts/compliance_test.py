#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
OpenResponses Compliance Test Suite
====================================
Replicates the 6 tests from https://www.openresponses.org/compliance

Validates responses against the full Zod schema used by the browser-based
compliance suite (generated from the OpenAI Responses API OpenAPI spec).
Each response field is type-checked, and each SSE streaming event is
validated against its specific event schema.

Usage:
    python scripts/compliance_test.py [--base-url URL] [--model MODEL] [--timeout SECS]

Defaults:
    --base-url  http://localhost:8000/v1
    --model     Qwen/Qwen3-0.6B
    --timeout   30
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Schema Validators
# ---------------------------------------------------------------------------
# These replicate the Zod schemas from openresponses/src/generated/kubb/zod/
# Each validator returns a list of error strings (empty = valid).


def _path_err(path: str, msg: str) -> str:
    """Format a validation error with a JSON path prefix."""
    return f"{path}: {msg}" if path else msg


def _check_type(value: Any, expected: str, path: str) -> list[str]:
    """Check that a value is of the expected type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected_type = type_map.get(expected)
    if expected_type is None:
        return [_path_err(path, f"Unknown type check: {expected}")]
    if not isinstance(value, expected_type):
        # int check should not match bool
        if expected in ("integer", "number") and isinstance(value, bool):
            return [_path_err(path, f"Expected {expected}, received boolean")]
        return [_path_err(path, f"Expected {expected}, received {type(value).__name__}")]
    if expected == "integer" and isinstance(value, float) and not value.is_integer():
        return [_path_err(path, f"Expected integer, received float")]
    return []


def _check_enum(value: Any, allowed: list, path: str) -> list[str]:
    """Check that a value is one of the allowed enum values."""
    if value not in allowed:
        return [_path_err(path, f"Expected one of {allowed}, received {value!r}")]
    return []


def _check_nullable(value: Any, type_name: str, path: str) -> list[str]:
    """Check that a value is either null or the expected type."""
    if value is None:
        return []
    return _check_type(value, type_name, path)


# ---------------------------------------------------------------------------
# ResponseResource Schema Validation
# ---------------------------------------------------------------------------
# Matches: openresponses/src/generated/kubb/zod/responseResourceSchema.ts
#
# All fields listed below are REQUIRED (not optional) in the spec.
# Fields marked "nullable" can be null but must be present.


TRUNCATION_VALUES = ["auto", "disabled"]
TOOL_CHOICE_VALUES = ["none", "auto", "required"]
STATUS_VALUES = ["completed", "failed", "in_progress", "incomplete", "queued"]


def validate_response_resource(data: Any, path: str = "") -> list[str]:
    """Full schema validation of a ResponseResource object.

    Validates all 31 required fields with correct types, matching the
    Zod schema from the OpenResponses compliance suite.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return [_path_err(path, f"Expected object, received {type(data).__name__}")]

    # --- Required string fields ---
    for field_name in ("id", "object", "status", "model", "service_tier"):
        val = data.get(field_name)
        if val is None:
            errors.append(_path_err(f"{path}.{field_name}" if path else field_name, "Required"))
        else:
            errors.extend(_check_type(val, "string", f"{path}.{field_name}" if path else field_name))

    # object must be "response"
    if data.get("object") is not None:
        errors.extend(_check_enum(data.get("object"), ["response"], f"{path}.object" if path else "object"))

    # status enum
    if isinstance(data.get("status"), str):
        errors.extend(_check_enum(data["status"], STATUS_VALUES, f"{path}.status" if path else "status"))

    # --- Required integer fields ---
    for field_name in ("created_at", "top_logprobs"):
        val = data.get(field_name)
        fpath = f"{path}.{field_name}" if path else field_name
        if val is None:
            errors.append(_path_err(fpath, "Required"))
        else:
            errors.extend(_check_type(val, "integer", fpath))

    # --- Required number fields ---
    for field_name in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        val = data.get(field_name)
        fpath = f"{path}.{field_name}" if path else field_name
        if val is None:
            errors.append(_path_err(fpath, "Required"))
        else:
            errors.extend(_check_type(val, "number", fpath))

    # --- Required boolean fields ---
    for field_name in ("parallel_tool_calls", "store", "background"):
        val = data.get(field_name)
        fpath = f"{path}.{field_name}" if path else field_name
        if val is None:
            errors.append(_path_err(fpath, "Required"))
        else:
            errors.extend(_check_type(val, "boolean", fpath))

    # --- Required array fields ---
    # tools: array of tool objects
    tools = data.get("tools")
    fpath = f"{path}.tools" if path else "tools"
    if tools is None:
        errors.append(_path_err(fpath, "Required"))
    else:
        errors.extend(_check_type(tools, "array", fpath))

    # output: array of output items
    output = data.get("output")
    fpath = f"{path}.output" if path else "output"
    if output is None:
        errors.append(_path_err(fpath, "Required"))
    elif not isinstance(output, list):
        errors.append(_path_err(fpath, f"Expected array, received {type(output).__name__}"))
    else:
        for i, item in enumerate(output):
            errors.extend(validate_output_item(item, f"{fpath}[{i}]"))

    # --- Required enum fields ---
    # truncation: "auto" | "disabled"
    trunc = data.get("truncation")
    fpath = f"{path}.truncation" if path else "truncation"
    if trunc is None:
        errors.append(_path_err(fpath, "Required"))
    else:
        errors.extend(_check_enum(trunc, TRUNCATION_VALUES, fpath))

    # tool_choice: string enum or object
    tc = data.get("tool_choice")
    fpath = f"{path}.tool_choice" if path else "tool_choice"
    if tc is None:
        errors.append(_path_err(fpath, "Required"))
    elif isinstance(tc, str):
        errors.extend(_check_enum(tc, TOOL_CHOICE_VALUES, fpath))
    elif not isinstance(tc, dict):
        errors.append(_path_err(fpath, f"Expected string or object, received {type(tc).__name__}"))

    # text: object with format field
    text = data.get("text")
    fpath = f"{path}.text" if path else "text"
    if text is None:
        errors.append(_path_err(fpath, "Required"))
    elif not isinstance(text, dict):
        errors.append(_path_err(fpath, f"Expected object, received {type(text).__name__}"))
    else:
        fmt = text.get("format")
        if fmt is None:
            errors.append(_path_err(f"{fpath}.format", "Required"))
        elif not isinstance(fmt, dict):
            errors.append(_path_err(f"{fpath}.format", f"Expected object, received {type(fmt).__name__}"))
        elif "type" not in fmt:
            errors.append(_path_err(f"{fpath}.format.type", "Required"))

    # metadata: any (required but can be any type including null)
    if "metadata" not in data:
        errors.append(_path_err(f"{path}.metadata" if path else "metadata", "Required"))

    # --- Nullable fields (must be present, can be null) ---
    nullable_int_fields = ["completed_at", "max_output_tokens", "max_tool_calls"]
    for field_name in nullable_int_fields:
        fpath = f"{path}.{field_name}" if path else field_name
        if field_name not in data:
            errors.append(_path_err(fpath, "Required"))
        else:
            errors.extend(_check_nullable(data[field_name], "integer", fpath))

    nullable_string_fields = [
        "previous_response_id", "instructions", "safety_identifier", "prompt_cache_key",
    ]
    for field_name in nullable_string_fields:
        fpath = f"{path}.{field_name}" if path else field_name
        if field_name not in data:
            errors.append(_path_err(fpath, "Required"))
        else:
            errors.extend(_check_nullable(data[field_name], "string", fpath))

    # error: null or object
    fpath = f"{path}.error" if path else "error"
    if "error" not in data:
        errors.append(_path_err(fpath, "Required"))

    # incomplete_details: null or object
    fpath = f"{path}.incomplete_details" if path else "incomplete_details"
    if "incomplete_details" not in data:
        errors.append(_path_err(fpath, "Required"))

    # reasoning: null or object
    fpath = f"{path}.reasoning" if path else "reasoning"
    if "reasoning" not in data:
        errors.append(_path_err(fpath, "Required"))

    # usage: null or object
    fpath = f"{path}.usage" if path else "usage"
    if "usage" not in data:
        errors.append(_path_err(fpath, "Required"))

    return errors


# ---------------------------------------------------------------------------
# Output Item Schema Validation
# ---------------------------------------------------------------------------
# Matches: openresponses/src/generated/kubb/zod/itemFieldSchema.ts
# Union of: messageSchema | functionCallSchema | functionCallOutputSchema | reasoningBodySchema


def validate_output_item(item: Any, path: str = "output") -> list[str]:
    """Validate a single output item (message, function_call, etc.)."""
    errors: list[str] = []

    if not isinstance(item, dict):
        return [_path_err(path, f"Expected object, received {type(item).__name__}")]

    item_type = item.get("type")
    if item_type is None:
        return [_path_err(f"{path}.type", "Required")]

    if item_type == "message":
        errors.extend(_validate_message_item(item, path))
    elif item_type == "function_call":
        errors.extend(_validate_function_call_item(item, path))
    elif item_type == "function_call_output":
        # Minimal validation for function_call_output
        if "call_id" not in item:
            errors.append(_path_err(f"{path}.call_id", "Required"))
    elif item_type == "reasoning":
        pass  # Minimal validation for reasoning
    else:
        errors.append(_path_err(f"{path}.type", f"Invalid discriminator value: {item_type!r}"))

    return errors


def _validate_message_item(item: dict, path: str) -> list[str]:
    """Validate a message output item.

    Matches: openresponses/src/generated/kubb/zod/messageSchema.ts
    Required fields: type, id, status, role, content
    """
    errors: list[str] = []

    # id: string
    if "id" not in item or not isinstance(item["id"], str):
        errors.append(_path_err(f"{path}.id", "Required string"))

    # status: string
    if "status" not in item or not isinstance(item["status"], str):
        errors.append(_path_err(f"{path}.status", "Required string"))

    # role: string
    if "role" not in item or not isinstance(item["role"], str):
        errors.append(_path_err(f"{path}.role", "Required string"))

    # content: array of content parts
    content = item.get("content")
    if content is None or not isinstance(content, list):
        errors.append(_path_err(f"{path}.content", "Required array"))
    else:
        for i, part in enumerate(content):
            errors.extend(_validate_content_part(part, f"{path}.content[{i}]"))

    return errors


def _validate_content_part(part: Any, path: str) -> list[str]:
    """Validate a content part within a message.

    Matches the union of content schemas. For output_text specifically:
    - type: "output_text"
    - text: string
    - annotations: array
    - logprobs: array (NOT nullable -- must be [] if empty)
    """
    errors: list[str] = []

    if not isinstance(part, dict):
        return [_path_err(path, f"Expected object, received {type(part).__name__}")]

    part_type = part.get("type")
    if part_type is None:
        return [_path_err(f"{path}.type", "Required")]

    if part_type == "output_text":
        # text: string
        if "text" not in part or not isinstance(part["text"], str):
            errors.append(_path_err(f"{path}.text", "Required string"))

        # annotations: array
        ann = part.get("annotations")
        if ann is None or not isinstance(ann, list):
            errors.append(_path_err(f"{path}.annotations", "Required array"))

        # logprobs: array (NOT nullable per spec)
        lp = part.get("logprobs")
        if lp is None:
            errors.append(_path_err(f"{path}.logprobs", "Expected array, received null"))
        elif not isinstance(lp, list):
            errors.append(_path_err(f"{path}.logprobs", f"Expected array, received {type(lp).__name__}"))

    # Other content types (input_text, refusal, etc.) are less common in output
    return errors


def _validate_function_call_item(item: dict, path: str) -> list[str]:
    """Validate a function_call output item.

    Matches: openresponses/src/generated/kubb/zod/functionCallSchema.ts
    Required fields: type, id, call_id, name, arguments, status
    """
    errors: list[str] = []

    for field_name in ("id", "call_id", "name", "arguments"):
        fpath = f"{path}.{field_name}"
        if field_name not in item or not isinstance(item[field_name], str):
            errors.append(_path_err(fpath, "Required string"))

    # status: required (functionCallStatusSchema)
    if "status" not in item:
        errors.append(_path_err(f"{path}.status", "Required"))

    # name must match pattern ^[a-zA-Z0-9_-]+$
    name = item.get("name")
    if isinstance(name, str) and not re.match(r'^[a-zA-Z0-9_-]+$', name):
        errors.append(_path_err(f"{path}.name", f"Invalid function name: {name!r}"))

    return errors


# ---------------------------------------------------------------------------
# SSE Streaming Event Schema Validation
# ---------------------------------------------------------------------------
# Matches: openresponses/src/lib/sse-parser.ts streamingEventSchema
# Each event is validated against its specific schema.


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
    "response.output_text_annotation.added",
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
    "error",
}

REQUIRED_LIFECYCLE_EVENTS = {
    "response.created",
    "response.in_progress",
    "response.completed",
}


def validate_sse_event(event: dict) -> list[str]:
    """Validate a single SSE event against its specific schema.

    Each event type has required fields defined in the corresponding
    Zod schema from openresponses/src/generated/kubb/zod/.
    """
    errors: list[str] = []
    evt_type = event.get("type")

    if evt_type is None:
        return ["Event missing 'type' field"]
    if evt_type not in VALID_SSE_EVENT_TYPES:
        return [f"Unknown event type: {evt_type!r}"]

    # All events require sequence_number (integer)
    if "sequence_number" not in event:
        errors.append(f"{evt_type}: missing 'sequence_number'")
    elif not isinstance(event["sequence_number"], int) or isinstance(event["sequence_number"], bool):
        errors.append(f"{evt_type}: 'sequence_number' must be integer")

    # Event-specific validation
    if evt_type in ("response.created", "response.in_progress",
                     "response.completed", "response.failed", "response.incomplete"):
        errors.extend(_validate_response_lifecycle_event(event, evt_type))
    elif evt_type == "response.output_item.added":
        errors.extend(_validate_output_item_event(event, evt_type))
    elif evt_type == "response.output_item.done":
        errors.extend(_validate_output_item_event(event, evt_type))
    elif evt_type == "response.content_part.added":
        errors.extend(_validate_content_part_event(event, evt_type))
    elif evt_type == "response.content_part.done":
        errors.extend(_validate_content_part_event(event, evt_type))
    elif evt_type == "response.output_text.delta":
        errors.extend(_validate_text_delta_event(event))
    elif evt_type == "response.output_text.done":
        errors.extend(_validate_text_done_event(event))
    elif evt_type == "response.function_call_arguments.delta":
        errors.extend(_validate_fc_args_delta_event(event))
    elif evt_type == "response.function_call_arguments.done":
        errors.extend(_validate_fc_args_done_event(event))

    return errors


def _validate_response_lifecycle_event(event: dict, evt_type: str) -> list[str]:
    """Validate response.created/in_progress/completed/failed/incomplete events.

    Required: sequence_number, response (full ResponseResource)
    """
    errors: list[str] = []
    resp = event.get("response")
    if resp is None:
        errors.append(f"{evt_type}: missing 'response' field")
    elif isinstance(resp, dict):
        sub_errors = validate_response_resource(resp, f"{evt_type}.response")
        errors.extend(sub_errors)
    else:
        errors.append(f"{evt_type}: 'response' must be object")
    return errors


def _validate_output_item_event(event: dict, evt_type: str) -> list[str]:
    """Validate response.output_item.added/done events.

    Required: sequence_number, output_index (integer), item (itemFieldSchema | null)
    """
    errors: list[str] = []
    if "output_index" not in event:
        errors.append(f"{evt_type}: missing 'output_index'")
    elif not isinstance(event["output_index"], int) or isinstance(event["output_index"], bool):
        errors.append(f"{evt_type}: 'output_index' must be integer")

    item = event.get("item")
    if item is not None and isinstance(item, dict):
        errors.extend(validate_output_item(item, f"{evt_type}.item"))
    # item can be null per spec
    return errors


def _validate_content_part_event(event: dict, evt_type: str) -> list[str]:
    """Validate response.content_part.added/done events.

    Required: sequence_number, item_id (string), output_index (integer),
              content_index (integer), part (content union)
    """
    errors: list[str] = []
    for field_name in ("item_id",):
        if field_name not in event or not isinstance(event[field_name], str):
            errors.append(f"{evt_type}: missing or invalid '{field_name}' (expected string)")
    for field_name in ("output_index", "content_index"):
        if field_name not in event:
            errors.append(f"{evt_type}: missing '{field_name}'")
        elif not isinstance(event[field_name], int) or isinstance(event[field_name], bool):
            errors.append(f"{evt_type}: '{field_name}' must be integer")

    part = event.get("part")
    if part is None:
        errors.append(f"{evt_type}: missing 'part'")
    elif isinstance(part, dict):
        errors.extend(_validate_content_part(part, f"{evt_type}.part"))
    return errors


def _validate_text_delta_event(event: dict) -> list[str]:
    """Validate response.output_text.delta event.

    Required: sequence_number, item_id (string), output_index (integer),
              content_index (integer), delta (string), logprobs (array)
    """
    evt_type = "response.output_text.delta"
    errors: list[str] = []
    if "item_id" not in event or not isinstance(event["item_id"], str):
        errors.append(f"{evt_type}: missing or invalid 'item_id' (expected string)")
    for field_name in ("output_index", "content_index"):
        if field_name not in event:
            errors.append(f"{evt_type}: missing '{field_name}'")
        elif not isinstance(event[field_name], int) or isinstance(event[field_name], bool):
            errors.append(f"{evt_type}: '{field_name}' must be integer")
    if "delta" not in event or not isinstance(event["delta"], str):
        errors.append(f"{evt_type}: missing or invalid 'delta' (expected string)")
    # logprobs: required array (not nullable)
    lp = event.get("logprobs")
    if lp is None:
        errors.append(f"{evt_type}: missing or null 'logprobs' (expected array)")
    elif not isinstance(lp, list):
        errors.append(f"{evt_type}: 'logprobs' must be array, got {type(lp).__name__}")
    return errors


def _validate_text_done_event(event: dict) -> list[str]:
    """Validate response.output_text.done event.

    Required: sequence_number, item_id (string), output_index (integer),
              content_index (integer), text (string), logprobs (array)
    """
    evt_type = "response.output_text.done"
    errors: list[str] = []
    if "item_id" not in event or not isinstance(event["item_id"], str):
        errors.append(f"{evt_type}: missing or invalid 'item_id' (expected string)")
    for field_name in ("output_index", "content_index"):
        if field_name not in event:
            errors.append(f"{evt_type}: missing '{field_name}'")
        elif not isinstance(event[field_name], int) or isinstance(event[field_name], bool):
            errors.append(f"{evt_type}: '{field_name}' must be integer")
    if "text" not in event or not isinstance(event["text"], str):
        errors.append(f"{evt_type}: missing or invalid 'text' (expected string)")
    # logprobs: required array
    lp = event.get("logprobs")
    if lp is None:
        errors.append(f"{evt_type}: missing or null 'logprobs' (expected array)")
    elif not isinstance(lp, list):
        errors.append(f"{evt_type}: 'logprobs' must be array, got {type(lp).__name__}")
    return errors


def _validate_fc_args_delta_event(event: dict) -> list[str]:
    """Validate response.function_call_arguments.delta event.

    Required: sequence_number, item_id (string), output_index (integer), delta (string)
    """
    evt_type = "response.function_call_arguments.delta"
    errors: list[str] = []
    if "item_id" not in event or not isinstance(event["item_id"], str):
        errors.append(f"{evt_type}: missing or invalid 'item_id'")
    if "output_index" not in event:
        errors.append(f"{evt_type}: missing 'output_index'")
    if "delta" not in event or not isinstance(event["delta"], str):
        errors.append(f"{evt_type}: missing or invalid 'delta'")
    return errors


def _validate_fc_args_done_event(event: dict) -> list[str]:
    """Validate response.function_call_arguments.done event.

    Required: sequence_number, item_id (string), output_index (integer), arguments (string)
    """
    evt_type = "response.function_call_arguments.done"
    errors: list[str] = []
    if "item_id" not in event or not isinstance(event["item_id"], str):
        errors.append(f"{evt_type}: missing or invalid 'item_id'")
    if "output_index" not in event:
        errors.append(f"{evt_type}: missing 'output_index'")
    if "arguments" not in event or not isinstance(event["arguments"], str):
        errors.append(f"{evt_type}: missing or invalid 'arguments'")
    return errors


# ---------------------------------------------------------------------------
# SSE Stream Parsing
# ---------------------------------------------------------------------------


def parse_sse_stream(raw_text: str) -> tuple[dict | None, list[dict], list[str]]:
    """Parse SSE text into (final_response, events, errors).

    SSE format:
        event: response.created
        data: {"type":"response.created","sequence_number":0,...}

        ...
        data: [DONE]

    Returns:
        final_response: The response object from the response.completed event
        events: List of all parsed event dicts
        errors: List of parsing/validation errors
    """
    events: list[dict] = []
    errors: list[str] = []
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

                # Validate each event against its specific schema
                evt_errors = validate_sse_event(parsed)
                errors.extend(evt_errors)

                if current_event in ("response.completed", "response.failed"):
                    if "response" in parsed:
                        final_response = parsed["response"]
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in SSE data: {e}")

            current_event = ""
            current_data = ""

    return final_response, events, errors


def validate_streaming_lifecycle(events: list[dict]) -> list[str]:
    """Validate that required lifecycle events are present and in order."""
    errors: list[str] = []
    event_types = [e.get("type") for e in events]
    event_types_set = set(event_types)

    for required in REQUIRED_LIFECYCLE_EVENTS:
        if required not in event_types_set:
            errors.append(f"Missing required SSE event: {required}")

    # Validate sequence numbers are monotonically increasing
    seq_numbers = [e.get("sequence_number") for e in events if "sequence_number" in e]
    for i in range(1, len(seq_numbers)):
        if seq_numbers[i] <= seq_numbers[i - 1]:
            errors.append(
                f"Sequence numbers not monotonically increasing: "
                f"event {i-1} has seq={seq_numbers[i-1]}, event {i} has seq={seq_numbers[i]}"
            )
            break

    return errors


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
# Base64 encoded 32x32 red heart PNG (same image used by OpenResponses)
# ---------------------------------------------------------------------------

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
            "validators": ["response_resource", "has_output", "completed_status"],
        },
        {
            "id": "2-streaming-response",
            "name": "Streaming Response",
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
                "streaming_lifecycle",
                "streaming_final_response",
            ],
        },
        {
            "id": "3-system-prompt",
            "name": "System Prompt",
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
            "validators": ["response_resource", "has_output", "completed_status"],
        },
        {
            "id": "4-tool-calling",
            "name": "Tool Calling",
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
                "response_resource",
                "has_output",
                "has_function_call",
            ],
        },
        {
            "id": "5-image-input",
            "name": "Image Input",
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
                            },
                        ],
                    }
                ],
            },
            "validators": ["response_resource", "has_output", "completed_status"],
        },
        {
            "id": "6-multi-turn",
            "name": "Multi-turn Conversation",
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
            "validators": ["response_resource", "has_output", "completed_status"],
        },
    ]


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------


def run_test(base_url: str, test_def: dict, timeout: int) -> TestResult:
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
        elif v == "streaming_lifecycle":
            # Include per-event schema validation errors from parsing
            errors.extend(parse_errors)
            # Check lifecycle events present and sequence numbers
            errors.extend(validate_streaming_lifecycle(events))
        elif v == "streaming_final_response":
            if final_response is None:
                errors.append("No response.completed event found")
            else:
                # Validate the final response against full ResponseResource schema
                errors.extend(validate_response_resource(final_response))
                errors.extend(_validate_completed_status(final_response))

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
        if v == "response_resource":
            errors.extend(validate_response_resource(data))
        elif v == "has_output":
            errors.extend(_validate_has_output(data))
        elif v == "completed_status":
            errors.extend(_validate_completed_status(data))
        elif v == "has_function_call":
            errors.extend(_validate_has_output_type(data, "function_call"))
    return errors


def _validate_has_output(data: dict) -> list[str]:
    """Check that at least one output item exists."""
    if not data.get("output"):
        return ["Response has no output items"]
    return []


def _validate_completed_status(data: dict) -> list[str]:
    """Check that status is 'completed'."""
    status = data.get("status")
    if status != "completed":
        return [f"Expected status='completed', got '{status}'"]
    return []


def _validate_has_output_type(data: dict, output_type: str) -> list[str]:
    """Check that at least one output item has the given type."""
    items = data.get("output", [])
    has_type = any(item.get("type") == output_type for item in items)
    if not has_type:
        types_found = [item.get("type", "unknown") for item in items]
        return [
            f"Expected output item of type '{output_type}' but found: {types_found}"
        ]
    return []


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
        for err in result.errors[:10]:  # Cap at 10 errors per test
            print(f"         {RED}- {err}{NC}")
        if len(result.errors) > 10:
            print(f"         {RED}  ... and {len(result.errors) - 10} more errors{NC}")

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
