#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate reasoning and tool calling against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import httpx

ThinkingExpectation = Literal["present", "absent", "either"]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You are a tool-using assistant. For weather requests, call get_weather "
    "with the exact city and unit requested. For other requests, answer directly."
)
WEATHER_PROMPT = (
    "Use get_weather to get the weather in Paris in celsius. Do not answer from memory."
)
DIRECT_PROMPT = "Do not call a tool. Reply with exactly: DIRECT-ANSWER"

REASONING_MARKERS = (
    "<think>",
    "</think>",
    "[think]",
    "[/think]",
)
TOOL_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<|tool_call|>",
    "<|tool_calls_section_begin|>",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    name: str
    messages: list[dict[str, Any]]
    tool_choice: Any | None = None
    stream: bool = False
    thinking: Literal["enabled", "disabled", "omitted"] = "omitted"
    expected_reasoning: ThinkingExpectation = "either"
    expected_tool: bool = False
    expected_content: bool = False
    structural_tag_only: bool = False
    parallel_tool_calls: bool | None = None


@dataclass
class NormalizedResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    done: bool = False


@dataclass
class CaseResult:
    case_id: str
    name: str
    passed: bool
    errors: list[str]
    latency_seconds: float
    request_file: str
    response_file: str
    normalized: dict[str, Any]


def _messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_cases(
    omitted_thinking: ThinkingExpectation,
    structural_tag_deployment: bool,
) -> list[Case]:
    """Build the fixed PR validation matrix."""
    named_choice = {"type": "function", "function": {"name": "get_weather"}}
    cases = [
        Case(
            "01",
            "tool_choice_none",
            _messages(DIRECT_PROMPT),
            tool_choice="none",
            expected_reasoning=omitted_thinking,
            expected_content=True,
        ),
        Case(
            "02",
            "tool_choice_auto_calls_tool",
            _messages(WEATHER_PROMPT),
            tool_choice="auto",
            expected_reasoning=omitted_thinking,
            expected_tool=True,
        ),
        Case(
            "03",
            "tool_choice_auto_answers_directly",
            _messages(DIRECT_PROMPT),
            tool_choice="auto",
            expected_reasoning=omitted_thinking,
            expected_content=True,
        ),
        Case(
            "04",
            "required_nonstream_thinking_enabled",
            _messages(WEATHER_PROMPT),
            tool_choice="required",
            thinking="enabled",
            expected_reasoning="present",
            expected_tool=True,
        ),
        Case(
            "05",
            "named_nonstream_thinking_enabled",
            _messages(WEATHER_PROMPT),
            tool_choice=named_choice,
            thinking="enabled",
            expected_reasoning="present",
            expected_tool=True,
        ),
        Case(
            "06",
            "required_stream_thinking_enabled",
            _messages(WEATHER_PROMPT),
            tool_choice="required",
            stream=True,
            thinking="enabled",
            expected_reasoning="present",
            expected_tool=True,
        ),
        Case(
            "07",
            "named_stream_thinking_enabled",
            _messages(WEATHER_PROMPT),
            tool_choice=named_choice,
            stream=True,
            thinking="enabled",
            expected_reasoning="present",
            expected_tool=True,
        ),
        Case(
            "08",
            "plain_reasoning_thinking_enabled",
            _messages("Is 143 a prime number? Give one short reason."),
            thinking="enabled",
            expected_reasoning="present",
            expected_content=True,
        ),
        Case(
            "09",
            "required_thinking_disabled",
            _messages(WEATHER_PROMPT),
            tool_choice="required",
            thinking="disabled",
            expected_reasoning="absent",
            expected_tool=True,
        ),
        Case(
            "10",
            "required_thinking_omitted",
            _messages(WEATHER_PROMPT),
            tool_choice="required",
            expected_reasoning=omitted_thinking,
            expected_tool=True,
        ),
        Case(
            "11",
            "consume_prior_tool_result",
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "What is the weather in Paris in celsius?",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "prior_weather_call",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps(
                                    {"city": "Paris", "unit": "celsius"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "prior_weather_call",
                    "content": json.dumps(
                        {"city": "Paris", "temperature": 21, "unit": "celsius"}
                    ),
                },
                {
                    "role": "user",
                    "content": "Use that tool result and answer in one sentence.",
                },
            ],
            tool_choice="auto",
            expected_reasoning=omitted_thinking,
            expected_content=True,
        ),
    ]
    if structural_tag_deployment:
        cases.append(
            Case(
                "12",
                "structural_tag_required_thinking_enabled",
                _messages(WEATHER_PROMPT),
                tool_choice="required",
                thinking="enabled",
                expected_reasoning="present",
                expected_tool=True,
                structural_tag_only=True,
                parallel_tool_calls=False,
            )
        )
    return cases


def build_request(
    case: Case,
    model: str,
    max_tokens: int,
    enabled_kwargs: Mapping[str, Any],
    disabled_kwargs: Mapping[str, Any],
    request_extra: Mapping[str, Any],
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": copy.deepcopy(case.messages),
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": case.stream,
    }
    if case.tool_choice is not None:
        request["tools"] = [copy.deepcopy(WEATHER_TOOL)]
        request["tool_choice"] = copy.deepcopy(case.tool_choice)
    if case.parallel_tool_calls is not None:
        request["parallel_tool_calls"] = case.parallel_tool_calls
    if case.thinking == "enabled":
        request["chat_template_kwargs"] = dict(enabled_kwargs)
    elif case.thinking == "disabled":
        request["chat_template_kwargs"] = dict(disabled_kwargs)
    request.update(copy.deepcopy(dict(request_extra)))
    return request


def _reasoning_from_message(message: Mapping[str, Any]) -> str:
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    return reasoning if isinstance(reasoning, str) else ""


def normalize_nonstream(data: Mapping[str, Any]) -> NormalizedResponse:
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("response choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("response choice has no message")
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    return NormalizedResponse(
        content=content if isinstance(content, str) else "",
        reasoning_content=_reasoning_from_message(message),
        tool_calls=tool_calls if isinstance(tool_calls, list) else [],
        finish_reason=choice.get("finish_reason"),
        done=True,
    )


def normalize_stream(raw_sse: str) -> NormalizedResponse:
    response = NormalizedResponse()
    calls: dict[int, dict[str, Any]] = {}
    for line in raw_sse.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            response.done = True
            continue
        event = json.loads(payload)
        choices = event.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                response.content += content
            response.reasoning_content += _reasoning_from_message(delta)
            if choice.get("finish_reason") is not None:
                response.finish_reason = choice["finish_reason"]
            for call_delta in delta.get("tool_calls") or []:
                index = int(call_delta.get("index", 0))
                call = calls.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if call_delta.get("id"):
                    call["id"] += call_delta["id"]
                function = call_delta.get("function") or {}
                if function.get("name"):
                    call["function"]["name"] += function["name"]
                if function.get("arguments"):
                    call["function"]["arguments"] += function["arguments"]
    response.tool_calls = [calls[index] for index in sorted(calls)]
    return response


def _tool_call_arguments(call: Mapping[str, Any]) -> tuple[str | None, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None, None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return function.get("name"), None
    return function.get("name"), arguments


def _is_complete_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        return isinstance(json.loads(stripped), (dict, list))
    except json.JSONDecodeError:
        return False


def _leak_errors(response: NormalizedResponse) -> list[str]:
    errors: list[str] = []
    for field_name, text in (
        ("content", response.content),
        ("reasoning_content", response.reasoning_content),
    ):
        lowered = text.lower()
        markers = [
            marker
            for marker in (*REASONING_MARKERS, *TOOL_MARKERS)
            if marker in lowered
        ]
        if markers:
            errors.append(f"{field_name} leaked parser marker(s): {markers}")
        if _is_complete_json(text):
            errors.append(f"{field_name} contains a raw guided JSON value")
    content = response.content.strip()
    if re.search(r'"(?:name|parameters|arguments)"\s*:', content):
        errors.append("content contains raw tool-call JSON fields")
    return errors


def validate_response(case: Case, response: NormalizedResponse) -> list[str]:
    errors = _leak_errors(response)
    reasoning = response.reasoning_content.strip()
    content = response.content.strip()

    if case.expected_reasoning == "present" and not reasoning:
        errors.append("reasoning_content is empty but reasoning was required")
    if case.expected_reasoning == "absent" and reasoning:
        errors.append("thinking-disabled request returned reasoning_content")

    if case.expected_tool:
        if response.finish_reason != "tool_calls":
            errors.append(
                f"finish_reason={response.finish_reason!r}; expected 'tool_calls'"
            )
        if len(response.tool_calls) != 1:
            errors.append(
                f"received {len(response.tool_calls)} tool calls; expected exactly one"
            )
        else:
            name, arguments = _tool_call_arguments(response.tool_calls[0])
            if name != "get_weather":
                errors.append(f"tool name={name!r}; expected 'get_weather'")
            if arguments != {"city": "Paris", "unit": "celsius"}:
                errors.append(
                    "tool arguments="
                    f"{arguments!r}; expected {{'city': 'Paris', 'unit': 'celsius'}}"
                )
        if content:
            errors.append("tool-call response contains non-whitespace content")
    else:
        if response.tool_calls:
            errors.append("request expected no tool calls")
        if response.finish_reason != "stop":
            errors.append(f"finish_reason={response.finish_reason!r}; expected 'stop'")

    if case.expected_content and not content:
        errors.append("direct-answer response content is empty")
    if case.stream and not response.done:
        errors.append("stream did not terminate with data: [DONE]")
    return errors


def _parse_json_object(value: str, option: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{option} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{option} must be a JSON object")
    return parsed


def _endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return f"{base_url}/chat/completions"


def _safe_name(case: Case) -> str:
    return f"{case.case_id}-{case.name}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _markdown_summary(
    model: str,
    endpoint: str,
    structural_tag_deployment: bool,
    results: list[CaseResult],
) -> str:
    passed = sum(result.passed for result in results)
    lines = [
        "# Guided Tool-Calling PR Validation",
        "",
        f"- Model: `{model}`",
        f"- Endpoint: `{endpoint}`",
        f"- Structural-tag deployment: `{str(structural_tag_deployment).lower()}`",
        f"- Result: **{passed}/{len(results)} passed**",
        "",
        "| Case | Result | Latency | Request | Response | Errors |",
        "|---|---:|---:|---|---|---|",
    ]
    for result in results:
        errors = "<br>".join(result.errors) if result.errors else ""
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"| `{result.case_id}` {result.name} | {status} | "
            f"{result.latency_seconds:.2f}s | "
            f"[{result.request_file}]({result.request_file}) | "
            f"[{result.response_file}]({result.response_file}) | {errors} |"
        )
    return "\n".join(lines) + "\n"


def _headers(api_key: str | None, extra_headers: list[str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for header in extra_headers:
        if ":" not in header:
            raise ValueError(f"invalid header {header!r}; expected NAME:VALUE")
        name, value = header.split(":", 1)
        headers[name.strip()] = value.strip()
    return headers


def run(args: argparse.Namespace) -> int:
    endpoint = _endpoint(args.base_url)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.omitted_thinking, args.structural_tag_deployment)
    if args.case:
        selected = set(args.case)
        known = {case.case_id for case in cases}
        unknown = selected - known
        if unknown:
            raise ValueError(f"unknown case(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.case_id in selected]

    headers = _headers(args.api_key, args.header)
    results: list[CaseResult] = []
    with httpx.Client(timeout=args.timeout) as client:
        for case in cases:
            request = build_request(
                case,
                args.model,
                args.max_tokens,
                args.thinking_enabled,
                args.thinking_disabled,
                args.request_extra,
            )
            prefix = _safe_name(case)
            request_path = output_dir / f"{prefix}.request.json"
            _write_json(request_path, request)
            started = datetime.now(timezone.utc)
            errors: list[str] = []
            normalized = NormalizedResponse()
            response_path = output_dir / f"{prefix}.response.error.txt"
            try:
                if case.stream:
                    with client.stream(
                        "POST", endpoint, headers=headers, json=request
                    ) as http_response:
                        raw_response = "\n".join(http_response.iter_lines()) + "\n"
                    response_path = output_dir / f"{prefix}.response.sse"
                    response_path.write_text(raw_response)
                    http_response.raise_for_status()
                    normalized = normalize_stream(raw_response)
                else:
                    http_response = client.post(endpoint, headers=headers, json=request)
                    response_path = output_dir / f"{prefix}.response.json"
                    try:
                        raw_json = http_response.json()
                        _write_json(response_path, raw_json)
                    except json.JSONDecodeError:
                        response_path = output_dir / f"{prefix}.response.txt"
                        response_path.write_text(http_response.text)
                        raise
                    http_response.raise_for_status()
                    normalized = normalize_nonstream(raw_json)
                errors.extend(validate_response(case, normalized))
            except Exception as exc:  # Continue the matrix and report every failure.
                error_path = output_dir / f"{prefix}.response.error.txt"
                error_path.write_text(f"{type(exc).__name__}: {exc}\n")
                if not response_path.exists():
                    response_path = error_path
                errors.append(f"request failed: {type(exc).__name__}: {exc}")
            latency = (datetime.now(timezone.utc) - started).total_seconds()
            result = CaseResult(
                case_id=case.case_id,
                name=case.name,
                passed=not errors,
                errors=errors,
                latency_seconds=latency,
                request_file=request_path.name,
                response_file=response_path.name,
                normalized=asdict(normalized),
            )
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {case.case_id} {case.name} ({latency:.2f}s)")
            for error in errors:
                print(f"       {error}")

    summary = {
        "model": args.model,
        "endpoint": endpoint,
        "structural_tag_deployment": args.structural_tag_deployment,
        "passed": sum(result.passed for result in results),
        "total": len(results),
        "results": [asdict(result) for result in results],
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        _markdown_summary(
            args.model,
            endpoint,
            args.structural_tag_deployment,
            results,
        )
    )
    print(f"\n{summary['passed']}/{summary['total']} passed; artifacts: {output_dir}")
    return 0 if summary["passed"] == summary["total"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--output-dir",
        default=(
            "pr-validator-results/"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        ),
    )
    parser.add_argument(
        "--thinking-enabled-json",
        default='{"enable_thinking": true}',
        help="chat_template_kwargs used for explicitly enabled cases",
    )
    parser.add_argument(
        "--thinking-disabled-json",
        default='{"enable_thinking": false}',
        help="chat_template_kwargs used for explicitly disabled cases",
    )
    parser.add_argument(
        "--omitted-thinking",
        choices=("present", "absent", "either"),
        default="either",
        help="expected reasoning_content when no thinking override is sent",
    )
    parser.add_argument(
        "--structural-tag-deployment",
        action="store_true",
        help=(
            "include case 12; use only against a worker launched with "
            "--dyn-enable-structural-tag"
        ),
    )
    parser.add_argument(
        "--request-extra-json",
        default="{}",
        help="JSON object merged into every request",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="additional HTTP header in NAME:VALUE form; may be repeated",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="run only this two-digit case ID; may be repeated",
    )
    args = parser.parse_args(argv)
    args.thinking_enabled = _parse_json_object(
        args.thinking_enabled_json, "--thinking-enabled-json"
    )
    args.thinking_disabled = _parse_json_object(
        args.thinking_disabled_json, "--thinking-disabled-json"
    )
    args.request_extra = _parse_json_object(
        args.request_extra_json, "--request-extra-json"
    )
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
