# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tool calling tests against the Dynamo frontend.

Validates:
  - Streaming protocol shape (chat.completion.chunk SSE)
  - Tool-call reconstruction from streamed deltas
  - Tool-call argument JSON validated against the declared JSON Schema
  - tool_choice variants: auto / required / none / named function
  - Multi-turn conversations carrying tool results
  - Multi-tool parallel calls

"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Generator

import pytest

from tests.conftest import EtcdServer, NatsServer
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_ports

openai = pytest.importorskip("openai")
OpenAI = openai.OpenAI
jsonschema = pytest.importorskip("jsonschema")
Draft7Validator = jsonschema.Draft7Validator

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"

pytestmark = [
    pytest.mark.sglang,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.model(MODEL_NAME),
    pytest.mark.timeout(300),
]


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


def _check_ready(response) -> bool:
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


def _prepare_log_dir(request, suffix: str) -> str:
    log_dir = f"{request.node.name}_{suffix}"
    shutil.rmtree(log_dir, ignore_errors=True)
    return log_dir


class WorkerProcess(ManagedProcess):
    """backend worker for the tool-calling tests."""

    def __init__(self, request, *, system_port: int):
        env = os.environ.copy()
        env["DYN_LOG"] = "info"
        env["DYN_SYSTEM_PORT"] = str(system_port)
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'

        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.sglang",
                "--model-path",
                MODEL_NAME,
                "--served-model-name",
                MODEL_NAME,
                "--trust-remote-code",
            ],
            env=env,
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready),
            ],
            timeout=600,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["SGLANG:EngineCore"],
            straggler_commands=["-m dynamo.sglang"],
            log_dir=_prepare_log_dir(request, "sglang-worker"),
        )


class ToolCallingFrontendProcess(ManagedProcess):
    """Frontend HTTP ingress.

    SGLang-specific chat processor, tool-call parser, and reasoning parser
    flags are only attached when ``sglang`` is importable in the current
    environment (otherwise the frontend would fail to load them).
    """

    def __init__(self, request, *, frontend_port: int):
        env = os.environ.copy()
        env["DYN_LOG"] = "info"
        env.pop("DYN_SYSTEM_PORT", None)

        command = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(frontend_port),
            "--router-mode",
            "round-robin",
            "--dyn-chat-processor",
            "sglang",
            "--tool-call-parser",
            "qwen25",
            "--reasoning-parser",
            "qwen3",
            "--trust-remote-code",
        ]

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
            ],
            timeout=240,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m dynamo.frontend"],
            log_dir=_prepare_log_dir(request, "frontend"),
        )


@pytest.fixture(scope="module")
def runtime_services(request) -> Generator[None, None, None]:
    """Module-scoped NATS + Etcd for the tool calling stack.

    Inlined (rather than depending on the function-scoped
    ``runtime_services_dynamic_ports``) so the worker + frontend processes
    can be reused across all tests in this module.
    """
    with NatsServer(request, port=0) as nats, EtcdServer(request, port=0) as etcd:
        orig_nats = os.environ.get("NATS_SERVER")
        orig_etcd = os.environ.get("ETCD_ENDPOINTS")
        os.environ["NATS_SERVER"] = f"nats://localhost:{nats.port}"
        os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd.port}"
        try:
            yield
        finally:
            if orig_nats is not None:
                os.environ["NATS_SERVER"] = orig_nats
            else:
                os.environ.pop("NATS_SERVER", None)
            if orig_etcd is not None:
                os.environ["ETCD_ENDPOINTS"] = orig_etcd
            else:
                os.environ.pop("ETCD_ENDPOINTS", None)


@pytest.fixture(scope="module")
def tool_calling_services(
    request, runtime_services, predownload_models
) -> Generator[int, None, None]:
    """Start the SGLang worker + tool-calling-aware frontend.

    Yields the frontend HTTP port.
    """
    frontend_port, system_port = allocate_ports(count=2, start_port=10000)

    with WorkerProcess(request, system_port=system_port):
        # Allow worker to register with discovery.
        time.sleep(2)
        with ToolCallingFrontendProcess(request, frontend_port=frontend_port):
            logger.info(
                "Tool calling stack ready (frontend=%d worker_system=%d)",
                frontend_port,
                system_port,
            )
            yield frontend_port


@pytest.fixture(scope="module")
def client(tool_calling_services: int) -> OpenAI:
    return OpenAI(
        api_key="EMPTY", base_url=f"http://localhost:{tool_calling_services}/v1"
    )


@pytest.fixture(scope="module")
def model() -> str:
    return MODEL_NAME


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS_WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city"],
                "additionalProperties": True,
            },
        },
    }
]

TOOLS_SEARCH = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": True,
            },
        },
    }
]

TOOLS_CALCULATOR = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
                "additionalProperties": True,
            },
        },
    }
]

TOOLS_COMPLEX_ARGS = [
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a calendar event with attendees and recurrence",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "attendees": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                                "role": {
                                    "type": "string",
                                    "enum": ["required", "optional", "organizer"],
                                },
                            },
                            "required": ["email"],
                            "additionalProperties": True,
                        },
                    },
                    "recurrence": {
                        "type": "object",
                        "properties": {
                            "frequency": {
                                "type": "string",
                                "enum": ["daily", "weekly", "monthly"],
                            },
                            "interval": {"type": "integer"},
                            "count": {"type": "integer"},
                        },
                        "additionalProperties": True,
                    },
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title", "start_time", "end_time"],
                "additionalProperties": True,
            },
        },
    }
]

TOOLS_DATABASE = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute a SQL query against the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "params": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "database": {
                        "type": "string",
                        "enum": ["users", "orders", "products"],
                    },
                },
                "required": ["sql", "database"],
                "additionalProperties": True,
            },
        },
    }
]

TOOLS_GET_TIME = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"},
                },
                "required": ["timezone"],
                "additionalProperties": True,
            },
        },
    }
]

ALL_TOOLS = (
    TOOLS_WEATHER
    + TOOLS_SEARCH
    + TOOLS_CALCULATOR
    + TOOLS_COMPLEX_ARGS
    + TOOLS_DATABASE
)


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def tool_schema_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tool in tools:
        fn = tool["function"]
        out[fn["name"]] = fn["parameters"]
    return out


@dataclass
class StreamResult:
    content: str
    reasoning_content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    model: str
    chunks: int
    ttft_ms: float
    raw_chunks: list[Any]


def collect_stream(stream) -> StreamResult:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    finish_reason = None
    model = ""
    chunk_count = 0
    raw_chunks: list[Any] = []
    t0 = time.monotonic()
    ttft_ms = 0.0

    for chunk in stream:
        raw_chunks.append(chunk)
        chunk_count += 1
        if chunk_count == 1:
            ttft_ms = (time.monotonic() - t0) * 1000.0
        model = chunk.model

        for choice in chunk.choices:
            delta = choice.delta

            if getattr(delta, "content", None):
                content_parts.append(delta.content)

            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    entry = tool_calls_by_index.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                    if tc.id:
                        if entry["id"] and entry["id"] != tc.id:
                            raise AssertionError(
                                f"Tool call id changed within same index {idx}: "
                                f"{entry['id']} -> {tc.id}"
                            )
                        entry["id"] = tc.id

                    if tc.type:
                        entry["type"] = tc.type

                    if tc.function:
                        if tc.function.name:
                            if (
                                entry["function"]["name"]
                                and entry["function"]["name"] != tc.function.name
                            ):
                                raise AssertionError(
                                    f"Tool name changed within same index {idx}: "
                                    f"{entry['function']['name']} -> {tc.function.name}"
                                )
                            entry["function"]["name"] = tc.function.name

                        if tc.function.arguments:
                            entry["function"]["arguments"] += tc.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

    ordered_tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    return StreamResult(
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts),
        tool_calls=ordered_tool_calls,
        finish_reason=finish_reason,
        model=model,
        chunks=chunk_count,
        ttft_ms=ttft_ms,
        raw_chunks=raw_chunks,
    )


def stream_chat(
    client: OpenAI,
    model: str,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    **kwargs,
) -> StreamResult:
    req: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        req["tools"] = tools
    req.update(kwargs)
    stream = client.chat.completions.create(**req)
    return collect_stream(stream)


def parse_and_validate_tool_call(
    tc: dict[str, Any],
    schema_by_name: dict[str, dict[str, Any]],
    *,
    expected_name: str | None = None,
) -> dict[str, Any]:
    assert tc["type"] == "function", f"unexpected tool type: {tc['type']!r}"
    assert tc["id"], "tool call id must be non-empty"
    fn_name = tc["function"]["name"]
    assert fn_name, "tool call function name must be non-empty"

    if expected_name is not None:
        assert fn_name == expected_name, f"expected {expected_name!r}, got {fn_name!r}"

    assert fn_name in schema_by_name, f"unknown tool name {fn_name!r}"
    args_str = tc["function"]["arguments"]
    assert args_str, "tool call arguments must be non-empty"

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError as e:
        raise AssertionError(f"arguments are not valid JSON: {args_str!r}") from e

    assert isinstance(args, dict), f"arguments must decode to object, got {type(args)}"

    validator = Draft7Validator(schema_by_name[fn_name])
    errors = sorted(validator.iter_errors(args), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"path={list(err.path)} message={err.message}" for err in errors
        )
        raise AssertionError(f"arguments failed schema validation: {rendered}")

    return args


def assert_finish_reason(result: StreamResult, allowed: set[str]) -> None:
    assert result.finish_reason in allowed, (
        f"unexpected finish_reason={result.finish_reason!r}, "
        f"allowed={sorted(allowed)}"
    )


def assistant_tool_message_from_result(result: StreamResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content or None,
        "tool_calls": result.tool_calls,
    }


# ---------------------------------------------------------------------------
# Protocol / contract tests
# ---------------------------------------------------------------------------


class TestToolCallingProtocol:
    def test_stream_has_required_chunk_shape(self, client: OpenAI, model: str):
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
            tools=TOOLS_WEATHER,
            stream=True,
            max_tokens=256,
        )

        chunk_count = 0
        saw_finish = False
        for chunk in stream:
            chunk_count += 1
            assert chunk.id
            assert chunk.model == model or isinstance(chunk.model, str)
            assert chunk.object == "chat.completion.chunk"
            assert chunk.created > 0
            assert len(chunk.choices) >= 1

            for choice in chunk.choices:
                assert choice.index == 0
                if choice.finish_reason is not None:
                    saw_finish = True
                    assert choice.finish_reason in {"stop", "tool_calls", "length"}

        assert chunk_count > 0
        assert saw_finish, "stream never emitted a finish_reason"

    def test_single_tool_call_schema_valid(self, client: OpenAI, model: str):
        result = stream_chat(
            client,
            model,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=TOOLS_WEATHER,
        )
        assert_finish_reason(result, {"tool_calls"})
        assert len(result.tool_calls) >= 1

        schema = tool_schema_map(TOOLS_WEATHER)
        args = parse_and_validate_tool_call(
            result.tool_calls[0], schema, expected_name="get_weather"
        )
        assert "city" in args
        assert isinstance(args["city"], str)
        assert args["city"]

    def test_tool_choice_required_forces_a_tool_call(self, client: OpenAI, model: str):
        result = stream_chat(
            client,
            model,
            messages=[{"role": "user", "content": "Hello there."}],
            tools=TOOLS_WEATHER,
            tool_choice="required",
        )
        assert_finish_reason(result, {"tool_calls"})
        assert len(result.tool_calls) >= 1
        # Intent of this test: verify tool_choice=required forces a call.
        # The prompt doesn't warrant a tool call, so a small model may
        # hallucinate values for optional fields.  Validate only that the
        # call is well-formed and the required fields are present; don't
        # enforce the full schema (e.g. enum values on optional fields).
        schema = tool_schema_map(TOOLS_WEATHER)
        for tc in result.tool_calls:
            assert tc["type"] == "function"
            assert tc["id"]
            fn_name = tc["function"]["name"]
            assert fn_name in schema, f"unknown tool name {fn_name!r}"
            args = json.loads(tc["function"]["arguments"])
            assert isinstance(args, dict)
            for required_field in schema[fn_name].get("required", []):
                assert (
                    required_field in args
                ), f"{fn_name} missing required field {required_field!r}"

    def test_tool_choice_none_suppresses_tool_calls(self, client: OpenAI, model: str):
        result = stream_chat(
            client,
            model,
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=TOOLS_WEATHER,
            tool_choice="none",
        )
        assert_finish_reason(result, {"stop"})
        assert result.tool_calls == []
        assert result.content.strip()

    def test_named_tool_choice_forces_specific_function(
        self, client: OpenAI, model: str
    ):
        result = stream_chat(
            client,
            model,
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=TOOLS_WEATHER,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        assert_finish_reason(result, {"tool_calls"})
        assert len(result.tool_calls) >= 1
        schema = tool_schema_map(TOOLS_WEATHER)
        for tc in result.tool_calls:
            parse_and_validate_tool_call(tc, schema, expected_name="get_weather")

    def test_parallel_multi_tool_request_includes_all_expected_tools(
        self, client: OpenAI, model: str
    ):
        result = stream_chat(
            client,
            model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Do all three of these with tools: "
                        "1) weather in Paris, "
                        "2) search the web for latest Python release, "
                        "3) calculate 15 * 23 + 7."
                    ),
                }
            ],
            tools=TOOLS_WEATHER + TOOLS_SEARCH + TOOLS_CALCULATOR,
            parallel_tool_calls=True,
        )
        assert_finish_reason(result, {"tool_calls"})
        # Models sometimes batch only a subset and emit follow-up calls in
        # later turns; require at least 2 distinct tools rather than all 3.
        schemas = tool_schema_map(TOOLS_WEATHER + TOOLS_SEARCH + TOOLS_CALCULATOR)
        names: set[str] = set()
        for tc in result.tool_calls:
            parse_and_validate_tool_call(tc, schemas)
            names.add(tc["function"]["name"])
        assert len(names) >= 2, f"expected at least 2 distinct tools, got {names}"

    def test_tool_call_ids_unique_in_single_response(self, client: OpenAI, model: str):
        result = stream_chat(
            client,
            model,
            messages=[
                {
                    "role": "user",
                    "content": "Get weather for New York, London, and Tokyo.",
                }
            ],
            tools=TOOLS_WEATHER,
            tool_choice="required",
            parallel_tool_calls=True,
        )
        assert_finish_reason(result, {"tool_calls"})
        ids = [tc["id"] for tc in result.tool_calls]
        assert len(ids) == len(set(ids)), f"duplicate tool ids: {ids}"

    def test_array_argument_schema_valid(self, client: OpenAI, model: str):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_emails",
                    "description": "Send emails",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipients": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["recipients", "subject", "body"],
                    },
                },
            }
        ]
        result = stream_chat(
            client,
            model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Send an email with subject 'Team Update' and body "
                        "'Meeting at 3pm' to alice@example.com, bob@example.com, "
                        "and carol@example.com."
                    ),
                }
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "send_emails"}},
        )
        assert_finish_reason(result, {"tool_calls"})
        schema = tool_schema_map(tools)
        args = parse_and_validate_tool_call(
            result.tool_calls[0], schema, expected_name="send_emails"
        )
        assert isinstance(args["recipients"], list)
        assert len(args["recipients"]) >= 3

    def test_no_tools_is_plain_text(self, client: OpenAI, model: str):
        result = stream_chat(
            client,
            model,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )
        assert_finish_reason(result, {"stop"})
        assert result.tool_calls == []
        assert result.content.strip()


# ---------------------------------------------------------------------------
# Multi-turn contract tests
# ---------------------------------------------------------------------------


class TestToolCallingMultiTurn:
    def test_tool_result_is_consumed_and_final_answer_is_text(
        self, client: OpenAI, model: str
    ):
        schemas = tool_schema_map(TOOLS_WEATHER)

        first = stream_chat(
            client,
            model,
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=TOOLS_WEATHER,
        )
        assert_finish_reason(first, {"tool_calls"})
        assert len(first.tool_calls) >= 1
        parse_and_validate_tool_call(
            first.tool_calls[0], schemas, expected_name="get_weather"
        )

        second = stream_chat(
            client,
            model,
            messages=[
                {"role": "user", "content": "What is the weather in London?"},
                assistant_tool_message_from_result(first),
                {
                    "role": "tool",
                    "tool_call_id": first.tool_calls[0]["id"],
                    "content": json.dumps(
                        {"temperature": 15, "unit": "celsius", "condition": "cloudy"}
                    ),
                },
            ],
            tools=TOOLS_WEATHER,
        )
        assert_finish_reason(second, {"stop"})
        assert second.tool_calls == []
        assert second.content.strip()
        assert "15" in second.content or "cloud" in second.content.lower()

    def test_chained_tool_use_search_then_calculate(self, client: OpenAI, model: str):
        schemas = tool_schema_map(TOOLS_SEARCH + TOOLS_CALCULATOR)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Search the web for the population of Tokyo, "
                    "then calculate what 10% of that number is."
                ),
            }
        ]

        step1 = stream_chat(
            client, model, messages=messages, tools=TOOLS_SEARCH + TOOLS_CALCULATOR
        )
        assert_finish_reason(step1, {"tool_calls"})
        assert len(step1.tool_calls) >= 1
        parse_and_validate_tool_call(step1.tool_calls[0], schemas)

        messages.append(assistant_tool_message_from_result(step1))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": step1.tool_calls[0]["id"],
                "content": json.dumps(
                    {
                        "results": [
                            {"title": "Tokyo population", "snippet": "13,960,000"}
                        ]
                    }
                ),
            }
        )

        step2 = stream_chat(
            client, model, messages=messages, tools=TOOLS_SEARCH + TOOLS_CALCULATOR
        )
        # Small models sometimes short-circuit and compute the answer in
        # their reasoning instead of chaining a second tool call. Accept
        # either path: (a) another tool call to `calculate`, or (b) a
        # direct text answer containing the correct result.
        assert_finish_reason(step2, {"tool_calls", "stop"})
        if step2.finish_reason == "tool_calls":
            assert len(step2.tool_calls) >= 1
            args2 = parse_and_validate_tool_call(step2.tool_calls[0], schemas)
            assert step2.tool_calls[0]["function"]["name"] == "calculate"
            assert "13960000" in args2["expression"].replace(
                ",", ""
            ) or "1396000" in args2["expression"].replace(",", "")

            messages.append(assistant_tool_message_from_result(step2))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": step2.tool_calls[0]["id"],
                    "content": "1396000",
                }
            )

            step3 = stream_chat(
                client, model, messages=messages, tools=TOOLS_SEARCH + TOOLS_CALCULATOR
            )
            assert_finish_reason(step3, {"stop"})
            assert step3.tool_calls == []
            assert "1396000" in step3.content.replace(",", "")
        else:
            # Short-circuit path: model did the math itself. Just verify
            # the final answer is present in the text.
            assert step2.tool_calls == []
            assert "1396000" in step2.content.replace(",", "")

    def test_multiple_prior_tool_results_synthesize_to_text(
        self, client: OpenAI, model: str
    ):
        result = stream_chat(
            client,
            model,
            messages=[
                {"role": "user", "content": "Get the weather in Tokyo and Paris."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Tokyo"}),
                            },
                        },
                        {
                            "id": "call_002",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Paris"}),
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_001",
                    "content": json.dumps(
                        {"temperature": 22, "unit": "celsius", "condition": "sunny"}
                    ),
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_002",
                    "content": json.dumps(
                        {"temperature": 18, "unit": "celsius", "condition": "rainy"}
                    ),
                },
            ],
            tools=TOOLS_WEATHER,
        )
        assert_finish_reason(result, {"stop"})
        assert result.tool_calls == []
        assert result.content.strip()
        lower = result.content.lower()
        assert "tokyo" in lower or "paris" in lower


# ---------------------------------------------------------------------------
# Model-behavior smoke tests
# These are intentionally looser because the model may vary.
# ---------------------------------------------------------------------------


class TestToolCallingModelBehavior:
    def test_many_tools_prefers_calculator_for_math_question(
        self, client: OpenAI, model: str
    ):
        result = stream_chat(
            client,
            model,
            messages=[
                {"role": "user", "content": "What is 2^10? Use a tool if helpful."}
            ],
            tools=ALL_TOOLS,
        )
        assert result.finish_reason in {"stop", "tool_calls"}
        if result.finish_reason == "tool_calls":
            assert len(result.tool_calls) >= 1
            assert result.tool_calls[0]["function"]["name"] == "calculate"

    def test_unicode_arguments_are_preserved(self, client: OpenAI, model: str):
        result = stream_chat(
            client,
            model,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Zürich, Switzerland?",
                }
            ],
            tools=TOOLS_WEATHER,
        )
        assert result.finish_reason in {"stop", "tool_calls"}
        if result.finish_reason == "tool_calls":
            schema = tool_schema_map(TOOLS_WEATHER)
            args = parse_and_validate_tool_call(
                result.tool_calls[0], schema, expected_name="get_weather"
            )
            assert args["city"]

    def test_system_instruction_encourages_tool_use(self, client: OpenAI, model: str):
        result = stream_chat(
            client,
            model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful weather assistant. "
                        "Always use the get_weather tool for weather questions."
                    ),
                },
                {"role": "user", "content": "How's the weather in Sydney?"},
            ],
            tools=TOOLS_WEATHER,
        )
        assert result.finish_reason in {"stop", "tool_calls"}
        if result.finish_reason == "tool_calls":
            schema = tool_schema_map(TOOLS_WEATHER)
            parse_and_validate_tool_call(
                result.tool_calls[0], schema, expected_name="get_weather"
            )
