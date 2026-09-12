# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-agnostic tool-calling helpers and end-to-end tool-execution scenarios.

Everything here depends only on an OpenAI-compatible ``client`` and a ``model``
name. Nothing knows how the endpoint came to exist -- a locally spawned worker,
a container, or a DynamoGraphDeployment on a Kubernetes cluster all work
identically. That is what lets the same scenarios run from
``tests/frontend/test_tool_calling_sglang.py`` (local processes) and
``tests/deploy/test_recipe_tool_execution.py`` (live cluster).

The ``assert_*`` scenarios execute real subprocesses **in the pytest process**,
not in the cluster, and the value each returns is generated per run and never
appears in any prompt. That is deliberate: it keeps the tests valid regardless
of where Dynamo itself is running, while still making hallucination unable to
satisfy the assertion.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

import pytest

openai = pytest.importorskip("openai")
OpenAI = openai.OpenAI
jsonschema = pytest.importorskip("jsonschema")
Draft7Validator = jsonschema.Draft7Validator


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
    assert (
        result.finish_reason in allowed
    ), f"unexpected finish_reason={result.finish_reason!r}, allowed={sorted(allowed)}"


def assistant_tool_message_from_result(result: StreamResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content or None,
        "tool_calls": result.tool_calls,
    }


MAX_TOOL_TURNS = 6

# These shell out deliberately. An in-process Python function would leave
# "did anything outside the test actually run?" unanswered.
_LOOKUP_CODE = (
    "import os, sys; "
    "print(os.environ['ACCESS_CODE'] if sys.argv[1].strip().lower() == 'alice' "
    "else 'DENIED')"
)
_LOOKUP_ID = (
    "import os, sys; "
    "print(os.environ['USER_ID'] if sys.argv[1].strip().lower() == 'alice' "
    "else 'UNKNOWN')"
)
_LOOKUP_QUOTA = (
    "import os, sys; "
    "print(os.environ['QUOTA'] if sys.argv[1].strip() == os.environ['USER_ID'] "
    "else '-1')"
)


def run_tool_cli(script: str, arg: str, env_extra: dict[str, str]) -> str:
    result = subprocess.run(
        [sys.executable, "-c", script, arg],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, **env_extra},
        check=True,
    )
    return result.stdout.strip()


def run_tool_loop(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    dispatch: dict[str, Any],
    max_turns: int = MAX_TOOL_TURNS,
) -> tuple[str, list[dict[str, Any]]]:
    """Drive a real tool-execution loop until the model answers with text.

    Returns (final_text, calls), where `calls` records every tool the model
    asked for and what that tool actually returned -- so a test can assert the
    tool ran, not merely that the final answer looks right.
    """
    calls: list[dict[str, Any]] = []
    convo = list(messages)
    for _ in range(max_turns):
        result = stream_chat(client, model, messages=convo, tools=tools)
        if not result.tool_calls:
            return (result.content or ""), calls

        # Echo the assistant turn back, tool_calls included, or the model has
        # no record of having asked.
        convo.append(assistant_tool_message_from_result(result))
        for tc in result.tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"model emitted unparseable arguments for {name}: "
                    f"{tc['function']['arguments']!r}"
                ) from e
            assert name in dispatch, (
                f"model called {name!r}, which was never offered. "
                f"Offered: {sorted(dispatch)}"
            )
            output = dispatch[name](args)  # the tool really runs here
            calls.append({"name": name, "args": args, "output": output})
            convo.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": str(output)}
            )

    raise AssertionError(
        f"model never produced a final text answer within {max_turns} turns; "
        f"calls so far: {calls}"
    )


# ---------------------------------------------------------------------------
# End-to-end tool-execution scenarios
# ---------------------------------------------------------------------------
# These are plain functions rather than test methods so that any deployment
# topology can drive them with its own (client, model) pair. Each raises
# AssertionError with a diagnostic message on failure, so callers can wrap them
# in flaky/xfail markers appropriate to their lane.


def assert_executes_real_tool_and_uses_output(client: OpenAI, model: str) -> None:
    """The answer must contain a secret only the executed tool could supply.

    The secret is generated per call and never appears in any prompt, so the
    assertion cannot be satisfied by hallucination, memorisation or luck.
    """
    secret = uuid.uuid4().hex[:12]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_access_code",
                "description": (
                    "Look up the access code for a user. This is the only "
                    "way to obtain an access code."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user": {"type": "string", "description": "the username"}
                    },
                    "required": ["user"],
                },
            },
        }
    ]

    def lookup_access_code(args: dict[str, Any]) -> str:
        return run_tool_cli(
            _LOOKUP_CODE, str(args.get("user", "")), {"ACCESS_CODE": secret}
        )

    text, calls = run_tool_loop(
        client,
        model,
        [
            {
                "role": "user",
                "content": (
                    "Look up the access code for user alice, then tell me "
                    "the code exactly as returned."
                ),
            }
        ],
        tools,
        {"lookup_access_code": lookup_access_code},
    )

    assert calls, "the model never called the tool, so nothing was executed"
    assert calls[0]["name"] == "lookup_access_code"
    assert calls[0]["args"].get("user", "").strip().lower() == "alice", (
        f"model passed an unusable argument: {calls[0]['args']!r} -- "
        "schema-valid but wrong, which protocol-only tests cannot catch"
    )
    assert calls[0]["output"] == secret, (
        f"tool returned {calls[0]['output']!r}, expected the generated "
        "secret; the subprocess did not get the argument it should have"
    )
    assert secret in text, (
        "final answer did not contain the executed tool's output.\n"
        f"secret={secret!r}\nanswer={text[:400]!r}"
    )


def assert_chained_tools_thread_real_output(client: OpenAI, model: str) -> None:
    """Two real executions, the second satisfiable only via the first.

    ``get_quota`` returns -1 unless handed the exact id ``get_user_id``
    produced, and both values are random per call. A correct final number
    therefore proves the model threaded real output from one execution into the
    next -- the capability that distinguishes models here.
    """
    user_id = f"U-{uuid.uuid4().hex[:8]}"
    quota = str(uuid.uuid4().int % 9000 + 1000)  # 4 digits, unguessable
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_user_id",
                "description": "Resolve a username to its internal user id.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_quota",
                "description": (
                    "Get the storage quota for an internal user id. "
                    "Requires the id from get_user_id, not a username."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                },
            },
        },
    ]

    dispatch = {
        "get_user_id": lambda a: run_tool_cli(
            _LOOKUP_ID, str(a.get("name", "")), {"USER_ID": user_id}
        ),
        "get_quota": lambda a: run_tool_cli(
            _LOOKUP_QUOTA,
            str(a.get("user_id", "")),
            {"USER_ID": user_id, "QUOTA": quota},
        ),
    }

    text, calls = run_tool_loop(
        client,
        model,
        [
            {
                "role": "user",
                "content": (
                    "What is alice's storage quota? Look up her user id "
                    "first, then use that id to get the quota. Report the "
                    "number."
                ),
            }
        ],
        tools,
        dispatch,
    )

    names = [c["name"] for c in calls]
    assert "get_user_id" in names, f"never resolved the id; calls={names}"
    assert "get_quota" in names, f"never fetched the quota; calls={names}"
    assert names.index("get_user_id") < names.index(
        "get_quota"
    ), f"called get_quota before get_user_id: {names}"

    quota_call = calls[names.index("get_quota")]
    assert quota_call["args"].get("user_id") == user_id, (
        f"second call used {quota_call['args'].get('user_id')!r} instead of "
        f"the id the first call actually returned ({user_id!r})"
    )
    assert (
        quota_call["output"] == quota
    ), f"get_quota returned {quota_call['output']!r} -- it was handed the wrong id"
    assert quota in text, (
        "final answer omitted the quota the tool actually returned.\n"
        f"quota={quota!r}\nanswer={text[:400]!r}"
    )
