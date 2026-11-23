# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU replay of the public contracts exercised by Custom regression cases."""

from __future__ import annotations

import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tool_calling_probe as probe  # noqa: E402

from benchmarks.tool_calling.custom_runner import apply_request_contract  # noqa: E402
from benchmarks.tool_calling.e2e_verifier.cli import _custom_verdict  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def tool(name="lookup", *, strict=True, namespace=None, schema=None):
    function = {
        "name": name,
        "strict": strict,
        "parameters": schema
        if schema is not None
        else {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 8,
                    "pattern": "^[A-Z]+$",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    }
    if namespace is not None:
        function["namespace"] = namespace
    return {"type": "function", "function": function}


def call(name="lookup", arguments=None, *, call_id="call_1", namespace=None):
    result = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                {"city": "PARIS"} if arguments is None else arguments
            ),
        },
    }
    if namespace is not None:
        result["namespace"] = namespace
    return result


def case(**overrides):
    return probe.Case(
        **{
            "case_id": "cpu_contract",
            "description": "CPU contract replay",
            "messages": (),
            "tools": (tool(),),
            "tool_choice": "auto",
            "min_tool_calls": 0,
            "expected_finish_reasons": ("stop", "tool_calls"),
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "strict,schema_mode,expected",
    [
        (True, "auto", "dynamo_api"),
        (False, "auto", "model_quality"),
        (False, "strict", "dynamo_api"),
        (True, "off", "model_quality"),
    ],
)
def test_decoded_constraints_follow_effective_schema_policy(
    strict, schema_mode, expected
):
    target = case(tools=(tool(strict=strict),), schema_mode=schema_mode)
    result = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[call(arguments={"city": "a", "extra": "b"})],
    )
    errors, _ = probe.validate_result(target, result)
    assert len(errors) >= 3
    assert probe.classify_failure(target, result, errors) == expected
    assert _custom_verdict("complete", {expected: 1}) == (
        "fail" if expected == "dynamo_api" else "pass"
    )


def test_schema_relaxation_does_not_relax_the_other_constraints():
    original = tool()["function"]["parameters"]
    effective = {**original, "properties": {"city": {"type": "string", "minLength": 3}}}
    target = case(tools=(tool(strict=False),), enforced_schemas={"lookup": effective})
    result = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[call(arguments={"city": "paris"})],
    )
    errors, _ = probe.validate_result(target, result)
    assert probe.classify_failure(target, result, errors) == "model_quality"
    result.tool_calls = [call(arguments={"city": "a"})]
    errors, _ = probe.validate_result(target, result)
    assert "enforced_schema_violation" in {e["kind"] for e in errors}
    assert probe.classify_failure(target, result, errors) == "dynamo_api"


def test_global_strict_overrides_a_case_relaxation():
    target = case(
        tools=(tool(strict=False),),
        schema_mode="strict",
        enforced_schemas={"lookup": True},
    )
    result = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[call(arguments={"city": "a"})],
    )
    errors, _ = probe.validate_result(target, result)
    assert probe.classify_failure(target, result, errors) == "dynamo_api"


@pytest.mark.parametrize(
    "schema",
    [
        False,
        {"$ref": "#/missing"},
        {"$ref": "https://invalid.example/schema"},
    ],
)
def test_boolean_and_unverifiable_schemas_never_silently_pass(schema):
    target = case(tools=(tool(schema=schema),))
    result = probe.ChatResult(finish_reason="tool_calls", tool_calls=[call()])
    errors, _ = probe.validate_result(target, result)
    expected = "dynamo_api" if schema is False else "unclassified"
    assert probe.classify_failure(target, result, errors) == expected
    assert _custom_verdict("complete", {expected: 1}) != "pass"


def test_actual_payload_controls_allowlist_and_single_call_policy():
    target = case(tools=(tool(), tool("delete")))
    payload = {
        "tools": list(target.tools),
        "tool_choice": {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
        },
        "parallel_tool_calls": False,
    }
    result = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[call("delete"), call(call_id="call_2")],
        request_payload=payload,
    )
    errors, _ = probe.validate_result(target, result)
    assert sum(e["kind"] == "tool_policy_violation" for e in errors) == 2
    assert probe.classify_failure(target, result, errors) == "dynamo_api"
    result.tool_calls, result.finish_reason, result.content = [], "stop", "READY"
    assert probe.validate_result(target, result)[0] == []


def test_permitted_parallelism_retains_semantic_count_classification():
    target = case(parallel_tool_calls=True, exact_tool_calls=1)
    result = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[call(), call(call_id="call_2")],
    )
    errors, _ = probe.validate_result(target, result)
    assert [e["kind"] for e in errors] == ["wrong_tool_call_count"]
    assert probe.classify_failure(target, result, errors) == "model_quality"


def test_namespaced_functions_use_their_own_schemas_and_policy():
    target = case(
        tools=(
            tool(namespace="crm"),
            tool(
                namespace="inventory",
                schema={"type": "object", "required": ["order_id"]},
            ),
        ),
        tool_choice={"type": "function", "name": "lookup", "namespace": "crm"},
    )
    result = probe.ChatResult(
        finish_reason="tool_calls",
        tool_calls=[call(namespace="crm")],
    )
    assert probe.validate_result(target, result)[0] == []
    result.tool_calls = [call(namespace="inventory", arguments={"order_id": 42})]
    errors, _ = probe.validate_result(target, result)
    assert [e["kind"] for e in errors] == ["tool_policy_violation"]


def test_failure_classification_uses_actual_followup_policy():
    target = case(tool_choice="required", expected_finish_reasons=("tool_calls",))
    result = probe.ChatResult(
        finish_reason="stop",
        content="READY",
        request_payload={"tool_choice": "auto"},
    )
    errors, _ = probe.validate_result(target, result)
    assert probe.classify_failure(target, result, errors) == "model_quality"


def test_capability_gaps_are_inconclusive_without_dispatch(monkeypatch):
    def no_dispatch(*args, **kwargs):
        pytest.fail("an unsupported vision case must not dispatch")

    monkeypatch.setattr(probe, "post_json", no_dispatch)
    target = case(required_capabilities=("vision",))
    record = probe.run_case(
        target,
        "nonstream",
        iteration=1,
        url="unused",
        api_key=None,
        model="cpu",
        temperature=0,
        max_tokens=64,
        timeout=1,
        extra_headers={},
        raw_chars=1000,
        record_success_raw=True,
    )
    assert record["pass"] is False
    assert record["failure_kinds"] == ["capability_unavailable"]
    assert (
        _custom_verdict("complete", {record["failure_category"]: 1}) == "inconclusive"
    )
    contracted = apply_request_contract(
        (target,),
        {
            "capabilities": ["vision"],
            "schema_mode": "off",
            "provenance": {"parser_version": "2", "grammar_backend": "xgrammar"},
        },
    )[0]
    assert contracted.available_capabilities == ("vision",)
    assert contracted.schema_mode == "off"
    assert contracted.runtime_provenance["parser_version"] == "2"


@pytest.fixture
def endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            streaming = payload["stream"]
            arguments = '{"token":"CATALOG_QUARTZ_73"}'
            if self.path.endswith("/responses"):
                item = {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "report_catalog",
                    "arguments": arguments,
                    "status": "completed",
                }
                response = {
                    "id": "resp_1",
                    "model": "cpu",
                    "status": "completed",
                    "output": [item],
                }
                events = [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {**item, "arguments": "", "status": "in_progress"},
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 0,
                        "item_id": "fc_1",
                        "delta": arguments,
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": 0,
                        "item_id": "fc_1",
                        "arguments": arguments,
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    },
                    {"type": "response.completed", "response": response},
                ]
                body = (
                    "".join(
                        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
                    )
                    if streaming
                    else json.dumps(response)
                )
            else:
                response = {
                    "id": "chat_1",
                    "model": "cpu",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    call(
                                        "report_catalog", {"token": "CATALOG_QUARTZ_73"}
                                    ),
                                ],
                            },
                        }
                    ],
                }
                delta = {
                    "index": 0,
                    **call("report_catalog", {"token": "CATALOG_QUARTZ_73"}),
                }
                chunks = [
                    {
                        "id": "chat_1",
                        "model": "cpu",
                        "choices": [
                            {
                                "delta": {"tool_calls": [delta]},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                ]
                body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
                body += "data: [DONE]\n\n"
                if not streaming:
                    body = json.dumps(response)
            wire = body.encode()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/event-stream" if streaming else "application/json",
            )
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        assert not worker.is_alive()


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
def test_http_transport_runs_targeted_cases_and_preserves_wire_evidence(
    endpoint, api, mode
):
    url, requests = endpoint
    target = next(
        c
        for c in probe.build_cases("generic")
        if c.case_id == f"regression_allowed_tools_required_catalog_{api}"
    )
    record = probe.run_case(
        target,
        mode,
        iteration=1,
        url=url,
        api_key=None,
        model="cpu",
        temperature=0,
        max_tokens=64,
        timeout=2,
        extra_headers={},
        raw_chars=20000,
        record_success_raw=True,
    )
    assert record["pass"], record["errors"]
    route, payload = requests[0]
    assert route.endswith("/responses" if api == "responses" else "/chat/completions")
    assert len(payload["tools"]) == 2
    assert record["request"] == payload
    assert record["response"]["tool_calls"][0]["id"] == "call_1"
    if api == "responses":
        assert "messages" not in payload and payload["max_output_tokens"] == 64
        assert payload["tool_choice"]["mode"] == "required"
        assert record["response"]["status"] == "completed"
        assert record["response"]["tool_calls"][0]["item_id"] == "fc_1"


@pytest.mark.parametrize(
    "tail,expected",
    [
        ("", "stream_incomplete"),
        (
            'data: {"error":{"message":"generation failed"}}\n\ndata: [DONE]\n\n',
            "stream_error",
        ),
    ],
)
def test_chat_error_or_missing_terminal_cannot_pass_after_a_valid_finish(
    monkeypatch,
    tail,
    expected,
):
    chunk = {"choices": [{"delta": {"content": "READY"}, "finish_reason": "stop"}]}
    body = f"data: {json.dumps(chunk)}\n\n" + tail
    monkeypatch.setattr(probe, "post_json", lambda *a, **k: io.BytesIO(body.encode()))
    result = probe.request_stream(
        "unused", {}, api_key=None, timeout=1, extra_headers={}
    )
    errors, _ = probe.validate_result(case(), result)
    assert expected in {e["kind"] for e in errors}
