# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The replay judge must catch changed requests and tool values independently."""

from __future__ import annotations

import copy
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests.frontend.trace_replay.contract import (
    MISSING,
    ReplayError,
    dumps,
    file_digest,
    judge_preprocessed,
    judge_response,
    load_corpus,
    loads,
    pointer,
    same_json,
    scheduled_events,
    validate_source,
)
from tests.frontend.trace_replay.runner import (
    _write_junit,
    assemble_sse,
    compare_baseline,
    normalize_unary,
)

FIXTURES = Path(__file__).parent / "trace_replay" / "fixtures" / "deepseek_v4"
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.parallel,
]


def test_startup_failure_is_a_junit_error_even_with_zero_executions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "junit.xml"
    _write_junit(path, [], ["frontend exited before readiness"])
    suite = ET.parse(path).getroot()
    assert suite.attrib["tests"] == "1"
    assert suite.attrib["errors"] == "1"
    assert suite.find("testcase/error").text == "frontend exited before readiness"


def test_v41_native_bundle_can_omit_nonexistent_generation_config(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(FIXTURES)
    manifest = copy.deepcopy(corpus.manifest)
    manifest["profile"]["runtime_config"].update(
        tool_call_parser="deepseek_v41", reasoning_parser="deepseek_v41"
    )
    manifest["profile"]["model"]["assets"].pop("generation_config.json")
    source = FIXTURES / manifest["traces"][0]["path"]
    (tmp_path / source.name).write_bytes(source.read_bytes())
    (tmp_path / "manifest.json").write_text(dumps(manifest))
    assert len(load_corpus(tmp_path).profile["model"]["assets"]) == 3
    manifest["profile"]["runtime_config"].update(
        tool_call_parser="deepseek_v4", reasoning_parser="deepseek_v4"
    )
    (tmp_path / "manifest.json").write_text(dumps(manifest))
    with pytest.raises(ReplayError, match="hashed native"):
        load_corpus(tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        {"kind": "authored"},
        {"kind": "benchmark", "context": {}},
        {
            "kind": "custom_report",
            "context": {"ref": "", "pr": "not a URL", "issue": "free text"},
        },
    ],
)
def test_minimal_and_optional_context_sources(source: dict) -> None:
    validate_source(source)


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"kind": "unknown"},
        {"kind": []},
        {"kind": "production", "context": "report"},
        {"kind": "agent_report", "context": {"ref": 5}},
        {"kind": "benchmark", "ref": "wrong nesting"},
    ],
)
def test_invalid_sources_are_rejected(source: dict) -> None:
    with pytest.raises(ReplayError):
        validate_source(source)


def test_json_comparison_preserves_numbers_without_boolean_or_string_coercion() -> None:
    assert same_json(loads("9007199254740993.0"), 9007199254740993)
    assert same_json(loads("1.0"), 1)
    assert not same_json(True, 1)
    assert not same_json("1", 1)
    assert not same_json(loads("9007199254740993.0"), 9007199254740992)
    assert same_json({"x": 1, "y": [True, None]}, {"y": [True, None], "x": 1})
    assert not same_json([1, 2], [2, 1])
    assert loads(dumps({"x": loads("1.2345678901234567890123456789")}))["x"] == loads(
        "1.2345678901234567890123456789"
    )


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', "NaN", "Infinity", "-Infinity"])
def test_ambiguous_or_nonfinite_json_is_rejected(text: str) -> None:
    with pytest.raises(ReplayError):
        loads(text)


def test_json_pointer_escapes_and_missing_are_distinct_from_null() -> None:
    value = {"a/b": {"~": [None, 3]}}
    assert pointer(value, "/a~1b/~0/1") == 3
    assert pointer(value, "/a~1b/~0/0") is None
    assert pointer(value, "/missing") is MISSING
    assert not same_json(pointer(value, "/missing"), None)


def test_wrong_prompt_and_effective_parameter_fail() -> None:
    case = load_corpus(FIXTURES, 1).cases[0]
    actual = {
        "token_ids": list(case.expected_preprocessed["token_ids"]),
        "sampling_options": {"temperature": 0, "seed": 0, "n": 1},
        "stop_conditions": {"max_tokens": 4096, "stop_token_ids_hidden": [1]},
        "eos_token_ids": [1],
    }
    assert not judge_preprocessed(case, actual)
    actual["token_ids"][0] += 1
    actual["sampling_options"]["seed"] = 9
    failures = judge_preprocessed(case, actual)
    assert {failure["path"] for failure in failures} == {
        "/token_ids",
        "/sampling_options/seed",
    }


def test_gzip_corpus_limit_does_not_change_total_or_silently_cap_full_run(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(FIXTURES)
    manifest = copy.deepcopy(corpus.manifest)
    source = FIXTURES / manifest["traces"][0]["path"]
    destination = tmp_path / "cases.jsonl.gz"
    with gzip.open(destination, "wb") as stream:
        stream.write(source.read_bytes())
    manifest["traces"] = [
        {"path": destination.name, "sha256": file_digest(destination), "count": 5}
    ]
    (tmp_path / "manifest.json").write_text(dumps(manifest))
    limited = load_corpus(tmp_path, 1)
    assert len(limited.cases) == 1 and limited.total_cases == 5
    assert len(load_corpus(tmp_path).cases) == 5
    with pytest.raises(ReplayError):
        load_corpus(tmp_path, 0)


def test_rechunking_preserves_terminal_eos_and_exact_generated_ids() -> None:
    for case in load_corpus(FIXTURES).cases:
        recorded = [
            token for event in case.worker_events for token in event["token_ids"]
        ]
        for schedule in ("whole", "one_token", "fixed32", "recorded"):
            events = scheduled_events(case, schedule)
            assert [
                token for event in events for token in event["token_ids"]
            ] == recorded
            assert recorded[-1] == 1
            assert [event.get("finish_reason") for event in events].count("stop") == 1
            assert events[-1]["finish_reason"] == "stop"


def _stream(chunks: list[dict], done: bool = True) -> list[str]:
    lines = [line for chunk in chunks for line in ("data: " + dumps(chunk), "")]
    return lines + (["data: [DONE]", ""] if done else [])


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "response-1",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        "usage": None,
    }


def test_sse_accumulates_typed_tool_arguments_and_detects_wrong_value() -> None:
    case = load_corpus(FIXTURES, 1).cases[0]
    chunks = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "discoverer_get",
                            "arguments": '{"element_name":',
                        },
                    }
                ],
            }
        ),
        _chunk(
            {"tool_calls": [{"index": 0, "function": {"arguments": '"radium"}'}}]},
            "tool_calls",
        ),
        {"id": "response-1", "choices": [], "usage": case.expected_response["usage"]},
    ]
    response = assemble_sse(_stream(chunks))
    assert not judge_response(case, response)
    response["message"]["tool_calls"][0]["function"]["arguments"][
        "element_name"
    ] = "uranium"
    assert judge_response(case, response)[0]["path"] == "/message"


@pytest.mark.parametrize(
    "variant", ["no_done", "after_finish", "double_finish", "changed_id"]
)
def test_invalid_sse_terminal_and_tool_id_sequences_fail(variant: str) -> None:
    chunks = [_chunk({"role": "assistant", "content": "a"}), _chunk({}, "stop")]
    if variant in ("after_finish", "double_finish"):
        chunks.append(
            _chunk(
                {"content": "b"} if variant == "after_finish" else {},
                "stop" if variant == "double_finish" else None,
            )
        )
    if variant == "changed_id":
        chunks = [
            _chunk(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "one",
                            "function": {"name": "f", "arguments": "{"},
                        }
                    ],
                }
            ),
            _chunk(
                {
                    "tool_calls": [
                        {"index": 0, "id": "two", "function": {"arguments": "}"}}
                    ]
                },
                "tool_calls",
            ),
        ]
    with pytest.raises(ReplayError):
        assemble_sse(_stream(chunks, done=variant != "no_done"))


def test_unary_requires_unique_generated_call_ids() -> None:
    call = {
        "id": "same",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }
    payload = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [call, call],
                },
            }
        ]
    }
    with pytest.raises(ReplayError):
        normalize_unary(payload)


@pytest.mark.parametrize("invalid", [False, 0, "", {}])
def test_wrong_wire_tool_calls_type_cannot_pass_a_no_tool_golden(invalid) -> None:
    message = {"role": "assistant", "content": "Six sides.", "tool_calls": invalid}
    payload = {"choices": [{"index": 0, "finish_reason": "stop", "message": message}]}
    with pytest.raises(ReplayError, match="tool_calls.*array"):
        normalize_unary(payload)
    with pytest.raises(ReplayError, match="tool_calls.*array"):
        assemble_sse(_stream([_chunk(message, "stop")]))


def test_baseline_separates_fix_from_unjudged_change() -> None:
    previous = {"status": "fail", "preprocessed_sha256": "a", "response_sha256": "a"}
    current = {"status": "pass", "preprocessed_sha256": "a", "response_sha256": "b"}
    assert compare_baseline(current, previous) == "fix"
    previous["status"] = "pass"
    assert compare_baseline(current, previous) == "unjudged_change"
    current["status"] = "fail"
    assert compare_baseline(current, previous) == "new_failure"
