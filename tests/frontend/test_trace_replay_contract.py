# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The replay judge must catch changed requests and tool values independently."""

from __future__ import annotations

import copy
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
import requests

from tests.frontend.trace_replay import runner
from tests.frontend.trace_replay.__main__ import main
from tests.frontend.trace_replay.contract import (
    MISSING,
    ReplayError,
    digest,
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
    write_json,
)
from tests.frontend.trace_replay.runner import (
    RunOptions,
    _baseline,
    _environment,
    _execution_configuration,
    _sse_lines,
    _write_junit,
    assemble_sse,
    compare_baseline,
    normalize_unary,
)

FIXTURES = Path(__file__).parent / "trace_replay" / "fixtures" / "deepseek_v41"
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.parallel,
]


@pytest.mark.parametrize(
    "status", ["infra_error", "target_unavailable", "provenance_rejected"]
)
def test_publication_failure_status_without_identity_is_an_incomplete_baseline(
    tmp_path: Path, status: str
) -> None:
    # Publication report generation is independently covered by its stdlib CI hook.
    write_json(tmp_path / "summary.json", {"status": status, "executions": 0})
    assert _baseline(tmp_path, {"harness": "current"}, set()) == (
        "baseline_incomplete",
        {},
    )


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
    manifest["profile"]["model"]["assets"]["generation_config.json"] = "0" * 64
    (tmp_path / "manifest.json").write_text(dumps(manifest))
    assert len(load_corpus(tmp_path).profile["model"]["assets"]) == 4


def test_public_v41_fixtures_have_five_reviewed_responses_and_native_assets() -> None:
    corpus = load_corpus(FIXTURES)
    assert [case.identifier for case in corpus.cases] == [
        "bfcl.simple_python_245.thinking-disabled",
        "bfcl.multiple_2.thinking-enabled",
        "bfcl.parallel_82.thinking-enabled",
        "bfcl.parallel_multiple_48.thinking-disabled",
        "bfcl.irrelevance_28.thinking-disabled",
    ]
    assert all(case.expected_response is not None for case in corpus.cases)
    assert corpus.profile["runtime_config"]["tool_call_parser"] == "deepseek_v41"
    assert corpus.profile["runtime_config"]["reasoning_parser"] == "deepseek_v41"
    assert set(corpus.profile["model"]["assets"]) == {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }


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


def test_semantic_fingerprints_ignore_number_spelling_without_rounding() -> None:
    for left, right in (
        ("1", "1.0"),
        ("100", "1e2"),
        ("-0.0", "0"),
        ("9007199254740993", "9007199254740993.0"),
        ("1.2345678901234567890123456789", "1.234567890123456789012345678900"),
    ):
        first, second = {"argument": loads(left)}, {"argument": loads(right)}
        assert same_json(first, second)
        assert digest(first) == digest(second)
    assert digest(1) != digest(True)
    assert digest(1) != digest("1")
    assert digest(loads("9007199254740993.0")) != digest(9007199254740992)
    assert dumps(loads("1.0")) == "1.0"


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
                "content": case.expected_response["message"]["content"],
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


@pytest.mark.parametrize("character", ["\u0085", "\u2028", "\u2029"])
def test_requests_sse_preserves_unicode_line_characters(character: str) -> None:
    text = "before" + character + "after"
    chunks = [_chunk({"role": "assistant", "content": text}), _chunk({}, "stop")]
    response = requests.Response()
    response.status_code = 200
    response.encoding = "utf-8"
    response._content = ("\n".join(_stream(chunks)) + "\n").encode("utf-8")
    response._content_consumed = True
    # Exercise requests' real byte-line splitter with a character that its
    # decode_unicode=True path incorrectly treats as a new HTTP line.
    assert assemble_sse(_sse_lines(response))["message"]["content"] == text


def test_requests_sse_rejects_invalid_utf8() -> None:
    response = requests.Response()
    response._content = b"data: \xff\n\n"
    response._content_consumed = True
    with pytest.raises(ReplayError, match="UTF-8"):
        list(_sse_lines(response))


@pytest.mark.parametrize("invalid", [None, False, [], "wrong"])
def test_nonobject_unary_and_streamed_choices_fail_as_replay_errors(
    invalid: Any,
) -> None:
    payload = {"id": "response-1", "choices": [invalid]}
    with pytest.raises(ReplayError, match="choice.*object"):
        normalize_unary(payload)
    with pytest.raises(ReplayError, match="choice.*object"):
        assemble_sse(_stream([payload]))


@pytest.mark.parametrize("invalid", [None, False, [], "wrong"])
@pytest.mark.parametrize("field", ["tool_call", "function"])
def test_nonobject_tool_fields_fail_as_replay_errors(invalid: Any, field: str) -> None:
    call = {"index": 0, "id": "call-1", "type": "function", "function": invalid}
    if field == "tool_call":
        call = invalid
    message = {"role": "assistant", "content": None, "tool_calls": [call]}
    payload = {
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]
    }
    with pytest.raises(ReplayError):
        normalize_unary(payload)
    with pytest.raises(ReplayError):
        assemble_sse(_stream([_chunk(message, "tool_calls")]))


@pytest.mark.parametrize("mode", ["unary", "sse"])
def test_malformed_wire_response_is_a_product_failure_with_complete_worker_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    corpus = load_corpus(FIXTURES, 1)
    case = corpus.cases[0]

    def malformed_response(
        session: requests.Session, url: str, **kwargs: Any
    ) -> requests.Response:
        request = loads(kwargs["data"])
        annotation = request["nvext"]["annotations"][-1]
        execution_id = loads(annotation.removeprefix(runner.ANNOTATION_PREFIX))[
            "execution_id"
        ]
        ingress = {
            "token_ids": case.expected_preprocessed["token_ids"],
            "sampling_options": {"temperature": 0, "seed": 0, "n": 1},
            "stop_conditions": {"max_tokens": 4096, "stop_token_ids_hidden": [1]},
            "eos_token_ids": [1],
        }
        write_json(tmp_path / "worker" / f"{execution_id}.request.json", ingress)
        write_json(
            tmp_path / "worker" / f"{execution_id}.emission.json",
            {
                "state": "completed",
                "terminal_count": 1,
                "token_ids": [
                    token
                    for event in case.worker_events
                    for token in event["token_ids"]
                ],
            },
        )
        payload = {"id": "response-1", "choices": [None]}
        response = requests.Response()
        response.status_code = 200
        response.encoding = "utf-8"
        response._content = (
            "\n".join(_stream([payload])) + "\n" if mode == "sse" else dumps(payload)
        ).encode("utf-8")
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "post", malformed_response)
    result = runner._exchange(
        case,
        mode,
        "whole",
        corpus,
        RunOptions(FIXTURES, tmp_path, tmp_path),
        "http://unused",
    )
    assert result["status"] == "fail"
    assert result["infra_errors"] == []
    assert result["failures"][0]["path"] == "/wire"


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
    assert compare_baseline(current, None) == "infrastructure_unjudged"


def _baseline_files(directory: Path, *, exit_code: int = 0, completed: int = 1) -> str:
    key = dumps(["case", "unary", "whole"])
    summary = {
        "identity": {"test": "identity"},
        "exit_code": exit_code,
        "coverage": {"executions_planned": 1, "executions_completed": completed},
    }
    (directory / "summary.json").write_text(dumps(summary))
    (directory / "observations.jsonl").write_text(
        dumps({"key": loads(key), "status": "pass"}) + "\n"
    )
    return key


@pytest.mark.parametrize(
    "variant", ["exit_2", "preflight_exit_2", "partial_coverage", "missing_row"]
)
def test_incomplete_baseline_does_not_create_product_attention(
    tmp_path: Path, variant: str
) -> None:
    key = _baseline_files(
        tmp_path,
        exit_code=2 if variant in ("exit_2", "preflight_exit_2") else 0,
        completed=0 if variant == "partial_coverage" else 1,
    )
    if variant == "missing_row":
        (tmp_path / "observations.jsonl").write_text("")
    if variant == "preflight_exit_2":
        write_json(tmp_path / "summary.json", {"exit_code": 2, "status": "infra_error"})
    assert _baseline(tmp_path, {"test": "identity"}, {key}) == (
        "baseline_incomplete",
        {},
    )


@pytest.mark.parametrize("record", [None, {}, {"key": None}, {"key": ["case"]}])
def test_malformed_baseline_rows_raise_replay_error(
    tmp_path: Path, record: Any
) -> None:
    key = _baseline_files(tmp_path)
    (tmp_path / "observations.jsonl").write_text(dumps(record) + "\n")
    with pytest.raises(ReplayError, match="baseline observation"):
        _baseline(tmp_path, {"test": "identity"}, {key})


def test_complete_baseline_is_comparable(tmp_path: Path) -> None:
    key = _baseline_files(tmp_path)
    status, rows = _baseline(tmp_path, {"test": "identity"}, {key})
    assert status == "compared" and rows[key]["status"] == "pass"


@pytest.mark.parametrize("failure", ["options", "corpus", "assets"])
def test_cli_preflight_failure_writes_summary_and_junit(
    tmp_path: Path, failure: str
) -> None:
    output = tmp_path / "results"
    args = [
        "run",
        "--corpus",
        str(tmp_path / "missing" if failure == "corpus" else FIXTURES),
        "--model-dir",
        str(tmp_path / "missing-assets"),
        "--output-dir",
        str(output),
    ]
    if failure == "options":
        args += ["--concurrency", "0"]
    assert main(args) == 2
    assert loads((output / "summary.json").read_bytes())["status"] == "infra_error"
    suite = ET.parse(output / "junit.xml").getroot()
    assert suite.attrib["errors"] == "1"
    assert suite.find("testcase/error") is not None


def test_cli_malformed_baseline_writes_summary_and_junit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    output = tmp_path / "results"
    options = RunOptions(FIXTURES, tmp_path, output, baseline=baseline)
    identity = runner.comparison_identity(load_corpus(FIXTURES), options, {})
    write_json(
        baseline / "summary.json",
        {
            "identity": identity,
            "exit_code": 0,
            "coverage": {"executions_planned": 1, "executions_completed": 1},
        },
    )
    (baseline / "observations.jsonl").write_text('{"status":"pass"}\n')
    monkeypatch.setattr(runner, "verify_assets", lambda *_: {})
    assert (
        main(
            [
                "run",
                "--corpus",
                str(FIXTURES),
                "--model-dir",
                str(tmp_path),
                "--output-dir",
                str(output),
                "--baseline",
                str(baseline),
            ]
        )
        == 2
    )
    summary = loads((output / "summary.json").read_bytes())
    assert "baseline observation" in summary["infrastructure_errors"][0]
    assert ET.parse(output / "junit.xml").getroot().attrib["errors"] == "1"


def test_replay_environment_drops_inherited_routes_and_does_not_report_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DYN_MIGRATION_LIMIT", "99")
    monkeypatch.setenv("DYN_DEFAULT_CHAT_TEMPLATE_KWARGS", '{"enable_thinking":false}')
    monkeypatch.setenv("DYN_TOKENIZER", "wrong-model")
    monkeypatch.setenv("DYN_REPLAY_HARNESS_REVISION", "parent-metadata")
    monkeypatch.setenv("HF_TOKEN", "test-credential-must-not-be-reported")
    env = _environment(tmp_path)
    assert (
        not {
            "DYN_MIGRATION_LIMIT",
            "DYN_DEFAULT_CHAT_TEMPLATE_KWARGS",
            "DYN_TOKENIZER",
            "DYN_REPLAY_HARNESS_REVISION",
        }
        & env.keys()
    )
    assert env["DYN_DISCOVERY_BACKEND"] == "file"
    report = _execution_configuration(["worker"], ["frontend"], env)
    assert report["environment"]["DYN_FILE_KV"] == str(tmp_path / "discovery")
    assert "HF_TOKEN" not in report["environment"]
    assert "test-credential-must-not-be-reported" not in dumps(report)


def test_startup_report_pins_migration_and_preserves_parent_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "verify_assets", lambda *_: {})
    monkeypatch.setattr(runner, "allocate_port", lambda *_: 12345)
    monkeypatch.setattr(runner, "deallocate_port", lambda *_: None)
    monkeypatch.setenv("DYN_REPLAY_HARNESS_REVISION", "harness-metadata")

    def unavailable_process(**kwargs: Any) -> None:
        raise RuntimeError("test startup failure")

    monkeypatch.setattr(runner, "ManagedProcess", unavailable_process)
    code, summary = runner.run(RunOptions(FIXTURES, tmp_path, tmp_path / "results"))
    assert code == 2
    command = summary["build"]["execution"]["frontend_command"]
    assert command[command.index("--migration-limit") + 1] == "0"
    assert summary["harness_revision"] == "harness-metadata"
    assert (
        "DYN_REPLAY_HARNESS_REVISION"
        not in summary["build"]["execution"]["environment"]
    )
