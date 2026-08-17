#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a static tool-calling report.

This is the cron-friendly counterpart to tool_calling_probe.py. It runs a
fixed OpenAI-compatible tool-calling sweep against NVIDIA's endpoint using a
server-side key, then publishes static HTML/JSON artifacts that nginx can serve
without exposing an interactive backend or API-key form.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import html
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tool_calling_probe as probe  # noqa: E402

DEFAULT_SITE_DIR = Path(probe.DEFAULT_OUTPUT_ROOT) / "static-site"
DEFAULT_RUNS_ROOT = Path(probe.DEFAULT_OUTPUT_ROOT) / "static-runs"
DEFAULT_TITLE = "Tool Calling Qualification Report"
ALLOWED_BASE_URL_HOSTS = {
    "inference-api.nvidia.com",
    "integrate.api.nvidia.com",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def local_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in value)
    return slug.strip("-")[:96] or "run"


def model_slug_from_model(model: str) -> str:
    return safe_slug(model.split("/")[-1].lower())


def model_label_from_model(model: str) -> str:
    suffix = model.split("/")[-1] or model
    normalized = suffix.replace("-", " ").replace("_", " ")
    parts = []
    for part in normalized.split():
        lower = part.lower()
        if lower in {"kimi", "deepseek", "v4", "k2.6", "k2.5", "pro", "flash"}:
            parts.append(part.upper() if lower in {"v4"} else part.capitalize())
        else:
            parts.append(part)
    return " ".join(parts) or model


def short_text(value: Any, max_chars: int = 1200) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n... truncated {len(value) - max_chars} chars"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=out)


def load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def validate_base_url(base_url: str, *, allow_other_base_url: bool) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must include an http(s) scheme and host")
    host = (parsed.hostname or "").lower()
    if not allow_other_base_url and host not in ALLOWED_BASE_URL_HOSTS:
        allowed = ", ".join(sorted(ALLOWED_BASE_URL_HOSTS))
        raise ValueError(
            f"base URL host {host!r} is not allowed for the static report. "
            f"Allowed hosts: {allowed}."
        )
    return base_url.rstrip("/")


def error_kinds(record: dict[str, Any]) -> list[str]:
    return [str(err.get("kind") or "unknown") for err in record.get("errors", [])]


LOW_PRIORITY_ERROR_KINDS = {
    "http_error",
    "request_error",
    "response_json_decode",
}

ERROR_KIND_HELP = {
    "unexpected_finish_reason": (
        "The response ended as a normal answer instead of ending with tool_calls, "
        "so a client may not know to execute tools."
    ),
    "too_few_tool_calls": (
        "The response contained fewer structured tool calls than the test expected."
    ),
    "missing_expected_tool": (
        "A specific function was expected, but it was missing from the structured tool_calls list."
    ),
    "missing_expected_argument_fragment": (
        "The returned tool arguments or final text did not contain a value that the prompt requested."
    ),
    "missing_expected_argument_value": (
        "The returned tool arguments did not contain an exact value that the prompt requested."
    ),
    "tool_marker_leaked_to_content": (
        "Raw model tool-call syntax appeared in user-visible content instead of structured tool_calls."
    ),
    "tool_marker_leaked_to_reasoning": (
        "Raw model tool-call syntax appeared in reasoning_content instead of structured tool_calls."
    ),
    "context_leak_to_tool_arguments": (
        "A sentinel or unrelated context value appeared inside tool arguments."
    ),
    "context_leak_to_content": (
        "A sentinel or unrelated context value appeared in assistant-visible content."
    ),
    "context_leak_to_reasoning": (
        "A sentinel or unrelated context value appeared in reasoning_content."
    ),
    "agent_loop_missing_final_content": (
        "After tools were executed, the assistant did not produce the expected final answer."
    ),
    "missing_expected_final_content_fragment": (
        "The final answer omitted information that should have come from the tool results."
    ),
    "missing_content": (
        "The assistant response did not include the expected text content."
    ),
    "missing_expected_executed_tool": (
        "The multi-turn loop never executed one of the tools that the scenario required."
    ),
    "invalid_arguments_json": (
        "A structured tool call was returned, but its arguments could not be decoded as valid JSON."
    ),
    "malformed_tool_arguments_json": (
        "A tool call was present, but its arguments were not valid JSON."
    ),
    "malformed_tool_calls": "The tool_calls field had an unexpected shape.",
    "malformed_message": "The assistant message had an unexpected shape.",
    "malformed_choice": "The response choice had an unexpected shape.",
    "missing_choice": "The response did not include a usable choice.",
    "reasoning_mismatch": (
        "The same reasoning text was reported differently in two response fields."
    ),
    "http_error": (
        "The endpoint returned an HTTP error. This is useful operational signal, "
        "but it is lower priority than parser or tool-call contract failures."
    ),
    "request_error": (
        "The request failed before a usable model response was returned."
    ),
    "response_json_decode": ("The response body could not be parsed as JSON."),
}


def is_low_priority_kind(kind: str) -> bool:
    return kind in LOW_PRIORITY_ERROR_KINDS or kind.endswith("_timeout")


PRIORITY_RANK = {
    "high": 0,
    "low": 1,
    "very low": 2,
    "pass": 3,
}


def priority_rank(priority: str) -> int:
    return PRIORITY_RANK.get(priority, PRIORITY_RANK["high"])


def priority_css_class(priority: str) -> str:
    if priority == "high":
        return "bad"
    if priority == "very low":
        return "very-low"
    return "low"


def combined_priority(records: list[dict[str, Any]]) -> str:
    if not records:
        return "pass"
    return min((record_priority(record) for record in records), key=priority_rank)


def record_priority(record: dict[str, Any]) -> str:
    kinds = error_kinds(record)
    if not kinds:
        return "pass"
    if likely_source_bucket(record) == "model":
        return "very low"
    if all(is_low_priority_kind(kind) for kind in kinds):
        return "low"
    return "high"


def error_kind_help(kind: str) -> str:
    if kind.endswith("_timeout"):
        return "The request timed out before a usable model response was returned."
    return ERROR_KIND_HELP.get(
        kind,
        "The probe reported this failure kind. Open Details to inspect the raw request and response.",
    )


def unique_error_kinds(record: dict[str, Any]) -> list[str]:
    seen = set()
    unique = []
    for kind in error_kinds(record):
        if kind not in seen:
            seen.add(kind)
            unique.append(kind)
    return unique


def response_has_tool_marker(record: dict[str, Any]) -> bool:
    text = "\n".join(
        str(record.get(key) or "")
        for key in ("content", "reasoning_content", "raw_response")
    )
    markers = probe.RAW_TOOL_MARKERS
    return any(marker in text for marker in markers)


def likely_source(record: dict[str, Any]) -> tuple[str, str]:
    kinds = set(error_kinds(record))
    response = record.get("response") or {}
    tool_calls = response.get("tool_calls") or []

    if kinds and all(is_low_priority_kind(kind) for kind in kinds):
        return (
            "Endpoint/deployment",
            "No usable model response was returned, so this is not enough signal to blame Dynamo parsing or the engine.",
        )
    if (
        "tool_marker_leaked_to_content" in kinds
        or "tool_marker_leaked_to_reasoning" in kinds
    ):
        return (
            "Dynamo/parser likely",
            "The engine produced recognizable raw tool-call markup, but the response exposed it instead of converting it to structured tool_calls.",
        )
    if "unexpected_finish_reason" in kinds and tool_calls:
        return (
            "Dynamo/API formatting likely",
            "Structured tool_calls are present, but finish_reason stayed stop. The API response should report finish_reason=tool_calls.",
        )
    expected_tool_missing = {
        "missing_expected_tool",
        "missing_expected_executed_tool",
        "too_few_tool_calls",
    } & kinds
    if (
        expected_tool_missing
        and not tool_calls
        and not response_has_tool_marker(record)
    ):
        detail = (
            "The echo diagnostic asked for a simple tool call, but the response had no structured call and no raw tool-call markup."
            if "echo" in str(record.get("case_id") or "")
            else "The response had no structured tool call and no raw tool-call markup to parse."
        )
        return (
            "Engine/model behavior likely",
            f"{detail} This points to model generation, guided decoding, or tool_choice enforcement rather than Dynamo response parsing.",
        )
    if (
        "missing_expected_argument_fragment" in kinds
        or "missing_expected_argument_value" in kinds
    ) and tool_calls:
        return (
            "Engine/tool arguments likely",
            "A structured tool call was returned, but the generated arguments missed a value requested by the prompt.",
        )
    return (
        "Mixed/needs inspection",
        "The failure combines signals that need the raw response in Details to separate engine behavior from response postprocessing.",
    )


def likely_source_bucket(record: dict[str, Any]) -> str:
    source, _ = likely_source(record)
    if source.startswith("Dynamo"):
        return "dynamo"
    if source.startswith("Engine"):
        return "model"
    if source.startswith("Endpoint"):
        return "endpoint"
    return "mixed"


def explain_failure(record: dict[str, Any]) -> str:
    kinds = set(error_kinds(record))
    response = record.get("response") or {}
    finish = response.get("finish_reason")

    if kinds and all(is_low_priority_kind(kind) for kind in kinds):
        return (
            "The request did not reach a usable model response because the endpoint "
            "returned a transport or deployment error."
        )
    if "tool_marker_leaked_to_content" in kinds:
        return (
            "The model emitted raw tool-call markup as assistant content instead of "
            "returning a structured tool_calls object."
        )
    if "tool_marker_leaked_to_reasoning" in kinds:
        return (
            "Raw tool-call markup landed in reasoning_content, so the client "
            "did not receive the tool call as structured data."
        )
    if "missing_expected_tool" in kinds and "unexpected_finish_reason" in kinds:
        return (
            f"The test expected a tool call, but the response finished as {finish!r} "
            "and did not include the required function."
        )
    if "missing_expected_tool" in kinds:
        return (
            "The response did not include the function that this scenario needed "
            "the client to execute."
        )
    if (
        "missing_expected_argument_fragment" in kinds
        or "missing_expected_argument_value" in kinds
    ):
        return "The tool call or answer was missing a specific value requested by the prompt."
    if "agent_loop_missing_final_content" in kinds:
        return (
            "The first tool step may have happened, but the loop did not end with "
            "a usable final answer."
        )
    if "unexpected_finish_reason" in kinds:
        return (
            f"The response reported finish_reason={finish!r}, which does not match "
            "the tool-calling contract for this case."
        )

    first = next(iter(kinds), "unknown")
    return error_kind_help(first)


def public_record(record: dict[str, Any], *, detail_chars: int) -> dict[str, Any]:
    response = record.get("response") or {}
    out: dict[str, Any] = {
        "timestamp": record.get("timestamp"),
        "iteration": record.get("iteration"),
        "case_id": record.get("case_id"),
        "description": record.get("description"),
        "mode": record.get("mode"),
        "pass": bool(record.get("pass")),
        "priority": record_priority(record),
        "errors": record.get("errors") or [],
        "warnings": record.get("warnings") or [],
        "agent_loop": bool(record.get("agent_loop")),
        "response": {
            "id": response.get("id"),
            "model": response.get("model"),
            "finish_reason": response.get("finish_reason"),
            "latency_ms": response.get("latency_ms"),
            "ttft_ms": response.get("ttft_ms"),
            "chunk_count": response.get("chunk_count"),
            "content_chars": response.get("content_chars"),
            "reasoning_chars": response.get("reasoning_chars"),
            "tool_calls": response.get("tool_calls") or [],
            "usage": response.get("usage"),
        },
    }
    if not record.get("pass"):
        out["explanation"] = explain_failure(out)
        source, source_detail = likely_source(record)
        out["likely_source"] = source
        out["likely_source_detail"] = source_detail
    for key in (
        "request",
        "raw_response",
        "content",
        "reasoning_content",
        "final_messages",
        "turns",
        "executed_tool_calls",
    ):
        if key in record:
            out[key] = short_text(record[key], detail_chars)
    return out


def latency_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return round(ordered[index], 3)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    failed_records = [record for record in records if not record.get("pass")]
    high_priority_failed = [
        record for record in failed_records if record_priority(record) == "high"
    ]
    dynamo_errors = [
        record for record in failed_records if likely_source_bucket(record) == "dynamo"
    ]
    model_behavior_failures = [
        record for record in failed_records if likely_source_bucket(record) == "model"
    ]
    needs_inspection_failures = [
        record for record in failed_records if likely_source_bucket(record) == "mixed"
    ]
    endpoint_failures = [
        record
        for record in failed_records
        if likely_source_bucket(record) == "endpoint"
    ]
    latencies = [
        float((record.get("response") or {}).get("latency_ms"))
        for record in records
        if isinstance((record.get("response") or {}).get("latency_ms"), (int, float))
    ]
    return {
        "total": total,
        "passed": total - len(failed_records),
        "failed": len(failed_records),
        "high_priority_failed": len(high_priority_failed),
        "low_priority_failed": len(failed_records) - len(high_priority_failed),
        "dynamo_errors": len(dynamo_errors),
        "model_behavior_failures": len(model_behavior_failures),
        "needs_inspection_failures": len(needs_inspection_failures),
        "endpoint_failures": len(endpoint_failures),
        "non_dynamo_failures": len(failed_records) - len(dynamo_errors),
        "pass_rate": None
        if total == 0
        else round((total - len(failed_records)) / total, 4),
        "latency_ms_p50": latency_quantile(latencies, 0.50),
        "latency_ms_p95": latency_quantile(latencies, 0.95),
    }


def grouped_counts(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    totals = Counter(str(record.get(key) or "unknown") for record in records)
    failures = Counter(
        str(record.get(key) or "unknown")
        for record in records
        if not record.get("pass")
    )
    return [
        {
            key: name,
            "total": total,
            "passed": total - failures.get(name, 0),
            "failed": failures.get(name, 0),
        }
        for name, total in sorted(totals.items())
    ]


def failure_kind_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(kind for record in records for kind in error_kinds(record))
    rows = [
        {
            "kind": kind,
            "count": count,
            "priority": "low" if is_low_priority_kind(kind) else "high",
        }
        for kind, count in counts.items()
    ]
    return sorted(
        rows,
        key=lambda row: (
            1 if row["priority"] == "low" else 0,
            -int(row["count"]),
            str(row["kind"]),
        ),
    )


def _error_messages(record: dict[str, Any]) -> str:
    return " ".join(str(err.get("message") or "") for err in record.get("errors", []))


def _tool_names(record: dict[str, Any]) -> list[str]:
    names = []
    for call in (record.get("response") or {}).get("tool_calls") or []:
        name = str(call.get("name") or "")
        if name:
            names.append(name)
    return names


def failure_issue_theme(record: dict[str, Any]) -> tuple[str, str, str]:
    kinds = set(error_kinds(record))
    case_id = str(record.get("case_id") or "")
    finish = str((record.get("response") or {}).get("finish_reason") or "")
    messages = _error_messages(record)
    tool_names = _tool_names(record)

    if kinds and all(is_low_priority_kind(kind) for kind in kinds):
        if "timed out" in messages or "TimeoutError" in messages:
            return (
                "endpoint-timeout",
                "Request timed out before a usable response",
                "The endpoint did not return a response before the probe timeout. Treat this as serving/latency signal, not a parser-format failure.",
            )
        return (
            "endpoint-transport",
            "Endpoint or transport request failure",
            "The request did not reach a usable model response, so this is operational signal rather than model/tool-parser signal.",
        )
    if (
        "tool_marker_leaked_to_content" in kinds
        or "tool_marker_leaked_to_reasoning" in kinds
    ):
        return (
            "raw-tool-marker-leak",
            "Raw tool-call marker leaked outside structured tool_calls",
            "The response exposed model-native tool syntax in assistant content or reasoning instead of keeping it in structured tool_calls.",
        )
    if "context_leak_to_tool_arguments" in kinds or "context_leak_to_content" in kinds:
        return (
            "context-isolation-leak",
            "Prompt/context sentinel leaked into output",
            "A sentinel from the prompt or surrounding context appeared in tool arguments or assistant-visible text.",
        )
    if "unexpected_finish_reason" in kinds and tool_names:
        return (
            f"finish-reason-mismatch-{finish or 'unknown'}",
            "Structured tool call returned with wrong finish_reason",
            "The response contains parseable structured tool calls, but the final finish_reason does not tell clients to execute them.",
        )
    if (
        "missing_expected_argument_fragment" in kinds
        or "missing_expected_argument_value" in kinds
    ) and "delimiter" in case_id:
        return (
            "delimiter-string-argument",
            "Delimiter-looking text inside a string argument was not preserved",
            "The response produced a structured tool call, but text that resembles native tool syntax inside one string argument was dropped or rewritten.",
        )
    if (
        "missing_expected_argument_fragment" in kinds
        or "missing_expected_argument_value" in kinds
    ) and tool_names:
        return (
            f"tool-argument-missing-{'-'.join(sorted(set(tool_names))) or 'tool'}",
            "Tool arguments omitted expected values",
            "The model returned a structured tool call, but one or more prompt-required argument values were missing.",
        )
    if "invalid_arguments_json" in kinds or "malformed_tool_arguments_json" in kinds:
        return (
            "tool-arguments-invalid-json",
            "Tool arguments were not valid JSON",
            "A structured tool call was present, but the argument payload could not be decoded reliably by a client.",
        )
    if "missing_content" in kinds:
        return (
            "missing-assistant-content",
            "Expected assistant content was empty",
            "The scenario expected a plain assistant answer, but the response content was empty or missing.",
        )
    if "missing_expected_executed_tool" in kinds:
        return (
            "agent-loop-missing-tool-step",
            "Multi-step tool loop skipped an expected tool",
            "The first or later tool step did not execute, so the scenario could not reach the expected tool-backed answer.",
        )
    if "agent_loop_missing_final_content" in kinds:
        return (
            "agent-loop-missing-final-answer",
            "Multi-step tool loop did not produce a final answer",
            "Tool execution may have happened, but the assistant did not finish with the expected final content.",
        )
    if "missing_expected_tool" in kinds or "too_few_tool_calls" in kinds:
        return (
            "missing-expected-tool-call",
            "Expected tool call was not produced",
            "The model did not return the required function call for this scenario.",
        )
    if "unexpected_finish_reason" in kinds:
        return (
            f"unexpected-finish-{finish or 'unknown'}",
            "Unexpected finish_reason for scenario",
            "The response stopped with a finish_reason that does not match the OpenAI-compatible contract expected by this test.",
        )

    first = sorted(kinds)[0] if kinds else "unknown"
    return (
        f"other-{first}",
        error_kind_help(first),
        explain_failure(record),
    )


def build_failure_summary(failures: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in failures:
        key, title, explanation = failure_issue_theme(record)
        issue = grouped.setdefault(
            key,
            {
                "key": key,
                "title": title,
                "explanation": explanation,
                "failure_count": 0,
                "priority": "very low",
                "likely_sources": [],
                "error_kinds": [],
                "affected_cases": {},
                "examples": [],
            },
        )
        issue["failure_count"] += 1
        issue["priority"] = min(
            (issue["priority"], record_priority(record)),
            key=priority_rank,
        )

        source, _ = likely_source(record)
        if source not in issue["likely_sources"]:
            issue["likely_sources"].append(source)
        for kind in unique_error_kinds(record):
            if kind not in issue["error_kinds"]:
                issue["error_kinds"].append(kind)

        case_id = str(record.get("case_id") or "unknown")
        modes = issue["affected_cases"].setdefault(case_id, [])
        mode = str(record.get("mode") or "unknown")
        if mode not in modes:
            modes.append(mode)
        if len(issue["examples"]) < 3:
            issue["examples"].append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "finish_reason": (record.get("response") or {}).get(
                        "finish_reason"
                    ),
                    "tool_names": _tool_names(record),
                    "explanation": str(
                        record.get("explanation") or explain_failure(record)
                    ),
                }
            )

    issues = sorted(
        grouped.values(),
        key=lambda issue: (
            priority_rank(str(issue["priority"])),
            -int(issue["failure_count"]),
            str(issue["title"]),
        ),
    )
    return {
        "failure_count": len(failures),
        "issue_count": len(issues),
        "issues": issues,
    }


AI_ANALYSIS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "headline": {"type": "string"},
        "risk_level": {
            "type": "string",
            "enum": ["clear", "watch", "action_needed"],
        },
        "notes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                },
                "required": ["title", "body", "evidence"],
            },
        },
    },
    "required": ["headline", "risk_level", "notes"],
}


def compact_failure_for_analysis(
    record: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    response = record.get("response") or {}
    source, source_detail = likely_source(record)
    return {
        "case_id": record.get("case_id"),
        "description": record.get("description"),
        "mode": record.get("mode"),
        "priority": record_priority(record),
        "likely_source": display_ownership_text(report, source),
        "likely_source_bucket": likely_source_bucket(record),
        "likely_source_detail": display_ownership_text(report, source_detail),
        "error_kinds": unique_error_kinds(record),
        "finish_reason": response.get("finish_reason"),
        "tool_names": _tool_names(record),
        "explanation": str(record.get("explanation") or explain_failure(record)),
    }


def build_ai_analysis_context(report: dict[str, Any]) -> dict[str, Any]:
    failures = [
        record for record in report.get("failures") or [] if not record.get("pass")
    ]
    summary = report.get("summary") or {}
    owner = serving_owner(report)
    failure_summary = report.get("failure_summary") or build_failure_summary(failures)
    return {
        "page": {
            "title": report.get("title"),
            "model_label": report.get("model_label"),
            "model": report.get("model"),
            "model_slug": report.get("model_slug"),
            "base_url": report.get("base_url"),
            "generated_at": report.get("generated_at"),
            "run_id": report.get("run_id"),
            "serving_owner": owner,
            "serving_error_label": owner_error_label(report),
            "serving_api_label": owner_api_label(report),
        },
        "summary": {
            "total": summary.get("total"),
            "passed": summary.get("passed"),
            "failed": summary.get("failed"),
            "dynamo_errors": summary_dynamo_errors(summary),
            "serving_errors": summary_dynamo_errors(summary),
            "research_cases": summary_research_cases(summary),
            "inspect_cases": summary_needs_inspection(summary)
            + summary_endpoint_failures(summary),
            "pass_rate": summary.get("pass_rate"),
        },
        "failure_summary": ownership_display_copy(report, failure_summary),
        "failures": [
            compact_failure_for_analysis(record, report) for record in failures
        ],
    }


def _analysis_note(
    title: str,
    body: str,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "body": body,
        "evidence": [item for item in (evidence or []) if str(item).strip()],
    }


def _failure_labels(records: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    labels = []
    for record in records[:limit]:
        kinds = ", ".join(unique_error_kinds(record)) or "unknown"
        labels.append(
            f"{record.get('case_id') or 'unknown'} ({record.get('mode') or 'unknown'}): {kinds}"
        )
    remaining = len(records) - len(labels)
    if remaining > 0:
        labels.append(
            f"{remaining} additional failing request(s) covered in the table below"
        )
    return labels


def deterministic_ai_analysis(
    report: dict[str, Any],
    *,
    engine: str = "deterministic-classifier",
    status_detail: str = "",
) -> dict[str, Any]:
    owner = serving_owner(report)
    error_label = owner_error_label(report)
    api_label = owner_api_label(report)
    failures = [
        record for record in report.get("failures") or [] if not record.get("pass")
    ]
    summary = report.get("summary") or {}
    total = int(summary.get("total") or 0)
    failed = len(failures)
    dynamo_records = [
        record for record in failures if likely_source_bucket(record) == "dynamo"
    ]
    research_records = [
        record for record in failures if likely_source_bucket(record) == "model"
    ]
    inspect_records = [
        record
        for record in failures
        if likely_source_bucket(record) in {"mixed", "endpoint"}
    ]
    issues = (report.get("failure_summary") or build_failure_summary(failures)).get(
        "issues"
    ) or []

    if failed == 0:
        notes = [
            _analysis_note(
                "No failures found",
                "All requests on this result page passed the deterministic tool-calling contract.",
                [f"{total} total request(s) passed"],
            )
        ]
        headline = "No failing cases on this page."
        risk_level = "clear"
    else:
        notes = [
            _analysis_note(
                "Failure ownership summary",
                (
                    f"This page has {failed} failing request(s): "
                    f"{len(dynamo_records)} {error_label}, "
                    f"{len(research_records)} Research Cases, and "
                    f"{len(inspect_records)} Inspect cases."
                ),
                [
                    f"{len(issues)} grouped issue theme(s)",
                    f"{error_label} are the customer-facing regression count for {owner} Serve pages"
                    if owner == "vLLM"
                    else "Dynamo Errors are the customer-facing regression count",
                    f"Research and Inspect remain visible but are separated from {error_label}",
                ],
            )
        ]
        if dynamo_records:
            notes.append(
                _analysis_note(
                    f"Likely {api_label} signal",
                    (
                        "These failures have response-shape or parser-contract evidence "
                        f"that should be treated as likely {owner} ownership until disproven."
                    ),
                    _failure_labels(dynamo_records),
                )
            )
        if research_records:
            notes.append(
                _analysis_note(
                    f"Likely not {owner}",
                    (
                        "These failures look like model generation, tool-choice, or "
                        f"argument-selection behavior rather than a {owner} parser regression."
                    ),
                    _failure_labels(research_records),
                )
            )
        if inspect_records:
            notes.append(
                _analysis_note(
                    "Needs inspection",
                    (
                        "These records are ambiguous or operational. Keep them out of the "
                        f"{error_label} count until raw response or endpoint evidence assigns ownership."
                    ),
                    _failure_labels(inspect_records),
                )
            )
        for issue in issues[:2]:
            affected = issue.get("affected_cases") or {}
            cases = ", ".join(
                f"{case} ({'/'.join(str(mode) for mode in modes)})"
                for case, modes in affected.items()
            )
            notes.append(
                _analysis_note(
                    str(issue.get("title") or "Issue theme"),
                    str(issue.get("explanation") or ""),
                    [
                        f"{int(issue.get('failure_count') or 0)} occurrence(s)",
                        f"Priority: {issue.get('priority') or 'unknown'}",
                        f"Affects: {cases or 'unknown'}",
                    ],
                )
            )
        headline = f"{failed} failing request(s) across {len(issues)} issue theme(s)."
        risk_level = "action_needed" if dynamo_records else "watch"

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "engine": engine,
        "status_detail": status_detail,
        "headline": headline,
        "risk_level": risk_level,
        "input_failure_count": failed,
        "notes": notes[:6],
    }


def _coerce_ai_analysis(value: Any, *, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return fallback
    notes = value.get("notes")
    if not isinstance(notes, list) or not notes:
        return fallback
    clean_notes = []
    for note in notes[:6]:
        if not isinstance(note, dict):
            continue
        title = str(note.get("title") or "").strip()
        body = str(note.get("body") or "").strip()
        evidence = note.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        elif not isinstance(evidence, list):
            evidence = [evidence]
        if title and body:
            clean_notes.append(
                _analysis_note(title, body, [str(item) for item in evidence[:8]])
            )
    if not clean_notes:
        return fallback
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "engine": "codex-cli",
        "status_detail": "",
        "headline": str(
            value.get("headline") or fallback.get("headline") or ""
        ).strip(),
        "risk_level": str(
            value.get("risk_level") or fallback.get("risk_level") or "watch"
        ),
        "input_failure_count": fallback.get("input_failure_count", 0),
        "notes": clean_notes,
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def run_codex_ai_analysis(
    report: dict[str, Any],
    *,
    codex_binary: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    fallback = deterministic_ai_analysis(report)
    codex_path = shutil.which(codex_binary)
    if not codex_path:
        fallback[
            "status_detail"
        ] = "Codex CLI unavailable; deterministic fallback used."
        return fallback

    context = build_ai_analysis_context(report)
    owner = serving_owner(report)
    error_label = owner_error_label(report)
    api_label = owner_api_label(report)
    prompt = (
        "You are the AI analysis stage of a static tool-calling result dashboard.\n"
        "Analyze every failure record in the JSON below. Use only the provided facts.\n"
        "Do not run tools or shell commands; this is a read-only summarization step.\n"
        f"Be concise and customer-facing. For this page, bucket 'dynamo' means "
        f"{api_label} ownership because the serving owner is {owner}. "
        f"Do not call a failure a {owner} issue unless the likely_source_bucket is "
        f"'dynamo'. Keep Research and Inspect separate from {error_label}. "
        "Do not mention internal field names or bucket names such as likely_source_bucket. "
        "Return JSON matching the supplied schema only.\n\n"
        f"{json.dumps(context, ensure_ascii=False, sort_keys=True)}"
    )

    with tempfile.TemporaryDirectory(prefix="tool-calling-ai-") as tmpdir:
        tmp = Path(tmpdir)
        schema_path = tmp / "schema.json"
        output_path = tmp / "analysis.json"
        write_json(schema_path, AI_ANALYSIS_SCHEMA)
        cmd = [
            codex_path,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "-C",
            str(SCRIPT_DIR),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"warning: Codex CLI analysis unavailable: {exc}", file=sys.stderr)
            fallback[
                "status_detail"
            ] = "Codex CLI unavailable; deterministic fallback used."
            return fallback

        if completed.returncode != 0 or not output_path.exists():
            detail = (completed.stderr or completed.stdout or "").strip()
            if detail:
                print(
                    f"warning: Codex CLI analysis failed: {detail[:500]}",
                    file=sys.stderr,
                )
            fallback["status_detail"] = "Codex CLI failed; deterministic fallback used."
            return fallback
        try:
            parsed = _parse_json_object(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"warning: Codex CLI returned unparsable analysis: {exc}",
                file=sys.stderr,
            )
            fallback[
                "status_detail"
            ] = "Codex CLI returned unparsable analysis; deterministic fallback used."
            return fallback
    return _coerce_ai_analysis(parsed, fallback=fallback)


def build_ai_analysis(
    report: dict[str, Any],
    *,
    mode: str,
    codex_binary: str,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    if mode == "off":
        return None
    if not (report.get("failures") or []):
        return deterministic_ai_analysis(
            report,
            engine="static-analysis",
            status_detail="No failures were present, so Codex CLI analysis was not needed.",
        )
    if mode == "codex":
        return run_codex_ai_analysis(
            report,
            codex_binary=codex_binary,
            timeout_seconds=timeout_seconds,
        )
    return deterministic_ai_analysis(report)


def run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    load_env_file(Path(args.env_file).expanduser() if args.env_file else None)
    base_url = validate_base_url(
        args.base_url, allow_other_base_url=args.allow_other_base_url
    )

    case_profile = (
        probe.model_case_profile(args.model)
        if args.case_profile == "auto"
        else args.case_profile
    )
    cases = probe.select_cases(
        probe.build_cases(case_profile), args.cases, args.exclude_cases
    )
    modes = probe.parse_modes(args.modes)
    extra_headers = probe.parse_headers(args.header)
    url = probe.endpoint_url(base_url)
    run_id = f"{local_timestamp()}-{safe_slug(args.model.split('/')[-1])}"
    output_dir = Path(args.output_root).expanduser() / run_id
    started_mono = time.monotonic()

    config = {
        "base_url": base_url,
        "url": url,
        "model": args.model,
        "api_key_env": None if args.no_auth else args.api_key_env,
        "auth": "disabled" if args.no_auth else "bearer",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "modes": modes,
        "case_ids": [case.case_id for case in cases],
        "case_profile": case_profile,
        "exclude_cases": args.exclude_cases,
        "iterations": args.iterations,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "source": "tool_calling_static_report",
    }

    if args.dry_run:
        return {
            "run_id": run_id,
            "generated_at": utc_now(),
            "config": config,
            "summary": {"total": 0, "passed": 0, "failed": 0},
        }, []

    api_key = None if args.no_auth else os.environ.get(args.api_key_env)
    if not args.no_auth and not api_key:
        source = args.env_file or "the environment"
        raise RuntimeError(f"missing API key: set ${args.api_key_env} in {source}")

    random.seed(args.seed)
    writer = probe.ReportWriter(output_dir, config=config, cases=cases)
    try:
        for iteration in range(1, args.iterations + 1):
            work = [(case, mode) for case in cases for mode in modes]
            if args.shuffle:
                random.shuffle(work)

            def submit_one(item: tuple[probe.Case, str]) -> dict[str, Any]:
                case, mode = item
                if args.case_delay_seconds > 0:
                    time.sleep(args.case_delay_seconds)
                return probe.run_case(
                    case,
                    mode,
                    iteration=iteration,
                    url=url,
                    api_key=api_key,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout_seconds,
                    extra_headers=extra_headers,
                    raw_chars=args.raw_chars,
                    record_success_raw=args.record_success_raw,
                )

            if args.concurrency <= 1:
                for item in work:
                    record = submit_one(item)
                    writer.record(record)
                    probe.print_progress(record)
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.concurrency
                ) as executor:
                    futures = [executor.submit(submit_one, item) for item in work]
                    for future in concurrent.futures.as_completed(futures):
                        record = future.result()
                        writer.record(record)
                        probe.print_progress(record)
            writer.write_summary()
    finally:
        writer.write_summary()
        writer.close()

    finished_at = utc_now()
    records = [
        public_record(record, detail_chars=args.detail_chars)
        for record in writer.records
    ]
    failures = [record for record in records if not record["pass"]]
    report = {
        "schema_version": 1,
        "title": args.title,
        "run_id": run_id,
        "generated_at": finished_at,
        "started_at": writer.started,
        "duration_seconds": round(time.monotonic() - started_mono, 3),
        "model": args.model,
        "base_url": base_url,
        "endpoint": url,
        "output_dir": str(output_dir),
        "config": config,
        "summary": summarize(records),
        "by_mode": grouped_counts(records, "mode"),
        "by_case": grouped_counts(records, "case_id"),
        "failure_kinds": failure_kind_counts(records),
        "failures": failures,
        "failure_summary": build_failure_summary(failures),
        "records": records,
    }
    if args.startup_command:
        report["startup_command"] = args.startup_command
    if args.startup_command_file:
        report["startup_command"] = (
            Path(args.startup_command_file)
            .expanduser()
            .read_text(encoding="utf-8")
            .strip()
        )
    if args.startup_command_source:
        report["startup_command_source"] = args.startup_command_source
    return report, records


def status_class(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    if summary.get("dynamo_errors", summary.get("high_priority_failed", 0)):
        return "fail"
    if summary.get("needs_inspection_failures", 0) or summary.get(
        "endpoint_failures", 0
    ):
        return "warn"
    return "pass"


def summary_dynamo_errors(summary: dict[str, Any]) -> int:
    if "dynamo_errors" in summary:
        return int(summary.get("dynamo_errors") or 0)
    return int(summary.get("high_priority_failed") or 0)


def summary_research_cases(summary: dict[str, Any]) -> int:
    if "model_behavior_failures" in summary:
        return int(summary.get("model_behavior_failures") or 0)
    failed = int(summary.get("failed") or 0)
    return max(0, failed - summary_dynamo_errors(summary))


def summary_needs_inspection(summary: dict[str, Any]) -> int:
    return int(summary.get("needs_inspection_failures") or 0)


def summary_endpoint_failures(summary: dict[str, Any]) -> int:
    if "endpoint_failures" in summary:
        return int(summary.get("endpoint_failures") or 0)
    return int(summary.get("low_priority_failed") or 0)


def is_vllm_report(report: dict[str, Any]) -> bool:
    text = " ".join(
        str(report.get(key) or "")
        for key in ("title", "model_label", "model_slug", "runtime")
    )
    deployment = report.get("deployment_info") or {}
    if isinstance(deployment, dict):
        text += " " + " ".join(
            str(deployment.get(key) or "")
            for key in ("serving", "runtime", "vllm_version", "startup_command")
        )
    return "vllm" in text.lower()


def serving_owner(report: dict[str, Any]) -> str:
    return "vLLM" if is_vllm_report(report) else "Dynamo"


def owner_error_label(report: dict[str, Any]) -> str:
    return f"{serving_owner(report)} Errors"


def owner_clear_label(report: dict[str, Any]) -> str:
    return f"{serving_owner(report)} Clear"


def owner_api_label(report: dict[str, Any]) -> str:
    return f"{serving_owner(report)}/API/parser"


def display_ownership_text(report: dict[str, Any], value: Any) -> str:
    text = str(value or "")
    if serving_owner(report) == "Dynamo":
        return text
    replacements = (
        ("Dynamo/API/parser", "vLLM/API/parser"),
        ("Dynamo/API formatting", "vLLM/API formatting"),
        ("Dynamo/API ", "vLLM/API "),
        ("Dynamo-formatting", "vLLM-formatting"),
        ("Dynamo XML", "vLLM XML"),
        ("Dynamo/parser", "vLLM/parser"),
        ("Dynamo response parsing", "vLLM response handling"),
        ("Dynamo parser", "vLLM parser"),
        ("Dynamo regressions", "vLLM regressions"),
        ("Dynamo regression", "vLLM regression"),
        ("Dynamo Error count", "vLLM Error count"),
        ("Dynamo Errors", "vLLM Errors"),
        ("Dynamo Error", "vLLM Error"),
        ("Dynamo errors", "vLLM errors"),
        ("Dynamo error", "vLLM error"),
        ("Dynamo ownership", "vLLM ownership"),
        ("Dynamo issue", "vLLM issue"),
        ("Dynamo issues", "vLLM issues"),
        ("not Dynamo", "not vLLM"),
        ("summary.dynamo_errors", "summary.serving_errors"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def ownership_display_copy(report: dict[str, Any], value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ownership_display_copy(report, item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [ownership_display_copy(report, item) for item in value]
    if isinstance(value, str):
        return display_ownership_text(report, value)
    return value


def render_metric(label: str, value: Any, tone: str = "") -> str:
    return (
        f'<div class="metric {tone}">'
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(str(value))}</strong>"
        "</div>"
    )


def render_deployment_metrics(report: dict[str, Any]) -> str:
    deployment = report.get("deployment_info") or {}
    if not isinstance(deployment, dict):
        return ""
    metrics = []
    if deployment.get("vllm_version"):
        metrics.append(render_metric("vLLM Version", deployment["vllm_version"]))
    return "".join(metrics)


def normalize_startup_command(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(part) for part in value).strip()
    if isinstance(value, dict):
        for key in ("startup_command", "command", "args"):
            command = normalize_startup_command(value.get(key))
            if command:
                return command
        return json.dumps(value, indent=2, sort_keys=True)
    return str(value).strip()


def startup_command_from_report(report: dict[str, Any]) -> str:
    deployment = report.get("deployment_info") or {}
    if not isinstance(deployment, dict):
        deployment = {}
    for value in (
        report.get("startup_command"),
        deployment.get("startup_command"),
        deployment.get("command"),
        deployment.get("args"),
    ):
        command = normalize_startup_command(value)
        if command:
            return command
    return ""


def startup_command_source(report: dict[str, Any]) -> str:
    deployment = report.get("deployment_info") or {}
    if not isinstance(deployment, dict):
        deployment = {}
    return str(
        report.get("startup_command_source")
        or deployment.get("startup_command_source")
        or deployment.get("source")
        or ""
    ).strip()


def extract_vllm_serve_command(command: str) -> str:
    lines = command.splitlines()
    for index, line in enumerate(lines):
        if "vllm serve " not in line:
            continue
        extracted = []
        for candidate in lines[index:]:
            stripped = candidate.strip()
            if not stripped:
                break
            extracted.append(stripped)
            if not stripped.endswith("\\"):
                break
        return "\n".join(extracted).strip()
    return ""


def vllm_serve_command_from_report(report: dict[str, Any]) -> str:
    deployment = report.get("deployment_info") or {}
    if not isinstance(deployment, dict):
        deployment = {}
    for value in (
        report.get("vllm_serve_command"),
        deployment.get("vllm_serve_command"),
    ):
        command = normalize_startup_command(value)
        if command:
            return command
    return extract_vllm_serve_command(startup_command_from_report(report))


def vllm_serve_command_source(report: dict[str, Any]) -> str:
    deployment = report.get("deployment_info") or {}
    if not isinstance(deployment, dict):
        deployment = {}
    return str(
        report.get("vllm_serve_command_source")
        or deployment.get("vllm_serve_command_source")
        or startup_command_source(report)
    ).strip()


def render_vllm_serve_command(report: dict[str, Any]) -> str:
    command = vllm_serve_command_from_report(report)
    if not command:
        return ""
    source = vllm_serve_command_source(report)
    source_html = (
        f'<div class="subtle">Source: {html.escape(source)}</div>' if source else ""
    )
    return (
        '<section class="startup-command vllm-serve-command">'
        "<h2>vLLM Serve Command</h2>"
        f"{source_html}"
        f"<pre>{html.escape(command)}</pre>"
        "</section>"
    )


def render_startup_command(report: dict[str, Any]) -> str:
    command = startup_command_from_report(report)
    if not command:
        return ""
    source = startup_command_source(report)
    source_html = (
        f'<div class="subtle">Source: {html.escape(source)}</div>' if source else ""
    )
    return (
        '<section class="startup-command">'
        "<h2>Startup Command</h2>"
        f"{source_html}"
        f"<pre>{html.escape(command)}</pre>"
        "</section>"
    )


def render_error_kind(kind: str) -> str:
    priority = "low" if is_low_priority_kind(kind) else "high"
    return (
        f'<span class="kind-pill {priority}" '
        f'tabindex="0" '
        f'data-tooltip="{html.escape(error_kind_help(kind), quote=True)}" '
        'aria-label="Failure kind explanation">'
        f"<code>{html.escape(kind)}</code>"
        "</span>"
    )


def render_error_kinds(kinds: list[str]) -> str:
    if not kinds:
        return '<span class="subtle">none</span>'
    return f'<div class="kind-list">{"".join(render_error_kind(kind) for kind in kinds)}</div>'


def render_source_badge(
    record: dict[str, Any], report: dict[str, Any] | None = None
) -> str:
    source, detail = likely_source(record)
    display_source = display_ownership_text(report or {}, source)
    display_detail = display_ownership_text(report or {}, detail)
    source_class = "source-mixed"
    if source.startswith("Dynamo"):
        source_class = "source-dynamo"
    elif source.startswith("Engine"):
        source_class = "source-engine"
    elif source.startswith("Endpoint"):
        source_class = "source-endpoint"
    return (
        f'<span class="source-badge {source_class}" tabindex="0" '
        f'data-tooltip="{html.escape(display_detail, quote=True)}" '
        'aria-label="Likely failure source explanation">'
        f"{html.escape(display_source)}"
        "</span>"
    )


def parse_model_link(value: str) -> dict[str, str]:
    parts = [part.strip() for part in value.split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError("--model-link must be formatted as 'slug|Label|/href/'")
    return {"slug": parts[0], "label": parts[1], "href": parts[2]}


def normalize_model_links(
    links: list[dict[str, str]],
    *,
    active_slug: str,
    active_label: str,
) -> list[dict[str, str]]:
    if not links:
        return [{"slug": active_slug, "label": active_label, "href": "./"}]
    seen = {link["slug"] for link in links}
    if active_slug not in seen:
        links = [
            {"slug": active_slug, "label": active_label, "href": "./"},
            *links,
        ]
    return links


def split_model_group(label: str) -> tuple[str | None, str]:
    if " · " not in label:
        return None, label
    group, display = label.split(" · ", 1)
    return group.strip() or None, display.strip() or label


def render_model_switcher(
    links: list[dict[str, str]],
    *,
    active_slug: str,
) -> str:
    return (
        '<div class="links summary-link"><a href="/models/summary/">Summary</a></div>'
    )


def render_count_table(
    rows: list[dict[str, Any]], name_key: str, name_label: str
) -> str:
    body = []
    for row in rows:
        failed = int(row.get("failed") or 0)
        cls = "bad" if failed else "ok"
        body.append(
            "<tr>"
            f"<td><code>{html.escape(str(row.get(name_key)))}</code></td>"
            f"<td>{row.get('total')}</td>"
            f"<td>{row.get('passed')}</td>"
            f'<td class="{cls}">{failed}</td>'
            "</tr>"
        )
    return (
        "<table>"
        f"<thead><tr><th>{html.escape(name_label)}</th><th>Total</th>"
        "<th>Passed</th><th>Failed</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def render_failure_kind_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">none</div>'
    body = []
    for row in rows:
        priority = str(row.get("priority") or "high")
        count_class = "low" if priority == "low" else "bad"
        kind = str(row.get("kind") or "unknown")
        body.append(
            "<tr>"
            f"<td>{render_error_kind(kind)}</td>"
            f"<td>{html.escape(priority)}</td>"
            f'<td class="{count_class}">{int(row.get("count") or 0)}</td>'
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Kind</th><th>Priority</th><th>Count</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def render_current_findings(report: dict[str, Any]) -> str:
    failures = report.get("failures") or []
    high_failures = [record for record in failures if record_priority(record) == "high"]
    low_failures = [record for record in failures if record_priority(record) != "high"]
    summary = report.get("summary") or {}
    pass_rate = summary.get("pass_rate")
    pass_rate_text = "n/a" if pass_rate is None else f"{pass_rate * 100:.1f}%"
    owner = serving_owner(report)
    error_label = owner_error_label(report)
    api_label = owner_api_label(report)

    def issue_table(records: list[dict[str, Any]], empty_text: str) -> str:
        if not records:
            return f'<div class="empty">{html.escape(empty_text)}</div>'
        rows = []
        for record in sorted(
            records,
            key=lambda item: (
                str(item.get("case_id") or ""),
                str(item.get("mode") or ""),
            ),
        ):
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(str(record.get('case_id') or 'unknown'))}</code>"
                f'<div class="subtle">{html.escape(str(record.get("description") or ""))}</div></td>'
                f"<td><code>{html.escape(str(record.get('mode') or ''))}</code></td>"
                f"<td>{html.escape(str(record.get('explanation') or explain_failure(record)))}</td>"
                f"<td>{render_error_kinds(unique_error_kinds(record))}</td>"
                f"<td>{render_source_badge(record, report)}</td>"
                "</tr>"
            )
        return (
            "<table><thead><tr><th>Case</th><th>Mode</th><th>Short Failure</th>"
            "<th>Kinds</th><th>Likely Source</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    dynamo_failures = [
        record for record in high_failures if likely_source_bucket(record) == "dynamo"
    ]
    model_failures = [
        record for record in failures if likely_source_bucket(record) == "model"
    ]
    mixed_failures = [
        record for record in high_failures if likely_source_bucket(record) == "mixed"
    ]
    endpoint_failures = [
        record for record in failures if likely_source_bucket(record) == "endpoint"
    ]
    return (
        '<section class="findings">'
        "<h2>Current Findings</h2>"
        "<table>"
        "<tbody>"
        f"<tr><th>Semantic failures</th><td>{len(high_failures)}</td></tr>"
        f"<tr><th>HTTP/transport failures</th><td>{len(low_failures)}</td></tr>"
        f"<tr><th>Pass rate</th><td>{html.escape(pass_rate_text)}</td></tr>"
        f"<tr><th>Focus</th><td>Separate likely {html.escape(owner)}/serving regressions from likely model/engine behavior. Unclear cases stay in needs-inspection.</td></tr>"
        "</tbody>"
        "</table>"
        f"<h3>Potential {html.escape(error_label)} / Serving Issues</h3>"
        f"<p>These are failures where the model response appears to contain parseable tool-call structure, but the {html.escape(api_label)} response shape is wrong.</p>"
        f"{issue_table(dynamo_failures, f'No likely {api_label} issues in this run.')}"
        "<h3>Likely Model / Engine Issues</h3>"
        "<p>These are failures where the model did not produce the requested tool behavior, or generated incomplete tool arguments.</p>"
        f"{issue_table(model_failures, 'No likely model or engine issues in this run.')}"
        "<h3>Needs Inspection</h3>"
        "<p>These failures need raw request/response inspection before assigning ownership.</p>"
        f"{issue_table(mixed_failures, 'No ambiguous high-priority failures in this run.')}"
        "<h3>Endpoint / Transport Issues</h3>"
        f"{issue_table(endpoint_failures, 'No endpoint or transport failures in this run.')}"
        "</section>"
    )


def render_ai_analysis(report: dict[str, Any]) -> str:
    analysis = report.get("ai_analysis") or report.get("analysis_notes") or []
    headline = ""
    engine = ""
    if isinstance(analysis, dict):
        headline = str(analysis.get("headline") or "")
        engine = str(analysis.get("engine") or "")
        notes = analysis.get("notes") or []
    else:
        notes = analysis
    if isinstance(notes, str):
        notes = [{"title": "Analysis", "body": notes}]
    if not isinstance(notes, list) or not notes:
        return ""

    rows = []
    for note in notes:
        if isinstance(note, str):
            title = "Analysis"
            body = note
            evidence = []
        elif isinstance(note, dict):
            title = str(note.get("title") or "Analysis")
            body = str(note.get("body") or note.get("summary") or "")
            evidence = note.get("evidence") or []
        else:
            continue
        title = display_ownership_text(report, title)
        body = display_ownership_text(report, body)
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence_items = "".join(
            f"<li>{html.escape(display_ownership_text(report, item))}</li>"
            for item in evidence
            if str(item).strip()
        )
        evidence_html = f"<ul>{evidence_items}</ul>" if evidence_items else ""
        rows.append(
            '<div class="analysis-note">'
            f"<h3>{html.escape(title)}</h3>"
            f"<p>{html.escape(body)}</p>"
            f"{evidence_html}"
            "</div>"
        )
    if not rows:
        return ""
    meta = ""
    if headline or engine:
        meta_parts = []
        if headline:
            meta_parts.append(html.escape(display_ownership_text(report, headline)))
        if engine:
            meta_parts.append(f"engine: {html.escape(engine)}")
        meta = f'<p class="analysis-meta">{" | ".join(meta_parts)}</p>'
    return (
        '<section class="ai-analysis">'
        "<h2>AI Analysis</h2>"
        f"{meta}"
        f"{''.join(rows)}"
        "</section>"
    )


def render_failure_rows(failures: list[dict[str, Any]], report: dict[str, Any]) -> str:
    if not failures:
        return (
            '<tr><td colspan="8" class="empty">No failing cases in this run.</td></tr>'
        )
    rows = []
    sorted_failures = sorted(
        failures,
        key=lambda record: (
            priority_rank(record_priority(record)),
            str(record.get("case_id") or ""),
            str(record.get("mode") or ""),
        ),
    )
    for record in sorted_failures:
        response = record.get("response") or {}
        priority = record_priority(record)
        priority_class = priority_css_class(priority)
        explanation = display_ownership_text(
            report, str(record.get("explanation") or explain_failure(record))
        )
        detail = json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True)
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(record.get('case_id')))}</code>"
            f'<div class="subtle">{html.escape(str(record.get("description") or ""))}</div></td>'
            f"<td><code>{html.escape(str(record.get('mode')))}</code></td>"
            f'<td class="{priority_class}">{html.escape(priority)}</td>'
            f"<td>{html.escape(explanation)}</td>"
            f"<td>{render_source_badge(record, report)}</td>"
            f"<td>{render_error_kinds(unique_error_kinds(record))}</td>"
            f"<td><code>{html.escape(str(response.get('finish_reason')))}</code></td>"
            "<td>"
            "<details><summary>Details</summary>"
            f"<pre>{html.escape(detail)}</pre>"
            "</details>"
            "</td>"
            "</tr>"
        )
    return "".join(rows)


def detail_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: detail_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detail_value(item) for item in value]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in '{["':
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def render_detail_pre(value: Any) -> str:
    normalized = detail_value(value)
    return html.escape(
        json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=True)
    )


def render_record_rows(records: list[dict[str, Any]]) -> str:
    if not records:
        return '<tr><td colspan="7" class="empty">No request records in this run.</td></tr>'
    rows = []
    sorted_records = sorted(
        records,
        key=lambda record: (
            str(record.get("case_id") or ""),
            str(record.get("mode") or ""),
        ),
    )
    for record in sorted_records:
        response = record.get("response") or {}
        passed = bool(record.get("pass"))
        result_class = "ok" if passed else "bad"
        result_label = "PASS" if passed else "FAIL"
        request_detail = record.get("request") or {
            "note": "Request body was not retained for this run. Future runs retain it.",
        }
        response_detail = {
            "pass": passed,
            "errors": record.get("errors") or [],
            "warnings": record.get("warnings") or [],
            "response": response,
        }
        for key in (
            "content",
            "reasoning_content",
            "raw_response",
            "turns",
            "executed_tool_calls",
            "final_messages",
        ):
            if key in record:
                response_detail[key] = record[key]
        tool_names = [
            str(call.get("name") or "<unknown>")
            for call in response.get("tool_calls") or []
        ]
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(record.get('case_id')))}</code>"
            f'<div class="subtle">{html.escape(str(record.get("description") or ""))}</div></td>'
            f"<td><code>{html.escape(str(record.get('mode')))}</code></td>"
            f'<td class="{result_class}">{result_label}</td>'
            f"<td><code>{html.escape(str(response.get('finish_reason')))}</code></td>"
            f"<td>{html.escape(', '.join(tool_names) if tool_names else 'none')}</td>"
            '<td class="record-details">'
            "<details><summary>Request</summary>"
            f"<pre>{render_detail_pre(request_detail)}</pre>"
            "</details>"
            "<details><summary>Response</summary>"
            f"<pre>{render_detail_pre(response_detail)}</pre>"
            "</details>"
            "</td>"
            "</tr>"
        )
    return "".join(rows)


def mode_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    mode = str(record.get("mode") or "")
    order = {"nonstream": 0, "stream": 1}
    return (order.get(mode, 99), mode)


def grouped_case_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("case_id") or "unknown")].append(record)
    rows = []
    for case_id, case_records in groups.items():
        case_records = sorted(case_records, key=mode_sort_key)
        failed_records = [record for record in case_records if not record.get("pass")]
        rows.append(
            {
                "case_id": case_id,
                "description": str(case_records[0].get("description") or ""),
                "records": case_records,
                "failed_records": failed_records,
                "failed": bool(failed_records),
                "priority": combined_priority(failed_records),
            }
        )
    return sorted(
        rows,
        key=lambda group: (
            priority_rank(str(group["priority"]))
            if group["failed"]
            else PRIORITY_RANK["pass"],
            0 if group["failed"] else 1,
            str(group["case_id"]),
        ),
    )


def record_tool_names(record: dict[str, Any]) -> list[str]:
    response = record.get("response") or {}
    return [
        str(call.get("name") or "<unknown>")
        for call in response.get("tool_calls") or []
    ]


def render_mode_summary(records: list[dict[str, Any]]) -> str:
    chips = []
    for record in sorted(records, key=mode_sort_key):
        response = record.get("response") or {}
        passed = bool(record.get("pass"))
        result_class = "ok" if passed else "bad"
        result_label = "PASS" if passed else "FAIL"
        finish = str(response.get("finish_reason"))
        tool_names = record_tool_names(record)
        tool_text = ", ".join(tool_names) if tool_names else "none"
        chips.append(
            f'<div class="mode-chip {result_class}">'
            f"<div><code>{html.escape(str(record.get('mode') or 'unknown'))}</code> "
            f"<strong>{result_label}</strong></div>"
            f'<div class="subtle">finish={html.escape(finish)} · tools={html.escape(tool_text)}</div>'
            "</div>"
        )
    return f'<div class="mode-stack">{"".join(chips)}</div>'


def response_detail_for_record(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response") or {}
    response_detail = {
        "pass": bool(record.get("pass")),
        "errors": record.get("errors") or [],
        "warnings": record.get("warnings") or [],
        "response": response,
    }
    for key in (
        "content",
        "reasoning_content",
        "raw_response",
        "turns",
        "executed_tool_calls",
        "final_messages",
    ):
        if key in record:
            response_detail[key] = record[key]
    return response_detail


def raw_response_for_record(record: dict[str, Any]) -> Any:
    if "raw_response" in record:
        return record.get("raw_response")
    return record.get("response") or {
        "note": "Raw response body was not retained for this run. Future runs retain it.",
    }


def render_grouped_request_details(records: list[dict[str, Any]]) -> str:
    parts = []
    for record in sorted(records, key=mode_sort_key):
        mode = str(record.get("mode") or "unknown")
        request_detail = record.get("request") or {
            "note": "Request body was not retained for this run. Future runs retain it.",
        }
        result = "PASS" if record.get("pass") else "FAIL"
        parts.append(
            "<details>"
            f"<summary>{html.escape(mode)} request · {result}</summary>"
            f"<pre>{render_detail_pre(request_detail)}</pre>"
            "</details>"
        )
    return f'<div class="record-details">{"".join(parts)}</div>'


def render_grouped_response_details(records: list[dict[str, Any]]) -> str:
    parts = []
    for record in sorted(records, key=mode_sort_key):
        mode = str(record.get("mode") or "unknown")
        result = "PASS" if record.get("pass") else "FAIL"
        finish = str((record.get("response") or {}).get("finish_reason"))
        parts.append(
            "<details>"
            f"<summary>{html.escape(mode)} response · {result} · finish={html.escape(finish)}</summary>"
            f"<pre>{render_detail_pre(raw_response_for_record(record))}</pre>"
            "</details>"
        )
    return f'<div class="record-details">{"".join(parts)}</div>'


def render_group_failure_summary(group: dict[str, Any], report: dict[str, Any]) -> str:
    failed_records = group["failed_records"]
    items = []
    for record in failed_records:
        mode = str(record.get("mode") or "unknown")
        explanation = str(record.get("explanation") or explain_failure(record))
        kinds = ", ".join(unique_error_kinds(record)) or "unknown"
        source, _ = likely_source(record)
        source = display_ownership_text(report, source)
        explanation = display_ownership_text(report, explanation)
        items.append(
            f"<div><code>{html.escape(mode)}</code>: "
            f"{html.escape(explanation)}"
            f'<div class="subtle">{html.escape(source)} · {html.escape(kinds)}</div></div>'
        )
    return "".join(items) if items else '<span class="subtle">none</span>'


def render_group_sources(group: dict[str, Any]) -> str:
    failed_records = group["failed_records"]
    if not failed_records:
        return '<span class="subtle">none</span>'
    seen = set()
    badges = []
    for record in failed_records:
        source, _ = likely_source(record)
        key = (str(record.get("mode") or ""), source)
        if key in seen:
            continue
        seen.add(key)
        badges.append(render_source_badge(record))
    return f'<div class="kind-list">{"".join(badges)}</div>'


def render_failure_issue_summary(report: dict[str, Any]) -> str:
    summary = report.get("failure_summary") or build_failure_summary(
        report.get("failures") or []
    )
    issues = summary.get("issues") or []
    failure_count = int(summary.get("failure_count") or 0)
    issue_count = int(summary.get("issue_count") or 0)
    if not issues:
        return (
            "<section>"
            "<h2>Case Notes</h2>"
            '<p class="subtle">No non-passing cases to cluster in this run.</p>'
            "</section>"
        )

    rows = []
    for index, issue in enumerate(issues, start=1):
        affected = []
        for case_id, modes in (issue.get("affected_cases") or {}).items():
            mode_text = ", ".join(str(mode) for mode in modes)
            affected.append(f"{case_id} ({mode_text})")
        source_text = ", ".join(
            display_ownership_text(report, source)
            for source in issue.get("likely_sources") or []
        )
        explanation = display_ownership_text(
            report, str(issue.get("explanation") or "")
        )
        kind_text = ", ".join(str(kind) for kind in issue.get("error_kinds") or [])
        priority = str(issue.get("priority") or "unknown")
        priority_class = priority_css_class(priority)
        rows.append(
            "<tr>"
            f"<td><strong>{index}. {html.escape(str(issue.get('title') or 'Failure issue'))}</strong>"
            f'<div class="subtle">{html.escape(explanation)}</div></td>'
            f'<td class="metric-count">{int(issue.get("failure_count") or 0)}</td>'
            f'<td class="{priority_class}">{html.escape(priority)}</td>'
            f"<td>{html.escape('; '.join(affected))}</td>"
            f"<td><div>{html.escape(source_text or 'unknown')}</div>"
            f'<div class="subtle">{html.escape(kind_text or "unknown")}</div></td>'
            "</tr>"
        )
    return (
        "<section>"
        "<h2>Case Notes</h2>"
        f'<p class="subtle">{failure_count} non-passing request(s) collapse into '
        f"{issue_count} likely note(s). Only cases classified as {owner_api_label(report)} "
        "issues count as dashboard errors; model-behavior cases stay visible here "
        "for follow-up.</p>"
        '<table class="issue-summary-table">'
        "<thead><tr><th>Likely Issue</th><th>Occurrences</th><th>Priority</th>"
        "<th>Affects</th><th>Likely Source / Kinds</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</section>"
    )


def filtered_failure_groups(
    groups: list[dict[str, Any]], buckets: set[str]
) -> list[dict[str, Any]]:
    filtered = []
    for group in groups:
        failed_records = [
            record
            for record in group["failed_records"]
            if likely_source_bucket(record) in buckets
        ]
        if not failed_records:
            continue
        priority = combined_priority(failed_records)
        filtered.append(
            {
                **group,
                "failed_records": failed_records,
                "failed": True,
                "priority": priority,
            }
        )
    return filtered


def render_management_failure_rows(
    groups: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    buckets: set[str] | None = None,
    empty_text: str = "No failing cases in this run.",
) -> str:
    failure_groups = (
        filtered_failure_groups(groups, buckets)
        if buckets
        else [group for group in groups if group["failed"]]
    )
    if not failure_groups:
        return f'<tr><td colspan="6" class="empty">{html.escape(empty_text)}</td></tr>'
    rows = []
    for index, group in enumerate(failure_groups):
        priority = str(group["priority"])
        priority_class = priority_css_class(priority)
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(group['case_id']))}</code>"
            f'<div class="subtle">{html.escape(str(group["description"]))}</div></td>'
            f"<td>{render_mode_summary(group['records'])}</td>"
            f'<td class="{priority_class}">{html.escape(priority)}</td>'
            f"<td>{render_group_failure_summary(group, report)}</td>"
            f"<td>{render_grouped_request_details(group['records'])}</td>"
            f"<td>{render_grouped_response_details(group['records'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_management_success_rows(groups: list[dict[str, Any]]) -> str:
    success_groups = [group for group in groups if not group["failed"]]
    if not success_groups:
        return (
            '<tr><td colspan="4" class="empty">No passing cases in this run.</td></tr>'
        )
    rows = []
    for group in success_groups:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(group['case_id']))}</code>"
            f'<div class="subtle">{html.escape(str(group["description"]))}</div></td>'
            f"<td>{render_mode_summary(group['records'])}</td>"
            f"<td>{render_grouped_request_details(group['records'])}</td>"
            f"<td>{render_grouped_response_details(group['records'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_history(history: list[dict[str, Any]], *, root_prefix: str) -> str:
    rows = []
    for item in history[:20]:
        summary = item.get("summary") or {}
        dynamo_errors = summary_dynamo_errors(summary)
        research_cases = summary_research_cases(summary)
        inspect_cases = summary_needs_inspection(summary) + summary_endpoint_failures(
            summary
        )
        cls = "bad" if dynamo_errors else "ok"
        rows.append(
            "<tr>"
            f'<td><a href="{root_prefix}runs/{html.escape(str(item.get("run_id")))}/index.html">'
            f"{html.escape(str(item.get('generated_at')))}</a></td>"
            f"<td><code>{html.escape(str(item.get('model')))}</code></td>"
            f"<td>{summary.get('total')}</td>"
            f'<td class="{cls}">{dynamo_errors}</td>'
            f"<td>{research_cases}</td>"
            f"<td>{inspect_cases}</td>"
            f"<td>{item.get('duration_seconds')}s</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="7" class="empty">No prior runs.</td></tr>')
    return (
        "<table><thead><tr><th>Run</th><th>Model</th><th>Total</th>"
        "<th>Serving Errors</th><th>Research Cases</th>"
        "<th>Inspect (ownership unclear or operational issue)</th>"
        "<th>Duration</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(
    report: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    history_prefix: str,
    model_links: list[dict[str, str]] | None = None,
    active_model_slug: str | None = None,
) -> str:
    summary = report["summary"]
    active_model_slug = active_model_slug or str(
        report.get("model_slug")
        or model_slug_from_model(str(report.get("model") or "model"))
    )
    model_label = str(
        report.get("model_label")
        or model_label_from_model(str(report.get("model") or "model"))
    )
    model_links = normalize_model_links(
        model_links or [],
        active_slug=active_model_slug,
        active_label=model_label,
    )
    dynamo_errors = summary_dynamo_errors(summary)
    model_research_cases = summary_research_cases(summary)
    needs_inspection = summary_needs_inspection(summary)
    endpoint_failures = summary_endpoint_failures(summary)
    inspection_cases = needs_inspection + endpoint_failures
    owner = serving_owner(report)
    error_label = owner_error_label(report)
    api_label = owner_api_label(report)
    research_label = f"Research Cases (likely not {owner})"
    if dynamo_errors:
        status = error_label
    elif inspection_cases:
        status = "Needs Inspection"
    else:
        status = owner_clear_label(report)
    status_tone = status_class(report)
    artifacts_prefix = "artifacts"
    artifact_links = [
        f'<a href="{artifacts_prefix}/latest.json">latest.json</a>',
        f'<a href="{artifacts_prefix}/results.public.jsonl">results.public.jsonl</a>',
        f'<a href="{artifacts_prefix}/failures.public.jsonl">failures.public.jsonl</a>',
        f'<a href="{artifacts_prefix}/failure_summary.json">failure_summary.json</a>',
        f'<a href="{artifacts_prefix}/ai_analysis.json">ai_analysis.json</a>',
        f'<a href="{artifacts_prefix}/summary.md">summary.md</a>',
    ]
    artifact_links_html = "".join(artifact_links)
    case_groups = grouped_case_records(report.get("records") or [])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(str(report.get("title") or DEFAULT_TITLE))}</title>
  <style>
    :root {{
      --nvidia-green: #76b900;
      --green-dark: #426b00;
      --page: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2428;
      --muted: #667085;
      --line: #d7dde5;
      --bad: #b42318;
      --good: #067647;
      --warn: #b54708;
      --code: #111827;
      --shadow: 0 10px 24px rgba(17, 24, 39, 0.07);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #ffffff 0, var(--page) 260px);
      color: var(--ink);
      font-family: "NVIDIA Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }}
    header {{
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      background: #0b0b0b;
      color: #fff;
    }}
    header strong {{
      text-transform: uppercase;
      letter-spacing: 0;
      font-size: 18px;
    }}
    header span {{ color: #d7dde5; font-size: 13px; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px 24px 44px; }}
    .topline {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0 0 7px; font-size: 28px; line-height: 1.15; letter-spacing: 0; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; line-height: 1.2; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 5px 10px;
      border-radius: 6px;
      font-weight: 700;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .badge.pass {{ color: var(--good); border-color: #a6d8bd; }}
    .badge.fail {{ color: var(--bad); border-color: #f0b8b2; }}
    .badge.warn {{ color: var(--warn); border-color: #f8d49b; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 78px;
      box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04);
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .metric strong {{ font-size: 24px; line-height: 1; }}
    .metric.fail strong {{ color: var(--bad); }}
    .metric.pass strong {{ color: var(--good); }}
    .metric.warn strong {{ color: var(--warn); }}
    .grid {{
      display: block;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 22px;
      box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04);
      overflow: visible;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{
      text-align: left;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      border-bottom: 1px solid var(--line);
      padding: 10px;
      vertical-align: bottom;
      background: #f8fafc;
    }}
    td {{
      vertical-align: top;
      border-bottom: 1px solid #eceff3;
      padding: 10px;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{
      color: var(--code);
      background: #f1f3f5;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      padding: 1px 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    pre {{
      max-height: 420px;
      overflow: auto;
      background: #111827;
      color: #f8fafc;
      padding: 10px;
      border-radius: 6px;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    .startup-command pre {{ max-height: 640px; }}
    details summary {{ cursor: pointer; color: var(--green-dark); font-weight: 650; }}
    .record-details details {{ margin-bottom: 6px; }}
    .record-details details:last-child {{ margin-bottom: 0; }}
    a {{ color: var(--green-dark); font-weight: 650; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .ok {{ color: var(--good); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .low {{ color: var(--warn); font-weight: 700; }}
    .very-low {{ color: var(--muted); font-weight: 700; }}
    .empty {{ color: var(--muted); text-align: center; padding: 22px; }}
    .metric-count {{
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      text-align: right;
      white-space: nowrap;
    }}
    .links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }}
    .links a {{
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 6px 9px;
    }}
    .summary-link {{
      margin: 0 0 12px;
    }}
    .contract-note {{
      max-width: 880px;
      margin-top: 10px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .model-switcher {{
      display: flex;
      flex-wrap: wrap;
      align-items: stretch;
      gap: 8px;
      margin: 8px 0 16px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 1px 2px rgba(17, 24, 39, 0.05);
    }}
    .model-nav-group {{
      display: grid;
      gap: 5px;
      align-content: start;
      min-width: 150px;
      padding: 7px;
      border: 1px solid #e8edf3;
      border-left-width: 4px;
      border-radius: 7px;
      background: #fafbfc;
    }}
    .model-group-tone-0 {{ background: #f4fbf7; border-color: #cfe9d9; border-left-color: #2f8f56; }}
    .model-group-tone-1 {{ background: #f4f8ff; border-color: #d4e1f7; border-left-color: #4778c7; }}
    .model-group-tone-2 {{ background: #fff9ef; border-color: #f1dfbd; border-left-color: #b7791f; }}
    .model-group-tone-3 {{ background: #f8f6ff; border-color: #ded8f2; border-left-color: #7c6bb2; }}
    .model-group-tone-4 {{ background: #f3fbfb; border-color: #cfe6e6; border-left-color: #358282; }}
    .model-group-tone-5 {{ background: #fff5f6; border-color: #efd2d7; border-left-color: #b55b6a; }}
    .model-group-tone-0 .model-group-label {{ color: #236b42; }}
    .model-group-tone-1 .model-group-label {{ color: #335f9f; }}
    .model-group-tone-2 .model-group-label {{ color: #8a5a16; }}
    .model-group-tone-3 .model-group-label {{ color: #67569c; }}
    .model-group-tone-4 .model-group-label {{ color: #2e7474; }}
    .model-group-tone-5 .model-group-label {{ color: #985063; }}
    .model-nav-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }}
    .model-group-label {{
      display: inline-flex;
      align-items: center;
      min-height: 18px;
      padding: 0 2px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 850;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    .model-tab {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 5px 9px;
      border-radius: 5px;
      color: var(--ink);
      font-weight: 700;
      text-decoration: none;
      background: #fff;
      border: 1px solid #e5eaf0;
    }}
    .model-tab:hover {{
      background: #eef7f2;
      border-color: #b8d9c8;
      text-decoration: none;
    }}
    .model-tab.active {{
      background: #111827;
      color: #fff;
    }}
    .model-tab-summary {{
      align-self: stretch;
      min-height: 58px;
      font-size: 14px;
      border-color: #d7dde5;
    }}
    .findings {{
      display: block;
      border-left: 5px solid {("var(--bad)" if dynamo_errors else ("var(--warn)" if inspection_cases else "var(--good)"))};
    }}
    .findings h3 {{
      margin: 18px 0 8px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .findings p {{
      color: var(--muted);
      margin: 8px 0 0;
      line-height: 1.45;
    }}
    .ai-analysis {{
      border-left: 5px solid var(--warn);
    }}
    .analysis-meta {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    .analysis-note + .analysis-note {{
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid #eceff3;
    }}
    .analysis-note h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--ink);
      letter-spacing: 0;
    }}
    .analysis-note p {{
      margin: 0;
      color: var(--ink);
      line-height: 1.5;
    }}
    .analysis-note ul {{
      margin: 10px 0 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .analysis-note li + li {{ margin-top: 5px; }}
    .finding-columns {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .finding-columns.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .finding-columns h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .finding-columns ul {{
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .finding-columns li {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 7px 0;
      border-bottom: 1px solid #eceff3;
    }}
    .finding-columns li:last-child {{ border-bottom: 0; }}
    .finding-columns li strong {{ color: var(--bad); font-size: 16px; }}
    .kind-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .kind-pill {{
      display: inline-flex;
      align-items: center;
      position: relative;
      cursor: help;
    }}
    .kind-pill::after,
    .source-badge::after {{
      content: attr(data-tooltip);
      position: absolute;
      left: 0;
      top: calc(100% + 8px);
      z-index: 20;
      width: min(340px, 80vw);
      padding: 9px 10px;
      border-radius: 6px;
      background: #111827;
      color: #fff;
      font-size: 12px;
      line-height: 1.35;
      font-weight: 500;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
      opacity: 0;
      pointer-events: none;
      transform: translateY(-3px);
      transition: opacity 120ms ease, transform 120ms ease;
    }}
    .kind-pill:hover::after,
    .kind-pill:focus::after,
    .source-badge:hover::after,
    .source-badge:focus::after {{
      opacity: 1;
      transform: translateY(0);
    }}
    .kind-pill code {{
      border-color: #d7dde5;
      white-space: nowrap;
    }}
    .kind-pill.high code {{ border-color: #f0b8b2; background: #fff5f3; }}
    .kind-pill.low code {{ border-color: #f8d49b; background: #fff8eb; }}
    .source-badge {{
      display: inline-flex;
      position: relative;
      align-items: center;
      max-width: 220px;
      padding: 4px 7px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-weight: 700;
      font-size: 12px;
      cursor: help;
      white-space: normal;
    }}
    .source-dynamo {{ color: var(--bad); background: #fff5f3; border-color: #f0b8b2; }}
    .source-engine {{ color: #7a4b00; background: #fff8eb; border-color: #f8d49b; }}
    .source-endpoint {{ color: var(--warn); background: #fff8eb; border-color: #f8d49b; }}
    .source-mixed {{ color: var(--ink); background: #f1f3f5; }}
    .mode-stack {{
      display: grid;
      gap: 6px;
      min-width: 190px;
    }}
    .mode-chip {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      background: #fff;
    }}
    .mode-chip.ok {{
      border-color: #a6d8bd;
      background: #f3fbf6;
    }}
    .mode-chip.bad {{
      border-color: #f0b8b2;
      background: #fff5f3;
    }}
    .mode-chip strong {{
      margin-left: 5px;
      font-size: 12px;
    }}
    .management-table th:first-child,
    .management-table td:first-child {{
      width: 260px;
    }}
    .management-table th:nth-child(2),
    .management-table td:nth-child(2) {{
      width: 230px;
    }}
    .management-table th:last-child,
    .management-table td:last-child {{
      width: 38%;
    }}
    @media (max-width: 980px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .findings {{ grid-template-columns: 1fr; }}
      .finding-columns,
      .finding-columns.three {{ grid-template-columns: 1fr; }}
      .topline {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <strong>NVIDIA</strong>
    <span>Tool Calling Probe</span>
  </header>
  <main>
    {render_model_switcher(model_links, active_slug=active_model_slug)}
    <div class="topline">
      <div>
        <h1>{html.escape(str(report.get("title") or DEFAULT_TITLE))}</h1>
        <div class="subtle">
          Generated {html.escape(str(report.get("generated_at")))} ·
          <strong>{html.escape(model_label)}</strong> ·
          <code>{html.escape(str(report.get("model")))}</code> ·
          <code>{html.escape(str(report.get("base_url")))}</code>
        </div>
        <div class="links">
          {artifact_links_html}
        </div>
        <div class="contract-note">
          Contract: this page checks the OpenAI-compatible API surface:
          structured <code>message.tool_calls</code> and <code>finish_reason=tool_calls</code>.
          Raw model-template tool syntax is model-specific and should not leak into
          assistant content or reasoning. Parser parity and malformed/recovery cases
          are tracked separately in
          <a href="http://speedoflight.nvidia.com/dynamo/commits/tests/parity/parser/">the parser parity chart</a>.
          The dashboard error count below includes only likely {html.escape(api_label)}
          issues for this serving path. Model-behavior cases remain listed as
          research cases so they do not look like {html.escape(owner)} regressions.
        </div>
      </div>
      <div class="badge {status_tone}">{status}</div>
    </div>

    <div class="metrics">
      {render_metric("Total Requests", summary.get("total"))}
      {render_metric("Passed", summary.get("passed"), "pass")}
      {render_metric(error_label, dynamo_errors, "fail" if dynamo_errors else "pass")}
      {render_metric(research_label, model_research_cases, "warn" if model_research_cases else "pass")}
      {render_metric("Inspect (ownership unclear or operational issue)", inspection_cases, "warn" if inspection_cases else "pass")}
      {render_metric("Pass Rate", "n/a" if summary.get("pass_rate") is None else f"{summary.get('pass_rate') * 100:.1f}%")}
      {render_metric("p50 Latency", "n/a" if summary.get("latency_ms_p50") is None else f"{summary.get('latency_ms_p50')} ms")}
      {render_deployment_metrics(report)}
    </div>

    {render_vllm_serve_command(report)}

    {render_startup_command(report)}

    {render_ai_analysis(report)}

    {render_failure_issue_summary(report)}

    <section>
      <h2>{html.escape(error_label)}</h2>
      <p class="subtle">Only likely {html.escape(api_label)} issues appear here. This is the customer-facing error list for this serving path.</p>
      <table class="management-table">
        <thead>
          <tr><th>Case</th><th>Stream / Nonstream</th><th>Priority</th><th>What Happened</th><th>Request</th><th>Response</th></tr>
        </thead>
        <tbody>{render_management_failure_rows(case_groups, report, buckets={"dynamo"}, empty_text=f"No likely {api_label} errors in this run.")}</tbody>
      </table>
    </section>

    <section>
      <h2>Research / Model-Behavior Cases (likely not {html.escape(owner)})</h2>
      <p class="subtle">These cases did not satisfy the probe contract, but the response shape points to model behavior, ambiguous ownership, or endpoint transport rather than a clear {html.escape(owner)} parser regression.</p>
      <table class="management-table">
        <thead>
          <tr><th>Case</th><th>Stream / Nonstream</th><th>Priority</th><th>What Happened</th><th>Request</th><th>Response</th></tr>
        </thead>
        <tbody>{render_management_failure_rows(case_groups, report, buckets={"model", "mixed", "endpoint"}, empty_text="No research or inspection cases in this run.")}</tbody>
      </table>
    </section>

    <section>
      <h2>Successes</h2>
      <table class="management-table">
        <thead>
          <tr><th>Case</th><th>Stream / Nonstream</th><th>Request</th><th>Response</th></tr>
        </thead>
        <tbody>{render_management_success_rows(case_groups)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def history_item(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": report["run_id"],
        "generated_at": report["generated_at"],
        "model": report["model"],
        "model_slug": report.get("model_slug"),
        "model_label": report.get("model_label"),
        "base_url": report["base_url"],
        "duration_seconds": report["duration_seconds"],
        "summary": report["summary"],
    }


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def write_static_site(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    site_dir: Path,
    keep_runs: int,
    model_slug: str | None = None,
    model_label: str | None = None,
    model_links: list[dict[str, str]] | None = None,
    root_alias: bool = True,
) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model_slug or model_slug_from_model(
        str(report.get("model") or "model")
    )
    model_label = model_label or model_label_from_model(
        str(report.get("model") or "model")
    )
    report["model_slug"] = model_slug
    report["model_label"] = model_label
    vllm_serve_command = vllm_serve_command_from_report(report)
    if vllm_serve_command:
        report.setdefault("vllm_serve_command", vllm_serve_command)
        command_source = vllm_serve_command_source(report)
        if command_source:
            report.setdefault("vllm_serve_command_source", command_source)
    if is_vllm_report(report):
        report["failure_summary"] = ownership_display_copy(
            report, report.get("failure_summary") or {}
        )
        report["failures"] = ownership_display_copy(
            report, report.get("failures") or []
        )
        report["records"] = ownership_display_copy(report, report.get("records") or [])
        records = ownership_display_copy(report, records)
    model_links = normalize_model_links(
        model_links or [],
        active_slug=model_slug,
        active_label=model_label,
    )

    def publish_page(page_dir: Path) -> None:
        runs_dir = page_dir / "runs"
        run_site_dir = runs_dir / report["run_id"]
        artifacts_dir = page_dir / "artifacts"
        run_artifacts_dir = run_site_dir / "artifacts"
        history_path = page_dir / "history.json"
        history = [
            item
            for item in read_history(history_path)
            if item.get("run_id") != report["run_id"]
        ]
        history.insert(0, history_item(report))
        history = history[:keep_runs]

        failures = [record for record in records if not record["pass"]]

        for directory in (artifacts_dir, run_site_dir):
            if directory.exists():
                shutil.rmtree(directory)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        for target in (artifacts_dir, run_artifacts_dir):
            write_json(target / "latest.json", report)
            write_json(
                target / "failure_summary.json",
                report.get("failure_summary") or build_failure_summary(failures),
            )
            write_json(target / "ai_analysis.json", report.get("ai_analysis") or {})
            write_jsonl(target / "results.public.jsonl", records)
            write_jsonl(target / "failures.public.jsonl", failures)
            summary = Path(report["output_dir"]) / "summary.md"
            config = Path(report["output_dir"]) / "run_config.json"
            cases = Path(report["output_dir"]) / "cases.json"
            if summary.exists():
                shutil.copy2(summary, target / "summary.md")
            if config.exists():
                shutil.copy2(config, target / "run_config.json")
            if cases.exists():
                shutil.copy2(cases, target / "cases.json")
        write_json(history_path, history)
        (page_dir / "index.html").write_text(
            render_html(
                report,
                history,
                history_prefix="",
                model_links=model_links,
                active_model_slug=model_slug,
            ),
            encoding="utf-8",
        )
        run_site_dir.mkdir(parents=True, exist_ok=True)
        (run_site_dir / "index.html").write_text(
            render_html(
                report,
                history,
                history_prefix="../../",
                model_links=model_links,
                active_model_slug=model_slug,
            ),
            encoding="utf-8",
        )

        if runs_dir.exists():
            keep = {item["run_id"] for item in history if item.get("run_id")}
            for child in runs_dir.iterdir():
                if child.is_dir() and child.name not in keep:
                    shutil.rmtree(child)

    model_page_dir = site_dir / "models" / model_slug
    model_page_dir.mkdir(parents=True, exist_ok=True)
    publish_page(model_page_dir)
    if root_alias:
        publish_page(site_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tool-calling tests and publish a static report."
    )
    parser.add_argument("--site-dir", default=str(DEFAULT_SITE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--base-url", default=probe.DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-slug",
        default=None,
        help="Stable slug for this model's static subpage under models/<slug>/.",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Short display label for the model switcher.",
    )
    parser.add_argument(
        "--model-link",
        action="append",
        default=[],
        help="Model switcher entry formatted as 'slug|Label|/href/'. May be repeated.",
    )
    parser.add_argument(
        "--no-root-alias",
        action="store_true",
        help="Only write this model under models/<slug>/, leaving the root page unchanged.",
    )
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument(
        "--allow-other-base-url",
        action="store_true",
        help="Allow a base URL outside NVIDIA's inference API hosts.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Extra request header, formatted as 'Name: value'. May be repeated.",
    )
    parser.add_argument("--cases", default="all")
    parser.add_argument(
        "--exclude-cases",
        default="",
        help="Comma-separated case ID glob patterns to omit from the report.",
    )
    parser.add_argument(
        "--case-profile",
        default="auto",
        choices=(
            "auto",
            *probe.INLINE_CASE_PROFILES,
            *probe.available_case_profiles(),
            "all",
        ),
        help="Case profile to run. auto infers from --model.",
    )
    parser.add_argument("--modes", default="nonstream,stream")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--case-delay-seconds", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--raw-chars", type=int, default=20000)
    parser.add_argument("--detail-chars", type=int, default=6000)
    parser.add_argument("--record-success-raw", action="store_true")
    parser.add_argument(
        "--ai-analysis-mode",
        choices=("off", "heuristic", "codex"),
        default=os.environ.get("TOOL_CALLING_AI_ANALYSIS_MODE", "codex"),
        help=(
            "Generate the AI Analysis panel. 'codex' invokes Codex CLI and "
            "falls back to the deterministic classifier if unavailable."
        ),
    )
    parser.add_argument(
        "--codex-binary",
        default=os.environ.get("TOOL_CALLING_CODEX_BIN", "codex"),
        help="Codex CLI binary to use when --ai-analysis-mode=codex.",
    )
    parser.add_argument(
        "--ai-analysis-timeout-seconds",
        type=float,
        default=float(
            os.environ.get("TOOL_CALLING_AI_ANALYSIS_TIMEOUT_SECONDS", "120")
        ),
        help="Maximum time to wait for Codex CLI AI analysis.",
    )
    parser.add_argument(
        "--startup-command",
        default="",
        help="Server startup command/arguments to show in the static report.",
    )
    parser.add_argument(
        "--startup-command-file",
        default="",
        help="File containing the server startup command/arguments to show.",
    )
    parser.add_argument(
        "--startup-command-source",
        default="",
        help="Human-readable source for the startup command metadata.",
    )
    parser.add_argument("--keep-runs", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-on-test-failure",
        action="store_true",
        help="Return non-zero when the tool-calling contract tests fail.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.ai_analysis_timeout_seconds <= 0:
        parser.error("--ai-analysis-timeout-seconds must be > 0")
    try:
        model_links = [parse_model_link(value) for value in args.model_link]
    except ValueError as exc:
        parser.error(str(exc))

    report, records = run_probe(args)
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    site_dir = Path(args.site_dir).expanduser()
    model_slug = args.model_slug or model_slug_from_model(args.model)
    model_label = args.model_label or model_label_from_model(args.model)
    report["model_slug"] = model_slug
    report["model_label"] = model_label

    ai_analysis = build_ai_analysis(
        report,
        mode=args.ai_analysis_mode,
        codex_binary=args.codex_binary,
        timeout_seconds=args.ai_analysis_timeout_seconds,
    )
    if ai_analysis:
        report["ai_analysis"] = ai_analysis

    write_static_site(
        report,
        records,
        site_dir=site_dir,
        keep_runs=max(1, args.keep_runs),
        model_slug=model_slug,
        model_label=model_label,
        model_links=model_links,
        root_alias=not args.no_root_alias,
    )
    model_site = site_dir / "models" / model_slug / "index.html"
    print(f"Static report: {model_site}", flush=True)
    if not args.no_root_alias:
        print(f"Root alias: {site_dir / 'index.html'}", flush=True)
    print(
        f"Latest JSON: {site_dir / 'models' / model_slug / 'artifacts' / 'latest.json'}",
        flush=True,
    )
    failed = int(report.get("summary", {}).get("failed") or 0)
    if failed and args.fail_on_test_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
