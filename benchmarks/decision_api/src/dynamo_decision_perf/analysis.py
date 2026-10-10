# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit pinned AIPerf artifacts without modifying raw evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aiperf.common.models.record_models import MetricRecordInfo, RawRecordInfo
from pydantic import ValidationError

from .contracts import DecisionContractError, validate_response


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _decode(text: str) -> Any:
    return json.loads(
        text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
    )


def _read(path: Path, blockers: list[str], model=None) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        blockers.append(f"{path.name}: {exc}")
        return []
    rows = []
    for line, content in enumerate(text.splitlines() if model else [text], start=1):
        try:
            row = _decode(content)
            if not isinstance(row, dict):
                raise TypeError("Expected JSON object")
            if model is not None:
                model.model_validate_json(content, strict=True)
            rows.append(row)
        except (TypeError, ValueError, ValidationError) as exc:
            blockers.append(f"{path.name}:{line}: {exc}")
    if not rows:
        blockers.append(f"{path.name}: no readable records")
    return rows


def latency_statistics(values: list[float]) -> dict:
    """Nearest-rank quantiles, with explicit minimum tail sample counts."""
    ordered = sorted(values)
    count = len(ordered)

    def quantile(fraction: float, minimum: int = 1):
        return (
            ordered[max(0, math.ceil(fraction * count) - 1)]
            if count >= minimum
            else None
        )

    return {
        "count": count,
        "p50": quantile(0.5),
        "max": max(ordered) if ordered else None,
        "p95": quantile(0.95, 100),
        "p99": quantile(0.99, 1000),
    }


def _index(rows: list[dict], label: str, blockers: list[str]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        request_id = row.get("metadata", {}).get("x_request_id")
        if not isinstance(request_id, str) or not request_id:
            blockers.append(f"{label}: missing request ID")
        elif request_id in indexed:
            blockers.append(f"{label}: duplicate request ID {request_id}")
        else:
            indexed[request_id] = row
    return indexed


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _aggregate_window(aggregate: dict, metadata: dict) -> tuple[int, int]:
    offset = metadata.get("export_timezone_offset_seconds")
    if type(offset) is not int or not -86400 < offset < 86400:
        raise ValueError(
            "Aggregate timestamps require recorded export_timezone_offset_seconds"
        )
    values = []
    for key in ("start_time", "end_time"):
        value = aggregate.get(key)
        if not isinstance(value, str):
            raise TypeError("Aggregate lacks controller phase timestamps")
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone(timedelta(seconds=offset)))
        delta = stamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
        values.append(
            ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds)
            * 1000
        )
    return values[0] - 1000, values[1] + 1000


def _window(
    directory: Path, metadata: dict, aggregate: dict, blockers: list[str]
) -> tuple[dict, float | None]:
    resolved = dict(metadata)
    if (
        resolved.get("measurement_start_ns") is None
        or resolved.get("measurement_end_ns") is None
    ):
        if (directory / "phase_manifest.json").is_file():
            manifests = _read(directory / "phase_manifest.json", blockers)
            phases = []
            for manifest in manifests:
                entries = manifest.get("phases")
                if not isinstance(entries, list):
                    blockers.append("Phase manifest requires a phases array")
                    continue
                phases.extend(
                    p
                    for p in entries
                    if isinstance(p, dict) and p.get("phase_kind") == "profiling"
                )
            if len(phases) == 1:
                phase = phases[0]
                resolved.update(
                    measurement_start_ns=phase.get("start_ns"),
                    measurement_end_ns=phase.get("end_ns"),
                    measurement_window_source="aiperf.phase_manifest.controller_phase",
                )
                if phase.get("was_cancelled"):
                    blockers.append("Profiling phase was cancelled")
            else:
                blockers.append(
                    "Expected exactly one profiling phase for the measurement window"
                )
        else:
            try:
                start, end = _aggregate_window(aggregate, resolved)
                resolved.update(
                    measurement_start_ns=start,
                    measurement_end_ns=end,
                    measurement_window_source="aiperf.aggregate.controller_phase_microsecond_precision",
                )
            except (TypeError, ValueError) as exc:
                blockers.append(str(exc))
    start, end = (
        resolved.get("measurement_start_ns"),
        resolved.get("measurement_end_ns"),
    )
    if not _count(start) or not _count(end) or end <= start:
        blockers.append("Missing or invalid explicit measurement window")
        return resolved, None
    resolved.setdefault(
        "measurement_window_source", "caller_explicit_controller_window"
    )
    return resolved, (end - start) / 1_000_000_000


def _latency(
    profile: dict | None, blockers: list[str], request_id: str
) -> float | None:
    if profile is None:
        return None
    metric = profile.get("metrics", {}).get("request_latency")
    if not isinstance(metric, dict):
        return None
    factors = {"ns": 0.000001, "us": 0.001, "ms": 1, "s": 1000}
    value, unit = metric.get("value"), metric.get("unit")
    if not _number(value) or value < 0 or unit not in factors:
        blockers.append(f"{request_id}: invalid latency value or unit")
        return None
    return value * factors[unit]


def _payload_identity(raw: dict, expected: dict, dialect: str) -> str:
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise TypeError("Missing request payload")
    if dialect == "native_score":
        request_id = raw["metadata"]["x_request_id"]
        salt = "decision-native-" + hashlib.sha256(request_id.encode()).hexdigest()
        if payload.get("cache_salt") != salt:
            raise ValueError(
                "Native baseline requires a fresh request-ID-bound cache salt"
            )
    original = {
        key: value
        for key, value in payload.items()
        if not (dialect == "native_score" and key == "cache_salt")
    }
    digest = hashlib.sha256(
        json.dumps(
            original, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    hashes = expected.get("payload_sha256", {})
    wanted = hashes.get(dialect) if isinstance(hashes, dict) else hashes
    if digest != wanted:
        raise ValueError("Wire payload does not match frozen workload hash")
    headers = {
        key.lower(): value for key, value in (raw.get("request_headers") or {}).items()
    }
    if headers.get("x-decision-payload-sha256", digest) != digest:
        raise ValueError("Original-payload header does not match frozen workload")
    return digest


def _classify(raw: dict, profile: dict | None) -> str:
    metadata = raw["metadata"]
    error = raw.get("error") or (profile or {}).get("error") or {}
    status = raw.get("status")
    if error.get("type") == "DecisionContractError" or "DecisionContractError" in (
        error.get("cause_chain") or []
    ):
        return "schema_error"
    if metadata.get("was_cancelled"):
        return "cancelled"
    if "timeout" in str(error.get("type", "")).lower() or status in (408, 504):
        return "timeout"
    if type(status) is int and 400 <= status < 500:
        return "rejection"
    if type(status) is int and status >= 500:
        return "http_error"
    if status != 200:
        return "transport_error"
    return "pipeline_error" if error else "response"


def _response_body(responses: list[dict], dialect: str) -> dict:
    if dialect != "native_score":
        if len(responses) != 1 or not isinstance(responses[0].get("text"), str):
            raise ValueError("Expected one complete non-streaming JSON response")
        return _decode(responses[0]["text"])
    terminals = []
    for response in responses:
        text = response.get("text")
        if text is None:
            text = "\n".join(
                packet["value"]
                for packet in response.get("packets", [])
                if packet.get("name") == "data" and packet.get("value") is not None
            )
        if not text or text == "[DONE]":
            continue
        body = _decode(text)
        if not isinstance(body, dict):
            raise TypeError("Native SSE data must be an object")
        meta = body.get("meta_info")
        if "error" in body or not isinstance(meta, dict):
            raise ValueError("Native SSE error or missing metadata")
        if body.get("output_ids", []) != [] or meta.get("completion_tokens", 0) != 0:
            raise ValueError("Native stream generated tokens before termination")
        if meta.get("finish_reason") is not None:
            terminals.append(body)
    if len(terminals) != 1:
        raise ValueError("Expected exactly one terminal native scoring response")
    return terminals[0]


def _request(
    raw: dict, profile: dict | None, workload: dict, metadata: dict, blockers: list[str]
) -> dict:
    fields = raw["metadata"]
    request_id, logical_id = fields["x_request_id"], fields.get("conversation_id")
    latency = _latency(profile, blockers, request_id)
    start, end = fields["request_start_ns"], fields["request_end_ns"]
    latency_source = "profile.request_latency" if latency is not None else None
    if (
        latency is None
        and end > start
        and (raw.get("payload") is not None or raw.get("status") is not None)
    ):
        latency = (end - start) / 1_000_000
        latency_source = "metadata.monotonic_delta"
    if end < start:
        blockers.append(f"{request_id}: request ends before it starts")
    window_start, window_end = (
        metadata.get("measurement_start_ns"),
        metadata.get("measurement_end_ns"),
    )
    if (
        _count(window_start)
        and _count(window_end)
        and not (window_start <= start <= end <= window_end)
    ):
        blockers.append(f"{request_id}: request outside the measurement window")
    if profile and any(
        profile["metadata"].get(key) != fields.get(key)
        for key in (
            "conversation_id",
            "benchmark_phase",
            "request_start_ns",
            "request_end_ns",
        )
    ):
        blockers.append(f"{request_id}: raw/profile metadata mismatch")
    result = {
        "request_id": request_id,
        "logical_id": logical_id,
        "latency_ms": latency,
        "latency_source": latency_source,
        "status": raw.get("status"),
        "outcome": _classify(raw, profile),
        "completed_questions": 0,
        "refusal_questions": 0,
        "api_usage": None,
        "payload_sha256": None,
        "workload_signature": None,
    }
    expected = workload.get(logical_id)
    if expected is None:
        blockers.append(f"{request_id}: unknown workload logical ID {logical_id!r}")
    elif raw.get("payload") is not None:
        try:
            result["payload_sha256"] = _payload_identity(
                raw, expected, metadata.get("dialect", "")
            )
            identity = {
                key: value
                for key, value in expected.items()
                if key not in ("payload_sha256", "request")
            }
            result["workload_signature"] = hashlib.sha256(
                json.dumps(identity, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        except (TypeError, ValueError) as exc:
            blockers.append(f"{request_id}: {exc}")
    if result["outcome"] == "schema_error":
        blockers.append(f"{request_id}: AIPerf recorded a decision-contract failure")
    if result["outcome"] not in ("response", "pipeline_error"):
        return result
    try:
        if not isinstance(raw.get("payload"), dict):
            raise TypeError("HTTP 200 has no request payload for workload correlation")
        dialect = metadata.get("dialect", "")
        body = _response_body(raw["responses"], dialect)
        summary = validate_response(body, dialect, raw.get("payload"))
        if expected is None or summary.question_count != expected.get(
            "expected_questions"
        ):
            raise ValueError("Response question count differs from the frozen workload")
        if result["outcome"] == "pipeline_error":
            blockers.append(f"{request_id}: HTTP 200 has an AIPerf processing error")
            return result
        result.update(
            outcome="valid",
            completed_questions=summary.question_count - summary.refusal_count,
            refusal_questions=summary.refusal_count,
            api_usage={
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "cached_tokens": summary.cached_tokens,
            },
        )
    except (TypeError, ValueError, DecisionContractError) as exc:
        result["outcome"] = "schema_error"
        blockers.append(f"{request_id}: invalid HTTP 200 response: {exc}")
    return result


def _summary_counts(requests: list[dict], metadata: dict, warmup: int) -> dict:
    outcomes = Counter(row["outcome"] for row in requests)
    offered, expected = (
        metadata.get("offered_requests"),
        metadata.get("expected_requests"),
    )
    attempted = len(requests)
    unattempted = max(0, offered - attempted) if _count(offered) else None
    unoffered = (
        max(0, expected - offered) if _count(expected) and _count(offered) else None
    )
    return {
        "expected_requests": expected,
        "offered_requests": offered,
        "attempted_requests": attempted,
        "completed_http_requests": sum(
            type(row["status"]) is int and 100 <= row["status"] <= 599
            for row in requests
        ),
        "unknown_http_status_requests": sum(row["status"] is None for row in requests),
        "valid_http_requests": outcomes["valid"],
        "completed_questions": sum(row["completed_questions"] for row in requests),
        "refusal_questions": sum(row["refusal_questions"] for row in requests),
        "refusal_requests": sum(row["refusal_questions"] > 0 for row in requests),
        "schema_errors": outcomes["schema_error"],
        "rejections": outcomes["rejection"],
        "timeouts": outcomes["timeout"],
        "http_errors": outcomes["http_error"],
        "transport_errors": outcomes["transport_error"],
        "cancelled": outcomes["cancelled"],
        "pipeline_errors": outcomes["pipeline_error"],
        "unattempted_offers": unattempted,
        "unoffered_expected": unoffered,
        "incomplete_requests": max(0, expected - attempted)
        if _count(expected)
        else unattempted,
        "warmup_excluded": warmup,
    }


def analyze_run(artifact_dir: str | Path, metadata: dict, workload: list[dict]) -> dict:
    """Join raw and metric exports by request ID; report all profiling outcomes.

    ``offered_requests`` is an observed controller count, never the configured
    target. Use None when unavailable. ``expected_requests`` is the plan count.
    The measurement window includes failed requests and drain, excluding warmup.
    """
    directory, blockers = Path(artifact_dir), []
    execution_rows = _read(directory.parent / "benchmark_execution.json", blockers)
    execution = execution_rows[0] if execution_rows else {}
    manifest = directory.parent / "manifest.json"
    if execution.get("exit_code") != 0:
        blockers.append("Benchmark execution did not exit successfully")
    if (
        not manifest.is_file()
        or execution.get("manifest_sha256")
        != hashlib.sha256(manifest.read_bytes()).hexdigest()
    ):
        blockers.append(
            "Benchmark execution does not bind the adjacent manifest SHA256"
        )
    manifest_rows = _read(manifest, blockers)
    frozen = manifest_rows[0].get("audit_metadata") if manifest_rows else None
    if not isinstance(frozen, dict):
        blockers.append("Manifest lacks frozen audit metadata")
    else:
        for key in (
            "run_id",
            "series_id",
            "profile_id",
            "cache_phase",
            "dialect",
            "aiperf_version",
            "expected_requests",
            "warmup_expected_requests",
            "cost_boundary",
        ):
            if key not in metadata or metadata[key] != frozen.get(key):
                blockers.append(f"Run metadata differs from frozen manifest: {key}")
    aggregate_rows = _read(directory / "profile_export_aiperf.json", blockers)
    aggregate = aggregate_rows[0] if aggregate_rows else {}
    source_metadata = dict(metadata)
    source_metadata.setdefault(
        "export_timezone_offset_seconds",
        execution.get("export_timezone_offset_seconds"),
    )
    source_metadata.setdefault("aiperf_version", aggregate.get("aiperf_version"))
    resolved, seconds = _window(directory, source_metadata, aggregate, blockers)
    for key in (
        "run_id",
        "series_id",
        "profile_id",
        "cache_phase",
        "dialect",
        "cost_boundary",
    ):
        if not isinstance(resolved.get(key), str) or not resolved[key]:
            blockers.append(f"Missing run metadata: {key}")
    raw_rows = _read(directory / "profile_export_raw.jsonl", blockers, RawRecordInfo)
    profiles = _read(directory / "profile_export.jsonl", blockers, MetricRecordInfo)
    if aggregate.get("is_complete") is not True or aggregate.get("was_cancelled"):
        blockers.append(
            "AIPerf aggregate is incomplete, cancelled, or lacks completeness evidence"
        )
    if (
        aggregate.get("aiperf_version") != "0.13.0"
        or resolved.get("aiperf_version", "0.13.0") != "0.13.0"
    ):
        blockers.append("Expected pinned AIPerf 0.13.0 artifacts and runtime")
    raw_index, profile_index = (
        _index(raw_rows, "raw", blockers),
        _index(profiles, "profile", blockers),
    )
    if raw_index.keys() != profile_index.keys():
        blockers.append("Raw/profile request ID sets do not match")
    known = {
        row["logical_id"]: row
        for row in workload
        if isinstance(row, dict)
        and isinstance(row.get("logical_id"), str)
        and _count(row.get("expected_questions"))
        and row["expected_questions"] > 0
    }
    if len(known) != len(workload):
        blockers.append(
            "Invalid or duplicate frozen workload logical IDs or question counts"
        )
    requests, warmup = [], 0
    for request_id, raw in raw_index.items():
        phase = raw["metadata"]["benchmark_phase"]
        if phase == "warmup":
            warmup += 1
        elif phase == "profiling":
            requests.append(
                _request(raw, profile_index.get(request_id), known, resolved, blockers)
            )
        else:
            blockers.append(f"{request_id}: unsupported phase {phase}")
    counts = _summary_counts(requests, resolved, warmup)
    for field in ("expected_requests", "offered_requests", "warmup_expected_requests"):
        if resolved.get(field) is not None and not _count(resolved[field]):
            blockers.append(f"Invalid nonnegative count: {field}")
    if (
        _count(counts["expected_requests"])
        and counts["attempted_requests"] != counts["expected_requests"]
    ):
        blockers.append(
            "Observed profiling requests differ from planned expected requests"
        )
    if (
        _count(counts["offered_requests"])
        and counts["attempted_requests"] != counts["offered_requests"]
    ):
        blockers.append(
            "Observed profiling requests differ from controller offered requests"
        )
    if (
        _count(resolved.get("warmup_expected_requests"))
        and warmup != resolved["warmup_expected_requests"]
    ):
        blockers.append("Warmup record count differs from plan")
    rates = {
        f"{key}_per_second": value / seconds if _number(value) and seconds else None
        for key, value in {
            "offered_http_requests": counts["offered_requests"],
            "attempted_http_requests": len(requests),
            "completed_http_requests": counts["completed_http_requests"],
            "valid_http_requests": counts["valid_http_requests"],
            "completed_questions": counts["completed_questions"],
        }.items()
    }
    usage = {}
    for field in ("input_tokens", "output_tokens", "cached_tokens"):
        values = [(row["api_usage"] or {}).get(field) for row in requests]
        measured = [value for value in values if value is not None]
        usage[field] = {
            "known_sum": sum(measured) if measured else None,
            "unknown_requests": values.count(None),
        }
    latencies = {
        "all_attempts": latency_statistics(
            [r["latency_ms"] for r in requests if r["latency_ms"] is not None]
        ),
        "valid_responses": latency_statistics(
            [
                r["latency_ms"]
                for r in requests
                if r["outcome"] == "valid" and r["latency_ms"] is not None
            ]
        ),
    }
    limitations = [
        "Metric conformance is not an accuracy measurement.",
        "API token accounting is not measured backend work.",
        "Single-run observations do not establish statistical significance.",
        "Completed questions exclude refusals; valid HTTP responses include genuine refusals.",
    ]
    if (
        resolved.get("measurement_window_source")
        == "aiperf.aggregate.controller_phase_microsecond_precision"
    ):
        limitations.append(
            "Aggregate controller ISO timestamps have microsecond precision; window expanded by 1000 ns at each boundary for rounding."
        )
    if resolved.get("offered_requests") is None:
        limitations.append(
            "Controller offered count is unknown; offered rate and unattempted-offer count remain null."
        )
    if latencies["all_attempts"]["count"] != len(requests):
        limitations.append(
            "Some failed/incomplete requests lack latency metrics; latency sample count excludes only those missing measurements."
        )
    artifacts = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (
            "profile_export_raw.jsonl",
            "profile_export.jsonl",
            "profile_export_aiperf.json",
            "phase_manifest.json",
        )
        if (directory / name).is_file()
    }
    return {
        "audit": {
            "status": "invalid" if blockers else "valid",
            "gain_claims_allowed": not blockers,
            "blockers": blockers,
            "limitations": limitations,
            "artifact_dir": str(directory.resolve()),
            "artifact_sha256": artifacts,
            "parser": "AIPerf 0.13.0 RawRecordInfo and MetricRecordInfo",
            "execution": execution,
            "next_action": "rerun_benchmark" if blockers else "continue_analysis",
        },
        "summary": {
            "metadata": resolved,
            "measurement_seconds": seconds,
            "counts": counts,
            "rates": rates,
            "fractions": {
                "valid_of_offered": counts["valid_http_requests"]
                / counts["offered_requests"]
                if _count(counts["offered_requests"]) and counts["offered_requests"]
                else None,
                "valid_of_attempted": counts["valid_http_requests"] / len(requests)
                if requests
                else None,
            },
            "latency_ms": latencies,
            "api_usage": usage,
            "actual_work": resolved.get("actual_work"),
        },
        "requests": requests,
    }


def paired_differences(baseline: dict, candidate: dict) -> dict:
    """Compute candidate-minus-baseline per-request deltas, never quantile deltas."""
    left, right = baseline["summary"]["metadata"], candidate["summary"]["metadata"]
    keys = ("series_id", "profile_id", "cache_phase", "aiperf_version")
    if any(run["audit"]["status"] != "valid" for run in (baseline, candidate)) or any(
        left.get(key) != right.get(key) for key in keys
    ):
        return {
            "status": "not_comparable",
            "reason": "Invalid audit or differing comparison series/profile/cache phase/tool version",
        }

    def unique(rows):
        counts = Counter(row["logical_id"] for row in rows)
        return {
            row["logical_id"]: row
            for row in rows
            if counts[row["logical_id"]] == 1
            and row["outcome"] == "valid"
            and row["latency_ms"] is not None
        }

    first, second = unique(baseline["requests"]), unique(candidate["requests"])
    pairs = [
        {
            "logical_id": key,
            "latency_difference_ms": second[key]["latency_ms"]
            - first[key]["latency_ms"],
        }
        for key in sorted(first.keys() & second.keys())
        if first[key]["workload_signature"] == second[key]["workload_signature"]
    ]
    return {
        "status": "comparable" if pairs else "no_unambiguous_pairs",
        "matched_requests": len(pairs),
        "baseline_unmatched_requests": len(baseline["requests"]) - len(pairs),
        "candidate_unmatched_requests": len(candidate["requests"]) - len(pairs),
        "latency_difference_ms": latency_statistics(
            [row["latency_difference_ms"] for row in pairs]
        ),
        "pairs": pairs,
        "baseline_cost_boundary": left["cost_boundary"],
        "candidate_cost_boundary": right["cost_boundary"],
        "interpretation": "Paired end-to-end cost difference, not isolated Dynamo overhead or an accuracy comparison.",
    }
