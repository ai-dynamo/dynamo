# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parsers for structured router log events used by e2e tests."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, cast

_ROUTING_MESSAGE_PATTERN = re.compile(
    r"\[ROUTING\].*with\s*(?P<overlap>\d+(?:\.\d+)?)/(?P<total>\d+)\s*blocks overlap"
)
_FIELD_PATTERN = re.compile(
    r"\b(?P<key>request_id|worker_id|dp_rank|overlap_blocks|total_blocks)="
    r"(?P<value>\"[^\"]*\"|\S+)"
)

KvEventDiagnosticCode = Literal[
    "kv_event_publisher_disabled",
    "kv_event_source_not_observed",
    "kv_event_source_ambiguous",
    "kv_event_source_recovered",
]

KV_EVENT_DIAGNOSTIC_CODES = frozenset(
    {
        "kv_event_publisher_disabled",
        "kv_event_source_not_observed",
        "kv_event_source_ambiguous",
        "kv_event_source_recovered",
    }
)
_SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class LogReadable(Protocol):
    def read_logs(self) -> str:
        ...


@dataclass(frozen=True)
class RouterKvOverlapRecord:
    overlap_blocks: int
    total_blocks: int
    request_id: str | None = None
    worker_id: int | None = None
    dp_rank: int | None = None


def _context_prefix(context: str) -> str:
    return f"{context}: " if context else ""


def _block_count(value: str) -> int:
    return int(float(value.strip('"')) + 0.5)


def _int_field(fields: dict[str, str], key: str) -> int | None:
    value = fields.get(key)
    if value is None:
        return None
    return int(value.strip('"'))


def _str_field(fields: dict[str, str], key: str) -> str | None:
    value = fields.get(key)
    if value is None:
        return None
    return value.strip('"')


def _record_from_line(line: str) -> RouterKvOverlapRecord | None:
    if "[ROUTING]" not in line:
        return None

    fields = {match["key"]: match["value"] for match in _FIELD_PATTERN.finditer(line)}
    overlap = fields.get("overlap_blocks")
    total = fields.get("total_blocks")
    if overlap is not None and total is not None:
        return RouterKvOverlapRecord(
            overlap_blocks=_block_count(overlap),
            total_blocks=_block_count(total),
            request_id=_str_field(fields, "request_id"),
            worker_id=_int_field(fields, "worker_id"),
            dp_rank=_int_field(fields, "dp_rank"),
        )

    match = _ROUTING_MESSAGE_PATTERN.search(line)
    if match is None:
        return None
    return RouterKvOverlapRecord(
        overlap_blocks=_block_count(match["overlap"]),
        total_blocks=_block_count(match["total"]),
    )


def extract_router_kv_overlap_records(log_text: str) -> list[RouterKvOverlapRecord]:
    return [
        record
        for line in log_text.splitlines()
        if (record := _record_from_line(line)) is not None
    ]


def wait_for_router_kv_overlap(
    logs_provider: Callable[[], str],
    *,
    start_offset: int,
    pre_request_record_count: int,
    context: str = "",
    log_label: str = "router",
    timeout_s: float = 60.0,
) -> tuple[int, int, str]:
    deadline = time.time() + timeout_s
    last_segment = ""

    while time.time() < deadline:
        full_logs = logs_provider()
        segment = full_logs[start_offset:]
        last_segment = segment
        records = extract_router_kv_overlap_records(full_logs)
        if len(records) >= pre_request_record_count + 1:
            record = records[pre_request_record_count]
            return record.overlap_blocks, record.total_blocks, segment
        time.sleep(1)

    segment_records = extract_router_kv_overlap_records(last_segment)
    if segment_records:
        record = segment_records[-1]
        return record.overlap_blocks, record.total_blocks, last_segment

    raise AssertionError(
        f"{_context_prefix(context)}Expected a structured router KV overlap log "
        f"event after the request.\nRecent {log_label} logs:\n{last_segment[-4000:]}"
    )


@dataclass(frozen=True)
class KvEventDiagnostic:
    """Typed KV-event diagnostic decoded from the runtime JSONL log stream."""

    diagnostic_code: KvEventDiagnosticCode
    model: str
    worker_role: str
    requirement: str
    worker_id: int
    serving_endpoint: str
    kv_event_publishing_enabled: bool
    waited_ms: int
    rank_count: int
    dp_ranks: str


def _require_string(record: dict[str, object], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(
            f"router diagnostic line {line_number} field {field!r} must be a string"
        )
    return value


def _require_int(record: dict[str, object], field: str, line_number: int) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"router diagnostic line {line_number} field {field!r} must be an integer"
        )
    return value


def _require_bool(record: dict[str, object], field: str, line_number: int) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise ValueError(
            f"router diagnostic line {line_number} field {field!r} must be a boolean"
        )
    return value


def _validate_dp_ranks(dp_ranks: str, rank_count: int, line_number: int) -> None:
    try:
        ranks = [] if not dp_ranks else [int(rank) for rank in dp_ranks.split(",")]
    except ValueError as error:
        raise ValueError(
            f"router diagnostic line {line_number} has invalid dp_ranks={dp_ranks!r}"
        ) from error

    canonical = ",".join(str(rank) for rank in sorted(set(ranks)))
    if dp_ranks != canonical:
        raise ValueError(
            f"router diagnostic line {line_number} dp_ranks must be unique, "
            "comma-separated, and ascending"
        )
    if rank_count != len(ranks):
        raise ValueError(
            f"router diagnostic line {line_number} rank_count={rank_count} "
            f"does not match dp_ranks={dp_ranks!r}"
        )


def _parse_diagnostic(record: dict[str, object], line_number: int) -> KvEventDiagnostic:
    diagnostic_code = _require_string(record, "diagnostic_code", line_number)
    if diagnostic_code not in KV_EVENT_DIAGNOSTIC_CODES:
        raise ValueError(
            f"router diagnostic line {line_number} has unknown "
            f"diagnostic_code={diagnostic_code!r}"
        )

    worker_role = _require_string(record, "worker_role", line_number)
    if _SNAKE_CASE_PATTERN.fullmatch(worker_role) is None:
        raise ValueError(
            f"router diagnostic line {line_number} has non-snake-case "
            f"worker_role={worker_role!r}"
        )

    requirement = _require_string(record, "requirement", line_number)
    if _SNAKE_CASE_PATTERN.fullmatch(requirement) is None:
        raise ValueError(
            f"router diagnostic line {line_number} has non-snake-case "
            f"requirement={requirement!r}"
        )

    waited_ms = _require_int(record, "waited_ms", line_number)
    rank_count = _require_int(record, "rank_count", line_number)
    dp_ranks = _require_string(record, "dp_ranks", line_number)
    if waited_ms < 0:
        raise ValueError(
            f"router diagnostic line {line_number} waited_ms must be non-negative"
        )
    if rank_count < 0:
        raise ValueError(
            f"router diagnostic line {line_number} rank_count must be non-negative"
        )
    _validate_dp_ranks(dp_ranks, rank_count, line_number)

    return KvEventDiagnostic(
        diagnostic_code=cast(KvEventDiagnosticCode, diagnostic_code),
        model=_require_string(record, "model", line_number),
        worker_role=worker_role,
        requirement=requirement,
        worker_id=_require_int(record, "worker_id", line_number),
        serving_endpoint=_require_string(record, "serving_endpoint", line_number),
        kv_event_publishing_enabled=_require_bool(
            record, "kv_event_publishing_enabled", line_number
        ),
        waited_ms=waited_ms,
        rank_count=rank_count,
        dp_ranks=dp_ranks,
    )


def parse_kv_event_diagnostics(log_content: str) -> list[KvEventDiagnostic]:
    """Parse typed KV-event diagnostics, ignoring unrelated or partial log lines."""

    diagnostics = []
    for line_number, raw_line in enumerate(log_content.splitlines(), start=1):
        json_start = raw_line.find("{")
        if json_start < 0:
            continue

        try:
            record = json.loads(raw_line[json_start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or "diagnostic_code" not in record:
            continue

        diagnostics.append(_parse_diagnostic(record, line_number))
    return diagnostics


def select_kv_event_diagnostics(
    diagnostics: Sequence[KvEventDiagnostic],
    *,
    diagnostic_code: KvEventDiagnosticCode,
    worker_role: str | None = None,
) -> list[KvEventDiagnostic]:
    return [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.diagnostic_code == diagnostic_code
        and (worker_role is None or diagnostic.worker_role == worker_role)
    ]


def wait_for_kv_event_diagnostics(
    process: LogReadable,
    *,
    diagnostic_code: KvEventDiagnosticCode,
    expected_count: int = 1,
    worker_role: str | None = None,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.1,
) -> list[KvEventDiagnostic]:
    """Poll a process JSONL log until at least ``expected_count`` records appear."""

    if expected_count < 1:
        raise ValueError("expected_count must be at least one")

    deadline = time.monotonic() + timeout_s
    while True:
        diagnostics = select_kv_event_diagnostics(
            parse_kv_event_diagnostics(process.read_logs()),
            diagnostic_code=diagnostic_code,
            worker_role=worker_role,
        )
        if len(diagnostics) >= expected_count:
            return diagnostics
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for {expected_count} {diagnostic_code!r} "
                f"diagnostic(s) for worker_role={worker_role!r}; "
                f"observed {len(diagnostics)}"
            )
        time.sleep(poll_interval_s)
