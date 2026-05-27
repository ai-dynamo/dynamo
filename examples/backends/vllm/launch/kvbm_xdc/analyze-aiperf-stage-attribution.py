#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Join AIPerf per-request records to KVBM hub role logs.

This is intentionally local to the KVBM cross-datacenter harness. It depends on
the Dynamo frontend request log shape and the KVBM `kvbm_audit` event names used
by this example, so it is not a generic AIPerf parser.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Iterable

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAW_TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+"
    r"\w+\s+(?P<rest>.*)$"
)
FIELD_RE = re.compile(r"(?P<key>\w+)=(?:\"(?P<qv>(?:[^\"\\]|\\.)*)\"|(?P<bv>\S+))")
X_REQUEST_RE = re.compile(r"\bx_request_id=\"?(?P<value>[^\"\s]+)\"?")
REQUEST_RECEIVED_RE = re.compile(r"\brequest received request_id=(?P<value>[0-9a-f-]{36})\b")
REQUEST_COMPLETED_RE = re.compile(
    r"\brequest completed request_id=(?P<value>[0-9a-f-]{36})\b"
)
UUID_PREFIX_RE = re.compile(
    r"^(?P<value>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})(?:-|$)"
)
URL_RE = re.compile(r"\b(?:https?|nats)://[^\s\"'<>]+")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,}[0-9a-fA-F:]{1,}(?:%\w+)?(?::\d+)?\b")
ABS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"/(?:Users|home|tmp|private|workspace|mnt|scratch|var|opt)"
    r"/[^\s\"'<>:,;)]*"
)
INTERNAL_HOST_RE = re.compile(
    r"\b(?=[A-Za-z0-9.-]*[.-])(?=[A-Za-z0-9.-]*\d)"
    r"[A-Za-z0-9.-]*(?:prod|cluster|login|compute|gpu|dgx|h100|a100)"
    r"[A-Za-z0-9.-]*(?::\d+)?\b",
    re.IGNORECASE,
)

STAGE_EVENTS = {
    "commit_usaa1_state_built": "commit",
    "worker_session_pull_call": "pull_call",
    "worker_session_pull_returned": "pull_return",
    "worker_g2_to_g1_done": "g2_to_g1",
    "mark_onboarding_complete": "mark_complete",
}
FAILURE_PATTERNS = {
    "kv_load_failure": re.compile(
        r"KV load failure|Onboarding failed|Failed to start onboarding|kv_load_failure"
    ),
    "nixl_create_xfer_req_failure": re.compile(
        r"createXferReq: no potential backend found|nixl_create_xfer_req_failures"
    ),
}
ZERO_SUMMARY_PATTERNS = {
    "kv_load_failure": re.compile(r"KV load failure events:\s*0|kv_load_failure_events=0"),
    "nixl_create_xfer_req_failure": re.compile(
        r"createXferReq failures:\s*0|nixl_create_xfer_req_failures=0"
    ),
}


@dataclass
class AiperfRecord:
    line_no: int
    session_num: str
    x_request_id: str
    request_start_ns: int
    ttft_ms: float | None

    @property
    def start_ms(self) -> float:
        return self.request_start_ns / 1_000_000

    @property
    def first_token_ms(self) -> float | None:
        if self.ttft_ms is None:
            return None
        return self.start_ms + self.ttft_ms


@dataclass
class FrontendRecord:
    request_id: str | None = None
    recv_ms: float | None = None
    select_ms: float | None = None
    completed_ms: float | None = None
    ttft_ms: float | None = None


@dataclass
class BackendRecord:
    recv_ms: float | None = None
    events: dict[str, float] = field(default_factory=dict)
    remote_slots_len: int | None = None
    expected_remote_hashes_len: int | None = None
    worker_pull_filled: int | None = None
    worker_g2_to_g1_blocks: int | None = None


@dataclass
class ParsedLogs:
    frontend_by_x_request_id: dict[str, FrontendRecord] = field(default_factory=dict)
    backend_by_request_id: dict[str, BackendRecord] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in FAILURE_PATTERNS}
    )


CSV_COLUMNS = [
    "order",
    "session_num",
    "x_request_id",
    "backend_request_id",
    "cohort",
    "aiperf_start_utc",
    "aiperf_first_token_utc",
    "aiperf_ttft_ms",
    "frontend_ttft_ms",
    "remote_slots_len",
    "expected_remote_hashes_len",
    "worker_pull_filled",
    "worker_g2_to_g1_blocks",
    "frontend_recv_to_select_ms",
    "frontend_recv_to_backend_recv_ms",
    "backend_recv_to_commit_ms",
    "commit_to_pull_call_ms",
    "pull_call_to_pull_return_ms",
    "pull_return_to_g2_to_g1_ms",
    "g2_to_g1_to_mark_complete_ms",
    "mark_complete_to_first_token_ms",
    "frontend_recv_to_first_token_ms",
    "commit_to_first_token_ms",
]


def scrub_text(value: str) -> str:
    value = URL_RE.sub("<url>", value)
    value = IPV4_RE.sub("<addr>", value)
    value = IPV6_RE.sub("<addr>", value)
    value = ABS_PATH_RE.sub("<path>", value)
    value = INTERNAL_HOST_RE.sub("<host>", value)
    return value


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_fields(rest: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in FIELD_RE.finditer(rest):
        value = match.group("qv") if match.group("qv") is not None else match.group("bv")
        if value is None:
            continue
        if match.group("qv") is not None:
            value = value.replace('\\"', '"').replace("\\\\", "\\")
        else:
            value = value.rstrip(",")
        fields[match.group("key")] = value
    return fields


def parse_ts_ms(line: str) -> float | None:
    match = RAW_TS_RE.match(line)
    if not match:
        return None
    try:
        ts = datetime.fromisoformat(match.group("ts").replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts.timestamp() * 1000


def ns_to_utc(ns: int | None) -> str:
    if ns is None:
        return ""
    dt = datetime.fromtimestamp(ns / 1_000_000_000, timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def ms_to_utc(ms: float | None) -> str:
    if ms is None:
        return ""
    dt = datetime.fromtimestamp(ms / 1000, timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def diff_ms(start_ms: float | None, end_ms: float | None) -> float | None:
    if start_ms is None or end_ms is None:
        return None
    return end_ms - start_ms


def request_prefix(request_id: str | None) -> str | None:
    if not request_id:
        return None
    match = UUID_PREFIX_RE.match(request_id)
    if not match:
        return None
    return match.group("value")


def first_request_id(pattern: re.Pattern[str], line: str) -> str | None:
    match = pattern.search(line)
    if not match:
        return None
    return match.group("value")


def line_x_request_id(line: str) -> str | None:
    match = X_REQUEST_RE.search(line)
    if not match:
        return None
    return match.group("value")


def load_aiperf_records(profile_jsonl: Path) -> list[AiperfRecord]:
    records: list[AiperfRecord] = []
    with profile_jsonl.open(errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{profile_jsonl}:{line_no}: invalid JSON: {exc}") from exc
            metadata = data.get("metadata", {})
            metrics = data.get("metrics", {})
            x_request_id = metadata.get("x_request_id")
            request_start_ns = parse_int(str(metadata.get("request_start_ns", "")))
            ttft_metric = metrics.get("time_to_first_token", {})
            ttft_ms = None
            if isinstance(ttft_metric, dict):
                ttft_ms = parse_float(str(ttft_metric.get("value", "")))
            if not x_request_id or request_start_ns is None:
                continue
            records.append(
                AiperfRecord(
                    line_no=line_no,
                    session_num=str(metadata.get("session_num", "")),
                    x_request_id=str(x_request_id),
                    request_start_ns=request_start_ns,
                    ttft_ms=ttft_ms,
                )
            )
    return sorted(records, key=lambda record: (record.request_start_ns, record.line_no))


def iter_log_files(log_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in log_dir.rglob("*.log") if path.is_file())


def parse_role_logs(log_dir: Path) -> ParsedLogs:
    parsed = ParsedLogs()
    for log_file in iter_log_files(log_dir):
        with log_file.open(errors="replace") as handle:
            for raw_line in handle:
                line = strip_ansi(raw_line.rstrip("\n"))
                ts_ms = parse_ts_ms(line)
                if ts_ms is None:
                    continue
                count_failures(parsed, line)
                parse_frontend_line(parsed, line, ts_ms)
                parse_backend_received_line(parsed, line, ts_ms)
                parse_kvbm_audit_line(parsed, line, ts_ms)
    return parsed


def count_failures(parsed: ParsedLogs, line: str) -> None:
    for name, pattern in FAILURE_PATTERNS.items():
        if not pattern.search(line):
            continue
        if ZERO_SUMMARY_PATTERNS[name].search(line):
            continue
        parsed.failure_counts[name] += 1


def parse_frontend_line(parsed: ParsedLogs, line: str, ts_ms: float) -> None:
    x_request_id = line_x_request_id(line)
    if not x_request_id:
        return
    record = parsed.frontend_by_x_request_id.setdefault(x_request_id, FrontendRecord())

    request_id = first_request_id(REQUEST_RECEIVED_RE, line)
    if request_id:
        record.request_id = request_id
        if record.recv_ms is None:
            record.recv_ms = ts_ms

    completed_id = first_request_id(REQUEST_COMPLETED_RE, line)
    if completed_id:
        record.request_id = record.request_id or completed_id
        record.completed_ms = ts_ms
        fields = parse_fields(line)
        ttft_ms = parse_float(fields.get("ttft_ms"))
        if ttft_ms is not None:
            record.ttft_ms = ttft_ms

    if "Selected worker:" in line and record.select_ms is None:
        record.select_ms = ts_ms


def parse_backend_received_line(parsed: ParsedLogs, line: str, ts_ms: float) -> None:
    if "request received" not in line:
        return
    if "component=\"backend\"" not in line and "component=backend" not in line:
        return
    request_id = first_request_id(REQUEST_RECEIVED_RE, line)
    if not request_id:
        return
    backend = parsed.backend_by_request_id.setdefault(request_id, BackendRecord())
    if backend.recv_ms is None:
        backend.recv_ms = ts_ms


def parse_kvbm_audit_line(parsed: ParsedLogs, line: str, ts_ms: float) -> None:
    if "kvbm_audit:" not in line:
        return
    rest = line.split("kvbm_audit:", 1)[1]
    fields = parse_fields(rest)
    event = fields.get("event")
    key = STAGE_EVENTS.get(event or "")
    request_id = fields.get("request_id")
    base_request_id = request_prefix(request_id)
    if not key or not base_request_id:
        return

    backend = parsed.backend_by_request_id.setdefault(base_request_id, BackendRecord())
    backend.events.setdefault(key, ts_ms)
    if event == "commit_usaa1_state_built":
        backend.remote_slots_len = parse_int(fields.get("remote_slots_len"))
        backend.expected_remote_hashes_len = parse_int(
            fields.get("expected_remote_hashes_len")
        )
    elif event == "worker_session_pull_returned":
        backend.worker_pull_filled = parse_int(fields.get("num_filled"))
    elif event == "worker_g2_to_g1_done":
        backend.worker_g2_to_g1_blocks = parse_int(fields.get("num_blocks"))


def classify_cohort(remote_slots_len: int | None) -> str:
    if remote_slots_len is None:
        return "missing_remote_slots"
    if remote_slots_len > 0:
        return "remote_slots_nonzero"
    return "remote_slots_empty"


def build_rows(aiperf_records: list[AiperfRecord], logs: ParsedLogs) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for order, record in enumerate(aiperf_records, start=1):
        frontend = logs.frontend_by_x_request_id.get(record.x_request_id, FrontendRecord())
        backend_request_id = frontend.request_id or ""
        backend = logs.backend_by_request_id.get(backend_request_id, BackendRecord())
        first_token_ms = record.first_token_ms
        commit_ms = backend.events.get("commit")
        pull_call_ms = backend.events.get("pull_call")
        pull_return_ms = backend.events.get("pull_return")
        g2_to_g1_ms = backend.events.get("g2_to_g1")
        mark_complete_ms = backend.events.get("mark_complete")

        rows.append(
            {
                "order": str(order),
                "session_num": record.session_num,
                "x_request_id": record.x_request_id,
                "backend_request_id": backend_request_id,
                "cohort": classify_cohort(backend.remote_slots_len),
                "aiperf_start_utc": ns_to_utc(record.request_start_ns),
                "aiperf_first_token_utc": ms_to_utc(first_token_ms),
                "aiperf_ttft_ms": format_float(record.ttft_ms),
                "frontend_ttft_ms": format_float(frontend.ttft_ms),
                "remote_slots_len": format_int(backend.remote_slots_len),
                "expected_remote_hashes_len": format_int(
                    backend.expected_remote_hashes_len
                ),
                "worker_pull_filled": format_int(backend.worker_pull_filled),
                "worker_g2_to_g1_blocks": format_int(backend.worker_g2_to_g1_blocks),
                "frontend_recv_to_select_ms": format_float(
                    diff_ms(frontend.recv_ms, frontend.select_ms)
                ),
                "frontend_recv_to_backend_recv_ms": format_float(
                    diff_ms(frontend.recv_ms, backend.recv_ms)
                ),
                "backend_recv_to_commit_ms": format_float(
                    diff_ms(backend.recv_ms, commit_ms)
                ),
                "commit_to_pull_call_ms": format_float(diff_ms(commit_ms, pull_call_ms)),
                "pull_call_to_pull_return_ms": format_float(
                    diff_ms(pull_call_ms, pull_return_ms)
                ),
                "pull_return_to_g2_to_g1_ms": format_float(
                    diff_ms(pull_return_ms, g2_to_g1_ms)
                ),
                "g2_to_g1_to_mark_complete_ms": format_float(
                    diff_ms(g2_to_g1_ms, mark_complete_ms)
                ),
                "mark_complete_to_first_token_ms": format_float(
                    diff_ms(mark_complete_ms, first_token_ms)
                ),
                "frontend_recv_to_first_token_ms": format_float(
                    diff_ms(frontend.recv_ms, first_token_ms)
                ),
                "commit_to_first_token_ms": format_float(
                    diff_ms(commit_ms, first_token_ms)
                ),
            }
        )
    return rows


def format_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def row_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def summarize_values(rows: list[dict[str, str]], key: str) -> dict[str, float | int | None]:
    values = [value for row in rows if (value := row_float(row, key)) is not None]
    if not values:
        return {"count": 0, "avg": None, "p50": None, "p90": None}
    return {
        "count": len(values),
        "avg": mean(values),
        "p50": median(values),
        "p90": percentile(values, 0.90),
    }


def cohort_rows(rows: list[dict[str, str]], cohort: str) -> list[dict[str, str]]:
    return [row for row in rows if row["cohort"] == cohort]


def build_summary(
    rows: list[dict[str, str]],
    failure_counts: dict[str, int],
    label: str | None,
) -> tuple[str, str]:
    cohorts = [
        "remote_slots_nonzero",
        "remote_slots_empty",
        "missing_remote_slots",
    ]
    joined = sum(1 for row in rows if row["backend_request_id"])
    env_lines = [
        f"requests_total={len(rows)}",
        f"joined_request_ids={joined}",
    ]
    markdown_lines = [
        "# AIPerf/KVBM Stage Attribution",
        "",
    ]
    if label:
        markdown_lines.extend([f"Label: `{scrub_text(label)}`", ""])
    markdown_lines.extend(
        [
            "This report joins per-request AIPerf JSONL records to Dynamo frontend "
            "and KVBM audit role logs. Stage timings are observed correlations from "
            "shared request IDs and timestamps; the remote pull interval can include "
            "network, hub dispatch, remote worker scheduling, and transport setup.",
            "",
            f"- Requests in AIPerf JSONL: {len(rows)}",
            f"- Requests joined to frontend/backend logs: {joined}",
            f"- KV load failure evidence: {failure_counts['kv_load_failure']}",
            f"- NIXL createXferReq failure evidence: {failure_counts['nixl_create_xfer_req_failure']}",
            "",
            "| Cohort | Requests | TTFT avg ms | TTFT p50 ms | TTFT p90 ms | Pull call->return avg ms | Pull call->return p50 ms | Pull call->return p90 ms | Mark complete->first token avg ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for cohort in cohorts:
        subset = cohort_rows(rows, cohort)
        ttft = summarize_values(subset, "aiperf_ttft_ms")
        pull = summarize_values(subset, "pull_call_to_pull_return_ms")
        mark_to_token = summarize_values(subset, "mark_complete_to_first_token_ms")
        env_prefix = cohort
        env_lines.append(f"{env_prefix}_count={len(subset)}")
        append_metric_env(env_lines, env_prefix, "ttft", ttft)
        append_metric_env(env_lines, env_prefix, "pull_call_to_return", pull)
        append_metric_env(env_lines, env_prefix, "mark_complete_to_first_token", mark_to_token)
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    cohort,
                    str(len(subset)),
                    metric_cell(ttft["avg"]),
                    metric_cell(ttft["p50"]),
                    metric_cell(ttft["p90"]),
                    metric_cell(pull["avg"]),
                    metric_cell(pull["p50"]),
                    metric_cell(pull["p90"]),
                    metric_cell(mark_to_token["avg"]),
                ]
            )
            + " |"
        )

    for name, count in failure_counts.items():
        env_lines.append(f"{name}={count}")

    markdown_lines.extend(
        [
            "",
            "Use `remote_slots_nonzero` as the KVBM remote-pull cohort and "
            "`remote_slots_empty` as the reuse/local cohort. Do not treat this "
            "report alone as proof of WAN latency; it narrows the observed "
            "slowdown to the KVBM pull path interval.",
            "",
        ]
    )
    return "\n".join(env_lines) + "\n", "\n".join(markdown_lines)


def append_metric_env(
    lines: list[str],
    prefix: str,
    metric_name: str,
    summary: dict[str, float | int | None],
) -> None:
    for key in ("avg", "p50", "p90"):
        value = summary[key]
        lines.append(f"{prefix}_{metric_name}_{key}_ms={metric_cell(value)}")


def metric_cell(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join AIPerf profile_export.jsonl records with KVBM XDC role logs."
    )
    parser.add_argument(
        "--profile-jsonl",
        required=True,
        type=Path,
        help="Path to AIPerf per-request profile_export.jsonl.",
    )
    parser.add_argument(
        "--role-log-dir",
        required=True,
        type=Path,
        help="Directory containing collected frontend/decode/prefill/hub .log files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for stage-attribution.csv, .env, and .md. Defaults to profile_jsonl parent / analysis.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional report label. Hostnames, IPs, URLs, and absolute paths are scrubbed in markdown output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_jsonl = args.profile_jsonl
    role_log_dir = args.role_log_dir
    output_dir = args.output_dir or profile_jsonl.parent / "analysis"

    if not profile_jsonl.is_file():
        raise SystemExit(f"profile JSONL not found: {profile_jsonl}")
    if not role_log_dir.is_dir():
        raise SystemExit(f"role log directory not found: {role_log_dir}")

    aiperf_records = load_aiperf_records(profile_jsonl)
    if not aiperf_records:
        raise SystemExit(f"no per-request records found in {profile_jsonl}")

    logs = parse_role_logs(role_log_dir)
    rows = build_rows(aiperf_records, logs)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "stage-attribution.csv")
    env_text, md_text = build_summary(rows, logs.failure_counts, args.label or None)
    (output_dir / "stage-attribution-summary.env").write_text(env_text)
    (output_dir / "stage-attribution-summary.md").write_text(md_text)

    joined = sum(1 for row in rows if row["backend_request_id"])
    print(
        "Wrote stage attribution: "
        f"requests={len(rows)} joined={joined} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
