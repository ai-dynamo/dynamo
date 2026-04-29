# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Dynamo agent trace JSONL files to Perfetto-compatible JSON.

The output uses Chrome Trace Event JSON, which can be opened directly in
Perfetto UI. Inputs can be uncompressed `.jsonl`, compressed `.jsonl.gz`,
directories containing trace shards, or glob patterns.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import heapq
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        expanded = [Path(path) for path in sorted(glob.glob(item))]
        candidates = expanded or [Path(item)]
        for candidate in candidates:
            if candidate.is_dir():
                paths.extend(sorted(candidate.glob("*.jsonl")))
                paths.extend(sorted(candidate.glob("*.jsonl.gz")))
            else:
                paths.append(candidate)
    return sorted(dict.fromkeys(paths))


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_records(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with _open_text(path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if isinstance(record, dict):
                    yield record


def _event_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    event = record.get("event", record)
    if not isinstance(event, dict):
        return None
    if event.get("schema") != "dynamo.agent.trace.v1":
        return None
    if event.get("event_type") != "request_end":
        return None
    return event


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ms_to_trace_us(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(round(number * 1000.0))


def _safe_label(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    label = str(value)
    return label if label else fallback


def _request_start_ms(event: dict[str, Any], request: dict[str, Any]) -> float | None:
    start = _as_float(request.get("request_received_ms"))
    if start is not None:
        return start
    return _as_float(event.get("event_time_unix_ms"))


def _request_duration_ms(
    event: dict[str, Any], request: dict[str, Any], start_ms: float
) -> float:
    duration = _as_float(request.get("total_time_ms"))
    if duration is not None and duration >= 0:
        return duration
    end_ms = _as_float(event.get("event_time_unix_ms"))
    if end_ms is not None and end_ms >= start_ms:
        return end_ms - start_ms
    return 0.001


def _flatten_args(
    event: dict[str, Any],
    agent_context: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "workflow_type_id": agent_context.get("workflow_type_id"),
        "workflow_id": agent_context.get("workflow_id"),
        "program_id": agent_context.get("program_id"),
        "parent_program_id": agent_context.get("parent_program_id"),
        "event_time_unix_ms": event.get("event_time_unix_ms"),
    }

    for key in (
        "request_id",
        "x_request_id",
        "model",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "request_received_ms",
        "prefill_wait_time_ms",
        "prefill_time_ms",
        "ttft_ms",
        "total_time_ms",
        "avg_itl_ms",
        "kv_hit_rate",
        "kv_transfer_estimated_latency_ms",
        "queue_depth",
    ):
        if key in request:
            args[key] = request[key]

    worker = request.get("worker")
    if isinstance(worker, dict):
        for key, value in worker.items():
            args[f"worker.{key}"] = value

    return {key: value for key, value in args.items() if value is not None}


class TrackTable:
    def __init__(self) -> None:
        self._workflow_pids: dict[str, int] = {}
        self._track_tids: dict[tuple[str, str, int, str], int] = {}
        self._active_lanes: dict[tuple[str, str], list[tuple[int, int]]] = {}
        self._next_lane: dict[tuple[str, str], int] = {}
        self._max_lanes: dict[tuple[str, str], int] = {}

    def lane_for(
        self,
        workflow_id: str,
        program_id: str,
        *,
        start_us: int,
        end_us: int,
    ) -> int:
        if workflow_id not in self._workflow_pids:
            self._workflow_pids[workflow_id] = len(self._workflow_pids) + 1

        program_key = (workflow_id, program_id)
        active = self._active_lanes.setdefault(program_key, [])
        while active and active[0][0] <= start_us:
            heapq.heappop(active)

        active_lanes = {lane for _, lane in active}
        lane = 0
        while lane in active_lanes:
            lane += 1
        if lane >= self._next_lane.get(program_key, 0):
            self._next_lane[program_key] = lane + 1
        self._max_lanes[program_key] = max(
            self._max_lanes.get(program_key, 0), lane + 1
        )
        heapq.heappush(active, (end_us, lane))
        return lane

    def track_for(
        self,
        workflow_id: str,
        program_id: str,
        lane: int,
        track_kind: str,
    ) -> tuple[int, int]:
        if workflow_id not in self._workflow_pids:
            self._workflow_pids[workflow_id] = len(self._workflow_pids) + 1
        pid = self._workflow_pids[workflow_id]

        track_key = (workflow_id, program_id, lane, track_kind)
        if track_key not in self._track_tids:
            self._track_tids[track_key] = (
                len([1 for existing in self._track_tids if existing[0] == workflow_id])
                + 1
            )
        return pid, self._track_tids[track_key]

    def metadata_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for workflow_id, pid in sorted(
            self._workflow_pids.items(), key=lambda item: item[1]
        ):
            events.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "args": {"name": f"workflow: {workflow_id}"},
                }
            )
        for (workflow_id, program_id, lane, track_kind), tid in sorted(
            self._track_tids.items(),
            key=lambda item: (self._workflow_pids[item[0][0]], item[1]),
        ):
            pid = self._workflow_pids[workflow_id]
            lane_count = self._max_lanes.get((workflow_id, program_id), 1)
            track_name = program_id
            if lane_count > 1:
                track_name = f"{program_id} [lane {lane + 1}]"
            if track_kind != "request":
                track_name = f"{track_name} {track_kind}"
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": track_name},
                }
            )
        return events


def _make_complete_event(
    *,
    name: str,
    category: str,
    pid: int,
    tid: int,
    ts_us: int,
    dur_us: int,
    args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": ts_us,
        "dur": max(1, dur_us),
        "args": args,
    }


def _make_instant_event(
    *,
    name: str,
    category: str,
    pid: int,
    tid: int,
    ts_us: int,
    args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "cat": category,
        "ph": "i",
        "s": "t",
        "pid": pid,
        "tid": tid,
        "ts": ts_us,
        "args": args,
    }


def _bounded_stage_duration(
    value_us: int | None,
    *,
    cursor_us: int,
    boundary_us: int,
) -> int | None:
    if value_us is None or value_us <= 0 or cursor_us >= boundary_us:
        return None
    return min(value_us, boundary_us - cursor_us)


def convert_records(
    records: Iterable[dict[str, Any]],
    *,
    include_stages: bool,
    include_markers: bool,
    separate_stage_tracks: bool = False,
) -> tuple[dict[str, Any], int]:
    prepared: list[dict[str, Any]] = []

    for record in records:
        event = _event_from_record(record)
        if event is None:
            continue

        agent_context = event.get("agent_context")
        request = event.get("request")
        if not isinstance(agent_context, dict) or not isinstance(request, dict):
            continue

        start_ms = _request_start_ms(event, request)
        if start_ms is None:
            continue
        duration_ms = _request_duration_ms(event, request, start_ms)
        ts_us = _ms_to_trace_us(start_ms)
        dur_us = _ms_to_trace_us(duration_ms)
        if ts_us is None or dur_us is None:
            continue

        prepared.append(
            {
                "request": request,
                "args": _flatten_args(event, agent_context, request),
                "ts_us": ts_us,
                "dur_us": max(1, dur_us),
                "workflow_id": _safe_label(
                    agent_context.get("workflow_id"), "unknown-workflow"
                ),
                "program_id": _safe_label(
                    agent_context.get("program_id"), "unknown-program"
                ),
            }
        )

    tracks = TrackTable()
    trace_events: list[dict[str, Any]] = []
    converted = 0

    for item in sorted(prepared, key=lambda item: item["ts_us"]):
        request = item["request"]
        args = item["args"]
        ts_us = item["ts_us"]
        dur_us = item["dur_us"]
        lane = tracks.lane_for(
            item["workflow_id"],
            item["program_id"],
            start_us=ts_us,
            end_us=ts_us + dur_us,
        )
        request_pid, request_tid = tracks.track_for(
            item["workflow_id"],
            item["program_id"],
            lane,
            "request",
        )
        stage_pid, stage_tid = (
            tracks.track_for(item["workflow_id"], item["program_id"], lane, "stages")
            if include_stages and separate_stage_tracks
            else (request_pid, request_tid)
        )

        trace_events.append(
            _make_complete_event(
                name=(
                    "LLM request: "
                    f"{_safe_label(request.get('model'), 'unknown-model')}"
                ),
                category="dynamo.llm",
                pid=request_pid,
                tid=request_tid,
                ts_us=ts_us,
                dur_us=dur_us,
                args=args,
            )
        )
        converted += 1

        ttft_us = _ms_to_trace_us(request.get("ttft_ms"))
        if include_markers and ttft_us is not None:
            trace_events.append(
                _make_instant_event(
                    name="first token",
                    category="dynamo.llm.marker",
                    pid=stage_pid,
                    tid=stage_tid,
                    ts_us=ts_us + ttft_us,
                    args={"ttft_ms": request.get("ttft_ms")},
                )
            )

        if include_stages:
            wait_us = _ms_to_trace_us(request.get("prefill_wait_time_ms"))
            prefill_us = _ms_to_trace_us(request.get("prefill_time_ms"))
            ttft_boundary_us = dur_us
            if ttft_us is not None:
                ttft_boundary_us = min(max(0, ttft_us), dur_us)
            stage_cursor_us = 0
            common_stage_args = {
                "request_id": request.get("request_id"),
                "x_request_id": request.get("x_request_id"),
                "model": request.get("model"),
            }
            # Chrome trace complete events on the same thread must form a valid
            # stack. Clamp visualization boundaries to avoid 1us overlaps from
            # independently rounded metrics while keeping raw values in args.
            wait_dur_us = _bounded_stage_duration(
                wait_us,
                cursor_us=stage_cursor_us,
                boundary_us=ttft_boundary_us,
            )
            if wait_dur_us is not None:
                trace_events.append(
                    _make_complete_event(
                        name="prefill wait",
                        category="dynamo.llm.stage",
                        pid=stage_pid,
                        tid=stage_tid,
                        ts_us=ts_us + stage_cursor_us,
                        dur_us=wait_dur_us,
                        args={
                            **common_stage_args,
                            "prefill_wait_time_ms": request.get("prefill_wait_time_ms"),
                        },
                    )
                )
                stage_cursor_us += wait_dur_us

            prefill_dur_us = _bounded_stage_duration(
                prefill_us,
                cursor_us=stage_cursor_us,
                boundary_us=ttft_boundary_us,
            )
            if prefill_dur_us is not None:
                trace_events.append(
                    _make_complete_event(
                        name="prefill",
                        category="dynamo.llm.stage",
                        pid=stage_pid,
                        tid=stage_tid,
                        ts_us=ts_us + stage_cursor_us,
                        dur_us=prefill_dur_us,
                        args={
                            **common_stage_args,
                            "prefill_time_ms": request.get("prefill_time_ms"),
                        },
                    )
                )
                stage_cursor_us += prefill_dur_us

            if ttft_us is not None and dur_us > ttft_boundary_us:
                trace_events.append(
                    _make_complete_event(
                        name="decode",
                        category="dynamo.llm.stage",
                        pid=stage_pid,
                        tid=stage_tid,
                        ts_us=ts_us + ttft_boundary_us,
                        dur_us=dur_us - ttft_boundary_us,
                        args={
                            **common_stage_args,
                            "output_tokens": request.get("output_tokens"),
                            "avg_itl_ms": request.get("avg_itl_ms"),
                        },
                    )
                )

    trace_events = tracks.metadata_events() + sorted(
        trace_events,
        key=lambda item: (item.get("ts", 0), item.get("pid", 0), item.get("tid", 0)),
    )
    return {
        "displayTimeUnit": "ms",
        "traceEvents": trace_events,
    }, converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Dynamo agent trace JSONL/JSONL.GZ files to Perfetto JSON.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input .jsonl/.jsonl.gz files, directories, or glob patterns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output Chrome Trace JSON path for Perfetto UI.",
    )
    parser.add_argument(
        "--include-stages",
        action="store_true",
        default=True,
        help=(
            "Emit prefill wait, prefill, and decode stage slices. This is the "
            "default; kept for compatibility."
        ),
    )
    parser.add_argument(
        "--no-stages",
        action="store_false",
        dest="include_stages",
        help="Emit only full LLM request slices.",
    )
    parser.add_argument(
        "--separate-stage-tracks",
        action="store_true",
        help=(
            "Place stage slices on adjacent stage tracks instead of stacking "
            "them under the request slice."
        ),
    )
    parser.add_argument(
        "--include-markers",
        action="store_true",
        help="Also emit first-token instant markers. Disabled by default to reduce clutter.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print output JSON. Defaults to compact JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = _expand_inputs(args.inputs)
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        print(f"error: input path does not exist: {missing[0]}", file=sys.stderr)
        return 2
    if not input_paths:
        print("error: no input files matched", file=sys.stderr)
        return 2

    trace, converted = convert_records(
        _iter_records(input_paths),
        include_stages=args.include_stages,
        include_markers=args.include_markers,
        separate_stage_tracks=args.separate_stage_tracks,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(trace, f, indent=2, sort_keys=True)
            f.write("\n")
        else:
            json.dump(trace, f, separators=(",", ":"))
            f.write("\n")

    print(
        f"wrote {converted} request events from {len(input_paths)} input file(s) to {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
