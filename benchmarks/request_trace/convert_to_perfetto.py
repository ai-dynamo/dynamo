# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Dynamo request trace JSONL files to Perfetto-compatible JSON.

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


_TOOL_EVENT_TYPES = {"tool_start", "tool_end", "tool_error"}
_SYNTHETIC_TOOL_DURATION_US = 1_000
_SUPPORTED_SCHEMA = "dynamo.request.trace.v1"


def _event_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    event = record.get("event", record)
    if not isinstance(event, dict):
        return None
    if event.get("schema") != _SUPPORTED_SCHEMA:
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
        "trajectory_id": agent_context.get("trajectory_id"),
        "parent_trajectory_id": agent_context.get("parent_trajectory_id"),
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

    finish = request.get("finish_reason_metadata")
    if isinstance(finish, dict):
        args["finish_reason_metadata"] = finish
        for key in ("finish_reason", "backend_finish_reason", "stop_reason"):
            if key in finish:
                args[f"finish.{key}"] = finish[key]

        tool_calls = finish.get("tool_calls")
        if isinstance(tool_calls, list):
            args["finish.tool_call_count"] = len(tool_calls)
            names = [
                str(call["name"])
                for call in tool_calls
                if isinstance(call, dict) and call.get("name")
            ]
            if names:
                args["finish.tool_call_names"] = ", ".join(names)

        choices = finish.get("choices")
        if isinstance(choices, list):
            args["finish.choice_count"] = len(choices)
            finish_reasons = [
                f"{choice.get('choice_index')}:{choice.get('finish_reason')}"
                for choice in choices
                if isinstance(choice, dict)
                and choice.get("choice_index") is not None
                and choice.get("finish_reason") is not None
            ]
            if finish_reasons:
                args["finish.choice_finish_reasons"] = ", ".join(finish_reasons)

    return {key: value for key, value in args.items() if value is not None}


def _flatten_tool_args(
    event: dict[str, Any],
    agent_context: dict[str, Any],
    tool: dict[str, Any],
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "trajectory_id": agent_context.get("trajectory_id"),
        "parent_trajectory_id": agent_context.get("parent_trajectory_id"),
        "event_type": event.get("event_type"),
        "event_source": event.get("event_source"),
        "event_time_unix_ms": event.get("event_time_unix_ms"),
    }

    for key in (
        "tool_call_id",
        "tool_class",
        "started_at_unix_ms",
        "ended_at_unix_ms",
        "status",
        "duration_ms",
        "output_tokens",
        "output_bytes",
        "tool_name_hash",
        "error_type",
    ):
        if key in tool:
            args[key] = tool[key]

    return {key: value for key, value in args.items() if value is not None}


def _inferred_tool_calls(request: dict[str, Any]) -> list[dict[str, Any]]:
    finish = request.get("finish_reason_metadata")
    tool_calls = finish.get("tool_calls") if isinstance(finish, dict) else None
    if not isinstance(tool_calls, list):
        return []
    return [call for call in tool_calls if isinstance(call, dict)]


class TrackTable:
    def __init__(self) -> None:
        self._track_tids: dict[tuple[str, int, str], int] = {}
        self._active_lanes: dict[str, list[tuple[int, int]]] = {}
        self._max_lanes: dict[str, int] = {}

    def lane_for(
        self,
        trajectory_id: str,
        *,
        start_us: int,
        end_us: int,
    ) -> int:
        active = self._active_lanes.setdefault(trajectory_id, [])
        while active and active[0][0] <= start_us:
            heapq.heappop(active)

        active_lanes = {lane for _, lane in active}
        lane = 0
        while lane in active_lanes:
            lane += 1
        self._max_lanes[trajectory_id] = max(
            self._max_lanes.get(trajectory_id, 0), lane + 1
        )
        heapq.heappush(active, (end_us, lane))
        return lane

    def track_for(
        self,
        trajectory_id: str,
        lane: int,
        track_kind: str,
    ) -> tuple[int, int]:
        track_key = (trajectory_id, lane, track_kind)
        if track_key not in self._track_tids:
            self._track_tids[track_key] = len(self._track_tids) + 1
        return 1, self._track_tids[track_key]

    def metadata_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._track_tids:
            events.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": 1,
                    "args": {"name": "Dynamo request trace"},
                }
            )
        for (trajectory_id, lane, track_kind), tid in sorted(
            self._track_tids.items(),
            key=lambda item: item[1],
        ):
            lane_count = self._max_lanes.get(trajectory_id, 1)
            track_name = trajectory_id
            if lane_count > 1:
                track_name = f"{trajectory_id} [lane {lane + 1}]"
            if track_kind != "request":
                track_name = f"{track_name} {track_kind}"
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 1,
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


def _prepare_tool_items(tool_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    starts: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for record in sorted(tool_records, key=lambda item: item["event_time_us"]):
        tool = record["tool"]
        tool_call_id = _safe_label(tool.get("tool_call_id"), "unknown-tool-call")
        key = (record["trajectory_id"], tool_call_id)
        event_type = record["event_type"]

        if event_type == "tool_start":
            starts.setdefault(key, []).append(record)
            continue

        matched_start = None
        start_stack = starts.get(key)
        if start_stack:
            matched_start = start_stack.pop()

        event_time_us = record["event_time_us"]
        started_at_us = _ms_to_trace_us(tool.get("started_at_unix_ms"))
        ended_at_us = _ms_to_trace_us(tool.get("ended_at_unix_ms"))
        duration_us = _ms_to_trace_us(tool.get("duration_ms"))
        if (
            started_at_us is not None
            and ended_at_us is not None
            and ended_at_us > started_at_us
        ):
            ts_us = started_at_us
            dur_us = ended_at_us - started_at_us
        elif started_at_us is not None and duration_us is not None and duration_us > 0:
            ts_us = started_at_us
            dur_us = duration_us
        elif duration_us is not None and duration_us > 0:
            end_us = ended_at_us if ended_at_us is not None else event_time_us
            ts_us = max(0, end_us - duration_us)
            dur_us = end_us - ts_us
        elif (
            matched_start is not None and event_time_us > matched_start["event_time_us"]
        ):
            ts_us = matched_start["event_time_us"]
            dur_us = event_time_us - ts_us
        else:
            synthetic_ts_us = (
                matched_start["event_time_us"]
                if matched_start is not None
                else event_time_us
            )
            synthetic_args = {
                **record["args"],
                "synthetic_duration": True,
                "visual_duration_ms": _SYNTHETIC_TOOL_DURATION_US / 1000.0,
            }
            items.append(
                {
                    **record,
                    "kind": "tool",
                    "ts_us": synthetic_ts_us,
                    "dur_us": _SYNTHETIC_TOOL_DURATION_US,
                    "args": synthetic_args,
                }
            )
            continue

        items.append(
            {**record, "kind": "tool", "ts_us": ts_us, "dur_us": max(1, dur_us)}
        )

    for start_stack in starts.values():
        for record in start_stack:
            items.append(
                {**record, "kind": "tool_instant", "ts_us": record["event_time_us"]}
            )

    return items


def _prepare_inferred_tool_items(
    request_items: list[dict[str, Any]],
    tool_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    real_tool_keys = {
        (
            record["trajectory_id"],
            _safe_label(record["tool"].get("tool_call_id"), ""),
        )
        for record in tool_records
        if record["event_type"] in {"tool_end", "tool_error"}
        and record["tool"].get("tool_call_id")
    }

    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for item in request_items:
        by_trajectory.setdefault(item["trajectory_id"], []).append(item)

    for requests in by_trajectory.values():
        requests.sort(key=lambda item: item["ts_us"])
        for index, request_item in enumerate(requests):
            next_request_item = (
                requests[index + 1] if index + 1 < len(requests) else None
            )
            tool_calls = _inferred_tool_calls(request_item["request"])
            if not tool_calls:
                continue

            start_us = request_item["ts_us"] + request_item["dur_us"]
            end_us = next_request_item["ts_us"] if next_request_item else start_us
            duration_unknown = next_request_item is None or end_us <= start_us

            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id")
                if (
                    request_item["trajectory_id"],
                    _safe_label(tool_call_id, ""),
                ) in real_tool_keys:
                    continue

                tool_name = _safe_label(tool_call.get("name"), "unknown-tool")
                args = {
                    "trajectory_id": request_item["trajectory_id"],
                    "parent_trajectory_id": request_item["args"].get(
                        "parent_trajectory_id"
                    ),
                    "event_type": "inferred_tool_call",
                    "event_source": "dynamo",
                    "source": "request.finish_reason_metadata.tool_calls",
                    "inferred": True,
                    "host_execution": False,
                    "source_request_id": request_item["request"].get("request_id"),
                    "tool_call_id": tool_call_id,
                    "tool_call_index": tool_call.get("tool_call_index"),
                    "choice_index": tool_call.get("choice_index"),
                    "tool_class": tool_name,
                    "started_at_unix_ms": start_us / 1000.0,
                }
                if duration_unknown:
                    args.update(
                        {
                            "duration_unknown": True,
                            "synthetic_duration": True,
                            "visual_duration_ms": _SYNTHETIC_TOOL_DURATION_US / 1000.0,
                            "overlapping_request_id": (
                                next_request_item["request"].get("request_id")
                                if next_request_item
                                else None
                            ),
                        }
                    )
                    dur_us = _SYNTHETIC_TOOL_DURATION_US
                else:
                    args.update(
                        {
                            "next_request_id": next_request_item["request"].get(
                                "request_id"
                            ),
                            "ended_at_unix_ms": end_us / 1000.0,
                            "duration_ms": (end_us - start_us) / 1000.0,
                        }
                    )
                    dur_us = end_us - start_us

                items.append(
                    {
                        "kind": "inferred_tool",
                        "tool_class": tool_name,
                        "args": {
                            key: value
                            for key, value in args.items()
                            if value is not None
                        },
                        "ts_us": start_us,
                        "dur_us": dur_us,
                        "trajectory_id": request_item["trajectory_id"],
                    }
                )

    return items


def convert_records(
    records: Iterable[dict[str, Any]],
    *,
    include_stages: bool,
    include_markers: bool,
) -> tuple[dict[str, Any], int]:
    prepared: list[dict[str, Any]] = []
    request_items: list[dict[str, Any]] = []
    tool_records: list[dict[str, Any]] = []

    for record in records:
        event = _event_from_record(record)
        if event is None:
            continue

        agent_context = event.get("agent_context")
        has_agent_context = isinstance(agent_context, dict)
        agent_context = agent_context if has_agent_context else {}

        event_type = event.get("event_type")

        if event_type == "request_end":
            request = event.get("request")
            if not isinstance(request, dict):
                continue
            trajectory_id = _safe_label(
                agent_context.get("trajectory_id"), "request-only"
            )

            start_ms = _request_start_ms(event, request)
            if start_ms is None:
                continue
            duration_ms = _request_duration_ms(event, request, start_ms)
            ts_us = _ms_to_trace_us(start_ms)
            dur_us = _ms_to_trace_us(duration_ms)
            if ts_us is None or dur_us is None:
                continue

            item = {
                "kind": "request",
                "request": request,
                "args": _flatten_args(event, agent_context, request),
                "ts_us": ts_us,
                "dur_us": max(1, dur_us),
                "trajectory_id": trajectory_id,
            }
            prepared.append(item)
            request_items.append(item)
            continue

        if event_type in _TOOL_EVENT_TYPES:
            if not has_agent_context:
                continue
            tool = event.get("tool")
            event_time_us = _ms_to_trace_us(event.get("event_time_unix_ms"))
            if not isinstance(tool, dict) or event_time_us is None:
                continue
            trajectory_id = _safe_label(
                agent_context.get("trajectory_id"), "unknown-trajectory"
            )

            tool_records.append(
                {
                    "tool": tool,
                    "event_type": event_type,
                    "args": _flatten_tool_args(event, agent_context, tool),
                    "event_time_us": event_time_us,
                    "trajectory_id": trajectory_id,
                }
            )

    prepared.extend(_prepare_inferred_tool_items(request_items, tool_records))
    prepared.extend(_prepare_tool_items(tool_records))

    tracks = TrackTable()
    trace_events: list[dict[str, Any]] = []
    converted = 0

    for item in sorted(prepared, key=lambda item: (item["ts_us"], item["kind"])):
        args = item["args"]
        ts_us = item["ts_us"]
        dur_us = item.get("dur_us", 1)
        lane = tracks.lane_for(
            item["trajectory_id"],
            start_us=ts_us,
            end_us=ts_us + dur_us,
        )

        if item["kind"] == "tool":
            tool = item["tool"]
            tool_pid, tool_tid = tracks.track_for(
                item["trajectory_id"],
                lane,
                "tools",
            )
            event_type = item["event_type"]
            trace_events.append(
                _make_complete_event(
                    name=("Tool error: " if event_type == "tool_error" else "Tool: ")
                    + _safe_label(tool.get("tool_class"), "unknown-tool"),
                    category="dynamo.agent.tool",
                    pid=tool_pid,
                    tid=tool_tid,
                    ts_us=ts_us,
                    dur_us=dur_us,
                    args=args,
                )
            )
            converted += 1
            continue

        if item["kind"] == "tool_instant":
            tool = item["tool"]
            tool_pid, tool_tid = tracks.track_for(
                item["trajectory_id"],
                lane,
                "tools",
            )
            trace_events.append(
                _make_instant_event(
                    name=(
                        "Tool start: "
                        if item["event_type"] == "tool_start"
                        else "Tool: "
                    )
                    + _safe_label(tool.get("tool_class"), "unknown-tool"),
                    category="dynamo.agent.tool.marker",
                    pid=tool_pid,
                    tid=tool_tid,
                    ts_us=ts_us,
                    args=args,
                )
            )
            converted += 1
            continue

        if item["kind"] == "inferred_tool":
            tool_pid, tool_tid = tracks.track_for(
                item["trajectory_id"],
                lane,
                "tools",
            )
            trace_events.append(
                _make_complete_event(
                    name="Tool: " + item["tool_class"],
                    category="dynamo.agent.tool",
                    pid=tool_pid,
                    tid=tool_tid,
                    ts_us=ts_us,
                    dur_us=dur_us,
                    args=args,
                )
            )
            converted += 1
            continue

        request = item["request"]
        request_pid, request_tid = tracks.track_for(
            item["trajectory_id"],
            lane,
            "request",
        )
        stage_pid, stage_tid = request_pid, request_tid
        trace_events.append(
            _make_complete_event(
                name=(
                    f"LLM request: {_safe_label(request.get('model'), 'unknown-model')}"
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
        description="Convert Dynamo request trace JSONL/JSONL.GZ files to Perfetto JSON.",
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
        "--no-stages",
        action="store_false",
        dest="include_stages",
        help="Emit only full LLM request slices.",
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
        f"wrote {converted} trace events from {len(input_paths)} input file(s) to {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
