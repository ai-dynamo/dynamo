# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone interactive HTML reports for Autoscaling Arena Match Configs."""

from __future__ import annotations

import gzip
import html
import json
import math
import shlex
import struct
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

_LOWER_IS_BETTER = {
    "gpu_hours",
    "duration_s",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2e_ms",
    "p99_e2e_ms",
    "p95_e2e_latency_ms",
    "oscillation_count",
    "scale_events",
}
_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#6F4E7C",
)
_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
_SYMBOLS = ("circle", "square", "diamond", "cross", "triangle-up", "x")
_MAX_TIMELINE_POINTS = 1_500
_FRONTEND_TIMELINE_FIELDS = frozenset(
    {
        "time_s",
        "latency_time_s",
        "window_start_s",
        "completed_requests",
        "total_queued_requests",
        "queued_prefill_requests",
        "queued_decode_requests",
        "scheduler_waiting_requests",
        "scheduler_waiting_prefill_requests",
        "scheduler_waiting_decode_requests",
        "router_pending_requests",
        "router_pending_prefill_requests",
        "router_pending_decode_requests",
        "queue_telemetry_semantics",
        "active_kv_cache_utilization",
        "physical_kv_cache_utilization",
        "prefill_active_kv_cache_utilization",
        "decode_active_kv_cache_utilization",
        "prefill_physical_kv_cache_utilization",
        "decode_physical_kv_cache_utilization",
        "scheduler_cache_reuse",
        "router_kv_hit_rate",
        "router_kv_hit_samples",
        "prefill_scheduler_cache_reuse",
        "decode_scheduler_cache_reuse",
        "active_kv_blocks",
        "inactive_kv_blocks",
        "total_kv_blocks",
        "scheduler_cache_hit_tokens",
        "scheduler_cache_total_tokens",
        "preemptions",
        "ttft_samples",
        "tpot_samples",
        "ttft_ms",
        "tpot_ms",
        "active_prefill",
        "active_decode",
        "provisioned_gpus",
        "provisioned_prefill",
        "provisioned_decode",
        "prefill_gpus_per_replica",
        "decode_gpus_per_replica",
        "requested_replicas",
        "requested_prefill",
        "requested_decode",
        "requested_gpus",
        "decision_only",
    }
)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        rendered = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return rendered if math.isfinite(rendered) else None


def _rounded(value: Any, digits: int = 4) -> Optional[float]:
    rendered = _number(value)
    return None if rendered is None else round(rendered, digits)


def _integer(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _optional_integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _first_present(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _configuration_id(result: Mapping[str, Any]) -> str:
    return f"{result.get('sla', 'default')}::r{_integer(result.get('repetition'))}"


def _configuration_label(result: Mapping[str, Any]) -> str:
    repetition = _integer(result.get("repetition")) + 1
    return f"{result.get('sla', 'default')} · repetition {repetition}"


def _sla_target(result: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _mapping(result.get("evaluation"))
    target = dict(_mapping(evaluation.get("sla")))
    return {
        "ttft_ms": _rounded(target.get("ttft_ms")),
        "tpot_ms": _rounded(target.get("itl_ms")),
        "e2e_ms": _rounded(target.get("e2e_ms")),
    }


def _nice_bucket_width(duration_s: float, target_bins: int = 100) -> float:
    raw = max(duration_s / target_bins, 0.1)
    exponent = math.floor(math.log10(raw))
    scale = 10.0**exponent
    fraction = raw / scale
    if fraction <= 1:
        nice = 1.0
    elif fraction <= 2:
        nice = 2.0
    elif fraction <= 5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * scale


def _trace_paths(paths: Path | str | Sequence[Path | str]) -> tuple[Path, ...]:
    if isinstance(paths, (Path, str)):
        return (Path(paths),)
    return tuple(Path(path) for path in paths)


def trace_arrival_series(
    paths: Path | str | Sequence[Path | str],
    *,
    trace_format: str = "mooncake",
    speedup: float = 1.0,
) -> dict[str, Any]:
    """Summarize arriving requests from one trace or native trace shards.

    Mooncake traces retain the original top-level timestamp behavior. Native
    Dynamo traces may be gzip-compressed and use
    ``event.request.request_received_ms`` as their arrival timestamp.
    """

    if not math.isfinite(speedup) or speedup <= 0:
        raise ValueError("speedup must be a finite positive number")
    normalized_format = trace_format.strip().lower()
    if normalized_format not in {"mooncake", "dynamo"}:
        raise ValueError("trace_format must be 'mooncake' or 'dynamo'")
    trace_paths = _trace_paths(paths)
    if not trace_paths:
        return {
            "status": "unavailable",
            "note": "The workload trace has no input files.",
            "bucket_width_s": None,
            "points": [],
        }

    def timestamps() -> Iterable[float]:
        for path in trace_paths:
            opener = gzip.open if path.suffix == ".gz" else Path.open
            with opener(path, "rt") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = _mapping(json.loads(line))
                    if normalized_format == "dynamo":
                        timestamp = _number(
                            _mapping(_mapping(record.get("event")).get("request")).get(
                                "request_received_ms"
                            )
                        )
                    else:
                        timestamp = _number(
                            _first_present(
                                record,
                                "timestamp",
                                "timestamp_ms",
                                "arrival_time_ms",
                            )
                        )
                    if timestamp is not None:
                        yield timestamp

    try:
        # Native collections can be many large gzip shards. Read each source
        # exactly once and spool only its numeric timestamps (8 bytes each) to
        # a private temporary file; the second pass is then over the compact
        # spool rather than decompressing and parsing the source again.
        timestamp_struct = struct.Struct("<d")
        with tempfile.TemporaryFile() as timestamp_spool:
            count = 0
            origin_ms = math.inf
            final_ms = -math.inf
            for timestamp in timestamps():
                count += 1
                origin_ms = min(origin_ms, timestamp)
                final_ms = max(final_ms, timestamp)
                timestamp_spool.write(timestamp_struct.pack(timestamp))

            if count == 0:
                return {
                    "status": "unavailable",
                    "note": "The workload trace contains no arrival timestamps.",
                    "bucket_width_s": None,
                    "points": [],
                }

            duration_s = max((final_ms - origin_ms) / 1000.0 / speedup, 0.1)
            bucket_width_s = _nice_bucket_width(duration_s)
            bucket_count = max(1, int(math.floor(duration_s / bucket_width_s)) + 1)
            counts = [0] * bucket_count
            timestamp_spool.seek(0)
            while chunk := timestamp_spool.read(timestamp_struct.size * 8192):
                for (timestamp,) in struct.iter_unpack("<d", chunk):
                    elapsed_s = max(0.0, (timestamp - origin_ms) / 1000.0 / speedup)
                    index = min(int(elapsed_s / bucket_width_s), bucket_count - 1)
                    counts[index] += 1
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "note": f"Could not read the workload trace ({type(exc).__name__}).",
            "bucket_width_s": None,
            "points": [],
        }

    points = [
        {
            "time_s": round(index * bucket_width_s, 4),
            "count": count,
            "rps": round(count / bucket_width_s, 5),
        }
        for index, count in enumerate(counts)
    ]
    return {
        "status": "ok",
        "note": "Shared arriving traffic; identical for every autoscaler.",
        "bucket_width_s": round(bucket_width_s, 4),
        "points": points,
    }


def _timeline_semantics(result: Mapping[str, Any]) -> str:
    value = result.get("timeline_semantics")
    if not isinstance(value, str):
        value = _mapping(result.get("evaluation")).get("timeline_semantics")
    return str(value or "")


def _coalesce_arrival_points(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(points) <= _MAX_TIMELINE_POINTS:
        return points
    group_size = math.ceil(len(points) / _MAX_TIMELINE_POINTS)
    output: list[dict[str, Any]] = []
    for offset in range(0, len(points), group_size):
        group = points[offset : offset + group_size]
        start_s = group[0]["time_s"]
        end_s = max(point["time_s"] + point["window_width_s"] for point in group)
        width_s = max(end_s - start_s, 0.0001)
        count = sum(point["count"] for point in group)
        output.append(
            {
                "time_s": round(start_s, 4),
                "count": count,
                "rps": round(count / width_s, 5),
                "window_width_s": round(width_s, 4),
            }
        )
    return output


def _saved_arrival_series(
    result: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Return saved arriving requests, never an implicit completion proxy."""

    semantics = _timeline_semantics(result)
    legacy = semantics == "legacy_num_req_as_completed"
    points: list[dict[str, Any]] = []
    saw_arriving_field = False
    saw_offered_field = False
    for raw in _sequence(result.get("timeline")):
        sample = _mapping(raw)
        arriving = _optional_integer(sample.get("arriving_requests"))
        if arriving is not None:
            saw_arriving_field = True
        else:
            arriving = _optional_integer(sample.get("offered_requests"))
            if arriving is not None:
                saw_offered_field = True
        if arriving is None and legacy:
            arriving = _optional_integer(sample.get("completed_requests"))
        if arriving is None:
            continue
        end_s = _number(_first_present(sample, "timestamp_s", "time_s", "at_s"))
        start_s = _number(sample.get("window_start_s"))
        if end_s is None or start_s is None or end_s <= start_s:
            continue
        width_s = end_s - start_s
        points.append(
            {
                "time_s": round(max(0.0, start_s), 4),
                "count": max(0, arriving),
                "rps": round(max(0, arriving) / width_s, 5),
                "window_width_s": round(width_s, 4),
            }
        )
    if not points:
        return None

    points.sort(key=lambda point: point["time_s"])
    points = _coalesce_arrival_points(points)
    widths = {point["window_width_s"] for point in points}
    saved_total = sum(point["count"] for point in points)
    evaluation = _mapping(result.get("evaluation"))
    expected_total = _optional_integer(
        _mapping(evaluation.get("trace")).get("request_rows")
    )
    if legacy and not saw_arriving_field and not saw_offered_field:
        note = (
            "Shared arriving traffic from saved replay windows. This legacy result "
            "explicitly marks its mislabeled completed_requests field as native "
            "arriving requests."
        )
        source = "legacy_marked_offered_requests"
    elif saw_arriving_field:
        note = "Shared arriving traffic from saved replay windows."
        source = "saved_arriving_requests"
    else:
        note = "Shared arriving traffic from saved replay windows."
        source = "saved_offered_requests"
    if expected_total is not None and saved_total < expected_total:
        missing = expected_total - saved_total
        note += (
            f" The saved windows cover {saved_total:,} of "
            f"{expected_total:,} arriving requests; {missing:,} arrived after "
            "the final saved window and have no time bucket."
        )
    return {
        "status": "ok",
        "note": note,
        "source": source,
        "bucket_width_s": next(iter(widths)) if len(widths) == 1 else None,
        "points": points,
    }


def _arrival_series(
    result: Mapping[str, Any],
    *,
    closed_loop: bool,
    cache: dict[tuple[tuple[str, ...], str, float, bool], dict[str, Any]],
) -> dict[str, Any]:
    if closed_loop:
        return {
            "status": "unavailable",
            "note": "Closed-loop replay ignores trace timestamps.",
            "bucket_width_s": None,
            "points": [],
        }

    evaluation = _mapping(result.get("evaluation"))
    embedded = evaluation.get("arrival_series")
    if isinstance(embedded, Mapping):
        return dict(embedded)
    saved_arrivals = _saved_arrival_series(result)
    if saved_arrivals is not None:
        return saved_arrivals
    trace = _mapping(evaluation.get("trace"))
    raw_paths: list[str] = []
    for raw_path in _sequence(trace.get("paths")):
        if isinstance(raw_path, str) and raw_path:
            raw_paths.append(raw_path)
        elif isinstance(raw_path, Mapping):
            nested_path = raw_path.get("path")
            if isinstance(nested_path, str) and nested_path:
                raw_paths.append(nested_path)
    if not raw_paths:
        raw_path = trace.get("path")
        if isinstance(raw_path, str) and raw_path:
            raw_paths.append(raw_path)
    if not raw_paths:
        return {
            "status": "unavailable",
            "note": "The materialized workload trace is unavailable.",
            "bucket_width_s": None,
            "points": [],
        }

    speedup = 1.0
    if result.get("backend") == "sim":
        configured_speedup = _number(evaluation.get("arrival_speedup"))
        if configured_speedup is not None and configured_speedup > 0:
            speedup = configured_speedup
    trace_format = str(
        _first_present(trace, "format", "trace_format") or "mooncake"
    ).lower()
    key = (tuple(raw_paths), trace_format, speedup, closed_loop)
    if key in cache:
        return cache[key]

    output = trace_arrival_series(
        [Path(raw_path) for raw_path in raw_paths],
        trace_format=trace_format,
        speedup=speedup,
    )
    cache[key] = output
    return output


def _fallback_timeline(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifact_dir = _mapping(result.get("artifacts")).get("directory")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return []
    path = Path(artifact_dir) / "planner.log.jsonl.gz"
    if not path.is_file():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("kind") != "snapshot":
                    continue
                rows.append(
                    {
                        "timestamp_s": record.get("timestamp_s"),
                        "window_start_s": record.get("timestamp_s"),
                        "mean_ttft_ms": record.get("observed_ttft_ms"),
                        "mean_tpot_ms": record.get("observed_itl_ms"),
                        "active_prefill_replicas": record.get("num_prefill_replicas"),
                        "active_decode_replicas": record.get("num_decode_replicas"),
                        "provisioned_prefill_replicas": record.get(
                            "num_prefill_replicas"
                        ),
                        "provisioned_decode_replicas": record.get(
                            "num_decode_replicas"
                        ),
                    }
                )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []
    return rows


def _engine_gpu_widths(result: Mapping[str, Any]) -> tuple[int, int]:
    runtime = _mapping(result.get("runtime"))
    engines = _mapping(runtime.get("engines"))

    def role_width(role: str) -> int:
        config = _mapping(_mapping(engines.get(role)).get("config"))
        return max(0, _integer(config.get("num_gpus"), 0))

    prefill = role_width("prefill")
    decode = role_width("decode")
    aggregate = role_width("aggregate")
    if aggregate:
        return (0, aggregate)
    return (prefill, decode)


def _downsample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= _MAX_TIMELINE_POINTS:
        return rows

    # Preserve each chart lane's global peak: otherwise a short KV, latency, or
    # cache-reuse spike can disappear merely because queue depth selected a
    # neighboring point from the same display bucket.
    indexes = {0, len(rows) - 1}
    for field in (
        "total_queued_requests",
        "ttft_ms",
        "tpot_ms",
        "active_kv_cache_utilization",
        "physical_kv_cache_utilization",
        "router_kv_hit_rate",
        "scheduler_cache_reuse",
    ):
        measured = [
            (value, index)
            for index, row in enumerate(rows)
            if (value := _number(row.get(field))) is not None
        ]
        if measured:
            indexes.add(max(measured)[1])

    telemetry_indexes = {
        index for index, row in enumerate(rows) if not bool(row.get("decision_only"))
    }

    # Reserve two thirds of the display budget for measured samples before
    # adding discrete events. Otherwise a controller that emits decisions more
    # frequently than telemetry is sampled can crowd queue, latency, and KV
    # history down to only global peaks.
    telemetry_target = min(
        len(telemetry_indexes),
        math.ceil(_MAX_TIMELINE_POINTS * 2 / 3),
    )
    telemetry_needed = max(0, telemetry_target - len(indexes & telemetry_indexes))
    indexes.update(
        _bucketed_timeline_indexes(
            sorted(telemetry_indexes - indexes), rows, telemetry_needed
        )
    )

    # Scaling decisions and observed capacity transitions share the remaining
    # event budget. All are retained for normal runs; pathological streams are
    # sampled evenly while measured telemetry remains representative.
    decisions = [
        index
        for index, row in enumerate(rows)
        if row["requested_replicas"] is not None and index not in indexes
    ]

    # Capacity is rendered as a step function. Preserve both sides of P/D and
    # provisioned-capacity transitions whenever the display budget allows,
    # including role rebalances whose total replica count does not change. A
    # decision-only row has no observed capacity and is intentionally skipped.
    capacity_fields = (
        "active_replicas",
        "active_prefill",
        "active_decode",
        "provisioned_replicas",
        "provisioned_prefill",
        "provisioned_decode",
        "provisioned_gpus",
    )
    capacity_boundaries: set[int] = set()
    capacity_samples = sorted(telemetry_indexes)
    for previous, index in zip(capacity_samples, capacity_samples[1:]):
        if any(
            rows[index].get(field) != rows[previous].get(field)
            for field in capacity_fields
        ):
            capacity_boundaries.update((previous, index))
    capacity_candidates = sorted(capacity_boundaries - indexes)
    event_budget = max(0, _MAX_TIMELINE_POINTS - len(indexes))
    decision_budget, capacity_budget = _shared_event_budgets(
        len(decisions), len(capacity_candidates), event_budget
    )
    indexes.update(_evenly_sampled_indexes(decisions, decision_budget))
    indexes.update(_evenly_sampled_indexes(capacity_candidates, capacity_budget))

    # Give any event budget left unused by normal runs back to measured time
    # buckets. Retain the largest queue depth in each bucket (or its center when
    # queue telemetry is unavailable) so local backlog spikes remain visible.
    remaining = max(0, _MAX_TIMELINE_POINTS - len(indexes))
    indexes.update(
        _bucketed_timeline_indexes(sorted(telemetry_indexes - indexes), rows, remaining)
    )
    return [row for index, row in enumerate(rows) if index in indexes]


def _bucketed_timeline_indexes(
    indexes: list[int], rows: list[dict[str, Any]], limit: int
) -> list[int]:
    if limit <= 0 or not indexes:
        return []
    if len(indexes) <= limit:
        return indexes
    selected: list[int] = []
    for bucket in range(limit):
        start = math.floor(bucket * len(indexes) / limit)
        end = math.floor((bucket + 1) * len(indexes) / limit)
        candidates = indexes[start : max(start + 1, end)]
        center = (indexes[start] + indexes[max(start, end - 1)]) / 2.0

        def priority(index: int) -> tuple[float, float]:
            queue_depth = _number(rows[index].get("total_queued_requests"))
            return (
                queue_depth if queue_depth is not None else -math.inf,
                -abs(index - center),
            )

        selected.append(max(candidates, key=priority))
    return selected


def _shared_event_budgets(decisions: int, capacity: int, total: int) -> tuple[int, int]:
    decision_budget = min(decisions, math.ceil(total / 2))
    capacity_budget = min(capacity, total - decision_budget)
    remaining = total - decision_budget - capacity_budget
    if remaining:
        extra_decisions = min(decisions - decision_budget, remaining)
        decision_budget += extra_decisions
        remaining -= extra_decisions
    if remaining:
        capacity_budget += min(capacity - capacity_budget, remaining)
    return decision_budget, capacity_budget


def _evenly_sampled_indexes(indexes: list[int], limit: int) -> list[int]:
    if limit <= 0 or not indexes:
        return []
    if len(indexes) <= limit:
        return indexes
    if limit == 1:
        return [indexes[-1]]
    return [
        indexes[round(position * (len(indexes) - 1) / (limit - 1))]
        for position in range(limit)
    ]


def _normalized_timeline(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _sequence(result.get("timeline")) or _fallback_timeline(result)
    semantics = _timeline_semantics(result)
    legacy = semantics == "legacy_num_req_as_completed"
    prefill_gpu_width, decode_gpu_width = _engine_gpu_widths(result)
    rows: list[dict[str, Any]] = []
    for raw in source:
        sample = _mapping(raw)
        timestamp_s = _number(_first_present(sample, "timestamp_s", "time_s", "at_s"))
        if timestamp_s is None:
            continue
        window_start_s = _number(sample.get("window_start_s"))
        if window_start_s is None:
            window_start_s = timestamp_s
        active_prefill = max(0, _integer(sample.get("active_prefill_replicas"), 0))
        active_decode = max(0, _integer(sample.get("active_decode_replicas"), 0))
        provisioned_prefill = max(
            0,
            _integer(
                sample.get("provisioned_prefill_replicas"),
                active_prefill,
            ),
        )
        provisioned_decode = max(
            0,
            _integer(
                sample.get("provisioned_decode_replicas"),
                active_decode,
            ),
        )
        has_scaling_decision = bool(sample.get("scaling_decision")) or (
            sample.get("requested_prefill_replicas") is not None
            or sample.get("requested_decode_replicas") is not None
        )
        requested_prefill = (
            max(
                0,
                _integer(
                    sample.get("requested_prefill_replicas"),
                    provisioned_prefill,
                ),
            )
            if has_scaling_decision
            else None
        )
        requested_decode = (
            max(
                0,
                _integer(
                    sample.get("requested_decode_replicas"),
                    provisioned_decode,
                ),
            )
            if has_scaling_decision
            else None
        )
        arriving_requests = _optional_integer(sample.get("arriving_requests"))
        if arriving_requests is None:
            arriving_requests = _optional_integer(sample.get("offered_requests"))
        completed_requests = _optional_integer(sample.get("completed_requests"))
        if arriving_requests is None and legacy:
            arriving_requests = completed_requests
            completed_requests = None
        total_queued_requests = _optional_integer(sample.get("total_queued_requests"))
        queued_prefill_requests = _optional_integer(
            sample.get("queued_prefill_requests")
        )
        queued_decode_requests = _optional_integer(sample.get("queued_decode_requests"))
        scheduler_waiting_requests = _optional_integer(
            sample.get("scheduler_waiting_requests")
        )
        scheduler_waiting_prefill_requests = _optional_integer(
            sample.get("scheduler_waiting_prefill_requests")
        )
        scheduler_waiting_decode_requests = _optional_integer(
            sample.get("scheduler_waiting_decode_requests")
        )
        router_pending_requests = _optional_integer(
            sample.get("router_pending_requests")
        )
        router_pending_prefill_requests = _optional_integer(
            sample.get("router_pending_prefill_requests")
        )
        router_pending_decode_requests = _optional_integer(
            sample.get("router_pending_decode_requests")
        )
        fpm_queued_requests = _optional_integer(sample.get("fpm_queued_requests"))
        fpm_queued_prefill_requests = _optional_integer(
            sample.get("fpm_queued_prefill_requests")
        )
        fpm_queued_decode_requests = _optional_integer(
            sample.get("fpm_queued_decode_requests")
        )
        raw_prefill_scheduler_metrics = [
            dict(row)
            for row in _sequence(sample.get("prefill_scheduler_metrics"))
            if isinstance(row, Mapping)
        ]
        raw_decode_scheduler_metrics = [
            dict(row)
            for row in _sequence(sample.get("decode_scheduler_metrics"))
            if isinstance(row, Mapping)
        ]
        rows.append(
            {
                "decision_only": bool(sample.get("decision_only")),
                "time_s": round(max(0.0, timestamp_s), 4),
                "latency_time_s": round(
                    max(0.0, (timestamp_s + window_start_s) / 2.0), 4
                ),
                "window_start_s": round(max(0.0, window_start_s), 4),
                "arriving_requests": (
                    max(0, arriving_requests) if arriving_requests is not None else None
                ),
                # Keep the old normalized alias for report-data consumers
                # while new producers and UI language use arriving_requests.
                "offered_requests": (
                    max(0, arriving_requests) if arriving_requests is not None else None
                ),
                "completed_requests": (
                    max(0, completed_requests)
                    if completed_requests is not None
                    else None
                ),
                # Keep queue availability distinct from a measured zero. In
                # particular, do not manufacture a total from role details:
                # older saved runs do not contain trustworthy queue telemetry.
                "total_queued_requests": (
                    max(0, total_queued_requests)
                    if total_queued_requests is not None
                    else None
                ),
                "queued_prefill_requests": (
                    max(0, queued_prefill_requests)
                    if queued_prefill_requests is not None
                    else None
                ),
                "queued_decode_requests": (
                    max(0, queued_decode_requests)
                    if queued_decode_requests is not None
                    else None
                ),
                "scheduler_waiting_requests": (
                    max(0, scheduler_waiting_requests)
                    if scheduler_waiting_requests is not None
                    else None
                ),
                "scheduler_waiting_prefill_requests": (
                    max(0, scheduler_waiting_prefill_requests)
                    if scheduler_waiting_prefill_requests is not None
                    else None
                ),
                "scheduler_waiting_decode_requests": (
                    max(0, scheduler_waiting_decode_requests)
                    if scheduler_waiting_decode_requests is not None
                    else None
                ),
                "router_pending_requests": (
                    max(0, router_pending_requests)
                    if router_pending_requests is not None
                    else None
                ),
                "router_pending_prefill_requests": (
                    max(0, router_pending_prefill_requests)
                    if router_pending_prefill_requests is not None
                    else None
                ),
                "router_pending_decode_requests": (
                    max(0, router_pending_decode_requests)
                    if router_pending_decode_requests is not None
                    else None
                ),
                "fpm_queued_requests": (
                    max(0, fpm_queued_requests)
                    if fpm_queued_requests is not None
                    else None
                ),
                "fpm_queued_prefill_requests": (
                    max(0, fpm_queued_prefill_requests)
                    if fpm_queued_prefill_requests is not None
                    else None
                ),
                "fpm_queued_decode_requests": (
                    max(0, fpm_queued_decode_requests)
                    if fpm_queued_decode_requests is not None
                    else None
                ),
                "queue_telemetry_semantics": (
                    str(sample.get("queue_telemetry_semantics"))
                    if sample.get("queue_telemetry_semantics") is not None
                    else None
                ),
                "active_kv_cache_utilization": _valid_ratio(
                    sample.get("active_kv_cache_utilization")
                ),
                "physical_kv_cache_utilization": _valid_ratio(
                    sample.get("physical_kv_cache_utilization")
                ),
                "prefill_active_kv_cache_utilization": _valid_ratio(
                    sample.get("prefill_active_kv_cache_utilization")
                ),
                "decode_active_kv_cache_utilization": _valid_ratio(
                    sample.get("decode_active_kv_cache_utilization")
                ),
                "prefill_physical_kv_cache_utilization": _valid_ratio(
                    sample.get("prefill_physical_kv_cache_utilization")
                ),
                "decode_physical_kv_cache_utilization": _valid_ratio(
                    sample.get("decode_physical_kv_cache_utilization")
                ),
                "scheduler_cache_reuse": _valid_ratio(
                    sample.get("scheduler_cache_reuse")
                ),
                "router_kv_hit_rate": _valid_ratio(
                    _first_present(
                        sample,
                        "mean_router_kv_hit_rate",
                        "router_kv_hit_rate",
                    )
                ),
                "router_kv_hit_samples": _optional_integer(
                    _first_present(
                        sample,
                        "router_kv_hit_sample_count",
                        "router_kv_hit_samples",
                    )
                ),
                "prefill_scheduler_cache_reuse": _valid_ratio(
                    sample.get("prefill_scheduler_cache_reuse")
                ),
                "decode_scheduler_cache_reuse": _valid_ratio(
                    sample.get("decode_scheduler_cache_reuse")
                ),
                "active_kv_blocks": _optional_integer(sample.get("active_kv_blocks")),
                "inactive_kv_blocks": _optional_integer(
                    sample.get("inactive_kv_blocks")
                ),
                "total_kv_blocks": _optional_integer(sample.get("total_kv_blocks")),
                "scheduler_cache_hit_tokens": _optional_integer(
                    sample.get("scheduler_cache_hit_tokens")
                ),
                "scheduler_cache_total_tokens": _optional_integer(
                    sample.get("scheduler_cache_total_tokens")
                ),
                "running_prefill_requests": _optional_integer(
                    sample.get("running_prefill_requests")
                ),
                "running_decode_requests": _optional_integer(
                    sample.get("running_decode_requests")
                ),
                "total_running_requests": _optional_integer(
                    sample.get("total_running_requests")
                ),
                "preemptions": _optional_integer(
                    _first_present(sample, "preemptions", "preemptions_total")
                ),
                "preemptions_total": _optional_integer(
                    _first_present(sample, "preemptions", "preemptions_total")
                ),
                "prefill_scheduler_metrics": (
                    raw_prefill_scheduler_metrics
                    if isinstance(sample.get("prefill_scheduler_metrics"), list)
                    else None
                ),
                "decode_scheduler_metrics": (
                    raw_decode_scheduler_metrics
                    if isinstance(sample.get("decode_scheduler_metrics"), list)
                    else None
                ),
                "scheduler_metrics_available": bool(
                    sample.get("scheduler_metrics_available")
                ),
                "scheduler_metrics_payload_available": bool(
                    sample.get("scheduler_metrics_payload_available")
                ),
                "active_scheduler_metrics_available": bool(
                    sample.get("active_scheduler_metrics_available")
                ),
                "ttft_samples": _optional_integer(
                    _first_present(
                        sample,
                        "ttft_sample_count",
                        "ttft_samples",
                        "ttft_count",
                    )
                ),
                "tpot_samples": _optional_integer(
                    _first_present(
                        sample,
                        "itl_sample_count",
                        "tpot_sample_count",
                        "tpot_samples",
                        "itl_samples",
                        "itl_count",
                    )
                ),
                "ttft_ms": _rounded(_first_present(sample, "mean_ttft_ms", "ttft_ms")),
                "tpot_ms": _rounded(
                    _first_present(
                        sample,
                        "mean_tpot_ms",
                        "mean_itl_ms",
                        "tpot_ms",
                        "itl_ms",
                    )
                ),
                "active_replicas": active_prefill + active_decode,
                "active_prefill": active_prefill,
                "active_decode": active_decode,
                "provisioned_replicas": (provisioned_prefill + provisioned_decode),
                "provisioned_prefill": provisioned_prefill,
                "provisioned_decode": provisioned_decode,
                "provisioned_gpus": (
                    provisioned_prefill * prefill_gpu_width
                    + provisioned_decode * decode_gpu_width
                ),
                "prefill_gpus_per_replica": prefill_gpu_width,
                "decode_gpus_per_replica": decode_gpu_width,
                "requested_replicas": (
                    requested_prefill + requested_decode
                    if has_scaling_decision
                    else None
                ),
                "requested_prefill": requested_prefill,
                "requested_decode": requested_decode,
                "requested_gpus": (
                    requested_prefill * prefill_gpu_width
                    + requested_decode * decode_gpu_width
                    if has_scaling_decision
                    else None
                ),
            }
        )
    rows.sort(key=lambda row: row["time_s"])
    return _downsample(rows)


def _valid_ratio(value: Any) -> Optional[float]:
    ratio = _number(value)
    if ratio is None or ratio < 0.0 or ratio > 1.0:
        return None
    return ratio


def _coalesce_cache_points(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(points) <= _MAX_TIMELINE_POINTS:
        return points
    group_size = math.ceil(len(points) / _MAX_TIMELINE_POINTS)
    output: list[dict[str, Any]] = []
    for offset in range(0, len(points), group_size):
        group = points[offset : offset + group_size]
        input_tokens = sum(point["input_tokens"] for point in group)
        reused_tokens = sum(point["reused_input_tokens"] for point in group)
        completed = [
            point["completed_requests"]
            for point in group
            if point["completed_requests"] is not None
        ]
        output.append(
            {
                "time_s": group[-1]["time_s"],
                "window_start_s": group[0]["window_start_s"],
                "input_tokens": input_tokens,
                "reused_input_tokens": reused_tokens,
                "completed_requests": sum(completed) if completed else None,
                "prefix_cache_reused_ratio": round(reused_tokens / input_tokens, 6),
            }
        )
    return output


def _normalized_cache(result: Mapping[str, Any]) -> dict[str, Any]:
    cache = _mapping(result.get("cache"))
    aggregate = _valid_ratio(cache.get("prefix_cache_reused_ratio"))
    first_admission = _valid_ratio(
        cache.get("first_admission_prefix_cache_reused_ratio")
    )
    source_points = _sequence(cache.get("timeline"))
    points: list[dict[str, Any]] = []
    for raw in source_points:
        point = _mapping(raw)
        time_s = _number(_first_present(point, "time_s", "timestamp_s", "window_end_s"))
        window_start_s = _number(point.get("window_start_s"))
        input_tokens = _optional_integer(point.get("input_tokens"))
        reused_tokens = _optional_integer(point.get("reused_input_tokens"))
        if (
            time_s is None
            or input_tokens is None
            or input_tokens <= 0
            or reused_tokens is None
            or reused_tokens < 0
            or reused_tokens > input_tokens
        ):
            continue
        if window_start_s is None:
            window_start_s = time_s
        completed = _optional_integer(point.get("completed_requests"))
        points.append(
            {
                "time_s": round(max(0.0, time_s), 4),
                "window_start_s": round(max(0.0, window_start_s), 4),
                "input_tokens": input_tokens,
                "reused_input_tokens": reused_tokens,
                "completed_requests": (
                    max(0, completed) if completed is not None else None
                ),
                # Recompute from the saved numerator and denominator so every
                # displayed interval remains token weighted.
                "prefix_cache_reused_ratio": round(reused_tokens / input_tokens, 6),
            }
        )
    points.sort(key=lambda point: point["time_s"])
    points = _coalesce_cache_points(points)

    timeline_claimed = bool(cache.get("timeline_available"))
    if points:
        status = "ok"
        note = (
            "Realized prefix-cache reuse over time; each point is weighted by "
            "input tokens."
        )
    elif aggregate is not None:
        status = "aggregate_only"
        note = (
            "Only the whole-run realized prefix-cache reuse aggregate was "
            "saved; an exact time series is unavailable."
        )
    else:
        status = "unavailable"
        note = (
            "Realized prefix-cache reuse was not saved for this run."
            if not timeline_claimed
            else "The saved prefix-cache timeline contains no usable token totals."
        )
    return {
        "status": status,
        "note": str(cache.get("note") or note),
        "prefix_cache_reused_ratio": (
            round(aggregate, 6) if aggregate is not None else None
        ),
        "first_admission_prefix_cache_reused_ratio": (
            round(first_admission, 6) if first_admission is not None else None
        ),
        "timeline_available": bool(points),
        "timeline_source": str(cache.get("timeline_source") or ""),
        "timeline": points,
    }


def _ranked_results(
    results: Iterable[dict[str, Any]], rank_by: str
) -> list[dict[str, Any]]:
    lower_is_better = rank_by in _LOWER_IS_BETTER

    def sort_key(result: Mapping[str, Any]) -> tuple[int, float, str]:
        value = _number(_mapping(result.get("metrics")).get(rank_by))
        if result.get("status") != "ok":
            return (2, 0.0, str(result.get("autoscaler", "")))
        if value is None:
            return (1, 0.0, str(result.get("autoscaler", "")))
        return (
            0,
            value if lower_is_better else -value,
            str(result.get("autoscaler", "")),
        )

    ordered = sorted(results, key=sort_key)
    rank = 0
    for result in ordered:
        value = _number(_mapping(result.get("metrics")).get(rank_by))
        if result.get("status") == "ok" and value is not None:
            rank += 1
            result["rank"] = rank
        else:
            result["rank"] = None
    return ordered


def _deployment_label(report: Mapping[str, Any]) -> str:
    backend = _mapping(_mapping(report.get("resolved_config")).get("backend"))
    backend_type = str(
        _mapping(report.get("match")).get("backend", backend.get("type", ""))
    )
    if backend_type != "sim":
        return "Online endpoints"
    model = str(_mapping(backend.get("model")).get("name", "model"))
    topology = str(backend.get("topology", "topology"))
    engines = _mapping(backend.get("engines"))
    roles: list[str] = []
    for role in ("prefill", "decode", "aggregate"):
        engine = _mapping(engines.get(role))
        if not engine:
            continue
        roles.append(
            f"{role}: {engine.get('backend', '?')} / "
            f"{engine.get('system', '?')} / {engine.get('num_gpus', '?')} GPU"
        )
    suffix = " · ".join(roles)
    return f"{model} · {topology}" + (f" · {suffix}" if suffix else "")


def _deployment_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    backend = _mapping(_mapping(report.get("resolved_config")).get("backend"))
    topology = str(backend.get("topology") or "")
    replay = _mapping(backend.get("replay"))
    engines = _mapping(backend.get("engines"))
    cold_start_s: dict[str, float | None] = {}
    for role in ("prefill", "decode", "aggregate"):
        runtime = _mapping(_mapping(engines.get(role)).get("runtime"))
        cold_start_s[role] = _rounded(runtime.get("cold_start_delay_s"))

    if topology == "disagg":
        cold_start_parts = [
            ("P", cold_start_s["prefill"]),
            ("D", cold_start_s["decode"]),
        ]
    else:
        cold_start_parts = [("Aggregate", cold_start_s["aggregate"])]
    configured = [
        f"{label} {value:g} s" for label, value in cold_start_parts if value is not None
    ]
    return {
        "topology": topology,
        "telemetry_sample_interval_s": _rounded(
            replay.get("telemetry_sample_interval_s")
        ),
        "cold_start_s": cold_start_s,
        "cold_start_label": " · ".join(configured) or "Not configured",
    }


def _replay_command(
    report: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    replay_config_path: Path | str | None,
) -> Optional[str]:
    provenance = _mapping(report.get("provenance"))
    configured_path = _replay_config_reference(
        report, replay_config_path=replay_config_path
    )
    # Replay commands are guarded by the portable digest produced by
    # ``match_replay_sha256``.  The legacy config digest includes local paths
    # and is intentionally not interchangeable with that contract.
    config_sha256 = _valid_sha256(provenance.get("replay_config_sha256"))
    run_id = str(result.get("run_id") or "")
    if not configured_path or config_sha256 is None or not run_id:
        return None
    config_argument = (
        '"$MATCH_CONFIG"'
        if str(configured_path) == "$MATCH_CONFIG"
        else shlex.quote(str(configured_path))
    )
    suffix = [
        "--expect-config-sha256",
        config_sha256,
        "--run-id",
        run_id,
    ]
    for digest in _replay_trace_sha256s(result):
        suffix.extend(("--expect-trace-sha256", digest))
    suffix.append("--no-publish")
    return " ".join(
        (
            "python",
            "scripts/run_match_config.py",
            config_argument,
            shlex.join(suffix),
        )
    )


def _replay_config_reference(
    report: Mapping[str, Any],
    *,
    replay_config_path: Path | str | None,
) -> Optional[str]:
    if replay_config_path is not None:
        return str(replay_config_path)
    replay = _mapping(_mapping(report.get("provenance")).get("replay"))
    if replay.get("kind") != "match_config" or not replay.get("config_path"):
        return None
    return str(replay["config_path"])


def _replay_trace_sha256s(result: Mapping[str, Any]) -> tuple[str, ...]:
    """Return saved source-trace digests that the replay CLI can verify."""

    trace = _mapping(_mapping(result.get("evaluation")).get("trace"))
    candidates: list[Any]
    if trace.get("format") == "dynamo":
        candidates = [
            _mapping(shard).get("sha256") for shard in _sequence(trace.get("shards"))
        ]
    elif trace.get("source_sha256") is not None:
        candidates = [trace.get("source_sha256")]
    elif trace.get("staged") is False:
        candidates = [trace.get("sha256")]
    else:
        candidates = []
    return tuple(
        digest for value in candidates if (digest := _valid_sha256(value)) is not None
    )


def _valid_sha256(value: Any) -> Optional[str]:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        return None
    return digest


def build_match_report_data(
    report: Mapping[str, Any],
    *,
    replay_config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Normalize one Match Config result for the standalone report UI."""

    summary = _mapping(report.get("summary"))
    rank_by = str(summary.get("rank_by", "goodput_per_gpu"))
    metric_columns = [str(value) for value in _sequence(summary.get("metrics"))]
    if rank_by not in metric_columns:
        metric_columns.insert(0, rank_by)

    configurations: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    workloads: "OrderedDict[str, None]" = OrderedDict()
    autoscalers: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    grouped: "OrderedDict[tuple[str, str], list[dict[str, Any]]]" = OrderedDict()
    replay_config_reference = _replay_config_reference(
        report, replay_config_path=replay_config_path
    )
    results = [_mapping(value) for value in _sequence(report.get("results"))]
    for raw_result in results:
        config_id = _configuration_id(raw_result)
        workload = str(raw_result.get("workload", "unknown"))
        autoscaler = str(raw_result.get("autoscaler", "unknown"))
        configurations.setdefault(
            config_id,
            {
                "id": config_id,
                "label": _configuration_label(raw_result),
                "sla": _sla_target(raw_result),
                "repetition": _integer(raw_result.get("repetition")),
            },
        )
        workloads.setdefault(workload, None)
        if autoscaler not in autoscalers:
            index = len(autoscalers)
            autoscalers[autoscaler] = {
                "name": autoscaler,
                "color": _COLORS[index % len(_COLORS)],
                "dash": _DASHES[index % len(_DASHES)],
                "symbol": _SYMBOLS[index % len(_SYMBOLS)],
            }
        grouped.setdefault((config_id, workload), []).append(dict(raw_result))

    resolved_backend = _mapping(_mapping(report.get("resolved_config")).get("backend"))
    closed_loop = (
        resolved_backend.get("type") == "sim"
        and _mapping(resolved_backend.get("replay")).get("concurrency") is not None
    )
    arrival_cache: dict[tuple[tuple[str, ...], str, float, bool], dict[str, Any]] = {}
    scopes: list[dict[str, Any]] = []
    for (config_id, workload), scope_results in grouped.items():
        normalized_results: list[dict[str, Any]] = []
        duration_s = 0.0
        for result in scope_results:
            metrics = {
                key: _rounded(value)
                for key, value in _mapping(result.get("metrics")).items()
            }
            timeline = _normalized_timeline(result)
            cache_data = _normalized_cache(result)
            if timeline:
                duration_s = max(duration_s, timeline[-1]["time_s"])
            cache_timeline = _sequence(cache_data.get("timeline"))
            if cache_timeline:
                duration_s = max(
                    duration_s,
                    _number(_mapping(cache_timeline[-1]).get("time_s")) or 0.0,
                )
            metric_duration = _number(metrics.get("duration_s"))
            if metric_duration is not None:
                duration_s = max(duration_s, metric_duration)
            autoscaler = str(result.get("autoscaler", "unknown"))
            style = autoscalers[autoscaler]
            error = _mapping(result.get("error"))
            normalized_results.append(
                {
                    "run_id": str(result.get("run_id", "")),
                    "autoscaler": autoscaler,
                    "status": str(result.get("status", "unknown")),
                    "metrics": metrics,
                    "error": {
                        "type": str(error.get("type", "")),
                        "message": str(error.get("message", "")),
                    },
                    "timeline": timeline,
                    "cache": cache_data,
                    "replay_command": _replay_command(
                        report,
                        result,
                        replay_config_path=replay_config_path,
                    ),
                    "replay_trace_guarded": bool(_replay_trace_sha256s(result)),
                    "replay_uses_config_env": (
                        replay_config_reference == "$MATCH_CONFIG"
                    ),
                    **style,
                }
            )
        representative = scope_results[0]
        arrivals = _arrival_series(
            representative, closed_loop=closed_loop, cache=arrival_cache
        )
        arrival_points = _sequence(arrivals.get("points"))
        if arrival_points:
            last_arrival = _number(_mapping(arrival_points[-1]).get("time_s")) or 0.0
            duration_s = max(
                duration_s,
                last_arrival
                + (
                    _number(_mapping(arrival_points[-1]).get("window_width_s"))
                    or _number(arrivals.get("bucket_width_s"))
                    or 0.0
                ),
            )
        ranked = _ranked_results(normalized_results, rank_by)
        scopes.append(
            {
                "id": f"{config_id}::{workload}",
                "configuration_id": config_id,
                "workload": workload,
                "duration_s": round(duration_s, 4),
                "arrivals": arrivals,
                "results": ranked,
            }
        )

    match = _mapping(report.get("match"))
    provenance = _mapping(report.get("provenance"))
    deployment_metadata = _deployment_metadata(report)
    return {
        "title": str(match.get("name", "Autoscaling Arena")),
        "description": str(match.get("description", "")),
        "backend": str(match.get("backend", "")),
        "deployment": _deployment_label(report),
        **deployment_metadata,
        "status": str(summary.get("status", "unknown")),
        "summary": {
            "planned_runs": _integer(summary.get("planned_runs")),
            "succeeded_runs": _integer(summary.get("succeeded_runs")),
            "failed_runs": _integer(summary.get("failed_runs")),
            "skipped_runs": _integer(summary.get("skipped_runs")),
        },
        "rank_by": rank_by,
        "rank_direction": ("lower" if rank_by in _LOWER_IS_BETTER else "higher"),
        "metric_columns": metric_columns,
        "configurations": list(configurations.values()),
        "workloads": list(workloads),
        "autoscalers": list(autoscalers.values()),
        "scopes": scopes,
        "provenance": {
            "finished_at": str(provenance.get("finished_at", "")),
            "git_commit": str(provenance.get("git_commit") or ""),
            "config_sha256": str(provenance.get("config_sha256") or ""),
            "replay_config_sha256": str(provenance.get("replay_config_sha256") or ""),
            "session_id": str(provenance.get("session_id") or ""),
            "config_file": str(provenance.get("config_file") or ""),
        },
    }


_REPORT_CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f5f7fb;
  --surface: #ffffff;
  --surface-2: #eef2f7;
  --text: #152033;
  --muted: #5d697a;
  --border: #d8dee8;
  --accent: #1769aa;
  --accent-soft: #e5f1fa;
  --ok: #007f5f;
  --bad: #b42318;
  --shadow: 0 16px 42px rgba(23, 36, 61, .08);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1118;
    --surface: #151b24;
    --surface-2: #1d2632;
    --text: #edf2f8;
    --muted: #aeb8c7;
    --border: #303b49;
    --accent: #68b5e8;
    --accent-soft: #173348;
    --ok: #60d4ae;
    --bad: #ff8b82;
    --shadow: 0 18px 45px rgba(0, 0, 0, .24);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}
main { width: min(1500px, 100%); margin: 0 auto; padding: 28px; }
.hero { display: grid; gap: 10px; margin: 4px 0 22px; }
.eyebrow {
  color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase;
}
h1, h2 { margin: 0; letter-spacing: -.025em; }
h1 { font-size: clamp(28px, 4vw, 44px); line-height: 1.05; }
h2 { font-size: 20px; }
.lede { max-width: 900px; margin: 0; color: var(--muted); }
.meta { display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
  display: inline-flex; align-items: center; min-height: 28px; padding: 4px 10px;
  border: 1px solid var(--border); border-radius: 999px; background: var(--surface);
  color: var(--muted); font-size: 12px;
}
.pill strong { color: var(--text); margin-right: 5px; }
.panel {
  margin-top: 18px; padding: 20px; border: 1px solid var(--border);
  border-radius: 16px; background: var(--surface); box-shadow: var(--shadow);
}
.controls {
  display: grid; grid-template-columns: minmax(260px, 400px) minmax(200px, 300px) 1fr;
  gap: 16px; align-items: start;
}
label, legend { font-weight: 650; }
select {
  width: 100%; margin-top: 6px; padding: 9px 34px 9px 10px;
  border: 1px solid var(--border); border-radius: 8px; background: var(--surface);
  color: var(--text); font: inherit;
}
select:focus-visible, input:focus-visible, button:focus-visible {
  outline: 3px solid var(--accent); outline-offset: 2px;
}
fieldset { min-width: 0; margin: 0; padding: 0; border: 0; }
.legend { display: flex; flex-wrap: wrap; gap: 7px 14px; margin-top: 7px; }
.legend label { display: inline-flex; align-items: center; gap: 7px; font-weight: 550; }
.swatch { width: 20px; height: 3px; border-radius: 3px; background: var(--series); }
.section-head {
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  flex-wrap: wrap; margin-bottom: 12px;
}
.section-head p, .chart-note { margin: 0; color: var(--muted); font-size: 13px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
caption { text-align: left; color: var(--muted); padding: 0 0 8px; }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: right; }
th { color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: .03em; }
th { white-space: normal; overflow-wrap: normal; min-width: 88px; }
th:nth-child(3) { min-width: 180px; }
td { white-space: nowrap; }
th:first-child, th:last-child { white-space: nowrap; overflow-wrap: normal; }
th:nth-child(2), td:nth-child(2), td.error { text-align: left; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr.is-muted { opacity: .42; }
.rank { width: 48px; font-variant-numeric: tabular-nums; }
.name { display: inline-flex; align-items: center; gap: 8px; font-weight: 650; }
.status-ok { color: var(--ok); }
.status-failed { color: var(--bad); }
.error { max-width: 440px; overflow: hidden; text-overflow: ellipsis; }
.replay-button, .dialog-button {
  padding: 6px 10px; border: 1px solid var(--border); border-radius: 7px;
  background: var(--surface-2); color: var(--text); font: inherit; cursor: pointer;
  white-space: nowrap;
}
.replay-button:hover, .dialog-button:hover { border-color: var(--accent); }
.replay-unavailable { color: var(--muted); font-size: 12px; }
.command-dialog {
  width: min(760px, calc(100% - 32px)); border: 1px solid var(--border);
  border-radius: 14px; padding: 20px; background: var(--surface); color: var(--text);
  box-shadow: var(--shadow);
}
.command-dialog::backdrop { background: rgba(9, 16, 27, .5); }
.command-dialog h2 { margin-bottom: 10px; }
.command-dialog p { margin: 0 0 10px; color: var(--muted); }
.command-dialog pre {
  margin: 0; padding: 14px; overflow-x: auto; border: 1px solid var(--border);
  border-radius: 9px; background: var(--surface-2); color: var(--text);
  white-space: pre-wrap; overflow-wrap: anywhere;
}
.dialog-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 14px; }
#arena-chart { width: 100%; min-height: 1460px; }
.empty { padding: 44px 12px; text-align: center; color: var(--muted); }
footer { margin: 18px 2px 0; color: var(--muted); font-size: 12px; }
.sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}
@media (max-width: 780px) {
  main { padding: 16px; }
  .panel { padding: 14px; border-radius: 12px; }
  .controls { grid-template-columns: 1fr; }
  #arena-chart { min-height: 1280px; }
}
@media print {
  body { background: #fff; }
  main { max-width: none; padding: 0; }
  .panel { break-inside: avoid; box-shadow: none; }
}
"""


_REPORT_JS = r"""
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("arena-report-data").textContent);
  const configSelect = document.getElementById("configuration");
  const workloadSelect = document.getElementById("workload");
  const legend = document.getElementById("autoscaler-legend");
  const tableBody = document.getElementById("leaderboard-body");
  const tableHead = document.getElementById("leaderboard-head");
  const tableCaption = document.getElementById("leaderboard-caption");
  const chart = document.getElementById("arena-chart");
  const chartNote = document.getElementById("chart-note");
  const live = document.getElementById("scope-status");
  const fallback = document.getElementById("chart-fallback");
  const replayDialog = document.getElementById("replay-dialog");
  const replayCommand = document.getElementById("replay-command");
  const replayCopy = document.getElementById("replay-copy");
  const replayCopyStatus = document.getElementById("replay-copy-status");
  const replayHelp = document.getElementById("replay-help");
  const enabled = new Map(data.autoscalers.map(item => [item.name, true]));

  const text = (tag, value, className) => {
    const node = document.createElement(tag);
    node.textContent = value;
    if (className) node.className = className;
    return node;
  };
  const option = (value, label) => {
    const node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    return node;
  };
  data.configurations.forEach(item => configSelect.append(option(item.id, item.label)));

  function scopesForConfig() {
    return data.scopes.filter(scope => scope.configuration_id === configSelect.value);
  }
  function syncWorkloads() {
    const previous = workloadSelect.value;
    workloadSelect.replaceChildren();
    scopesForConfig().forEach(scope => workloadSelect.append(option(scope.workload, scope.workload)));
    if ([...workloadSelect.options].some(item => item.value === previous)) {
      workloadSelect.value = previous;
    }
  }
  function currentScope() {
    return data.scopes.find(scope =>
      scope.configuration_id === configSelect.value &&
      scope.workload === workloadSelect.value
    );
  }
  function currentConfiguration() {
    return data.configurations.find(item => item.id === configSelect.value);
  }
  function formatMetric(metric, value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const numeric = Number(value);
    if (metric === "good_rate") return `${(numeric * 100).toFixed(1)}%`;
    if (metric.endsWith("_ms")) return numeric >= 1000 ? numeric.toFixed(0) : numeric.toFixed(1);
    if (metric === "gpu_hours") return numeric.toFixed(3);
    if (Number.isInteger(numeric)) return String(numeric);
    if (Math.abs(numeric) >= 100) return numeric.toFixed(1);
    if (Math.abs(numeric) >= 1) return numeric.toFixed(2);
    return numeric.toFixed(4);
  }
  function metricLabel(metric) {
    const labels = {
      goodput_per_gpu: "Goodput / average GPU",
      goodput_rps: "Goodput (req/s)",
      good_rate: "Good rate",
      good_count: "Good requests",
      gpu_hours: "GPU-hours",
      request_throughput_rps: "Throughput (req/s)",
      completed_requests: "Completed",
      duration_s: "Duration (s)",
      mean_ttft_ms: "Mean TTFT (ms)",
      p95_ttft_ms: "p95 TTFT (ms)",
      p99_ttft_ms: "p99 TTFT (ms)",
      mean_itl_ms: "Mean TPOT (ms)",
      p99_itl_ms: "p99 TPOT (ms)",
      oscillation_count: "Oscillations",
      scale_events: "Scale events"
    };
    return labels[metric] || metric.replaceAll("_", " ");
  }
  function renderLegend(scope) {
    legend.replaceChildren();
    data.autoscalers
      .filter(style => scope.results.some(result => result.autoscaler === style.name))
      .forEach(style => {
      const id = `series-${btoa(unescape(encodeURIComponent(style.name))).replace(/=+$/g, "")}`;
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = id;
      input.checked = enabled.get(style.name) !== false;
      input.addEventListener("change", () => {
        enabled.set(style.name, input.checked);
        renderLeaderboard(scope);
        renderPlot(scope);
      });
      const swatch = text("span", "", "swatch");
      swatch.style.setProperty("--series", style.color);
      label.append(input, swatch, document.createTextNode(style.name));
      legend.append(label);
    });
  }
  function renderLeaderboard(scope) {
    tableHead.replaceChildren();
    const header = document.createElement("tr");
    ["Rank", "Autoscaler", ...data.metric_columns, "KV reuse (whole run)", "Replay", "Status"].forEach(value => {
      const label = data.metric_columns.includes(value) ? metricLabel(value) : value;
      const th = text("th", value === data.rank_by ? `${label} (${data.rank_direction} is better)` : label);
      th.scope = "col";
      header.append(th);
    });
    tableHead.append(header);
    tableBody.replaceChildren();
    scope.results.forEach(result => {
      const row = document.createElement("tr");
      if (enabled.get(result.autoscaler) === false) row.className = "is-muted";
      row.append(text("td", result.rank === null ? "—" : String(result.rank), "rank"));
      const nameCell = document.createElement("td");
      const name = text("span", "", "name");
      const swatch = text("span", "", "swatch");
      swatch.style.setProperty("--series", result.color);
      name.append(swatch, document.createTextNode(result.autoscaler));
      nameCell.append(name);
      row.append(nameCell);
      data.metric_columns.forEach(metric => {
        row.append(text("td", formatMetric(metric, result.metrics[metric])));
      });
      const cacheRatio = result.cache.prefix_cache_reused_ratio;
      const cacheLabel = cacheRatio === null ? "Unavailable" :
        `${(cacheRatio * 100).toFixed(2)}%${result.cache.status === "aggregate_only" ? " · whole run only" : ""}`;
      const cacheCell = text("td", cacheLabel);
      cacheCell.title = result.cache.note;
      row.append(cacheCell);
      const replayCell = document.createElement("td");
      if (result.replay_command) {
        const replayButton = text("button", "Replay cmd", "replay-button");
        replayButton.type = "button";
        replayButton.title = `Show the single-run replay command for ${result.autoscaler}`;
        replayButton.addEventListener("click", () => {
          replayCommand.textContent = result.replay_command;
          const traceGuard = result.replay_trace_guarded ?
            " It also refuses changed source-trace content." :
            " No source-trace digest was available for this saved run.";
          const configHint = result.replay_uses_config_env ?
            " Set MATCH_CONFIG to the original Match Config path before running it." : "";
          replayHelp.textContent = "Run from gyms/planner-gym with its environment active. Runs only this matrix cell, refuses a changed resolved Match Config, and keeps artifacts without replacing the published leaderboard." + traceGuard + configHint;
          replayCopyStatus.textContent = "";
          if (typeof replayDialog.showModal === "function") replayDialog.showModal();
          else replayDialog.setAttribute("open", "");
        });
        replayCell.append(replayButton);
      } else {
        const unavailable = text("span", "Unavailable", "replay-unavailable");
        unavailable.title = "A validated source Match Config reference was not available for this saved result.";
        replayCell.append(unavailable);
      }
      row.append(replayCell);
      const status = document.createElement("td");
      if (result.status === "ok") {
        status.textContent = "OK";
        status.className = "status-ok";
      } else {
        status.textContent = `${result.error.type || "Failed"}: ${result.error.message || ""}`;
        status.className = "status-failed error";
      }
      row.append(status);
      tableBody.append(row);
    });
    tableCaption.textContent = `${scope.workload} · ranked by ${data.rank_by}`;
  }
  replayCopy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(replayCommand.textContent);
      replayCopyStatus.textContent = "Copied";
    } catch (_error) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(replayCommand);
      selection.removeAllRanges();
      selection.addRange(range);
      replayCopyStatus.textContent = "Selected — press Ctrl/Cmd+C";
    }
  });
  function laneAnnotation(textValue, y) {
    return {
      text: textValue, x: 0.5, y, xref: "paper", yref: "paper", showarrow: false,
      font: {color: getComputedStyle(document.documentElement).getPropertyValue("--muted").trim()}
    };
  }
  function elapsedTimeTicks(xmin, xmax) {
    const span = Math.max(xmax - xmin, .001);
    const units = span >= 2 * 86400 ? [86400, "days", "d"] :
      (span >= 2 * 3600 ? [3600, "hours", "h"] :
        (span >= 2 * 60 ? [60, "minutes", "m"] : [1, "seconds", "s"]));
    const rawStep = span / units[0] / 7;
    const magnitude = 10 ** Math.floor(Math.log10(Math.max(rawStep, 1e-9)));
    const fraction = rawStep / magnitude;
    const niceFraction = fraction <= 1 ? 1 : (fraction <= 2 ? 2 : (fraction <= 5 ? 5 : 10));
    const stepUnits = niceFraction * magnitude;
    const stepSeconds = stepUnits * units[0];
    const first = Math.ceil(xmin / stepSeconds - 1e-9) * stepSeconds;
    const tickvals = [];
    for (let value = first; value <= xmax + stepSeconds * 1e-6; value += stepSeconds) {
      tickvals.push(Number(value.toFixed(6)));
    }
    const digits = Math.min(
      4, Math.max(0, -Math.floor(Math.log10(Math.max(stepUnits, 1e-9))))
    );
    return {
      tickvals,
      ticktext: tickvals.map(value => `${(value / units[0]).toFixed(digits)}${units[2]}`),
      title: `Elapsed time (${units[1]})`
    };
  }
  function syncElapsedTimeTicks() {
    const range = chart._fullLayout?.xaxis?.range;
    if (!range || range.length !== 2) return;
    const ticks = elapsedTimeTicks(Number(range[0]), Number(range[1]));
    const signature = `${ticks.title}:${ticks.tickvals.join(",")}`;
    if (chart._arenaTimeTickSignature === signature) return;
    chart._arenaTimeTickSignature = signature;
    const update = {
      "xaxis.title.text": ticks.title,
      "xaxis.tickmode": "array",
      "xaxis.tickvals": ticks.tickvals,
      "xaxis.ticktext": ticks.ticktext
    };
    Plotly.relayout(chart, update);
  }
  function renderPlot(scope) {
    const selected = scope.results.filter(result =>
      result.status === "ok" && enabled.get(result.autoscaler) !== false
    );
    const traces = [];
    const isDisaggregated = data.topology === "disagg";
    const secondaryRoleShort = isDisaggregated ? "D" : "Agg";
    const secondaryRoleName = isDisaggregated ? "decode" : "aggregate";
    const arrivals = scope.arrivals.points || [];
    if (scope.arrivals.status === "ok" && arrivals.length) {
      const arrivalWidths = arrivals.map(point =>
        point.window_width_s || scope.arrivals.bucket_width_s || 1
      );
      traces.push({
        type: "bar", name: "Arriving requests", x: arrivals.map(point => point.time_s),
        y: arrivals.map(point => point.rps),
        customdata: arrivals.map((point, index) => [point.count, arrivalWidths[index]]),
        marker: {color: "rgba(116, 129, 151, .55)", line: {width: 0}},
        width: arrivalWidths.map(width => width * .88),
        hovertemplate: "t=%{x:.1f}s<br>%{y:.2f} arriving req/s<br>%{customdata[0]} arriving requests / %{customdata[1]}s<extra>shared workload</extra>",
        xaxis: "x", yaxis: "y", showlegend: false
      });
    }
    const laneCounts = {
      queue: 0, queueLayers: 0, ttft: 0, tpot: 0, kv: 0,
      routerKvHit: 0, schedulerReuse: 0,
      cacheTimeline: 0, cacheAggregate: 0, replicas: 0, gpus: 0
    };
    selected.forEach(result => {
      const queueMeasurements = result.timeline.filter(point => point.total_queued_requests !== null);
      const schedulerQueue = result.timeline.filter(point => point.scheduler_waiting_requests !== null);
      const routerQueue = result.timeline.filter(point => point.router_pending_requests !== null);
      const latency = result.timeline.filter(point => point.ttft_ms !== null);
      const tpot = result.timeline.filter(point => point.tpot_ms !== null);
      const activeKv = result.timeline.filter(point => point.active_kv_cache_utilization !== null);
      const physicalKv = result.timeline.filter(point => point.physical_kv_cache_utilization !== null);
      const schedulerReuse = result.timeline.filter(point => point.scheduler_cache_reuse !== null);
      const routerKvHit = result.timeline.filter(point => point.router_kv_hit_rate !== null);
      const decisions = result.timeline.filter(point => point.requested_replicas !== null);
      const capacity = result.timeline.filter(point => !point.decision_only);
      const cachePoints = result.cache.timeline || [];
      const cacheAggregate = result.cache.prefix_cache_reused_ratio;
      if (queueMeasurements.length) laneCounts.queue += 1;
      if (result.timeline.some(point => point.queue_telemetry_semantics === "scheduler_waiting_plus_router_pending")) laneCounts.queueLayers += 1;
      if (latency.length) laneCounts.ttft += 1;
      if (tpot.length) laneCounts.tpot += 1;
      if (activeKv.length || physicalKv.length) laneCounts.kv += 1;
      if (schedulerReuse.length) laneCounts.schedulerReuse += 1;
      if (routerKvHit.length) laneCounts.routerKvHit += 1;
      if (cachePoints.length) laneCounts.cacheTimeline += 1;
      if (cacheAggregate !== null) laneCounts.cacheAggregate += 1;
      if (capacity.length) {
        laneCounts.replicas += 1;
        laneCounts.gpus += 1;
      }
      const common = {
        type: "scatter", mode: "lines", name: result.autoscaler,
        legendgroup: result.autoscaler, showlegend: false,
        line: {color: result.color, width: 2, dash: result.dash}
      };
      if (queueMeasurements.length) traces.push({
        ...common, line: {...common.line, shape: "hv"}, connectgaps: false,
        x: queueMeasurements.map(point => point.time_s),
        y: queueMeasurements.map(point => point.total_queued_requests),
        xaxis: "x", yaxis: "y2",
        customdata: queueMeasurements.map(point => [
          point.queued_prefill_requests === null ? "unavailable" : point.queued_prefill_requests,
          point.queued_decode_requests === null ? "unavailable" : point.queued_decode_requests,
          point.scheduler_waiting_prefill_requests === null ? "unavailable" : point.scheduler_waiting_prefill_requests,
          point.scheduler_waiting_decode_requests === null ? "unavailable" : point.scheduler_waiting_decode_requests,
          point.router_pending_prefill_requests === null ? "unavailable" : point.router_pending_prefill_requests,
          point.router_pending_decode_requests === null ? "unavailable" : point.router_pending_decode_requests,
          point.preemptions === null ? "unavailable" : point.preemptions
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>%{y} total requests in queue<br>prefill total=%{customdata[0]}, ${secondaryRoleName} total=%{customdata[1]}<br>scheduler waiting P=%{customdata[2]}, ${secondaryRoleShort}=%{customdata[3]}<br>router pending P=%{customdata[4]}, ${secondaryRoleShort}=%{customdata[5]}<br>preemptions in window=%{customdata[6]}<extra>total</extra>`
      });
      if (schedulerQueue.length) traces.push({
        ...common, mode: "lines", name: `${result.autoscaler} · scheduler waiting`,
        line: {color: result.color, width: 1, dash: "dot", shape: "hv"},
        x: schedulerQueue.map(point => point.time_s),
        y: schedulerQueue.map(point => point.scheduler_waiting_requests),
        xaxis: "x", yaxis: "y2",
        customdata: schedulerQueue.map(point => [
          point.scheduler_waiting_prefill_requests,
          point.scheduler_waiting_decode_requests
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>%{y} scheduler-waiting requests<br>P=%{customdata[0]}, ${secondaryRoleShort}=%{customdata[1]}<extra>queue layer</extra>`
      });
      if (routerQueue.length) traces.push({
        ...common, mode: "lines", name: `${result.autoscaler} · router pending`,
        line: {color: result.color, width: 1, dash: "dash", shape: "hv"},
        x: routerQueue.map(point => point.time_s),
        y: routerQueue.map(point => point.router_pending_requests),
        xaxis: "x", yaxis: "y2",
        customdata: routerQueue.map(point => [
          point.router_pending_prefill_requests,
          point.router_pending_decode_requests
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>%{y} router-pending requests<br>P=%{customdata[0]}, ${secondaryRoleShort}=%{customdata[1]}<extra>queue layer</extra>`
      });
      if (latency.length) traces.push({
        ...common, x: latency.map(point => point.latency_time_s),
        y: latency.map(point => point.ttft_ms), xaxis: "x", yaxis: "y3",
        customdata: latency.map(point => [
          point.window_start_s, point.time_s,
          point.ttft_samples !== null ? `${point.ttft_samples} TTFT samples` :
            (point.completed_requests !== null ? `${point.completed_requests} completed requests` : "sample count unavailable")
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>mean TTFT=%{y:.2f} ms<br>window %{customdata[0]:.1f}–%{customdata[1]:.1f}s<br>%{customdata[2]}<extra></extra>`
      });
      if (tpot.length) traces.push({
        ...common, x: tpot.map(point => point.latency_time_s),
        y: tpot.map(point => point.tpot_ms), xaxis: "x", yaxis: "y4",
        customdata: tpot.map(point => [
          point.window_start_s, point.time_s,
          point.tpot_samples !== null ? `${point.tpot_samples} TPOT samples` :
            (point.completed_requests !== null ? `${point.completed_requests} completed requests` : "sample count unavailable")
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>mean TPOT=%{y:.2f} ms<br>window %{customdata[0]:.1f}–%{customdata[1]:.1f}s<br>%{customdata[2]}<extra></extra>`
      });
      if (activeKv.length) traces.push({
        ...common, line: {...common.line, shape: "hv"},
        name: `${result.autoscaler} · active KV`,
        x: activeKv.map(point => point.time_s),
        y: activeKv.map(point => point.active_kv_cache_utilization * 100),
        xaxis: "x", yaxis: "y5",
        customdata: activeKv.map(point => [
          point.active_kv_blocks, point.total_kv_blocks,
          point.prefill_active_kv_cache_utilization === null ? "unavailable" : `${(point.prefill_active_kv_cache_utilization * 100).toFixed(2)}%`,
          point.decode_active_kv_cache_utilization === null ? "unavailable" : `${(point.decode_active_kv_cache_utilization * 100).toFixed(2)}%`
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>active KV pressure=%{y:.2f}%<br>%{customdata[0]} / %{customdata[1]} blocks<br>P=%{customdata[2]}, ${secondaryRoleShort}=%{customdata[3]}<extra>active blocks</extra>`
      });
      if (physicalKv.length) traces.push({
        ...common, line: {color: result.color, width: 1, dash: "dot", shape: "hv"},
        name: `${result.autoscaler} · physical KV`,
        x: physicalKv.map(point => point.time_s),
        y: physicalKv.map(point => point.physical_kv_cache_utilization * 100),
        xaxis: "x", yaxis: "y5",
        customdata: physicalKv.map(point => [
          point.active_kv_blocks, point.inactive_kv_blocks, point.total_kv_blocks,
          point.prefill_physical_kv_cache_utilization === null ? "unavailable" : `${(point.prefill_physical_kv_cache_utilization * 100).toFixed(2)}%`,
          point.decode_physical_kv_cache_utilization === null ? "unavailable" : `${(point.decode_physical_kv_cache_utilization * 100).toFixed(2)}%`
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>physical KV residency=%{y:.2f}%<br>active=%{customdata[0]}, inactive/reusable=%{customdata[1]}, capacity=%{customdata[2]} blocks<br>P=%{customdata[3]}, ${secondaryRoleShort}=%{customdata[4]}<extra>resident blocks</extra>`
      });
      if (routerKvHit.length) traces.push({
        ...common,
        name: `${result.autoscaler} · router KV hit rate`,
        x: routerKvHit.map(point => point.latency_time_s),
        y: routerKvHit.map(point => point.router_kv_hit_rate * 100),
        xaxis: "x", yaxis: "y6",
        customdata: routerKvHit.map(point => [
          point.window_start_s, point.time_s,
          point.router_kv_hit_samples === null ? "sample count unavailable" : `${point.router_kv_hit_samples} routed requests`
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>router KV hit rate=%{y:.2f}%<br>window %{customdata[0]:.1f}–%{customdata[1]:.1f}s<br>%{customdata[2]}<extra>router overlap</extra>`
      });
      if (schedulerReuse.length) traces.push({
        ...common,
        name: `${result.autoscaler} · scheduler cache reuse`,
        line: {color: result.color, width: 1, dash: "dot"},
        x: schedulerReuse.map(point => point.time_s),
        y: schedulerReuse.map(point => point.scheduler_cache_reuse * 100),
        xaxis: "x", yaxis: "y6",
        customdata: schedulerReuse.map(point => [
          point.scheduler_cache_hit_tokens, point.scheduler_cache_total_tokens,
          point.prefill_scheduler_cache_reuse === null ? "unavailable" : `${(point.prefill_scheduler_cache_reuse * 100).toFixed(2)}%`,
          point.decode_scheduler_cache_reuse === null ? "unavailable" : `${(point.decode_scheduler_cache_reuse * 100).toFixed(2)}%`
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>scheduler cache reuse=%{y:.2f}%<br>%{customdata[0]} / %{customdata[1]} observed tokens hit<br>P=%{customdata[2]}, ${secondaryRoleShort}=%{customdata[3]}<extra>telemetry window</extra>`
      });
      if (!schedulerReuse.length && cachePoints.length) traces.push({
        ...common, x: cachePoints.map(point => point.time_s),
        y: cachePoints.map(point => point.prefix_cache_reused_ratio * 100),
        xaxis: "x", yaxis: "y6",
        customdata: cachePoints.map(point => [
          point.window_start_s, point.time_s, point.reused_input_tokens,
          point.input_tokens, point.completed_requests === null ? "completion count unavailable" : `${point.completed_requests} completed requests`
        ]),
        hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>realized prefix-cache reuse=%{y:.2f}%<br>window %{customdata[0]:.1f}–%{customdata[1]:.1f}s<br>%{customdata[2]} / %{customdata[3]} input tokens reused<br>%{customdata[4]}<extra></extra>`
      });
      if (!schedulerReuse.length && !cachePoints.length && cacheAggregate !== null) traces.push({
        ...common, mode: "lines", line: {...common.line, width: 2, dash: "dot"},
        x: [0, Math.max(scope.duration_s, 1)], y: [cacheAggregate * 100, cacheAggregate * 100],
        xaxis: "x", yaxis: "y6",
        customdata: ["whole-run summary", "whole-run summary"],
        hovertemplate: `${result.autoscaler}<br>realized prefix-cache reuse=%{y:.2f}%<br>%{customdata}<br>exact time series unavailable<extra></extra>`
      });
      if (capacity.length) {
        const readyRoles = isDisaggregated ? [
          {name: "prefill", field: "active_prefill", width: 2.5, opacity: 1},
          {name: "decode", field: "active_decode", width: 1.4, opacity: .58}
        ] : [
          {name: "aggregate", field: "active_decode", width: 2, opacity: 1}
        ];
        readyRoles.forEach(role => traces.push({
          ...common, name: `${result.autoscaler} · ${role.name} ready`,
          line: {...common.line, width: role.width, shape: "hv"},
          opacity: role.opacity,
          x: capacity.map(point => point.time_s),
          y: capacity.map(point => point[role.field]),
          xaxis: "x", yaxis: "y7",
          hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>%{y} ready ${role.name} replicas<extra></extra>`
        }));
        traces.push({
          ...common, line: {...common.line, shape: "hv"},
          x: capacity.map(point => point.time_s),
          y: capacity.map(point => point.provisioned_gpus),
          xaxis: "x", yaxis: "y8",
          customdata: capacity.map(point => [
            point.provisioned_prefill, point.prefill_gpus_per_replica,
            point.provisioned_decode, point.decode_gpus_per_replica
          ]),
          hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>%{y} observed provisioned GPUs<br>P=%{customdata[0]}×%{customdata[1]}, ${secondaryRoleShort}=%{customdata[2]}×%{customdata[3]}<extra></extra>`
        });
      }
      if (decisions.length) {
        const decisionRoles = isDisaggregated ? [
          {name: "prefill", field: "requested_prefill", symbol: "triangle-up-open"},
          {name: "decode", field: "requested_decode", symbol: "triangle-down-open"}
        ] : [
          {name: "aggregate", field: "requested_decode", symbol: "triangle-up-open"}
        ];
        decisionRoles.forEach(role => traces.push({
          ...common, mode: "markers",
          marker: {
            color: result.color, size: 10, symbol: role.symbol,
            line: {color: result.color, width: 2}
          },
          x: decisions.map(point => point.time_s),
          y: decisions.map(point => point[role.field]),
          xaxis: "x", yaxis: "y7",
          hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>requested %{y} ${role.name} replicas<extra></extra>`
        }));
        const decisionMarker = {
          color: result.color, size: 10, symbol: "triangle-up-open",
          line: {color: result.color, width: 2}
        };
        traces.push({
          ...common, mode: "markers", marker: decisionMarker,
          x: decisions.map(point => point.time_s),
          y: decisions.map(point => point.requested_gpus),
          xaxis: "x", yaxis: "y8",
          customdata: decisions.map(point => [
            point.requested_prefill, point.prefill_gpus_per_replica,
            point.requested_decode, point.decode_gpus_per_replica
          ]),
          hovertemplate: `${result.autoscaler}<br>t=%{x:.1f}s<br>requested %{y} GPUs<br>P=%{customdata[0]}×%{customdata[1]}, ${secondaryRoleShort}=%{customdata[2]}×%{customdata[3]}<extra></extra>`
        });
      }
    });
    const style = getComputedStyle(document.documentElement);
    const foreground = style.getPropertyValue("--text").trim();
    const muted = style.getPropertyValue("--muted").trim();
    const border = style.getPropertyValue("--border").trim();
    const surface = style.getPropertyValue("--surface").trim();
    const annotations = [];
    if (scope.arrivals.status !== "ok") annotations.push(laneAnnotation(scope.arrivals.note, .945));
    const queueUnavailable = selected
      .filter(result => !result.timeline.some(point => point.total_queued_requests !== null))
      .map(result => result.autoscaler);
    if (!laneCounts.queue) {
      annotations.push(laneAnnotation("Total requests in queue unavailable; no values inferred.", .8125));
    } else if (queueUnavailable.length) {
      annotations.push(laneAnnotation(`Queue depth unavailable for ${queueUnavailable.join(", ")}; no values inferred.`, .8125));
    }
    if (!laneCounts.ttft) annotations.push(laneAnnotation("TTFT timeline unavailable for the selected series.", .6825));
    if (!laneCounts.tpot) annotations.push(laneAnnotation("TPOT timeline unavailable for the selected series.", .5525));
    if (!laneCounts.kv) {
      annotations.push(laneAnnotation("KV cache utilization timeline unavailable for the selected series.", .4225));
    }
    if (!laneCounts.schedulerReuse && !laneCounts.cacheTimeline && laneCounts.cacheAggregate) {
      annotations.push(laneAnnotation("Whole-run cache summaries shown; exact time series unavailable.", .2925));
    } else if (!laneCounts.routerKvHit && !laneCounts.schedulerReuse && !laneCounts.cacheTimeline && !laneCounts.cacheAggregate) {
      annotations.push(laneAnnotation("Router KV hit rate and scheduler cache reuse are unavailable for the selected series.", .2925));
    }
    if (!laneCounts.replicas) annotations.push(laneAnnotation("Replica timeline unavailable for the selected series.", .1675));
    if (!laneCounts.gpus) annotations.push(laneAnnotation("Provisioned GPU timeline unavailable for the selected series.", .0475));
    const target = currentConfiguration()?.sla || {};
    const shapes = [];
    if (target.ttft_ms !== null && target.ttft_ms !== undefined) shapes.push({
      type: "line", xref: "x", yref: "y3", x0: 0, x1: Math.max(scope.duration_s, 1),
      y0: target.ttft_ms, y1: target.ttft_ms, line: {color: muted, width: 1, dash: "dot"}
    });
    if (target.tpot_ms !== null && target.tpot_ms !== undefined) shapes.push({
      type: "line", xref: "x", yref: "y4", x0: 0, x1: Math.max(scope.duration_s, 1),
      y0: target.tpot_ms, y1: target.tpot_ms, line: {color: muted, width: 1, dash: "dot"}
    });
    const axis = (title, domain) => ({
      title: {text: title, standoff: 8}, domain, rangemode: "tozero",
      gridcolor: border, zerolinecolor: border, color: foreground,
      automargin: true, fixedrange: true
    });
    const initialTicks = elapsedTimeTicks(0, Math.max(scope.duration_s, 1));
    const layout = {
      height: 1460, margin: {l: 112, r: 28, t: 18, b: 62}, paper_bgcolor: surface,
      plot_bgcolor: surface, font: {family: "Inter, system-ui, sans-serif", color: foreground, size: 12},
      hovermode: "x unified", hoversubplots: "axis", hoverdistance: 5,
      bargap: .08,
      showlegend: false, annotations, shapes,
      xaxis: {
        title: {text: initialTicks.title},
        range: [0, Math.max(scope.duration_s, 1)],
        tickmode: "array", tickvals: initialTicks.tickvals,
        ticktext: initialTicks.ticktext,
        gridcolor: border, color: foreground, zerolinecolor: border,
        automargin: true, anchor: "free", position: 0
      },
      yaxis: axis("Arriving requests (req/s)", [.89, 1]),
      yaxis2: axis("Requests in queue", [.76, .865]),
      yaxis3: axis("TTFT (ms)", [.63, .735]),
      yaxis4: axis("TPOT (ms)", [.50, .605]),
      yaxis5: {...axis("KV cache utilization (%)", [.37, .475]), range: [0, 100]},
      yaxis6: {...axis("KV hit / cache reuse (%)", [.24, .345]), range: [0, 100]},
      yaxis7: axis("Ready replicas", [.12, .215]),
      yaxis8: axis("Provisioned GPUs", [0, .095]),
      uirevision: scope.id
    };
    const renderVersion = (chart._arenaRenderVersion || 0) + 1;
    chart._arenaRenderVersion = renderVersion;
    chart._arenaTimeTickSignature = null;
    Plotly.react(chart, traces, layout, {
      responsive: true, displaylogo: false, scrollZoom: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }).then(() => {
      if (chart._arenaRenderVersion !== renderVersion) return;
      if (chart._arenaRelayoutHandler && typeof chart.removeListener === "function") {
        chart.removeListener("plotly_relayout", chart._arenaRelayoutHandler);
      }
      let frame = null;
      chart._arenaRelayoutHandler = event => {
        const changedRange = Object.keys(event).some(key =>
          /^xaxis\d*\.(range(?:\[[01]\])?|autorange)$/.test(key)
        );
        if (!changedRange) return;
        if (frame !== null) cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => {
          frame = requestAnimationFrame(() => {
            frame = null;
            syncElapsedTimeTicks();
          });
        });
      };
      chart.on("plotly_relayout", chart._arenaRelayoutHandler);
      syncElapsedTimeTicks();
    });
    const selectedNames = selected.map(result => result.autoscaler);
    fallback.textContent = `Eight aligned time lanes for ${scope.workload}. ` +
      `Selected autoscalers: ${selectedNames.join(", ") || "none"}. ` +
      `${scope.arrivals.note} ` +
      (laneCounts.queue ? `${laneCounts.queue} autoscaler queue timelines available.` :
        "Queue telemetry unavailable; no values inferred.");
    const aggregateOnly = selected.filter(result =>
      result.cache.status === "aggregate_only" &&
      !result.timeline.some(point => point.scheduler_cache_reuse !== null)
    ).map(result => result.autoscaler);
    const cacheNote = aggregateOnly.length ?
      ` Cache reuse is available only as a whole-run summary for ${aggregateOnly.join(", ")}; no time series is inferred.` : "";
    const queueNote = !laneCounts.queue ?
      " Queue depth is unavailable; no values are inferred." :
      (queueUnavailable.length ?
        ` Queue depth is unavailable for ${queueUnavailable.join(", ")}; no values are inferred.` :
        (laneCounts.queueLayers === laneCounts.queue ?
          " Queue depth is scheduler waiting plus router pending; dotted and dashed lines show those layers." :
          (laneCounts.queueLayers ?
            " New queue timelines include scheduler waiting plus router pending; legacy series omit router pending." :
            " Legacy queue depth contains scheduler waiting only; router pending was not captured.")));
    const latencyWindowNote = data.telemetry_sample_interval_s === null ?
      "TTFT and TPOT are means over each saved telemetry window, not whole-run means." :
      `TTFT and TPOT are means over each saved telemetry window; ${Number(data.telemetry_sample_interval_s).toLocaleString()} seconds is the nominal sampling cadence, and the final window may be shorter.`;
    const capacityNote = isDisaggregated ?
      " Ready replicas are split by role: thicker lines and upward triangles are prefill; lighter lines and downward triangles are decode." :
      " Ready replicas show the aggregate role; hollow triangles are controller decisions.";
    const gpuCapacityNote = isDisaggregated ?
      " Provisioned GPUs remain the P+D total, with role composition in hover." :
      " Provisioned GPUs show the aggregate total.";
    chartNote.textContent = `${scope.arrivals.note}${queueNote} ${latencyWindowNote} ` +
      `Router KV hit rate uses the same saved windows (hover for exact bounds and sample counts). For vLLM, active KV excludes inactive reusable blocks and physical ` +
      `residency includes them; SGLang's legacy occupancy makes those lines equal. ` +
      `Router KV hit rate is request-weighted routing overlap; scheduler cache reuse is backend-observed hit tokens / cache tokens. ` +
      `Capacity lines are telemetry samples.${capacityNote}${gpuCapacityNote} ` +
      `Dotted horizontal latency lines are SLO targets.${cacheNote}`;
    live.textContent = `${currentConfiguration()?.label || ""}; ${scope.workload}; ` +
      `${selectedNames.length} autoscalers selected.`;
  }
  function renderScope() {
    const scope = currentScope();
    if (!scope) {
      chart.replaceChildren(text("div", "No results are available for this selection.", "empty"));
      return;
    }
    renderLegend(scope);
    renderLeaderboard(scope);
    renderPlot(scope);
  }
  configSelect.addEventListener("change", () => { syncWorkloads(); renderScope(); });
  workloadSelect.addEventListener("change", renderScope);
  if (data.configurations.length) configSelect.value = data.configurations[0].id;
  syncWorkloads();
  renderScope();
})();
"""


def _safe_script_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            separators=(",", ":"),
            sort_keys=False,
            allow_nan=False,
        )
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _frontend_report_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Drop persisted diagnostic detail that the standalone UI never reads."""

    output = dict(data)
    projected_scopes: list[dict[str, Any]] = []
    for raw_scope in _sequence(data.get("scopes")):
        scope = dict(_mapping(raw_scope))
        projected_results: list[dict[str, Any]] = []
        for raw_result in _sequence(scope.get("results")):
            result = dict(_mapping(raw_result))
            result["timeline"] = [
                {
                    key: value
                    for key, value in _mapping(raw_point).items()
                    if key in _FRONTEND_TIMELINE_FIELDS
                }
                for raw_point in _sequence(result.get("timeline"))
            ]
            projected_results.append(result)
        scope["results"] = projected_results
        projected_scopes.append(scope)
    output["scopes"] = projected_scopes
    return output


def render_match_report(
    report: Mapping[str, Any],
    *,
    replay_config_path: Path | str | None = None,
) -> str:
    """Return one complete, offline-capable interactive HTML document."""

    from plotly.offline import get_plotlyjs

    data = build_match_report_data(report, replay_config_path=replay_config_path)
    title = html.escape(data["title"])
    description = html.escape(data["description"])
    deployment = html.escape(data["deployment"])
    cold_start = html.escape(data["cold_start_label"])
    status = html.escape(data["status"].upper())
    summary = data["summary"]
    payload = _safe_script_json(_frontend_report_data(data))
    plotly_js = get_plotlyjs().replace("</script", r"<\/script")
    provenance = data["provenance"]
    provenance_text = " · ".join(
        value
        for value in (
            f"finished {provenance['finished_at']}"
            if provenance["finished_at"]
            else "",
            f"session {provenance['session_id']}" if provenance["session_id"] else "",
            f"git {provenance['git_commit'][:12]}" if provenance["git_commit"] else "",
        )
        if value
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title} · Autoscaling Arena</title>\n"
        f"<style>{_REPORT_CSS}</style>\n"
        f"<script>{plotly_js}</script>\n"
        "</head>\n<body>\n<main>\n"
        '<header class="hero">\n<div class="eyebrow">Autoscaling Arena</div>\n'
        f'<h1 id="report-title">{title}</h1>\n'
        f'<p class="lede">{description}</p>\n'
        '<div class="meta">\n'
        f'<span class="pill"><strong>Deployment</strong>{deployment}</span>\n'
        f'<span class="pill"><strong>Cold start</strong>{cold_start}</span>\n'
        f'<span class="pill"><strong>Status</strong>{status}</span>\n'
        f'<span class="pill"><strong>Runs</strong>{summary["succeeded_runs"]} succeeded · '
        f'{summary["failed_runs"]} failed · {summary["skipped_runs"]} skipped</span>\n'
        "</div>\n</header>\n"
        '<section class="panel controls" aria-label="Report filters">\n'
        '<label for="configuration">Configuration (SLA / repetition)'
        '<select id="configuration"></select></label>\n'
        '<label for="workload">Workload<select id="workload"></select></label>\n'
        "<fieldset><legend>Autoscalers</legend>"
        '<div class="legend" id="autoscaler-legend"></div></fieldset>\n'
        '<p id="scope-status" class="sr-only" aria-live="polite"></p>\n'
        "</section>\n"
        '<section class="panel" aria-labelledby="leaderboard-title">\n'
        '<div class="section-head"><h2 id="leaderboard-title">Leaderboard</h2>'
        f'<p>Ranked by <code>{html.escape(data["rank_by"])}</code>; '
        f'{html.escape(data["rank_direction"])} is better.</p></div>\n'
        '<div class="table-wrap"><table><caption id="leaderboard-caption"></caption>'
        '<thead id="leaderboard-head"></thead><tbody id="leaderboard-body"></tbody>'
        "</table></div>\n</section>\n"
        '<section class="panel" aria-labelledby="behavior-title">\n'
        '<div class="section-head"><h2 id="behavior-title">Behavior over time</h2>'
        "<p>Pan or box-zoom time; metric ranges stay fixed. Double-click to reset.</p></div>\n"
        '<p class="chart-note" id="chart-note"></p>\n'
        '<div id="arena-chart" role="img" aria-labelledby="behavior-title"></div>\n'
        '<p id="chart-fallback" class="sr-only"></p>\n'
        "</section>\n"
        '<dialog class="command-dialog" id="replay-dialog" '
        'aria-labelledby="replay-dialog-title">\n'
        '<h2 id="replay-dialog-title">Replay command</h2>\n'
        '<p id="replay-help"></p>\n'
        '<pre id="replay-command"></pre>\n'
        '<div class="dialog-actions"><span id="replay-copy-status" '
        'role="status"></span><button class="dialog-button" type="button" '
        'id="replay-copy">Copy</button><form method="dialog"><button '
        'class="dialog-button" type="submit">Close</button></form></div>\n'
        "</dialog>\n"
        f"<footer>{html.escape(provenance_text)}</footer>\n"
        f'<script id="arena-report-data" type="application/json">{payload}</script>\n'
        f"<script>{_REPORT_JS}</script>\n"
        "</main>\n</body>\n</html>\n"
    )


__all__ = [
    "build_match_report_data",
    "render_match_report",
    "trace_arrival_series",
]
