#!/usr/bin/env python3
"""Analyze one NUMA frontend run or a complete mirrored experiment matrix.

This module is deliberately standalone and uses only the Python standard
library.  It understands both AIPerf's per-request JSONL export and its
aggregate JSON export, raw sysstat pidstat/mpstat text, and telemetry JSONL
written by ``telemetry.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_REMOTE_ROLES = ("mocker", "aiperf")
CLK_TCK = os.sysconf("SC_CLK_TCK")


def _read_text(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text()
    if "\n" not in source and len(source) < 4096:
        try:
            candidate = Path(source)
            if candidate.exists():
                return candidate.read_text()
        except OSError:
            pass
    return source


def _metric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Mapping):
        for key in ("avg", "value", "mean"):
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
    return None


def _percentile(sorted_values: Sequence[float], percent: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    data = sorted(float(value) for value in values)
    if not data:
        return {"count": 0, "mean": None, "min": None, "max": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "min": data[0],
        "max": data[-1],
        "p50": _percentile(data, 50),
        "p95": _percentile(data, 95),
        "p99": _percentile(data, 99),
    }


def _json_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _aggregate_aiperf(document: Mapping[str, Any]) -> dict[str, Any]:
    request_count = _metric_value(document.get("request_count"))
    latency = document.get("request_latency", {})
    ttft = document.get("time_to_first_token", document.get("time_to_first_output_token", {}))
    output_length = document.get("output_sequence_length", document.get("output_token_count", {}))
    benchmark_duration = _metric_value(document.get("benchmark_duration"))
    result: dict[str, Any] = {
        "source_format": "aggregate_json",
        "request_count": int(request_count or _metric_value(latency.get("count") if isinstance(latency, Mapping) else None) or 0),
        "request_throughput_rps": _metric_value(document.get("request_throughput")),
        "benchmark_duration_seconds": benchmark_duration,
        "request_latency_ms": _metric_value(latency),
        "request_latency_p50_ms": _metric_value(latency.get("p50")) if isinstance(latency, Mapping) else None,
        "request_latency_p95_ms": _metric_value(latency.get("p95")) if isinstance(latency, Mapping) else None,
        "request_latency_p99_ms": _metric_value(latency.get("p99")) if isinstance(latency, Mapping) else None,
        "ttft_ms": _metric_value(ttft),
        "ttft_p50_ms": _metric_value(ttft.get("p50")) if isinstance(ttft, Mapping) else None,
        "ttft_p95_ms": _metric_value(ttft.get("p95")) if isinstance(ttft, Mapping) else None,
        "ttft_p99_ms": _metric_value(ttft.get("p99")) if isinstance(ttft, Mapping) else None,
        "output_tokens_per_request": _metric_value(output_length),
        "output_tokens_per_request_min": _metric_value(output_length.get("min")) if isinstance(output_length, Mapping) else None,
        "output_tokens_per_request_max": _metric_value(output_length.get("max")) if isinstance(output_length, Mapping) else None,
        "total_output_tokens": _metric_value(document.get("total_output_tokens")),
        "cancelled_count": int(bool(document.get("was_cancelled", False))),
        "error_count": len(document.get("error_summary") or []),
    }
    if result["request_throughput_rps"] is None and request_count and benchmark_duration:
        result["request_throughput_rps"] = request_count / benchmark_duration
    if result["total_output_tokens"] is None and request_count and result["output_tokens_per_request"] is not None:
        result["total_output_tokens"] = request_count * result["output_tokens_per_request"]
    branch_stats = document.get("branch_stats")
    if isinstance(branch_stats, Mapping):
        result["error_count"] += int(branch_stats.get("children_errored", 0) or 0)
        result["error_count"] += int(branch_stats.get("parents_failed_due_to_child_error", 0) or 0)
    return result


def _request_metric(record: Mapping[str, Any], name: str) -> float | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    return _metric_value(metrics.get(name))


def _requests_aiperf(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    # Keep only scalar distributions.  A request record can contain a very
    # large inter-token-latency array, so retaining the records themselves can
    # exhaust memory on a high-throughput 150-second run.
    latencies: list[float] = []
    ttfts: list[float] = []
    output_lengths: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    request_count = 0
    cancelled = 0
    errors = 0
    token_count_mismatches = 0
    for record in records:
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        if metadata.get("benchmark_phase") not in (None, "profiling"):
            continue
        request_count += 1
        latency = _request_metric(record, "request_latency")
        if latency is not None:
            latencies.append(latency)
        ttft = _request_metric(record, "time_to_first_token")
        if ttft is None:
            ttft = _request_metric(record, "time_to_first_output_token")
        if ttft is not None:
            ttfts.append(ttft)
        output_length = _request_metric(record, "output_sequence_length")
        if output_length is None:
            output_length = _request_metric(record, "output_token_count")
        if output_length is not None:
            output_lengths.append(output_length)
        if metadata.get("was_cancelled"):
            cancelled += 1
        if metadata.get("error") or metadata.get("error_code") or record.get("error"):
            errors += 1
        output = _request_metric(record, "output_token_count")
        usage = _request_metric(record, "usage_completion_tokens")
        if output is not None and usage is not None and output != usage:
            token_count_mismatches += 1
        start = metadata.get("request_start_ns")
        end = metadata.get("request_end_ns")
        if isinstance(start, int):
            starts.append(start)
        if isinstance(end, int):
            ends.append(end)
    duration = (max(ends) - min(starts)) / 1e9 if starts and ends and max(ends) > min(starts) else None
    latency_dist = _distribution(latencies)
    ttft_dist = _distribution(ttfts)
    result = {
        "source_format": "request_jsonl",
        "request_count": request_count,
        "request_throughput_rps": request_count / duration if duration else None,
        "benchmark_duration_seconds": duration,
        "request_latency_ms": latency_dist["mean"],
        "request_latency_p50_ms": latency_dist["p50"],
        "request_latency_p95_ms": latency_dist["p95"],
        "request_latency_p99_ms": latency_dist["p99"],
        "ttft_ms": ttft_dist["mean"],
        "ttft_p50_ms": ttft_dist["p50"],
        "ttft_p95_ms": ttft_dist["p95"],
        "ttft_p99_ms": ttft_dist["p99"],
        "output_tokens_per_request": statistics.fmean(output_lengths) if output_lengths else None,
        "output_tokens_per_request_min": min(output_lengths) if output_lengths else None,
        "output_tokens_per_request_max": max(output_lengths) if output_lengths else None,
        "total_output_tokens": sum(output_lengths) if output_lengths else None,
        "cancelled_count": cancelled,
        "error_count": errors,
        "token_count_mismatch_count": token_count_mismatches,
    }
    return result


def parse_aiperf(path: str | Path) -> dict[str, Any]:
    """Parse an AIPerf aggregate JSON file or per-request JSONL file."""

    path = Path(path)
    if path.suffix == ".jsonl":
        def records() -> Iterable[Mapping[str, Any]]:
            with path.open() as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                    if isinstance(record, Mapping):
                        yield record

        return _requests_aiperf(records())
    document = json.loads(path.read_text())
    if isinstance(document, list):
        return _requests_aiperf([record for record in document if isinstance(record, Mapping)])
    if not isinstance(document, Mapping):
        raise ValueError(f"unsupported AIPerf JSON root in {path}")
    if "request_throughput" in document or "request_count" in document:
        return _aggregate_aiperf(document)
    # A JSON file may still contain one request object.
    return _requests_aiperf([document])


def _parse_sysstat_rows(text: str, required_column: str) -> list[dict[str, str]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if required_column in fields:
            header = fields
            continue
        if header is None or line.startswith("Linux"):
            continue
        # Some locales insert AM/PM after the time in both header and rows, but
        # an `Average:` row does not.  Align the stable metric suffix from right.
        if len(fields) < len(header):
            continue
        if len(fields) > len(header):
            prefix = len(fields) - len(header)
            fields = fields[prefix:]
        rows.append(dict(zip(header, fields)))
    return rows


def parse_pidstat(source: str | Path) -> dict[str, Any]:
    rows = _parse_sysstat_rows(_read_text(source), "%CPU")
    samples: list[dict[str, float | int | str]] = []
    for row in rows:
        try:
            samples.append(
                {
                    "pid": int(row.get("PID", "0")),
                    "user_percent": float(row.get("%usr", 0)),
                    "system_percent": float(row.get("%system", row.get("%sys", 0))),
                    "cpu_percent": float(row["%CPU"]),
                    "command": row.get("Command", row.get("Command", "")),
                }
            )
        except ValueError:
            continue
    # Prefer explicit Average rows when present; ordinary sample rows remain a
    # portable fallback across sysstat versions.
    average_rows = [sample for sample, raw in zip(samples, rows) if "Average:" in raw.values()]
    selected = average_rows or samples
    return {
        "sample_count": len(selected),
        "cpu_percent": statistics.fmean(float(row["cpu_percent"]) for row in selected) if selected else None,
        "user_percent": statistics.fmean(float(row["user_percent"]) for row in selected) if selected else None,
        "system_percent": statistics.fmean(float(row["system_percent"]) for row in selected) if selected else None,
        "max_cpu_percent": max((float(row["cpu_percent"]) for row in selected), default=None),
        "samples": selected,
    }


def parse_mpstat(source: str | Path) -> list[dict[str, float | int]]:
    rows = _parse_sysstat_rows(_read_text(source), "%idle")
    samples: list[dict[str, float | int]] = []
    for row in rows:
        cpu_text = row.get("CPU")
        if cpu_text is None or not cpu_text.isdigit():
            continue
        try:
            user = float(row.get("%usr", 0)) + float(row.get("%nice", 0))
            system = float(row.get("%sys", 0)) + float(row.get("%irq", 0)) + float(row.get("%soft", 0))
            idle = float(row["%idle"])
            iowait = float(row.get("%iowait", 0))
        except ValueError:
            continue
        samples.append(
            {
                "cpu": int(cpu_text),
                "user_percent": user,
                "system_percent": system,
                "busy_percent": max(0.0, 100.0 - idle - iowait),
            }
        )
    return samples


def summarize_memory_samples(samples: Sequence[Mapping[int, int]]) -> dict[str, Any]:
    """Summarize private-anonymous page percentages across samples."""

    nodes = sorted({int(node) for sample in samples for node in sample})
    percentages: dict[int, list[float]] = {node: [] for node in nodes}
    totals: list[int] = []
    for sample in samples:
        total = sum(max(0, int(value)) for value in sample.values())
        if total <= 0:
            continue
        totals.append(total)
        for node in nodes:
            percentages[node].append(max(0, int(sample.get(node, 0))) / total * 100.0)
    return {
        "sample_count": len(totals),
        "mean_total_pages": statistics.fmean(totals) if totals else None,
        "node_percentages": {
            node: statistics.fmean(values) if values else None for node, values in percentages.items()
        },
        "min_node_percentages": {node: min(values) if values else None for node, values in percentages.items()},
        "max_node_percentages": {node: max(values) if values else None for node, values in percentages.items()},
    }


def _process_ticks_from_stat(text: str) -> tuple[int, int] | None:
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    try:
        return int(fields[11]), int(fields[12])
    except (IndexError, ValueError):
        return None


def _normalize_sampler_records(
    records: Sequence[Mapping[str, Any]], source_name: str
) -> list[dict[str, Any]]:
    """Normalize the harness's compact sampler format to telemetry schema v1.

    Keeping this adapter here makes the analysis usable with an interrupted
    run even when the separate telemetry process was not able to emit its final
    record.
    """

    normalized: list[dict[str, Any]] = []
    previous_ticks: tuple[int, int] | None = None
    previous_time: float | None = None
    previous_nic: Mapping[str, Any] | None = None
    previous_retransmits: int | None = None
    previous_out_segments: int | None = None
    for raw in records:
        monotonic = raw.get("monotonic_seconds")
        monotonic_value = float(monotonic) if isinstance(monotonic, (int, float)) else None
        elapsed = (
            monotonic_value - previous_time
            if monotonic_value is not None and previous_time is not None
            else None
        )
        roles: dict[str, Any] = {}
        if isinstance(raw.get("private_anon_pages"), Mapping):
            frontend: dict[str, Any] = {
                "alive": not bool(raw.get("process_disappeared")),
                "private_anon_pages_by_node": {
                    str(node): int(count)
                    for node, count in raw["private_anon_pages"].items()
                },
            }
            ticks = (
                _process_ticks_from_stat(str(raw["process_stat"]))
                if isinstance(raw.get("process_stat"), str)
                else None
            )
            if ticks is not None:
                frontend["user_ticks"], frontend["system_ticks"] = ticks
                frontend["cpu_ticks"] = sum(ticks)
                if previous_ticks is not None and elapsed and elapsed > 0:
                    user_delta = max(0, ticks[0] - previous_ticks[0])
                    system_delta = max(0, ticks[1] - previous_ticks[1])
                    frontend["cpu_seconds_delta"] = (user_delta + system_delta) / CLK_TCK
                    frontend["cpu_user_percent"] = user_delta / CLK_TCK / elapsed * 100.0
                    frontend["cpu_system_percent"] = system_delta / CLK_TCK / elapsed * 100.0
                    frontend["cpu_percent"] = (user_delta + system_delta) / CLK_TCK / elapsed * 100.0
                previous_ticks = ticks
            roles["frontend"] = frontend

        nic_entry: dict[str, Any] = {}
        current_nic = raw.get("nic")
        if isinstance(current_nic, Mapping):
            nic_entry.update(current_nic)
            if previous_nic is not None:
                for field in (
                    "rx_bytes",
                    "tx_bytes",
                    "rx_dropped",
                    "tx_dropped",
                    "rx_errors",
                    "tx_errors",
                ):
                    now, old = current_nic.get(field), previous_nic.get(field)
                    if isinstance(now, int) and isinstance(old, int) and now >= 0 and old >= 0:
                        nic_entry[f"{field}_delta"] = max(0, now - old)
            previous_nic = current_nic

        raw_tcp = raw.get("tcp") if isinstance(raw.get("tcp"), Mapping) else {}
        retransmits = raw.get("tcp_retransmits", raw_tcp.get("RetransSegs"))
        out_segments = raw_tcp.get("OutSegs")
        tcp: dict[str, Any] = {
            "retrans_segs": retransmits,
            "out_segs": out_segments,
        }
        if isinstance(retransmits, int) and isinstance(previous_retransmits, int):
            tcp["retrans_segs_delta"] = max(0, retransmits - previous_retransmits)
        if isinstance(retransmits, int):
            previous_retransmits = retransmits

        if isinstance(out_segments, int) and isinstance(previous_out_segments, int):
            tcp["out_segs_delta"] = max(0, out_segments - previous_out_segments)
        if isinstance(out_segments, int):
            previous_out_segments = out_segments

        speed = raw.get("link_speed_bits_per_second")
        if (
            isinstance(speed, int)
            and speed > 0
            and elapsed
            and elapsed > 0
            and isinstance(previous_nic, Mapping)
        ):
            # ``previous_nic`` has already been advanced above, so utilization
            # is populated while the per-counter deltas are calculated.
            for direction in ("rx", "tx"):
                delta = nic_entry.get(f"{direction}_bytes_delta")
                if isinstance(delta, int):
                    nic_entry[f"{direction}_link_utilization_percent"] = (
                        100.0 * delta * 8 / elapsed / speed
                    )

        normalized.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "sample",
                "timestamp_ns": len(normalized),
                "monotonic_ns": int(monotonic_value * 1e9) if monotonic_value is not None else 0,
                "elapsed_seconds": elapsed,
                "roles": roles,
                "pools": raw.get("pools", {}),
                "nics": {source_name: nic_entry} if nic_entry else {},
                "tcp": tcp,
            }
        )
        if monotonic_value is not None:
            previous_time = monotonic_value
    return normalized


def parse_telemetry(paths: Sequence[str | Path]) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        records = _json_records(path)
        if records and all(record.get("record_type") is None for record in records):
            records = _normalize_sampler_records(records, path.stem)
        for record in records:
            if record.get("record_type") == "sample":
                samples.append(record)
            elif record.get("record_type") == "metadata":
                metadata.append(record)
    samples.sort(key=lambda sample: int(sample.get("timestamp_ns", 0)))
    return {"metadata": metadata, "samples": samples}


def _role_telemetry(telemetry: Mapping[str, Any], role: str) -> dict[str, Any]:
    entries = [
        sample.get("roles", {}).get(role)
        for sample in telemetry.get("samples", [])
        if isinstance(sample.get("roles", {}).get(role), Mapping)
    ]
    entries = [entry for entry in entries if entry.get("alive")]
    cpu_deltas = [float(entry["cpu_seconds_delta"]) for entry in entries if isinstance(entry.get("cpu_seconds_delta"), (int, float))]
    user = [float(entry["cpu_user_percent"]) for entry in entries if isinstance(entry.get("cpu_user_percent"), (int, float))]
    system = [float(entry["cpu_system_percent"]) for entry in entries if isinstance(entry.get("cpu_system_percent"), (int, float))]
    memory_samples: list[dict[int, int]] = []
    for entry in entries:
        pages = entry.get("private_anon_pages_by_node")
        if isinstance(pages, Mapping):
            memory_samples.append({int(node): int(count) for node, count in pages.items()})
    return {
        "sample_count": len(entries),
        "cpu_seconds": sum(cpu_deltas) if cpu_deltas else None,
        "cpu_user_percent": statistics.fmean(user) if user else None,
        "cpu_system_percent": statistics.fmean(system) if system else None,
        "memory": summarize_memory_samples(memory_samples),
    }


def _pool_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    by_role: dict[str, list[float]] = defaultdict(list)
    for sample in telemetry.get("samples", []):
        for role, entry in sample.get("pools", {}).items():
            busy = entry.get("busy_percent") if isinstance(entry, Mapping) else None
            if isinstance(busy, (int, float)):
                by_role[str(role)].append(float(busy))
    return {
        role: {
            "sample_count": len(values),
            "mean_busy_percent": statistics.fmean(values),
            "max_busy_percent": max(values),
        }
        for role, values in by_role.items()
    }


def _nic_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    by_nic: dict[str, dict[str, Any]] = {}
    retransmits = 0
    retransmit_observed = False
    out_segments = 0
    out_segments_observed = False
    for sample in telemetry.get("samples", []):
        tcp_delta = sample.get("tcp", {}).get("retrans_segs_delta")
        if isinstance(tcp_delta, int):
            retransmits += tcp_delta
            retransmit_observed = True
        out_delta = sample.get("tcp", {}).get("out_segs_delta")
        if isinstance(out_delta, int):
            out_segments += out_delta
            out_segments_observed = True
        for name, entry in sample.get("nics", {}).items():
            if not isinstance(entry, Mapping):
                continue
            aggregate = by_nic.setdefault(
                str(name),
                {
                    "rx_dropped_delta": 0,
                    "tx_dropped_delta": 0,
                    "rx_errors_delta": 0,
                    "tx_errors_delta": 0,
                    "max_rx_link_utilization_percent": None,
                    "max_tx_link_utilization_percent": None,
                    "sample_count": 0,
                },
            )
            aggregate["sample_count"] += 1
            for field in ("rx_dropped_delta", "tx_dropped_delta", "rx_errors_delta", "tx_errors_delta"):
                if isinstance(entry.get(field), int):
                    aggregate[field] += entry[field]
            for direction in ("rx", "tx"):
                value = entry.get(f"{direction}_link_utilization_percent")
                key = f"max_{direction}_link_utilization_percent"
                if isinstance(value, (int, float)):
                    aggregate[key] = max(float(value), aggregate[key] if aggregate[key] is not None else 0.0)
    fraction = (
        retransmits / out_segments
        if retransmit_observed and out_segments_observed and out_segments > 0
        else None
    )
    return {
        "interfaces": by_nic,
        "tcp_retransmits_delta": retransmits if retransmit_observed else None,
        "tcp_out_segments_delta": out_segments if out_segments_observed else None,
        "tcp_retransmit_fraction": fraction,
    }


def summarize_mpstat(samples: Sequence[Mapping[str, Any]], node_cpus: Mapping[int, set[int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node, cpus in node_cpus.items():
        selected = [sample for sample in samples if int(sample["cpu"]) in cpus]
        result[str(node)] = {
            "sample_count": len(selected),
            "busy_percent": statistics.fmean(float(sample["busy_percent"]) for sample in selected) if selected else None,
            "user_percent": statistics.fmean(float(sample["user_percent"]) for sample in selected) if selected else None,
            "system_percent": statistics.fmean(float(sample["system_percent"]) for sample in selected) if selected else None,
        }
    return result


def summarize_pool_per_node_cpu(
    telemetry: Mapping[str, Any],
    node_cpus: Mapping[int, set[int]],
    pool: str = "frontend",
) -> dict[str, Any]:
    """Aggregate the harness sampler's per-CPU utilization by NUMA node."""

    by_node: dict[int, list[float]] = {node: [] for node in node_cpus}
    for sample in telemetry.get("samples", []):
        entry = sample.get("pools", {}).get(pool)
        if not isinstance(entry, Mapping):
            continue
        per_cpu = entry.get("per_cpu_busy_percent")
        if not isinstance(per_cpu, Mapping):
            continue
        values = {
            int(cpu): float(value)
            for cpu, value in per_cpu.items()
            if isinstance(value, (int, float))
        }
        for node, cpus in node_cpus.items():
            selected = [value for cpu, value in values.items() if cpu in cpus]
            if selected:
                by_node[node].append(statistics.fmean(selected))
    if not any(by_node.values()):
        return {}
    return {
        str(node): {
            "sample_count": len(values),
            "busy_percent": statistics.fmean(values) if values else None,
            "min_busy_percent": min(values) if values else None,
            "max_busy_percent": max(values) if values else None,
            "source": "sampler_per_cpu",
        }
        for node, values in by_node.items()
    }


def _check(name: str, status: str, observed: Any, requirement: str) -> dict[str, Any]:
    if status not in ("pass", "fail", "skipped"):
        raise ValueError(status)
    return {"name": name, "status": status, "observed": observed, "requirement": requirement}


def _memory_checks(arm: str, memory: Mapping[str, Any]) -> list[dict[str, Any]]:
    count = int(memory.get("sample_count", 0) or 0)
    means = memory.get("node_percentages", {})
    minimums = memory.get("min_node_percentages", {})
    maximums = memory.get("max_node_percentages", {})
    if count == 0:
        return [_check("frontend_memory_residency", "fail", None, "at least one private-anonymous memory sample")]

    def value(mapping: Mapping[Any, Any], node: int) -> float | None:
        candidate = mapping.get(node, mapping.get(str(node)))
        return float(candidate) if isinstance(candidate, (int, float)) else None

    if arm == "8+0":
        observed = value(minimums, 0)
        return [_check("frontend_memory_residency", "pass" if observed is not None and observed >= 99.0 else "fail", observed, "node 0 >= 99% in every sample")]
    if arm == "0+8":
        observed = value(minimums, 1)
        return [_check("frontend_memory_residency", "pass" if observed is not None and observed >= 99.0 else "fail", observed, "node 1 >= 99% in every sample")]
    if arm == "4+4":
        observed = {
            "node0_min": value(minimums, 0),
            "node0_max": value(maximums, 0),
            "node1_min": value(minimums, 1),
            "node1_max": value(maximums, 1),
            "node0_mean": value(means, 0),
            "node1_mean": value(means, 1),
        }
        valid = all(value is not None and 40.0 <= value <= 60.0 for value in observed.values())
        return [_check("frontend_memory_residency", "pass" if valid else "fail", observed, "each node remains between 40% and 60% in every sample")]
    return [_check("frontend_memory_residency", "skipped", None, "known arm placement")]


def _find_first(run_dir: Path, candidates: Sequence[str]) -> Path | None:
    for candidate in candidates:
        matches = sorted(run_dir.glob(candidate))
        if matches:
            return matches[0]
    return None


def _load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def _manifest_value(manifest: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in manifest:
            return manifest[name]
    workload = manifest.get("workload")
    if isinstance(workload, Mapping):
        for name in names:
            if name in workload:
                return workload[name]
    placement = manifest.get("placement")
    if isinstance(placement, Mapping):
        for name in names:
            if name in placement:
                return placement[name]
    return None


def analyze_run(
    run_dir: str | Path,
    *,
    arm: str | None = None,
    workload: str | None = None,
    aiperf_path: str | Path | None = None,
    telemetry_paths: Sequence[str | Path] = (),
    pidstat_path: str | Path | None = None,
    mpstat_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    expected_output_tokens: int | None = None,
    expected_total_output_tokens: int | None = None,
    node_cpus: Mapping[int, str | Sequence[int]] | None = None,
    remote_roles: Sequence[str] = DEFAULT_REMOTE_ROLES,
    max_remote_busy_percent: float = 85.0,
    max_nic_utilization_percent: float = 70.0,
    max_retransmits: int | None = None,
    max_retransmit_fraction: float = 0.001,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest_file = (
        Path(manifest_path)
        if manifest_path
        else _find_first(run_dir, ("run.json", "manifest.json", "run_manifest.json"))
    )
    manifest = _load_optional_json(manifest_file)
    arm = arm or _manifest_value(manifest, "arm", "placement_arm") or run_dir.name
    manifest_workload = manifest.get("workload")
    workload = (
        workload
        or _manifest_value(manifest, "workload_name", "name")
        or (manifest_workload if isinstance(manifest_workload, str) else None)
        or run_dir.parent.name
    )

    if aiperf_path is None:
        aiperf_file = _find_first(
            run_dir,
            (
                "**/profile_export_aiperf.json",
                "**/aiperf_summary.json",
                "**/profile_export.jsonl",
                "**/aiperf*.jsonl",
            ),
        )
    else:
        aiperf_file = Path(aiperf_path)
    if aiperf_file is None:
        raise FileNotFoundError(f"no AIPerf result found beneath {run_dir}")
    aiperf = parse_aiperf(aiperf_file)

    if not telemetry_paths:
        telemetry_paths = sorted(
            set(run_dir.glob("**/*telemetry*.jsonl"))
            | set(run_dir.glob("**/*samples.jsonl"))
        )
    telemetry = parse_telemetry(telemetry_paths) if telemetry_paths else {"metadata": [], "samples": []}
    frontend = _role_telemetry(telemetry, "frontend")
    pools = _pool_telemetry(telemetry)
    nics = _nic_telemetry(telemetry)

    if pidstat_path is None:
        pidstat_file = _find_first(run_dir, ("**/frontend*pidstat*.txt", "**/pidstat*.txt"))
    else:
        pidstat_file = Path(pidstat_path)
    pidstat = parse_pidstat(pidstat_file) if pidstat_file else {}

    duration = aiperf.get("benchmark_duration_seconds") or _manifest_value(manifest, "duration", "duration_seconds")
    request_count = int(aiperf.get("request_count", 0) or 0)
    cpu_seconds = frontend.get("cpu_seconds")
    if cpu_seconds is None and pidstat.get("cpu_percent") is not None and duration:
        cpu_seconds = float(pidstat["cpu_percent"]) / 100.0 * float(duration)
    cpu_ms_per_request = float(cpu_seconds) * 1000.0 / request_count if cpu_seconds is not None and request_count else None
    user_percent = frontend.get("cpu_user_percent", pidstat.get("user_percent"))
    system_percent = frontend.get("cpu_system_percent", pidstat.get("system_percent"))
    system_share = (
        float(system_percent) / (float(user_percent) + float(system_percent)) * 100.0
        if isinstance(user_percent, (int, float))
        and isinstance(system_percent, (int, float))
        and float(user_percent) + float(system_percent) > 0
        else None
    )

    node_cpu_map: dict[int, set[int]] = {}
    raw_node_map = node_cpus or _manifest_value(manifest, "node_cpus", "numa_node_cpus")
    if isinstance(raw_node_map, Mapping):
        for node, value in raw_node_map.items():
            if isinstance(value, str):
                # Local import keeps analyze.py executable by itself.
                from telemetry import parse_cpu_list

                node_cpu_map[int(node)] = parse_cpu_list(value)
            elif isinstance(value, Sequence):
                node_cpu_map[int(node)] = {int(cpu) for cpu in value}
    if mpstat_path is None:
        mpstat_file = _find_first(run_dir, ("**/frontend*mpstat*.txt", "**/mpstat*.txt"))
    else:
        mpstat_file = Path(mpstat_path)
    mpstat = parse_mpstat(mpstat_file) if mpstat_file else []
    per_node_cpu = summarize_mpstat(mpstat, node_cpu_map) if node_cpu_map and mpstat else {}
    if node_cpu_map and not per_node_cpu:
        per_node_cpu = summarize_pool_per_node_cpu(telemetry, node_cpu_map)

    metrics = {
        key: value
        for key, value in {
            "request_throughput_rps": aiperf.get("request_throughput_rps"),
            "request_latency_ms": aiperf.get("request_latency_ms"),
            "request_latency_p50_ms": aiperf.get("request_latency_p50_ms"),
            "request_latency_p95_ms": aiperf.get("request_latency_p95_ms"),
            "request_latency_p99_ms": aiperf.get("request_latency_p99_ms"),
            "ttft_ms": aiperf.get("ttft_ms"),
            "ttft_p50_ms": aiperf.get("ttft_p50_ms"),
            "ttft_p95_ms": aiperf.get("ttft_p95_ms"),
            "ttft_p99_ms": aiperf.get("ttft_p99_ms"),
            "frontend_cpu_ms_per_request": cpu_ms_per_request,
            "frontend_cpu_user_percent": user_percent,
            "frontend_cpu_system_percent": system_percent,
            "frontend_cpu_system_share_percent": system_share,
        }.items()
        if isinstance(value, (int, float))
    }

    checks: list[dict[str, Any]] = []
    checks.append(_check("requests_present", "pass" if request_count > 0 else "fail", request_count, "> 0"))
    errors = int(aiperf.get("error_count", 0) or 0)
    cancelled = int(aiperf.get("cancelled_count", 0) or 0)
    checks.append(_check("zero_errors_and_cancellations", "pass" if errors == 0 and cancelled == 0 else "fail", {"errors": errors, "cancelled": cancelled}, "both equal zero"))
    token_mismatches = aiperf.get("token_count_mismatch_count")
    if token_mismatches is not None:
        checks.append(
            _check(
                "server_and_client_token_counts_match",
                "pass" if int(token_mismatches) == 0 else "fail",
                int(token_mismatches),
                "zero mismatches",
            )
        )

    if expected_output_tokens is None:
        expected_output_tokens = _manifest_value(manifest, "expected_output_tokens", "output_sequence_length")
    if expected_output_tokens is not None:
        observed = {
            "min": aiperf.get("output_tokens_per_request_min"),
            "max": aiperf.get("output_tokens_per_request_max"),
        }
        valid = observed["min"] == int(expected_output_tokens) and observed["max"] == int(expected_output_tokens)
        checks.append(_check("output_tokens_per_request", "pass" if valid else "fail", observed, f"exactly {expected_output_tokens}"))
    elif expected_total_output_tokens is not None or _manifest_value(manifest, "expected_total_output_tokens") is not None:
        expected = int(expected_total_output_tokens or _manifest_value(manifest, "expected_total_output_tokens"))
        observed_total = aiperf.get("total_output_tokens")
        checks.append(_check("total_output_tokens", "pass" if observed_total == expected else "fail", observed_total, f"exactly {expected}"))
    else:
        checks.append(_check("output_token_count", "skipped", aiperf.get("total_output_tokens"), "expected count was not supplied"))

    checks.extend(_memory_checks(str(arm), frontend.get("memory", {})))
    for role in remote_roles:
        pool = pools.get(role)
        if pool is None:
            checks.append(_check(f"remote_{role}_headroom", "fail", None, f"mean busy < {max_remote_busy_percent}%"))
        else:
            observed = pool.get("mean_busy_percent")
            checks.append(_check(f"remote_{role}_headroom", "pass" if observed is not None and observed < max_remote_busy_percent else "fail", observed, f"mean busy < {max_remote_busy_percent}%"))

    if not nics["interfaces"]:
        checks.append(_check("nic_health", "skipped", None, "NIC counters available"))
    else:
        for name, nic in nics["interfaces"].items():
            drops = sum(int(nic.get(key, 0)) for key in ("rx_dropped_delta", "tx_dropped_delta", "rx_errors_delta", "tx_errors_delta"))
            utilizations = [nic.get("max_rx_link_utilization_percent"), nic.get("max_tx_link_utilization_percent")]
            known_utilizations = [float(value) for value in utilizations if isinstance(value, (int, float))]
            max_utilization = max(known_utilizations) if known_utilizations else None
            valid = drops == 0 and (max_utilization is None or max_utilization < max_nic_utilization_percent)
            checks.append(_check(f"nic_{name}_health", "pass" if valid else "fail", {"drops_and_errors": drops, "max_utilization_percent": max_utilization}, f"zero drops/errors and utilization < {max_nic_utilization_percent}% when speed is known"))
    retransmits = nics.get("tcp_retransmits_delta")
    retransmit_fraction = nics.get("tcp_retransmit_fraction")
    if isinstance(retransmit_fraction, (int, float)):
        retransmit_status = (
            "pass" if float(retransmit_fraction) <= max_retransmit_fraction else "fail"
        )
        retransmit_requirement = f"fraction <= {max_retransmit_fraction}"
        retransmit_observed: Any = {
            "delta": retransmits,
            "out_segments": nics.get("tcp_out_segments_delta"),
            "fraction": retransmit_fraction,
        }
    elif isinstance(retransmits, int) and max_retransmits is not None:
        retransmit_status = "pass" if retransmits <= max_retransmits else "fail"
        retransmit_requirement = f"delta <= {max_retransmits}"
        retransmit_observed = retransmits
    else:
        retransmit_status = "skipped"
        retransmit_requirement = "TCP retransmit fraction available"
        retransmit_observed = retransmits
    checks.append(
        _check(
            "tcp_retransmits",
            retransmit_status,
            retransmit_observed,
            retransmit_requirement,
        )
    )

    placement_validation = _load_optional_json(_find_first(run_dir, ("placement_validation.json", "**/placement_validation.json")))
    if placement_validation:
        valid = bool(placement_validation.get("valid", placement_validation.get("accepted", False)))
        checks.append(_check("cpu_affinity_and_memory_policy", "pass" if valid else "fail", placement_validation, "placement validator reports valid"))
    else:
        checks.append(_check("cpu_affinity_and_memory_policy", "skipped", None, "placement validation artifact available"))

    sse_witness = _load_optional_json(_find_first(run_dir, ("sse_witness.json", "**/sse_witness.json")))
    if sse_witness:
        valid = bool(sse_witness.get("valid", sse_witness.get("accepted", False)))
        checks.append(_check("one_sse_event_per_token", "pass" if valid else "fail", sse_witness, "witness validator reports valid"))
    else:
        checks.append(_check("one_sse_event_per_token", "skipped", None, "SSE witness artifact available"))

    accepted = not any(check["status"] == "fail" for check in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "workload": str(workload),
        "arm": str(arm),
        "run_dir": str(run_dir.resolve()),
        "accepted": accepted,
        "metrics": metrics,
        "requests": aiperf,
        "frontend": frontend,
        "per_node_cpu": per_node_cpu,
        "measurement_availability": {
            "per_node_cpu": {
                "available": bool(per_node_cpu),
                "reason": None
                if per_node_cpu
                else (
                    "no NUMA node CPU map supplied"
                    if not node_cpu_map
                    else "no raw mpstat or sampler per-CPU samples found"
                ),
            },
            "pidstat": {"available": bool(pidstat.get("sample_count", 0))},
            "telemetry": {"available": bool(telemetry.get("samples"))},
        },
        "remote_pools": pools,
        "network": nics,
        "acceptance": {"accepted": accepted, "checks": checks},
        "inputs": {
            "aiperf": str(aiperf_file),
            "telemetry": [str(path) for path in telemetry_paths],
            "pidstat": str(pidstat_file) if pidstat_file else None,
            "mpstat": str(mpstat_file) if mpstat_file else None,
            "manifest": str(manifest_file) if manifest_file else None,
        },
    }


def _numeric_metrics(summaries: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for summary in summaries:
        metrics = summary.get("metrics", {})
        if isinstance(metrics, Mapping):
            result.update(key for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool))
    return result


def calculate_matrix_metrics(run_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate repeats and calculate endpoint-normalized placement effects.

    Only accepted runs contribute.  For every numeric metric:

    * endpoint mean = mean(``8+0``, ``0+8``)
    * node asymmetry = (``8+0`` - ``0+8``) / endpoint mean
    * spanning penalty = (``4+4`` - endpoint mean) / endpoint mean

    The sign is intentionally not inverted for latency: positive always means
    the measured 4+4 value is larger than the endpoint baseline.
    """

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for summary in run_summaries:
        if not summary.get("accepted", False):
            continue
        grouped[str(summary.get("workload", "unknown"))][str(summary.get("arm", "unknown"))].append(summary)

    result: dict[str, Any] = {}
    for workload, arms in grouped.items():
        workload_summaries = [summary for summaries in arms.values() for summary in summaries]
        metric_names = sorted(_numeric_metrics(workload_summaries))
        arm_output: dict[str, Any] = {}
        arm_means: dict[str, dict[str, float]] = {}
        for arm, summaries in sorted(arms.items()):
            metric_output: dict[str, Any] = {}
            means: dict[str, float] = {}
            for metric in metric_names:
                values = [float(summary["metrics"][metric]) for summary in summaries if isinstance(summary.get("metrics", {}).get(metric), (int, float))]
                if not values:
                    continue
                mean = statistics.fmean(values)
                means[metric] = mean
                metric_output[metric] = {
                    "mean": mean,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                }
            arm_means[arm] = means
            arm_output[arm] = {"repeat_count": len(summaries), "metrics": metric_output}

        endpoint_mean: dict[str, float] = {}
        node_asymmetry: dict[str, float] = {}
        spanning_penalty: dict[str, float] = {}
        for metric in metric_names:
            node0 = arm_means.get("8+0", {}).get(metric)
            node1 = arm_means.get("0+8", {}).get(metric)
            split = arm_means.get("4+4", {}).get(metric)
            if node0 is None or node1 is None:
                continue
            baseline = (node0 + node1) / 2.0
            endpoint_mean[metric] = baseline
            if baseline != 0:
                node_asymmetry[metric] = (node0 - node1) / baseline * 100.0
                if split is not None:
                    spanning_penalty[metric] = (split - baseline) / baseline * 100.0
        result[workload] = {
            "arms": arm_output,
            "endpoint_mean": endpoint_mean,
            "node_asymmetry_pct": node_asymmetry,
            "spanning_penalty_pct": spanning_penalty,
        }
    return result


def _write_or_print(document: Mapping[str, Any], output: str | None) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output in (None, "-"):
        sys.stdout.write(encoded)
    else:
        Path(output).write_text(encoded)


def command_run(args: argparse.Namespace) -> int:
    node_cpus: dict[int, str] = {}
    for assignment in args.node_cpus:
        node, cpus = assignment.split("=", 1)
        node_cpus[int(node)] = cpus
    summary = analyze_run(
        args.run_dir,
        arm=args.arm,
        workload=args.workload,
        aiperf_path=args.aiperf,
        telemetry_paths=args.telemetry,
        pidstat_path=args.pidstat,
        mpstat_path=args.mpstat,
        manifest_path=args.manifest,
        expected_output_tokens=args.expected_output_tokens,
        expected_total_output_tokens=args.expected_total_output_tokens,
        node_cpus=node_cpus or None,
        remote_roles=args.remote_role or DEFAULT_REMOTE_ROLES,
        max_remote_busy_percent=args.max_remote_busy,
        max_nic_utilization_percent=args.max_nic_utilization,
        max_retransmits=args.max_retransmits,
        max_retransmit_fraction=args.max_retransmit_fraction,
    )
    _write_or_print(summary, args.output)
    return 0 if summary["accepted"] else 2


def command_matrix(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir)
    paths: list[Path] = []
    for pattern in args.pattern:
        paths.extend(job_dir.glob(pattern))
    unique_paths = sorted(set(path.resolve() for path in paths))
    summaries = [json.loads(path.read_text()) for path in unique_paths]
    matrix_manifest = _load_optional_json(job_dir / "matrix_manifest.json")
    sequence = matrix_manifest.get("sequence", [])
    expected_run_count = (
        sum(
            1
            for item in sequence
            if isinstance(item, Mapping) and not bool(item.get("calibration"))
        )
        if isinstance(sequence, list) and sequence
        else None
    )
    complete = bool(summaries) and (
        expected_run_count is None or len(summaries) == expected_run_count
    )
    accepted_run_count = sum(bool(summary.get("accepted")) for summary in summaries)
    accepted = complete and accepted_run_count == len(summaries)
    matrix = {
        "schema_version": SCHEMA_VERSION,
        "job_dir": str(job_dir.resolve()),
        "run_count": len(summaries),
        "expected_run_count": expected_run_count,
        "complete": complete,
        "accepted": accepted,
        "accepted_run_count": accepted_run_count,
        "workloads": calculate_matrix_metrics(summaries),
        "inputs": [str(path) for path in unique_paths],
    }
    _write_or_print(matrix, args.output)
    return 0 if accepted else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="analyze and validate one measured run")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--arm", choices=("8+0", "4+4", "0+8"))
    run.add_argument("--workload")
    run.add_argument("--aiperf")
    run.add_argument("--telemetry", action="append", default=[])
    run.add_argument("--pidstat")
    run.add_argument("--mpstat")
    run.add_argument("--manifest")
    run.add_argument("--expected-output-tokens", type=int)
    run.add_argument("--expected-total-output-tokens", type=int)
    run.add_argument(
        "--node-cpus",
        action="append",
        default=[],
        metavar="NODE=CPULIST",
        help="NUMA CPU membership used to summarize raw mpstat samples",
    )
    run.add_argument("--remote-role", action="append")
    run.add_argument("--max-remote-busy", type=float, default=85.0)
    run.add_argument("--max-nic-utilization", type=float, default=70.0)
    run.add_argument("--max-retransmits", type=int)
    run.add_argument("--max-retransmit-fraction", type=float, default=0.001)
    run.add_argument("--output", help="summary JSON path; default stdout")
    run.set_defaults(func=command_run)

    matrix = subparsers.add_parser("matrix", help="aggregate accepted run summaries")
    matrix.add_argument("--job-dir", required=True)
    matrix.add_argument("--pattern", action="append", default=["runs/**/summary.json"])
    matrix.add_argument("--output", help="matrix JSON path; default stdout")
    matrix.set_defaults(func=command_matrix)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
