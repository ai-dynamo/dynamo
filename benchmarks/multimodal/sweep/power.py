# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional


POWER_SUMMARY_FILENAME = "power_summary.json"
GPU_TELEMETRY_FILENAME = "gpu_telemetry_export.jsonl"


@dataclass(frozen=True)
class GpuPowerSample:
    gpu_index: str
    power_watts: float
    timestamp_s: Optional[float] = None
    utilization_gpu_pct: Optional[float] = None
    memory_used_mib: Optional[float] = None


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _normalize_key(key: Any) -> str:
    normalized = re.sub(r"\[[^\]]*\]", "", str(key).strip().lower())
    normalized = normalized.replace("%", " pct ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _normalized_map(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {_normalize_key(k): v for k, v in record.items()}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered or lowered in {"n/a", "na", "nan", "none", "null"}:
            return None
        match = _NUMBER_RE.search(lowered)
        if not match:
            return None
        number = float(match.group(0))
        return number if math.isfinite(number) else None
    return None


def _timestamp_to_seconds(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if _NUMBER_RE.fullmatch(text):
        return _to_float(text)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _first_value(record: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _sample_from_mapping(record: Mapping[str, Any]) -> Optional[GpuPowerSample]:
    normalized = _normalized_map(record)

    power_watts = _to_float(
        _first_value(
            normalized,
            (
                "power_watts",
                "power_w",
                "power_draw_watts",
                "power_draw_w",
                "power_draw",
                "power_usage_watts",
                "power_usage",
                "power",
            ),
        )
    )
    if power_watts is None:
        return None

    gpu_index = _first_value(
        normalized,
        (
            "gpu_index",
            "gpu_id",
            "device_index",
            "device_id",
            "index",
            "gpu",
            "uuid",
            "pci_bus_id",
        ),
    )
    if gpu_index is None:
        gpu_index = "0"

    timestamp_s = _timestamp_to_seconds(
        _first_value(
            normalized,
            (
                "timestamp_s",
                "timestamp_sec",
                "timestamp_seconds",
                "timestamp",
                "time_s",
                "time",
                "ts",
            ),
        )
    )
    utilization_gpu_pct = _to_float(
        _first_value(
            normalized,
            (
                "utilization_gpu_pct",
                "utilization_gpu",
                "gpu_utilization_pct",
                "gpu_utilization",
                "util_gpu_pct",
                "util_gpu",
                "utilization",
            ),
        )
    )
    memory_used_mib = _to_float(
        _first_value(
            normalized,
            (
                "memory_used_mib",
                "memory_used",
                "used_memory_mib",
                "used_memory",
            ),
        )
    )

    return GpuPowerSample(
        gpu_index=str(gpu_index),
        power_watts=power_watts,
        timestamp_s=timestamp_s,
        utilization_gpu_pct=utilization_gpu_pct,
        memory_used_mib=memory_used_mib,
    )


def _child_records(record: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    normalized = _normalized_map(record)
    parent_timestamp = _first_value(
        normalized,
        (
            "timestamp_s",
            "timestamp_sec",
            "timestamp_seconds",
            "timestamp",
            "time_s",
            "time",
            "ts",
        ),
    )

    for raw_key, raw_value in record.items():
        key = _normalize_key(raw_key)
        if key not in {
            "gpus",
            "gpu",
            "gpu_metrics",
            "gpu_telemetry",
            "devices",
            "samples",
            "metrics",
        }:
            continue

        if isinstance(raw_value, list):
            for child in raw_value:
                if isinstance(child, Mapping):
                    merged = dict(child)
                    if parent_timestamp is not None:
                        merged.setdefault("timestamp", parent_timestamp)
                    yield merged
        elif isinstance(raw_value, Mapping):
            for child_key, child in raw_value.items():
                if isinstance(child, Mapping):
                    merged = dict(child)
                    if parent_timestamp is not None:
                        merged.setdefault("timestamp", parent_timestamp)
                    merged.setdefault("gpu_index", child_key)
                    yield merged


def _samples_from_record(record: Any) -> Iterator[GpuPowerSample]:
    if isinstance(record, list):
        for item in record:
            yield from _samples_from_record(item)
        return

    if not isinstance(record, Mapping):
        return

    sample = _sample_from_mapping(record)
    if sample is not None:
        yield sample

    for child in _child_records(record):
        child_sample = _sample_from_mapping(child)
        if child_sample is not None:
            yield child_sample


def load_power_samples(telemetry_path: Path) -> List[GpuPowerSample]:
    samples: List[GpuPowerSample] = []
    with telemetry_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {telemetry_path} at line {line_no}: {exc}"
                ) from exc
            samples.extend(_samples_from_record(record))

    return sorted(
        samples,
        key=lambda s: (
            s.gpu_index,
            float("inf") if s.timestamp_s is None else s.timestamp_s,
        ),
    )


def _average(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _peak(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return max(present)


def _round_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 6)


def _summarize_gpu(samples: List[GpuPowerSample]) -> Dict[str, Any]:
    with_timestamps = [s for s in samples if s.timestamp_s is not None]
    energy_j = 0.0
    duration_s = 0.0

    if len(with_timestamps) >= 2:
        with_timestamps.sort(key=lambda s: s.timestamp_s or 0.0)
        start_s = with_timestamps[0].timestamp_s or 0.0
        end_s = with_timestamps[-1].timestamp_s or start_s
        duration_s = max(0.0, end_s - start_s)
        for left, right in zip(with_timestamps, with_timestamps[1:]):
            left_ts = left.timestamp_s or 0.0
            right_ts = right.timestamp_s or left_ts
            delta_s = right_ts - left_ts
            if delta_s > 0:
                energy_j += ((left.power_watts + right.power_watts) / 2.0) * delta_s

    arithmetic_avg_power_w = sum(s.power_watts for s in samples) / len(samples)
    average_power_w = (
        energy_j / duration_s if duration_s > 0 else arithmetic_avg_power_w
    )

    return {
        "gpu_index": samples[0].gpu_index,
        "sample_count": len(samples),
        "duration_s": _round_float(duration_s),
        "energy_j": _round_float(energy_j),
        "energy_wh": _round_float(energy_j / 3600.0),
        "average_power_w": _round_float(average_power_w),
        "min_power_w": _round_float(min(s.power_watts for s in samples)),
        "peak_power_w": _round_float(max(s.power_watts for s in samples)),
        "average_utilization_gpu_pct": _round_float(
            _average(s.utilization_gpu_pct for s in samples)
        ),
        "peak_utilization_gpu_pct": _round_float(
            _peak(s.utilization_gpu_pct for s in samples)
        ),
        "average_memory_used_mib": _round_float(
            _average(s.memory_used_mib for s in samples)
        ),
        "peak_memory_used_mib": _round_float(_peak(s.memory_used_mib for s in samples)),
    }


def summarize_power_telemetry(telemetry_path: Path) -> Dict[str, Any]:
    samples = load_power_samples(telemetry_path)
    by_gpu: Dict[str, List[GpuPowerSample]] = {}
    for sample in samples:
        by_gpu.setdefault(sample.gpu_index, []).append(sample)

    gpus = [_summarize_gpu(by_gpu[gpu_index]) for gpu_index in sorted(by_gpu)]
    total_energy_j = sum(gpu["energy_j"] for gpu in gpus)

    timestamped = [s.timestamp_s for s in samples if s.timestamp_s is not None]
    duration_s = max(timestamped) - min(timestamped) if len(timestamped) >= 2 else 0.0
    if duration_s > 0:
        average_power_w = total_energy_j / duration_s
    else:
        average_power_w = sum(gpu["average_power_w"] for gpu in gpus)

    return {
        "schema_version": 1,
        "telemetry_file": telemetry_path.name,
        "sample_count": len(samples),
        "gpu_count": len(gpus),
        "duration_s": _round_float(duration_s),
        "total_energy_j": _round_float(total_energy_j),
        "total_energy_wh": _round_float(total_energy_j / 3600.0),
        "average_power_w": _round_float(average_power_w),
        "sum_peak_power_w": _round_float(sum(gpu["peak_power_w"] for gpu in gpus)),
        "gpus": gpus,
    }


def write_power_summary(
    artifact_dir: Path,
    telemetry_filename: str = GPU_TELEMETRY_FILENAME,
    output_filename: str = POWER_SUMMARY_FILENAME,
) -> Optional[Path]:
    telemetry_path = artifact_dir / telemetry_filename
    if not telemetry_path.is_file():
        return None

    summary = summarize_power_telemetry(telemetry_path)
    if summary["sample_count"] == 0:
        return None

    output_path = artifact_dir / output_filename
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return output_path
