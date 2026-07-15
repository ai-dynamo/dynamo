# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict input schema for explicit vLLM self-benchmark points."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

BENCHMARK_POINTS_SCHEMA_VERSION = 1

BenchmarkMode = Literal["prefill", "decode", "agg"]


class PrefillBenchmarkPoint(TypedDict):
    total_prefill_tokens: int
    total_kv_read_tokens: int
    batch_size: int


class DecodeBenchmarkPoint(TypedDict):
    total_kv_read_tokens: int
    batch_size: int


class ExplicitBenchmarkPoints(TypedDict):
    schema_version: int
    prefill: list[PrefillBenchmarkPoint]
    decode: list[DecodeBenchmarkPoint]


_TOP_LEVEL_FIELDS = {"schema_version", "prefill", "decode"}
_POINT_FIELDS = {
    "prefill": ("total_prefill_tokens", "total_kv_read_tokens", "batch_size"),
    "decode": ("total_kv_read_tokens", "batch_size"),
}


def load_benchmark_points_file(path: str, mode: str) -> ExplicitBenchmarkPoints:
    """Read and normalize one explicit-points file in the parent process."""

    try:
        contents = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        detail = error.strerror or str(error)
        raise ValueError(
            f"--benchmark-points-file {path!r} could not be read: {detail}"
        ) from error
    except UnicodeError as error:
        raise ValueError(
            f"--benchmark-points-file {path!r} is not valid UTF-8: {error}"
        ) from error

    try:
        payload = json.loads(contents)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"--benchmark-points-file {path!r} contains invalid JSON at "
            f"line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    try:
        return normalize_benchmark_points(payload, mode)
    except ValueError as error:
        raise ValueError(f"--benchmark-points-file {path!r}: {error}") from error


def normalize_benchmark_points(payload: object, mode: str) -> ExplicitBenchmarkPoints:
    """Validate and return a canonical, rank-independent pure-point manifest."""

    if not isinstance(mode, str) or mode not in {"prefill", "decode", "agg"}:
        raise ValueError("benchmark mode must be one of prefill, decode, or agg")
    if not isinstance(payload, dict):
        raise ValueError("top level must be an object")

    unknown_fields = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    if unknown_fields:
        raise ValueError(
            "top level contains unsupported field(s): " + ", ".join(unknown_fields)
        )

    missing_fields = sorted(_TOP_LEVEL_FIELDS - set(payload))
    if missing_fields:
        raise ValueError(
            "top level is missing required field(s): " + ", ".join(missing_fields)
        )

    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("schema_version must be the integer 1")
    if version != BENCHMARK_POINTS_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be 1, got {version}")

    normalized: ExplicitBenchmarkPoints = {
        "schema_version": BENCHMARK_POINTS_SCHEMA_VERSION,
        "prefill": _normalize_phase(payload["prefill"], "prefill"),
        "decode": _normalize_phase(payload["decode"], "decode"),
    }

    if mode in {"prefill", "agg"} and not normalized["prefill"]:
        raise ValueError(
            f"prefill must contain at least one point for benchmark mode {mode!r}"
        )
    if mode in {"decode", "agg"} and not normalized["decode"]:
        raise ValueError(
            f"decode must contain at least one point for benchmark mode {mode!r}"
        )

    return normalized


def benchmark_points_digest(points: ExplicitBenchmarkPoints) -> str:
    payload = json.dumps(
        points,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_phase(value: object, phase: Literal["prefill", "decode"]):
    if not isinstance(value, list):
        raise ValueError(f"{phase} must be an array")

    normalized: list[dict[str, int]] = []
    expected_fields = set(_POINT_FIELDS[phase])
    for index, raw_point in enumerate(value):
        path = f"{phase}[{index}]"
        if not isinstance(raw_point, dict):
            raise ValueError(f"{path} must be an object")

        unknown_fields = sorted(set(raw_point) - expected_fields)
        if unknown_fields:
            raise ValueError(
                f"{path} contains unsupported field(s): " + ", ".join(unknown_fields)
            )
        missing_fields = sorted(expected_fields - set(raw_point))
        if missing_fields:
            raise ValueError(
                f"{path} is missing required field(s): " + ", ".join(missing_fields)
            )

        point = {
            field: _require_integer(raw_point[field], f"{path}.{field}")
            for field in _POINT_FIELDS[phase]
        }
        batch_size = point["batch_size"]
        if batch_size < 1:
            raise ValueError(f"{path}.batch_size must be positive")

        total_kv_read_tokens = point["total_kv_read_tokens"]
        if phase == "prefill":
            total_prefill_tokens = point["total_prefill_tokens"]
            if total_prefill_tokens < 1:
                raise ValueError(f"{path}.total_prefill_tokens must be positive")
            if total_prefill_tokens < batch_size:
                raise ValueError(
                    f"{path}.total_prefill_tokens must be at least batch_size"
                )
            if total_kv_read_tokens < 0:
                raise ValueError(f"{path}.total_kv_read_tokens must be non-negative")
            if 0 < total_kv_read_tokens < batch_size:
                raise ValueError(
                    f"{path}.total_kv_read_tokens must be zero or at least batch_size"
                )
        elif total_kv_read_tokens < batch_size:
            raise ValueError(f"{path}.total_kv_read_tokens must be at least batch_size")

        normalized.append(point)

    if phase == "prefill":
        return cast(list[PrefillBenchmarkPoint], normalized)
    return cast(list[DecodeBenchmarkPoint], normalized)


def _require_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value
