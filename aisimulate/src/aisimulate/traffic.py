# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-neutral materialization of configured replay traffic."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import JsonValue

_INLINE_REQUEST_FIELDS = {
    "id",
    "arrival_time_ms",
    "input_tokens",
    "input_token_ids",
    "output_tokens",
    "session_id",
    "metadata",
}
_TRACE_CONFIG_FIELDS = {
    "trace",
    "trace_path",
    "format",
    "trace_block_size",
    "speedup",
}
_MOONCAKE_ROW_FIELDS = {
    "request_id",
    "session_id",
    "input_length",
    "input_tokens",
    "output_length",
    "output_tokens",
    "output_token_ids",
    "hash_ids",
    "timestamp",
    "created_time",
    "delay",
    "delay_ms",
    "priority",
    "strict_priority",
    "policy_class",
}


def materialize_configured_traffic(
    config: Mapping[str, JsonValue],
) -> list[dict[str, JsonValue]]:
    """Materialize inline requests or one static, single-turn Mooncake trace.

    Trace paths are opened only when this function runs. The resulting requests
    are fully serializable and no longer depend on the source file or Dynamo.
    """

    if not isinstance(config, Mapping):
        raise ValueError("configured traffic must be a mapping")

    if "requests" in config:
        unexpected = set(config) - {"requests"}
        if unexpected:
            raise ValueError(
                "inline traffic has unsupported field(s): "
                + ", ".join(sorted(unexpected))
            )
        return _materialize_inline_requests(config.get("requests"))

    unexpected = set(config) - _TRACE_CONFIG_FIELDS
    if unexpected:
        raise ValueError(
            "trace traffic has unsupported field(s): " + ", ".join(sorted(unexpected))
        )

    selectors = [
        selector
        for selector in ("trace", "trace_path")
        if config.get(selector) is not None
    ]
    if len(selectors) != 1:
        raise ValueError(
            "trace traffic requires exactly one of 'trace' or 'trace_path'"
        )
    path = _trace_path(config[selectors[0]], selectors[0])

    trace_format = config.get("format", "mooncake")
    if trace_format != "mooncake":
        raise ValueError(
            "engine trace traffic supports only format='mooncake'; "
            f"got {trace_format!r}"
        )

    trace_block_size = _positive_int(
        config.get("trace_block_size", 512), "trace_block_size"
    )
    speedup = _positive_number(config.get("speedup", 1.0), "speedup")
    return _materialize_mooncake(path, trace_block_size, speedup)


def _materialize_inline_requests(
    raw_requests: JsonValue,
) -> list[dict[str, JsonValue]]:
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("inline traffic 'requests' must be a non-empty list")

    materialized: list[dict[str, JsonValue]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_requests):
        label = f"traffic requests[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a mapping")
        unexpected = set(raw) - _INLINE_REQUEST_FIELDS
        if unexpected:
            raise ValueError(
                f"{label} has unexpected field(s): " + ", ".join(sorted(unexpected))
            )

        request_id = raw.get("id", f"request-{index}")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"{label}.id must be a non-empty string")
        if request_id in seen_ids:
            raise ValueError(f"duplicate traffic request id {request_id!r}")
        seen_ids.add(request_id)

        if "arrival_time_ms" not in raw:
            raise ValueError(f"{label}.arrival_time_ms is required")
        arrival_time_ms = _nonnegative_number(
            raw["arrival_time_ms"], f"{label}.arrival_time_ms"
        )

        has_length = "input_tokens" in raw
        has_ids = "input_token_ids" in raw
        if has_length == has_ids:
            raise ValueError(
                f"{label} requires exactly one of input_tokens or input_token_ids"
            )

        request: dict[str, JsonValue] = {
            "id": request_id,
            "arrival_time_ms": arrival_time_ms,
            "output_tokens": _positive_int(
                raw.get("output_tokens"), f"{label}.output_tokens"
            ),
            "metadata": raw.get("metadata"),
        }
        if has_ids:
            token_ids = raw["input_token_ids"]
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError(f"{label}.input_token_ids must be a non-empty list")
            if any(not _is_uint(token, 32) for token in token_ids):
                raise ValueError(
                    f"{label}.input_token_ids must contain unsigned 32-bit integers"
                )
            request["input_tokens"] = len(token_ids)
            request["input_token_ids"] = token_ids
        else:
            request["input_tokens"] = _positive_int(
                raw["input_tokens"], f"{label}.input_tokens"
            )

        session_id = raw.get("session_id")
        if session_id is not None:
            if not isinstance(session_id, str):
                raise ValueError(f"{label}.session_id must be a string")
            request["session_id"] = session_id
        materialized.append(request)
    return materialized


def _materialize_mooncake(
    path: Path,
    trace_block_size: int,
    speedup: float,
) -> list[dict[str, JsonValue]]:
    rows = _read_mooncake_rows(path)

    explicit_ids: set[str] = set()
    for line_number, row in rows:
        request_id = row.get("request_id")
        if request_id is None:
            continue
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(
                f"Mooncake trace {path} line {line_number}: "
                "request_id must be a non-empty string"
            )
        if request_id in explicit_ids:
            raise ValueError(
                f"Mooncake trace {path} line {line_number}: "
                f"duplicate request_id {request_id!r}"
            )
        explicit_ids.add(request_id)

    interned_hash_ids: dict[int, int] = {}
    used_ids = set(explicit_ids)
    seen_sessions: set[str] = set()
    requests: list[dict[str, JsonValue]] = []
    for line_number, row in rows:
        label = f"Mooncake trace {path} line {line_number}"
        unexpected = set(row) - _MOONCAKE_ROW_FIELDS
        if unexpected:
            raise ValueError(
                f"{label}: unsupported field(s): " + ", ".join(sorted(unexpected))
            )
        if "output_token_ids" in row:
            raise ValueError(
                f"{label}: authored output_token_ids are not supported by "
                "static engine replay"
            )
        raw_delay = _aliased_value(row, "delay", "delay_ms", label)
        if (
            raw_delay is not None
            and _nonnegative_number(raw_delay, f"{label}: delay") != 0.0
        ):
            raise ValueError(
                f"{label}: a non-zero delay requires session follow-up replay, "
                "which is not supported"
            )

        session_id = row.get("session_id")
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f"{label}: session_id must be a non-empty string")
            if session_id in seen_sessions:
                raise ValueError(
                    f"{label}: multiple rows for session {session_id!r} require "
                    "multi-turn replay, which is not supported"
                )
            seen_sessions.add(session_id)

        request_id = row.get("request_id")
        if request_id is None:
            request_id = _generated_request_id(line_number, used_ids)
            used_ids.add(request_id)

        raw_hash_ids = row.get("hash_ids")
        if not isinstance(raw_hash_ids, list) or not raw_hash_ids:
            raise ValueError(f"{label}: hash_ids must be a non-empty list")
        if any(not _is_uint(hash_id, 64) for hash_id in raw_hash_ids):
            raise ValueError(f"{label}: hash_ids must contain unsigned 64-bit integers")
        canonical_hash_ids = [
            _intern_hash_id(hash_id, interned_hash_ids) for hash_id in raw_hash_ids
        ]

        capacity = len(canonical_hash_ids) * trace_block_size
        raw_input_length = _aliased_value(row, "input_length", "input_tokens", label)
        input_length = (
            capacity
            if raw_input_length is None
            else min(
                _nonnegative_int(raw_input_length, f"{label}: input_length"), capacity
            )
        )
        if input_length == 0:
            raise ValueError(
                f"{label}: input_length must remain positive after clamping "
                f"to synthesized capacity {capacity}"
            )

        output_length = _aliased_value(row, "output_length", "output_tokens", label)
        if output_length is None:
            raise ValueError(f"{label}: output_length is required")

        timestamp = _aliased_value(row, "timestamp", "created_time", label)

        metadata: dict[str, JsonValue] = {
            "priority": _signed_int(row.get("priority", 0), 32, f"{label}: priority"),
            "strict_priority": _unsigned_int(
                row.get("strict_priority", 0), 32, f"{label}: strict_priority"
            ),
        }
        policy_class = row.get("policy_class")
        if policy_class is not None:
            if not isinstance(policy_class, str):
                raise ValueError(f"{label}: policy_class must be a string")
            metadata["policy_class"] = policy_class

        request: dict[str, JsonValue] = {
            "id": request_id,
            # Match the native static-trace bridge: an omitted first-arrival
            # timestamp is a burst arrival at zero.
            "arrival_time_ms": (
                0.0
                if timestamp is None
                else _finite_number(timestamp, f"{label}: timestamp")
            ),
            "input_tokens": input_length,
            "input_token_ids": _synthesize_tokens(
                input_length, canonical_hash_ids, trace_block_size
            ),
            "output_tokens": _positive_int(output_length, f"{label}: output_length"),
            "metadata": metadata,
        }
        if session_id is not None:
            request["session_id"] = session_id
        requests.append(request)

    first_arrival = min(float(request["arrival_time_ms"]) for request in requests)
    requests.sort(key=lambda request: float(request["arrival_time_ms"]))
    for request in requests:
        request["arrival_time_ms"] = (
            float(request["arrival_time_ms"]) - first_arrival
        ) / speedup
    return requests


def _read_mooncake_rows(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        source = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"failed to open Mooncake trace {path}: {exc}") from exc

    rows: list[tuple[int, Mapping[str, Any]]] = []
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"failed to parse Mooncake trace {path} line "
                    f"{line_number} as JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"Mooncake trace {path} line {line_number}: "
                    "each row must be a JSON object"
                )
            rows.append((line_number, row))
    if not rows:
        raise ValueError(f"Mooncake trace {path} did not contain any requests")
    return rows


def _trace_path(value: JsonValue, name: str) -> Path:
    if isinstance(value, str) and value:
        return Path(value)
    if isinstance(value, Mapping):
        unexpected = set(value) - {"path"}
        if unexpected:
            raise ValueError(
                f"{name} has unsupported field(s): " + ", ".join(sorted(unexpected))
            )
        path = value.get("path")
        if isinstance(path, str) and path:
            return Path(path)
        raise ValueError(f"{name}.path must be a non-empty path string")
    raise ValueError(f"{name} must be a non-empty path string or {{path: ...}}")


def _aliased_value(
    row: Mapping[str, Any],
    canonical: str,
    alias: str,
    label: str,
) -> Any:
    if canonical in row and alias in row:
        raise ValueError(
            f"{label}: {canonical} and its alias {alias} cannot both be set"
        )
    if canonical in row:
        return row[canonical]
    return row.get(alias)


def _intern_hash_id(hash_id: int, interned: dict[int, int]) -> int:
    canonical = interned.get(hash_id)
    if canonical is not None:
        return canonical
    canonical = len(interned)
    if canonical > 0xFFFF_FFFF:
        raise ValueError("Mooncake trace contains more hash identities than u32 allows")
    interned[hash_id] = canonical
    return canonical


def _synthesize_tokens(
    input_length: int,
    canonical_hash_ids: list[int],
    trace_block_size: int,
) -> list[int]:
    tokens: list[int] = []
    for hash_id in canonical_hash_ids:
        remaining = input_length - len(tokens)
        tokens.extend([hash_id] * min(remaining, trace_block_size))
        if len(tokens) == input_length:
            break
    return tokens


def _generated_request_id(line_number: int, used_ids: set[str]) -> str:
    base = f"trace-{line_number}"
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _is_uint(value: Any, bits: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= (1 << bits) - 1
    )


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _unsigned_int(value: Any, bits: int, name: str) -> int:
    if not _is_uint(value, bits):
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return value


def _signed_int(value: Any, bits: int, name: str) -> int:
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not lower <= value <= upper
    ):
        raise ValueError(f"{name} must be a signed {bits}-bit integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _nonnegative_number(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_number(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number
