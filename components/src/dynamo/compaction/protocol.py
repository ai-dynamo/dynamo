# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, bounded source snapshot protocol for the optional local helper."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

MAX_INPUT_BYTES = 1_048_576
MAX_OUTPUT_BYTES = 65_536
PROTECTED_ROLES = frozenset({"user", "system", "developer"})


class Invalid(ValueError):
    """Stable rejection code, without reflecting potentially sensitive evidence."""


def require(valid: bool, reason: str) -> None:
    if not valid:
        raise Invalid(reason)


def canonical(value: Any) -> str:
    try:
        result = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        result.encode("utf-8")
        return result
    except (TypeError, ValueError, UnicodeError) as error:
        raise Invalid("invalid_json_value") from error


def strict_json(raw: str | bytes) -> Any:
    def pairs(items):
        result = dict(items)
        require(len(result) == len(items), "duplicate_json_key")
        return result

    def constant(_):
        raise Invalid("nonfinite_json")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise Invalid("invalid_json") from error


def bounded_int(value: Any, maximum: int, reason: str) -> int:
    require(type(value) is int and 0 < value <= maximum, reason)
    return value


@dataclass(frozen=True)
class Unit:
    id: str
    role: str
    text: str
    protected: bool

    @property
    def must_retain(self) -> bool:
        return self.protected or self.role in PROTECTED_ROLES

    @classmethod
    def parse(cls, value: Any) -> "Unit":
        require(
            type(value) is dict and set(value) == {"id", "role", "text", "protected"},
            "invalid_unit_schema",
        )
        require(
            type(value["id"]) is str
            and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value["id"]) is not None,
            "invalid_unit_id",
        )
        require(
            type(value["role"]) is str
            and value["role"] in {"user", "assistant", "tool", "system", "developer"},
            "invalid_role",
        )
        require(type(value["text"]) is str and bool(value["text"]), "invalid_text")
        require(type(value["protected"]) is bool, "invalid_protection")
        return cls(**value)


@dataclass(frozen=True)
class Budget:
    max_output_tokens: int
    max_input_bytes: int = MAX_INPUT_BYTES
    max_output_bytes: int = MAX_OUTPUT_BYTES
    max_units: int = 256
    max_summary_bytes: int = 32_768
    max_model_calls: int = 3
    deadline_seconds: float = 30

    @classmethod
    def parse(cls, value: Any) -> "Budget":
        require(type(value) is dict and "max_output_tokens" in value, "invalid_budget")
        require(not set(value) - set(cls.__dataclass_fields__), "unknown_budget_field")
        result = cls(**value)
        for key, maximum in (
            ("max_output_tokens", 65_536),
            ("max_input_bytes", MAX_INPUT_BYTES),
            ("max_output_bytes", MAX_OUTPUT_BYTES),
            ("max_units", 256),
            ("max_summary_bytes", 32_768),
            ("max_model_calls", 3),
        ):
            bounded_int(getattr(result, key), maximum, "invalid_budget")
        require(
            type(result.deadline_seconds) in (int, float)
            and math.isfinite(result.deadline_seconds)
            and 0 < result.deadline_seconds <= 30,
            "invalid_deadline",
        )
        return result


@dataclass(frozen=True)
class Request:
    snapshot_sha256: str
    units: tuple[Unit, ...]
    context_units: tuple[Unit, ...]
    session_id: str
    budget: Budget

    @classmethod
    def parse(cls, value: Any) -> "Request":
        require(
            type(value) is dict
            and set(value)
            == {
                "version",
                "snapshot_sha256",
                "units",
                "context_units",
                "session_id",
                "budget",
            },
            "invalid_request_schema",
        )
        require(
            type(value["version"]) is int and value["version"] == 1,
            "unsupported_version",
        )
        budget = Budget.parse(value["budget"])
        require(
            len(canonical(value).encode()) <= budget.max_input_bytes,
            "input_bytes_exceeded",
        )
        raw_units = value["units"]
        raw_context = value["context_units"]
        require(
            type(raw_units) is list
            and type(raw_context) is list
            and 0 < len(raw_units)
            and len(raw_units) + len(raw_context) <= budget.max_units,
            "unit_count_exceeded",
        )
        units = tuple(Unit.parse(item) for item in raw_units)
        context = tuple(Unit.parse(item) for item in raw_context)
        require(
            len({unit.id for unit in units + context}) == len(units) + len(context),
            "duplicate_source_id",
        )
        require(
            type(value["session_id"]) is str
            and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value["session_id"])
            is not None,
            "invalid_session_id",
        )
        digest = hashlib.sha256(
            canonical(
                {
                    "session_id": value["session_id"],
                    "units": raw_units,
                    "context_units": raw_context,
                }
            ).encode()
        ).hexdigest()
        require(
            type(value["snapshot_sha256"]) is str
            and value["snapshot_sha256"] == digest,
            "snapshot_mismatch",
        )
        return cls(digest, units, context, value["session_id"], budget)


@dataclass(frozen=True)
class Result:
    status: str
    snapshot_sha256: str | None
    summary: str | None
    retained_ids: tuple[str, ...]
    reason: str
    mode: str
    model_calls: int | None = 0
    counted_output: int | None = None
    counter_name: str | None = None
    qualified_counter: bool = False
    version: int = 1

    def wire(self) -> dict:
        return asdict(self)
