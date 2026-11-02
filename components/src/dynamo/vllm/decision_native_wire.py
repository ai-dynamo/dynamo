# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate native request wires and account for bounded tokenization and usage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .decision_native_errors import NativeExecutionError, NativeValidationError
from .decision_native_loader import NativeApi
from .decision_native_registry import PROFILES

MAX_NATIVE_QUESTIONS = 64
MAX_NATIVE_CANDIDATES = 255
MAX_NATIVE_INPUT_TOKENS = 1_048_576
MAX_NATIVE_TOKENIZATION_WORK = 1_048_576
MAX_NATIVE_ENCODING_BYTES = 16 * 1024 * 1024


def _positive_integer(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise NativeValidationError(f"invalid {name}")
    return value


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as cause:
        raise NativeValidationError(
            "native decision input must contain valid JSON"
        ) from cause


def _wire_json(raw: dict, profile: str) -> str:
    selected = PROFILES.get(profile)
    if selected is None:
        raise NativeExecutionError("unsupported native wire profile")
    try:
        return json.dumps(
            {f"decision_{selected.kind}": raw},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as cause:
        raise NativeValidationError(
            "native decision input must contain valid JSON"
        ) from cause


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return _json(value)
    raise NativeValidationError("question text must be a string, object, or array")


def _object(value: Any, keys: set[str], required: set[str]) -> dict:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - keys
    ):
        raise NativeExecutionError("invalid internal native decision request fields")
    return value


@dataclass(frozen=True)
class NativeRequest:
    profile: str
    state: Any
    questions: tuple[Any, ...]
    cache_salt: str
    max_input_tokens: int
    max_tokenization_work: int
    max_encoding_bytes: int


def _candidate_options(raw: dict, native: Any) -> tuple[Any, ...]:
    candidates = raw["candidates"]
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= MAX_NATIVE_CANDIDATES
    ):
        raise NativeValidationError("invalid native decision candidate count")
    typed = any(
        isinstance(c, dict) and type(c.get("value")) is bool for c in candidates
    )
    options, seen = [], set()
    for index, candidate in enumerate(candidates):
        item = _object(candidate, {"value", "description", "label"}, {"value"})
        value = item["value"]
        if type(value) not in (str, bool) or (type(value), value) in seen:
            raise NativeValidationError(
                "native decision candidates must have unique string/bool values"
            )
        seen.add((type(value), value))
        label = item.get("label")
        if label is not None and not isinstance(label, str):
            raise NativeValidationError("invalid native decision score label")
        description = (
            _text(item.get("description"))
            if item.get("description") is not None
            else None
        )
        name = _json(value) if typed else value
        if raw["kind"] == "Score":
            if value != str(index):
                raise NativeExecutionError("score values must be canonical ordinals")
            name = label if label is not None else description or ""
            if label is not None and description:
                name = f"{label} - {description}"
            description = None
        options.append(native.questions.Option(name, description))
    if raw["kind"] == "Predicate" and [c["value"] for c in candidates] != [True, False]:
        raise NativeExecutionError("predicate candidates must be true then false")
    if raw["kind"] == "Score" and not 2 <= len(options) <= 10:
        raise NativeValidationError("native decision scores require two to ten levels")
    return tuple(options)


def _question(raw: Any, ordinal: int, native: Any, max_options: int) -> Any:
    item = _object(
        raw,
        {"ordinal", "id", "kind", "instructions", "candidates"},
        {"ordinal", "kind", "candidates"},
    )
    if type(item["ordinal"]) is not int or item["ordinal"] != ordinal:
        raise NativeExecutionError(
            "native decision question ordinals must be contiguous and ordered"
        )
    if item.get("id") is not None and not isinstance(item["id"], str):
        raise NativeExecutionError("invalid native decision question ID")
    kinds = {"Predicate": "noul", "Choice": "choice", "Score": "score"}
    if not isinstance(item["kind"], str) or item["kind"] not in kinds:
        raise NativeExecutionError("unsupported native decision question kind")
    options = _candidate_options(item, native)
    if len(options) > max_options:
        raise NativeValidationError("question exceeds the native decision option limit")
    return native.questions.Question(
        id=f"question-{ordinal}",
        type=native.questions.get_question_type(kinds[item["kind"]]),
        instructions=_text(item.get("instructions")),
        options=options,
        labels=(),
    )


def parse_native_request(
    raw: Any, profile: str, native: NativeApi, limits: Any
) -> NativeRequest:
    fields = {
        "protocol_version",
        "profile",
        "model",
        "input",
        "questions",
        "temperature",
        "cache_salt",
        "max_input_tokens",
        "max_tokenization_work",
        "max_encoding_bytes",
    }
    raw = _object(raw, fields, fields)
    if type(raw["protocol_version"]) is not int or raw["protocol_version"] != 1:
        raise NativeExecutionError("unsupported native decision protocol version")
    if raw["profile"] != profile:
        raise NativeExecutionError(
            "native decision profile differs from the selected worker"
        )
    if not isinstance(raw["model"], str) or not raw["model"]:
        raise NativeExecutionError("native decision model must be a nonempty string")
    if type(raw["temperature"]) not in (int, float) or raw["temperature"] != 1:
        raise NativeExecutionError("native decision profiles require temperature one")
    if not isinstance(raw["cache_salt"], str) or not 1 <= len(raw["cache_salt"]) <= 256:
        raise NativeExecutionError(
            "native decision cache salt must be bounded and nonempty"
        )
    budgets = tuple(
        _positive_integer(raw[name], name, maximum)
        for name, maximum in (
            ("max_input_tokens", MAX_NATIVE_INPUT_TOKENS),
            ("max_tokenization_work", MAX_NATIVE_TOKENIZATION_WORK),
            ("max_encoding_bytes", MAX_NATIVE_ENCODING_BYTES),
        )
    )
    try:
        if len(_wire_json(raw, profile).encode("utf-8")) > budgets[2]:
            raise NativeValidationError("native decision encoding byte budget exceeded")
    except UnicodeEncodeError as cause:
        raise NativeValidationError(
            "native decision input must contain valid UTF-8"
        ) from cause
    questions = raw["questions"]
    if not isinstance(questions, list) or not 1 <= len(questions) <= min(
        limits.max_questions, MAX_NATIVE_QUESTIONS
    ):
        raise NativeValidationError("invalid native decision question count")
    parsed = tuple(
        _question(q, i, native, limits.max_options) for i, q in enumerate(questions)
    )
    return NativeRequest(profile, raw["input"], parsed, raw["cache_salt"], *budgets)


class BudgetTokenizer:
    """Track actual tokenization work and preserve the original tokenizer API."""

    def __init__(self, tokenizer: Any, token_budget: int, byte_budget: int):
        self.tokenizer = tokenizer
        self.tokens, self.bytes = token_budget, byte_budget

    def consume_bytes(self, text: str) -> None:
        self.bytes -= len(text.encode("utf-8"))
        if self.bytes < 0:
            raise NativeValidationError("native encoding byte budget exceeded")

    def consume_tokens(self, count: int) -> None:
        self.tokens -= count
        if self.tokens < 0:
            raise NativeValidationError("native tokenization work budget exceeded")

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        self.consume_bytes(text)
        result = self.tokenizer.encode(text, **kwargs)
        self.consume_tokens(len(result))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


@dataclass(frozen=True)
class NativeUsage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int


def measured_cache_tokens(cached: Any, input_tokens: int) -> int:
    """Validate counters; zero's provenance relies on the pinned scheduler.

    The upstream output processor also defaults absent prefill statistics to
    zero. Integer validation alone cannot distinguish that from a measured zero.
    """
    if type(cached) is not int or not 0 <= cached <= input_tokens:
        raise NativeExecutionError("native executor returned invalid cache accounting")
    return cached


def aggregate_usage(rows: list[NativeUsage]):
    """Combine the measured execution counters validated at the worker boundary."""
    return {
        "input_tokens": sum(row.input_tokens for row in rows),
        "output_tokens": sum(row.output_tokens for row in rows),
        "cached_tokens": sum(row.cached_tokens for row in rows),
    }
