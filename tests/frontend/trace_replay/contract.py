# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned corpus contract and independent, lossless JSON comparisons."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

SCHEMA = "dynamo-frontend-replay/v1"
SOURCE_KINDS = frozenset(
    {"benchmark", "production", "custom_report", "agent_report", "authored"}
)
ASSET_NAMES = frozenset(
    {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}
)
MISSING = object()


class ReplayError(ValueError):
    """The corpus or replay mechanism cannot establish a valid observation."""


def _reject_constant(value: str) -> None:
    raise ReplayError(f"non-finite JSON number: {value}")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def loads(text: str | bytes) -> Any:
    """Keep decimal numbers exact; do not equate booleans with integers."""
    try:
        return json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_object,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReplayError(f"invalid JSON: {exc}") from exc


def dumps(value: Any, *, canonical: bool = False) -> str:
    """Serialize Decimal as a JSON number, never a quoted or rounded value."""
    if (
        canonical
        and not isinstance(value, bool)
        and isinstance(value, (int, float, Decimal))
    ):
        number = Decimal(str(value))
        if not number.is_finite():
            raise ReplayError("non-finite number")
        sign, digits, exponent = number.as_tuple()
        if not any(digits):
            return "0"
        while digits[-1] == 0:
            digits, exponent = digits[:-1], exponent + 1
        # Avoid Decimal.normalize(), which rounds to the current decimal context.
        coefficient = ("-" if sign else "") + "".join(str(item) for item in digits)
        return coefficient + "e" + str(exponent)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ReplayError("non-finite decimal")
        return str(value)
    if isinstance(value, dict):
        keys = sorted(value) if canonical else value.keys()
        if any(not isinstance(key, str) for key in value):
            raise ReplayError("JSON object key must be a string")
        return (
            "{"
            + ",".join(
                json.dumps(key, ensure_ascii=False)
                + ":"
                + dumps(value[key], canonical=canonical)
                for key in keys
            )
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(dumps(item, canonical=canonical) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value, canonical=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Publish an observation atomically; private corpora never need broad modes."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".replay-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(dumps(value) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def numeric(value: Any) -> bool:
    return type(value) in (int, float, Decimal)


def same_json(actual: Any, expected: Any) -> bool:
    if actual is MISSING or expected is MISSING:
        return actual is expected
    if numeric(actual) and numeric(expected):
        if isinstance(actual, float) and not math.isfinite(actual):
            return False
        if isinstance(expected, float) and not math.isfinite(expected):
            return False
        return Decimal(str(actual)) == Decimal(str(expected))
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same_json(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_json(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def pointer(value: Any, path: str) -> Any:
    if path == "":
        return value
    if not isinstance(path, str) or not path.startswith("/"):
        raise ReplayError(f"invalid JSON Pointer: {path!r}")
    current = value
    for encoded in path[1:].split("/"):
        if re.search(r"~(?![01])", encoded):
            raise ReplayError(f"invalid JSON Pointer escape: {path}")
        key = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current.get(key, MISSING)
        elif isinstance(current, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
            index = int(key)
            current = current[index] if index < len(current) else MISSING
        else:
            return MISSING
        if current is MISSING:
            return MISSING
    return current


def validate_source(source: Any) -> None:
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("kind"), str)
        or source["kind"] not in SOURCE_KINDS
    ):
        raise ReplayError("source.kind must name one of the five supported origins")
    if set(source) - {"kind", "context"}:
        raise ReplayError("source accepts only kind and optional context")
    if "context" in source:
        context = source["context"]
        if not isinstance(context, dict) or set(context) - {"ref", "pr", "issue"}:
            raise ReplayError(
                "source.context must be an object with ref/pr/issue strings"
            )
        if any(not isinstance(value, str) for value in context.values()):
            raise ReplayError("source.context values must be strings")


def _token_ids(value: Any, name: str) -> None:
    if not isinstance(value, list) or any(
        type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in value
    ):
        raise ReplayError(f"{name} must be an array of unsigned 32-bit token IDs")


@dataclass(frozen=True)
class Case:
    identifier: str
    source: dict[str, Any]
    request: dict[str, Any]
    worker_events: tuple[dict[str, Any], ...]
    expected_preprocessed: dict[str, Any]
    expected_response: dict[str, Any] | None
    reference: dict[str, Any]


def parse_case(value: Any) -> Case:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("id"), str)
        or not value["id"]
    ):
        raise ReplayError("case.id must be a nonempty string")
    validate_source(value.get("source"))
    if not isinstance(value.get("request"), dict):
        raise ReplayError("case.request must be an object")
    replay = value.get("replay")
    events = replay.get("worker_events") if isinstance(replay, dict) else None
    if not isinstance(events, list) or not events:
        raise ReplayError("replay.worker_events must be a nonempty array")
    terminals = 0
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ReplayError("worker event must be an object")
        _token_ids(event.get("token_ids"), "worker event token_ids")
        if event.get("index", 0) != 0 or type(event.get("index", 0)) is not int:
            raise ReplayError("the replay MVP supports choice index 0 only")
        if event.get("tokens") is not None or event.get("text") is not None:
            raise ReplayError(
                "token replay leaves tokens/text unset for frontend decoding"
            )
        finish = event.get("finish_reason")
        if finish is not None:
            if finish not in ("stop", "length") or index != len(events) - 1:
                raise ReplayError("worker stop/length must occur on the last event")
            terminals += 1
    if terminals != 1:
        raise ReplayError("worker events require exactly one terminal")
    expect = value.get("expect")
    preprocessed = expect.get("preprocessed") if isinstance(expect, dict) else None
    if not isinstance(preprocessed, dict):
        raise ReplayError("expect.preprocessed must be an object")
    _token_ids(preprocessed.get("token_ids"), "expected prompt token_ids")
    fields = preprocessed.get("fields_equal", {})
    if not isinstance(fields, dict):
        raise ReplayError("fields_equal must map JSON Pointers to expected values")
    if "/token_ids" in fields:
        raise ReplayError("put the prompt expectation in preprocessed.token_ids only")
    for path in fields:
        pointer({}, path)
    response = expect.get("response")
    if response is not None:
        if not isinstance(response, dict) or not isinstance(
            response.get("message"), dict
        ):
            raise ReplayError("expect.response requires a message object")
        if response.get("http_status") != 200 or response.get("finish_reason") not in (
            "stop",
            "tool_calls",
            "length",
        ):
            raise ReplayError(
                "response expectation requires HTTP 200 and a supported finish"
            )
        if not isinstance(response.get("usage"), dict):
            raise ReplayError("response expectation requires usage")
    reference = value.get("reference", {})
    if not isinstance(reference, dict):
        raise ReplayError("reference must be an object")
    return Case(
        value["id"],
        value["source"],
        value["request"],
        tuple(events),
        preprocessed,
        response,
        reference,
    )


@dataclass(frozen=True)
class Corpus:
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    cases: tuple[Case, ...]
    total_cases: int

    @property
    def profile(self) -> dict[str, Any]:
        return self.manifest["profile"]


def manifest_path(path: Path) -> Path:
    return (path / "manifest.json" if path.is_dir() else path).resolve()


def _relative(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ReplayError("manifest file paths must be relative")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ReplayError("manifest file path escapes corpus directory")
    return result


def load_corpus(path: Path, limit: int | None = None) -> Corpus:
    if limit is not None and limit <= 0:
        raise ReplayError("limit must be positive; omit it to run every case")
    path = manifest_path(path)
    manifest = loads(path.read_bytes())
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ReplayError(f"manifest schema must be {SCHEMA}")
    profile = manifest.get("profile")
    if (
        not isinstance(profile, dict)
        or not isinstance(profile.get("name"), str)
        or not isinstance(profile.get("model_name"), str)
    ):
        raise ReplayError("profile requires name and model_name strings")
    model = profile.get("model")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("repo_id"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", "")))
    ):
        raise ReplayError(
            "profile.model requires repo_id and immutable 40-character revision"
        )
    assets = model.get("assets")
    config = profile.get("runtime_config")
    required_assets = ASSET_NAMES
    if isinstance(config, dict) and config.get("tool_call_parser") == "deepseek_v41":
        required_assets = ASSET_NAMES - {"generation_config.json"}
    if (
        not isinstance(assets, dict)
        or not required_assets <= set(assets) <= ASSET_NAMES
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(checksum))
            for checksum in assets.values()
        )
    ):
        raise ReplayError(
            "DeepSeek replay requires hashed native tokenizer/config assets"
        )
    if (
        not isinstance(config, dict)
        or config.get("tool_call_parser") not in ("deepseek_v4", "deepseek_v41")
        or config.get("reasoning_parser") != config.get("tool_call_parser")
    ):
        raise ReplayError(
            "this replay requires a matching DeepSeek V4 or V4.1 parser pair"
        )
    if type(config.get("context_length")) is not int or config["context_length"] <= 0:
        raise ReplayError("runtime_config.context_length must be positive")
    entries = manifest.get("traces")
    if not isinstance(entries, list) or not entries:
        raise ReplayError("manifest.traces must be nonempty")
    cases: list[Case] = []
    seen: set[str] = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ReplayError("trace entry must be an object")
        trace = _relative(path.parent, entry.get("path"))
        if file_digest(trace) != entry.get("sha256"):
            raise ReplayError(f"trace hash mismatch: {trace.name}")
        opener = gzip.open if trace.suffix == ".gz" else open
        count = 0
        with opener(trace, "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                case = parse_case(loads(line))
                if case.identifier in seen:
                    raise ReplayError(f"duplicate case ID: {case.identifier}")
                seen.add(case.identifier)
                count += 1
                total += 1
                if limit is None or len(cases) < limit:
                    cases.append(case)
        if type(entry.get("count")) is not int or count != entry["count"]:
            raise ReplayError(f"trace count mismatch: {trace.name}")
    if not cases:
        raise ReplayError("corpus selected no cases")
    return Corpus(path, manifest, file_digest(path), tuple(cases), total)


def verify_assets(corpus: Corpus, model_dir: Path) -> dict[str, str]:
    hashes = {}
    for name, expected in corpus.profile["model"]["assets"].items():
        path = model_dir / name
        if not path.is_file() or file_digest(path) != expected:
            raise ReplayError(f"missing or mismatched model asset: {name}")
        hashes[name] = expected
    return hashes


def scheduled_events(case: Case, schedule: str) -> list[dict[str, Any]]:
    if schedule == "recorded":
        return [dict(event) for event in case.worker_events]
    widths = {"whole": None, "one_token": 1, "fixed32": 32}
    if schedule not in widths:
        raise ReplayError(f"unsupported schedule: {schedule}")
    for event in case.worker_events:
        if event.get("log_probs") is not None or event.get("top_logprobs") is not None:
            raise ReplayError("logprob-bearing frames require the recorded schedule")
    ids = [token for event in case.worker_events for token in event["token_ids"]]
    width = widths[schedule] or max(1, len(ids))
    groups = [ids[start : start + width] for start in range(0, len(ids), width)] or [[]]
    result = []
    for index, group in enumerate(groups):
        final = index == len(groups) - 1
        frame = dict(case.worker_events[-1]) if final else {"index": 0}
        frame.update(token_ids=group, tokens=None, text=None)
        if not final:
            frame["finish_reason"] = None
        result.append(frame)
    return result


def mismatch(path: str, actual: Any, expected: Any) -> dict[str, Any]:
    """Bound diagnostics while retaining exact observations in their own files."""

    def preview(value: Any) -> Any:
        if value is MISSING:
            return {"missing": True}
        if isinstance(value, list) and len(value) > 20:
            return {"length": len(value), "first": value[:10], "sha256": digest(value)}
        return value

    return {"path": path, "actual": preview(actual), "expected": preview(expected)}


def judge_preprocessed(case: Case, actual: dict[str, Any]) -> list[dict[str, Any]]:
    expected = case.expected_preprocessed
    pairs = {"/token_ids": expected["token_ids"], **expected.get("fields_equal", {})}
    return [
        mismatch(path, pointer(actual, path), value)
        for path, value in pairs.items()
        if not same_json(pointer(actual, path), value)
    ]


def canonical_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ReplayError("response message must be an object")
    if "reasoning" in message:
        raise ReplayError("expected the configured reasoning_content wire field")
    if message.get("role") != "assistant" or "content" not in message:
        raise ReplayError("response requires assistant role and content")
    content = message["content"]
    reasoning = message.get("reasoning_content")
    if content is not None and not isinstance(content, str):
        raise ReplayError("response content must be string or null")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ReplayError("response reasoning_content must be string or null")
    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise ReplayError("tool_calls must be an array")
    normalized = []
    seen: set[str] = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ReplayError("tool call must have type function")
        identifier = call.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ReplayError("tool-call IDs must be nonempty and unique")
        seen.add(identifier)
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ReplayError("tool call requires a function name")
        arguments = function.get("arguments", MISSING)
        if not isinstance(arguments, str):
            raise ReplayError("wire tool arguments must be a JSON string")
        normalized.append(
            {
                "type": "function",
                "function": {"name": function["name"], "arguments": loads(arguments)},
            }
        )
    # The two optional OpenAI fields may be absent or null. Content is deliberately
    # required, and its null/empty-string distinction is preserved.
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": normalized,
    }


def expected_message(message: dict[str, Any]) -> dict[str, Any]:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = loads(arguments)
        calls.append(
            {
                "type": "function",
                "function": {"name": function["name"], "arguments": arguments},
            }
        )
    return {
        "role": message.get("role", "assistant"),
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": calls,
    }


def judge_response(case: Case, response: dict[str, Any]) -> list[dict[str, Any]]:
    expected = case.expected_response
    if expected is None:
        return []
    failures = []
    for key in ("http_status", "finish_reason"):
        if not same_json(response.get(key, MISSING), expected[key]):
            failures.append(
                mismatch("/" + key, response.get(key, MISSING), expected[key])
            )
    wanted = expected_message(expected["message"])
    if not same_json(response.get("message", MISSING), wanted):
        failures.append(mismatch("/message", response.get("message", MISSING), wanted))
    for key, value in expected["usage"].items():
        actual = pointer(response, "/usage/" + key)
        if not same_json(actual, value):
            failures.append(mismatch("/usage/" + key, actual, value))
    return failures
