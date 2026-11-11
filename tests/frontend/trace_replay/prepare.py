# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare offline DeepSeek replay corpora with a pinned, independent encoder.

Supported controls map to the IR through CONTROL_POINTERS. Verified native EOS
IDs additionally supply /eos_token_ids and /stop_conditions/stop_token_ids_hidden.
This deliberately accepts a closed text/function-call subset; unsupported input
must fail rather than silently disappear from the independent prompt oracle.
The optional tokenizers==0.21.4 dependency is needed only during preparation.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from . import contract
from .contract import (
    SCHEMA,
    ReplayError,
    dumps,
    file_digest,
    load_corpus,
    loads,
    numeric,
    parse_case,
    same_json,
    validate_profile,
    write_json,
)

CONTROL_POINTERS = {
    "temperature": "/sampling_options/temperature",
    "top_p": "/sampling_options/top_p",
    "seed": "/sampling_options/seed",
    "n": "/sampling_options/n",
    "max_tokens": "/stop_conditions/max_tokens",
    "max_completion_tokens": "/stop_conditions/max_tokens",
}
SELECTORS = frozenset({"stream", "return_token_ids"})
REQUEST_KEYS = (
    frozenset(CONTROL_POINTERS)
    | SELECTORS
    | {"model", "messages", "tools", "tool_choice", "chat_template_kwargs"}
)
CAPTURE_SCHEMA = "dynamo-frontend-replay-capture/v1"
RECORD_SCHEMA = "dynamo-frontend-replay-capture-record/v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ids(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int or not 0 <= item <= 0xFFFFFFFF for item in value)
    ):
        raise ReplayError(f"{label} requires nonempty unsigned 32-bit token IDs")
    return value


def _local(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ReplayError("evidence paths must be relative")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ReplayError("evidence path escapes its directory")
    return result


def load_profile(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = loads(path.read_bytes())
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ReplayError(f"profile document schema must be {SCHEMA}")
    profile = validate_profile(document.get("profile"))
    reference = document.get("reference")
    if reference is None:
        provenance = document.get("provenance", {})
        review = (
            provenance.get("golden_review", {})
            if isinstance(provenance, dict)
            else None
        )
        if not isinstance(review, dict) or not isinstance(review.get("encoder"), dict):
            raise ReplayError("profile requires a pinned reference descriptor")
        reference = {
            **review.get("encoder", {}),
            "tokenizers_version": review.get("tokenizers_version"),
        }
    if not isinstance(reference, dict):
        raise ReplayError("profile requires a pinned reference descriptor")
    model = profile["model"]
    if any(reference.get(key) != model[key] for key in ("repo_id", "revision")):
        raise ReplayError("reference repository/revision must match the profile")
    if (
        not isinstance(reference.get("path"), str)
        or Path(reference["path"]).is_absolute()
        or ".." in Path(reference["path"]).parts
        or not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256", "")))
        or reference.get("tokenizers_version") != "0.21.4"
    ):
        raise ReplayError(
            "reference requires relative path, SHA-256 and tokenizers 0.21.4"
        )
    if (
        reference.get("tokenizer_sha256", model["assets"]["tokenizer.json"])
        != model["assets"]["tokenizer.json"]
    ):
        raise ReplayError("reference tokenizer hash must match the profile")
    return profile, reference


def read_seeds(data: bytes, profile: dict[str, Any]) -> list[dict[str, Any]]:
    return contract.read_request_seeds(data, profile["model_name"])


def _closed(value: Any, keys: set[str] | frozenset[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) - keys:
        raise ReplayError(f"unsupported {label} fields")


def rendering_input(
    request: dict[str, Any], *, captured: bool
) -> tuple[list[dict[str, Any]], str, Any, list[str]]:
    _closed(request, REQUEST_KEYS, "request")
    if (
        request.get("stream", False) is not False
        or request.get("return_token_ids", True) is not True
    ):
        raise ReplayError(
            "capture selectors must be stream:false and return_token_ids:true"
        )
    if "max_tokens" in request and "max_completion_tokens" in request:
        raise ReplayError("conflicting token caps")
    for key in ("max_tokens", "max_completion_tokens"):
        if key in request and (
            type(request[key]) is not int or not 0 < request[key] <= 0xFFFFFFFF
        ):
            raise ReplayError("token cap must be a positive uint32")
    if "n" in request and (type(request["n"]) is not int or request["n"] != 1):
        raise ReplayError("only n=1 is supported")
    if "seed" in request and (
        type(request["seed"]) is not int
        or not 0 <= request["seed"] <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ReplayError("seed must be uint64")
    for key, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1)):
        if key in request and (
            not numeric(request[key])
            or not lower <= request[key] <= upper
            or (key == "top_p" and request[key] == 0)
        ):
            raise ReplayError("invalid sampling control: " + key)
    if "tool_choice" in request and request["tool_choice"] != "auto":
        raise ReplayError("only tool_choice:auto is supported")
    kwargs = request.get("chat_template_kwargs")
    _closed(
        kwargs,
        {"thinking", "thinking_mode", "reasoning_effort"},
        "chat_template_kwargs",
    )
    if "thinking" not in kwargs and "thinking_mode" not in kwargs:
        raise ReplayError("an explicit thinking mode is required")
    mode = kwargs.get("thinking_mode")
    if "thinking_mode" in kwargs and mode not in ("chat", "thinking"):
        raise ReplayError("thinking_mode must be chat or thinking")
    if "thinking" in kwargs:
        if type(kwargs["thinking"]) is not bool:
            raise ReplayError("thinking must be boolean")
        alias = "thinking" if kwargs["thinking"] else "chat"
        if mode is not None and mode != alias:
            raise ReplayError("thinking aliases disagree")
        mode = alias
    effort = kwargs.get("reasoning_effort")
    if effort is not None and not (
        (type(effort) is int and 1 <= effort <= 100)
        or (isinstance(effort, str) and effort in {"low", "high", "max"})
    ):
        raise ReplayError("unsupported reasoning_effort")
    original = request.get("messages")
    if not isinstance(original, list) or not original:
        raise ReplayError("messages must be nonempty")
    messages = copy.deepcopy(original)
    transformations: list[str] = []
    for message in messages:
        _closed(
            message,
            {"role", "content", "reasoning_content", "tool_calls", "tool_call_id"},
            "message",
        )
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ReplayError("unsupported message role")
        if message.get("content") is not None and not isinstance(
            message["content"], str
        ):
            raise ReplayError("only string/null message content is supported")
        reasoning = message.get("reasoning_content")
        if (
            isinstance(reasoning, list)
            and captured
            and all(isinstance(part, str) for part in reasoning)
        ):
            message["reasoning_content"] = "\n".join(part for part in reasoning if part)
            transformations.append(
                "captured reasoning segments: nonempty strings joined with newline; verified against captured prompt IDs"
            )
        elif reasoning is not None and not isinstance(reasoning, str):
            raise ReplayError(
                "array reasoning requires matching captured prompt evidence"
            )
        if "reasoning_content" in message and role != "assistant":
            raise ReplayError(
                "reasoning_content is supported only for assistant messages"
            )
        if "tool_call_id" in message and (
            role != "tool" or not isinstance(message["tool_call_id"], str)
        ):
            raise ReplayError("tool_call_id requires a tool message")
        if role == "tool" and not isinstance(message.get("tool_call_id"), str):
            raise ReplayError("tool messages require tool_call_id")
        if "tool_calls" in message:
            if role != "assistant" or not isinstance(message["tool_calls"], list):
                raise ReplayError("tool_calls requires an assistant array")
            for call in message["tool_calls"]:
                _closed(call, {"id", "type", "function"}, "tool call")
                if call.get("type") != "function" or not isinstance(
                    call.get("id"), str
                ):
                    raise ReplayError("tool call requires function type and string id")
                function = call.get("function")
                _closed(function, {"name", "arguments"}, "tool call function")
                if (
                    not isinstance(function.get("name"), str)
                    or not isinstance(function.get("arguments"), str)
                    or not isinstance(loads(function["arguments"]), dict)
                ):
                    raise ReplayError(
                        "function call requires name and JSON-object argument string"
                    )
    tools = request.get("tools", [])
    if not isinstance(tools, list):
        raise ReplayError("tools must be an array")
    for tool in tools:
        _closed(tool, {"type", "function"}, "tool")
        if tool.get("type") != "function":
            raise ReplayError("only function tools are supported")
        function = tool.get("function")
        _closed(
            function,
            {"name", "description", "parameters", "strict"},
            "function definition",
        )
        if (
            not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(function.get("parameters"), dict)
        ):
            raise ReplayError("function definition requires name and parameters object")
        if "description" in function and not isinstance(function["description"], str):
            raise ReplayError("function description must be string")
        if "strict" in function and type(function["strict"]) is not bool:
            raise ReplayError("function strict must be boolean")
    if tools:
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = copy.deepcopy(tools)
    # The official Python renderer uses ordinary JSON number handling. Preserve
    # lossless originals separately; never round them when binding expectations.
    return json.loads(dumps(messages)), str(mode), effort, transformations


class ReferenceEncoder:
    """Execute only hash-verified vendor bytes and a verified tokenizer string."""

    def __init__(
        self,
        profile: dict[str, Any],
        reference: dict[str, Any],
        model_dir: Path,
        encoder: Path | None = None,
    ) -> None:
        self.profile = profile
        self.reference = reference
        self.assets: dict[str, bytes] = {}
        for name, expected in profile["model"]["assets"].items():
            data = (model_dir / name).read_bytes()
            if _sha(data) != expected:
                raise ReplayError("model asset hash mismatch: " + name)
            self.assets[name] = data
        code = (encoder or _local(model_dir, reference["path"])).read_bytes()
        if _sha(code) != reference["sha256"]:
            raise ReplayError("reference encoder hash mismatch")
        tree = ast.parse(code)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ReplayError(
                        "reference encoder relative imports are unsupported"
                    )
                modules = [node.module or ""]
            if any(
                module.split(".")[0] not in sys.stdlib_module_names
                for module in modules
            ):
                raise ReplayError(
                    "reference encoder requires a non-stdlib/sibling import"
                )
        self.tokenizer = self._tokenizer(
            self.assets["tokenizer.json"], reference["tokenizers_version"]
        )
        self.namespace: dict[str, Any] = {"__name__": "_pinned_replay_reference"}
        exec(compile(code, reference["path"], "exec"), self.namespace)
        if not all(
            callable(self.namespace.get(name))
            for name in ("encode_messages", "render_message")
        ):
            raise ReplayError("reference encoder lacks encode_messages/render_message")
        self.eos_ids = self._eos()

    @staticmethod
    def _tokenizer(data: bytes, version: str) -> Any:
        try:
            actual = importlib.metadata.version("tokenizers")
            if actual != version:
                raise ReplayError(
                    f"preparation requires tokenizers=={version}; found {actual}"
                )
            module = importlib.import_module("tokenizers")
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise ReplayError(
                f"preparation requires optional tokenizers=={version}"
            ) from exc
        return module.Tokenizer.from_str(data.decode("utf-8"))

    def _eos(self) -> list[int]:
        candidates: list[list[int]] = []
        for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
            if name not in self.assets:
                continue
            config = loads(self.assets[name])
            if not isinstance(config, dict):
                raise ReplayError("native model configuration must be an object")
            if "eos_token_id" in config:
                value = config["eos_token_id"]
                ids = _ids(value if isinstance(value, list) else [value], name + " EOS")
                candidates.append(sorted(set(ids)))
            if "eos_token" in config:
                token = config["eos_token"]
                token = token.get("content") if isinstance(token, dict) else token
                if (
                    not isinstance(token, str)
                    or self.tokenizer.token_to_id(token) is None
                ):
                    raise ReplayError(
                        "native EOS token is absent from the verified tokenizer"
                    )
                candidates.append([self.tokenizer.token_to_id(token)])
        if not candidates or any(
            candidate != candidates[0] for candidate in candidates
        ):
            raise ReplayError("missing or conflicting native EOS identities")
        return candidates[0]

    def prompt(
        self, request: dict[str, Any], *, captured: bool
    ) -> tuple[list[int], list[str]]:
        messages, mode, effort, transformations = rendering_input(
            request, captured=captured
        )
        try:
            rendered = self.namespace["encode_messages"](
                messages, mode, reasoning_effort=effort
            )
            ids = self.tokenizer.encode(rendered, add_special_tokens=False).ids
        except (
            AssertionError,
            TypeError,
            ValueError,
            KeyError,
            NotImplementedError,
        ) as exc:
            raise ReplayError(
                "vendor encoder rejected the supported request: " + str(exc)
            ) from exc
        return _ids(ids, "reference prompt"), transformations

    def synthetic(self, request: dict[str, Any]) -> list[int]:
        messages, mode, effort, _ = rendering_input(request, captured=False)
        messages.append({"role": "assistant", "content": "OK", "reasoning_content": ""})
        rendered = self.namespace["render_message"](
            len(messages) - 1, messages, mode, reasoning_effort=effort
        )
        ids = _ids(
            self.tokenizer.encode(rendered, add_special_tokens=False).ids,
            "synthetic output",
        )
        if ids[-1] not in self.eos_ids or any(
            token in self.eos_ids for token in ids[:-1]
        ):
            raise ReplayError("synthetic completion must end with one native EOS")
        cap = request.get("max_tokens", request.get("max_completion_tokens"))
        if cap is not None and len(ids) > cap:
            raise ReplayError("synthetic completion exceeds the explicit token cap")
        return ids


def normalize_terminal(
    request: dict[str, Any], response: dict[str, Any], eos_ids: list[int]
) -> tuple[list[int], list[int], str]:
    prompt = _ids(response.get("prompt_token_ids"), "captured prompt")
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or type(choices[0].get("index", 0)) is not int
        or choices[0].get("index", 0) != 0
    ):
        raise ReplayError("capture requires exactly choice index 0")
    choice = choices[0]
    output = _ids(choice.get("token_ids"), "captured output")
    if choice.get("stop_reason") is not None:
        raise ReplayError("nonnull captured stop_reason is unsupported")
    if any(token in eos_ids for token in output[:-1]):
        raise ReplayError("EOS appears before the captured output ends")
    cap = request.get("max_tokens", request.get("max_completion_tokens"))
    if cap is not None and len(output) > cap:
        raise ReplayError("captured output exceeds explicit token cap")
    finish = choice.get("finish_reason")
    if finish in ("stop", "tool_calls") and output[-1] in eos_ids:
        terminal = "stop"
    elif (
        finish in ("length", "tool_calls")
        and cap is not None
        and len(output) == cap
        and output[-1] not in eos_ids
    ):
        terminal = "length"
    else:
        raise ReplayError(
            "capture lacks unambiguous EOS or token-cap terminal evidence"
        )
    expected_usage = {
        "prompt_tokens": len(prompt),
        "completion_tokens": len(output),
        "total_tokens": len(prompt) + len(output),
    }
    usage = response.get("usage")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] != value
        for key, value in expected_usage.items()
    ):
        raise ReplayError("captured usage disagrees with prompt/output token arrays")
    return prompt, output, terminal


def _capture_records(
    root: Path, profile: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    path = root / "capture.json"
    document = loads(path.read_bytes())
    if (
        not isinstance(document, dict)
        or document.get("schema") != CAPTURE_SCHEMA
        or document.get("complete") is not True
    ):
        raise ReplayError("prepare requires a complete capture manifest")
    if not same_json(document.get("profile"), profile):
        raise ReplayError("capture profile does not match preparation profile")
    seed_bytes = _local(root, document.get("requests_path")).read_bytes()
    if _sha(seed_bytes) != document.get("requests_sha256"):
        raise ReplayError("archived seed hash mismatch")
    seeds = read_seeds(seed_bytes, profile)
    entries = document.get("records")
    if (
        not isinstance(entries, list)
        or len(entries) != len(seeds)
        or type(document.get("count")) is not int
        or document["count"] != len(seeds)
        or type(document.get("successful_count")) is not int
        or document["successful_count"] != len(seeds)
    ):
        raise ReplayError("capture seed/record/success counts must match exactly")
    records: list[dict[str, Any]] = []
    for seed, entry in zip(seeds, entries):
        if (
            not isinstance(entry, dict)
            or entry.get("id") != seed["id"]
            or entry.get("status") != "captured"
        ):
            raise ReplayError(
                "capture records must match seed IDs and order without errors"
            )
        data = _local(root, entry.get("path")).read_bytes()
        if _sha(data) != entry.get("sha256"):
            raise ReplayError("capture record hash mismatch")
        record = loads(data)
        if (
            not isinstance(record, dict)
            or record.get("schema") != RECORD_SCHEMA
            or record.get("id") != seed["id"]
            or record.get("error") is not None
            or record.get("http_status") != 200
        ):
            raise ReplayError("invalid successful capture record")
        if (
            not same_json(record.get("original_request"), seed["request"])
            or not same_json(record.get("source"), seed["source"])
            or not same_json(record.get("reference", {}), seed.get("reference", {}))
        ):
            raise ReplayError("capture record disagrees with archived seed")
        wire = record.get("wire_request_bytes")
        if not isinstance(wire, str) or _sha(wire.encode("utf-8")) != record.get(
            "request_sha256"
        ):
            raise ReplayError("captured request bytes hash mismatch")
        expected_wire = {**seed["request"], "stream": False, "return_token_ids": True}
        if not same_json(loads(wire), expected_wire):
            raise ReplayError("captured wire request disagrees with archived seed")
        try:
            response_bytes = base64.b64decode(
                record["response_bytes_base64"], validate=True
            )
            parsed = loads(response_bytes.decode("utf-8"))
        except (
            KeyError,
            TypeError,
            ValueError,
            binascii.Error,
            UnicodeDecodeError,
        ) as exc:
            raise ReplayError("invalid captured response bytes") from exc
        if (
            _sha(response_bytes) != record.get("response_sha256")
            or not same_json(parsed, record.get("response"))
            or not isinstance(parsed, dict)
        ):
            raise ReplayError("captured response bytes/hash/object mismatch")
        records.append(record)
    return seeds, records, document, file_digest(path)


def _effective(request: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in request.items() if key not in SELECTORS}


def _reviewed(
    path: Path, profile: dict[str, Any], reference: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    corpus = load_corpus(path)
    if (
        not same_json(corpus.profile["model"], profile["model"])
        or corpus.profile["model_name"] != profile["model_name"]
        or not same_json(corpus.profile["runtime_config"], profile["runtime_config"])
    ):
        raise ReplayError("reviewed corpus model/assets do not match")
    provenance = corpus.manifest.get("provenance", {})
    review = (
        provenance.get("golden_review", {}) if isinstance(provenance, dict) else None
    )
    if not isinstance(review, dict):
        raise ReplayError("reviewed corpus requires a golden_review object")
    ids = review.get("ids")
    if (
        not isinstance(ids, list)
        or any(not isinstance(item, str) for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ReplayError("reviewed corpus requires explicit unique golden_review.ids")
    encoder = review.get("encoder", {})
    if (
        not isinstance(encoder, dict)
        or any(
            encoder.get(key) != reference[key]
            for key in ("repo_id", "revision", "path", "sha256")
        )
        or review.get("tokenizers_version") != reference["tokenizers_version"]
    ):
        raise ReplayError("reviewed corpus reference pins do not match")
    identities = review.get("approved_capture_identities", {})
    source = {
        "repo_id": reference["repo_id"],
        "revision": reference["revision"],
        "encoder_sha256": reference["sha256"],
        "tokenizer_sha256": profile["model"]["assets"]["tokenizer.json"],
    }
    if (
        not isinstance(identities, dict)
        or not same_json(identities.get("source"), source)
        or not isinstance(identities.get("records"), dict)
    ):
        raise ReplayError(
            "reviewed corpus requires matching approved capture identities"
        )
    cases = {case.identifier: case for case in corpus.cases}
    if set(ids) - set(cases) or any(
        case.expected_response is not None and case.identifier not in ids
        for case in corpus.cases
    ):
        raise ReplayError(
            "reviewed response expectations must match the explicit approval list"
        )
    for identifier in ids:
        case = cases[identifier]
        if case.expected_response is None:
            raise ReplayError("approved ID lacks a response expectation")
        ledger = identities["records"].get(identifier)
        output = [token for event in case.worker_events for token in event["token_ids"]]
        if (
            not isinstance(ledger, dict)
            or ledger.get("prompt_token_ids_sha256")
            != _sha(dumps(case.expected_preprocessed["token_ids"]).encode())
            or ledger.get("output_token_ids_sha256") != _sha(dumps(output).encode())
        ):
            raise ReplayError("reviewed corpus token ledger mismatch: " + identifier)
        # Legacy approvals hash their documented compact lossless serializer,
        # with return_token_ids appended before the unary stream selector.
        wire = {
            **_effective(case.request),
            "return_token_ids": True,
            "stream": False,
        }
        legacy = dumps(wire)
        if _sha(legacy.encode()) != ledger.get("request_bytes_sha256"):
            raise ReplayError(
                "reviewed corpus legacy request ledger missing or mismatched: "
                + identifier
            )
    return {identifier: cases[identifier] for identifier in ids}, {
        "manifest_sha256": corpus.manifest_sha256,
        "golden_review": review,
    }


def prepare_corpus(
    profile_path: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    encoder: Path | None = None,
    captures: Path | None = None,
    requests: Path | None = None,
    reviewed_corpus: Path | None = None,
) -> dict[str, Any]:
    """Export a fresh deterministic corpus; captured API messages are diagnostic."""
    if (captures is None) == (requests is None):
        raise ReplayError("choose exactly one capture directory or request JSONL")
    if reviewed_corpus is not None and captures is None:
        raise ReplayError("reviewed responses require captured evidence")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ReplayError("preparation output must be fresh or empty")
    profile, reference = load_profile(profile_path)
    native = ReferenceEncoder(profile, reference, model_dir, encoder)
    evidence: dict[str, Any]
    if captures is not None:
        seeds, records, capture, capture_sha = _capture_records(captures, profile)
        metadata = capture.get("reference", {})
        evidence = {
            "kind": "captured",
            "capture_manifest_sha256": capture_sha,
            "requests_sha256": capture["requests_sha256"],
            "reference": {
                key: metadata[key] for key in ("version", "models") if key in metadata
            },
        }
    else:
        if requests is None:
            raise ReplayError("reference-only preparation requires request JSONL")
        data = requests.read_bytes()
        seeds, records = read_seeds(data, profile), []
        evidence = {"kind": "synthetic", "requests_sha256": _sha(data)}
    reviewed: dict[str, Any] = {}
    review_evidence: dict[str, Any] | None = None
    if reviewed_corpus is not None:
        reviewed, review_evidence = _reviewed(reviewed_corpus, profile, reference)
    lines: list[str] = []
    for index, seed in enumerate(seeds):
        request = seed["request"]
        prompt, transformations = native.prompt(request, captured=captures is not None)
        effective = _effective(request)
        case_reference = copy.deepcopy(seed.get("reference", {}))
        if "preparation" in case_reference:
            raise ReplayError(
                "seed.reference.preparation is reserved for generated evidence"
            )
        preparation: dict[str, Any] = {}
        case_reference["preparation"] = preparation
        if captures is not None:
            record = records[index]
            observed_prompt, output, terminal = normalize_terminal(
                request, record["response"], native.eos_ids
            )
            if observed_prompt != prompt:
                raise ReplayError(
                    "vendor prompt IDs disagree with captured prompt: " + seed["id"]
                )
            preparation["capture"] = {
                "request_sha256": record["request_sha256"],
                "response_sha256": record["response_sha256"],
                "diagnostic_response": {
                    "message": record["response"]["choices"][0].get("message"),
                    "usage": record["response"].get("usage"),
                },
                "terminal": {
                    "api_finish_reason": record["response"]["choices"][0].get(
                        "finish_reason"
                    ),
                    "stop_reason": record["response"]["choices"][0].get("stop_reason"),
                    "worker_finish_reason": terminal,
                },
            }
        else:
            output, terminal = native.synthetic(request), "stop"
            preparation["completion"] = {
                "kind": "synthetic",
                "method": "pinned vendor render_message at assistant continuation; content OK",
            }
        fields = {
            pointer: request[key]
            for key, pointer in CONTROL_POINTERS.items()
            if key in request
        }
        fields.update(
            {
                "/eos_token_ids": native.eos_ids,
                "/stop_conditions/stop_token_ids_hidden": native.eos_ids,
            }
        )
        expected: dict[str, Any] = {
            "preprocessed": {"token_ids": prompt, "fields_equal": fields}
        }
        if seed["id"] in reviewed:
            golden = reviewed[seed["id"]]
            golden_output = [
                token for event in golden.worker_events for token in event["token_ids"]
            ]
            if (
                not same_json(_effective(golden.request), effective)
                or golden.expected_preprocessed["token_ids"] != prompt
                or golden_output != output
                or golden.worker_events[-1]["finish_reason"] != terminal
                or golden.worker_events[-1].get("stop_reason") is not None
            ):
                raise ReplayError("reviewed response binding mismatch: " + seed["id"])
            expected["response"] = golden.expected_response
        if transformations:
            preparation["reference_transformations"] = transformations
        value = {
            "id": seed["id"],
            "source": seed["source"],
            "request": effective,
            "replay": {
                "worker_events": [
                    {
                        "index": 0,
                        "token_ids": output,
                        "tokens": None,
                        "text": None,
                        "finish_reason": terminal,
                    }
                ]
            },
            "expect": expected,
            "reference": case_reference,
        }
        parse_case(value)
        lines.append(dumps(value) + "\n")
    trace_bytes = "".join(lines).encode("utf-8")
    provenance: dict[str, Any] = {
        "preparation": {
            "method": "pinned official vendor encoder and native tokenizer; never candidate Dynamo rendering",
            "encoder": {
                key: reference[key] for key in ("repo_id", "revision", "path", "sha256")
            },
            "tokenizers_version": reference["tokenizers_version"],
            "tool_sha256": {
                "prepare.py": file_digest(Path(__file__)),
                "contract.py": file_digest(Path(contract.__file__)),
            },
            "evidence": evidence,
            "response_policy": "captured API responses are diagnostic unless explicitly bound to reviewed response goldens",
        }
    }
    if review_evidence is not None:
        provenance["reviewed_corpus"] = review_evidence
        # Keep only promoted IDs in the new approval list and preserve its pins.
        provenance["golden_review"] = {
            **review_evidence["golden_review"],
            "ids": [seed["id"] for seed in seeds if seed["id"] in reviewed],
        }
    manifest = {
        "schema": SCHEMA,
        "profile": profile,
        "traces": [
            {"path": "cases.jsonl", "sha256": _sha(trace_bytes), "count": len(seeds)}
        ],
        "provenance": provenance,
    }
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        output_dir / "cases.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(trace_bytes)
    write_json(output_dir / "manifest.json", manifest)
    load_corpus(output_dir)
    return manifest


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument(
        "--output", "--output-dir", dest="output", type=Path, required=True
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-dir", "--capture", dest="captures", type=Path)
    source.add_argument("--requests", type=Path)
    parser.add_argument("--reviewed-corpus", type=Path)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_corpus(
        args.profile,
        args.model_dir,
        args.output,
        encoder=args.encoder,
        captures=args.captures,
        requests=args.requests,
        reviewed_corpus=args.reviewed_corpus,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        manifest = execute(args)
    except (ReplayError, OSError) as exc:
        print(dumps({"status": "error", "error": str(exc)}))
        return 2
    print(dumps({"status": "prepared", "count": manifest["traces"][0]["count"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
