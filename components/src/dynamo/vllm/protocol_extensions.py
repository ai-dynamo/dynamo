# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded vLLM sampling extensions at the frontend/worker wire boundary.

These fields affect decode sampling/output, not prompt preprocessing or KV cache
identity. This is deliberately not the union of SamplingParams attributes: every
new field needs an ownership, cache/disaggregation, validation and N-2 review.
The module has no engine imports so both boundaries can use the same contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from dynamo.common.legacy_vllm import LegacyVllmRelease

CAPABILITY_KEY = "vllm_protocol_extensions"
PROMPT_LOGPROBS_CAPABILITY_KEY = "vllm_prompt_logprobs"
FULL_VOCAB_COUNT = 0xFFFFFFFF
SCHEMA_VERSION = 1
MAX_EXTENSION_BYTES = 65536
MAX_EXTENSION_DEPTH = 8
MAX_EXTENSION_VALUES = 16384
SAMPLING_FIELDS = frozenset(
    {"allowed_token_ids", "bad_words_token_ids", "logprob_token_ids"}
)
LEGACY_FIELDS = SAMPLING_FIELDS | {"detokenize"}


class ProtocolExtensionError(ValueError):
    """A payload-free, client-actionable extension contract error."""

    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(
            f"vLLM protocol extension `{field}` rejected at transport: {reason}. "
            "Remove the field or use a compatible worker."
        )


def _bounded(value: Any) -> None:
    pending = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        # Reject an oversized scalar before the JSON encoder can allocate an
        # equally large escaped copy. Strings are not accepted sampling values,
        # but malformed wire input still reaches this boundary.
        if isinstance(item, str) and len(item) > MAX_EXTENSION_BYTES:
            raise ProtocolExtensionError("backend_extensions", "size limit exceeded")
        if depth > MAX_EXTENSION_DEPTH or visited + len(pending) > MAX_EXTENSION_VALUES:
            raise ProtocolExtensionError(
                "backend_extensions", "nesting or value-count limit exceeded"
            )
        if (
            isinstance(item, (dict, list))
            and visited + len(pending) + len(item) > MAX_EXTENSION_VALUES
        ):
            raise ProtocolExtensionError(
                "backend_extensions", "value-count limit exceeded"
            )
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ProtocolExtensionError("backend_extensions", "invalid JSON key")
            if any(len(key) > MAX_EXTENSION_BYTES for key in item):
                raise ProtocolExtensionError(
                    "backend_extensions", "size limit exceeded"
                )
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    try:
        size = 0
        encoder = json.JSONEncoder(
            separators=(",", ":"), allow_nan=False, ensure_ascii=False
        )
        for chunk in encoder.iterencode(value):
            size += len(chunk.encode("utf-8"))
            if size > MAX_EXTENSION_BYTES:
                raise ProtocolExtensionError(
                    "backend_extensions", "size limit exceeded"
                )
    except ProtocolExtensionError:
        raise
    except (TypeError, ValueError) as error:
        raise ProtocolExtensionError(
            "backend_extensions", "invalid JSON value"
        ) from error


def validate_sampling_fields(fields: Any, *, legacy: bool = False) -> dict[str, Any]:
    """Validate exact wire types, including bool-versus-int and u32 token IDs."""
    if not isinstance(fields, dict):
        raise ProtocolExtensionError("vllm", "expected an object")
    allowed = LEGACY_FIELDS if legacy else SAMPLING_FIELDS
    if fields.keys() - allowed:
        # Do not echo arbitrary unknown keys or their values into error/log labels.
        raise ProtocolExtensionError(
            "vllm", "unknown or canonical field in extension map"
        )
    _bounded(fields)
    for name, value in fields.items():
        # Legacy Rust Option fields use null as omission. This normalization lives
        # at the reader boundary; no null sentinel leaks into the core adapter.
        if value is None:
            continue
        if name == "detokenize":
            if type(value) is not bool:
                raise ProtocolExtensionError(name, "expected a boolean")
            continue
        groups = value if name == "bad_words_token_ids" else [value]
        if not isinstance(groups, list) or any(
            not isinstance(group, list)
            or any(
                type(token) is not int or not 0 <= token <= 0xFFFFFFFF
                for token in group
            )
            for group in groups
        ):
            raise ProtocolExtensionError(
                name, "expected unsigned 32-bit token ID arrays"
            )
    return {name: value for name, value in fields.items() if value is not None}


def resolve_sampling_extensions(
    request: Mapping[str, Any], supported_fields: set[str] | frozenset[str]
) -> dict[str, Any]:
    """Normalize N-2 legacy and v1 representations without overriding canonical data.

    Compatibility for pre-v1 senders in Dynamo 1.4/1.5 during the 1.6 rollout.
    TODO(1.8): remove the legacy reader when 1.5 leaves the N-2 window.
    Agreeing dual writes are allowed; conflicting representations are rejected.
    """
    extra = request.get("extra_args")
    if extra is None:
        return {}
    if not isinstance(extra, dict):
        raise ProtocolExtensionError("extra_args", "expected an object")
    old = extra.get("sampling_options")
    legacy = {} if old is None else validate_sampling_fields(old, legacy=True)
    current = {}
    envelope = extra.get("backend_extensions")
    if envelope is not None:
        if not isinstance(envelope, dict):
            raise ProtocolExtensionError("backend_extensions", "expected an object")
        _bounded(envelope)
        if set(envelope) != {"schema_version", "vllm"}:
            raise ProtocolExtensionError(
                "backend_extensions", "unknown backend or envelope key"
            )
        if (
            type(envelope["schema_version"]) is not int
            or envelope["schema_version"] != SCHEMA_VERSION
        ):
            raise ProtocolExtensionError("schema_version", "unsupported schema version")
        current = validate_sampling_fields(envelope["vllm"])
        for name, value in envelope["vllm"].items():
            if name in legacy and legacy[name] != value:
                raise ProtocolExtensionError(
                    name, "conflicting current and legacy values"
                )
    canonical = request.get("sampling_options")
    if canonical is None:
        canonical = {}
    if not isinstance(canonical, dict):
        raise ProtocolExtensionError("sampling_options", "expected an object")
    for name, value in current.items():
        if name in legacy and legacy[name] != value:
            raise ProtocolExtensionError(name, "conflicting current and legacy values")
    merged = {**legacy, **current}
    for name, value in merged.items():
        if (
            name in canonical
            and canonical[name] is not None
            and canonical[name] != value
        ):
            raise ProtocolExtensionError(
                name, "conflicts with canonical sampling field"
            )
        if name != "detokenize" and name not in supported_fields:
            raise ProtocolExtensionError(
                name, "installed engine cannot preserve this field"
            )
    return merged


def supported_sampling_extensions(sampling_params: Any) -> set[str]:
    """Resolve version-dependent engine support rather than guessing from a version string."""
    return {
        name
        for name in SAMPLING_FIELDS
        if hasattr(
            sampling_params,
            "_bad_words_token_ids" if name == "bad_words_token_ids" else name,
        )
    }


def apply_sampling_extensions(sampling_params: Any, request: Mapping[str, Any]) -> None:
    """Apply the validated wire subset with native-server normalization.

    vLLM's chat and completion adapters turn an empty logprob_token_ids list into
    None. Preserve an empty allowed_token_ids list instead: it is a different
    directive and the engine owns its vocabulary-dependent validation.
    """
    fields = resolve_sampling_extensions(
        request, supported_sampling_extensions(sampling_params)
    )
    for name, value in fields.items():
        attribute = "_bad_words_token_ids" if name == "bad_words_token_ids" else name
        if name == "logprob_token_ids" and not value:
            value = None
        setattr(sampling_params, attribute, value)


def lower_sampling_extensions(
    fields: dict[str, Any],
    runtime_data: Mapping[str, Any],
    *,
    legacy_target: LegacyVllmRelease | None = None,
) -> dict[str, Any]:
    """Frontend writer; return fields to merge into internal extra_args only."""
    fields = validate_sampling_fields(fields)
    if not fields:
        return {}
    capability = runtime_data.get(CAPABILITY_KEY)
    if CAPABILITY_KEY in runtime_data:
        if (
            not isinstance(capability, dict)
            or type(capability.get("schema_version")) is not int
            or capability["schema_version"] != SCHEMA_VERSION
            or capability.get("target") != "vllm"
            or not isinstance(capability.get("engine_version"), str)
            or not isinstance(capability.get("sampling_fields"), list)
            or any(not isinstance(name, str) for name in capability["sampling_fields"])
        ):
            raise ProtocolExtensionError(
                "backend_extensions", "malformed or unsupported worker capability"
            )
        for name in fields:
            if name not in capability["sampling_fields"]:
                raise ProtocolExtensionError(
                    name, "selected worker cannot preserve this field"
                )
        envelope = {"schema_version": SCHEMA_VERSION, "vllm": fields}
        _bounded(envelope)
        # The decode card cannot establish the capability of every downstream
        # prefill hop. Dual-write only the old, representable subset for N-2.
        # TODO(1.8): remove when Dynamo 1.5 leaves the support window.
        return {"backend_extensions": envelope, "sampling_options": dict(fields)}
    if legacy_target is not None:
        if not isinstance(legacy_target, LegacyVllmRelease):
            raise ProtocolExtensionError("vllm", "invalid explicit legacy target")
        for name in fields:
            if not legacy_target.supports_field(name):
                raise ProtocolExtensionError(
                    name, "declared legacy vLLM release cannot preserve this field"
                )
        return {"sampling_options": fields}
    if runtime_data.get("vllm_inference_v1_generate") is True:
        # Only identifies legacy-envelope workers advertising native Generate.
        # Dynamo 1.4 has no such marker; its fallback still needs explicit
        # target/profile resolution, not an assumption that missing means vLLM.
        # TODO(1.8): remove with the legacy reader when 1.5 leaves N-2.
        return {"sampling_options": fields}
    raise ProtocolExtensionError(
        next(iter(fields)), "selected worker has no verified vLLM extension capability"
    )


def protocol_capability(sampling_params: Any, engine_version: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target": "vllm",
        "engine_version": engine_version,
        "sampling_fields": sorted(supported_sampling_extensions(sampling_params)),
    }


def prompt_logprobs_capability(model_config: Any) -> dict[str, Any]:
    """Publish effective engine limits and the worker's explicit wire reader.

    Dynamo 1.4/1.5 readers do not reverse the u32-max sentinel. This declaration
    must never be inferred from their older Generate capability or version name.
    """
    return {
        "schema_version": 1,
        "wire_count": "u32_max",
        "max_logprobs": model_config.max_logprobs,
        "vocab_size": model_config.get_vocab_size(),
    }


def _prompt_logprobs_contract(runtime_data: Mapping[str, Any]) -> dict[str, Any] | None:
    if PROMPT_LOGPROBS_CAPABILITY_KEY not in runtime_data:
        return None
    capability = runtime_data[PROMPT_LOGPROBS_CAPABILITY_KEY]
    if (
        not isinstance(capability, dict)
        or type(capability.get("schema_version")) is not int
        or capability["schema_version"] != 1
        or capability.get("wire_count") != "u32_max"
        or type(capability.get("max_logprobs")) is not int
        or not -1 <= capability["max_logprobs"] <= 0x7FFFFFFFFFFFFFFF
        or type(capability.get("vocab_size")) is not int
        or not 0 < capability["vocab_size"] < FULL_VOCAB_COUNT
    ):
        raise ProtocolExtensionError(
            "prompt_logprobs", "malformed or unsupported worker count capability"
        )
    return capability


def prompt_logprobs_model_limit(runtime_data: Mapping[str, Any]) -> int | None:
    """Keep the Python frontend validator aligned with the admitted worker."""
    capability = _prompt_logprobs_contract(runtime_data)
    return None if capability is None else capability["max_logprobs"]


def prompt_logprobs_to_wire(
    value: int | None, runtime_data: Mapping[str, Any]
) -> int | None:
    """Preserve signed public semantics in the existing unsigned output options."""
    if value is None:
        return None
    if type(value) is not int or not -1 <= value < FULL_VOCAB_COUNT:
        raise ProtocolExtensionError(
            "prompt_logprobs", "expected -1 or a non-negative count below 4294967295"
        )
    if value != -1:
        return value
    capability = _prompt_logprobs_contract(runtime_data)
    if capability is None:
        raise ProtocolExtensionError(
            "prompt_logprobs",
            "selected worker has not advertised full-vocabulary wire support",
        )
    if (
        capability["max_logprobs"] != -1
        and capability["max_logprobs"] < capability["vocab_size"]
    ):
        raise ProtocolExtensionError(
            "prompt_logprobs",
            "requested full vocabulary exceeds worker max_logprobs limit",
        )
    return FULL_VOCAB_COUNT
