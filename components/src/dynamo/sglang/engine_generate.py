# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang request adaptation for Dynamo's engine-native generate API."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from dynamo.common.utils.structural_tag import serialize_structural_tag

SGLANG_GENERATE_CAPABILITY = "sglang_generate"
_PAYLOAD_KEY = "sglang_tito"
_INTERNAL_SAMPLING_FIELDS = frozenset(
    {
        "stop_strs",
        "stop_regex_strs",
        "stop_str_max_len",
        "stop_regex_max_len",
        "is_normalized",
    }
)
_SAMPLING_ALIASES = {
    "max_tokens": "max_new_tokens",
    "min_tokens": "min_new_tokens",
    "seed": "sampling_seed",
}
_LOGPROB_FIELDS = frozenset({"logprobs", "prompt_logprobs"})
_IGNORABLE_STRUCTURED_DEFAULTS = {
    "disable_any_whitespace": False,
    "disable_additional_properties": False,
    "whitespace_pattern": None,
}


@lru_cache(maxsize=1)
def _sglang_sampling_fields() -> frozenset[str]:
    """Return public SGLang sampling fields without importing SGLang eagerly.

    Dynamo's SGLang modules must remain importable in collection and tooling
    environments that do not install the engine runtime. Resolve the pinned
    SGLang ``SamplingParams`` type only when an engine-native request actually
    needs field validation, and use msgspec's public introspection API rather
    than the type's private ``__struct_fields__`` attribute.
    """
    import msgspec
    from sglang.srt.sampling.sampling_params import SamplingParams

    return (
        frozenset(field.name for field in msgspec.structs.fields(SamplingParams))
        - _INTERNAL_SAMPLING_FIELDS
    )


def build_sampling_params(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """Translate an SGLang Generate payload, or return ``None`` when absent."""
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict):
        return None
    sglang_tito = extra_args.get(_PAYLOAD_KEY)
    if not isinstance(sglang_tito, dict):
        return None
    raw_params = sglang_tito.get("sampling_params")
    if not isinstance(raw_params, dict):
        raise ValueError("extra_args.sglang_tito.sampling_params must be an object")
    return _build_sampling_params(raw_params)


def _build_sampling_params(raw_params: dict[str, Any]) -> dict[str, Any]:
    sampling_params: dict[str, Any] = {}
    sources: dict[str, str] = {}
    supported_fields = _sglang_sampling_fields()

    for source_name, value in raw_params.items():
        if source_name in _LOGPROB_FIELDS:
            continue
        if source_name == "structured_outputs":
            for target_name, target_value in _structured_output_params(value).items():
                _insert_sampling_param(
                    sampling_params,
                    sources,
                    target_name,
                    target_value,
                    source_name,
                )
            continue

        target_name = _SAMPLING_ALIASES.get(source_name, source_name)
        if target_name not in supported_fields:
            raise ValueError(
                f"unsupported sampling parameter for this SGLang: {source_name}"
            )
        _insert_sampling_param(
            sampling_params,
            sources,
            target_name,
            value,
            source_name,
        )

    return sampling_params


def _insert_sampling_param(
    sampling_params: dict[str, Any],
    sources: dict[str, str],
    target_name: str,
    value: Any,
    source_name: str,
) -> None:
    previous_source = sources.get(target_name)
    if previous_source is not None:
        raise ValueError(
            f"sampling parameters {previous_source} and {source_name} both map to "
            f"SGLang's {target_name}"
        )
    sampling_params[target_name] = value
    sources[target_name] = source_name


def _structured_output_params(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("sampling_params.structured_outputs must be an object")

    constraints: dict[str, Any] = {}
    json_schema = value.get("json")
    if json_schema is not None:
        constraints["json_schema"] = (
            json_schema if isinstance(json_schema, str) else json.dumps(json_schema)
        )
    if value.get("json_object"):
        if "json_schema" in constraints:
            raise ValueError("structured_outputs sets both json and json_object")
        constraints["json_schema"] = json.dumps({"type": "object"})
    if value.get("regex") is not None:
        constraints["regex"] = value["regex"]
    if value.get("grammar") is not None:
        constraints["ebnf"] = value["grammar"]
    if value.get("structural_tag") is not None:
        constraints["structural_tag"] = serialize_structural_tag(
            value["structural_tag"]
        )

    choices = value.get("choice")
    if choices:
        if not isinstance(choices, list) or not all(
            isinstance(choice, str) for choice in choices
        ):
            raise ValueError("structured_outputs.choice must be an array of strings")
        if "regex" in constraints:
            raise ValueError("structured_outputs sets both regex and choice")
        constraints["regex"] = (
            "(" + "|".join(re.escape(choice) for choice in choices) + ")"
        )

    active_constraints = [
        name
        for name in ("json_schema", "regex", "ebnf", "structural_tag")
        if name in constraints
    ]
    if len(active_constraints) > 1:
        raise ValueError(
            "SGLang supports one structured-output constraint per request; got "
            + ", ".join(active_constraints)
        )

    known_fields = {
        "json",
        "json_object",
        "regex",
        "grammar",
        "structural_tag",
        "choice",
        *_IGNORABLE_STRUCTURED_DEFAULTS,
    }
    unsupported = sorted(
        name
        for name, item in value.items()
        if name not in known_fields and item is not None
    )
    unsupported.extend(
        name
        for name, default in _IGNORABLE_STRUCTURED_DEFAULTS.items()
        if value.get(name, default) is not default
    )
    if unsupported:
        raise ValueError(
            "unsupported structured_outputs field(s) for SGLang: "
            + ", ".join(sorted(unsupported))
        )
    return constraints
