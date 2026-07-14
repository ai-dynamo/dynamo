# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang request adaptation for Dynamo's engine-native generate API."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.sampling.sampling_params import SamplingParams

from dynamo.common.backend import logprobs as _shared_logprobs
from dynamo.common.utils.structural_tag import serialize_structural_tag

SGLANG_INFERENCE_V1_GENERATE_CAPABILITY = "sglang_inference_v1_generate"
_PAYLOAD_KEY = "sglang_tito"
_STANDARD_REQUEST_FIELDS = frozenset(
    {
        "request_id",
        "sampling_params",
        "model",
        "stream",
        "stream_options",
        "cache_salt",
        "priority",
        "kv_transfer_params",
    }
)
_INTERNAL_SAMPLING_FIELDS = frozenset(
    {
        "stop_strs",
        "stop_regex_strs",
        "stop_str_max_len",
        "stop_regex_max_len",
        "is_normalized",
    }
)
_SGLANG_SAMPLING_FIELDS = (
    frozenset(SamplingParams.__struct_fields__) - _INTERNAL_SAMPLING_FIELDS
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


@dataclass(frozen=True)
class EngineGenerateRequest:
    """Typed worker adapter for one SGLang-native generate request."""

    sglang_tito: dict[str, Any]

    @classmethod
    def from_request(cls, request: dict[str, Any]) -> Optional["EngineGenerateRequest"]:
        sglang_tito = payload(request)
        return cls(sglang_tito) if sglang_tito is not None else None

    def build_sampling_params(self) -> dict[str, Any]:
        self._validate_request_fields()
        return _build_sampling_params(self.sglang_tito)

    def build_logprob_kwargs(self) -> dict[str, Any]:
        self._validate_request_fields()
        return _build_logprob_kwargs(self.sglang_tito)

    def _validate_request_fields(self) -> None:
        unsupported = sorted(set(self.sglang_tito) - _STANDARD_REQUEST_FIELDS)
        if unsupported:
            raise ValueError(
                "unsupported top-level generate field(s) for SGLang: "
                + ", ".join(unsupported)
            )
        if self.sglang_tito.get("cache_salt") is not None:
            raise ValueError("cache_salt is not supported by SGLang generate")
        if self.sglang_tito.get("kv_transfer_params") is not None:
            raise ValueError(
                "kv_transfer_params is managed by Dynamo for SGLang disaggregation"
            )


def payload(request: dict[str, Any]) -> Optional[dict[str, Any]]:
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict):
        return None
    value = extra_args.get(_PAYLOAD_KEY)
    return value if isinstance(value, dict) else None


def _raw_sampling_params(sglang_tito: dict[str, Any]) -> dict[str, Any]:
    raw_params = sglang_tito.get("sampling_params")
    if not isinstance(raw_params, dict):
        raise ValueError("extra_args.sglang_tito.sampling_params must be an object")
    return raw_params


def _build_sampling_params(sglang_tito: dict[str, Any]) -> dict[str, Any]:
    raw_params = _raw_sampling_params(sglang_tito)
    sampling_params: dict[str, Any] = {}
    sources: dict[str, str] = {}

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
        if target_name not in _SGLANG_SAMPLING_FIELDS:
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
        if value.get(name, default) != default
    )
    if unsupported:
        raise ValueError(
            "unsupported structured_outputs field(s) for SGLang: "
            + ", ".join(sorted(unsupported))
        )
    return constraints


def _build_logprob_kwargs(sglang_tito: dict[str, Any]) -> dict[str, Any]:
    raw_params = _raw_sampling_params(sglang_tito)
    logprobs = _logprob_count(raw_params.get("logprobs"), "logprobs")
    prompt_logprobs = _logprob_count(
        raw_params.get("prompt_logprobs"), "prompt_logprobs"
    )
    if logprobs is None and prompt_logprobs is None:
        return {}

    output_options = {
        "logprobs": logprobs,
        "prompt_logprobs": prompt_logprobs,
    }
    kwargs = _shared_logprobs.build_sglang_logprob_kwargs(
        output_options,
        allow_top_logprobs=_shared_logprobs.sglang_top_logprobs_allowed(),
    )
    kwargs.setdefault("logprob_start_len", -1)
    kwargs["return_text_in_logprobs"] = False
    return kwargs


def _logprob_count(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"sampling_params.{name} must be a non-negative integer for SGLang"
        )
    return value
