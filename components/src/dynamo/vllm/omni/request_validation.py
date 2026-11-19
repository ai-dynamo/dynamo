# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured-output admission and parameter conversion for vLLM-Omni."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any

from dynamo.common.utils.guided_json import reject_nonprogressing_guided_json_ref_cycles
from dynamo.llm.exceptions import InvalidArgument

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams


UNSUPPORTED_JSON_SCHEMA_MESSAGE = (
    "response_format.type=json_schema is not supported by this Dynamo "
    "vLLM-Omni configuration"
)


def is_json_schema_request(request: dict[str, Any]) -> bool:
    """Identify the chat constraint format without touching media formats."""
    response_format = request.get("response_format")
    return (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_schema"
    )


def _stage_value(stage: Any, key: str) -> Any:
    value = stage.get(key) if isinstance(stage, Mapping) else getattr(stage, key, None)
    return value.value if isinstance(value, Enum) else value


def is_llm_stage(stage: Any) -> bool:
    """Match both native and legacy LLM stage metadata."""
    return _stage_value(stage, "stage_type") == "llm"


def json_schema_stage_index(
    request: dict[str, Any],
    stage_configs: Sequence[Any] | None = None,
    *,
    staged: bool = False,
    output_modalities: Sequence[str] | None = None,
) -> int | None:
    """Select a unique public text stage; unsupported topologies fail closed."""
    if not is_json_schema_request(request):
        return None
    configs = list(stage_configs or [])
    if request.get("modalities") not in (None, ["text"]):
        raise InvalidArgument(UNSUPPORTED_JSON_SCHEMA_MESSAGE)
    if output_modalities is not None and "text" not in {
        str(modality).lower() for modality in output_modalities
    }:
        raise InvalidArgument(UNSUPPORTED_JSON_SCHEMA_MESSAGE)
    text_stages = [
        index
        for index, stage in enumerate(configs)
        if _stage_value(stage, "final_output") is True
        and _stage_value(stage, "final_output_type") == "text"
    ]
    # The staged router formats only its last stage, and only stage 0 receives
    # the raw request. Other topologies need explicit per-stage wire fields.
    if (
        len(text_stages) != 1
        or not is_llm_stage(configs[text_stages[0]])
        or _stage_value(configs[text_stages[0]], "is_comprehension") is not True
        or (staged and (len(configs) != 1 or _stage_value(configs[0], "stage_id") != 0))
    ):
        raise InvalidArgument(UNSUPPORTED_JSON_SCHEMA_MESSAGE)
    return text_stages[0]


@cache
def _response_format_type_adapter():
    # Optional native protocol imports must not run during lightweight admission.
    from pydantic import TypeAdapter
    from vllm.entrypoints.generate.base.protocol import AnyResponseFormat

    return TypeAdapter(AnyResponseFormat)


def apply_json_schema_response_format(
    request: dict[str, Any], sampling_params: SamplingParams | None = None
) -> SamplingParams:
    """Convert the wire format locally without mutating stage defaults."""
    # Keep optional native protocol imports out of lightweight admission checks.
    from pydantic import ValidationError
    from vllm.entrypoints.generate.base.protocol import (
        structured_outputs_from_response_format,
    )
    from vllm.exceptions import VLLMClientError
    from vllm.sampling_params import SamplingParams

    params = (
        deepcopy(sampling_params) if sampling_params is not None else SamplingParams()
    )
    try:
        response_format = _response_format_type_adapter().validate_python(
            request["response_format"]
        )
    except ValidationError as exc:
        raise InvalidArgument("Invalid response_format") from exc
    if (
        response_format.type != "json_schema"
        or response_format.json_schema is None
        or response_format.json_schema.json_schema is None
    ):
        raise InvalidArgument("Invalid response_format")
    reject_nonprogressing_guided_json_ref_cycles(
        response_format.json_schema.json_schema
    )
    try:
        params.structured_outputs = structured_outputs_from_response_format(
            params.structured_outputs, response_format
        )
    except VLLMClientError as exc:
        raise InvalidArgument("Invalid structured-output constraint") from exc
    return params
