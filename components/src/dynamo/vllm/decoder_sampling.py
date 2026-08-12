# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dynamo request to native vLLM sampling conversion."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Dict, Final

from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)

from dynamo.common.backend import logprobs as _shared_logprobs
from dynamo.common.utils.guided_json import reject_nonprogressing_guided_json_ref_cycles
from dynamo.common.utils.structural_tag import serialize_structural_tag

logger = logging.getLogger(__name__)

_DELTA_REQUEST_OUTPUT_KIND = RequestOutputKind.DELTA
_KV_TRANSFER_PARAMS_EXTRA_ARGS_KEY: Final = "kv_transfer_params"
_ROUTER_HINT_EXTRA_ARGS_KEY: Final = "router_hint"


def _iter_nvext_sources(request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield request ``nvext`` dictionaries in protocol priority order."""

    extra_args = request.get("extra_args")
    for source in (
        request.get("nvext"),
        extra_args.get("nvext") if isinstance(extra_args, dict) else None,
    ):
        if isinstance(source, dict):
            yield source


def _is_token_in_request(request: Dict[str, Any]) -> bool:
    return any(
        source.get("token_data") or source.get("token_in")
        for source in _iter_nvext_sources(request)
    )


def build_sampling_params(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
    model_max_len: int | None = None,
    enable_rl: bool = False,
    *,
    prompt_token_count_override: int | None = None,
) -> SamplingParams:
    """Build native vLLM sampling parameters from a Dynamo request."""

    if enable_rl and _is_token_in_request(request):
        sampling_params = SamplingParams()
    else:
        sampling_params = SamplingParams(**default_sampling_params)

    sampling_options = dict(request.get("sampling_options") or {})
    extra_args = request.get("extra_args") or {}
    if isinstance(extra_args, dict):
        passthrough_sampling_options = extra_args.get("sampling_options")
        if isinstance(passthrough_sampling_options, dict):
            sampling_options.update(passthrough_sampling_options)
    guided_decoding = sampling_options.get("guided_decoding")
    if guided_decoding is not None and isinstance(guided_decoding, dict):
        json_schema = guided_decoding.get("json")
        if json_schema is not None:
            reject_nonprogressing_guided_json_ref_cycles(json_schema)
        sampling_params.structured_outputs = StructuredOutputsParams(
            json=json_schema,
            regex=guided_decoding.get("regex"),
            choice=guided_decoding.get("choice"),
            grammar=guided_decoding.get("grammar"),
            whitespace_pattern=guided_decoding.get("whitespace_pattern"),
            structural_tag=serialize_structural_tag(
                guided_decoding.get("structural_tag")
            ),
        )

    for key, value in sampling_options.items():
        if key == "guided_decoding":
            continue
        if key == "bad_words_token_ids" and value is not None:
            if not hasattr(sampling_params, "_bad_words_token_ids"):
                raise AttributeError(
                    "vLLM SamplingParams._bad_words_token_ids missing; TITO "
                    "bad_words_token_ids passthrough needs updating for this "
                    "vLLM version"
                )
            sampling_params._bad_words_token_ids = value
            continue
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    routed_experts_start = getattr(sampling_params, "routed_experts_prompt_start", None)
    if routed_experts_start is not None and (
        isinstance(routed_experts_start, bool)
        or not isinstance(routed_experts_start, int)
        or routed_experts_start < 0
    ):
        logger.warning(
            "Ignoring invalid routed_experts_prompt_start=%r "
            "(want non-negative int)",
            routed_experts_start,
        )
        sampling_params.routed_experts_prompt_start = 0

    for key, value in request.get("stop_conditions", {}).items():
        if value is not None and hasattr(sampling_params, key):
            if key == "stop":
                continue
            setattr(sampling_params, key, value)
        if (
            key == "stop_token_ids_hidden"
            and value is not None
            and hasattr(sampling_params, "stop_token_ids")
        ):
            existing = sampling_params.stop_token_ids or []
            sampling_params.stop_token_ids = list(set(existing).union(value))
        if (
            key == "max_thinking_tokens"
            and value is not None
            and hasattr(sampling_params, "thinking_token_budget")
        ):
            sampling_params.thinking_token_budget = value

    output_options = request.get("output_options", {}) or {}
    logprobs, prompt_logprobs = _shared_logprobs.parse_logprob_options(output_options)
    if logprobs is not None:
        sampling_params.logprobs = logprobs
    if prompt_logprobs is not None:
        sampling_params.prompt_logprobs = prompt_logprobs

    provided_max_tokens = request.get("stop_conditions", {}).get("max_tokens", None)
    token_ids = request.get("token_ids", [])
    input_length = (
        prompt_token_count_override
        if prompt_token_count_override is not None
        else len(token_ids)
    )
    if model_max_len is not None and provided_max_tokens is None:
        dynamic_default = max(1, model_max_len - input_length)
        configured_default = default_sampling_params.get("max_tokens", dynamic_default)
        sampling_params.max_tokens = min(configured_default, dynamic_default)

    if isinstance(extra_args, dict):
        request_kv_transfer_params = extra_args.get(_KV_TRANSFER_PARAMS_EXTRA_ARGS_KEY)
        if isinstance(request_kv_transfer_params, dict):
            passthrough_router_hint = request_kv_transfer_params.get(
                _ROUTER_HINT_EXTRA_ARGS_KEY
            )
            if isinstance(passthrough_router_hint, dict):
                passthrough_extra_args = (
                    dict(sampling_params.extra_args)
                    if isinstance(sampling_params.extra_args, dict)
                    else {}
                )
                existing_kv_transfer_params = passthrough_extra_args.get(
                    _KV_TRANSFER_PARAMS_EXTRA_ARGS_KEY
                )
                passthrough_kv_transfer_params = (
                    dict(existing_kv_transfer_params)
                    if isinstance(existing_kv_transfer_params, dict)
                    else {}
                )
                passthrough_kv_transfer_params[
                    _ROUTER_HINT_EXTRA_ARGS_KEY
                ] = passthrough_router_hint
                passthrough_extra_args[
                    _KV_TRANSFER_PARAMS_EXTRA_ARGS_KEY
                ] = passthrough_kv_transfer_params
                sampling_params.extra_args = passthrough_extra_args

    sampling_params.detokenize = False
    sampling_params.output_kind = _DELTA_REQUEST_OUTPUT_KIND
    return sampling_params
