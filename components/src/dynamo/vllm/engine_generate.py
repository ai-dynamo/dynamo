# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM request adaptation for Dynamo's engine-native generate API."""

import base64
import io
import logging
from typing import Any, Dict, Optional

import msgspec
import numpy as np
from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest
from vllm.inputs import TokensPrompt, mm_input
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams

logger = logging.getLogger(__name__)


def serialize_routed_experts(routed_experts: Any) -> Optional[str]:
    """Encode routed experts using vLLM's base64-of-NumPy wire format."""
    if routed_experts is None:
        return None
    try:
        buffer = io.BytesIO()
        np.save(buffer, routed_experts)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        logger.warning(
            "Unable to encode routed_experts for generate API", exc_info=True
        )
        return None


def payload(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict):
        return None
    value = extra_args.get("vllm_tito")
    return value if isinstance(value, dict) else None


def priority(request: Dict[str, Any]) -> int:
    generate_request = payload(request)
    if generate_request is None:
        routing = request.get("routing") or {}
        return -int(routing.get("priority", 0))
    return int(generate_request.get("priority", 0))


def merge_kv_transfer_params(caller: Any, framework: Any) -> Dict[str, Any] | Any:
    if caller is None:
        return framework
    if framework is None:
        return caller
    if not isinstance(caller, dict) or not isinstance(framework, dict):
        raise ValueError(
            "kv_transfer_params from both caller and framework must be objects"
        )
    duplicate = caller.keys() & framework.keys()
    if duplicate:
        raise ValueError(
            "caller and framework kv_transfer_params collide on: "
            + ", ".join(sorted(duplicate))
        )
    return {**caller, **framework}


def build_prompt(request: Dict[str, Any]) -> Any:
    generate_request = payload(request)
    if generate_request is None:
        raise ValueError("extra_args.vllm_tito is missing from token-native request")

    token_ids = list(request.get("token_ids") or [])
    envelope_token_ids = list(generate_request.get("token_ids") or [])
    if token_ids != envelope_token_ids:
        raise ValueError("core token_ids do not match extra_args.vllm_tito.token_ids")

    cache_salt = generate_request.get("cache_salt")
    features = generate_request.get("features")
    if not isinstance(features, dict):
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        if cache_salt is not None:
            prompt["cache_salt"] = cache_salt
        return prompt

    mm_hashes = features.get("mm_hashes") or {}
    placeholders = features.get("mm_placeholders") or {}
    kwargs_data = features.get("kwargs_data")
    mm_placeholders = {
        modality: [
            PlaceholderRange(offset=int(item["offset"]), length=int(item["length"]))
            for item in ranges
        ]
        for modality, ranges in placeholders.items()
    }
    mm_kwargs: Dict[str, list[MultiModalKwargsItem | None]] = {}
    if isinstance(kwargs_data, dict):
        for modality, items in kwargs_data.items():
            mm_kwargs[modality] = [
                decode_mm_kwargs_item(item) if item is not None else None
                for item in items
            ]
    else:
        for modality, hashes in mm_hashes.items():
            mm_kwargs[modality] = [None] * len(hashes)

    return mm_input(
        prompt_token_ids=token_ids,
        mm_kwargs=MultiModalKwargsItems(mm_kwargs),
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholders,
        cache_salt=cache_salt,
    )


def build_sampling_params(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
    model_max_len: int | None,
) -> SamplingParams:
    generate_request = payload(request)
    if generate_request is None:
        raise ValueError("extra_args.vllm_tito is missing from token-native request")

    parsed = GenerateRequest.model_validate(generate_request)
    sampling_params = parsed.sampling_params
    if isinstance(sampling_params, dict):
        sampling_params = msgspec.convert(sampling_params, type=SamplingParams)
    if not isinstance(sampling_params, SamplingParams):
        raise TypeError(
            "vLLM GenerateRequest returned unsupported sampling_params type "
            f"{type(sampling_params).__name__}"
        )

    raw_params = generate_request.get("sampling_params") or {}
    if not isinstance(raw_params, dict):
        raise ValueError("extra_args.vllm_tito.sampling_params must be an object")

    if "max_tokens" not in raw_params and model_max_len is not None:
        input_length = len(request.get("token_ids") or [])
        dynamic_default = max(1, model_max_len - input_length)
        configured_default = default_sampling_params.get("max_tokens", dynamic_default)
        sampling_params.max_tokens = min(configured_default, dynamic_default)

    caller_kv = generate_request.get("kv_transfer_params")
    if caller_kv is not None:
        extensions = dict(sampling_params.extra_args or {})
        if "kv_transfer_params" in extensions:
            raise ValueError(
                "kv_transfer_params appears in both the request and sampling extensions"
            )
        extensions["kv_transfer_params"] = caller_kv
        sampling_params.extra_args = extensions

    # Dynamo's worker, router accounting, and migration path consume deltas.
    sampling_params.output_kind = RequestOutputKind.DELTA
    return sampling_params
