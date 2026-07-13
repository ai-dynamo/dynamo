# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt vLLM's token-in/token-out request envelope for Dynamo workers."""

import base64
import io
import logging
from functools import lru_cache
from typing import Any

import msgspec
import numpy as np
import torch
from vllm.inputs import TokensPrompt, mm_input
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams

from .constants import DYNAMO_CACHE_SALT_PREFIX

GENERATE_CAPABILITY = "vllm_inference_v1_generate"
logger = logging.getLogger(__name__)


def serialize_routed_experts(routed_experts: Any) -> str | None:
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


@lru_cache(maxsize=1)
def _native_generate_api() -> tuple[Any, Any]:
    """Load the vLLM-native endpoint adapter only when `/generate` is used."""
    try:
        from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
            decode_mm_kwargs_item,
        )
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            GenerateRequest,
        )
    except ModuleNotFoundError as exc:
        expected_module = "vllm.entrypoints.scale_out.token_in_token_out"
        if exc.name is None or not expected_module.startswith(exc.name):
            raise
        from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item
        from vllm.entrypoints.serve.disagg.protocol import GenerateRequest

    return decode_mm_kwargs_item, GenerateRequest


def payload(request: dict[str, Any]) -> dict[str, Any] | None:
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict):
        return None
    value = extra_args.get("vllm_tito")
    return value if isinstance(value, dict) else None


def reconstructed_payload(request: dict[str, Any]) -> dict[str, Any] | None:
    generate_request = payload(request)
    if generate_request is None:
        return None

    token_ids = list(request.get("token_ids") or [])
    envelope_token_ids = generate_request.get("token_ids")
    if envelope_token_ids is not None and token_ids != list(envelope_token_ids):
        raise ValueError("core token_ids do not match extra_args.vllm_tito.token_ids")
    return {**generate_request, "token_ids": token_ids}


def priority(request: dict[str, Any]) -> int:
    generate_request = payload(request)
    if generate_request is not None:
        return int(generate_request.get("priority", 0))
    routing = request.get("routing") or {}
    return -int(routing.get("priority", 0))


def merge_kv_transfer_params(caller: Any, framework: Any) -> dict[str, Any] | Any:
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


def build_prompt(request: dict[str, Any]) -> Any:
    generate_request = reconstructed_payload(request)
    if generate_request is None:
        raise ValueError("extra_args.vllm_tito is missing from token-native request")

    token_ids = generate_request["token_ids"]
    cache_salt = generate_request.get("cache_salt")
    event_cache_salt = f"{DYNAMO_CACHE_SALT_PREFIX}{cache_salt}" if cache_salt else None
    features = generate_request.get("features")
    if not isinstance(features, dict):
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        if event_cache_salt is not None:
            prompt["cache_salt"] = event_cache_salt
        return prompt

    mm_hashes = features.get("mm_hashes") or {}
    placeholders = features.get("mm_placeholders") or {}
    kwargs_data = features.get("kwargs_data")
    mm_placeholders: dict[str, list[PlaceholderRange]] = {}
    for modality, ranges in placeholders.items():
        modality_ranges = []
        for item in ranges:
            length = int(item["length"])
            is_embed_raw = item.get("is_embed")
            is_embed = (
                None
                if is_embed_raw is None
                else torch.as_tensor(is_embed_raw, dtype=torch.bool)
            )
            if is_embed is not None and (
                is_embed.ndim != 1 or is_embed.numel() != length
            ):
                raise ValueError(
                    "placeholder is_embed must be one-dimensional and match length"
                )
            modality_ranges.append(
                PlaceholderRange(
                    offset=int(item["offset"]),
                    length=length,
                    is_embed=is_embed,
                )
            )
        mm_placeholders[modality] = modality_ranges

    mm_kwargs: dict[str, list[MultiModalKwargsItem | None]] = {}
    if isinstance(kwargs_data, dict):
        decode_mm_kwargs_item, _ = _native_generate_api()
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
        cache_salt=event_cache_salt,
    )


def build_sampling_params(
    request: dict[str, Any],
    default_sampling_params: dict[str, Any],
    model_max_len: int | None,
) -> SamplingParams:
    generate_request = reconstructed_payload(request)
    if generate_request is None:
        raise ValueError("extra_args.vllm_tito is missing from token-native request")

    _, generate_request_type = _native_generate_api()
    parsed = generate_request_type.model_validate(generate_request)
    sampling_params = parsed.sampling_params
    if isinstance(sampling_params, dict):
        sampling_params = msgspec.convert(sampling_params, type=SamplingParams)
    if not isinstance(sampling_params, SamplingParams):
        raise TypeError("vLLM GenerateRequest returned invalid sampling_params")

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
                "kv_transfer_params appears in both request and sampling extensions"
            )
        extensions["kv_transfer_params"] = caller_kv
        sampling_params.extra_args = extensions

    sampling_params.output_kind = RequestOutputKind.DELTA
    return sampling_params
