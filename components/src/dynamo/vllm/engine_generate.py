# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime capability metadata for vLLM's native Generate API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange
    from vllm.sampling_params import SamplingParams

from dynamo.common.utils.guided_json import reject_nonprogressing_guided_json_ref_cycles
from dynamo.llm import HttpError, ModelInput, ModelRuntimeConfig, ModelType, WorkerType

VLLM_GENERATE_CAPABILITY = "vllm_inference_v1_generate"
VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY = "vllm_enable_tower_connector_lora"
DYNAMO_CACHE_SALT_PREFIX = "dynamo-cache-salt:"


def publish_engine_generate_capability(
    runtime_config: ModelRuntimeConfig,
    model_input: ModelInput,
    model_type: ModelType,
    worker_type: WorkerType,
    tower_connector_lora_enabled: bool,
) -> bool:
    """Publish native Generate support and its MM-routing-relevant config."""
    if model_input != ModelInput.Tokens or worker_type not in (
        WorkerType.Aggregated,
        WorkerType.Decode,
    ):
        return False
    if not (model_type.supports_chat() or model_type == ModelType.Completions):
        return False

    runtime_config.set_engine_specific(
        VLLM_GENERATE_CAPABILITY,
        json.dumps(True),
    )
    runtime_config.set_engine_specific(
        VLLM_ENABLE_TOWER_CONNECTOR_LORA_RUNTIME_KEY,
        json.dumps(tower_connector_lora_enabled),
    )
    return True


@dataclass(frozen=True)
class EngineGenerateInput:
    prompt: Any
    sampling_params: SamplingParams
    priority: int
    # Raw media for the worker to expand into ``prompt`` before generation.
    content_parts: list[dict[str, Any]] | None = None
    return_token_ids: bool = False


@lru_cache(maxsize=1)
def _native_generate_api() -> tuple[Any, Any]:
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


def _image_features(
    request: dict[str, Any],
    features: dict[str, Any],
) -> tuple[
    dict[str, list[str]],
    MultiModalKwargsItems,
    dict[str, list[PlaceholderRange]],
]:
    import torch
    from vllm.multimodal.inputs import (
        MultiModalKwargsItem,
        MultiModalKwargsItems,
        PlaceholderRange,
    )

    mm_hashes = features.get("mm_hashes")
    mm_placeholders = features.get("mm_placeholders")
    kwargs_data = features.get("kwargs_data")
    if not isinstance(mm_hashes, dict) or not isinstance(mm_placeholders, dict):
        raise TypeError("TITO features require mm_hashes and mm_placeholders objects")
    if not isinstance(kwargs_data, dict):
        raise TypeError("TITO features kwargs_data must be an object")

    modalities = set(mm_hashes) | set(mm_placeholders)
    modalities.update(kwargs_data)
    if modalities != {"image"}:
        raise ValueError("TITO preprocessed features currently support image only")

    hashes = mm_hashes.get("image")
    ranges = mm_placeholders.get("image")
    if not isinstance(hashes, list) or not isinstance(ranges, list):
        raise TypeError("TITO image hashes and placeholders must be lists")
    if len(hashes) != len(ranges):
        raise ValueError("TITO image hash and placeholder counts must match")
    if not all(isinstance(value, str) and value for value in hashes):
        raise ValueError("TITO image hashes must be non-empty strings")

    routing_hashes = (request.get("extra_args") or {}).get("dynamo_mm_routing_hashes")
    if routing_hashes is not None:
        if (
            not isinstance(routing_hashes, list)
            or len(routing_hashes) != len(hashes)
            or not all(isinstance(value, str) and value for value in routing_hashes)
        ):
            raise ValueError("TITO image routing hash count or value is invalid")
        hashes = routing_hashes

    prompt_length = len(request["token_ids"])
    restored_ranges: list[PlaceholderRange] = []
    for item in ranges:
        if not isinstance(item, dict):
            raise TypeError("TITO image placeholders must be objects")
        offset = item.get("offset")
        length = item.get("length")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 1
            or offset + length > prompt_length
        ):
            raise ValueError("TITO image placeholder range is invalid")
        is_embed_raw = item.get("is_embed")
        if is_embed_raw is not None:
            if not isinstance(is_embed_raw, (list, tuple)):
                raise TypeError("TITO image placeholder is_embed must be a sequence")
            if len(is_embed_raw) != length:
                raise ValueError(
                    "TITO image placeholder is_embed must match placeholder length"
                )
            if not all(isinstance(value, bool) for value in is_embed_raw):
                raise ValueError(
                    "TITO image placeholder is_embed values must be booleans"
                )
        is_embed = (
            None
            if is_embed_raw is None
            else torch.as_tensor(is_embed_raw, dtype=torch.bool)
        )
        restored_ranges.append(
            PlaceholderRange(offset=offset, length=length, is_embed=is_embed)
        )

    restored_kwargs: list[MultiModalKwargsItem]
    image_data = kwargs_data.get("image")
    if not isinstance(image_data, list) or len(image_data) != len(hashes):
        raise ValueError("TITO image tensor and hash counts must match")
    decode_mm_kwargs_item, _ = _native_generate_api()
    restored_kwargs = [decode_mm_kwargs_item(value) for value in image_data]

    return (
        {"image": hashes},
        MultiModalKwargsItems({"image": restored_kwargs}),
        {"image": restored_ranges},
    )


def adapt_engine_generate_request(
    request: dict[str, Any],
    *,
    enable_multimodal: bool,
    decode_capable: bool,
    vllm_config: Any,
    default_sampling_params: dict[str, Any],
    allow_multimodal_features: bool = True,
) -> EngineGenerateInput | None:
    """Adapt one Rust-frontend TITO envelope at the Python engine boundary."""
    import msgspec
    from vllm.inputs import TokensPrompt, mm_input
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict) or "vllm_tito" not in extra_args:
        return None
    envelope = extra_args["vllm_tito"]
    if not isinstance(envelope, dict):
        raise TypeError("extra_args.vllm_tito must be an object")
    if not decode_capable:
        raise ValueError("TITO requests require an aggregated or decode vLLM worker")
    raw_sampling_params = envelope.get("sampling_params")
    if not isinstance(raw_sampling_params, dict):
        raise TypeError("extra_args.vllm_tito.sampling_params must be an object")
    features = envelope.get("features")
    if features is not None and not allow_multimodal_features:
        raise ValueError("TITO multimodal features require an aggregated vLLM worker")
    if envelope.get("content_parts"):
        if not allow_multimodal_features:
            raise ValueError(
                "TITO multimodal content_parts require an aggregated vLLM worker"
            )
        if not enable_multimodal:
            raise ValueError(
                "TITO multimodal content_parts require --enable-multimodal"
            )
    if isinstance(features, dict):
        kwargs_data = features.get("kwargs_data")
        if not isinstance(kwargs_data, dict):
            raise TypeError("TITO features kwargs_data must be an object")
    token_ids = list(request.get("token_ids") or [])
    raw_prompt_start = raw_sampling_params.get("routed_experts_prompt_start", 0)
    if (
        isinstance(raw_prompt_start, bool)
        or not isinstance(raw_prompt_start, int)
        or raw_prompt_start < 0
    ):
        raise ValueError(
            "sampling_params.routed_experts_prompt_start must be a non-negative integer"
        )
    if raw_prompt_start >= len(token_ids):
        raise ValueError(
            "sampling_params.routed_experts_prompt_start must be smaller than "
            "the prompt token count"
        )
    reconstructed = {**envelope, "token_ids": token_ids}
    _, generate_request_type = _native_generate_api()
    native_request = generate_request_type.model_validate(reconstructed)
    sampling_params = native_request.sampling_params
    if isinstance(sampling_params, dict):
        sampling_params = msgspec.convert(sampling_params, type=SamplingParams)
    if not isinstance(sampling_params, SamplingParams):
        raise TypeError("vLLM GenerateRequest returned invalid sampling_params")

    structured_outputs = sampling_params.structured_outputs
    if structured_outputs is not None and structured_outputs.json is not None:
        try:
            reject_nonprogressing_guided_json_ref_cycles(structured_outputs.json)
        except HttpError as exc:
            raise ValueError(str(exc)) from exc

    if native_request.kv_transfer_params is not None:
        sampling_params.extra_args = {
            **(sampling_params.extra_args or {}),
            "kv_transfer_params": native_request.kv_transfer_params,
        }
    if not sampling_params.stop:
        sampling_params.detokenize = False
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    if sampling_params.n > max_num_seqs:
        raise ValueError(
            "sampling_params.n must be at most the server's max_num_seqs "
            f"({max_num_seqs}), got {sampling_params.n}."
        )
    if not native_request.is_sampling_param_provided("max_tokens"):
        # Older supported vLLM builds do not expose this helper. Keep the import
        # lazy so workers that do not adapt native Generate requests still start.
        from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

        model_config = vllm_config.model_config
        generation_config = getattr(model_config, "generation_config", "vllm")
        override_generation_config = (
            getattr(model_config, "override_generation_config", {}) or {}
        )
        override_max_tokens = (
            default_sampling_params.get("max_tokens")
            if generation_config not in ("auto", "vllm")
            else override_generation_config.get("max_new_tokens")
        )
        sampling_params.max_tokens = get_max_tokens(
            max_model_len=model_config.max_model_len,
            max_tokens=None,
            input_length=len(token_ids),
            default_sampling_params=default_sampling_params,
            override_max_tokens=override_max_tokens,
        )
    sampling_params.output_kind = RequestOutputKind.DELTA

    cache_salt = envelope.get("cache_salt")
    engine_cache_salt = (
        f"{DYNAMO_CACHE_SALT_PREFIX}{cache_salt}" if cache_salt else None
    )
    if features is None:
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        if engine_cache_salt is not None:
            prompt["cache_salt"] = engine_cache_salt
    else:
        if not enable_multimodal:
            raise ValueError("TITO multimodal features require --enable-multimodal")
        if not isinstance(features, dict):
            raise ValueError("extra_args.vllm_tito.features must be an object")
        mm_hashes, mm_kwargs, mm_placeholders = _image_features(request, features)
        prompt = mm_input(
            prompt_token_ids=token_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            cache_salt=engine_cache_salt,
        )

    return EngineGenerateInput(
        prompt=prompt,
        sampling_params=sampling_params,
        priority=native_request.priority,
        content_parts=native_request.content_parts or None,
        return_token_ids=bool(native_request.return_token_ids),
    )


_CONTENT_PART_PARSERS = {
    "image_url": "parse_image",
    "audio_url": "parse_audio",
    "video_url": "parse_video",
}


async def render_content_parts(
    renderer: Any,
    model_config: Any,
    prompt: dict[str, Any],
    content_parts: list[dict[str, Any]],
) -> Any:
    """Expand raw media into ``prompt`` as vLLM's own Generate API does."""
    import aiohttp
    from vllm.renderers.chat_utils import AsyncMultiModalItemTracker

    from dynamo.common.http import HttpStatusError
    from dynamo.common.http.url_validator import describe_media_source

    tracker = AsyncMultiModalItemTracker(model_config)
    parser = tracker.create_parser()
    for part in content_parts:
        method = _CONTENT_PART_PARSERS.get(part.get("type", ""))
        if method is None:
            raise ValueError(
                f"TITO content_parts type {part.get('type')!r} is not supported"
            )
        getattr(parser, method)(part.get("url"), part.get("uuid"))
    try:
        mm_data, mm_uuids = await tracker.resolve_items()
    except aiohttp.ClientConnectionError as e:
        # An unreachable user-supplied URL is a client error, not a worker fault.
        sources = ", ".join(
            describe_media_source(str(part.get("url"))) for part in content_parts
        )
        raise HttpStatusError(
            400, f"Connection error loading media: '{sources}': {e}", sources
        ) from e
    prompt = dict(prompt)
    if mm_data:
        prompt["multi_modal_data"] = mm_data
    if mm_uuids:
        prompt["multi_modal_uuids"] = mm_uuids
    (engine_input,) = await renderer.render_cmpl_async([prompt])
    return engine_input


def prompt_response_fields(prompt: Any) -> dict[str, Any]:
    """Return vLLM's ``prompt_token_ids`` and ``mm_placeholders`` response fields."""
    mm_placeholders = None
    if prompt.get("type") == "multimodal":
        mm_placeholders = {
            modality: [{"offset": p.offset, "length": p.length} for p in ranges]
            for modality, ranges in prompt["mm_placeholders"].items()
        }
    return {
        "prompt_token_ids": list(prompt["prompt_token_ids"]),
        "mm_placeholders": mm_placeholders,
    }
