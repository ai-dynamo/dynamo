# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the vLLM-Omni backend."""

import asyncio
import logging
from typing import Any, cast

import torch
from vllm.sampling_params import SamplingParams
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.entrypoints.stage_utils import shm_read_bytes
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.common.utils.video_utils import compute_num_frames, parse_size

DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_VIDEO_SIZE = "832x480"
MAX_IMAGE_DIMENSION = 4096
# Longest a client-supplied ``size`` may render as inside an error or log line.
SIZE_LABEL_LIMIT = 32


def _coerce_dimension(value: Any, name: str) -> int:
    """Convert a client-supplied width/height to a bounded int, rejecting
    non-numeric or out-of-range values instead of letting ``int()`` raise."""
    # bool is an int subclass and float truncates silently; reject both so
    # true/1.5 don't slip through as 1.
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be an integer")
    try:
        dim = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= dim <= MAX_IMAGE_DIMENSION:
        raise ValueError(f"{name} must be between 1 and {MAX_IMAGE_DIMENSION}")
    return dim


def streaming_sampling_params(
    engine_client: Any, sampling_params_list: list[Any] | None = None
) -> list[Any]:
    """Return request parameters or engine defaults configured for streaming."""
    source = (
        sampling_params_list
        if sampling_params_list is not None
        else engine_client.default_sampling_params_list
    )
    return coerce_param_message_types(list(source or []), is_streaming=True)


def shm_deserialize(shm_meta: dict) -> Any:
    """Read and deserialize an OmniRequestOutput from shared memory."""
    return OmniSerializer.deserialize(shm_read_bytes(shm_meta))


async def ensure_awaited(value: Any) -> Any:
    """Await a value if it is a coroutine, otherwise return it directly."""
    if asyncio.iscoroutine(value):
        return await value
    return value


def unwrap_connector_payload(payload: Any) -> Any:
    """Unpack connector return value (some return (payload,) tuples)."""
    return payload[0] if isinstance(payload, tuple) else payload


def is_empty_payload(value: Any) -> bool:
    """Check if a payload value is empty/None (tensor-aware)."""
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    if isinstance(value, (list, tuple, dict, str, bytes, bytearray, set)):
        return len(value) == 0
    return False


def coerce_token_ids_to_list(token_ids: Any) -> list[Any]:
    """Normalize token_ids (tensor, list, tuple, or other) to a Python list."""
    if token_ids is None:
        return []
    if isinstance(token_ids, torch.Tensor):
        return token_ids.detach().cpu().tolist()
    if isinstance(token_ids, (list, tuple)):
        return list(token_ids)
    try:
        return list(token_ids)
    except TypeError:
        return [token_ids]


def image_generation_mm_processor_kwargs(height: int, width: int) -> dict[str, int]:
    """Build processor kwargs that force image prompts through multimodal preprocessing."""
    return {"target_h": height, "target_w": width}


def _size_dimension_fields(size: Any) -> tuple[str, str]:
    """Error labels naming ``size`` as the source of a width/height.

    A dimension derived from ``size`` must not be reported as ``width``: the
    client never sent that field and would have nothing to correct.

    ``size`` is unbounded client input and this label reaches both the error
    returned to the caller and the log line at the handler, so echo at most
    ``SIZE_LABEL_LIMIT`` characters of it -- enough to identify a plausible
    ``WxH`` value, and never a megabyte of it per request.
    """
    if isinstance(size, str) and len(size) > SIZE_LABEL_LIMIT:
        shown: Any = size[:SIZE_LABEL_LIMIT] + "..."
    else:
        shown = size
    return f"width in size={shown!r}", f"height in size={shown!r}"


def image_generation_size_from_str(
    size: str | None, *, default_w: int = 1024, default_h: int = 1024
) -> tuple[int, int]:
    """Resolve bounded image dimensions from a ``WxH`` size string.

    ``parse_size`` falls back to the defaults for an unparseable string but does
    not bound what it does parse, so every entry point that accepts a
    client-supplied size needs this rather than ``parse_size`` alone.
    """
    width, height = parse_size(size, default_w=default_w, default_h=default_h)
    width_field, height_field = _size_dimension_fields(size)
    return _coerce_dimension(width, width_field), _coerce_dimension(
        height, height_field
    )


def resolve_image_dimensions(request: dict) -> tuple[Any, str, Any, str]:
    """Resolve width/height through the request's precedence chain, uncoerced.

    Returns each value paired with the field it came from. Coercion is left to
    the caller so that callers with a further override (``nvext``) can resolve
    the *complete* chain first: a value a later source replaces never reaches
    the engine, so validating it here would reject a request over a number that
    was discarded.
    """
    extra_body = request.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}

    size = request.get("size") or extra_body.get("size") or DEFAULT_IMAGE_SIZE
    width, height = parse_size(size, default_w=1024, default_h=1024)
    width_field, height_field = _size_dimension_fields(size)

    for source in (extra_body, request):
        if source.get("width") is not None:
            width, width_field = source["width"], "width"
        if source.get("height") is not None:
            height, height_field = source["height"], "height"
    return width, width_field, height, height_field


def image_generation_size_from_request(request: dict) -> tuple[int, int]:
    """Resolve image output dimensions from OpenAI-style image or chat requests."""
    width, width_field, height, height_field = resolve_image_dimensions(request)
    # One coercion covers both the size-derived dims and any explicit override,
    # and reports whichever field the surviving value actually came from.
    return _coerce_dimension(width, width_field), _coerce_dimension(
        height, height_field
    )


def image_generation_sampling_overrides(
    request: dict, height: int, width: int
) -> dict[str, Any]:
    """Collect diffusion sampling overrides for image-generation chat requests."""
    overrides: dict[str, Any] = {"height": height, "width": width}
    for source_name in ("extra_body", "nvext"):
        source = request.get(source_name)
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if key not in {"height", "width", "size"} and value is not None:
                overrides[key] = value
    return overrides


def image_generation_negative_prompt_from_request(request: dict) -> str | None:
    """Resolve negative prompt from the places image requests commonly carry it."""
    for source in (
        request,
        request.get("extra_body"),
        request.get("nvext"),
    ):
        if not isinstance(source, dict):
            continue
        negative_prompt = source.get("negative_prompt")
        if negative_prompt is not None:
            return negative_prompt
    return None


def _normalize_nvext(request: dict) -> dict[str, Any]:
    nvext = request.get("nvext")
    if isinstance(nvext, dict):
        return nvext
    model_dump = getattr(nvext, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return {}


def build_image_generation_prompt(
    prompt: str,
    height: int,
    width: int,
    *,
    negative_prompt: str | None = None,
    multi_modal_data: dict[str, Any] | None = None,
) -> OmniTextPrompt:
    """Build the prompt shape expected by AR-to-diffusion image pipelines."""
    image_prompt = OmniTextPrompt(prompt=prompt)
    if negative_prompt is not None:
        image_prompt["negative_prompt"] = negative_prompt
    if multi_modal_data:
        image_prompt["multi_modal_data"] = multi_modal_data
    image_prompt["modalities"] = ["image"]
    image_prompt["mm_processor_kwargs"] = image_generation_mm_processor_kwargs(
        height, width
    )
    return image_prompt


def build_original_prompt(request: dict, nvext: dict, height: int, width: int) -> Any:
    """Build the rich prompt dict that processor functions (ar2diffusion etc.) read."""
    prompt = OmniTextPrompt(
        prompt=request.get("prompt", ""),
        negative_prompt=request.get("negative_prompt", None),
    )
    if request.get("multi_modal_data"):
        prompt["multi_modal_data"] = request["multi_modal_data"]
    return prompt


async def parse_omni_request(
    request: dict,
    output_modalities: list,
    default_video_fps: int = 16,
    tokenizer_getter=None,
) -> dict:
    """Parse a raw frontend request into engine_inputs, original_prompt, sampling_params_list.

    Args:
      tokenizer_getter: async callable returning a tokenizer (e.g. engine.get_tokenizer).
          When provided, chat requests are formatted through the model's chat template
          so the thinker receives the same prompt as native ``vllm serve --omni``.

    Returns:
      engine_inputs:        text prompt (str or OmniTextPrompt) for the stage 0 engine
      original_prompt:      rich prompt dict with geometry/params for processor functions
      sampling_params_list: raw user overrides dict (height/width/nvext) or None for chat
    """
    _, request_type = parse_request_type(request, output_modalities)

    if request_type in (RequestType.VIDEO_GENERATION, RequestType.IMAGE_GENERATION):
        is_video = request_type == RequestType.VIDEO_GENERATION
        nvext = _normalize_nvext(request)
        default_size = DEFAULT_VIDEO_SIZE if is_video else DEFAULT_IMAGE_SIZE
        size_kwargs = {} if is_video else {"default_w": 1024, "default_h": 1024}
        if is_video:
            width, height = parse_size(request.get("size", default_size), **size_kwargs)
        else:
            # nvext is the highest-priority source, so resolve the whole chain
            # before coercing: coercing the helper's result first would reject
            # a size that nvext goes on to replace, and a bare int() here would
            # reintroduce every failure the helper rejects.
            (
                raw_width,
                width_field,
                raw_height,
                height_field,
            ) = resolve_image_dimensions(request)
            if nvext.get("width") is not None:
                raw_width, width_field = nvext["width"], "nvext.width"
            if nvext.get("height") is not None:
                raw_height, height_field = nvext["height"], "nvext.height"
            width = _coerce_dimension(raw_width, width_field)
            height = _coerce_dimension(raw_height, height_field)
        sp: dict = {**nvext, "height": height, "width": width}
        if is_video:
            sp["num_frames"] = compute_num_frames(
                num_frames=nvext.get("num_frames"),
                fps=nvext.get("fps"),
                default_fps=default_video_fps,
            )
            engine_inputs = OmniTextPrompt(prompt=request.get("prompt", ""))
            original_prompt = build_original_prompt(request, nvext, height, width)
        else:
            engine_inputs = build_image_generation_prompt(
                request.get("prompt", ""),
                height,
                width,
                negative_prompt=image_generation_negative_prompt_from_request(request),
                multi_modal_data=request.get("multi_modal_data"),
            )
            original_prompt = dict(engine_inputs)
        return {
            "engine_inputs": engine_inputs,
            "original_prompt": original_prompt,
            "sampling_params_list": sp,
        }

    # Chat / text
    messages = request.get("messages", [])
    text = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        request.get("prompt", ""),
    )

    # Apply chat template when a tokenizer is available.  The native
    # OpenAI API server applies the template before the engine sees it;
    # without it the thinker receives bare text instead of the full
    # chat-formatted prompt.
    if messages and tokenizer_getter is not None:
        try:
            tokenizer = await tokenizer_getter()
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            logging.getLogger(__name__).debug(
                "Chat template not available, using raw text"
            )

    if any(str(modality).lower() == "image" for modality in output_modalities):
        width, height = image_generation_size_from_request(request)
        engine_prompt = build_image_generation_prompt(
            text,
            height,
            width,
            negative_prompt=image_generation_negative_prompt_from_request(request),
            multi_modal_data=request.get("multi_modal_data"),
        )
        return {
            "engine_inputs": engine_prompt,
            "original_prompt": dict(engine_prompt),
            "sampling_params_list": image_generation_sampling_overrides(
                request, height, width
            ),
        }

    return {
        "engine_inputs": text,
        "original_prompt": {"prompt": text},
        "sampling_params_list": None,
    }


def _build_sampling_params(stage_config: Any, overrides: dict | None) -> list | None:
    """Construct typed sampling params from YAML default_sampling_params."""
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    defaults = getattr(stage_config, "default_sampling_params", None)
    if not defaults:
        return None

    if OmegaConf.is_config(defaults):
        params = OmegaConf.to_container(defaults, resolve=True)
    else:
        params = dict(defaults)
    params_dict = cast(dict[str, Any], params)

    stage_type = getattr(stage_config, "stage_type", "llm")
    if stage_type == "diffusion":
        diffusion_params = OmniDiffusionSamplingParams(**params_dict)
        if overrides:
            for arg, value in overrides.items():
                if hasattr(diffusion_params, arg):
                    setattr(diffusion_params, arg, value)
        return [diffusion_params]

    llm_params = SamplingParams(**params_dict)
    if overrides:
        for arg, value in overrides.items():
            if hasattr(llm_params, arg):
                setattr(llm_params, arg, value)
    return [llm_params]
