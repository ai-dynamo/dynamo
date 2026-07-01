# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the vLLM-Omni backend."""

import logging
from typing import Any, cast

from vllm.sampling_params import SamplingParams
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.entrypoints.stage_utils import shm_read_bytes
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.common.utils.video_utils import compute_num_frames, parse_size

DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_VIDEO_SIZE = "832x480"


def shm_deserialize(shm_meta: dict) -> Any:
    """Read and deserialize an OmniRequestOutput from shared memory."""
    return OmniSerializer.deserialize(shm_read_bytes(shm_meta))


def image_generation_mm_processor_kwargs(height: int, width: int) -> dict[str, int]:
    """Build processor kwargs that force image prompts through multimodal preprocessing."""
    return {"target_h": height, "target_w": width}


def image_generation_size_from_request(request: dict) -> tuple[int, int]:
    """Resolve image output dimensions from OpenAI-style image or chat requests."""
    extra_body = request.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}

    size = request.get("size") or extra_body.get("size") or DEFAULT_IMAGE_SIZE
    width, height = parse_size(size, default_w=1024, default_h=1024)

    for source in (extra_body, request):
        if source.get("width") is not None:
            width = int(source["width"])
        if source.get("height") is not None:
            height = int(source["height"])
    return width, height


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
            width, height = image_generation_size_from_request(request)
            if nvext.get("width") is not None:
                width = int(nvext["width"])
            if nvext.get("height") is not None:
                height = int(nvext["height"])
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
