# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the vLLM-Omni backend."""

import asyncio
import logging
from typing import Any, cast

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.sampling_params import SamplingParams
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.stage_utils import shm_read_bytes
from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.common.utils.video_utils import compute_num_frames, parse_size

DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_VIDEO_SIZE = "832x480"


def load_omni_stage_configs(model: str, stage_configs_path: str | None) -> list:
    """Load resolved vLLM-Omni stage configs from deploy-format YAML."""
    if not stage_configs_path:
        return []
    _, stage_configs = load_and_resolve_stage_configs(
        model,
        stage_configs_path,
        kwargs={},
    )
    return stage_configs


def stage_configs_use_async_chunk(stage_configs: list) -> bool:
    if not stage_configs:
        return False
    engine_args = getattr(stage_configs[0], "engine_args", None)
    return bool(getattr(engine_args, "async_chunk", False))


class OmniChatPreprocessor(OmniOpenAIServingChat):
    """Adapter for vLLM-Omni's OpenAI chat preprocessing."""

    def __init__(self, model_config: Any):
        self.model_config = model_config
        self._supported_speakers = None


async def preprocess_chat_with_omni(
    request: dict[str, Any],
    chat_preprocessor: Any,
    renderer: Any,
) -> dict[str, Any] | None:
    """Render chat through vLLM-Omni's OpenAI chat preprocessing."""
    if chat_preprocessor is None or renderer is None:
        return None

    try:
        chat_request = ChatCompletionRequest.model_validate(request)
        audio_options = (
            request.get("audio") if isinstance(request.get("audio"), dict) else {}
        )
        for attr in ("voice", "speaker", "language", "instructions"):
            value = request.get(attr) or audio_options.get(attr)
            if value is not None:
                object.__setattr__(chat_request, attr, value)
        (
            _conversation,
            engine_prompts,
        ) = await chat_preprocessor._preprocess_chat(
            chat_request,
            chat_request.messages,
            default_template=chat_request.chat_template,
            default_template_content_format="auto",
            default_template_kwargs=getattr(chat_request, "chat_template_kwargs", None),
            tool_dicts=(
                [tool.model_dump() for tool in chat_request.tools]
                if chat_request.tools is not None
                else None
            ),
            renderer=renderer,
            add_generation_prompt=chat_request.add_generation_prompt,
            continue_final_message=chat_request.continue_final_message,
            documents=getattr(chat_request, "documents", None),
            add_special_tokens=chat_request.add_special_tokens,
        )
        if not engine_prompts:
            return None
        return engine_prompts[0] if isinstance(engine_prompts[0], dict) else None
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.getLogger(__name__).debug(
            "Failed to render chat with vLLM-Omni; falling back",
            exc_info=True,
        )
        return None


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


def _chat_content_to_text(content: Any) -> str:
    """Extract plain text from OpenAI chat content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)


def _audio_additional_information(request: dict) -> dict[str, list[str]]:
    """Build optional speaker/language metadata for omni talker processors."""
    additional_information: dict[str, list[str]] = {}
    audio_options = (
        request.get("audio") if isinstance(request.get("audio"), dict) else {}
    )
    speaker = (
        request.get("voice")
        or request.get("speaker")
        or audio_options.get("voice")
        or audio_options.get("speaker")
    )
    if isinstance(speaker, str) and speaker.strip():
        additional_information["speaker"] = [speaker.strip().lower()]
    language = request.get("language") or audio_options.get("language")
    if isinstance(language, str) and language.strip():
        additional_information["language"] = [language.strip()]
    return additional_information


async def parse_omni_request(
    request: dict,
    output_modalities: list,
    default_video_fps: int = 16,
    tokenizer_getter=None,
    chat_preprocessor: Any | None = None,
    renderer: Any | None = None,
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
    parsed_request, request_type = parse_request_type(request, output_modalities)

    if request_type == RequestType.AUDIO_GENERATION:
        input_text = getattr(parsed_request, "input", request.get("input", ""))
        engine_inputs = OmniTextPrompt(prompt=input_text)
        additional_information = _audio_additional_information(request)
        if additional_information:
            engine_inputs["additional_information"] = additional_information
        return {
            "engine_inputs": engine_inputs,
            "original_prompt": dict(engine_inputs),
            "sampling_params_list": None,
        }

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
    text = _chat_content_to_text(
        next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            request.get("prompt", ""),
        )
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

    additional_information = _audio_additional_information(request)
    if messages:
        native_prompt = await preprocess_chat_with_omni(
            request,
            chat_preprocessor,
            renderer,
        )
        if native_prompt is not None:
            if additional_information:
                existing = native_prompt.setdefault("additional_information", {})
                if isinstance(existing, dict):
                    for key, value in additional_information.items():
                        existing.setdefault(key, value)
            return {
                "engine_inputs": native_prompt,
                "original_prompt": native_prompt,
                "sampling_params_list": None,
            }

    # Apply chat template when a tokenizer is available. Some Omni models,
    # including Qwen3-Omni, do not publish tokenizer.chat_template; those
    # models rely on vLLM-Omni's renderer path above.
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

    return {
        "engine_inputs": (
            OmniTextPrompt(prompt=text, additional_information=additional_information)
            if additional_information
            else text
        ),
        "original_prompt": {
            "prompt": text,
            **(
                {"additional_information": additional_information}
                if additional_information
                else {}
            ),
        },
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
