# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the vLLM-Omni backend."""

import json
import logging
from pathlib import Path
from typing import Any, cast

from huggingface_hub import scan_cache_dir
from vllm.sampling_params import SamplingParams
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.entrypoints.stage_utils import shm_read_bytes
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.utils.output_modalities import (
    RequestType,
    normalize_output_modalities,
    parse_request_type,
)
from dynamo.common.utils.video_utils import compute_num_frames, parse_size

DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_VIDEO_SIZE = "832x480"


def shm_deserialize(shm_meta: dict) -> Any:
    """Read and deserialize an OmniRequestOutput from shared memory."""
    return OmniSerializer.deserialize(shm_read_bytes(shm_meta))


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


async def parse_omni_request(
    request: dict,
    output_modalities: list,
    default_video_fps: int = 16,
    tokenizer_getter=None,
    input_preprocessor_getter=None,
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
    requested_modalities = _requested_output_modalities(
        request, request_type, output_modalities
    )

    if request_type == RequestType.AUDIO_GENERATION or "input" in request:
        text = request.get("input", "")
        return {
            "engine_inputs": OmniTextPrompt(prompt=text),
            "original_prompt": {"prompt": text},
            "sampling_params_list": None,
            "output_modalities": requested_modalities,
            "request_type": request_type,
        }

    if request_type in (RequestType.VIDEO_GENERATION, RequestType.IMAGE_GENERATION):
        is_video = request_type == RequestType.VIDEO_GENERATION
        nvext = request.get("nvext") or {}
        default_size = DEFAULT_VIDEO_SIZE if is_video else DEFAULT_IMAGE_SIZE
        size_kwargs = {} if is_video else {"default_w": 1024, "default_h": 1024}
        width, height = parse_size(request.get("size", default_size), **size_kwargs)
        sp: dict = {"height": height, "width": width, **nvext}
        if is_video:
            sp["num_frames"] = compute_num_frames(
                num_frames=nvext.get("num_frames"),
                fps=nvext.get("fps"),
                default_fps=default_video_fps,
            )
        return {
            "engine_inputs": OmniTextPrompt(prompt=request.get("prompt", "")),
            "original_prompt": build_original_prompt(request, nvext, height, width),
            "sampling_params_list": sp,
            "output_modalities": requested_modalities,
            "request_type": request_type,
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

    rendered_prompt = await _render_chat_engine_prompt(
        messages,
        request,
        input_preprocessor_getter=input_preprocessor_getter,
    )
    if rendered_prompt is not None:
        return {
            "engine_inputs": rendered_prompt,
            "original_prompt": rendered_prompt,
            "sampling_params_list": None,
            "output_modalities": requested_modalities,
            "request_type": request_type,
        }

    # Apply chat template when a tokenizer is available.  The native
    # OpenAI API server applies the template before the engine sees it;
    # without it the thinker receives bare text instead of the full
    # chat-formatted prompt.
    if messages and tokenizer_getter is not None:
        try:
            tokenizer = await tokenizer_getter()
            if tokenizer is not None:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            logging.getLogger(__name__).debug(
                "Chat template not available, using raw text"
            )

    return {
        "engine_inputs": text,
        "original_prompt": {"prompt": text},
        "sampling_params_list": None,
        "output_modalities": requested_modalities,
        "request_type": request_type,
    }


async def _render_chat_engine_prompt(
    messages: list[dict],
    request: dict,
    *,
    input_preprocessor_getter=None,
) -> dict | None:
    if not messages or input_preprocessor_getter is None:
        return None

    try:
        input_preprocessor = await input_preprocessor_getter()
    except Exception:
        logging.getLogger(__name__).debug("vLLM input preprocessor unavailable")
        return None

    renderer = getattr(input_preprocessor, "renderer", None)
    if renderer is None or not hasattr(renderer, "render_chat_async"):
        return None

    from vllm.entrypoints.openai.chat_completion.protocol import (  # type: ignore[import-not-found]
        ChatCompletionRequest,
    )

    chat_request = ChatCompletionRequest.model_validate(request)
    chat_params = chat_request.build_chat_params(None, "auto").with_defaults(
        {"tools": chat_request.tools, "tokenize": False},
        default_mm_processor_kwargs=request.get("mm_processor_kwargs"),
    )

    _, engine_prompts = await renderer.render_chat_async(
        [chat_request.messages],
        chat_params,
    )
    if not engine_prompts:
        return None
    prompt = engine_prompts[0]
    if isinstance(prompt, dict):
        _copy_chat_audio_metadata(request, prompt)
        return prompt
    return None


def _copy_chat_audio_metadata(request: dict, prompt: dict) -> None:
    audio = request.get("audio") if isinstance(request.get("audio"), dict) else {}
    speaker = request.get("speaker") or request.get("voice") or audio.get("voice")
    language = request.get("language") or audio.get("language")
    instruction = request.get("instructions") or audio.get("instructions")
    values = {
        "speaker": speaker.lower().strip() if isinstance(speaker, str) else "",
        "language": language.strip() if isinstance(language, str) else "",
        "instruction": instruction.strip() if isinstance(instruction, str) else "",
    }
    if not any(values.values()):
        return

    additional_information = prompt.get("additional_information")
    if not isinstance(additional_information, dict):
        additional_information = {}
        prompt["additional_information"] = additional_information

    if values["speaker"]:
        additional_information["speaker"] = [values["speaker"]]
    if values["language"]:
        additional_information["language"] = [values["language"]]
    if values["instruction"]:
        additional_information["instruction"] = values["instruction"]


def _requested_output_modalities(
    request: dict, request_type: RequestType, configured_output_modalities: list
) -> list[str]:
    if request_modalities := _modalities_from_request(request):
        return request_modalities

    return {
        RequestType.CHAT_COMPLETION: ["text"],
        RequestType.AUDIO_GENERATION: ["audio"],
        RequestType.IMAGE_GENERATION: ["image"],
        RequestType.VIDEO_GENERATION: ["video"],
    }.get(request_type, normalize_output_modalities(configured_output_modalities))


def stage_satisfies_request(
    stage_config: Any, request_type: RequestType, request: dict
) -> bool:
    terminal_modality = _terminal_output_modality(request_type, request)
    return bool(
        terminal_modality and stage_output_requested(stage_config, [terminal_modality])
    )


def stage_output_requested(
    stage_config: Any, request_output_modalities: list[str] | None
) -> bool:
    if not getattr(stage_config, "final_output", False):
        return False
    final_output_type = str(getattr(stage_config, "final_output_type", "")).lower()
    return bool(
        final_output_type and final_output_type in (request_output_modalities or [])
    )


def _modalities_from_request(request: dict) -> list[str]:
    raw_modalities = request.get("modalities") or request.get("output_modalities")
    if isinstance(raw_modalities, str):
        return normalize_output_modalities([raw_modalities])
    if isinstance(raw_modalities, list):
        return normalize_output_modalities(raw_modalities)
    return []


def _terminal_output_modality(request_type: RequestType, request: dict) -> str | None:
    if request_type == RequestType.CHAT_COMPLETION:
        requested = _modalities_from_request(request)
        return next((m for m in ("audio", "video", "image") if m in requested), "text")
    return {
        RequestType.AUDIO_GENERATION: "audio",
        RequestType.IMAGE_GENERATION: "image",
        RequestType.VIDEO_GENERATION: "video",
    }.get(request_type)


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
        return coerce_param_message_types([diffusion_params], is_streaming=False)

    llm_params = SamplingParams(**params_dict)
    if overrides:
        for arg, value in overrides.items():
            if hasattr(llm_params, arg):
                setattr(llm_params, arg, value)
    return coerce_param_message_types([llm_params], is_streaming=False)


def ensure_dummy_tokenizer_for_tts(model: str) -> list[Path]:
    """Create a minimal tokenizer.json for TTS models that lack one.

    Audio/TTS models (e.g., Qwen3-TTS) use a custom speech tokenizer and don't
    ship the standard tokenizer.json expected by the Rust ModelDeploymentCard
    loader. This writes a placeholder so register_model doesn't fail.

    Returns the list of created dummy paths so the caller can delete them
    after registration (otherwise the fake tokenizer poisons vLLM-Omni's
    inference-time AutoTokenizer.from_pretrained call).

    This is a short-term workaround. The long-term fix is making TokenizerKind
    optional in ModelDeploymentCard::from_repo_checkout().
    """
    created: list[Path] = []
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == model:
            for revision in repo.revisions:
                tokenizer_path = Path(revision.snapshot_path) / "tokenizer.json"
                if not tokenizer_path.exists():
                    logging.warning(
                        "TTS model %s has no tokenizer.json; "
                        "creating a minimal placeholder at %s",
                        model,
                        tokenizer_path,
                    )
                    minimal_tokenizer = {
                        "version": "1.0",
                        "model": {"type": "BPE", "vocab": {}, "merges": []},
                    }
                    tokenizer_path.write_text(json.dumps(minimal_tokenizer))
                    created.append(tokenizer_path)
            return created
    return created


def cleanup_dummy_tokenizer_for_tts(paths: list[Path]):
    """Remove dummy tokenizer.json files created by ensure_dummy_tokenizer_for_tts.

    Must be called after register_model() completes so the fake tokenizer
    doesn't interfere with vLLM-Omni's inference-time tokenizer loading
    (AutoTokenizer.from_pretrained picks up our stub and crashes).
    """
    for path in paths:
        try:
            path.unlink(missing_ok=True)
            logging.info("Removed dummy tokenizer placeholder: %s", path)
        except OSError as e:
            logging.warning("Failed to remove dummy tokenizer %s: %s", path, e)
