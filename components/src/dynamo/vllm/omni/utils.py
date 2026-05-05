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
    """Extract the text portions from OpenAI chat content."""
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
      input_preprocessor_getter: async callable returning vLLM's input preprocessor.
          When available, chat requests are rendered through vLLM's renderer so
          processor-owned chat templates and multimodal content are handled the
          same way as vLLM's OpenAI frontend.

    Returns:
      engine_inputs:        text prompt (str or OmniTextPrompt) for the stage 0 engine
      original_prompt:      rich prompt dict with geometry/params for processor functions
      sampling_params_list: raw user overrides dict (height/width/nvext) or None for chat
      output_modalities:    per-request modalities to pass to vLLM-Omni generation
    """
    _, request_type = parse_request_type(request, output_modalities)
    requested_modalities = _requested_output_modalities(
        request, request_type, output_modalities
    )

    if request_type == RequestType.AUDIO_GENERATION or "input" in request:
        text = request.get("input", "")
        engine_inputs: Any = OmniTextPrompt(prompt=text)
        templated = await _apply_chat_template(
            [{"role": "user", "content": text}],
            fallback=text,
            tokenizer_getter=tokenizer_getter,
        )
        if templated != text:
            engine_inputs = templated
        return {
            "engine_inputs": engine_inputs,
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
    text = await _apply_chat_template(
        messages,
        fallback=text,
        tokenizer_getter=tokenizer_getter,
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
    """Render chat messages through vLLM's renderer when the engine exposes it."""
    if not messages or input_preprocessor_getter is None:
        return None

    try:
        input_preprocessor = await input_preprocessor_getter()
    except Exception:
        logging.getLogger(__name__).debug(
            "vLLM input preprocessor unavailable, falling back to tokenizer template",
            exc_info=True,
        )
        return None

    renderer = getattr(input_preprocessor, "renderer", None)
    if renderer is None or not hasattr(renderer, "render_chat_async"):
        return None

    from vllm.renderers.params import ChatParams  # type: ignore[import-not-found]

    chat_template_kwargs = dict(request.get("chat_template_kwargs") or {})
    chat_template_kwargs.setdefault("tools", request.get("tools"))
    chat_template_kwargs.setdefault("documents", request.get("documents"))
    chat_template_kwargs.setdefault(
        "add_generation_prompt", request.get("add_generation_prompt", True)
    )
    chat_template_kwargs.setdefault(
        "continue_final_message", request.get("continue_final_message", False)
    )
    chat_template_kwargs.setdefault(
        "add_special_tokens", request.get("add_special_tokens", False)
    )
    chat_template_kwargs.setdefault("tokenize", False)

    chat_params = ChatParams(
        chat_template=request.get("chat_template"),
        chat_template_content_format=request.get(
            "chat_template_content_format", "auto"
        ),
        chat_template_kwargs=chat_template_kwargs,
        media_io_kwargs=request.get("media_io_kwargs"),
        mm_processor_kwargs=request.get("mm_processor_kwargs"),
    )
    prompt_extras = {
        key: value
        for key in ("mm_processor_kwargs", "cache_salt")
        if (value := request.get(key)) is not None
    }
    _, engine_prompts = await renderer.render_chat_async(
        [messages],
        chat_params,
        prompt_extras=prompt_extras or None,
    )
    if not engine_prompts:
        return None
    prompt = engine_prompts[0]
    if isinstance(prompt, dict):
        _copy_chat_audio_metadata(request, prompt)
        return prompt
    return None


def _copy_chat_audio_metadata(request: dict, prompt: dict) -> None:
    """Forward chat audio metadata in the shape vLLM-Omni processors expect."""
    audio = request.get("audio") if isinstance(request.get("audio"), dict) else {}
    speaker = request.get("speaker") or request.get("voice") or audio.get("voice")
    language = request.get("language") or audio.get("language")
    instruction = request.get("instructions") or audio.get("instructions")
    if not any(
        isinstance(value, str) and value.strip()
        for value in (speaker, language, instruction)
    ):
        return

    additional_information = prompt.get("additional_information")
    if not isinstance(additional_information, dict):
        additional_information = {}
        prompt["additional_information"] = additional_information

    if isinstance(speaker, str) and speaker.strip():
        additional_information["speaker"] = [speaker.lower().strip()]
    if isinstance(language, str) and language.strip():
        additional_information["language"] = [language.strip()]
    if isinstance(instruction, str) and instruction.strip():
        additional_information["instruction"] = instruction.strip()


async def _apply_chat_template(
    messages: list[dict], *, fallback: str, tokenizer_getter=None
) -> str:
    """Render messages with the model chat template when one is available."""
    if not messages or tokenizer_getter is None:
        return fallback
    try:
        tokenizer = await tokenizer_getter()
        if tokenizer is None:
            return fallback
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        logging.getLogger(__name__).debug("Chat template not available, using raw text")
        return fallback


def _requested_output_modalities(
    request: dict, request_type: RequestType, configured_output_modalities: list
) -> list[str]:
    """Return the modalities that this specific request asks the engine to emit."""
    raw_modalities = request.get("modalities") or request.get("output_modalities")
    if raw_modalities is not None:
        if isinstance(raw_modalities, str):
            tokens = [raw_modalities]
        elif isinstance(raw_modalities, list):
            tokens = [str(token) for token in raw_modalities]
        else:
            tokens = []
        normalized = normalize_output_modalities(tokens)
        if normalized:
            return normalized

    if request_type == RequestType.CHAT_COMPLETION:
        return ["text"]
    if request_type == RequestType.AUDIO_GENERATION:
        return ["audio"]
    if request_type == RequestType.IMAGE_GENERATION:
        return ["image"]
    if request_type == RequestType.VIDEO_GENERATION:
        return ["video"]
    return normalize_output_modalities(configured_output_modalities)


def stage_satisfies_request(
    stage_config: Any, request_type: RequestType, request: dict
) -> bool:
    """Return whether a stage's final output is the response this request needs."""
    if not getattr(stage_config, "final_output", False):
        return False

    final_output_type = str(getattr(stage_config, "final_output_type", "")).lower()
    if request_type == RequestType.CHAT_COMPLETION:
        if request_includes_output_modality(request, "audio"):
            return final_output_type == "audio"
        return final_output_type == "text"

    return final_output_type == {
        RequestType.AUDIO_GENERATION: "audio",
        RequestType.IMAGE_GENERATION: "image",
        RequestType.VIDEO_GENERATION: "video",
    }.get(request_type)


def request_includes_output_modality(request: dict, modality: str) -> bool:
    """Check chat-style output modality hints from OpenAI-compatible payloads."""
    raw_modalities = request.get("modalities") or request.get("output_modalities")
    if raw_modalities is None:
        return False
    if isinstance(raw_modalities, str):
        tokens = [raw_modalities]
    elif isinstance(raw_modalities, list):
        tokens = [str(token) for token in raw_modalities]
    else:
        return False
    return modality.lower() in normalize_output_modalities(tokens)


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
