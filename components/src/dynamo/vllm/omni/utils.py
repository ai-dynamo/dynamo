# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the vLLM-Omni backend."""

import copy
import json
import logging
from pathlib import Path
from typing import Any, cast

from huggingface_hub import scan_cache_dir
from vllm.sampling_params import SamplingParams
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.entrypoints.stage_utils import shm_read_bytes
from vllm_omni.entrypoints.utils import load_stage_configs_from_yaml
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.common.utils.video_utils import compute_num_frames, parse_size

DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_VIDEO_SIZE = "832x480"
_STAGE_OVERRIDES_KEY = "__stage_overrides__"


def load_omni_stage_configs(model: str, stage_configs_path: str | None) -> list:
    """Load vLLM-Omni stage configs from legacy or deploy-format YAML."""
    if not stage_configs_path:
        return []

    try:
        from vllm_omni.engine.async_omni_engine import load_and_resolve_stage_configs
    except (AttributeError, ImportError, ModuleNotFoundError):
        logging.getLogger(__name__).debug(
            "Falling back to legacy vLLM-Omni stage config loader",
            exc_info=True,
        )
        return load_stage_configs_from_yaml(stage_configs_path)

    _, stage_configs = load_and_resolve_stage_configs(
        model,
        stage_configs_path,
        {},
        default_stage_cfg_factory=None,
    )
    return stage_configs


def stage_configs_use_async_chunk(stage_configs: list) -> bool:
    if not stage_configs:
        return False
    engine_args = getattr(stage_configs[0], "engine_args", None)
    return bool(getattr(engine_args, "async_chunk", False))


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


def _json_safe_prompt(prompt: Any) -> dict:
    """Keep only prompt fields that can cross the router as JSON."""
    if not isinstance(prompt, dict):
        return {"prompt": prompt}

    result: dict[str, Any] = {}
    for key in (
        "prompt",
        "prompt_token_ids",
        "additional_information",
        "modalities",
        "mm_processor_kwargs",
    ):
        if key not in prompt:
            continue
        value = copy.deepcopy(prompt[key])
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        result[key] = value
    return result or {"prompt": prompt.get("prompt", "")}


def _chat_sampling_overrides(request: Any) -> dict | None:
    """Return OpenAI sampling overrides for the comprehension stage."""
    sampling_fields = _openai_sampling_fields(request)
    explicit_fields = getattr(request, "model_fields_set", None)
    overrides: dict[str, Any] = {}

    for field in sampling_fields:
        if explicit_fields is not None:
            if field not in explicit_fields:
                continue
            value = getattr(request, field, None)
        elif isinstance(request, dict):
            if field not in request:
                continue
            value = request.get(field)
        else:
            continue
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        overrides[field] = value

    if explicit_fields is not None and "max_completion_tokens" in explicit_fields:
        max_completion_tokens = getattr(request, "max_completion_tokens", None)
        if max_completion_tokens is not None:
            overrides["max_tokens"] = max_completion_tokens
    elif isinstance(request, dict) and request.get("max_completion_tokens") is not None:
        overrides["max_tokens"] = request["max_completion_tokens"]

    if not overrides:
        return None
    return {_STAGE_OVERRIDES_KEY: {"0": overrides}}


def _openai_sampling_fields(request: Any) -> set[str]:
    model_fields = getattr(request.__class__, "model_fields", None)
    if model_fields is not None:
        return set(getattr(SamplingParams, "__struct_fields__", ())).intersection(
            model_fields
        )
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
    except ImportError:
        return set()

    return set(getattr(SamplingParams, "__struct_fields__", ())).intersection(
        ChatCompletionRequest.model_fields
    )


async def _render_chat_request(
    request: dict,
    renderer: Any,
    model_config: Any,
) -> tuple[dict, Any] | None:
    """Render chat through vLLM-Omni's OpenAI server preprocessing path."""
    messages = request.get("messages")
    if not messages:
        return None
    if renderer is None or model_config is None:
        raise ValueError("Cannot process chat without a vLLM renderer")

    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
    except ImportError as e:
        raise ValueError(
            "Cannot process chat without ChatCompletionRequest support"
        ) from e

    try:
        chat_request = ChatCompletionRequest.model_validate(
            _normalize_chat_request(request)
        )
        engine_prompt = await _render_chat_with_vllm(
            chat_request, renderer, model_config
        )
        _attach_omni_chat_metadata(engine_prompt, chat_request)
        return engine_prompt, chat_request
    except Exception as e:
        raise ValueError(f"Failed to render chat request: {e}") from e


async def _render_chat_with_vllm(
    chat_request: Any,
    renderer: Any,
    model_config: Any,
) -> dict:
    chat_params = chat_request.build_chat_params(None, "auto")
    tok_params = chat_request.build_tok_params(model_config)
    prompt_extras = {
        key: value
        for key in ("mm_processor_kwargs", "cache_salt")
        if (value := getattr(chat_request, key, None)) is not None
    }
    (_,), (engine_prompt,) = await renderer.render_chat_async(
        [chat_request.messages],
        chat_params,
        tok_params,
        prompt_extras=prompt_extras,
    )
    return engine_prompt


def _attach_omni_chat_metadata(prompt: Any, request: Any) -> None:
    if not isinstance(prompt, dict):
        return
    additional_information = prompt.setdefault("additional_information", {})
    if not isinstance(additional_information, dict):
        additional_information = {}
        prompt["additional_information"] = additional_information

    speaker = getattr(request, "speaker", None) or getattr(request, "voice", None)
    if isinstance(speaker, str) and speaker.strip():
        additional_information["speaker"] = [speaker.lower().strip()]
    elif isinstance(speaker, list) and speaker:
        additional_information["speaker"] = [
            item.lower().strip() if isinstance(item, str) else item for item in speaker
        ]

    language = getattr(request, "language", None)
    if isinstance(language, str) and language.strip():
        additional_information["language"] = [language.strip()]
    elif isinstance(language, list) and language:
        additional_information["language"] = language

    instructions = getattr(request, "instructions", None)
    if isinstance(instructions, str) and instructions.strip():
        additional_information["instruction"] = instructions.strip()


def _normalize_chat_request(request: dict) -> dict:
    chat_request = dict(request)
    if "voice" in chat_request and "speaker" not in chat_request:
        chat_request["speaker"] = chat_request["voice"]
    return chat_request


async def parse_omni_request(
    request: dict,
    output_modalities: list,
    default_video_fps: int = 16,
    tokenizer_getter=None,
    renderer=None,
    model_config=None,
) -> dict:
    """Parse a raw frontend request into engine_inputs, original_prompt, sampling_params_list.

    Args:
      tokenizer_getter: async callable returning a tokenizer (e.g. engine.get_tokenizer).
          Retained for compatibility with older call sites.
      renderer/model_config: vLLM renderer inputs.  When available, chat
          requests are rendered like native ``vllm serve --omni`` so stage 0
          receives tokenized multimodal/chat-template inputs.

    Returns:
      engine_inputs:        text prompt (str or OmniTextPrompt) for the stage 0 engine
      original_prompt:      rich prompt dict with geometry/params for processor functions
      sampling_params_list: raw user overrides dict (height/width/nvext) or None for chat
    """
    _, request_type = parse_request_type(request, output_modalities)

    if request_type == RequestType.AUDIO_GENERATION or (
        "input" in request and "messages" not in request
    ):
        text = request.get("input", "")
        chat_request = {
            **request,
            "messages": [{"role": "user", "content": text}],
            "modalities": request.get("modalities") or ["audio"],
        }
        rendered = None
        if renderer is not None and model_config is not None:
            rendered = await _render_chat_request(chat_request, renderer, model_config)
        if rendered is not None:
            engine_prompt, chat_request_model = rendered
            return {
                "engine_inputs": engine_prompt,
                "original_prompt": _json_safe_prompt(engine_prompt),
                "sampling_params_list": _chat_sampling_overrides(chat_request_model),
            }
        return {
            "engine_inputs": OmniTextPrompt(prompt=text),
            "original_prompt": {"prompt": text},
            "sampling_params_list": _chat_sampling_overrides(chat_request),
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
        }

    # Chat / text
    rendered = await _render_chat_request(request, renderer, model_config)
    if rendered is not None:
        engine_prompt, chat_request = rendered
        return {
            "engine_inputs": engine_prompt,
            "original_prompt": _json_safe_prompt(engine_prompt),
            "sampling_params_list": _chat_sampling_overrides(chat_request),
        }

    text = request.get("prompt", "")

    return {
        "engine_inputs": text,
        "original_prompt": {"prompt": text},
        "sampling_params_list": _chat_sampling_overrides(request),
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
    stage_overrides = None
    if overrides and _STAGE_OVERRIDES_KEY in overrides:
        stage_map = overrides.get(_STAGE_OVERRIDES_KEY) or {}
        stage_id = getattr(stage_config, "stage_id", 0)
        stage_overrides = stage_map.get(str(stage_id), stage_map.get(stage_id))
    else:
        stage_overrides = overrides

    stage_type = getattr(stage_config, "stage_type", "llm")
    if stage_type == "diffusion":
        diffusion_params = OmniDiffusionSamplingParams(**params_dict)
        if stage_overrides:
            for arg, value in stage_overrides.items():
                if hasattr(diffusion_params, arg):
                    setattr(diffusion_params, arg, value)
        return [diffusion_params]

    llm_params = SamplingParams(**params_dict)
    if stage_overrides:
        for arg, value in stage_overrides.items():
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
