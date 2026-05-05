# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the vLLM-Omni backend."""

import copy
import json
import logging
from pathlib import Path
from typing import Any, cast

from huggingface_hub import scan_cache_dir
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.utils import get_max_tokens
from vllm.sampling_params import SamplingParams
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.engine.async_omni_engine import load_and_resolve_stage_configs
from vllm_omni.entrypoints.stage_utils import _to_dict, shm_read_bytes
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.common.utils.video_utils import compute_num_frames, parse_size

DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_VIDEO_SIZE = "832x480"
_STAGE_OVERRIDES_KEY = "__stage_overrides__"
_CHAT_TO_SAMPLING_FIELDS = {
    "max_completion_tokens": {"max_tokens"},
    "top_logprobs": {"logprobs"},
    "response_format": {"structured_outputs"},
    "stream": {"output_kind"},
    "vllm_xargs": {"extra_args"},
    "kv_transfer_params": {"extra_args"},
}
_JSON_UNSAFE_SAMPLING_FIELDS = {
    "output_kind",
    "structured_outputs",
    "output_text_buffer_length",
    "skip_clone",
    "_eos_token_id",
    "_all_stop_token_ids",
    "_bad_words_token_ids",
}


def load_omni_stage_configs(model: str, stage_configs_path: str | None) -> list:
    """Load resolved vLLM-Omni stage configs from deploy-format YAML."""
    if not stage_configs_path:
        return []

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


def _chat_sampling_overrides(
    chat_request: ChatCompletionRequest,
    default_sampling_params: Any = None,
    *,
    prompt_len: int = 0,
    max_model_len: int | None = None,
) -> dict | None:
    """Return vLLM OpenAI sampling overrides for the comprehension stage."""
    explicit_fields = set(chat_request.model_fields_set or ())
    if not explicit_fields:
        return None

    defaults = _to_dict(default_sampling_params or {})
    max_tokens = _chat_max_tokens(
        chat_request,
        defaults,
        prompt_len=prompt_len,
        max_model_len=max_model_len,
    )
    sampling_params = chat_request.to_sampling_params(max_tokens, defaults)
    overrides = _sampling_overrides_from_vllm(sampling_params, explicit_fields)

    if not overrides:
        return None
    return {_STAGE_OVERRIDES_KEY: {"0": overrides}}


def _chat_max_tokens(
    request: ChatCompletionRequest,
    defaults: dict,
    *,
    prompt_len: int,
    max_model_len: int | None,
) -> int:
    requested = getattr(request, "max_completion_tokens", None)
    if requested is None:
        requested = getattr(request, "max_tokens", None)

    if max_model_len is not None:
        return int(
            get_max_tokens(
                max_model_len,
                requested,
                prompt_len,
                defaults,
            )
        )

    fallback = requested if requested is not None else defaults.get("max_tokens")
    return int(fallback if fallback is not None else 16)


def _sampling_overrides_from_vllm(
    sampling_params: SamplingParams,
    explicit_fields: set[str],
) -> dict[str, Any]:
    sampling_fields = set(getattr(SamplingParams, "__struct_fields__", ()))
    fields = sampling_fields.intersection(explicit_fields)
    for request_field in explicit_fields:
        fields.update(_CHAT_TO_SAMPLING_FIELDS.get(request_field, ()))

    overrides: dict[str, Any] = {}
    for field in fields:
        if field in _JSON_UNSAFE_SAMPLING_FIELDS:
            continue
        value = getattr(sampling_params, field, None)
        if value is None or (isinstance(value, list) and not value):
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        overrides[field] = value
    return overrides


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
    chat_request = _strip_none_values(request)
    if "voice" in chat_request and "speaker" not in chat_request:
        chat_request["speaker"] = chat_request["voice"]
    return chat_request


def _strip_none_values(value: Any) -> Any:
    """Drop JSON nulls before vLLM validates OpenAI chat payloads.

    The Dynamo frontend materializes absent optional fields as ``None`` in some
    multimodal content parts (for example ``image_url.detail``). vLLM's OpenAI
    protocol expects those optional fields to be omitted, not set to null.
    """
    if isinstance(value, dict):
        return {k: _strip_none_values(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none_values(item) for item in value]
    return value


async def parse_omni_request(
    request: dict,
    output_modalities: list,
    default_video_fps: int = 16,
    renderer=None,
    model_config=None,
    default_sampling_params: Any = None,
) -> dict:
    """Convert a Dynamo frontend request into the inputs expected by AsyncOmni.

    Dynamo receives several OpenAI-style request shapes at the same router
    endpoint, but the stage-0 vLLM-Omni engine does not consume those raw
    payloads directly.  This function normalizes each public request shape into
    the three values the disaggregated stage worker needs:

    * ``engine_inputs``: the object passed to ``AsyncOmni.generate``.  For chat,
      this must be the vLLM-rendered prompt containing token ids and multimodal
      processor data, not raw ``messages``.
    * ``original_prompt``: a JSON-safe prompt copy that can cross Dynamo's
      router/stage boundary and still provide metadata needed by vLLM-Omni stage
      processor functions.
    * ``sampling_params_list``: request-level overrides to merge with the
      per-stage YAML defaults.  Chat overrides are scoped to stage 0 so user
      OpenAI sampling options do not overwrite talker/code2wav defaults.

    The branch structure mirrors the request families:

    * Audio generation uses ``input`` rather than ``messages``.  When a renderer
      is available we wrap it as a single-user chat request so vLLM applies the
      model's normal chat/audio template.
    * Image/video generation are diffusion-style requests.  They do not go
      through chat rendering, and their geometry/nvext fields become generation
      sampling params.
    * Chat/text requests are rendered through vLLM's OpenAI renderer.  Passing
      raw chat JSON to vLLM's ``InputProcessor`` skips chat-template and
      multimodal preprocessing and can produce errors such as "Token prompt
      should be a list of integers".

    Args:
      renderer/model_config: vLLM renderer inputs.  Chat requests are rendered
          like native ``vllm serve --omni`` so stage 0
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
        engine_prompt, chat_request_model = await _render_chat_request(
            chat_request, renderer, model_config
        )
        return {
            "engine_inputs": engine_prompt,
            "original_prompt": _json_safe_prompt(engine_prompt),
            "sampling_params_list": _chat_sampling_overrides(
                chat_request_model,
                default_sampling_params,
                prompt_len=_prompt_len(engine_prompt),
                max_model_len=getattr(model_config, "max_model_len", None),
            ),
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
            "sampling_params_list": _chat_sampling_overrides(
                chat_request,
                default_sampling_params,
                prompt_len=_prompt_len(engine_prompt),
                max_model_len=getattr(model_config, "max_model_len", None),
            ),
        }

    text = request.get("prompt", "")

    return {
        "engine_inputs": text,
        "original_prompt": {"prompt": text},
        "sampling_params_list": None,
    }


def _build_sampling_params(stage_config: Any, overrides: dict | None) -> list | None:
    """Construct typed sampling params from YAML default_sampling_params."""
    defaults = getattr(stage_config, "default_sampling_params", None)
    if not defaults:
        return None

    params_dict = cast(dict[str, Any], _to_dict(defaults))
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


def _prompt_len(prompt: Any) -> int:
    if isinstance(prompt, dict):
        token_ids = prompt.get("prompt_token_ids")
    else:
        token_ids = getattr(prompt, "prompt_token_ids", None)
    return len(token_ids or [])


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
