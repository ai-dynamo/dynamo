# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang reasoning-aware guided-decoding integration.

SGLang's grammar backend can defer a structured-output grammar until native
reasoning reaches its end marker.  That behavior is request-local: callers
must pass ``require_reasoning`` to ``Engine.async_generate`` when the chat
template selected reasoning for the request.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from dynamo.sglang._compat import filter_supported_async_generate_kwargs

REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY = "reasoning_aware_guided_decoding"
_BUILTIN_REASONING_GRAMMAR_BACKENDS = frozenset({"llguidance", "outlines", "xgrammar"})
_CAPABILITY_CACHE_ATTR = "_dynamo_reasoning_aware_guided_decoding"

_DYNAMO_TO_SGLANG_REASONING_PARSER = {
    "basic": "qwen3",
    "deepseek_r1": "deepseek-r1",
    "deepseek_v3": "deepseek-v3",
    "deepseek_v3_1": "deepseek-v3",
    "deepseek_v3_2": "deepseek-v3",
    "deepseek_v4": "deepseek-v4",
    "deepseek-v4": "deepseek-v4",
    "deepseekv4": "deepseek-v4",
    "gemma4": "gemma4",
    "gemma-4": "gemma4",
    "glm45": "glm45",
    "gpt_oss": "gpt-oss",
    "kimi": "kimi",
    "kimi_k25": "kimi_k2",
    "minimax_append_think": "minimax-append-think",
    "minimax_m3": "minimax-m3",
    "minimax-m3": "minimax-m3",
    "minimax_m3_nom": "minimax-m3",
    "minimax-m3-nom": "minimax-m3",
    "mistral": "mistral",
    "nemotron_deci": "qwen3",
    "nemotron_nano": "nemotron_3",
    "nemotron3": "nemotron_3",
    "nemotron_v3": "nemotron_3",
    "qwen3": "qwen3",
    "step3": "step3",
}


def reasoning_parsers_compatible(
    native_parser: str | None, dynamo_parser: str | None
) -> bool:
    """Whether native SGLang and Dynamo parse the same reasoning wire format."""
    if not native_parser or not dynamo_parser:
        return True
    expected = _DYNAMO_TO_SGLANG_REASONING_PARSER.get(dynamo_parser.strip().lower())
    return expected is not None and native_parser.strip().lower() == expected


def _reasoning_parser_class() -> Any | None:
    """Load SGLang's native parser without making older installs unusable."""
    try:
        from sglang.srt.parser.reasoning_parser import ReasoningParser
    except (ImportError, AttributeError):
        return None
    return ReasoningParser


def _native_reasoning_parser_name(engine: Any, server_args: Any | None = None) -> str:
    effective_server_args = getattr(engine, "server_args", None) or server_args
    if effective_server_args is None:
        tokenizer_manager = getattr(engine, "tokenizer_manager", None)
        effective_server_args = getattr(tokenizer_manager, "server_args", None)
    value = getattr(effective_server_args, "reasoning_parser", None)
    return value if isinstance(value, str) else ""


def _effective_server_args(engine: Any, server_args: Any | None = None) -> Any | None:
    effective_server_args = getattr(engine, "server_args", None) or server_args
    if effective_server_args is None:
        tokenizer_manager = getattr(engine, "tokenizer_manager", None)
        effective_server_args = getattr(tokenizer_manager, "server_args", None)
    return effective_server_args


def _request_reasoning_metadata(
    request: Mapping[str, Any],
) -> tuple[bool | None, dict[str, Any]]:
    """Read normalized or tokenizer-mode reasoning metadata.

    Token-input requests carry the frontend's normalized parser metadata.
    Text-input (``--use-sglang-tokenizer``) requests bypass that frontend
    preprocessing and retain the public chat-template dictionaries at the
    root.  Match :class:`InputParamManager` precedence there: legacy
    ``chat_template_args`` overrides ``chat_template_kwargs``.
    """
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, Mapping):
        extra_args = {}

    reasoning_ended = request.get("reasoning_ended")
    if reasoning_ended is None:
        reasoning_ended = extra_args.get("reasoning_ended")
    if not isinstance(reasoning_ended, bool):
        reasoning_ended = None

    parser_kwargs = request.get("reasoning_parser_kwargs")
    if parser_kwargs is None:
        parser_kwargs = extra_args.get("reasoning_parser_kwargs")
    if isinstance(parser_kwargs, Mapping):
        chat_template_kwargs = parser_kwargs.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, Mapping):
            return reasoning_ended, dict(chat_template_kwargs)

    root_kwargs: dict[str, Any] = {}
    chat_template_kwargs = request.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, Mapping):
        root_kwargs.update(chat_template_kwargs)
    chat_template_args = request.get("chat_template_args")
    if isinstance(chat_template_args, Mapping):
        root_kwargs.update(chat_template_args)
    return reasoning_ended, root_kwargs


@lru_cache(maxsize=32)
def _reasoning_default(parser_name: str) -> str | None:
    parser_class = _reasoning_parser_class()
    if parser_class is None:
        return None
    try:
        parser = parser_class(model_type=parser_name, stream_reasoning=False)
    except (TypeError, ValueError):
        return None
    value = getattr(getattr(parser, "detector", None), "reasoning_default", None)
    return value if isinstance(value, str) else None


def _requires_reasoning_from_mode(
    mode: str | None,
    chat_template_kwargs: Mapping[str, Any],
    reasoning_effort: Any,
) -> bool:
    if mode == "always":
        return True
    if mode == "mistral":
        return reasoning_effort is not None and reasoning_effort != "none"
    if mode in ("thinking", "enable_thinking"):
        return chat_template_kwargs.get(mode) is not False
    if mode in ("explicit_thinking", "explicit_enable_thinking"):
        toggle_param = mode.removeprefix("explicit_")
        return chat_template_kwargs.get(toggle_param) is True
    return False


def resolve_require_reasoning(engine: Any, request: Mapping[str, Any]) -> bool:
    """Mirror SGLang's OpenAI request-to-``require_reasoning`` decision.

    Explicit ``reasoning_ended`` metadata wins.  Otherwise use the detected
    chat-template toggle and fall back to the configured native parser's
    detector defaults, matching SGLang's ``OpenAIServingChat`` behavior.
    Unknown parser/config shapes fail closed.
    """
    parser_name = _native_reasoning_parser_name(engine)
    if not parser_name:
        return False

    reasoning_ended, chat_template_kwargs = _request_reasoning_metadata(request)
    if reasoning_ended is not None:
        return not reasoning_ended

    reasoning_effort = request.get("reasoning_effort")
    if reasoning_effort is None:
        reasoning_effort = chat_template_kwargs.get("reasoning_effort")
    if parser_name.lower() == "hunyuan":
        return reasoning_effort not in (None, "none", "no_think")

    template_manager = getattr(engine, "template_manager", None)
    config = getattr(template_manager, "reasoning_config", None)
    if config is None:
        return _requires_reasoning_from_mode(
            _reasoning_default(parser_name),
            chat_template_kwargs,
            reasoning_effort,
        )

    special_case = getattr(config, "special_case", None)
    if special_case in ("always", "mistral"):
        return _requires_reasoning_from_mode(
            special_case, chat_template_kwargs, reasoning_effort
        )

    toggle_param = getattr(config, "toggle_param", None)
    default_enabled = getattr(config, "default_enabled", None)
    if not isinstance(toggle_param, str) or not isinstance(default_enabled, bool):
        return False
    if default_enabled:
        return chat_template_kwargs.get(toggle_param) is not False
    return chat_template_kwargs.get(toggle_param) is True


def request_reasoning_kwargs(
    engine: Any, request: Mapping[str, Any]
) -> dict[str, bool]:
    """Build a version-compatible ``async_generate`` reasoning kwarg."""
    return filter_supported_async_generate_kwargs(
        engine, {"require_reasoning": resolve_require_reasoning(engine, request)}
    )


def _is_builtin_grammar_backend(backend_name: str) -> bool:
    """Reject custom backends, including overrides of built-in names."""
    if backend_name not in _BUILTIN_REASONING_GRAMMAR_BACKENDS:
        return False
    try:
        from sglang.srt.constrained.base_grammar_backend import GRAMMAR_BACKEND_REGISTRY
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )
    except (ImportError, AttributeError):
        return False

    del ReasonerGrammarBackend
    return backend_name not in GRAMMAR_BACKEND_REGISTRY


def _xgrammar_tokenizer_supported(tokenizer: Any, model_config: Any) -> bool:
    """Run the same tokenizer compatibility probe as SGLang's xgrammar backend."""
    try:
        if hasattr(tokenizer, "init_xgrammar"):
            tokenizer_info, _ = tokenizer.init_xgrammar()
            return tokenizer_info is not None

        from xgrammar import TokenizerInfo

        vocab_size = getattr(model_config, "vocab_size", None)
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            return False
        eos_token_ids = getattr(model_config, "hf_eos_token_id", None)
        if isinstance(eos_token_ids, int) and not isinstance(eos_token_ids, bool):
            eos_token_ids = [eos_token_ids]
        elif eos_token_ids:
            eos_token_ids = list(eos_token_ids)
        TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=vocab_size,
            stop_token_ids=eos_token_ids,
        )
    except Exception:
        return False
    return True


def _has_reasoner_grammar_prerequisites(
    server_args: Any,
    native_parser: str,
    tokenizer: Any,
    model_config: Any,
) -> bool:
    """Mirror the prerequisites for SGLang's ``ReasonerGrammarBackend`` wrapper."""
    backend_name = getattr(server_args, "grammar_backend", None)
    if not isinstance(backend_name, str):
        return False
    backend_name = backend_name.strip().lower()
    if not _is_builtin_grammar_backend(backend_name):
        return False

    if tokenizer is None or model_config is None:
        return False

    parser_class = _reasoning_parser_class()
    if parser_class is None:
        return False
    try:
        parser = parser_class(model_type=native_parser, stream_reasoning=False)
        think_end_token = parser.detector.think_end_token
        think_end_ids = tokenizer.encode(think_end_token, add_special_tokens=False)
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        not isinstance(think_end_ids, list)
        or len(think_end_ids) != 1
        or not isinstance(think_end_ids[0], int)
        or isinstance(think_end_ids[0], bool)
    ):
        return False

    if backend_name == "xgrammar" and not _xgrammar_tokenizer_supported(
        tokenizer, model_config
    ):
        return False
    return True


def supports_reasoning_aware_guided_decoding_components(
    *,
    native_parser: str | None,
    dynamo_parser: str | None,
    server_args: Any,
    tokenizer: Any,
    model_config: Any,
    async_generate: Any,
) -> bool:
    """Validate delayed grammar support without requiring a live engine.

    Multimodal encode workers own the public model card but delegate generation
    to a homogeneous SGLang backend. They already load the same tokenizer,
    model config, server args, and SGLang ``Engine`` API, which are precisely
    the inputs needed to prove the reasoner-grammar prerequisites.
    """
    if getattr(server_args, "skip_tokenizer_init", False):
        return False
    if getattr(server_args, "dllm_algorithm", None) is not None:
        return False
    if not isinstance(native_parser, str) or not native_parser.strip():
        return False
    if not dynamo_parser or not reasoning_parsers_compatible(
        native_parser, dynamo_parser
    ):
        return False
    if not _has_reasoner_grammar_prerequisites(
        server_args,
        native_parser,
        tokenizer,
        model_config,
    ):
        return False

    supported_kwargs = filter_supported_async_generate_kwargs(
        SimpleNamespace(async_generate=async_generate),
        {"require_reasoning": True},
    )
    return "require_reasoning" in supported_kwargs


def _supports_reasoning_aware_guided_decoding_uncached(
    engine: Any, server_args: Any, dynamo_args: Any
) -> bool:
    native_parser = _native_reasoning_parser_name(engine, server_args)
    dynamo_parser = getattr(dynamo_args, "dyn_reasoning_parser", None)
    effective_server_args = _effective_server_args(engine, server_args)
    if effective_server_args is None:
        return False
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    return supports_reasoning_aware_guided_decoding_components(
        native_parser=native_parser,
        dynamo_parser=dynamo_parser,
        server_args=effective_server_args,
        tokenizer=getattr(tokenizer_manager, "tokenizer", None),
        model_config=getattr(tokenizer_manager, "model_config", None),
        async_generate=getattr(engine, "async_generate", None),
    )


def supports_reasoning_aware_guided_decoding(
    engine: Any, server_args: Any, dynamo_args: Any
) -> bool:
    """Whether this worker can safely defer its grammar for reasoning.

    A custom/disabled grammar backend, unsupported xgrammar tokenizer, or
    missing reasoning end token must not advertise the capability. Cache the
    probe on the initialized engine because xgrammar builds tokenizer metadata
    and dynamic LoRA cards reuse the same result.
    """
    cached = getattr(engine, _CAPABILITY_CACHE_ATTR, None)
    if isinstance(cached, bool):
        return cached

    supported = _supports_reasoning_aware_guided_decoding_uncached(
        engine, server_args, dynamo_args
    )
    try:
        setattr(engine, _CAPABILITY_CACHE_ATTR, supported)
    except (AttributeError, TypeError):
        pass
    return supported


def reasoning_aware_guided_decoding_runtime_data(
    engine: Any, server_args: Any, dynamo_args: Any
) -> dict[str, bool] | None:
    if not supports_reasoning_aware_guided_decoding(engine, server_args, dynamo_args):
        return None
    return {REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY: True}


def publish_reasoning_aware_guided_decoding(
    runtime_config: Any, engine: Any, server_args: Any, dynamo_args: Any
) -> bool:
    """Publish the capability through ModelRuntimeConfig's JSON binding."""
    runtime_data = reasoning_aware_guided_decoding_runtime_data(
        engine, server_args, dynamo_args
    )
    if runtime_data is None:
        return False
    runtime_config.set_engine_specific(
        REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY, json.dumps(True)
    )
    return True


__all__ = [
    "REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY",
    "publish_reasoning_aware_guided_decoding",
    "reasoning_parsers_compatible",
    "reasoning_aware_guided_decoding_runtime_data",
    "request_reasoning_kwargs",
    "resolve_require_reasoning",
    "supports_reasoning_aware_guided_decoding",
    "supports_reasoning_aware_guided_decoding_components",
]
