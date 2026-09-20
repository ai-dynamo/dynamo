# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Mapping
from typing import Any

from sglang.srt.parser.reasoning_parser import ReasoningParser

from dynamo.llm.exceptions import InvalidArgument

_U32_MAX = 2**32 - 1


def extract_thinking_budget(request: Mapping[str, Any]) -> int | None:
    """Read and validate a thinking budget from supported request shapes."""
    stop_conditions = request.get("stop_conditions")
    if (
        isinstance(stop_conditions, Mapping)
        and stop_conditions.get("max_thinking_tokens") is not None
    ):
        value = stop_conditions["max_thinking_tokens"]
    elif request.get("thinking_token_budget") is not None:
        value = request["thinking_token_budget"]
    else:
        nvext = request.get("nvext")
        value = nvext.get("max_thinking_tokens") if isinstance(nvext, Mapping) else None

    if value is None:
        return None
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _U32_MAX
    ):
        raise InvalidArgument(
            "thinking_token_budget must be an integer between 0 and 4294967295"
        )
    return value


def thinking_budget_requested(request: Mapping[str, Any]) -> bool:
    """Return whether the request contains a valid thinking budget."""
    return extract_thinking_budget(request) is not None


def _resolved_reasoning_parser(server_args: Any, engine: Any | None) -> str | None:
    reasoning_parser = getattr(server_args, "reasoning_parser", None)
    if reasoning_parser != "auto":
        return reasoning_parser

    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    config_value = getattr(tokenizer_manager, "config_value", None)
    if callable(config_value):
        resolved = config_value("reasoning_parser")
        return resolved if isinstance(resolved, str) and resolved != "auto" else None
    return None


def _validate_server_config(server_args: Any, engine: Any | None) -> None:
    if not getattr(server_args, "enable_strict_thinking", False):
        raise InvalidArgument(
            "thinking_token_budget requires SGLang --enable-strict-thinking"
        )

    reasoning_parser = _resolved_reasoning_parser(server_args, engine)
    if not reasoning_parser:
        raise InvalidArgument(
            "thinking_token_budget requires an SGLang --reasoning-parser"
        )

    if getattr(server_args, "skip_tokenizer_init", False):
        raise InvalidArgument(
            "thinking_token_budget is unavailable with --skip-tokenizer-init"
        )

    if getattr(server_args, "grammar_backend", None) == "none":
        raise InvalidArgument(
            "thinking_token_budget requires an enabled SGLang grammar backend"
        )

    if _token_filter_is_active(reasoning_parser):
        return

    raise InvalidArgument(
        f"SGLang cannot enforce per-request thinking_token_budget with reasoning "
        f"parser {reasoning_parser!r} in this configuration because its reasoning "
        "token filter is inactive"
    )


def _token_filter_is_active(reasoning_parser: str) -> bool:
    """Mirror SGLang 0.5.18/0.5.19 reasoner-grammar activation."""
    if int(os.getenv("SGLANG_MAX_THINK_TOKENS", "-1")) >= 0:
        return True

    try:
        parser = ReasoningParser(model_type=reasoning_parser)
    except (TypeError, ValueError) as exc:
        raise InvalidArgument(
            f"Unable to validate SGLang reasoning parser {reasoning_parser!r}: {exc}"
        ) from exc
    return bool(parser.detector.think_excluded_tokens)


def _reject_custom_logit_processor(
    request: Mapping[str, Any], sampling_params: Mapping[str, Any]
) -> None:
    sampling_options = request.get("sampling_options")
    custom_processor = sampling_params.get("custom_logit_processor")
    if custom_processor is None:
        custom_processor = request.get("custom_logit_processor")
    if custom_processor is None and isinstance(sampling_options, Mapping):
        custom_processor = sampling_options.get("custom_logit_processor")
    if custom_processor is not None:
        raise InvalidArgument(
            "thinking_token_budget cannot be combined with custom_logit_processor "
            "because SGLang applies custom processors after its reasoning-token mask"
        )


def apply_thinking_budget(
    request: Mapping[str, Any],
    sampling_params: Mapping[str, Any],
    server_args: Any,
    *,
    engine: Any | None = None,
) -> dict[str, Any]:
    """Translate Dynamo's canonical budget to SGLang sampling parameters."""
    result = dict(sampling_params)
    budget = extract_thinking_budget(request)
    if budget is None:
        custom_params = result.get("custom_params")
        if (
            isinstance(custom_params, Mapping)
            and custom_params.get("thinking_budget") is not None
        ):
            raise InvalidArgument(
                "SGLang sampling_params.custom_params.thinking_budget requires a "
                "canonical stop_conditions.max_thinking_tokens value"
            )
        return result

    stop_conditions = request.get("stop_conditions")
    if not (
        isinstance(stop_conditions, Mapping)
        and stop_conditions.get("max_thinking_tokens") is not None
    ):
        raise InvalidArgument(
            "thinking_token_budget requires Dynamo frontend preprocessing and is "
            "unavailable with --use-sglang-tokenizer"
        )

    if request.get("require_reasoning") is not True:
        raise InvalidArgument(
            "thinking_token_budget requires reasoning to be enabled for the request"
        )

    _validate_server_config(server_args, engine)
    _reject_custom_logit_processor(request, result)

    custom_params = result.get("custom_params")
    if custom_params is None:
        custom_params = {}
    elif not isinstance(custom_params, Mapping):
        raise InvalidArgument(
            "SGLang sampling_params.custom_params must be an object when "
            "thinking_token_budget is set"
        )

    result["custom_params"] = {**custom_params, "thinking_budget": budget}
    return result
