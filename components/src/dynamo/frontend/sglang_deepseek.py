# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility with SGLang's recipe-based DeepSeek encoding and parsing."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import sglang.srt.parser.deepseek_recipe_renderer as recipe_renderer
except ModuleNotFoundError as exc:
    if exc.name != "sglang.srt.parser.deepseek_recipe_renderer":
        raise
    recipe_renderer = None

try:
    from sglang.srt.parser.deepseek_recipe_output import DeepSeekRecipeOutputParser
except ModuleNotFoundError as exc:
    if exc.name != "sglang.srt.parser.deepseek_recipe_output":
        raise
    DeepSeekRecipeOutputParser = None


def resolve_deepseek_spec(model_path: str) -> str | None:
    """Resolve the encoder from the downloaded checkpoint, independent of aliases."""
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    arch = architectures[0] if architectures else ""
    if config.get("model_type") == "deepseek_v41" or "DeepseekV41" in arch:
        return "dsv41"
    if config.get("model_type") == "deepseek_v4" or "DeepseekV4" in arch:
        return "dsv4"
    return None


class RecipeRequestError(ValueError):
    """A client request rejected by the recipe encoder."""


@lru_cache(maxsize=8)
def _encoder(tokenizer: Any, spec: str):
    return recipe_renderer.DeepSeekRecipe.from_hf_tokenizer(spec, tokenizer)


def supports_recipe_output(
    *,
    spec: str | None,
    reasoning_parser_name: str | None,
    tool_call_parser_name: str | None,
) -> bool:
    """Use native token parsing only for the checkpoint's configured family."""
    if DeepSeekRecipeOutputParser is None or spec is None:
        return False
    reasoning_name, tool_name = {
        "dsv4": ("deepseek-v4", "deepseekv4"),
        "dsv41": ("deepseek-v41", "deepseekv41"),
    }[spec]
    if not (
        reasoning_parser_name == reasoning_name or tool_call_parser_name == tool_name
    ):
        return False
    if reasoning_parser_name not in (
        None,
        reasoning_name,
    ) or tool_call_parser_name not in (
        None,
        tool_name,
    ):
        return False
    return True


def create_recipe_output_parser(
    request: dict[str, Any],
    *,
    tokenizer: Any,
    spec: str | None,
    reasoning_parser_name: str | None,
    tool_call_parser_name: str | None,
    parse_tool_calls: bool,
):
    if not supports_recipe_output(
        spec=spec,
        reasoning_parser_name=reasoning_parser_name,
        tool_call_parser_name=tool_call_parser_name,
    ):
        return None
    kwargs = (
        request.get("chat_template_kwargs") or request.get("chat_template_args") or {}
    )
    encoder = _encoder(tokenizer, spec)
    messages = request.get("messages") or []
    continued = (
        {**messages[-1], "content": messages[-1].get("content") or ""}
        if request.get("continue_final_message")
        and messages
        and messages[-1]["role"] == "assistant"
        else None
    )
    stop = request.get("stop") or []
    return DeepSeekRecipeOutputParser(
        encoder,
        thinking_stage=encoder.parse_start(
            thinking=bool(kwargs.get("thinking")), continued=continued
        ),
        parse_tool_calls=parse_tool_calls,
        parse_json_output=(request.get("response_format") or {}).get("type")
        in ("json_object", "json_schema"),
        separate_reasoning=reasoning_parser_name is not None
        and request.get("separate_reasoning", True),
        stream_reasoning=request.get("stream_reasoning", True),
        stop=[stop]
        if isinstance(stop, str)
        else [item for item in stop if isinstance(item, str)],
    )


def render_recipe_prompt(
    request: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None,
    spec: str,
    image_urls: list[str],
) -> list[int]:
    """Encode a prompt on SGLang builds that provide the recipe adapter."""
    kwargs = (
        request.get("chat_template_kwargs") or request.get("chat_template_args") or {}
    )
    effort = request.get("reasoning_effort")
    if effort is None:
        effort = kwargs.get("reasoning_effort")
    if request.get("continue_final_message") and kwargs.get("thinking"):
        raise RecipeRequestError(
            "continue_final_message with thinking is not supported "
            "with DeepSeek recipe encoding"
        )
    messages = [
        {**m, "reasoning_content": "\n".join(s for s in m["reasoning_content"] if s)}
        if isinstance(m.get("reasoning_content"), list)
        else m
        for m in messages
    ]
    # The caller already filters template tools; the host enforces tool choice.
    body = {
        "messages": messages,
        "thinking": bool(kwargs.get("thinking")),
        "tools": tools,
        "tool_choice": "auto" if tools else None,
        "response_format": request.get("response_format"),
        "continue_final_message": request.get("continue_final_message", False),
    }
    encoder = _encoder(tokenizer, spec)
    budget = None
    try:
        if spec == "dsv41" and isinstance(effort, (int, float)):
            budget = recipe_renderer.parse_effort_budget(effort)
        elif effort is not None:
            body["reasoning_effort"] = effort
        rendered_images: list[str] | None = [] if image_urls else None
        tokens = encoder.encode(body, effort_budget=budget, image_urls=rendered_images)
        if rendered_images is not None and rendered_images != image_urls:
            raise ValueError(
                "DeepSeek recipe image order differs from worker media order"
            )
    except ValueError as exc:
        raise RecipeRequestError(str(exc)) from exc
    return tokens
