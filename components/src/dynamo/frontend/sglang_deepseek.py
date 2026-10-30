# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility with SGLang's recipe-based DeepSeek prompt encoder."""

from functools import lru_cache
from typing import Any

try:
    import sglang.srt.parser.deepseek_recipe_renderer as recipe_renderer
except ModuleNotFoundError as exc:
    if exc.name != "sglang.srt.parser.deepseek_recipe_renderer":
        raise
    recipe_renderer = None


class RecipeRequestError(ValueError):
    """A client request rejected by the recipe encoder."""


@lru_cache(maxsize=8)
def _encoder(tokenizer: Any, spec: str):
    # SGLang renamed RecipeEncoder when adding recipe output parsing.
    encoder_type = getattr(recipe_renderer, "DeepSeekRecipe", None)
    if encoder_type is None:
        encoder_type = recipe_renderer.RecipeEncoder
    return encoder_type.from_hf_tokenizer(spec, tokenizer)


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
    # The caller already filters template tools; the host enforces tool choice.
    body = {
        "messages": messages,
        "thinking": bool(kwargs.get("thinking")),
        "tools": tools,
        "tool_choice": "auto" if tools else None,
        "response_format": request.get("response_format"),
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
