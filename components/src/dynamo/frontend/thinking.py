#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_THINKING_MODE_RUNTIME_KEY = "default_thinking_mode"
THINKING_CONTROL_KEYS = (
    "thinking",
    "enable_thinking",
    "thinking_mode",
    "reasoning_effort",
)

# Reasoning parsers whose generation mode follows the rendered prompt rather
# than the request's thinking flags, as in the Rust preprocessor (#17660).
GLM_REASONING_PARSERS = frozenset({"glm45", "glm47"})
_GLM_THINK_START = "<think>"
_GLM_THINK_END = "</think>"
# Enough trailing prompt tokens to contain either reasoning marker.
_GLM_PROMPT_TAIL_TOKENS = 16


def runtime_default_thinking_mode(runtime_config: dict[str, Any] | None) -> str | None:
    """Read deployment-level default thinking mode from model runtime metadata."""
    if not isinstance(runtime_config, dict):
        return None

    runtime_data = runtime_config.get("runtime_data")
    if not isinstance(runtime_data, dict):
        return None

    value = runtime_data.get(DEFAULT_THINKING_MODE_RUNTIME_KEY)
    return value if isinstance(value, str) and value else None


def apply_default_thinking_mode_to_template_kwargs(
    chat_template_kwargs: dict[str, Any],
    default_thinking_mode: str | None,
    *,
    request_has_root_thinking: bool = False,
) -> dict[str, Any]:
    """Merge deployment thinking default unless the request already controls it."""
    if default_thinking_mode is None:
        return chat_template_kwargs

    if request_has_root_thinking or any(
        key in chat_template_kwargs for key in THINKING_CONTROL_KEYS
    ):
        return chat_template_kwargs

    if default_thinking_mode not in ("enabled", "disabled"):
        logger.warning(
            "Ignoring invalid default_thinking_mode=%r; expected 'enabled' or 'disabled'",
            default_thinking_mode,
        )
        return chat_template_kwargs

    enabled = default_thinking_mode == "enabled"
    merged = dict(chat_template_kwargs)
    merged["thinking"] = enabled
    merged["enable_thinking"] = enabled
    merged["thinking_mode"] = default_thinking_mode
    return merged


def glm_prompt_reasoning_open(
    prompt_token_ids: Sequence[int],
    tokenizer: Any,
    *,
    add_generation_prompt: bool,
    continue_final_message: bool,
) -> bool | None:
    """Whether a rendered GLM generation prompt leaves reasoning open.

    ``True`` when the prompt (trailing whitespace ignored) ends with ``<think>``,
    ``False`` when it ends with ``</think>``, and ``None`` when the state is
    unknown: no generation prompt, a continued final message, or neither marker.
    Mirrors ``reasoning_closed_by_prompt`` for ``glm45``/``glm47`` in the Rust
    preprocessor. A GLM template may ignore a false thinking flag and still open
    ``<think>``, so a known state overrides the request flags; callers keep the
    request-based behavior when this returns ``None``.
    """
    if not add_generation_prompt or continue_final_message or not prompt_token_ids:
        return None
    tail = tokenizer.decode(
        list(prompt_token_ids[-_GLM_PROMPT_TAIL_TOKENS:]),
        skip_special_tokens=False,
    ).rstrip()
    if tail.endswith(_GLM_THINK_START):
        return True
    if tail.endswith(_GLM_THINK_END):
        return False
    return None
