# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses metadata at the Python chat-template boundary."""

import json
from collections.abc import Callable
from functools import lru_cache

from jinja2 import Environment

# vLLM rebuilds the conversation without unknown fields such as recipient.
# Restore them after that normalization, immediately before the original Jinja
# source runs. No shared tokenizer/renderer state is changed for this request.
_HISTORY_PREAMBLE = """{%- set _dynamo_phases = PHASE_MAP -%}
{%- set _dynamo_history = namespace(messages=[], assistant_index=0) -%}
{%- for _dynamo_message in messages -%}
    {%- if _dynamo_message.role == 'assistant' -%}
        {%- set _dynamo_recipient = _dynamo_phases.get(_dynamo_history.assistant_index | string) -%}
        {%- if _dynamo_recipient -%}
            {%- set _dynamo_message = dict(_dynamo_message, recipient=_dynamo_recipient) -%}
        {%- endif -%}
        {%- set _dynamo_history.assistant_index = _dynamo_history.assistant_index + 1 -%}
    {%- endif -%}
    {%- set _dynamo_history.messages = _dynamo_history.messages + [_dynamo_message] -%}
{%- endfor -%}
{%- set messages = _dynamo_history.messages -%}
"""


def responses_history_recipients(
    history_phases: list[str | None] | None,
) -> dict[str, str]:
    recipients = {"commentary": "commentary", "final_answer": "user"}
    return {
        str(index): recipients[phase]
        for index, phase in enumerate(history_phases or [])
        if phase in recipients
    }


def responses_history_template(
    chat_template: str | None,
    history_phases: list[str | None] | None,
    resolve_template: Callable[[], str | None],
) -> str | None:
    phases = responses_history_recipients(history_phases)
    if not phases:
        return chat_template
    # Keep the renderer's processor/tokenizer/template-variant precedence.
    source = resolve_template()
    if source is None:
        return chat_template
    return _HISTORY_PREAMBLE.replace("PHASE_MAP", json.dumps(phases)) + source


@lru_cache(maxsize=128)
def _supports_native_response_tools(source: str) -> bool:
    # Match the Rust renderer's executable-Jinja capability marker. Lexing
    # excludes prose, quoted strings, comments and raw blocks, and accepts
    # templates with the transformers-specific generation extension.
    return any(
        kind == "name" and value == "deferred_render_mode"
        for _, kind, value in Environment().lex(source)
    )


def responses_tools_request(request: dict, template_source: str | None) -> dict:
    """Select renderer-only Responses metadata after resolving the template."""
    native = request.get("responses_native_tools")
    if (
        not native
        or not template_source
        or not _supports_native_response_tools(template_source)
    ):
        return request
    # The engine still sees only function tools. Switch their names, the
    # selected choice and replay together so native namespaces cannot drift.
    selected = {
        **request,
        **{key: native[key] for key in ("tools", "messages", "tool_choice")},
    }
    selected["responses_template_tools"] = native["template_tools"]
    choice = selected.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = choice["function"]["name"]
        # A selected tool must expose its schema even when advertised deferred.
        selected["responses_template_tools"] = [
            tool for tool in native["tools"] if tool["function"]["name"] == name
        ]
    return selected
