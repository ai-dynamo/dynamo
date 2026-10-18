# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses history metadata at the Python chat-template boundary."""

import json
from collections.abc import Callable

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


def responses_history_template(
    chat_template: str | None,
    history_phases: list[str | None] | None,
    resolve_template: Callable[[], str | None],
) -> str | None:
    recipients = {"commentary": "commentary", "final_answer": "user"}
    phases = {
        str(index): recipients[phase]
        for index, phase in enumerate(history_phases or [])
        if phase in recipients
    }
    if not phases:
        return chat_template
    # Keep the renderer's processor/tokenizer/template-variant precedence.
    source = resolve_template()
    if source is None:
        return chat_template
    return _HISTORY_PREAMBLE.replace("PHASE_MAP", json.dumps(phases)) + source
