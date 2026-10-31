# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from transformers import PreTrainedTokenizerBase

from dynamo.frontend import prepost

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_DIRECT = (
    "{% for m in messages if m.role == 'assistant' %}"
    "<think>{{ m.reasoning_content }}</think>{% endfor %}"
)
_SEGMENTS = (
    "{% for m in messages if m.role == 'assistant' %}"
    "{% for segment in m.reasoning_content %}{{ segment }}"
    "{% if not loop.last %}{{ m.tool_calls[loop.index0].function.name }}"
    "{% endif %}{% endfor %}{% endfor %}"
)
_JSON = (
    "{% for m in messages if m.role == 'assistant' %}"
    "{{ m.reasoning_content | tojson }}{% endfor %}"
)


def request(reasoning, with_tools):
    value = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "calculate"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        ],
    }
    if with_tools:
        value["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "calculator",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    return value


class CapturingRenderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.messages = None
        self.prompt = None

    async def render_messages_async(self, messages, params):
        self.messages = copy.deepcopy(messages)
        rendered_messages = [
            {**message, "reasoning_content": message["reasoning"]}
            if message.get("role") == "assistant" and "reasoning" in message
            else message
            for message in messages
        ]
        kwargs = {
            **params.chat_template_kwargs,
            "chat_template": params.chat_template,
            "tokenize": False,
            "return_dict": False,
        }
        self.prompt = self.tokenizer.apply_chat_template(rendered_messages, **kwargs)
        return rendered_messages, {"prompt_token_ids": [1]}


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize(
    "template,reasoning,expected",
    [
        (_DIRECT, ["first", "second"], "<think>first\nsecond</think>"),
        (_DIRECT, "first", "<think>first</think>"),
        (_SEGMENTS, ["first", "second"], "firstcalcsecond"),
        (_JSON, ["first", "second"], '"first\\nsecond"'),
    ],
)
async def test_preprocess_preserves_reasoning(
    with_tools, template, reasoning, expected
):
    tokenizer = PreTrainedTokenizerBase(chat_template=template)
    renderer = CapturingRenderer(tokenizer)
    value = request(reasoning, with_tools)
    original = copy.deepcopy(value)

    await prepost.preprocess_chat_request(
        value,
        tokenizer=tokenizer,
        renderer=renderer,
        tool_parser_class=None,
        reasoning_parser_class=None,
    )

    assert renderer.prompt == expected
    assert value == original
    expected_reasoning = (
        reasoning
        if isinstance(reasoning, str) or template == _SEGMENTS
        else "first\nsecond"
    )
    assert renderer.messages[1]["reasoning"] == expected_reasoning


@pytest.mark.asyncio
async def test_explicit_reasoning_takes_precedence():
    tokenizer = PreTrainedTokenizerBase(chat_template=_DIRECT)
    renderer = CapturingRenderer(tokenizer)
    value = request("old", with_tools=True)
    value["messages"][1]["reasoning"] = "new"
    original = copy.deepcopy(value)

    await prepost.preprocess_chat_request(
        value,
        tokenizer=tokenizer,
        renderer=renderer,
        tool_parser_class=None,
        reasoning_parser_class=None,
    )

    assert renderer.prompt == "<think>new</think>"
    assert value == original


def test_reasoning_with_iterator_tool_calls():
    value = request(["first", "second"], with_tools=True)
    calls = value["messages"][1]["tool_calls"]
    value["messages"][1]["tool_calls"] = iter(calls)

    validated = prepost._validate_chat_completion_request(value)

    assert validated.messages[1]["tool_calls"] == calls
    assert validated.messages[1]["reasoning"] == ["first", "second"]
