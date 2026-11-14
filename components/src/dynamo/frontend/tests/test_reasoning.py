# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from functools import partial

import pytest
from transformers import PreTrainedTokenizerBase

from dynamo.frontend.reasoning import normalize_reasoning_content

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

_DIRECT = "{% for message in messages %}{{ message.reasoning_content }}{% endfor %}"
_STRING_ONLY = (
    "{% for message in messages %}"
    "{% if message.reasoning_content is string %}"
    "{{ message.reasoning_content }}{% endif %}{% endfor %}"
)
_CONCAT = (
    "{% for message in messages if message.role == 'assistant' %}"
    "{{ '<think>' + message.reasoning_content + '</think>' }}{% endfor %}"
)
_SEGMENTS = (
    "{% for message in messages %}"
    "{% for segment in message.reasoning_content %}{{ segment }}"
    "{% if not loop.last %}{{ message.tool_calls[loop.index0].function.name }}"
    "{% endif %}{% endfor %}{% endfor %}"
)
_TOOL = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "calculator",
        "parameters": {"type": "object", "properties": {}},
    },
}


def assistant(reasoning):
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "tool_calls": [
            {"type": "function", "function": {"name": "calc", "arguments": {}}}
        ],
    }


def template_renderer(template, **kwargs):
    tokenizer = PreTrainedTokenizerBase(chat_template=template)
    return partial(
        tokenizer.apply_chat_template, tokenize=False, return_dict=False, **kwargs
    )


@pytest.mark.parametrize("template", [_DIRECT, _STRING_ONLY])
def test_string_templates_render_segments_as_unescaped_text(template):
    segments = ["Check 'quotes' and \\slashes.\n", "", "Then add.", ""]
    messages = [assistant(segments)]
    render = template_renderer(template)
    normalize_reasoning_content(messages, render_template=render, tools=[_TOOL])
    rendered = render(messages)
    assert rendered == "Check 'quotes' and \\slashes.\n\nThen add."
    assert segments == ["Check 'quotes' and \\slashes.\n", "", "Then add.", ""]


def test_segment_aware_template_preserves_interleaving():
    messages = [assistant(["before", "after"])]
    original = copy.deepcopy(messages)
    render = template_renderer(_SEGMENTS)
    normalize_reasoning_content(messages, render_template=render, tools=[_TOOL])
    assert messages == original
    assert render(messages) == "beforecalcafter"


def test_probe_handles_templates_that_read_tool_schemas():
    template = (
        "{% for tool in tools %}{{ tool.function.description | tojson }}"
        "{% for name, _ in tool.function.parameters.properties.items() %}{{ name }}"
        "{% endfor %}{% endfor %}" + _STRING_ONLY
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "c",
                "parameters": {"type": "object", "properties": {"x": {}}},
            },
        }
    ]
    messages = [assistant(["before", "after"])]
    render = template_renderer(template, tools=tools)
    normalize_reasoning_content(messages, render_template=render, tools=tools)
    assert render(messages) == '"c"xbefore\nafter'


@pytest.mark.parametrize("tool_name", ["calc", "other"])
def test_probe_uses_request_tools_when_template_branches_on_tool_name(tool_name):
    template = (
        "{% for message in messages if message.role == 'assistant' %}"
        "{% if tools[0].function.name == 'calc' %}"
        "{{ message.reasoning_content }}{% else %}"
        "{% for segment in message.reasoning_content %}{{ segment }}"
        "{% if not loop.last %}{{ message.tool_calls[loop.index0].function.name }}"
        "{% endif %}{% endfor %}{% endif %}{% endfor %}"
    )
    tools = copy.deepcopy([_TOOL])
    tools[0]["function"]["name"] = tool_name
    original_tools = copy.deepcopy(tools)
    messages = [assistant(["before", "after"])]
    render = template_renderer(template, tools=tools)

    normalize_reasoning_content(messages, render_template=render, tools=tools)

    assert render(messages) == (
        "before\nafter" if tool_name == "calc" else "beforecalcafter"
    )
    assert tools == original_tools


@pytest.mark.parametrize("separately", [False, True])
def test_json_serialization_distinguishes_lists_from_segments(separately):
    template = (
        "{% for message in messages if message.role == 'assistant' %}"
        + (
            "{% if message.reasoning_content is string %}"
            "{{ message.reasoning_content | tojson }}{% else %}"
            "{% for segment in message.reasoning_content %}{{ segment | tojson }}"
            "{% if not loop.last %}{{ message.tool_calls[loop.index0].function.name }}"
            "{% endif %}{% endfor %}{% endif %}"
            if separately
            else "{{ message.reasoning_content | tojson }}"
        )
        + "{% endfor %}"
    )
    messages = [assistant(["before", "after"])]
    render = template_renderer(template)
    normalize_reasoning_content(messages, render_template=render, tools=[_TOOL])
    assert render(messages) == (
        '"before"calc"after"' if separately else '"before\\nafter"'
    )


def test_template_rejecting_arrays_receives_string():
    messages = [assistant(["before", "after", ""])]
    render = template_renderer(_CONCAT)
    normalize_reasoning_content(messages, render_template=render)
    assert render(messages) == "<think>before\nafter</think>"


@pytest.mark.parametrize("has_tools", [False, True])
def test_template_selection_matches_tool_use_default(has_tools):
    messages = [assistant(["before", "after"])]
    normalize_reasoning_content(
        messages,
        render_template=template_renderer({"default": _DIRECT, "tool_use": _SEGMENTS}),
        tools=[_TOOL] if has_tools else None,
    )
    assert messages[0]["reasoning_content"] == (
        ["before", "after"] if has_tools else "before\nafter"
    )


def test_native_encoder_normalization_is_idempotent():
    text = "Check 'quotes', \\slashes, and literal ['brackets'].\nThen add."
    messages = [assistant([text, ""])]
    normalize_reasoning_content(messages, string_only=True)
    normalized = copy.deepcopy(messages)
    rendered = "<think>{reasoning_content}</think>".format(**messages[0])
    assert rendered == f"<think>{text}</think>"

    normalize_reasoning_content(messages, string_only=True)
    assert messages == normalized
    assert "<think>{reasoning_content}</think>".format(**messages[0]) == rendered


def test_probe_uses_request_options_without_reusing_another_requests_result():
    template = (
        "{% if reasoning_effort != 'high' or options.modes[0] != 'thinking' %}"
        "{{ raise_exception('reasoning options required') }}{% endif %}"
        "{% if options.segmented %}"
        + _SEGMENTS
        + "{% else %}"
        + _DIRECT
        + "{% endif %}"
    )
    for segmented in [False, True]:
        options = {"modes": ["thinking"], "segmented": segmented}
        original_options = copy.deepcopy(options)
        render = template_renderer(template, reasoning_effort="high", options=options)
        messages = [assistant(["before", "after"])]
        normalize_reasoning_content(messages, render_template=render, tools=[_TOOL])
        assert messages[0]["reasoning_content"] == (
            ["before", "after"] if segmented else "before\nafter"
        )
        assert render(messages) == ("beforecalcafter" if segmented else "before\nafter")
        assert options == original_options


@pytest.mark.parametrize("reasoning", [None, "", "['literal', '']", []])
def test_existing_strings_and_empty_reasoning(reasoning):
    messages = [assistant(reasoning)]
    normalize_reasoning_content(messages, string_only=True)
    assert messages[0]["reasoning_content"] == (
        "" if isinstance(reasoning, list) else reasoning
    )


def test_non_assistant_and_unrecognized_templates_are_unchanged():
    messages = [
        {"role": "user", "content": "hello", "reasoning_content": ["literal"]},
        assistant(["before", "after"]),
    ]
    for template in [None, "{{ missing.required.value }}", "{{ messages[0].content }}"]:
        normalized = copy.deepcopy(messages)
        normalize_reasoning_content(
            normalized, render_template=template_renderer(template)
        )
        assert normalized == messages
