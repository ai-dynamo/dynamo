# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for Responses replay after vLLM message normalization."""

from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from dynamo.frontend.responses_history import responses_history_template

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

TEMPLATE = (
    Path(__file__).resolve().parents[5]
    / "lib/llm/tests/data/muse_spark_open_deferred_tools.jinja"
).read_text()


@pytest.fixture
def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        bos_token="",
        chat_template=TEMPLATE,
    )


def render(tokenizer, messages, phases, template=None, tools=None):
    source = responses_history_template(
        template,
        phases,
        lambda: tokenizer.get_chat_template(chat_template=template, tools=tools),
    )
    return tokenizer.apply_chat_template(
        messages,
        chat_template=source,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        current_date="2026-10-06",
    )


@pytest.mark.parametrize("split_user", [False, True])
def test_commentary_and_final_replay_survive_normalization(tokenizer, split_user):
    # vLLM has already rebuilt the messages without recipient. Its multimodal
    # parser can split user messages and its template adapter can merge system
    # messages; neither should shift the assistant phase alignment.
    messages = [{"role": "system", "content": "Merged instructions."}]
    messages.extend(
        [
            {"role": "user", "content": "Part one."},
            {"role": "user", "content": "Part two."},
        ]
        if split_user
        else [{"role": "user", "content": "Part one. Part two."}]
    )
    messages.extend(
        [
            {"role": "assistant", "content": "Checking."},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Continue."},
        ]
    )
    before = deepcopy(messages)
    prompt = render(tokenizer, messages, ["commentary", "final_answer"])
    assert "assistant to=commentary<|message|>Checking.<|eom|>" in prompt
    assert "assistant to=user<|message|>Done.<|eot|>" in prompt
    assert messages == before


def test_tool_history_does_not_shift_later_phases(tokenizer):
    messages = [
        {"role": "assistant", "content": "Checking."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Found."},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Continue."},
    ]
    prompt = render(tokenizer, messages, ["commentary", None, "final_answer"])
    assert "assistant to=commentary<|message|>Checking.<|eom|>" in prompt
    assert "assistant to=lookup<|message|>" in prompt
    assert "assistant to=user<|message|>Done.<|eot|>" in prompt


@pytest.mark.parametrize("phases", [None, [], [None], ["unknown"]])
def test_unphased_chat_keeps_original_template(tokenizer, phases):
    def unexpected_resolution():
        pytest.fail("ordinary Chat Completions must keep the existing rendering path")

    assert responses_history_template(None, phases, unexpected_resolution) is None
    assert (
        responses_history_template(TEMPLATE, phases, unexpected_resolution) == TEMPLATE
    )


def test_processor_template_precedence_is_preserved(tokenizer):
    template = responses_history_template(
        None, ["commentary"], lambda: "PROCESSOR:" + TEMPLATE
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "Checking."}],
        chat_template=template,
        tokenize=False,
        current_date="2026-10-06",
    )
    assert prompt.startswith("PROCESSOR:")
    assert "assistant to=commentary<|message|>Checking.<|eom|>" in prompt


def test_template_variant_and_request_override_are_preserved(tokenizer):
    tokenizer.chat_template = {"default": TEMPLATE, "tool_use": "TOOLS: " + TEMPLATE}
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    messages = [{"role": "assistant", "content": "Checking."}]
    assert render(tokenizer, messages, ["commentary"], tools=tools).startswith("TOOLS:")
    assert render(
        tokenizer,
        messages,
        ["commentary"],
        template="OVERRIDE: " + TEMPLATE,
        tools=tools,
    ).startswith("OVERRIDE:")
    assert tokenizer.chat_template["default"] == TEMPLATE


def test_phase_metadata_is_request_local(tokenizer):
    messages = [{"role": "assistant", "content": "Checking."}]
    assert "to=commentary" in render(tokenizer, messages, ["commentary"])
    assert "assistant to=user<|message|>Checking.<|eot|>" in render(
        tokenizer, messages, ["final_answer"]
    )
    assert "assistant to=user<|message|>Checking.<|eot|>" in render(
        tokenizer, messages, None
    )


@pytest.mark.parametrize("eager", [False, True], ids=["generic", "spark"])
def test_selected_deferred_tool_is_eager_and_replay_names_match(tokenizer, eager):
    from dynamo.frontend.responses_history import responses_tools_request

    function = {
        "type": "function",
        "function": {
            "name": "fs.read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "fs.read", "arguments": {"path": "a"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "user", "content": "Read again"},
    ]
    request = {
        "responses_native_tools": {
            "tools": [function],
            "render_named_tool_eagerly": eager,
            "messages": messages,
            "tool_choice": {"type": "function", "function": {"name": "fs.read"}},
            "template_tools": [
                {
                    "type": "namespace",
                    "name": "fs",
                    "defer_loading": True,
                    "tools": [
                        {
                            "name": "read",
                            "parameters": function["function"]["parameters"],
                        }
                    ],
                }
            ],
        }
    }
    before = deepcopy(request)
    selected = responses_tools_request(request, TEMPLATE)
    prompt = tokenizer.apply_chat_template(
        selected["messages"],
        chat_template=TEMPLATE,
        tools=selected["responses_template_tools"],
        tokenize=False,
        current_date="2026-10-06",
    )
    if eager:
        assert selected["responses_template_tools"] == [function]
        assert '"name": "fs.read"' in prompt
        assert "[deferred:" not in prompt
    else:
        assert (
            selected["responses_template_tools"]
            == request["responses_native_tools"]["template_tools"]
        )
        assert '"has_deferred_functions": true' in prompt
    assert "assistant to=fs.read<|message|>" in prompt
    assert "tool fs.read<|message|>" in prompt
    assert request == before
