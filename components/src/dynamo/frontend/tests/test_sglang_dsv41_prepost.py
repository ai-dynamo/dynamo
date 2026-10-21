# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 prompt rendering on the sglang chat-processor path.

V4.1 ships no chat template. SGLang renders it with ``encoding_dsv41``, which
differs from the V4 encoder and places images itself; these tests compare
Dynamo's prompt with that encoder's output.
"""

import json

import pytest

from dynamo.frontend.sglang_prepost import (
    _deepseek_chat_encoding,
    preprocess_chat_request,
)
from dynamo.frontend.utils import PreprocessError

encoding_dsv41 = pytest.importorskip("sglang.srt.entrypoints.openai.encoding_dsv41")

# Needs sglang packages (gpu_1 container) but allocates no GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]

IMAGE_TOKEN_ID = 129264
IMAGE_TOKEN = "<IMAGE>"
IMAGE_URL = "data:image/png;base64,aW1hZ2U="


class RecordingTokenizer:
    """No chat template; records the rendered prompt instead of tokenizing it."""

    chat_template = None

    def __init__(self, name_or_path: str = ""):
        self.name_or_path = name_or_path
        self.prompts: list[str] = []

    def encode(self, prompt: str) -> list[int]:
        self.prompts.append(prompt)
        return [1, 2, 3]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        assert token_id == IMAGE_TOKEN_ID
        return IMAGE_TOKEN


@pytest.fixture
def v41_model_dir(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "architectures": ["DeepseekV41ForCausalLM"],
                "image_token_id": IMAGE_TOKEN_ID,
            }
        )
    )
    return str(tmp_path)


def _render(request, tokenizer, **kwargs):
    preprocess_chat_request(
        request,
        tokenizer=tokenizer,
        tool_call_parser_name=kwargs.get("tool_call_parser_name"),
        reasoning_parser_name=kwargs.get("reasoning_parser_name"),
    )
    assert len(tokenizer.prompts) == 1
    return tokenizer.prompts[0]


@pytest.mark.parametrize(
    "request_model,tool_parser,reasoning_parser",
    [
        ("accounts/fireworks/models/deepseek-v4p1-flash-1017", None, None),
        ("deepseek-ai/DeepSeek-V4.1-Flash", None, None),
        ("served-name", "deepseekv41", None),
        ("served-name", None, "deepseek-v41"),
    ],
)
def test_v41_hints_select_the_v41_encoder(request_model, tool_parser, reasoning_parser):
    assert (
        _deepseek_chat_encoding(
            {"model": request_model},
            tokenizer=RecordingTokenizer(),
            tool_call_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
        )
        == "dsv41"
    )


def test_v41_config_selects_the_v41_encoder_without_hints(v41_model_dir):
    assert (
        _deepseek_chat_encoding(
            {"model": "served-name"},
            tokenizer=RecordingTokenizer(v41_model_dir),
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
        == "dsv41"
    )


def test_v4_hints_keep_the_v4_encoder():
    assert (
        _deepseek_chat_encoding(
            {"model": "deepseek-ai/DeepSeek-V4-Pro"},
            tokenizer=RecordingTokenizer(),
            tool_call_parser_name=None,
            reasoning_parser_name="deepseek-v4",
        )
        == "dsv4"
    )


def test_text_prompt_matches_encoding_dsv41(v41_model_dir):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ]
    prompt = _render(
        {"model": "served-name", "messages": messages},
        RecordingTokenizer(v41_model_dir),
    )

    expected = encoding_dsv41.encode_messages(
        messages,
        thinking_mode="chat",
        reasoning_effort=encoding_dsv41.DEFAULT_REASONING_EFFORT,
    )
    assert prompt == expected


def test_image_becomes_one_image_token_in_place(v41_model_dir):
    content = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
    ]
    prompt = _render(
        {"model": "served-name", "messages": [{"role": "user", "content": content}]},
        RecordingTokenizer(v41_model_dir),
    )

    expected, media = encoding_dsv41.encode_messages(
        [{"role": "user", "content": content}],
        thinking_mode="chat",
        reasoning_effort=encoding_dsv41.DEFAULT_REASONING_EFFORT,
        return_multi_modal_data=True,
    )
    assert media["images"] == [{"url": IMAGE_URL}]
    assert prompt == expected.replace(encoding_dsv41.IMAGE_PLACEHOLDER, IMAGE_TOKEN)
    assert prompt.count(IMAGE_TOKEN) == 1


def test_tools_render_only_client_fields_for_the_chosen_tool(v41_model_dir):
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
        {
            "type": "function",
            "function": {
                "parameters": {"type": "object"},
                "name": "get_time",
                "description": "Current time",
            },
        },
    ]
    request = {
        "model": "served-name",
        "messages": [{"role": "user", "content": "What time is it?"}],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "get_time"}},
    }
    prompt = _render(request, RecordingTokenizer(v41_model_dir))

    # OpenAI field order, no strict=false default, only the named tool.
    chosen = {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current time",
            "parameters": {"type": "object"},
        },
    }
    expected = encoding_dsv41.encode_messages(
        [
            {"role": "system", "content": "", "tools": [chosen]},
            {"role": "user", "content": "What time is it?"},
        ],
        thinking_mode="chat",
        reasoning_effort=encoding_dsv41.DEFAULT_REASONING_EFFORT,
    )
    assert prompt == expected


def test_reasoning_effort_and_thinking_reach_the_encoder(v41_model_dir):
    messages = [{"role": "user", "content": "Hi"}]
    prompt = _render(
        {
            "model": "served-name",
            "messages": messages,
            "reasoning_effort": "max",
            "chat_template_kwargs": {"thinking": True},
        },
        RecordingTokenizer(v41_model_dir),
    )

    expected = encoding_dsv41.encode_messages(
        messages, thinking_mode="thinking", reasoning_effort="max"
    )
    assert prompt == expected


def _tool_turn(results):
    calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": "screenshot", "arguments": "{}"},
        }
        for call_id in ("call_1", "call_2")
    ]
    return [
        {"role": "user", "content": "Look at the tool output."},
        {"role": "assistant", "content": "", "tool_calls": calls},
    ] + [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": [{"type": "image_url", "image_url": {"url": url}}],
        }
        for call_id, url in results
    ]


def test_tool_result_images_are_placed_in_call_order(v41_model_dir):
    request = {
        "model": "served-name",
        "messages": _tool_turn([("call_1", IMAGE_URL), ("call_2", "https://b/2.png")]),
    }
    prompt = _render(request, RecordingTokenizer(v41_model_dir))
    assert prompt.count(IMAGE_TOKEN) == 2


def test_image_without_a_url_is_rejected(v41_model_dir):
    request = {
        "model": "served-name",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {}, "uuid": "abcd"}],
            }
        ],
    }
    with pytest.raises(PreprocessError):
        _render(request, RecordingTokenizer(v41_model_dir))


def test_image_without_image_token_id_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    request = {
        "model": "served-name",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": IMAGE_URL}}],
            }
        ],
    }
    with pytest.raises(PreprocessError, match="not supported"):
        _render(request, RecordingTokenizer(str(tmp_path)))
