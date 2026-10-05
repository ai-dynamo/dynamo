#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Checkpoint token metadata must reach SGLang's public Hunyuan parsers."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest
from _routed_engine_fakes import FakeRoutedEngine
from sglang.srt.function_call.function_call_parser import FunctionCallParser

import dynamo.frontend.sglang_processor as processor_module
from dynamo.frontend.sglang_prepost import preprocess_chat_request
from dynamo.frontend.sglang_processor import SglangProcessor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class HunyuanTokenizer:
    """Synthetic vocab with public Hunyuan's checkpoint-specific delimiters."""

    chat_template = ""

    def get_vocab(self):
        names = ("think", "tool_calls", "tool_call", "tool_sep", "arg_key", "arg_value")
        return {f"<{name}:opensource>": i for i, name in enumerate(names)}

    def apply_chat_template(self, messages, **kwargs):
        return [ord("P")]

    def decode(self, token_ids, **kwargs):
        return "".join(map(chr, token_ids))


@pytest.fixture
def weather_request():
    return {
        "model": "test",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "object"}},
                    },
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("tool_choice", "use_pool"),
    [("auto", False), ("auto", True), ("required", False)],
    ids=["auto-inline", "auto-pool", "required-inline"],
)
def test_hunyuan_checkpoint_tokens(weather_request, use_pool, tool_choice, monkeypatch):
    tokenizer = HunyuanTokenizer()
    weather_request["tool_choice"] = tool_choice
    reasoning = "Check the weather." if tool_choice == "auto" else ""
    text = (
        f"<think:opensource>{reasoning}</think:opensource>" if reasoning else ""
    ) + (
        "<tool_calls:opensource><tool_call:opensource>get_weather<tool_sep:opensource>"
        "<arg_key:opensource>location</arg_key:opensource>"
        '<arg_value:opensource>{"city":"Paris"}</arg_value:opensource>'
        "</tool_call:opensource></tool_calls:opensource>"
    )
    # A required-tool request receiving native output exercises finish-time
    # fallback from JsonArrayParser, as when a backend cannot enforce guidance.
    engine = FakeRoutedEngine(
        items=[{"token_ids": list(map(ord, text)), "finish_reason": "stop"}]
    )
    if use_pool:
        # Exercise worker preprocessing and parent parser reconstruction without
        # loading a checkpoint or spawning a process.
        for name, value in {
            "tokenizer": tokenizer,
            "tool_call_parser_name": "hunyuan",
            "reasoning_parser_name": "hunyuan",
            "exclude_tools_when_tool_choice_none": True,
            "template_force_reasoning": False,
            "default_thinking_mode": None,
        }.items():
            monkeypatch.setattr(processor_module, f"_w_{name}", value)

    with ThreadPoolExecutor(max_workers=1) if use_pool else nullcontext() as pool:
        processor = SglangProcessor(
            tokenizer,
            engine,
            "hunyuan",
            "hunyuan",
            None,
            preprocess_pool=pool,
            preprocess_workers=1 if use_pool else 0,
        )

        async def collect():
            return [item async for item in processor.generator(weather_request)]

        output = asyncio.run(collect())

    choices = [
        choice for item in output for choice in item.get("data", {}).get("choices", [])
    ]
    assert (
        "".join(c["delta"].get("reasoning_content", "") for c in choices) == reasoning
    )
    if tool_choice == "auto":
        assert "".join(c["delta"].get("content", "") for c in choices) == ""
    calls = [call for c in choices for call in c["delta"].get("tool_calls", [])]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "location": {"city": "Paris"}
    }
    assert choices[-1]["finish_reason"] == "tool_calls"


def test_auto_guidance_receives_checkpoint_tokens(weather_request, monkeypatch):
    # Hunyuan currently has no auto grammar. Inspect the real parser at the
    # guidance boundary to ensure this separate construction gets the tokenizer.
    def constraint(parser, tool_choice):
        assert tool_choice == "auto"
        assert parser.detector.bot_token == "<tool_calls:opensource>"
        return "structural_tag", {"type": "object"}

    monkeypatch.setattr(FunctionCallParser, "get_structure_constraint", constraint)
    result = preprocess_chat_request(
        weather_request,
        tokenizer=HunyuanTokenizer(),
        tool_call_parser_name="hunyuan",
        reasoning_parser_name="hunyuan",
    )
    assert result.guided_decoding == {"structural_tag": {"type": "object"}}
