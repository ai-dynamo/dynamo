# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe encoding against SGLang's real adapter, without loading a model."""

import copy
import json

import pytest

pytest.importorskip("sglang.srt.parser")
from dynamo.frontend import sglang_deepseek  # noqa: E402 -- optional SGLang guard

pytest_plugins = ["_deepseek_recipe_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def tokenizer(recipe_tokenizer):
    return recipe_tokenizer


def render(request, tokenizer, *, spec="dsv4", images=None):
    return sglang_deepseek.render_recipe_prompt(
        request,
        messages=request["messages"],
        tokenizer=tokenizer,
        tools=request.get("tools"),
        spec=spec,
        image_urls=images or [],
    ).token_ids


@pytest.mark.parametrize("spec,effort", [("dsv4", "max"), ("dsv41", 37)])
@pytest.mark.parametrize("thinking", [False, True])
def test_recipe_uses_served_tokenizer_and_request_effort(
    tokenizer, spec, effort, thinking
):
    request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": effort,
        "chat_template_kwargs": {"thinking": thinking, "reasoning_effort": "low"},
        "temperature": 0,
        "max_tokens": 19,
    }
    original = copy.deepcopy(request)
    ids = render(request, tokenizer, spec=spec)
    prompt = tokenizer.backend_tokenizer.decode(ids, skip_special_tokens=False)
    assert "Hello" in prompt
    assert prompt.endswith("<think>" if thinking else "</think>")
    if thinking:
        assert "Reasoning Effort: 37" in prompt if spec == "dsv41" else "max" in prompt
    assert ids[0] == tokenizer.backend_tokenizer.token_to_id("<｜begin▁of▁sentence｜>")
    assert request == original


def test_filtered_tools_leave_tool_choice_to_host_grammar(tokenizer):
    request = {
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "tool_choice": {"type": "function", "function": {"name": "weather"}},
    }
    ids = render(request, tokenizer)
    prompt = tokenizer.backend_tokenizer.decode(ids, skip_special_tokens=False)
    assert "weather" in prompt
    assert prompt.endswith("<｜Assistant｜></think>")


def test_v41_image_slots_match_worker_order(tokenizer):
    images = ["https://example.com/first.png", "https://example.com/second.png"]
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}} for url in images
                ],
            }
        ]
    }
    ids = render(request, tokenizer, spec="dsv41", images=images)
    image_id = tokenizer.backend_tokenizer.token_to_id("<｜deepseek_image｜>")
    assert ids.count(image_id) == 2
    rendered = sglang_deepseek.render_recipe_prompt(
        request,
        messages=request["messages"],
        tokenizer=tokenizer,
        tools=None,
        spec="dsv41",
        image_urls=list(reversed(images)),
    )
    assert rendered.image_urls == images
    with pytest.raises(sglang_deepseek.RecipeRequestError, match="images differ"):
        render(request, tokenizer, spec="dsv41", images=[images[0], images[0]])


def test_recipe_request_errors_do_not_hide_tokenizer_configuration_errors(tokenizer):
    request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": True,
    }
    with pytest.raises(sglang_deepseek.RecipeRequestError, match="boolean"):
        render(request, tokenizer, spec="dsv41")
    with pytest.raises(ValueError, match="fast tokenizer") as error:
        render(request, object(), spec="dsv41")
    assert not isinstance(error.value, sglang_deepseek.RecipeRequestError)


@pytest.mark.parametrize("spec", ["dsv4", "dsv41"])
@pytest.mark.parametrize("thinking", [False, True])
def test_recipe_continues_final_assistant_message(tokenizer, spec, thinking):
    request = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "The answer is"},
        ],
        "continue_final_message": True,
        "chat_template_kwargs": {"thinking": thinking},
    }
    original = copy.deepcopy(request)
    if thinking:
        with pytest.raises(
            sglang_deepseek.RecipeRequestError,
            match="continue_final_message with thinking",
        ):
            render(request, tokenizer, spec=spec)
    else:
        ids = render(request, tokenizer, spec=spec)
        prompt = tokenizer.backend_tokenizer.decode(ids, skip_special_tokens=False)
        assert prompt.endswith("The answer is")
    assert request == original


@pytest.mark.parametrize(
    "model_type,architecture,expected",
    [
        ("deepseek_v41", "DeepseekV4ForCausalLM", "dsv41"),
        ("deepseek_v4", "DeepseekV4ForCausalLM", "dsv4"),
        ("other", "OtherForCausalLM", None),
    ],
)
def test_recipe_spec_comes_from_checkpoint(
    tmp_path, model_type, architecture, expected
):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [architecture]})
    )
    assert sglang_deepseek.resolve_deepseek_spec(str(tmp_path)) == expected


def test_recipe_normalizes_reasoning_history_without_mutating_request(tokenizer):
    request = {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": ["Check the weather", "", "Use the tool"],
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ],
        "chat_template_kwargs": {"thinking": True},
    }
    original = copy.deepcopy(request)
    ids = render(request, tokenizer, spec="dsv41")
    expected = copy.deepcopy(request)
    expected["messages"][1]["reasoning_content"] = "Check the weather\nUse the tool"
    assert ids == render(expected, tokenizer, spec="dsv41")
    assert request == original


def test_recipe_returns_tool_images_in_prompt_order(tokenizer, tool_image_request):
    request, expected = tool_image_request
    original = copy.deepcopy(request)
    wire_images = [
        part["image_url"]["url"]
        for message in request["messages"]
        if message["role"] == "tool"
        for part in message["content"]
    ]
    rendered = sglang_deepseek.render_recipe_prompt(
        request,
        messages=request["messages"],
        tokenizer=tokenizer,
        tools=None,
        spec="dsv41",
        image_urls=wire_images,
    )
    assert rendered.image_urls == expected
    image_id = tokenizer.backend_tokenizer.token_to_id("<｜deepseek_image｜>")
    assert rendered.token_ids.count(image_id) == len(expected)
    assert request == original
