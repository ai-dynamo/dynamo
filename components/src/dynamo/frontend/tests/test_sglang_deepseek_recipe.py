# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe encoding against SGLang's real adapter, without loading a model."""

import copy

import pytest

from dynamo.frontend import sglang_deepseek

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def tokenizer():
    if sglang_deepseek.recipe_renderer is None:
        pytest.skip("SGLang predates the recipe renderer")
    recipe = pytest.importorskip("deepseek_recipe")
    tokenizers = pytest.importorskip("tokenizers")
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.add_special_tokens(
        [
            getattr(recipe.SpecialToken, name).default_str()
            for name in dir(recipe.SpecialToken)
            if name[0].isupper()
        ]
        + ["<｜deepseek_image｜>"]
    )

    class ServedTokenizer:
        backend_tokenizer = backend

    return ServedTokenizer()


def render(request, tokenizer, *, spec="dsv4", images=None):
    return sglang_deepseek.render_recipe_prompt(
        request,
        messages=request["messages"],
        tokenizer=tokenizer,
        tools=request.get("tools"),
        spec=spec,
        image_urls=images or [],
    )


@pytest.mark.parametrize("spec,effort", [("dsv4", "max"), ("dsv41", 37)])
def test_recipe_uses_served_tokenizer_and_request_effort(tokenizer, spec, effort):
    request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": effort,
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"},
        "temperature": 0,
        "max_tokens": 19,
    }
    original = copy.deepcopy(request)
    ids = render(request, tokenizer, spec=spec)
    prompt = tokenizer.backend_tokenizer.decode(ids, skip_special_tokens=False)
    assert "Hello" in prompt
    assert prompt.endswith("<think>")
    assert "Reasoning Effort: 37" in prompt if spec == "dsv41" else "max" in prompt
    assert ids[0] == tokenizer.backend_tokenizer.token_to_id("<｜begin▁of▁sentence｜>")
    assert request == original


@pytest.mark.parametrize(
    "choice", ["none", {"type": "function", "function": {"name": "weather"}}]
)
def test_filtered_tools_leave_tool_choice_to_host_grammar(tokenizer, choice):
    request = {
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "tool_choice": choice,
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
    with pytest.raises(sglang_deepseek.RecipeRequestError, match="image order"):
        render(request, tokenizer, spec="dsv41", images=list(reversed(images)))


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
