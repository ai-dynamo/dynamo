# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, model-free fixtures shared by recipe and processor contract tests."""

import pytest

from dynamo.frontend import sglang_deepseek


@pytest.fixture
def recipe_tokenizer():
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


@pytest.fixture(params=[1, 2], ids=["distinct-images", "repeated-image"])
def tool_image_request(request):
    first, second = "https://example.com/first.png", "https://example.com/second.png"
    call_images = {"first": [first] * request.param, "second": [second]}
    body = {
        "model": "deepseek-v41-prod",
        "messages": [
            {"role": "user", "content": "Compare the images"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": name,
                        "type": "function",
                        "function": {"name": "image", "arguments": "{}"},
                    }
                    for name in call_images
                ],
            },
            *[
                {
                    "role": "tool",
                    "tool_call_id": name,
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in call_images[name]
                    ],
                }
                for name in reversed(call_images)
            ],
        ],
    }
    return body, call_images["first"] + call_images["second"]
