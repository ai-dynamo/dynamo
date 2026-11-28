# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for how InputParamManager tokenizes a completions prompt.

As in vLLM's /v1/completions, the tokenizer adds its special tokens (such as
BOS) unless the request sets add_special_tokens.
"""

import pytest

from dynamo.common.utils.input_params import InputParamManager

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

BOS = 1


class RecordingTokenizer:
    """Adds BOS unless add_special_tokens=False, like a Llama tokenizer."""

    def __init__(self):
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append(kwargs)
        ids = [ord(char) for char in text]
        if kwargs.get("add_special_tokens", True):
            return [BOS, *ids]
        return ids


def test_prompt_keeps_the_tokenizer_default_when_unset():
    tokenizer = RecordingTokenizer()
    manager = InputParamManager(tokenizer)

    assert manager.get_input_param({"prompt": "hi"}, use_tokenizer=True) == [
        BOS,
        ord("h"),
        ord("i"),
    ]
    assert tokenizer.calls == [{}]


@pytest.mark.parametrize(
    ("add_special_tokens", "expected"),
    [(True, [BOS, ord("h"), ord("i")]), (False, [ord("h"), ord("i")])],
)
def test_prompt_honors_add_special_tokens(add_special_tokens, expected):
    tokenizer = RecordingTokenizer()
    manager = InputParamManager(tokenizer)

    request = {"prompt": "hi", "add_special_tokens": add_special_tokens}
    assert manager.get_input_param(request, use_tokenizer=True) == expected
    assert tokenizer.calls == [{"add_special_tokens": add_special_tokens}]
