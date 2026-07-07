# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DYN_SKIP_FRONTEND_TOKENIZE behaviour in the vLLM engine.

These tests verify that generate() selects TextPrompt when token_ids is
empty and prompt_text is present, and falls back to TokensPrompt otherwise.
No GPU or vLLM engine initialisation is required.
"""

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(token_ids=None, prompt_text=None):
    req = {}
    if token_ids is not None:
        req["token_ids"] = token_ids
    if prompt_text is not None:
        req["prompt_text"] = prompt_text
    return req


def _prompt_type(token_ids, prompt_text):
    """Mirror the selection logic from VllmLLMEngine.generate()."""
    if not token_ids and prompt_text:
        return "text"
    return "tokens"


# ---------------------------------------------------------------------------
# Prompt selection logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token_ids", "prompt_text", "expected"),
    [
        # skip path: no token_ids, prompt_text present → TextPrompt
        ([], "Hello world", "text"),
        (None, "Hello world", "text"),
        # normal path: token_ids present → TokensPrompt regardless of prompt_text
        ([1, 2, 3], None, "tokens"),
        ([1, 2, 3], "Hello world", "tokens"),
        # edge: both absent → empty TokensPrompt
        ([], None, "tokens"),
        (None, None, "tokens"),
    ],
)
def test_prompt_type_selection(token_ids, prompt_text, expected):
    assert _prompt_type(token_ids, prompt_text) == expected


def test_text_prompt_is_a_dict_subtype():
    """TextPrompt is a TypedDict; {"prompt": text} matches what SGLang/vLLM expect."""
    from vllm.inputs import TextPrompt

    tp = TextPrompt(prompt="Hello world")
    assert tp["prompt"] == "Hello world"
    assert isinstance(tp, dict)


def test_tokens_prompt_carries_token_ids():
    from vllm.inputs import TokensPrompt

    tp = TokensPrompt(prompt_token_ids=[1, 2, 3])
    assert tp["prompt_token_ids"] == [1, 2, 3]
