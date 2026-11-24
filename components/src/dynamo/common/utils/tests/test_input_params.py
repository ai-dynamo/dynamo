#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""InputParamManager hands the engine token ids, never rendered chat text.

Re-encoding the rendered chat would add a second BOS wherever the template
already emitted one (Llama, Gemma, Mistral) and lose Mistral's ``[INST]``
control tokens (ai-dynamo/dynamo#18810), so the chat branch must tokenize in
the template and return ``list[int]``.
"""

from types import SimpleNamespace

import pytest

from dynamo.common.utils.input_params import InputParamManager

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

BOS = 128000


class _BosTemplateTokenizer:
    """Mimics an HF tokenizer whose chat template starts with the BOS token.

    ``encode`` adds BOS (the tokenizer default); ``apply_chat_template``
    renders ``<bos>`` as text and, with ``tokenize=True``, encodes the rendered
    text without special tokens, as transformers does.
    """

    chat_template = "<bos>{% for m in messages %}{{ m.content }}{% endfor %}"
    bos_token = "<bos>"

    def __init__(self):
        self.calls: list[dict] = []

    def _ids(self, text, add_special_tokens):
        ids = [BOS] if add_special_tokens else []
        for piece in text.replace("<bos>", " <bos> ").split():
            ids.append(BOS if piece == "<bos>" else len(piece))
        return ids

    def encode(self, text, add_special_tokens=True):
        return self._ids(text, add_special_tokens)

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        rendered = "<bos>" + "".join(m["content"] for m in messages)
        if kwargs.get("add_generation_prompt"):
            rendered += " assistant:"
        if not kwargs.get("tokenize", True):
            return rendered
        ids = self._ids(rendered, add_special_tokens=False)
        if kwargs.get("return_dict", False):
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids


def _request(**extra):
    return {"messages": [{"role": "user", "content": "What is 2 + 3?"}], **extra}


def test_chat_prompt_carries_exactly_one_bos():
    tokenizer = _BosTemplateTokenizer()
    ids = InputParamManager(tokenizer).get_input_param(_request(), use_tokenizer=True)

    expected = tokenizer.apply_chat_template(
        _request()["messages"], tokenize=True, add_generation_prompt=True
    )
    assert ids == expected
    assert ids[0] == BOS and ids[1] != BOS
    # The engine must never see rendered text for a chat request.
    assert isinstance(ids, list) and all(isinstance(t, int) for t in ids)


def test_chat_template_is_called_with_tokenize_true():
    tokenizer = _BosTemplateTokenizer()
    InputParamManager(tokenizer).get_input_param(_request(), use_tokenizer=True)

    (kwargs,) = tokenizer.calls
    assert kwargs["tokenize"] is True
    assert kwargs["return_dict"] is False
    assert kwargs["add_generation_prompt"] is True


def test_caller_cannot_override_tokenize_or_return_dict():
    tokenizer = _BosTemplateTokenizer()
    request = _request(
        chat_template_kwargs={
            "tokenize": False,
            "return_dict": True,
            "add_generation_prompt": False,
            "enable_thinking": False,
        }
    )
    ids = InputParamManager(tokenizer).get_input_param(request, use_tokenizer=True)

    (kwargs,) = tokenizer.calls
    assert kwargs["tokenize"] is True
    assert kwargs["return_dict"] is False
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["enable_thinking"] is False
    assert isinstance(ids, list)


def test_batch_encoding_return_is_unwrapped_to_ids():
    """A tokenizer that ignores return_dict=False still yields plain ids."""

    class _DictTokenizer(_BosTemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            kwargs["return_dict"] = True
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = _DictTokenizer()
    ids = InputParamManager(tokenizer).get_input_param(_request(), use_tokenizer=True)

    expected = tokenizer.apply_chat_template(
        _request()["messages"], tokenize=True, add_generation_prompt=True
    )
    assert ids == expected["input_ids"]


def test_tensor_like_return_is_converted_to_a_list():
    class _TensorTokenizer(_BosTemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            return SimpleNamespace(tolist=lambda: [ids])

    ids = InputParamManager(_TensorTokenizer()).get_input_param(
        _request(), use_tokenizer=True
    )

    assert isinstance(ids, list) and ids[0] == BOS


def test_text_return_from_the_template_is_rejected():
    tokenizer = _BosTemplateTokenizer()
    tokenizer.apply_chat_template = lambda messages, **kwargs: "<bos>rendered"

    with pytest.raises(TypeError, match="expected a list of token ids"):
        InputParamManager(tokenizer).get_input_param(_request(), use_tokenizer=True)


def test_completions_prompt_still_gets_bos_once():
    tokenizer = _BosTemplateTokenizer()
    ids = InputParamManager(tokenizer).get_input_param(
        {"prompt": "What is 2 + 3?"}, use_tokenizer=True
    )

    assert ids[0] == BOS and BOS not in ids[1:]


def test_token_ids_passthrough_without_tokenizer():
    assert InputParamManager(None).get_input_param(
        {"token_ids": [5, 6]}, use_tokenizer=False
    ) == [5, 6]
