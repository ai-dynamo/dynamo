# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.custom_encoder.

The base CustomEncoder defines the *contract* only: subclasses implement
load / async encode / get_image_placeholder_token_id. How those are realized
(encode's execution model; the id source) is a subclass detail — the serial
default is covered in test_vllm_serialized_custom_encoder.py and Qwen resolution
in test_vllm_qwen_serialized_custom_encoder.py. These tests pin the abstract
contract,
validate()'s fail-fast dispatch to the subclass impl, and the reusable
placeholder_token_id_from_tokenizer helper.
"""

from typing import List

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder import (
    CustomEncoder,
    placeholder_token_id_from_tokenizer,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_TOKEN = "<|other_model_image_token|>"


class _FakeTokenizer:
    """Minimal tokenizer stub: convert_tokens_to_ids + unk_token_id."""

    def __init__(self, mapping: dict, unk_token_id=None):
        self._mapping = mapping
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token: str):
        return self._mapping.get(token, self.unk_token_id)


class _Encoder(CustomEncoder):
    """Concrete encoder returning a fixed placeholder id."""

    def load(self, model_id: str, device: str) -> None:
        ...

    async def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
        return []

    def get_image_placeholder_token_id(self) -> int:
        return 42


class _MisconfiguredEncoder(CustomEncoder):
    """Encoder whose id resolution fails, to exercise validate() propagation."""

    def load(self, model_id: str, device: str) -> None:
        ...

    async def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
        return []

    def get_image_placeholder_token_id(self) -> int:
        raise ValueError("placeholder id unresolved")


def test_helper_resolves_id():
    """The helper resolves a defined token to its id."""
    tok = _FakeTokenizer({_TOKEN: 42})
    assert placeholder_token_id_from_tokenizer(tok, _TOKEN) == 42


def test_helper_returns_none_when_token_maps_to_unk():
    """A token mapping to unk_token_id is treated as undefined -> None."""
    tok = _FakeTokenizer({"something_else": 1}, unk_token_id=0)
    assert placeholder_token_id_from_tokenizer(tok, _TOKEN) is None


def test_helper_returns_none_without_convert_method():
    """An object without convert_tokens_to_ids resolves to None, not an error."""
    assert placeholder_token_id_from_tokenizer(object(), _TOKEN) is None


def test_encode_is_abstract():
    """A subclass that omits encode cannot instantiate."""

    class _NoEncode(CustomEncoder):
        def load(self, model_id: str, device: str) -> None:
            ...

        def get_image_placeholder_token_id(self) -> int:
            return 1

    with pytest.raises(TypeError):
        _NoEncode()


def test_get_image_placeholder_token_id_is_abstract():
    """A subclass that omits get_image_placeholder_token_id cannot instantiate."""

    class _Missing(CustomEncoder):
        def load(self, model_id: str, device: str) -> None:
            ...

        async def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
            return []

    with pytest.raises(TypeError):
        _Missing()


def test_validate_passes_when_id_resolves():
    """validate() is a no-op when the subclass returns an id."""
    _Encoder().validate()  # must not raise


def test_validate_propagates_subclass_error():
    """validate() surfaces the subclass's resolution error at startup."""
    with pytest.raises(ValueError, match="placeholder id unresolved"):
        _MisconfiguredEncoder().validate()
