# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.custom_encoder.

The image placeholder token is identified by its token *string*, set on the
subclass via ``image_placeholder_token``; the numeric id is resolved from the
model tokenizer (``placeholder_token_id_from_tokenizer``). These tests pin that
generic resolution, the fail-fast errors (unset token string, unset tokenizer,
token the tokenizer does not define), and ``validate()`` as the post-load early
check. Qwen-specific behavior lives in test_vllm_qwen_custom_encoder.py.
"""

from typing import List

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder import CustomEncoder

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
    image_placeholder_token = _TOKEN

    def load(self, model_id: str, device: str) -> None:  # pragma: no cover - stub
        ...

    def encode(self, image_urls: List[str]) -> List[torch.Tensor]:  # pragma: no cover
        return []


class _NoTokenEncoder(CustomEncoder):
    def load(self, model_id: str, device: str) -> None:  # pragma: no cover - stub
        ...

    def encode(self, image_urls: List[str]) -> List[torch.Tensor]:  # pragma: no cover
        return []


def test_resolves_id_from_tokenizer():
    """A subclass that sets image_placeholder_token resolves it via the tokenizer."""
    enc = _Encoder()
    enc.tokenizer = _FakeTokenizer({_TOKEN: 42})
    assert enc.get_image_placeholder_token_id() == 42


def test_token_not_defined_raises():
    """Token mapping to unk_token_id is treated as undefined -> ValueError."""
    enc = _Encoder()
    enc.tokenizer = _FakeTokenizer({"something_else": 1}, unk_token_id=0)
    with pytest.raises(ValueError, match="does not define placeholder token"):
        enc.get_image_placeholder_token_id()


def test_unset_token_string_raises():
    """An encoder that never sets image_placeholder_token -> ValueError."""
    enc = _NoTokenEncoder()
    enc.tokenizer = _FakeTokenizer({_TOKEN: 42})
    with pytest.raises(ValueError, match="image_placeholder_token is not set"):
        enc.get_image_placeholder_token_id()


def test_unset_tokenizer_raises():
    """No tokenizer assigned in load() -> ValueError."""
    enc = _Encoder()
    with pytest.raises(ValueError, match="self.tokenizer is not set"):
        enc.get_image_placeholder_token_id()


def test_validate_passes_when_resolvable():
    """validate() is a no-op when the placeholder id resolves."""
    enc = _Encoder()
    enc.tokenizer = _FakeTokenizer({_TOKEN: 42})
    enc.validate()  # must not raise


def test_validate_fails_fast_on_bad_config():
    """validate() surfaces a misconfigured encoder before the first request."""
    enc = _Encoder()  # tokenizer never set
    with pytest.raises(ValueError):
        enc.validate()
