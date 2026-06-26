# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.qwen_custom_encoder.

QwenSerializedCustomEncoder is a serial (SerializedCustomEncoder) base that
implements get_image_placeholder_token_id by resolving the Qwen image
placeholder token string against the model tokenizer (assigned by the subclass
in load()). These tests pin the preset string, per-version id resolution (151655
for Qwen3-VL, 248056 for Qwen3.5 — why the tokenizer is authoritative rather
than a static id table), the fail-fast errors (unset tokenizer, token the
tokenizer does not define), and that the class stays abstract.
"""

from typing import List

import pytest
import torch

from dynamo.vllm.multimodal_utils.qwen_custom_encoder import (
    QWEN_IMAGE_PLACEHOLDER_TOKEN,
    QwenSerializedCustomEncoder,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeTokenizer:
    """Minimal tokenizer stub: convert_tokens_to_ids + unk_token_id."""

    def __init__(self, mapping: dict, unk_token_id=None):
        self._mapping = mapping
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token: str):
        return self._mapping.get(token, self.unk_token_id)


class _ConcreteQwen(QwenSerializedCustomEncoder):
    def load(self, model_id: str, device: str) -> None:
        ...

    def _encode_blocking(self, image_urls: List[str]) -> List[torch.Tensor]:
        return []


def test_preset_token_string():
    """QwenSerializedCustomEncoder presets the Qwen placeholder token string."""
    assert QWEN_IMAGE_PLACEHOLDER_TOKEN == "<|image_pad|>"


@pytest.mark.parametrize("token_id", [151655, 248056])
def test_same_string_resolves_per_version_id(token_id):
    """The single preset string resolves to whatever id the tokenizer defines
    (151655 Qwen3-VL, 248056 Qwen3.5) — what a static id table could not do."""
    enc = _ConcreteQwen()
    enc.tokenizer = _FakeTokenizer({QWEN_IMAGE_PLACEHOLDER_TOKEN: token_id})
    assert enc.get_image_placeholder_token_id() == token_id
    enc.validate()  # must not raise


def test_unset_tokenizer_raises():
    """No tokenizer assigned in load() -> ValueError."""
    enc = _ConcreteQwen()
    with pytest.raises(ValueError, match="tokenizer is not set"):
        enc.get_image_placeholder_token_id()


def test_token_not_defined_raises():
    """Token mapping to unk_token_id is treated as undefined -> ValueError."""
    enc = _ConcreteQwen()
    enc.tokenizer = _FakeTokenizer({"something_else": 1}, unk_token_id=0)
    with pytest.raises(ValueError, match="does not define placeholder token"):
        enc.get_image_placeholder_token_id()


def test_qwen_serialized_custom_encoder_is_abstract():
    """load / _encode_blocking stay abstract -> cannot be instantiated."""
    with pytest.raises(TypeError):
        QwenSerializedCustomEncoder()
