# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.qwen_custom_encoder.

QwenCustomEncoder presets the Qwen image placeholder token string; the numeric
id is resolved from the model tokenizer. These tests pin that the preset token
is used, that the same string resolves to the per-version id (151655 for
Qwen3-VL, 248056 for Qwen3.5 — the reason the tokenizer is authoritative rather
than a static id table), and that the class stays abstract.
"""

from typing import List

import pytest
import torch

from dynamo.vllm.multimodal_utils.qwen_custom_encoder import (
    QWEN_IMAGE_PLACEHOLDER_TOKEN,
    QwenCustomEncoder,
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


class _ConcreteQwen(QwenCustomEncoder):
    def load(self, model_id: str, device: str) -> None:  # pragma: no cover - stub
        ...

    def encode(self, image_urls: List[str]) -> List[torch.Tensor]:  # pragma: no cover
        return []


def test_preset_token_string():
    """QwenCustomEncoder presets the Qwen placeholder token string."""
    assert QWEN_IMAGE_PLACEHOLDER_TOKEN == "<|image_pad|>"
    assert _ConcreteQwen.image_placeholder_token == QWEN_IMAGE_PLACEHOLDER_TOKEN


@pytest.mark.parametrize("token_id", [151655, 248056])
def test_same_string_resolves_per_version_id(token_id):
    """The single preset string resolves to whatever id the tokenizer defines
    (151655 Qwen3-VL, 248056 Qwen3.5) — what a static id table could not do."""
    enc = _ConcreteQwen()
    enc.tokenizer = _FakeTokenizer({QWEN_IMAGE_PLACEHOLDER_TOKEN: token_id})
    assert enc.get_image_placeholder_token_id() == token_id
    enc.validate()  # must not raise


def test_qwen_custom_encoder_is_abstract():
    """load/encode stay abstract -> QwenCustomEncoder cannot be instantiated."""
    with pytest.raises(TypeError):
        QwenCustomEncoder()
