# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.vision_encoder_backend.

Pin the author-facing contract surface: the ``Preprocessed`` carrier, the
placeholder-id resolution helper + Qwen mixin, and that the ABC cannot be
instantiated without the required methods.
"""

import pytest
import torch

from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    QWEN_IMAGE_PLACEHOLDER_TOKEN,
    Preprocessed,
    QwenPlaceholderMixin,
    VisionEncoderBackend,
    placeholder_token_id_from_tokenizer,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeTokenizer:
    def __init__(self, mapping, unk=None):
        self._m = mapping
        self.unk_token_id = unk

    def convert_tokens_to_ids(self, token):
        return self._m.get(token, self.unk_token_id)


class _MinimalBackend(QwenPlaceholderMixin, VisionEncoderBackend):
    """Smallest concrete backend — enough to instantiate and test the mixin."""

    def build(self, model_id, device):
        ...

    def preprocess(self, raw):
        return Preprocessed(item=raw, cost=1, bucket_key=None)

    def forward_batch(self, items, target_bucket=None):
        return [torch.zeros(1, 1) for _ in items]


def test_preprocessed_is_frozen():
    p = Preprocessed(item="x", cost=3, bucket_key=(1, 2, 3))
    assert (p.item, p.cost, p.bucket_key) == ("x", 3, (1, 2, 3))
    with pytest.raises(Exception):  # FrozenInstanceError
        p.cost = 4  # type: ignore[misc]


def test_placeholder_resolves_from_tokenizer():
    tok = _FakeTokenizer({QWEN_IMAGE_PLACEHOLDER_TOKEN: 151655})
    assert (
        placeholder_token_id_from_tokenizer(tok, QWEN_IMAGE_PLACEHOLDER_TOKEN) == 151655
    )


def test_placeholder_none_when_token_maps_to_unk():
    tok = _FakeTokenizer({}, unk=0)  # missing token → returns unk id
    assert (
        placeholder_token_id_from_tokenizer(tok, QWEN_IMAGE_PLACEHOLDER_TOKEN) is None
    )


def test_placeholder_none_when_tokenizer_has_no_convert():
    assert placeholder_token_id_from_tokenizer(object(), "x") is None


def test_qwen_mixin_resolves_id_from_self_tokenizer():
    e = _MinimalBackend()
    e.tokenizer = _FakeTokenizer({QWEN_IMAGE_PLACEHOLDER_TOKEN: 42})
    assert e.get_image_placeholder_token_id() == 42


def test_qwen_mixin_raises_without_tokenizer():
    e = _MinimalBackend()
    with pytest.raises(ValueError, match="tokenizer is not set"):
        e.get_image_placeholder_token_id()


def test_qwen_mixin_raises_when_token_undefined():
    e = _MinimalBackend()
    e.tokenizer = _FakeTokenizer({}, unk=0)
    with pytest.raises(ValueError, match="does not define placeholder"):
        e.get_image_placeholder_token_id()


def test_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        VisionEncoderBackend()  # type: ignore[abstract]


def test_default_attrs_and_close_noop():
    e = _MinimalBackend()
    # Defaults from the ABC: eager (no ladder) + pass-through (no cost cap).
    assert e.buckets is None
    assert e.max_batch_cost is None
    assert e.close() is None


def test_preprocessed_defaults_for_passthrough():
    # A pass-through author can omit cost / bucket_key entirely.
    p = Preprocessed(item="x")
    assert p.cost == 1
    assert p.bucket_key is None
