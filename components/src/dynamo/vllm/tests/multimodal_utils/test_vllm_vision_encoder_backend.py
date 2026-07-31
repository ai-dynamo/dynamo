# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.vision_encoder_backend.

Pin the author-facing contract surface: the ``Preprocessed`` carrier (item +
scalar cost, no bucket_key), the hardcoded ``image_token_id`` attribute, the
no-device ``build`` signature, and that the ABC cannot be instantiated without
the required methods.
"""

import pytest
import torch

from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    Preprocessed,
    VisionEncoderBackend,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _MinimalBackend(VisionEncoderBackend):
    """Smallest concrete backend — hardcodes the image token id."""

    image_token_id = 151655

    def build(self, model_id):
        ...

    def preprocess(self, raw):
        return Preprocessed(item=raw, cost=1)

    def forward_batch(self, items, target_bucket=None):
        return [torch.zeros(1, 1) for _ in items]


class _PassthroughBackend(VisionEncoderBackend):
    """Backend that does NOT override preprocess — exercises the identity default
    (no preprocess phase; raws go straight to forward_batch)."""

    image_token_id = 151655

    def build(self, model_id):
        ...

    def forward_batch(self, items, target_bucket=None):
        return [torch.zeros(1, 1) for _ in items]


def test_preprocessed_is_frozen():
    p = Preprocessed(item="x", cost=3)
    assert (p.item, p.cost) == ("x", 3)
    with pytest.raises(Exception):  # FrozenInstanceError
        p.cost = 4  # type: ignore[misc]


def test_preprocessed_cost_defaults_to_1():
    # A pass-through author can omit cost entirely.
    assert Preprocessed(item="x").cost == 1


def test_preprocessed_has_no_bucket_key():
    # Batching is one-dimensional (scalar cost only) — there is no bucket_key.
    assert not hasattr(Preprocessed(item="x"), "bucket_key")


def test_abc_cannot_be_instantiated():
    # build + forward_batch are still abstract (preprocess is not).
    with pytest.raises(TypeError):
        VisionEncoderBackend()  # type: ignore[abstract]


def test_preprocess_defaults_to_identity_passthrough():
    # Not overriding preprocess ⇒ raw IS the item, cost 1 (no preprocess phase).
    p = _PassthroughBackend().preprocess("http://img")
    assert isinstance(p, Preprocessed)
    assert (p.item, p.cost) == ("http://img", 1)


def test_preprocess_concurrency_defaults_to_0():
    # No pool by default — authors opt in by overriding preprocess + setting > 0.
    assert _PassthroughBackend().preprocess_concurrency == 0
    assert _MinimalBackend().preprocess_concurrency == 0


def test_default_attrs_and_close_noop():
    e = _MinimalBackend()
    # Defaults from the ABC: eager (no ladder) + pass-through (no cost cap) + no
    # preprocess pool.
    assert e.buckets is None
    assert e.max_batch_cost is None
    assert e.preprocess_concurrency == 0
    assert e.image_token_id == 151655
    assert e.close() is None
