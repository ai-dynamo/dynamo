# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.embed_assembler.

build_mixed_embeds assembles the mixed token-ids/embeds EmbedsPrompt for the
aggregated CustomEncoder path: each placeholder token is expanded to its encoder
tensor's row count, image rows carry the encoder embeddings, and text rows stay
zero (vLLM fills them from the model's embedding table). These tests pin that
layout, the per-image expansion (including back-to-back images), and the input
validation that surfaces a bad encoder output as a clear ValueError.
"""

import pytest
import torch

from dynamo.vllm.multimodal_utils.embed_assembler import build_mixed_embeds

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_HIDDEN = 8
_PLACEHOLDER = 999


def test_build_mixed_embeds_multi_image_token_expand():
    """Each placeholder token is one image and is expanded to its encoder
    tensor's row count; back-to-back placeholders (adjacent images, no
    separator) yield one block per image."""
    # img_a, then text, then img_b and img_c back-to-back (no separator).
    token_ids = [1, 2, _PLACEHOLDER, 3, _PLACEHOLDER, _PLACEHOLDER, 4, 5]
    img_a = torch.ones(2, _HIDDEN, dtype=torch.float16)
    img_b = torch.ones(3, _HIDDEN, dtype=torch.float16) * 2.0
    img_c = torch.ones(1, _HIDDEN, dtype=torch.float16) * 3.0

    embeds, out_ids, is_tok = build_mixed_embeds(
        token_ids, [img_a, img_b, img_c], _PLACEHOLDER
    )

    # Layout: [1,2] + img_a(2) + [3] + img_b(3) + img_c(1) + [4,5] -> 11 rows.
    assert embeds.shape == (11, _HIDDEN)
    assert embeds.dtype == torch.float16
    assert embeds.device.type == "cpu"
    assert len(out_ids) == 11 and len(is_tok) == 11
    assert out_ids == [1, 2, 999, 999, 3, 999, 999, 999, 999, 4, 5]
    assert is_tok == [
        True,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    # Each image's rows carry its encoder values; text rows stay zero.
    assert torch.all(embeds[2:4] == 1.0)  # img_a
    assert torch.all(embeds[5:8] == 2.0)  # img_b
    assert torch.all(embeds[8] == 3.0)  # img_c (adjacent to img_b, no separator)
    assert torch.all(embeds[:2] == 0)
    assert torch.all(embeds[4] == 0)
    assert torch.all(embeds[9:] == 0)


@pytest.mark.parametrize(
    "token_ids, n_tensors",
    [
        pytest.param([1, _PLACEHOLDER, 2], 2, id="more_tensors_than_placeholders"),
        pytest.param([1, 2, 3], 1, id="no_placeholders_but_tensors"),
    ],
)
def test_build_mixed_embeds_raises_on_placeholder_tensor_mismatch(token_ids, n_tensors):
    """A placeholder-token count that differs from the image-tensor count is a
    caller error and must raise ValueError, not silently mis-scatter."""
    tensors = [torch.ones(1, _HIDDEN, dtype=torch.float16)] * n_tensors
    with pytest.raises(ValueError):
        build_mixed_embeds(token_ids, tensors, _PLACEHOLDER)


def test_build_mixed_embeds_raises_on_empty_tensors():
    with pytest.raises(ValueError):
        build_mixed_embeds([1, _PLACEHOLDER, 2], [], _PLACEHOLDER)


def test_build_mixed_embeds_raises_on_bad_tensor_shape():
    """A 1D encoder tensor (missing the hidden dim) must raise before the row
    copy, not surface as an opaque RuntimeError."""
    with pytest.raises(ValueError):
        build_mixed_embeds(
            [1, _PLACEHOLDER, 2], [torch.ones(4, dtype=torch.float16)], _PLACEHOLDER
        )


def test_build_mixed_embeds_raises_on_empty_rows():
    """A (0, hidden) encoder tensor passes the 2D/hidden checks but would erase
    the image's placeholder run, silently dropping the image — must raise."""
    with pytest.raises(ValueError, match="0 rows"):
        build_mixed_embeds(
            [1, _PLACEHOLDER, 2],
            [torch.empty(0, _HIDDEN, dtype=torch.float16)],
            _PLACEHOLDER,
        )
