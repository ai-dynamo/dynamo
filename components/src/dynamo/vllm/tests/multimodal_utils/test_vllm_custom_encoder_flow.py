# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interaction test: a VisionEncoderBackend feeding build_mixed_embeds.

The unit tests cover each side with the other mocked out — the backend test never
touches the assembler, and the assembler test hand-builds tensors instead of a
backend — so neither pins the seam. This walks the whole contract for one request
(build -> preprocess -> forward_batch -> build_mixed_embeds) and doubles as a
worked example of how the APIs fit: forward_batch emits one CPU
(n_tokens, hidden) tensor per image, and build_mixed_embeds splices them at the
one-placeholder-token-per-image positions.
"""

import pytest
import torch

from dynamo.vllm.multimodal_utils.embed_assembler import build_mixed_embeds
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

_HIDDEN = 4
_IMG = 151655  # the toy backend's hardcoded image placeholder token id


class _ToyEncoder(VisionEncoderBackend):
    """Stand-in backend: one visual token per input char, distinct per-image embeds."""

    image_token_id = _IMG

    def build(self, model_id):
        ...

    def preprocess(self, raw):
        n = len(raw)
        return Preprocessed(item=(raw, n), cost=n)

    def forward_batch(self, items, target_bucket=None):
        # Distinct non-zero fill per image (1.0, 2.0, ...) so the splice is
        # unambiguous against the zero-filled text rows. CPU, per the contract.
        return [
            torch.full((n, _HIDDEN), float(i + 1)) for i, (_, n) in enumerate(items)
        ]


def test_backend_and_assembler_work_together():
    """Worked example of the whole contract for one two-image request."""
    enc = _ToyEncoder()
    enc.build("toy-model")

    # Caller flow: preprocess each raw off-thread, then one batched forward.
    pre = [enc.preprocess(r) for r in ("ab", "cde")]
    assert [p.cost for p in pre] == [2, 3]
    img_tensors = enc.forward_batch([p.item for p in pre])
    assert [tuple(t.shape) for t in img_tensors] == [(2, _HIDDEN), (3, _HIDDEN)]

    # Prompt: 10 <img> 11 12 <img> — one placeholder token per image.
    prompt_embeds, out_ids, is_token = build_mixed_embeds(
        [10, _IMG, 11, 12, _IMG], img_tensors, enc.image_token_id
    )

    # Each placeholder is expanded to its image's row count (2 and 3).
    assert out_ids == [10, _IMG, _IMG, 11, 12, _IMG, _IMG, _IMG]
    assert is_token == [True, False, False, True, True, False, False, False]
    # Image spans carry the encoder embeds; text rows stay zero.
    assert torch.equal(prompt_embeds[1:3], torch.full((2, _HIDDEN), 1.0))
    assert torch.equal(prompt_embeds[5:8], torch.full((3, _HIDDEN), 2.0))
    assert torch.equal(prompt_embeds[0], torch.zeros(_HIDDEN))
