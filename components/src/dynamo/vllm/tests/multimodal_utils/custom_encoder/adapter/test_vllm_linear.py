# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the linear custom encoder adapter.

build_mixed_embeds assembles the mixed token-ids/embeds EmbedsPrompt for the
aggregated CustomEncoder path: each placeholder token is expanded to its encoder
tensor's row count, image rows carry the encoder embeddings, and text rows stay
zero (vLLM fills them from the model's embedding table). These tests pin that
layout, the per-image expansion (including back-to-back images), and the input
validation that surfaces a bad encoder output as a clear ValueError.
"""

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder.adapter import (
    build_mixed_embeds,
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.backend import VisionEncoderBackend

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_HIDDEN = 8
_PLACEHOLDER = 999
_ADAPTER_PLACEHOLDER = 99


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


class _Backend(VisionEncoderBackend):
    image_token_id = _ADAPTER_PLACEHOLDER

    def build(self, model_id: str) -> None:
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


def _model_config(
    *,
    multimodal: bool = False,
    callable_flag: bool = False,
    architecture: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        architectures=[architecture] if architecture is not None else [],
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        is_multimodal_model=(lambda: multimodal) if callable_flag else multimodal,
    )


def _engine_args(*, enable_prompt_embeds: bool = True):
    return SimpleNamespace(enable_prompt_embeds=enable_prompt_embeds)


def test_text_decoder_selects_linear_adapter_and_builds_final_prompt():
    adapter = create_custom_encoder_adapter(_Backend(), _model_config(), _engine_args())

    prompt = adapter.prepare_prompt(
        [1, _ADAPTER_PLACEHOLDER, 2],
        [torch.ones((2, 4), dtype=torch.bfloat16)],
    )

    assert tuple(prompt["prompt_embeds"].shape) == (4, 4)
    assert prompt["prompt_token_ids"] == [
        1,
        _ADAPTER_PLACEHOLDER,
        _ADAPTER_PLACEHOLDER,
        2,
    ]
    assert prompt["prompt_is_token_ids"] == [True, False, False, True]


def test_text_only_qwen_decoder_selects_linear_adapter():
    adapter = create_custom_encoder_adapter(
        _Backend(),
        _model_config(architecture="Qwen2ForCausalLM"),
        _engine_args(),
    )

    assert type(adapter).__name__ == "LinearEmbedsAdapter"


def test_linear_adapter_requires_prompt_embeds_flag():
    with pytest.raises(ValueError, match="--enable-prompt-embeds"):
        create_custom_encoder_adapter(
            _Backend(),
            _model_config(),
            _engine_args(enable_prompt_embeds=False),
        )


def test_linear_adapter_requires_image_token_id():
    backend = _Backend()
    backend.image_token_id = None

    with pytest.raises(ValueError, match="image_token_id"):
        create_custom_encoder_adapter(backend, _model_config(), _engine_args())


def test_linear_adapter_rejects_multimodal_decoder():
    with pytest.raises(ValueError, match="multimodal decoder"):
        create_custom_encoder_adapter(
            _Backend(), _model_config(multimodal=True), _engine_args()
        )


def test_linear_adapter_calls_real_model_config_multimodal_method():
    adapter = create_custom_encoder_adapter(
        _Backend(), _model_config(callable_flag=True), _engine_args()
    )

    assert adapter is not None


@pytest.mark.parametrize(
    "encoding, match",
    [
        (torch.ones((2, 3), dtype=torch.bfloat16), "decoder hidden size 4"),
        (torch.ones((2, 4), dtype=torch.float16), "decoder dtype"),
        ("not-a-tensor", "must return tensors"),
    ],
)
def test_linear_adapter_validates_encoder_artifacts(encoding, match):
    adapter = create_custom_encoder_adapter(_Backend(), _model_config(), _engine_args())

    with pytest.raises((TypeError, ValueError), match=match):
        adapter.prepare_prompt([_ADAPTER_PLACEHOLDER], [encoding])
