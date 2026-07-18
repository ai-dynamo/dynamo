# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder_adapter import (
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.vision_encoder_backend import VisionEncoderBackend

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_IMAGE_TOKEN_ID = 99


class _Backend(VisionEncoderBackend):
    image_token_id = _IMAGE_TOKEN_ID

    def build(self, model_id: str) -> None:
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


def _model_config(*, multimodal: bool = False, callable_flag: bool = False):
    return SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        is_multimodal_model=(lambda: multimodal) if callable_flag else multimodal,
    )


def _engine_args(*, enable_prompt_embeds: bool = True):
    return SimpleNamespace(enable_prompt_embeds=enable_prompt_embeds)


def test_text_decoder_selects_linear_adapter_and_builds_final_prompt():
    adapter = create_custom_encoder_adapter(_Backend(), _model_config(), _engine_args())

    prompt = adapter.prepare_prompt(
        [1, _IMAGE_TOKEN_ID, 2],
        [torch.ones((2, 4), dtype=torch.bfloat16)],
    )

    assert tuple(prompt["prompt_embeds"].shape) == (4, 4)
    assert prompt["prompt_token_ids"] == [1, 99, 99, 2]
    assert prompt["prompt_is_token_ids"] == [True, False, False, True]


def test_linear_adapter_requires_prompt_embeds_flag():
    with pytest.raises(ValueError, match="--enable-prompt-embeds"):
        create_custom_encoder_adapter(
            _Backend(),
            _model_config(),
            _engine_args(enable_prompt_embeds=False),
        )


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
        adapter.prepare_prompt([_IMAGE_TOKEN_ID], [encoding])
