# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder import (
    Qwen3VLImageEncoding,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _QwenBackend(VisionEncoderBackend):
    def build(self, model_id: str) -> None:
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


def _adapter():
    return create_custom_encoder_adapter(
        _QwenBackend(),
        SimpleNamespace(
            is_multimodal_model=True,
            hf_config=SimpleNamespace(
                architectures=["Qwen3VLForConditionalGeneration"]
            ),
        ),
        SimpleNamespace(),
    )


def test_qwen3_vl_decoder_selects_native_adapter():
    assert type(_adapter()).__name__ == "Qwen3VLNativeAdapter"


def test_qwen3_vl_adapter_builds_tokens_prompt_in_image_order():
    token_ids = [100, 101, 102, 7, 100, 101, 102]
    first = Qwen3VLImageEncoding(
        torch.full((1, 8), 1, dtype=torch.bfloat16),
        (1, 2, 2),
    )
    second = Qwen3VLImageEncoding(
        torch.full((2, 8), 2, dtype=torch.bfloat16),
        (1, 2, 4),
    )

    prompt = _adapter().prepare_prompt(token_ids, [first, second])

    assert prompt["prompt_token_ids"] == token_ids
    image = prompt["multi_modal_data"]["image"]
    assert image["image_embeds"].shape == (3, 8)
    assert image["image_embeds"][:, 0].tolist() == [1, 2, 2]
    assert image["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]
