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


def _adapter(architecture: str = "Qwen3VLForConditionalGeneration"):
    return create_custom_encoder_adapter(
        _QwenBackend(),
        SimpleNamespace(
            is_multimodal_model=True,
            architectures=[architecture],
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2)
            ),
        ),
        SimpleNamespace(),
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
    ],
)
def test_qwen3_vl_decoder_selects_native_adapter(architecture):
    assert type(_adapter(architecture)).__name__ == "Qwen3VLNativeAdapter"


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


def test_qwen3_vl_adapter_rejects_empty_artifacts():
    with pytest.raises(ValueError, match="must not be empty"):
        _adapter().prepare_prompt([100, 101, 102], [])


@pytest.mark.parametrize(
    "artifact, match",
    [
        ("not-an-encoding", "must return Qwen3VLImageEncoding"),
        (
            Qwen3VLImageEncoding(torch.empty((0, 8)), (1, 2, 2)),
            "no embedding rows",
        ),
        (
            Qwen3VLImageEncoding(torch.empty((1, 8), device="meta"), (1, 2, 2)),
            "must be on CPU",
        ),
        (
            Qwen3VLImageEncoding(torch.empty((1, 8)), (1, 3, 2)),
            "must be divisible",
        ),
        (
            Qwen3VLImageEncoding(torch.empty((2, 8)), (1, 2, 2)),
            "grid .* requires 1",
        ),
    ],
)
def test_qwen3_vl_adapter_validates_artifacts(artifact, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _adapter().prepare_prompt([100, 101, 102], [artifact])
