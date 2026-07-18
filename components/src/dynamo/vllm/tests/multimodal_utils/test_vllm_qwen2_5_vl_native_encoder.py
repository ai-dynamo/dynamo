# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the native Qwen2.5-VL custom encoder producer."""

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder_adapter import (
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.vision_encoder_backend import Qwen2VLImageEncoding
from examples.custom_encoder.qwen2_5_vl_benchmark_encoder import (
    Qwen2_5VLBenchmarkEncoder,
    Qwen2VLImageInputs,
)
from examples.custom_encoder.qwen2_5_vl_native_encoder import Qwen2_5VLNativeEncoder

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _item(grid: tuple[int, int, int]) -> Qwen2VLImageInputs:
    return Qwen2VLImageInputs(
        pixel_values=torch.zeros((grid[0] * grid[1] * grid[2], 6)),
        image_grid_thw=torch.tensor([grid], dtype=torch.long),
    )


def _model_config():
    return SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        is_multimodal_model=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            image_token_id=101,
            vision_start_token_id=100,
            vision_end_token_id=102,
            video_token_id=103,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
    )


def _engine_args():
    return SimpleNamespace(
        enable_mm_embeds=True,
        language_model_only=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=False),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )


def test_native_encoder_wraps_projected_rows_with_ordered_grids(monkeypatch):
    encoder = Qwen2_5VLNativeEncoder()
    items = [_item((1, 4, 8)), _item((1, 8, 4))]
    projected = [
        torch.full((8, 4), 1, dtype=torch.bfloat16),
        torch.full((8, 4), 2, dtype=torch.bfloat16),
    ]
    monkeypatch.setattr(
        Qwen2_5VLBenchmarkEncoder,
        "forward_batch",
        lambda self, values, target_bucket=None: projected,
    )

    outputs = encoder.forward_batch(items)

    assert encoder.output_format == "qwen2_vl_projected_grid"
    assert all(isinstance(output, Qwen2VLImageEncoding) for output in outputs)
    assert [output.grid_thw for output in outputs] == [(1, 4, 8), (1, 8, 4)]
    assert outputs[0].projected is projected[0]
    assert outputs[1].projected is projected[1]


def test_native_producer_and_consumer_adapter_build_final_prompt(monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.custom_encoder_adapter.version",
        lambda package: "0.25.1",
    )
    monkeypatch.setattr(
        Qwen2_5VLBenchmarkEncoder,
        "forward_batch",
        lambda self, values, target_bucket=None: [
            torch.full((8, 4), index + 1, dtype=torch.bfloat16)
            for index, _ in enumerate(values)
        ],
    )
    encoder = Qwen2_5VLNativeEncoder()
    items = [_item((1, 4, 8)), _item((1, 8, 4))]
    adapter = create_custom_encoder_adapter(encoder, _model_config(), _engine_args())

    prompt = adapter.prepare_prompt(
        [100, 101, 102, 7, 100, 101, 102],
        encoder.forward_batch(items),
    )

    image = prompt["multi_modal_data"]["image"]
    assert prompt["prompt_token_ids"] == [100, 101, 102, 7, 100, 101, 102]
    assert image["image_grid_thw"].tolist() == [[1, 4, 8], [1, 8, 4]]
    assert image["image_embeds"][:, 0].tolist() == [1] * 8 + [2] * 8
    assert set(prompt["multi_modal_data"]) == {"image"}
