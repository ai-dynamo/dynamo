# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler tests for custom-encoder prompt-plan selection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from dynamo.vllm.handlers import DecodeWorkerHandler
from dynamo.vllm.multimodal_utils.custom_encoder_adapter import (
    BoundCustomEncoderAdapter,
    NativeMMPlan,
)
from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    BackendEncodingSpecV1,
    Qwen2VLImageEncodingV1,
    VisionEncoderBackend,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _QwenBackend(VisionEncoderBackend):
    encoding_spec = BackendEncodingSpecV1(
        adapter_abi="vllm-qwen2-vl-external-v1",
        producer_fingerprint="handler-test-v1",
        expected_decoder_config_fingerprint=(
            "model=None:revision=None:Qwen2_5_VLForConditionalGeneration:"
            "hidden=4:merge=2:dtype=torch.bfloat16"
        ),
        output_dtype="bfloat16",
        hidden_size=4,
        spatial_merge_size=2,
    )

    def build(self, model_id):
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


def _adapter() -> BoundCustomEncoderAdapter:
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            image_token_id=101,
            vision_start_token_id=100,
            vision_end_token_id=102,
            video_token_id=103,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
    )
    engine_args = SimpleNamespace(
        enable_mm_embeds=True,
        enable_prompt_embeds=False,
        language_model_only=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=False),
    )
    return BoundCustomEncoderAdapter(_QwenBackend(), model_config, engine_args)


async def test_native_qwen_custom_encoder_prepares_external_mm_plan():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(
            return_value=[
                Qwen2VLImageEncodingV1(
                    torch.zeros((1, 4), dtype=torch.bfloat16),
                    (1, 2, 2),
                )
            ]
        )
    )

    plan, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [100, 101, 102],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
        None,
    )

    assert error is None
    assert isinstance(plan, NativeMMPlan)
    assert plan.multi_modal_data["image"]["image_grid_thw"].tolist() == [[1, 2, 2]]


async def test_native_qwen_custom_encoder_rejects_processor_kwargs():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(encode=AsyncMock())

    plan, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [100, 101, 102],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
            "mm_processor_kwargs": {"min_pixels": 1},
        },
        "request-id",
        None,
    )

    assert plan is None
    assert error is not None
    assert "mm_processor_kwargs" in error["finish_reason"]["error"]
    handler._custom_encoder.encode.assert_not_awaited()
