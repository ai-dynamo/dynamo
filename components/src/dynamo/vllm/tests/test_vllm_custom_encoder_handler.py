# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from dynamo.vllm.handlers import DecodeWorkerHandler
from dynamo.vllm.multimodal_utils.custom_encoder_adapter import (
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    Qwen2VLImageEncoding,
    VisionEncoderBackend,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _Backend(VisionEncoderBackend):
    image_token_id = 99

    def build(self, model_id: str) -> None:
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


class _QwenBackend(_Backend):
    output_format = "qwen2_vl_projected_grid"
    image_token_id = None


def _adapter():
    return create_custom_encoder_adapter(
        _Backend(),
        SimpleNamespace(
            dtype=torch.bfloat16,
            get_hidden_size=lambda: 4,
            is_multimodal_model=False,
        ),
        SimpleNamespace(enable_prompt_embeds=True),
    )


def _qwen_adapter(monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.custom_encoder_adapter.version",
        lambda package: "0.25.1",
    )
    return create_custom_encoder_adapter(
        _QwenBackend(),
        SimpleNamespace(
            dtype=torch.bfloat16,
            get_hidden_size=lambda: 4,
            is_multimodal_model=lambda: True,
            hf_config=SimpleNamespace(
                architectures=["Qwen2_5_VLForConditionalGeneration"],
                image_token_id=101,
                vision_start_token_id=100,
                vision_end_token_id=102,
                video_token_id=103,
                vision_config=SimpleNamespace(spatial_merge_size=2),
            ),
        ),
        SimpleNamespace(
            enable_mm_embeds=True,
            language_model_only=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            compilation_config=SimpleNamespace(cudagraph_mm_encoder=False),
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
        ),
    )


async def test_custom_encoder_handler_returns_adapter_prepared_prompt():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(return_value=[torch.ones((2, 4), dtype=torch.bfloat16)])
    )

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [1, 99, 2],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert error is None
    assert prompt is not None
    assert tuple(prompt["prompt_embeds"].shape) == (4, 4)
    assert prompt["prompt_token_ids"] == [1, 99, 99, 2]


async def test_custom_encoder_handler_preserves_string_error_contract():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(side_effect=RuntimeError("encoder failed"))
    )

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [99],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert prompt is None
    assert error is not None
    assert error["finish_reason"] == "error: CustomEncoder failed: encoder failed"


async def test_custom_encoder_handler_returns_native_qwen_prompt(monkeypatch):
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _qwen_adapter(monkeypatch)
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(
            return_value=[
                Qwen2VLImageEncoding(
                    torch.zeros((1, 4), dtype=torch.bfloat16), (1, 2, 2)
                )
            ]
        )
    )

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [100, 101, 102],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert error is None
    assert prompt is not None
    assert prompt["prompt_token_ids"] == [100, 101, 102]
    image = prompt["multi_modal_data"]["image"]
    assert image["image_embeds"].shape == (1, 4)
    assert image["image_grid_thw"].tolist() == [[1, 2, 2]]


async def test_custom_encoder_handler_passes_processor_kwargs_to_adapter(monkeypatch):
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _qwen_adapter(monkeypatch)
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(
            return_value=[
                Qwen2VLImageEncoding(
                    torch.zeros((1, 4), dtype=torch.bfloat16), (1, 2, 2)
                )
            ]
        )
    )

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [100, 101, 102],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
            "mm_processor_kwargs": {"min_pixels": 1},
        },
        "request-id",
    )

    assert prompt is None
    assert error is not None
    assert "mm_processor_kwargs" in error["finish_reason"]
