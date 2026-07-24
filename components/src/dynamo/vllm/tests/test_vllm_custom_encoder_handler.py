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
from dynamo.vllm.multimodal_utils.vision_encoder_backend import VisionEncoderBackend

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
