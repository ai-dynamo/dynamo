# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for rejecting vLLM-only multimodal cache UUIDs."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping because TensorRT-LLM imports require a CUDA-capable GPU.",
        allow_module_level=True,
    )

from dynamo.trtllm.request_handlers.aggregated_handler import AggregatedHandler
from dynamo.trtllm.request_handlers.handler_base import HandlerBase
from dynamo.trtllm.request_handlers.handlers import EncodeHandler, PrefillHandler
from dynamo.trtllm.tests.request_handlers.utils import create_mock_context
from dynamo.trtllm.tests.utils import create_mock_request_handler_config

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_type", "disaggregation_mode"),
    [
        (EncodeHandler, "encode"),
        (PrefillHandler, "prefill"),
        (AggregatedHandler, "prefill_and_decode"),
    ],
)
async def test_handler_rejects_multimodal_cache_uuid(
    handler_type: type[HandlerBase], disaggregation_mode: str
) -> None:
    config = create_mock_request_handler_config(disaggregation_mode=disaggregation_mode)
    handler = handler_type(config)
    request = {"multi_modal_uuids": {"image_url": ["cached-image"]}}

    with pytest.raises(ValueError, match="supported only by the vLLM backend"):
        async for _ in handler.generate(request, create_mock_context()):
            pass
