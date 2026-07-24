# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""process_openai_request must let client-error types from image loading
propagate (so the frontend returns a 4xx) instead of swallowing them to None."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

# multimodal_processor imports tensorrt_llm, whose import needs CUDA. -m selection
# happens after collection, so this module is still imported on the CPU-only step;
# skip at collection there to avoid an ImportError. GPU runs it via the gpu_1 marker.
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )

from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import UrlValidationError
from dynamo.trtllm import multimodal_processor as mmp
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        UrlValidationError("blocked IP literal"),
        HttpStatusError(415, "Unsupported Media Type", "https://example.com/x.png"),
    ],
)
async def test_client_errors_propagate(error) -> None:
    # Mock tokenizer skips tokenizer_factory; mock loader forces the failure.
    processor = MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )
    processor.image_loader.load_image_batch = AsyncMock(side_effect=error)

    request = {
        "multi_modal_data": {"image_url": [{"Url": "https://example.com/x.png"}]}
    }
    with pytest.raises(type(error)):
        await processor.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )


@pytest.mark.asyncio
async def test_internal_video_uses_dynamo_fetcher_when_allowed(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    processor = MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )
    fetch = AsyncMock(return_value=b"video bytes")
    load_video = AsyncMock(return_value=object())
    monkeypatch.setattr(mmp, "fetch_bytes", fetch, raising=False)
    monkeypatch.setattr(mmp, "async_load_video", load_video)
    url = "http://169.254.169.254/latest/meta-data/"

    await processor.process_openai_request(
        {
            "multi_modal_data": {"video_url": [{"Url": url}]},
            "token_ids": [1],
        },
        embeddings=None,
        ep_disaggregated_params=None,
    )

    fetch.assert_awaited_once_with(url, 30.0, policy=processor._url_policy)
    assert load_video.await_args.args[0] != url
