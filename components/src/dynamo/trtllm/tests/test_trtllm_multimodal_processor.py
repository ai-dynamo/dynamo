# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error-propagation contract for MultimodalRequestProcessor.process_openai_request.

The PD image-loading branch must let client-error types (a rejected media URL
or an unsupported-media HTTP status) propagate so the frontend maps them to a
4xx, while still swallowing genuine server/transport failures into a None
(degrade-gracefully) so they surface as a 500 upstream.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import UrlValidationError
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
]

_BLOCKED_URL = "https://169.254.169.254/latest/meta-data/"
# PD flow: pre-tokenized by the Rust frontend, images loaded here by URL.
_PD_IMAGE_REQUEST = {"multi_modal_data": {"image_url": [{"Url": _BLOCKED_URL}]}}


def _make_processor() -> MultimodalRequestProcessor:
    # A mock tokenizer lets __init__ skip tokenizer_factory (no model load).
    return MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused-with-mock-tokenizer",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        UrlValidationError("IP literal '169.254.169.254' is in a blocked range"),
        HttpStatusError(415, "Unsupported Media Type", _BLOCKED_URL),
    ],
)
async def test_process_openai_request_propagates_client_errors(error) -> None:
    """A client-error from load_image_batch must propagate out of
    process_openai_request instead of being swallowed into a silent None."""
    processor = _make_processor()
    processor.image_loader.load_image_batch = AsyncMock(side_effect=error)

    with pytest.raises(type(error)):
        await processor.process_openai_request(
            _PD_IMAGE_REQUEST, embeddings=None, ep_disaggregated_params=None
        )


@pytest.mark.asyncio
async def test_process_openai_request_swallows_generic_load_failure() -> None:
    """A generic (server/transport) load failure is still swallowed to None,
    preserving the pre-existing degrade-gracefully behavior for non-client errors."""
    processor = _make_processor()
    processor.image_loader.load_image_batch = AsyncMock(
        side_effect=RuntimeError("nixl transfer failed")
    )

    result = await processor.process_openai_request(
        _PD_IMAGE_REQUEST, embeddings=None, ep_disaggregated_params=None
    )
    assert result is None
