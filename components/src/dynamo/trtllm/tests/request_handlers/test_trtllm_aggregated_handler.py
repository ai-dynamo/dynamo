# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AggregatedHandler."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.trtllm.request_handlers.aggregated_handler import AggregatedHandler
from dynamo.trtllm.tests.request_handlers.utils import (
    create_mock_context,
    create_mock_encoder_cache,
    setup_multimodal_config,
)
from dynamo.trtllm.tests.utils import create_mock_request_handler_config

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
]


class TestAggregatedHandlerGenerate:
    """Tests for AggregatedHandler.generate method."""

    @pytest.mark.asyncio
    async def test_fetch_embeddings_called_with_image_urls(self):
        """fetch_embeddings_from_encoder called when image URLs present."""
        config = create_mock_request_handler_config(
            disaggregation_mode="prefill_and_decode"
        )
        setup_multimodal_config(config, ["http://example.com/image.jpg"])
        encoder_cache = create_mock_encoder_cache()

        handler = AggregatedHandler(config, encoder_cache=encoder_cache)
        ctx = create_mock_context()

        request: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://example.com/image.jpg"},
                        },
                    ],
                }
            ]
        }

        async def mock_generate_locally(req, ctx, embeddings):
            yield {"result": "mock"}

        with patch(
            "dynamo.trtllm.request_handlers.aggregated_handler.fetch_embeddings_from_encoder",
            new_callable=AsyncMock,
            return_value=[MagicMock()],
        ) as mock_fetch:
            with patch.object(handler, "generate_locally", mock_generate_locally):
                async for _ in handler.generate(request, ctx):
                    pass

        mock_fetch.assert_called_once()
