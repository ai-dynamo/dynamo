# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PrefillHandler."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from tensorrt_llm.llmapi import DisaggregatedParams

from dynamo.trtllm.request_handlers.handlers import PrefillHandler
from dynamo.trtllm.tests.utils import create_mock_request_handler_config

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
]


@pytest.fixture
def mock_config():
    """Create a mock RequestHandlerConfig."""
    return create_mock_request_handler_config(disaggregation_mode="prefill")


@pytest.fixture
def mock_encoder_cache():
    """Create a mock EncoderCacheManager."""
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock(return_value=True)
    return cache


@pytest.fixture
def mock_context():
    """Create a mock Context."""
    ctx = MagicMock()
    ctx.id = MagicMock(return_value="test-id")
    ctx.is_stopped = MagicMock(return_value=False)
    ctx.is_killed = MagicMock(return_value=False)
    return ctx


@pytest.fixture
def image_request() -> dict[str, Any]:
    """Create a request with one image URL."""
    return {
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


def setup_multimodal_config(mock_config):
    """Configure mock_config for multimodal requests."""
    mock_config.multimodal_processor = MagicMock()
    mock_config.multimodal_processor.extract_prompt_and_media = MagicMock(
        return_value=("text", ["http://example.com/image.jpg"], [])
    )
    mock_config.encode_client = MagicMock()


class TestPrefillHandlerInit:
    """Tests for PrefillHandler initialization."""

    def test_init_with_encoder_cache(self, mock_config, mock_encoder_cache):
        """Test PrefillHandler can be initialized with encoder_cache."""
        handler = PrefillHandler(mock_config, encoder_cache=mock_encoder_cache)

        assert handler.engine == mock_config.engine
        assert handler._encoder_cache == mock_encoder_cache


class TestPrefillHandlerGenerate:
    """Tests for PrefillHandler.generate method."""

    @pytest.mark.asyncio
    async def test_embeddings_passed_to_generate_locally(
        self, mock_config, mock_encoder_cache, mock_context, image_request
    ):
        """Test embeddings from fetch_embeddings_from_encoder passed to generate_locally."""
        setup_multimodal_config(mock_config)
        handler = PrefillHandler(mock_config, encoder_cache=mock_encoder_cache)

        expected_embeddings = [torch.randn(10, 256)]
        captured_embeddings = None

        async def mock_generate_locally(request, context, embeddings, ep_params):
            nonlocal captured_embeddings
            captured_embeddings = embeddings
            yield {"result": "mock"}

        with patch(
            "dynamo.trtllm.request_handlers.handlers.fetch_embeddings_from_encoder",
            new_callable=AsyncMock,
            return_value=expected_embeddings,
        ) as mock_fetch:
            with patch.object(handler, "generate_locally", mock_generate_locally):
                async for _ in handler.generate(image_request, mock_context):
                    pass

        mock_fetch.assert_called_once()
        assert captured_embeddings is expected_embeddings

    @pytest.mark.asyncio
    async def test_disaggregated_params_passed_to_generate_locally(
        self, mock_config, mock_context, image_request
    ):
        """Test DisaggregatedParams from fetch_embeddings_from_encoder passed to generate_locally."""
        setup_multimodal_config(mock_config)
        handler = PrefillHandler(mock_config, encoder_cache=None)

        expected_params = DisaggregatedParams(request_type="context_only")
        captured_ep_params = None

        async def mock_generate_locally(request, context, embeddings, ep_params):
            nonlocal captured_ep_params
            captured_ep_params = ep_params
            yield {"result": "mock"}

        with patch(
            "dynamo.trtllm.request_handlers.handlers.fetch_embeddings_from_encoder",
            new_callable=AsyncMock,
            return_value=expected_params,
        ) as mock_fetch:
            with patch.object(handler, "generate_locally", mock_generate_locally):
                async for _ in handler.generate(image_request, mock_context):
                    pass

        mock_fetch.assert_called_once()
        assert captured_ep_params is expected_params
