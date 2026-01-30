# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PrefillHandler."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import torch

from dynamo.trtllm.request_handlers.handlers import PrefillHandler
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


class TestPrefillHandlerInit:
    """Tests for PrefillHandler initialization."""

    def test_init_with_encoder_cache(self):
        """Test PrefillHandler can be initialized with encoder_cache."""
        config = create_mock_request_handler_config(disaggregation_mode="prefill")
        cache = create_mock_encoder_cache()

        handler = PrefillHandler(config, encoder_cache=cache)

        assert handler.engine == config.engine
        assert handler._encoder_cache == cache


class TestPrefillHandlerGenerate:
    """Tests for PrefillHandler.generate method."""

    @pytest.mark.asyncio
    async def test_embeddings_passed_to_generate_locally(self):
        """Test embeddings from fetch_embeddings_from_encoder passed to generate_locally."""
        config = create_mock_request_handler_config(disaggregation_mode="prefill")
        setup_multimodal_config(config, ["http://example.com/image.jpg"])

        handler = PrefillHandler(config, encoder_cache=create_mock_encoder_cache())

        expected_embeddings = [torch.randn(10, 256)]
        captured_embeddings = None

        async def mock_generate_locally(request, context, embeddings, ep_params):
            nonlocal captured_embeddings
            captured_embeddings = embeddings
            yield {"result": "mock"}

        request: dict[str, Any] = {"messages": []}

        with patch(
            "dynamo.trtllm.request_handlers.handlers.fetch_embeddings_from_encoder",
            new_callable=AsyncMock,
            return_value=expected_embeddings,
        ) as mock_fetch:
            with patch.object(handler, "generate_locally", mock_generate_locally):
                async for _ in handler.generate(request, create_mock_context()):
                    pass

        mock_fetch.assert_called_once()
        assert captured_embeddings is expected_embeddings
