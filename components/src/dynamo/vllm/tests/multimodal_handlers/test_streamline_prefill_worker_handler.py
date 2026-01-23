# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultimodalStreamlinePrefillWorkerHandler."""

from unittest.mock import Mock, patch

import pytest

from dynamo.vllm.multimodal_handlers import MultimodalStreamlinePrefillWorkerHandler

# Import shared fixtures
from .utils import (
    mock_component,
    mock_config,
    mock_decode_client,
    mock_encoder_client,
    mock_engine_client,
    mock_runtime,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
]


class TestMultimodalStreamlinePrefillWorkerHandler:
    """Test suite for MultimodalStreamlinePrefillWorkerHandler."""

    @patch("dynamo.vllm.multimodal_handlers.multimodal_streamline_prefill_worker_handler.BaseWorkerHandler.__init__")
    def test_initialization(
        self,
        mock_base_init,
        mock_runtime,
        mock_component,
        mock_engine_client,
        mock_config,
        mock_encoder_client,
        mock_decode_client,
    ):
        """Test that PrefillWorkerHandler initializes correctly."""
        mock_base_init.return_value = None

        handler = MultimodalStreamlinePrefillWorkerHandler(
            runtime=mock_runtime,
            component=mock_component,
            engine_client=mock_engine_client,
            config=mock_config,
            encoder_worker_client=mock_encoder_client,
            decode_worker_client=mock_decode_client,
        )

        # Verify BaseWorkerHandler.__init__ was called with correct parameters
        mock_base_init.assert_called_once()
        call_args = mock_base_init.call_args
        assert call_args[0][0] == mock_runtime
        assert call_args[0][1] == mock_component
        assert call_args[0][2] == mock_engine_client
        assert call_args[1]["enable_multimodal"] is True

        # Verify handler attributes
        assert handler.config == mock_config
        assert handler.encoder_worker_client == mock_encoder_client
        assert handler.decode_worker_client == mock_decode_client

    @patch("dynamo.vllm.multimodal_handlers.multimodal_streamline_prefill_worker_handler.BaseWorkerHandler.__init__")
    @pytest.mark.asyncio
    async def test_generate_not_implemented(
        self,
        mock_base_init,
        mock_runtime,
        mock_component,
        mock_engine_client,
        mock_config,
    ):
        """Test that generate method raises NotImplementedError."""
        mock_base_init.return_value = None

        handler = MultimodalStreamlinePrefillWorkerHandler(
            runtime=mock_runtime,
            component=mock_component,
            engine_client=mock_engine_client,
            config=mock_config,
        )

        mock_request = Mock()
        mock_context = Mock()

        with pytest.raises(NotImplementedError, match="MultimodalStreamlinePrefillWorkerHandler.generate"):
            async for _ in handler.generate(mock_request, mock_context):
                pass