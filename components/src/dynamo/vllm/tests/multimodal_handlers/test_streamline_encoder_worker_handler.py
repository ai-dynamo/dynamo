# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultimodalStreamlineEncoderWorkerHandler."""

from unittest.mock import Mock

import pytest

from dynamo.vllm.multimodal_handlers import MultimodalStreamlineEncoderWorkerHandler

# Import shared fixtures
from .utils import (
    mock_component,
    mock_config,
    mock_runtime,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
]


class TestMultimodalStreamlineEncoderWorkerHandler:
    """Test suite for MultimodalStreamlineEncoderWorkerHandler."""

    def test_initialization(self, mock_runtime, mock_component, mock_config):
        """Test that EncoderWorkerHandler initializes correctly."""
        handler = MultimodalStreamlineEncoderWorkerHandler(
            runtime=mock_runtime,
            component=mock_component,
            config=mock_config,
        )

        assert handler.runtime == mock_runtime
        assert handler.component == mock_component
        assert handler.config == mock_config
        assert handler._vision_model is None
        assert handler._image_processor is None
        assert handler._vision_encoder is None
        assert handler._projector is None
        assert handler.image_loader is not None

    @pytest.mark.asyncio
    async def test_encode_not_implemented(self, mock_runtime, mock_component, mock_config):
        """Test that encode method raises NotImplementedError."""
        handler = MultimodalStreamlineEncoderWorkerHandler(
            runtime=mock_runtime,
            component=mock_component,
            config=mock_config,
        )

        mock_request = Mock()
        mock_context = Mock()

        with pytest.raises(NotImplementedError, match="MultimodalStreamlineEncoderWorkerHandler.encode"):
            async for _ in handler.encode(mock_request, mock_context):
                pass