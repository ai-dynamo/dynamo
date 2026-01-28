# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RequestHandlerFactory."""

import pytest

from dynamo.common.memory.encoder_cache_manager import EncoderCacheManager
from dynamo.trtllm.request_handlers.handlers import (
    AggregatedHandler,
    PrefillHandler,
    RequestHandlerFactory,
)
from dynamo.trtllm.tests.utils import create_mock_request_handler_config

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
]


@pytest.fixture
def mock_config():
    """Create a mock RequestHandlerConfig."""
    return create_mock_request_handler_config()


class TestRequestHandlerFactory:
    """Tests for RequestHandlerFactory."""

    def test_creates_aggregated_handler(self, mock_config):
        """Test factory creates AggregatedHandler for prefill_and_decode mode."""
        factory = RequestHandlerFactory()
        handler = factory.get_request_handler(mock_config)

        assert isinstance(handler, AggregatedHandler)

    def test_creates_prefill_handler(self, mock_config):
        """Test factory creates PrefillHandler for prefill mode."""
        mock_config.disaggregation_mode.value = "prefill"
        factory = RequestHandlerFactory()
        handler = factory.get_request_handler(mock_config)

        assert isinstance(handler, PrefillHandler)

    def test_invalid_mode_raises(self, mock_config):
        """Test factory raises ValueError for invalid disaggregation_mode."""
        mock_config.disaggregation_mode.value = "invalid_mode"
        factory = RequestHandlerFactory()

        with pytest.raises(ValueError, match="Invalid disaggregation_mode"):
            factory.get_request_handler(mock_config)

    def test_prefill_handler_with_encoder_cache(self):
        """Test factory creates PrefillHandler with EncoderCacheManager when capacity > 0."""
        mock_config = create_mock_request_handler_config(
            disaggregation_mode="prefill",
            encoder_cache_capacity_gb=1.0,
        )
        factory = RequestHandlerFactory()
        handler = factory.get_request_handler(mock_config)

        assert isinstance(handler, PrefillHandler)
        assert isinstance(handler._encoder_cache, EncoderCacheManager)

    def test_prefill_handler_without_encoder_cache(self):
        """Test factory creates PrefillHandler with no cache when capacity is 0."""
        mock_config = create_mock_request_handler_config(
            disaggregation_mode="prefill",
            encoder_cache_capacity_gb=0,
        )
        factory = RequestHandlerFactory()
        handler = factory.get_request_handler(mock_config)

        assert isinstance(handler, PrefillHandler)
        assert handler._encoder_cache is None
