# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RequestHandlerFactory."""

import pytest

from dynamo.trtllm.request_handlers.handlers import (
    AggregatedHandler,
    PrefillHandler,
    RequestHandlerFactory,
)
from dynamo.trtllm.tests.utils import create_mock_request_handler_config

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def mock_config():
    """Create a mock RequestHandlerConfig."""
    return create_mock_request_handler_config()


class TestRequestHandlerFactory:
    """Tests for RequestHandlerFactory."""

    def test_factory_creates_aggregated_handler(self, mock_config):
        """Test factory creates AggregatedHandler for prefill_and_decode mode."""
        factory = RequestHandlerFactory()
        handler = factory.get_request_handler(mock_config)

        assert isinstance(handler, AggregatedHandler)

    def test_factory_creates_prefill_handler(self, mock_config):
        """Test factory creates PrefillHandler for prefill mode."""
        mock_config.disaggregation_mode.value = "prefill"
        factory = RequestHandlerFactory()
        handler = factory.get_request_handler(mock_config)

        assert isinstance(handler, PrefillHandler)

    def test_factory_invalid_mode_raises(self, mock_config):
        """Test factory raises ValueError for invalid disaggregation_mode."""
        mock_config.disaggregation_mode.value = "invalid_mode"
        factory = RequestHandlerFactory()

        with pytest.raises(ValueError, match="Invalid disaggregation_mode"):
            factory.get_request_handler(mock_config)
