# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PrefillHandler."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.trtllm.request_handlers.handlers import PrefillHandler
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
    return create_mock_request_handler_config(disaggregation_mode="prefill")


@pytest.fixture
def mock_encoder_cache():
    """Create a mock AsyncEncoderCache."""
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.get_or_compute = AsyncMock()
    cache.stats = {"hits": 0, "misses": 0, "in_flight": 0}
    return cache


class TestPrefillHandlerInit:
    """Tests for PrefillHandler initialization."""

    def test_init_with_encoder_cache(self, mock_config, mock_encoder_cache):
        """Test PrefillHandler can be initialized with encoder_cache."""
        handler = PrefillHandler(mock_config, encoder_cache=mock_encoder_cache)

        assert handler.engine == mock_config.engine
