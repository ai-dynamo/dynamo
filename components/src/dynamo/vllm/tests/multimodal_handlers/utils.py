# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mock fixtures and utilities for multimodal streamline handler tests."""

from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_runtime():
    """Create a mock DistributedRuntime."""
    runtime = Mock()
    runtime.component_name = "test_component"
    return runtime


@pytest.fixture
def mock_component():
    """Create a mock Component."""
    component = Mock()
    component.name = "test_component"
    return component


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = Mock()
    config.model = "test-model"
    config.enable_multimodal = True

    # Mock engine_args for handlers that inherit from BaseWorkerHandler
    mock_engine_args = Mock()
    mock_model_config = Mock()
    mock_model_config.get_diff_sampling_param = Mock(return_value={"temperature": 0.7})
    mock_engine_args.create_model_config = Mock(return_value=mock_model_config)
    config.engine_args = mock_engine_args

    return config


@pytest.fixture
def mock_engine_client():
    """Create a mock AsyncLLM engine client."""
    engine_client = AsyncMock()
    return engine_client


@pytest.fixture
def mock_encoder_client():
    """Create a mock encoder worker client."""
    client = Mock()
    return client


@pytest.fixture
def mock_decode_client():
    """Create a mock decode worker client."""
    client = Mock()
    return client
