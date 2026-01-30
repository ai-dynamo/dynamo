# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for request handler tests."""

from typing import List
from unittest.mock import MagicMock


def create_mock_encoder_cache() -> MagicMock:
    """Create mock EncoderCacheManager."""
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock(return_value=True)
    return cache


def create_mock_context(request_id: str = "test-id") -> MagicMock:
    """Create mock Context."""
    ctx = MagicMock()
    ctx.id = MagicMock(return_value=request_id)
    ctx.is_stopped = MagicMock(return_value=False)
    ctx.is_killed = MagicMock(return_value=False)
    return ctx


def setup_multimodal_config(config: MagicMock, image_urls: List[str]) -> None:
    """Configure multimodal_processor and encode_client on config."""
    config.multimodal_processor = MagicMock()
    config.multimodal_processor.extract_prompt_and_media = MagicMock(
        return_value=("text", image_urls, [])
    )
    config.encode_client = MagicMock()
