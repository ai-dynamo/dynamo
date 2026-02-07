# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for request handler tests."""

from typing import Any, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch


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


async def run_generate_with_mock_fetch(
    handler: Any,
    fetch_patch_path: str,
    mock_return_value: Any,
) -> Tuple[Any, Any]:
    """
    Run handler.generate() with mocked fetch_embeddings_from_encoder.

    Args:
        handler: Handler instance (PrefillHandler or AggregatedHandler)
        fetch_patch_path: Full path to patch fetch_embeddings_from_encoder
        mock_return_value: Value to return from mocked fetch

    Returns:
        Tuple of (captured_embeddings, captured_ep_params)
    """
    captured_embeddings = None
    captured_ep_params = None

    async def mock_generate_locally(request, context, embeddings, ep_params):
        nonlocal captured_embeddings, captured_ep_params
        captured_embeddings = embeddings
        captured_ep_params = ep_params
        yield {"result": "mock"}

    request: dict[str, Any] = {"messages": []}

    with patch(
        fetch_patch_path,
        new_callable=AsyncMock,
        return_value=mock_return_value,
    ) as mock_fetch:
        with patch.object(handler, "generate_locally", mock_generate_locally):
            async for _ in handler.generate(request, create_mock_context()):
                pass

    mock_fetch.assert_called_once()
    return captured_embeddings, captured_ep_params
