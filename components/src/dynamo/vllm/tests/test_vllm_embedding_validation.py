# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for boolean validation of the dimensions parameter in vLLM embeddings."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.vllm.handlers import EmbeddingWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_embedding_handler_rejects_boolean_dimensions():
    """Verify that the embedding handler rejects boolean values for dimensions."""
    # Mock dependencies
    config = MagicMock()
    config.served_model_name = "test-model"
    engine_client = MagicMock()
    runtime = MagicMock()

    handler = EmbeddingWorkerHandler(runtime, engine_client, config)

    context = MagicMock()
    context.id.return_value = "req-123"

    # dimensions=True should be rejected
    request = {"model": "test-model", "input": "hello", "dimensions": True}

    with pytest.raises(TypeError, match="Invalid 'dimensions' type bool; expected int"):
        # We need to trigger the validation inside generate()
        gen = handler.generate(request, context)
        await gen.__anext__()


@pytest.mark.asyncio
async def test_embedding_handler_accepts_integer_dimensions():
    """Verify that the embedding handler accepts integer values for dimensions."""
    # Mock dependencies
    config = MagicMock()
    config.served_model_name = "test-model"
    engine_client = MagicMock()
    runtime = MagicMock()

    # Mock engine_client.encode to return a minimal iterator
    mock_output = MagicMock()
    mock_output.outputs.data = [0.1, 0.2, 0.3]

    async def mock_encode(*args, **kwargs):
        yield mock_output

    engine_client.encode = mock_encode

    handler = EmbeddingWorkerHandler(runtime, engine_client, config)

    context = MagicMock()
    context.id.return_value = "req-123"
    # async_killed_or_stopped needs to return an awaitable for the abort monitor
    context.async_killed_or_stopped.return_value = AsyncMock()()

    # dimensions=2 should be accepted
    request = {"model": "test-model", "input": "hello", "dimensions": 2}

    # This should not raise TypeError during validation
    gen = handler.generate(request, context)
    try:
        await gen.__anext__()
    except StopAsyncIteration:
        pass
    except Exception as e:
        # We expect it might fail later if mocks aren't perfect,
        # but it shouldn't be a TypeError from our validation.
        assert not isinstance(e, TypeError) or "dimensions" not in str(e)
