# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal unit tests for boolean validation of the dimensions parameter."""

from unittest.mock import MagicMock

import pytest

from dynamo.vllm.handlers import EmbeddingWorkerHandler

pytestmark = [pytest.mark.unit, pytest.mark.vllm, pytest.mark.pre_merge]


@pytest.mark.asyncio
async def test_embedding_handler_rejects_boolean_dimensions():
    """Verify that dimensions=True is rejected."""
    handler = EmbeddingWorkerHandler(MagicMock(), MagicMock(), MagicMock())
    request = {"input": "test", "dimensions": True}

    with pytest.raises(TypeError, match="Invalid 'dimensions' type bool"):
        async for _ in handler.generate(request, MagicMock()):
            pass
