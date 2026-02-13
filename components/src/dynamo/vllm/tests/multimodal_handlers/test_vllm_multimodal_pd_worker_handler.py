# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultimodalPDWorkerHandler.__init__."""

from unittest.mock import MagicMock, patch

import pytest

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.multimodal_handlers import multimodal_pd_worker_handler as mod

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _make_config(
    model: str = "test-model",
    is_prefill_worker: bool = False,
    enable_multimodal: bool = True,
    multimodal_embedding_cache_capacity_gb: float = 0,
) -> MagicMock:
    """Create a mock Config with the fields used by MultimodalPDWorkerHandler.__init__."""
    config = MagicMock()
    config.model = model
    config.is_prefill_worker = is_prefill_worker
    config.enable_multimodal = enable_multimodal
    config.multimodal_embedding_cache_capacity_gb = (
        multimodal_embedding_cache_capacity_gb
    )
    config.engine_args.create_model_config.return_value.get_diff_sampling_param.return_value = (
        {}
    )
    return config


class TestMultimodalPDWorkerHandlerInit:
    """Tests for MultimodalPDWorkerHandler.__init__ focusing on embedding cache."""

    def test_init_with_embedding_cache(self):
        """When capacity > 0, a MultimodalEmbeddingCacheManager is created with correct byte size."""
        capacity_gb = 0.1
        config = _make_config(multimodal_embedding_cache_capacity_gb=capacity_gb)

        with (
            patch.object(mod.BaseWorkerHandler, "__init__", return_value=None),
            patch.object(mod, "ImageLoader", new_callable=MagicMock),
        ):
            handler = mod.MultimodalPDWorkerHandler(
                runtime=MagicMock(),
                component=MagicMock(),
                engine_client=MagicMock(),
                config=config,
            )

        assert isinstance(
            handler.embedding_cache_manager, MultimodalEmbeddingCacheManager
        )
        expected_bytes = int(capacity_gb * 1024**3)
        assert handler.embedding_cache_manager._capacity_bytes == expected_bytes
