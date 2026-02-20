# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMultimodalEmbeddingCacheConnector."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from dynamo.vllm.multimodal_utils import multimodal_embedding_cache_connector as mod

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _make_vllm_config(capacity_gb: float = 1.0) -> MagicMock:
    config = MagicMock()
    config.ec_transfer_config.ec_connector_extra_config = {
        "multimodal_embedding_cache_capacity_gb": capacity_gb,
    }
    config.model_config.get_hidden_size.return_value = 4096
    config.model_config.dtype = torch.float16
    return config


class TestVersionCheck:
    def test_warns_old_vllm(self):
        with (
            patch.object(mod, "_vllm_version", "0.16.5"),
            patch.object(mod.ECConnectorBase, "__init__", return_value=None),
            patch.object(mod.logger, "warning") as mock_warn,
        ):
            connector = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(),
                role=MagicMock(),
            )
            assert connector is not None
            mock_warn.assert_called_once()
            assert mock_warn.call_args[0][1] == mod.MINIMUM_VLLM_VERSION
            assert mock_warn.call_args[0][2] == "0.16.5"


class TestSchedulerSideLRU:
    """Test the scheduler-side logical LRU cache and metadata generation."""

    def _make_connector(self, capacity_gb: float = 1.0):
        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            return mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(capacity_gb),
                role=MagicMock(),
            )

    def _make_request(self, hashes_and_embeds: list[tuple[str, int]]) -> MagicMock:
        request = MagicMock()
        features = []
        for h, _ in hashes_and_embeds:
            f = MagicMock()
            f.identifier = h
            features.append(f)
        request.mm_features = features

        def get_num_encoder_embeds(idx):
            return hashes_and_embeds[idx][1]

        request.get_num_encoder_embeds = get_num_encoder_embeds
        return request

    def test_has_cache_item_miss_then_hit(self):
        conn = self._make_connector()
        assert not conn.has_cache_item("hash_a")

        request = self._make_request([("hash_a", 100)])
        conn.update_state_after_alloc(request, 0)

        assert conn.has_cache_item("hash_a")

    def test_update_state_plans_save(self):
        conn = self._make_connector()
        request = self._make_request([("hash_a", 100)])
        conn.update_state_after_alloc(request, 0)

        scheduler_output = MagicMock()
        meta = conn.build_connector_meta(scheduler_output)
        assert isinstance(meta, mod.MultimodalEmbeddingCacheConnectorMetadata)
        assert "hash_a" in meta.saves
        assert meta.loads == []
        assert meta.evicts == []

    def test_update_state_plans_load_for_cached(self):
        conn = self._make_connector()
        request = self._make_request([("hash_a", 100)])

        conn.update_state_after_alloc(request, 0)
        conn.build_connector_meta(MagicMock())

        conn.update_state_after_alloc(request, 0)
        meta = conn.build_connector_meta(MagicMock())
        assert "hash_a" in meta.loads
        assert meta.saves == []

    def test_eviction_under_pressure(self):
        # 4096 hidden_size * 2 bytes (fp16) = 8192 bytes per embed
        conn = self._make_connector()
        bpe = conn._bytes_per_embed  # 8192
        # Set capacity to hold exactly 200 embeds worth of bytes
        conn._capacity_bytes = 200 * bpe

        req_a = self._make_request([("hash_a", 100)])
        conn.update_state_after_alloc(req_a, 0)
        conn.build_connector_meta(MagicMock())

        req_b = self._make_request([("hash_b", 100)])
        conn.update_state_after_alloc(req_b, 0)
        conn.build_connector_meta(MagicMock())

        assert conn._num_used_bytes == 200 * bpe

        # Adding hash_c (100 embeds) should evict hash_a (LRU)
        req_c = self._make_request([("hash_c", 100)])
        conn.update_state_after_alloc(req_c, 0)
        meta = conn.build_connector_meta(MagicMock())

        assert "hash_c" in meta.saves
        assert "hash_a" in meta.evicts
        assert "hash_a" not in conn._cache_order
        assert "hash_c" in conn._cache_order

    def test_skip_oversized_item(self):
        conn = self._make_connector()
        bpe = conn._bytes_per_embed
        conn._capacity_bytes = 50 * bpe

        request = self._make_request([("huge_hash", 100)])
        conn.update_state_after_alloc(request, 0)
        meta = conn.build_connector_meta(MagicMock())

        assert meta.saves == []
        assert meta.loads == []
        assert "huge_hash" not in conn._cache_order
