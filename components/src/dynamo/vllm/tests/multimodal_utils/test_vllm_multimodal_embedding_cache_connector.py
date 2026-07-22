# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMultimodalEmbeddingCacheConnector."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from dynamo.vllm.multimodal_utils import cache_config as cache_config_mod
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


class TestCacheConfiguration:
    def test_disabled_capacity_leaves_engine_args_unchanged(self):
        engine_args = SimpleNamespace()

        cache_config_mod.configure_multimodal_embedding_cache(
            engine_args,
            route_to_encoder=False,
            capacity_gb=0,
            namespace="dynamo",
            component="backend",
        )

        assert not hasattr(engine_args, "ec_transfer_config")

    def test_encoder_routing_leaves_engine_args_unchanged(self):
        engine_args = SimpleNamespace()

        cache_config_mod.configure_multimodal_embedding_cache(
            engine_args,
            route_to_encoder=True,
            capacity_gb=1,
            namespace="dynamo",
            component="backend",
        )

        assert not hasattr(engine_args, "ec_transfer_config")

    def test_enabled_capacity_configures_dynamo_connector(self):
        engine_args = SimpleNamespace()
        transfer_config = object()

        with patch("vllm.config.ECTransferConfig", return_value=transfer_config) as cls:
            cache_config_mod.configure_multimodal_embedding_cache(
                engine_args,
                route_to_encoder=False,
                capacity_gb=2.5,
                namespace="deployment",
                component="prefill",
            )

        assert engine_args.ec_transfer_config is transfer_config
        cls.assert_called_once_with(
            engine_id="deployment.prefill.backend.0",
            ec_role="ec_both",
            ec_connector="DynamoMultimodalEmbeddingCacheConnector",
            ec_connector_module_path=(
                "dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector"
            ),
            ec_connector_extra_config={
                "multimodal_embedding_cache_capacity_gb": 2.5,
            },
        )


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
        opaque_uuid = "catalog/image:v2"
        assert not conn.has_cache_item(opaque_uuid)

        request = self._make_request([(opaque_uuid, 100)])
        conn.update_state_after_alloc(request, 0)

        with patch.object(mod.logger, "debug") as log_debug:
            assert conn.has_cache_item(opaque_uuid)
        log_debug.assert_called_once_with(
            mod.EMBEDDING_CACHE_HIT_LOG,
            opaque_uuid,
        )

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
