#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.cache_info."""

import pytest

from dynamo.vllm.cache_info import (
    DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
    configure_kv_event_block_size,
    detect_mamba_hybrid_model,
    get_configured_kv_event_block_size,
    select_main_attention_block_size,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class TestDetectMambaHybridModel:
    """Test detect_mamba_hybrid_model helper."""

    def test_pure_attention_model_returns_false(self):
        """Non-Mamba, non-speculative model is not detected as hybrid."""
        vllm_config = pytest.importorskip("vllm.config").VllmConfig()
        vllm_config.speculative_config = None
        vllm_config.model_config.hf_config.architectures = [
            "QWenLMHeadModel",
            "Qwen2ForCausalLM",
        ]
        assert detect_mamba_hybrid_model(vllm_config) is False

    def test_mamba_architecture_returns_true(self):
        """Mamba architecture strings are correctly identified."""
        vllm_config = pytest.importorskip("vllm.config").VllmConfig()
        vllm_config.speculative_config = None
        for arch in [
            "MambaLMHeadModel",
            "MambaDecodeModel",
            "MambaForCausalLM",
            "FalconMambaForCausalLM",
            "Mamba2ForCausalLM",
            "JambaForCausalLM",
        ]:
            vllm_config.model_config.hf_config.architectures = [arch]
            assert (
                detect_mamba_hybrid_model(vllm_config) is True
            ), f"{arch} not detected"

    def test_speculative_config_returns_true(self):
        """Non-None speculative_config marks the model as hybrid."""
        vllm_config = pytest.importorskip("vllm.config").VllmConfig()
        # Set a simple object as speculative config (None-check only)
        vllm_config.speculative_config = object()
        vllm_config.model_config.hf_config.architectures = ["Qwen2ForCausalLM"]
        assert detect_mamba_hybrid_model(vllm_config) is True

    def test_missing_hf_config_returns_false(self):
        """Model without hf_config is not flagged as Mamba/hybrid."""
        vllm_config = pytest.importorskip("vllm.config").VllmConfig()
        vllm_config.speculative_config = None
        del vllm_config.model_config.hf_config
        assert detect_mamba_hybrid_model(vllm_config) is False

    def test_empty_architectures_returns_false(self):
        """Empty architectures list is not flagged as Mamba/hybrid."""
        vllm_config = pytest.importorskip("vllm.config").VllmConfig()
        vllm_config.speculative_config = None
        vllm_config.model_config.hf_config.architectures = []
        assert detect_mamba_hybrid_model(vllm_config) is False


class TestSelectMainAttentionBlockSize:
    """Test select_main_attention_block_size."""

    def test_empty_metadata_returns_fallback(self):
        """Empty group metadata uses the fallback."""
        result = select_main_attention_block_size([], fallback_block_size=16)
        assert result == 16

    def test_full_attention_block_size_returned(self):
        """The block_size from full_attention group is returned."""
        group_metadata = [
            {"kind": "other", "block_size": 99},
            {"kind": "full_attention", "block_size": 42},
        ]
        result = select_main_attention_block_size(
            group_metadata, fallback_block_size=16
        )
        assert result == 42

    def test_mla_attention_block_size_returned(self):
        """The block_size from mla_attention group is returned."""
        group_metadata = [
            {"kind": "mla_attention", "block_size": 128},
        ]
        result = select_main_attention_block_size(
            group_metadata, fallback_block_size=16
        )
        assert result == 128

    def test_only_sink_full_attention_returns_fallback(self):
        """When no main-attention kind is present, fallback is used."""
        group_metadata = [
            {"kind": "sink_full_attention", "block_size": 999},
        ]
        result = select_main_attention_block_size(
            group_metadata, fallback_block_size=16
        )
        assert result == 16

    def test_missing_block_size_uses_fallback(self):
        """Group with no block_size key falls back."""
        group_metadata = [
            {"kind": "full_attention"},
        ]
        result = select_main_attention_block_size(
            group_metadata, fallback_block_size=16
        )
        assert result == 16


class TestConfigureKvEventBlockSizeSuccess:
    """Test configure_kv_event_block_size when the utility call succeeds."""

    @pytest.mark.asyncio
    async def test_success_stores_block_size_in_additional_config(self):
        """On success the block size is stored in vllm_config.additional_config."""
        VllmConfig = pytest.importorskip("vllm.config").VllmConfig
        vllm_config = VllmConfig()
        vllm_config.additional_config = {}
        vllm_config.cache_config.block_size = 16

        group_metadata = [
            {"kind": "full_attention", "block_size": 2096},
        ]

        from unittest.mock import AsyncMock, MagicMock

        mock_engine = MagicMock()
        mock_engine.engine_core = MagicMock()
        mock_engine.engine_core.call_utility_async = AsyncMock(
            return_value=group_metadata
        )

        result = await configure_kv_event_block_size(mock_engine, vllm_config)

        assert result == 2096
        assert vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == 2096


class TestConfigureKvEventBlockSizeFallback:
    """Test configure_kv_event_block_size when the utility call fails."""

    @pytest.mark.asyncio
    async def test_pure_attention_model_falls_back_silently(self):
        """Non-Mamba/non-hybrid models fall back to cache_config.block_size."""
        VllmConfig = pytest.importorskip("vllm.config").VllmConfig
        vllm_config = VllmConfig()
        vllm_config.additional_config = {}
        vllm_config.cache_config.block_size = 16
        vllm_config.speculative_config = None
        vllm_config.model_config.hf_config.architectures = ["Qwen2ForCausalLM"]

        from unittest.mock import AsyncMock, MagicMock

        mock_engine = MagicMock()
        mock_engine.engine_core = MagicMock()
        mock_engine.engine_core.call_utility_async = AsyncMock(
            side_effect=RuntimeError("utility unavailable")
        )

        result = await configure_kv_event_block_size(mock_engine, vllm_config)

        assert result == 16  # fallback block_size
        assert vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == 16

    @pytest.mark.asyncio
    async def test_mamba_model_raises_value_error(self):
        """Mamba models raise ValueError when the utility call fails."""
        VllmConfig = pytest.importorskip("vllm.config").VllmConfig
        vllm_config = VllmConfig()
        vllm_config.additional_config = {}
        vllm_config.cache_config.block_size = 16
        vllm_config.speculative_config = None
        vllm_config.model_config.hf_config.architectures = ["MambaLMHeadModel"]

        from unittest.mock import AsyncMock, MagicMock

        mock_engine = MagicMock()
        mock_engine.engine_core = MagicMock()
        mock_engine.engine_core.call_utility_async = AsyncMock(
            side_effect=RuntimeError("utility unavailable")
        )

        with pytest.raises(
            ValueError, match="Mamba|hybrid|get_kv_cache_group_metadata"
        ):
            await configure_kv_event_block_size(mock_engine, vllm_config)

    @pytest.mark.asyncio
    async def test_speculative_hybrid_model_raises_value_error(self):
        """Speculative/hybrid models raise ValueError when the utility fails."""
        VllmConfig = pytest.importorskip("vllm.config").VllmConfig
        vllm_config = VllmConfig()
        vllm_config.additional_config = {}
        vllm_config.cache_config.block_size = 16
        vllm_config.speculative_config = object()  # non-None = hybrid
        vllm_config.model_config.hf_config.architectures = ["Qwen2ForCausalLM"]

        from unittest.mock import AsyncMock, MagicMock

        mock_engine = MagicMock()
        mock_engine.engine_core = MagicMock()
        mock_engine.engine_core.call_utility_async = AsyncMock(
            side_effect=RuntimeError("utility unavailable")
        )

        with pytest.raises(ValueError, match="hybrid|get_kv_cache_group_metadata"):
            await configure_kv_event_block_size(mock_engine, vllm_config)

    @pytest.mark.asyncio
    async def test_speculative_model_without_hf_config_raises_value_error(self):
        """Speculative models without hf_config raise ValueError, not AttributeError."""
        VllmConfig = pytest.importorskip("vllm.config").VllmConfig
        vllm_config = VllmConfig()
        vllm_config.additional_config = {}
        vllm_config.cache_config.block_size = 16
        vllm_config.speculative_config = object()  # non-None = hybrid
        # Simulate missing hf_config (speculative configs may not have it)
        del vllm_config.model_config.hf_config

        from unittest.mock import AsyncMock, MagicMock

        mock_engine = MagicMock()
        mock_engine.engine_core = MagicMock()
        mock_engine.engine_core.call_utility_async = AsyncMock(
            side_effect=RuntimeError("utility unavailable")
        )

        # Must raise ValueError, not AttributeError
        with pytest.raises(ValueError, match="hybrid|get_kv_cache_group_metadata"):
            await configure_kv_event_block_size(mock_engine, vllm_config)


class TestGetConfiguredKvEventBlockSize:
    """Test get_configured_kv_event_block_size."""

    def test_returns_cached_value_when_present(self):
        """If DYNAMO_KV_EVENT_BLOCK_SIZE_KEY is set, return it."""
        VllmConfig = pytest.importorskip("vllm.config").VllmConfig
        vllm_config = VllmConfig()
        vllm_config.additional_config = {DYNAMO_KV_EVENT_BLOCK_SIZE_KEY: 2096}
        vllm_config.cache_config.block_size = 16

        result = get_configured_kv_event_block_size(vllm_config)
        assert result == 2096

    def test_returns_cache_block_size_when_not_cached(self):
        """If the key is absent, fall back to cache_config.block_size."""
        VllmConfig = pytest.importorskip("vllm.config").VllmConfig
        vllm_config = VllmConfig()
        vllm_config.additional_config = {}
        vllm_config.cache_config.block_size = 16

        result = get_configured_kv_event_block_size(vllm_config)
        assert result == 16
