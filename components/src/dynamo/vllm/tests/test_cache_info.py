# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM KV event block-size helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm.cache_info import (
    DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
    configure_kv_event_block_size,
    get_configured_kv_event_block_size,
    select_main_attention_block_size,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _make_vllm_config(block_size: int = 16, additional_config=None):
    """Build a minimal VllmConfig-like namespace for testing."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        additional_config=additional_config,
    )


def _make_engine(side_effect=None, return_value=None):
    """Build a minimal AsyncLLM-like namespace with a mocked engine_core."""
    engine_core = Mock()
    engine_core.call_utility_async = AsyncMock(
        side_effect=side_effect,
        return_value=return_value,
    )
    return SimpleNamespace(engine_core=engine_core)


class TestSelectMainAttentionBlockSize:
    """Tests for select_main_attention_block_size."""

    def test_returns_fallback_when_metadata_empty(self):
        """Empty metadata returns the fallback block size."""
        assert select_main_attention_block_size([], 16) == 16

    def test_selects_main_attention_cache_group(self):
        """Block size is taken from the first main-attention group, ignoring others."""
        group_metadata = [
            {"kind": "mamba_state", "block_size": 512},
            {"kind": "full_attention", "block_size": 2096},
        ]

        assert select_main_attention_block_size(group_metadata, 16) == 2096


class TestGetConfiguredKvEventBlockSize:
    """Tests for get_configured_kv_event_block_size."""

    def test_reads_env_override(self, monkeypatch):
        """DYN_VLLM_KV_EVENT_BLOCK_SIZE env var takes highest precedence."""
        monkeypatch.setenv("DYN_VLLM_KV_EVENT_BLOCK_SIZE", "4096")

        assert get_configured_kv_event_block_size(_make_vllm_config()) == 4096

    def test_falls_back_to_additional_config_then_cache_config(self):
        """additional_config key takes precedence over cache_config.block_size."""
        assert (
            get_configured_kv_event_block_size(
                _make_vllm_config(
                    additional_config={DYNAMO_KV_EVENT_BLOCK_SIZE_KEY: 1024}
                )
            )
            == 1024
        )
        assert get_configured_kv_event_block_size(_make_vllm_config()) == 16


class TestConfigureKvEventBlockSize:
    """Tests for configure_kv_event_block_size."""

    @pytest.mark.asyncio
    async def test_preserves_existing_additional_config_override(self):
        """Existing additional_config value is not overwritten by engine metadata."""
        vllm_config = _make_vllm_config(
            additional_config={DYNAMO_KV_EVENT_BLOCK_SIZE_KEY: 2048}
        )
        engine = _make_engine(
            return_value=[{"kind": "full_attention", "block_size": 32}]
        )

        result = await configure_kv_event_block_size(engine, vllm_config)

        assert result == 2048
        engine.engine_core.call_utility_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_env_override(self, monkeypatch):
        """DYN_VLLM_KV_EVENT_BLOCK_SIZE skips engine metadata fetch entirely."""
        monkeypatch.setenv("DYN_VLLM_KV_EVENT_BLOCK_SIZE", "4096")
        vllm_config = _make_vllm_config()
        engine = _make_engine(
            return_value=[{"kind": "full_attention", "block_size": 32}]
        )

        result = await configure_kv_event_block_size(engine, vllm_config)

        assert result == 4096
        assert vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == 4096
        engine.engine_core.call_utility_async.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("require_exact_match", [False, True])
    async def test_warns_and_falls_back_when_metadata_api_unavailable(
        self, caplog, require_exact_match
    ):
        """A failed metadata API call always warns and falls back to block_size.

        This covers older vLLM builds that don't expose get_kv_cache_group_metadata.
        require_exact_match does NOT raise in this case because the API being absent
        means block_size is the correct fallback for that build.
        """
        vllm_config = _make_vllm_config()
        engine = _make_engine(side_effect=AttributeError("missing method"))

        result = await configure_kv_event_block_size(
            engine, vllm_config, require_exact_match=require_exact_match
        )

        assert result == 16
        assert "falling back to vLLM cache_config.block_size" in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "group_metadata", [[], [{"kind": "mamba_state", "block_size": 32}]]
    )
    async def test_raises_when_exact_match_required_and_main_attention_metadata_missing(
        self,
        group_metadata,
    ):
        """require_exact_match=True raises RuntimeError when metadata is returned but has no main-attention group."""
        vllm_config = _make_vllm_config()
        engine = _make_engine(return_value=group_metadata)

        with pytest.raises(RuntimeError, match="DYN_VLLM_KV_EVENT_BLOCK_SIZE"):
            await configure_kv_event_block_size(
                engine,
                vllm_config,
                require_exact_match=True,
            )
