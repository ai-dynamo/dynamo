#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for the routing_priority pool fallback chain in the global router.

`routing_priority` is an optional list of pool indices configured per strategy.
When set, it replaces grid-based selection and `priority_overrides`: the handler
attempts each pool in order, falling back on setup errors only.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from dynamo.global_router.handler import GlobalRouterHandler, _setup_with_fallback
    from dynamo.global_router.pool_selection import (
        AggPoolSelectionStrategy,
        DecodePoolSelectionStrategy,
        GlobalRouterConfig,
        PrefillPoolSelectionStrategy,
        PriorityPoolOverride,
        load_config,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]


# --- Strategy helpers ---


def _make_prefill_strategy(routing_priority=None) -> PrefillPoolSelectionStrategy:
    return PrefillPoolSelectionStrategy(
        ttft_min=10,
        ttft_max=3000,
        ttft_resolution=2,
        isl_min=0,
        isl_max=32000,
        isl_resolution=2,
        prefill_pool_mapping=[[0, 1], [0, 1]],
        priority_overrides=[
            PriorityPoolOverride(min_priority=1, max_priority=10, target_pool=1)
        ],
        routing_priority=routing_priority,
    )


def _make_decode_strategy(routing_priority=None) -> DecodePoolSelectionStrategy:
    return DecodePoolSelectionStrategy(
        itl_min=10,
        itl_max=500,
        itl_resolution=2,
        context_length_min=0,
        context_length_max=32000,
        context_length_resolution=2,
        decode_pool_mapping=[[0, 1], [0, 1]],
        priority_overrides=[
            PriorityPoolOverride(min_priority=1, max_priority=10, target_pool=1)
        ],
        routing_priority=routing_priority,
    )


def _make_agg_strategy(routing_priority=None) -> AggPoolSelectionStrategy:
    return AggPoolSelectionStrategy(
        ttft_min=10,
        ttft_max=3000,
        ttft_resolution=2,
        itl_min=5,
        itl_max=200,
        itl_resolution=2,
        agg_pool_mapping=[[0, 1], [1, 1]],
        priority_overrides=[
            PriorityPoolOverride(min_priority=1, max_priority=10, target_pool=0)
        ],
        routing_priority=routing_priority,
    )


# --- select_pool_chain tests ---


class TestPrefillSelectPoolChain:
    def test_unset_returns_single_grid_result(self):
        strategy = _make_prefill_strategy(routing_priority=None)
        # Grid selects 0 for ISL=100, TTFT=50
        assert strategy.select_pool_chain(isl=100, ttft_target=50) == [0]

    def test_unset_with_priority_override_uses_grid_path(self):
        strategy = _make_prefill_strategy(routing_priority=None)
        # priority=5 matches override -> pool 1
        assert strategy.select_pool_chain(isl=100, ttft_target=50, priority=5) == [1]

    def test_set_returns_chain_ignoring_grid(self):
        strategy = _make_prefill_strategy(routing_priority=[2, 0, 1])
        # Grid would pick 0; priority override would pick 1; chain wins.
        assert strategy.select_pool_chain(isl=100, ttft_target=50, priority=5) == [
            2,
            0,
            1,
        ]

    def test_chain_is_copied_not_aliased(self):
        rp = [1, 0]
        strategy = _make_prefill_strategy(routing_priority=rp)
        chain = strategy.select_pool_chain(isl=100, ttft_target=50)
        chain.append(99)
        assert strategy.routing_priority == [1, 0]


class TestDecodeSelectPoolChain:
    def test_unset_returns_single_grid_result(self):
        strategy = _make_decode_strategy(routing_priority=None)
        assert strategy.select_pool_chain(context_length=100, itl_target=20) == [0]

    def test_set_returns_chain(self):
        strategy = _make_decode_strategy(routing_priority=[1, 0])
        assert strategy.select_pool_chain(
            context_length=100, itl_target=20, priority=5
        ) == [1, 0]


class TestAggSelectPoolChain:
    def test_unset_returns_single_grid_result(self):
        strategy = _make_agg_strategy(routing_priority=None)
        # Grid mapping [[0,1],[1,1]] at ttft=50, itl=10 => pool 0
        assert strategy.select_pool_chain(ttft_target=50, itl_target=10) == [0]

    def test_set_returns_chain(self):
        strategy = _make_agg_strategy(routing_priority=[1, 0])
        assert strategy.select_pool_chain(
            ttft_target=50, itl_target=10, priority=5
        ) == [1, 0]


# --- Validation tests ---


def _disagg_config(**strategy_kwargs) -> GlobalRouterConfig:
    prefill = _make_prefill_strategy(
        routing_priority=strategy_kwargs.get("prefill_routing_priority")
    )
    decode = _make_decode_strategy(
        routing_priority=strategy_kwargs.get("decode_routing_priority")
    )
    return GlobalRouterConfig(
        mode="disagg",
        num_prefill_pools=2,
        num_decode_pools=2,
        prefill_pool_dynamo_namespaces=["a", "b"],
        decode_pool_dynamo_namespaces=["c", "d"],
        prefill_pool_selection_strategy=prefill,
        decode_pool_selection_strategy=decode,
    )


def _agg_config(routing_priority=None) -> GlobalRouterConfig:
    return GlobalRouterConfig(
        mode="agg",
        num_agg_pools=2,
        agg_pool_dynamo_namespaces=["a", "b"],
        agg_pool_selection_strategy=_make_agg_strategy(
            routing_priority=routing_priority
        ),
    )


class TestValidationDisagg:
    def test_valid_chain_passes(self):
        _disagg_config(prefill_routing_priority=[1, 0]).validate()
        _disagg_config(decode_routing_priority=[0, 1]).validate()

    def test_unset_passes(self):
        _disagg_config().validate()

    def test_empty_chain_rejected(self):
        with pytest.raises(
            ValueError, match="Prefill routing_priority must be non-empty"
        ):
            _disagg_config(prefill_routing_priority=[]).validate()

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="invalid pool index 5"):
            _disagg_config(prefill_routing_priority=[0, 5]).validate()

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="invalid pool index -1"):
            _disagg_config(decode_routing_priority=[-1, 0]).validate()

    def test_duplicates_rejected(self):
        with pytest.raises(ValueError, match="duplicate pool index 1"):
            _disagg_config(prefill_routing_priority=[1, 0, 1]).validate()


class TestValidationAgg:
    def test_valid_chain_passes(self):
        _agg_config(routing_priority=[1, 0]).validate()

    def test_unset_passes(self):
        _agg_config().validate()

    def test_empty_chain_rejected(self):
        with pytest.raises(ValueError, match="Agg routing_priority must be non-empty"):
            _agg_config(routing_priority=[]).validate()

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="invalid pool index 7"):
            _agg_config(routing_priority=[0, 7]).validate()

    def test_duplicates_rejected(self):
        with pytest.raises(ValueError, match="duplicate pool index 0"):
            _agg_config(routing_priority=[0, 0]).validate()


# --- JSON load round-trip ---


def _write_config(tmp_dir: Path, config_data: dict) -> Path:
    config_path = tmp_dir / "config.json"
    config_path.write_text(json.dumps(config_data))
    return config_path


def _disagg_json(**overrides) -> dict:
    config = {
        "num_prefill_pools": 2,
        "num_decode_pools": 2,
        "prefill_pool_dynamo_namespaces": ["ns-prefill-0", "ns-prefill-1"],
        "decode_pool_dynamo_namespaces": ["ns-decode-0", "ns-decode-1"],
        "prefill_pool_selection_strategy": {
            "isl_min": 0,
            "isl_max": 32000,
            "isl_resolution": 2,
            "ttft_min": 10,
            "ttft_max": 3000,
            "ttft_resolution": 2,
            "prefill_pool_mapping": [[0, 1], [0, 1]],
        },
        "decode_pool_selection_strategy": {
            "context_length_min": 0,
            "context_length_max": 32000,
            "context_length_resolution": 2,
            "itl_min": 10,
            "itl_max": 500,
            "itl_resolution": 2,
            "decode_pool_mapping": [[0, 1], [0, 1]],
        },
    }
    config.update(overrides)
    return config


class TestLoadConfig:
    def test_disagg_loads_routing_priority(self, tmp_path):
        config_data = _disagg_json()
        config_data["prefill_pool_selection_strategy"]["routing_priority"] = [1, 0]
        config_data["decode_pool_selection_strategy"]["routing_priority"] = [0, 1]
        config = load_config(_write_config(tmp_path, config_data))

        assert config.prefill_pool_selection_strategy.routing_priority == [1, 0]
        assert config.decode_pool_selection_strategy.routing_priority == [0, 1]

    def test_disagg_omitted_routing_priority_defaults_none(self, tmp_path):
        config = load_config(_write_config(tmp_path, _disagg_json()))
        assert config.prefill_pool_selection_strategy.routing_priority is None
        assert config.decode_pool_selection_strategy.routing_priority is None

    def test_agg_loads_routing_priority(self, tmp_path):
        config_data = {
            "mode": "agg",
            "num_agg_pools": 2,
            "agg_pool_dynamo_namespaces": ["ns-agg-0", "ns-agg-1"],
            "agg_pool_selection_strategy": {
                "ttft_min": 10,
                "ttft_max": 3000,
                "ttft_resolution": 2,
                "itl_min": 5,
                "itl_max": 200,
                "itl_resolution": 2,
                "agg_pool_mapping": [[0, 1], [1, 1]],
                "routing_priority": [1, 0],
            },
        }
        config = load_config(_write_config(tmp_path, config_data))
        assert config.agg_pool_selection_strategy.routing_priority == [1, 0]


# --- Handler-level fallback ---


def _make_async_stream(items):
    """Return an async iterator that yields the given items."""

    async def _gen():
        for item in items:
            yield item

    return _gen()


def _make_failing_stream(exc):
    """Return an async iterator that raises `exc` on first iteration."""

    async def _gen():
        raise exc
        yield  # pragma: no cover — unreachable, makes this a generator

    return _gen()


@pytest.mark.asyncio
class TestSetupWithFallback:
    async def test_first_pool_succeeds(self):
        client_a = MagicMock()
        client_a.generate = AsyncMock(return_value="stream-a")
        client_b = MagicMock()
        client_b.generate = AsyncMock(return_value="stream-b")

        stream, idx = await _setup_with_fallback(
            pool_chain=[0, 1],
            namespaces=["ns-a", "ns-b"],
            clients={"ns-a": client_a, "ns-b": client_b},
            request={"x": 1},
            pool_kind="agg",
        )
        assert stream == "stream-a"
        assert idx == 0
        client_a.generate.assert_awaited_once()
        client_b.generate.assert_not_called()

    async def test_first_pool_fails_falls_back(self):
        client_a = MagicMock()
        client_a.generate = AsyncMock(side_effect=ConnectionError("nope"))
        client_b = MagicMock()
        client_b.generate = AsyncMock(return_value="stream-b")

        stream, idx = await _setup_with_fallback(
            pool_chain=[0, 1],
            namespaces=["ns-a", "ns-b"],
            clients={"ns-a": client_a, "ns-b": client_b},
            request={"x": 1},
            pool_kind="prefill",
        )
        assert stream == "stream-b"
        assert idx == 1

    async def test_all_pools_fail_raises_last(self):
        client_a = MagicMock()
        client_a.generate = AsyncMock(side_effect=ConnectionError("first"))
        client_b = MagicMock()
        client_b.generate = AsyncMock(side_effect=RuntimeError("last"))

        with pytest.raises(RuntimeError, match="last"):
            await _setup_with_fallback(
                pool_chain=[0, 1],
                namespaces=["ns-a", "ns-b"],
                clients={"ns-a": client_a, "ns-b": client_b},
                request={"x": 1},
                pool_kind="decode",
            )


def _build_disagg_handler(prefill_clients, decode_clients, prefill_chain=None):
    """Build a GlobalRouterHandler in disagg mode without invoking initialize().

    Returns a handler whose prefill_clients / decode_clients are the supplied
    mocks, so we can exercise dispatch logic without a live runtime.
    """
    config = GlobalRouterConfig(
        mode="disagg",
        num_prefill_pools=2,
        num_decode_pools=2,
        prefill_pool_dynamo_namespaces=["ns-prefill-0", "ns-prefill-1"],
        decode_pool_dynamo_namespaces=["ns-decode-0", "ns-decode-1"],
        prefill_pool_selection_strategy=_make_prefill_strategy(
            routing_priority=prefill_chain
        ),
        decode_pool_selection_strategy=_make_decode_strategy(),
    )
    config.validate()
    handler = GlobalRouterHandler.__new__(GlobalRouterHandler)
    handler.runtime = None
    handler.config = config
    handler.model_name = "test-model"
    handler.default_ttft_target = None
    handler.default_itl_target = None
    handler.prefill_clients = prefill_clients
    handler.decode_clients = decode_clients
    handler.agg_clients = {}
    handler.prefill_namespace_to_idx = {
        ns: i for i, ns in enumerate(config.prefill_pool_dynamo_namespaces)
    }
    handler.decode_namespace_to_idx = {
        ns: i for i, ns in enumerate(config.decode_pool_dynamo_namespaces)
    }
    return handler


@pytest.mark.asyncio
class TestHandlerFallback:
    async def test_prefill_fallback_on_setup_error(self):
        # Pool 0 fails setup; pool 1 succeeds. With routing_priority=[0, 1],
        # handler should fall through and stream pool 1's output.
        bad = MagicMock()
        bad.generate = AsyncMock(side_effect=ConnectionError("pool 0 down"))
        good = MagicMock()
        good.generate = AsyncMock(return_value=_make_async_stream([{"text": "hi"}]))

        handler = _build_disagg_handler(
            prefill_clients={"ns-prefill-0": bad, "ns-prefill-1": good},
            decode_clients={},
            prefill_chain=[0, 1],
        )
        outputs = [
            chunk async for chunk in handler.handle_prefill({"token_ids": [1, 2, 3]})
        ]
        assert outputs == [{"text": "hi"}]
        bad.generate.assert_awaited_once()
        good.generate.assert_awaited_once()

    async def test_prefill_no_fallback_on_mid_stream_error(self):
        # Pool 0 setup succeeds, but the stream raises on first iteration.
        # Handler must propagate the error rather than fall back to pool 1.
        bad = MagicMock()
        bad.generate = AsyncMock(
            return_value=_make_failing_stream(RuntimeError("mid-stream"))
        )
        good = MagicMock()
        good.generate = AsyncMock(return_value=_make_async_stream([{"text": "hi"}]))

        handler = _build_disagg_handler(
            prefill_clients={"ns-prefill-0": bad, "ns-prefill-1": good},
            decode_clients={},
            prefill_chain=[0, 1],
        )
        with pytest.raises(RuntimeError, match="mid-stream"):
            async for _ in handler.handle_prefill({"token_ids": [1, 2, 3]}):
                pass
        bad.generate.assert_awaited_once()
        good.generate.assert_not_called()

    async def test_prefill_all_pools_fail_raises(self):
        bad0 = MagicMock()
        bad0.generate = AsyncMock(side_effect=ConnectionError("first"))
        bad1 = MagicMock()
        bad1.generate = AsyncMock(side_effect=RuntimeError("last"))

        handler = _build_disagg_handler(
            prefill_clients={"ns-prefill-0": bad0, "ns-prefill-1": bad1},
            decode_clients={},
            prefill_chain=[0, 1],
        )
        with pytest.raises(RuntimeError, match="last"):
            async for _ in handler.handle_prefill({"token_ids": [1, 2, 3]}):
                pass
