# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for inheriting backend model-card runtime policy into the proxy card."""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from dynamo.common.token_budget import TOKEN_BUDGET_RUNTIME_KEY
from dynamo.thunderagent_router.proxy_card import (
    apply_runtime_config_mapping,
    kv_cache_block_size_from_card_json,
    runtime_config_from_card_json,
    select_backend_card,
    wait_for_backend_card,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class FakeRuntimeConfig:
    """Records inherited fields without requiring the PyO3 ModelRuntimeConfig."""

    def __init__(self) -> None:
        self.engine_specific: dict[str, Any] = {}
        self.structural_tag_mode: Optional[str] = None
        self.bootstrap_host: Optional[str] = None
        self.bootstrap_port: Optional[int] = None

    def set_engine_specific(self, key: str, value: str) -> None:
        self.engine_specific[key] = json.loads(value)

    def get_engine_specific(self, key: str) -> Optional[str]:
        if key not in self.engine_specific:
            return None
        return json.dumps(self.engine_specific[key])

    def set_structural_tag_mode(self, mode: str) -> None:
        self.structural_tag_mode = mode

    def set_disaggregated_endpoint(
        self,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
    ) -> None:
        self.bootstrap_host = bootstrap_host
        self.bootstrap_port = bootstrap_port


def _token_budget(
    combined_limit: int = 131_072,
    *,
    reject_prompt_overflow: bool = True,
    reject_total_overflow: bool = True,
) -> dict[str, Any]:
    return {
        "combined_limit": combined_limit,
        "reject_prompt_overflow": reject_prompt_overflow,
        "reject_total_overflow": reject_total_overflow,
    }


def _card(
    *,
    token_budget: Optional[dict[str, Any]] = None,
    context_length: Optional[int] = None,
    extra_runtime: Optional[dict[str, Any]] = None,
    kv_cache_block_size: Optional[int] = 64,
    runtime_data: Optional[dict[str, Any]] = None,
) -> str:
    runtime_config: dict[str, Any] = dict(extra_runtime or {})
    if context_length is not None:
        runtime_config["context_length"] = context_length
    data = dict(runtime_config.get("runtime_data") or {})
    if runtime_data:
        data.update(runtime_data)
    if token_budget is not None:
        data[TOKEN_BUDGET_RUNTIME_KEY] = token_budget
    if data:
        runtime_config["runtime_data"] = data
    body: dict[str, Any] = {"runtime_config": runtime_config}
    if kv_cache_block_size is not None:
        body["kv_cache_block_size"] = kv_cache_block_size
    return json.dumps(body)


def test_select_backend_card_prefers_token_budget():
    without_budget = _card(context_length=262_144)
    with_budget = _card(token_budget=_token_budget(), context_length=131_072)
    selected = select_backend_card({"1": without_budget, "2": with_budget})
    assert selected == with_budget


def test_select_backend_card_skips_malformed_and_returns_first_valid():
    first = _card(context_length=4096)
    selected = select_backend_card(
        {
            "bad": "{not json",
            "also-bad": json.dumps(["not", "an", "object"]),
            "ok": first,
            "later": _card(context_length=8192),
        }
    )
    assert selected == first


def test_select_backend_card_returns_none_when_empty():
    assert select_backend_card({}) is None


def test_inherit_copies_token_budget_and_typed_runtime_fields():
    card = _card(
        token_budget=_token_budget(131_072),
        context_length=131_072,
        extra_runtime={
            "tool_call_parser": "hermes",
            "reasoning_parser": "qwen",
            "total_kv_blocks": 1000,
            "taints": ["user.taint/example"],
            "structural_tag_mode": "on",
            "disaggregated_endpoint": {
                "bootstrap_host": "10.0.0.1",
                "bootstrap_port": 8998,
            },
        },
        runtime_data={"sglang_generate": True},
    )
    cfg = runtime_config_from_card_json(card, FakeRuntimeConfig)

    budget = json.loads(cfg.get_engine_specific(TOKEN_BUDGET_RUNTIME_KEY))
    assert budget == _token_budget(131_072)
    assert cfg.engine_specific["sglang_generate"] is True
    assert cfg.context_length == 131_072
    assert cfg.tool_call_parser == "hermes"
    assert cfg.reasoning_parser == "qwen"
    assert cfg.total_kv_blocks == 1000
    assert cfg.taints == {"user.taint/example"}
    assert cfg.structural_tag_mode == "on"
    assert cfg.bootstrap_host == "10.0.0.1"
    assert cfg.bootstrap_port == 8998


def test_parser_overlay_does_not_drop_inherited_token_budget():
    cfg = runtime_config_from_card_json(
        _card(
            token_budget=_token_budget(131_072),
            extra_runtime={"tool_call_parser": "hermes"},
        ),
        FakeRuntimeConfig,
    )
    cfg.tool_call_parser = "glm47"
    budget = json.loads(cfg.get_engine_specific(TOKEN_BUDGET_RUNTIME_KEY))
    assert budget["combined_limit"] == 131_072
    assert cfg.tool_call_parser == "glm47"


def test_inherit_skips_unknown_fields_that_reject_setattr():
    class StrictRuntimeConfig(FakeRuntimeConfig):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "not_a_real_field":
                raise AttributeError(name)
            super().__setattr__(name, value)

    cfg = StrictRuntimeConfig()
    apply_runtime_config_mapping(
        cfg,
        {
            "context_length": 2048,
            "not_a_real_field": "drop-me",
            "runtime_data": {TOKEN_BUDGET_RUNTIME_KEY: _token_budget(2048)},
        },
    )
    assert cfg.context_length == 2048
    assert TOKEN_BUDGET_RUNTIME_KEY in cfg.engine_specific


def test_kv_cache_block_size_from_card_json():
    assert kv_cache_block_size_from_card_json(_card(kv_cache_block_size=64)) == 64
    assert kv_cache_block_size_from_card_json(_card(kv_cache_block_size=0)) is None
    assert kv_cache_block_size_from_card_json(_card(kv_cache_block_size=None)) is None
    assert kv_cache_block_size_from_card_json("{not json") is None


@pytest.mark.asyncio
async def test_wait_for_backend_card_returns_when_cards_appear():
    card = _card(token_budget=_token_budget())
    calls = {"n": 0}

    def get_cards() -> dict[str, str]:
        calls["n"] += 1
        if calls["n"] < 3:
            return {}
        return {"7": card}

    got = await wait_for_backend_card(
        get_cards, timeout_seconds=1.0, poll_interval_seconds=0.01
    )
    assert got == card
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_wait_for_backend_card_times_out():
    got = await wait_for_backend_card(
        lambda: {}, timeout_seconds=0.05, poll_interval_seconds=0.01
    )
    assert got is None
