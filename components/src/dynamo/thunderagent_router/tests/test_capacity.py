# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the WorkerCapacityProvider MDC parser. No runtime needed."""

from __future__ import annotations

import json
from typing import Optional

import pytest

from dynamo.thunderagent_router.capacity import WorkerCapacity, WorkerCapacityProvider

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _FakeSubscriber:
    def __init__(self, cards: dict[str, str]) -> None:
        self._cards = cards
        self.get_calls = 0

    def get_model_cards(self) -> dict[str, str]:
        self.get_calls += 1
        return self._cards


def _make_provider(
    cards: dict[str, str],
) -> tuple[WorkerCapacityProvider, _FakeSubscriber]:
    provider = WorkerCapacityProvider(endpoint=None)  # type: ignore[arg-type]
    subscriber = _FakeSubscriber(cards)
    provider._subscriber = subscriber  # type: ignore[assignment]
    return provider, subscriber


def _card(
    block_size: Optional[int],
    total_blocks: Optional[int],
    host_total_tokens: Optional[int] = None,
    capacity_total_tokens: object = None,
) -> str:
    body: dict = {}
    if block_size is not None:
        body["kv_cache_block_size"] = block_size
    if total_blocks is not None:
        body["runtime_config"] = {"total_kv_blocks": total_blocks}
    if host_total_tokens is not None:
        body.setdefault("runtime_config", {}).setdefault("runtime_data", {})[
            "native_offloading_capacity"
        ] = {"total_tokens": host_total_tokens}
    if capacity_total_tokens is not None:
        body.setdefault("runtime_config", {}).setdefault("runtime_data", {})[
            "kv_cache_capacity"
        ] = {"total_tokens": capacity_total_tokens}
    return json.dumps(body)


def test_snapshot_falls_back_to_physical_kv_pool_tokens():
    provider, _ = _make_provider({"1": _card(16, 1000), "2": _card(8, 2000)})
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=16_000, block_size=16),
        2: WorkerCapacity(retention_tokens=16_000, block_size=8),
    }


def test_snapshot_prefers_authoritative_token_capacity():
    provider, _ = _make_provider(
        {"1": _card(4160, 1000, capacity_total_tokens=1_234_567)}
    )
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=1_234_567, block_size=4160)
    }


def test_snapshot_adds_native_offloading_tokens_to_retention_budget():
    provider, _ = _make_provider(
        {
            "1": _card(
                16,
                1_000,
                host_total_tokens=300,
                capacity_total_tokens=15_000,
            )
        }
    )
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=15_300, block_size=16)
    }


def test_snapshot_ignores_invalid_native_offloading_capacity():
    card = json.loads(_card(16, 1_000))
    card["runtime_config"]["runtime_data"] = {
        "native_offloading_capacity": {"total_tokens": "300"}
    }
    provider, _ = _make_provider({"1": json.dumps(card)})
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=16_000, block_size=16)
    }


@pytest.mark.parametrize(
    "invalid_capacity", ["16000", 0, -1, True, float("inf"), float("nan")]
)
def test_snapshot_falls_back_when_explicit_capacity_is_invalid(invalid_capacity):
    provider, _ = _make_provider(
        {"1": _card(16, 1_000, capacity_total_tokens=invalid_capacity)}
    )
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=16_000, block_size=16)
    }


def test_snapshot_skips_malformed_cards():
    provider, _ = _make_provider(
        {
            "1": _card(16, 1000),
            "2": "{not json",
            "3": _card(None, 1000),
            "4": _card(16, None),
            "5": _card(0, 1000),
            "6": _card(16, "abc"),  # type: ignore[arg-type]
        }
    )
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=16_000, block_size=16)
    }


def test_snapshot_skips_unparseable_worker_ids():
    provider, _ = _make_provider({"not-an-int": _card(16, 1000)})
    assert provider.snapshot() == {}


def test_parsed_cards_cache_hits_on_repeat_snapshot():
    """The MDC body never changes per worker; a repeat snapshot should
    hit the cache instead of re-parsing JSON."""
    cards = {"1": _card(16, 1000)}
    provider, _ = _make_provider(cards)
    provider.snapshot()
    assert cards["1"] in provider._parsed
    # Second snapshot returns same result without touching json.loads;
    # we sentinel-check by mutating the cached value.
    provider._parsed[cards["1"]] = WorkerCapacity(
        retention_tokens=999_999, block_size=32
    )
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=999_999, block_size=32)
    }


def test_snapshot_returns_empty_when_subscriber_unset():
    provider = WorkerCapacityProvider(endpoint=None)  # type: ignore[arg-type]
    assert provider.snapshot() == {}
