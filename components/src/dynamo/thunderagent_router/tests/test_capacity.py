# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the WorkerCapacityProvider MDC parser. No runtime needed."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Optional

import pytest

from dynamo.thunderagent_router.capacity import WorkerCapacityProvider

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _FakeSubscriber:
    def __init__(self, cards: dict[str, str]) -> None:
        self._cards = cards
        self.get_calls = 0

    def get_model_cards(self) -> dict[str, str]:
        self.get_calls += 1
        return self._cards


class _FakeClient:
    def __init__(self, worker_ids: list[int] | None = None) -> None:
        self._worker_ids = worker_ids or []

    def instance_ids(self) -> list[int]:
        return list(self._worker_ids)


def _make_provider(
    cards: Optional[dict[str, str]],
) -> tuple[WorkerCapacityProvider, Optional[_FakeSubscriber]]:
    """Build a provider over *cards*; ``None`` leaves the subscriber unset."""
    provider = WorkerCapacityProvider(  # type: ignore[arg-type]
        endpoint=None,
        client=_FakeClient(),
    )
    if cards is None:
        return provider, None
    subscriber = _FakeSubscriber(cards)
    provider._subscriber = subscriber  # type: ignore[assignment]
    return provider, subscriber


def _card(
    block_size: Optional[int],
    total_blocks: Optional[int],
    host_total_tokens: Optional[int] = None,
    dp_size: Optional[int] = None,
    start_rank: Optional[int] = None,
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
    if dp_size is not None:
        body.setdefault("runtime_config", {})["data_parallel_size"] = dp_size
    if start_rank is not None:
        body.setdefault("runtime_config", {})["data_parallel_start_rank"] = start_rank
    return json.dumps(body)


def test_snapshot_extracts_kv_pool_tokens():
    provider, _ = _make_provider({"1": _card(16, 1000), "2": _card(8, 2000)})
    assert provider.snapshot() == {(1, 0): 16_000, (2, 0): 16_000}


@pytest.mark.parametrize(
    "dp_size, start_rank, expected_ranks",
    [
        (4, None, [0, 1, 2, 3]),
        (4, 2, [2, 3, 4, 5]),
        (1, None, [0]),
    ],
    ids=["implicit_start", "declared_start", "single_rank"],
)
def test_snapshot_fans_out_one_entry_per_dp_rank(dp_size, start_rank, expected_ranks):
    """total_kv_blocks is per rank, so a D-rank worker yields D entries of it."""
    provider, _ = _make_provider(
        {"1": _card(16, 1000, dp_size=dp_size, start_rank=start_rank)}
    )
    assert provider.snapshot() == {(1, rank): 16_000 for rank in expected_ranks}


def test_snapshot_adds_native_offloading_tokens_to_retention_budget():
    provider, _ = _make_provider({"1": _card(16, 1_000, host_total_tokens=300)})
    assert provider.snapshot() == {(1, 0): 16_300}


def test_snapshot_ignores_invalid_native_offloading_capacity():
    card = json.loads(_card(16, 1_000))
    card["runtime_config"]["runtime_data"] = {
        "native_offloading_capacity": {"total_tokens": "300"}
    }
    provider, _ = _make_provider({"1": json.dumps(card)})
    assert provider.snapshot() == {(1, 0): 16_000}


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
    assert provider.snapshot() == {(1, 0): 16_000}


def test_snapshot_skips_unparseable_worker_ids():
    provider, _ = _make_provider({"not-an-int": _card(16, 1000)})
    assert provider.snapshot() == {}


def test_one_cache_entry_serves_every_field_of_a_card():
    """Parsed once for all fields. Sentinel-checked by mutating the cached record: a
    re-read of any field -- pool tokens, rank count, start rank -- would lose it."""
    cards = {"1": _card(16, 1000, dp_size=4)}
    provider, _ = _make_provider(cards)
    provider.snapshot()

    cached = provider._parsed[cards["1"]]
    provider._parsed[cards["1"]] = replace(
        cached, pool_tokens=999_999, start_rank=7, dp_size=2
    )

    assert provider.snapshot() == {(1, 7): 999_999, (1, 8): 999_999}


def test_snapshot_returns_empty_when_subscriber_unset():
    provider = WorkerCapacityProvider(  # type: ignore[arg-type]
        endpoint=None,
        client=_FakeClient(),
    )
    assert provider.snapshot() == {}


def test_live_worker_ids_uses_endpoint_client():
    provider = WorkerCapacityProvider(  # type: ignore[arg-type]
        endpoint=None,
        client=_FakeClient([1, 2]),
    )
    assert provider.live_worker_ids() == {1, 2}
