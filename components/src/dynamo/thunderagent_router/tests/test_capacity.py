# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the WorkerCapacityProvider MDC parser. No runtime needed."""

from __future__ import annotations

import json

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
    block_size: object,
    total_blocks: object,
    host_total_tokens: object = None,
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
    return json.dumps(body)


def test_snapshot_falls_back_to_physical_kv_pool_tokens():
    provider, _ = _make_provider({"1": _card(16, 1000), "2": _card(8, 2000)})
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=16_000, block_size=16),
        2: WorkerCapacity(retention_tokens=16_000, block_size=8),
    }


def test_snapshot_adds_native_offloading_tokens_to_physical_pool():
    provider, _ = _make_provider({"1": _card(16, 1_000, host_total_tokens=300)})
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=16_300, block_size=16)
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


def test_snapshot_skips_malformed_cards():
    provider, _ = _make_provider(
        {
            "1": _card(16, 1000),
            "2": "{not json",
            "3": _card(None, 1000),
            "4": _card(16, None),
            "5": _card(0, 1000),
            "6": _card(16, "1000"),
            "7": _card(1.5, 1000),
            "8": _card(16, 1.5),
            "9": _card(True, 1000),
            "10": _card(16, True),
        }
    )
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=16_000, block_size=16)
    }


def test_snapshot_skips_unparseable_worker_ids():
    provider, _ = _make_provider({"not-an-int": _card(16, 1000)})
    assert provider.snapshot() == {}


def test_snapshot_isolates_parser_failure_to_one_worker():
    provider, _ = _make_provider({"1": "bad", "2": "good"})

    def parse(card_json: str) -> WorkerCapacity:
        if card_json == "bad":
            raise ValueError("bad card")
        return WorkerCapacity(retention_tokens=32_000, block_size=32)

    provider._parse_capacity = parse  # type: ignore[method-assign]
    assert provider.snapshot() == {
        2: WorkerCapacity(retention_tokens=32_000, block_size=32)
    }


def test_parsed_cards_cache_hits_on_repeat_snapshot():
    cards = {"1": _card(16, 1000)}
    provider, _ = _make_provider(cards)
    provider.snapshot()
    assert provider._parsed == {
        1: (cards["1"], WorkerCapacity(retention_tokens=16_000, block_size=16))
    }
    provider._parsed[1] = (
        cards["1"],
        WorkerCapacity(retention_tokens=999_999, block_size=32),
    )
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=999_999, block_size=32)
    }


def test_parsed_cards_cache_tracks_only_current_worker_cards():
    cards = {"1": _card(16, 1000), "2": _card(8, 2000)}
    provider, subscriber = _make_provider(cards)
    provider.snapshot()

    updated_card = _card(32, 500)
    subscriber._cards = {"1": updated_card}
    assert provider.snapshot() == {
        1: WorkerCapacity(retention_tokens=16_000, block_size=32)
    }
    assert provider._parsed == {
        1: (updated_card, WorkerCapacity(retention_tokens=16_000, block_size=32))
    }


def test_snapshot_returns_empty_when_subscriber_unset():
    provider = WorkerCapacityProvider(endpoint=None)  # type: ignore[arg-type]
    assert provider.snapshot() == {}


@pytest.mark.parametrize(
    ("retention_tokens", "block_size", "field"),
    [
        (0, 16, "retention_tokens"),
        (-1, 16, "retention_tokens"),
        (True, 16, "retention_tokens"),
        (1.5, 16, "retention_tokens"),
        (100, 0, "block_size"),
        (100, -1, "block_size"),
        (100, False, "block_size"),
        (100, 1.5, "block_size"),
    ],
)
def test_worker_capacity_requires_positive_integers(
    retention_tokens, block_size, field
):
    with pytest.raises(ValueError, match=field):
        WorkerCapacity(
            retention_tokens=retention_tokens,
            block_size=block_size,
        )
