# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine
except ImportError:
    pytest.skip("tensorrt_llm backend not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


def _stored_kv_event(cache_salt: str | None = "tenant-a") -> dict:
    return {
        "event_id": 1,
        "attention_dp_rank": 0,
        "data": {
            "type": "stored",
            "parent_hash": None,
            "blocks": [
                {
                    "type": "stored_block",
                    "block_hash": 123,
                    "cache_salt": cache_salt,
                    "tokens": [
                        {"token_id": 1},
                        {"token_id": 2},
                        {"token_id": 3},
                        {"token_id": 4},
                    ],
                }
            ],
        },
    }


def test_dispatch_kv_event_forwards_per_block_cache_salt() -> None:
    engine = TrtllmLLMEngine.__new__(TrtllmLLMEngine)
    publisher = MagicMock()
    engine._kv_publishers = {0: publisher}
    engine._last_event_id_by_rank = {}
    engine._warned_unknown_dp_rank = False
    engine._additional_metrics = None
    engine._partial_block_hashes_by_rank = {}
    engine.kv_block_size = 4

    engine._dispatch_kv_event(_stored_kv_event())

    publisher.publish_batch.assert_called_once()
    events = publisher.publish_batch.call_args.args[0]
    assert len(events) == 1
    assert events[0]["cache_salt"] == "tenant-a"


def test_dispatch_kv_events_groups_native_drain_by_rank_in_order() -> None:
    engine = TrtllmLLMEngine.__new__(TrtllmLLMEngine)
    rank_0 = MagicMock()
    rank_1 = MagicMock()
    engine._kv_publishers = {0: rank_0, 1: rank_1}
    engine._last_event_id_by_rank = {}
    engine._warned_unknown_dp_rank = False
    engine._warned_dispatch_failed = False
    engine._additional_metrics = None
    engine._partial_block_hashes_by_rank = {}
    engine.kv_block_size = 4

    first = _stored_kv_event()
    second = {
        "event_id": 1,
        "attention_dp_rank": 1,
        "data": {"type": "removed", "block_hashes": [200, 201]},
    }
    third = _stored_kv_event(cache_salt="tenant-b")
    third["event_id"] = 2
    third["data"]["blocks"][0]["block_hash"] = 124

    engine._dispatch_kv_events([first, second, third])

    rank_0.publish_batch.assert_called_once()
    rank_0_events = rank_0.publish_batch.call_args.args[0]
    assert [event["type"] for event in rank_0_events] == ["stored", "stored"]
    assert [event["cache_salt"] for event in rank_0_events] == [
        "tenant-a",
        "tenant-b",
    ]
    rank_1.publish_batch.assert_called_once_with(
        [{"type": "removed", "block_hashes": [200, 201]}]
    )


def test_dispatch_kv_events_skips_partial_blocks_without_empty_batches() -> None:
    engine = TrtllmLLMEngine.__new__(TrtllmLLMEngine)
    publisher = MagicMock()
    engine._kv_publishers = {0: publisher}
    engine._last_event_id_by_rank = {}
    engine._warned_unknown_dp_rank = False
    engine._warned_dispatch_failed = False
    engine._additional_metrics = None
    engine._partial_block_hashes_by_rank = {0: {123}}
    engine.kv_block_size = 4

    engine._dispatch_kv_events(
        [
            {
                "event_id": 1,
                "attention_dp_rank": 0,
                "data": {"type": "removed", "block_hashes": [123]},
            }
        ]
    )

    publisher.publish_batch.assert_not_called()
    assert engine._partial_block_hashes_by_rank[0] == set()
