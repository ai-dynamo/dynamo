# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _build_zmq_dispatch_publisher():
    from dynamo.trtllm import publisher as publisher_mod

    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub.zmq_kv_event_publisher = MagicMock()
    pub.kv_event_publishers = None
    pub.lower_tier_kv_event_publishers = {1: MagicMock()}
    return pub


def test_zmq_path_keeps_device_store_on_consolidator():
    pub = _build_zmq_dispatch_publisher()

    pub._dispatch_stored(
        token_ids=[1, 2],
        num_block_tokens=[2],
        block_hashes=[123],
        parent_hash=99,
        block_mm_infos=[None],
        lora_name=None,
        attention_dp_rank=1,
        storage_tier="device",
    )

    pub.zmq_kv_event_publisher.publish_stored.assert_called_once_with(
        [1, 2],
        [2],
        [123],
        99,
        [None],
        1,
        None,
    )
    pub.lower_tier_kv_event_publishers[1].publish_stored.assert_not_called()


def test_zmq_path_sends_host_pinned_store_to_direct_tier_publisher():
    pub = _build_zmq_dispatch_publisher()

    pub._dispatch_stored(
        token_ids=[1, 2],
        num_block_tokens=[2],
        block_hashes=[123],
        parent_hash=99,
        block_mm_infos=[None],
        lora_name="adapter",
        attention_dp_rank=1,
        storage_tier="host_pinned",
    )

    pub.zmq_kv_event_publisher.publish_stored.assert_not_called()
    pub.lower_tier_kv_event_publishers[1].publish_stored.assert_called_once_with(
        [1, 2],
        [2],
        [123],
        99,
        [None],
        lora_name="adapter",
        storage_tier="host_pinned",
    )


def test_zmq_path_sends_host_pinned_remove_to_direct_tier_publisher():
    pub = _build_zmq_dispatch_publisher()

    pub._dispatch_removed(
        block_hashes=[123],
        attention_dp_rank=1,
        storage_tier="host_pinned",
    )

    pub.zmq_kv_event_publisher.publish_removed.assert_not_called()
    pub.lower_tier_kv_event_publishers[1].publish_removed.assert_called_once_with(
        [123],
        storage_tier="host_pinned",
    )


def test_lower_tier_publishers_reuse_borrowed_rank_zero_and_create_missing_ranks(
    monkeypatch,
):
    from dynamo.trtllm import publisher as publisher_mod

    created = []

    def fake_kv_event_publisher(**kwargs):
        created.append(kwargs)
        return MagicMock(name=f"lower-tier-rank-{kwargs['dp_rank']}")

    monkeypatch.setattr(publisher_mod, "KvEventPublisher", fake_kv_event_publisher)

    borrowed_rank_zero = MagicMock()
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub.endpoint = MagicMock()
    pub.worker_id = 1234
    pub.kv_block_size = 64
    pub.attention_dp_size = 2
    pub.enable_local_indexer = True
    pub.zmq_kv_event_publisher = MagicMock()
    pub.lower_tier_kv_event_publishers = {0: borrowed_rank_zero}
    pub._owned_lower_tier_publisher_ranks = set()

    pub._init_lower_tier_kv_event_publishers()

    assert pub.lower_tier_kv_event_publishers[0] is borrowed_rank_zero
    assert pub._owned_lower_tier_publisher_ranks == {1}
    assert len(created) == 1
    assert created[0]["dp_rank"] == 1
    assert created[0]["enable_local_indexer"] is True


def test_direct_publishers_initialize_kv_event_polling_thread(monkeypatch):
    from dynamo.trtllm import publisher as publisher_mod

    class FakeTask:
        def add_done_callback(self, callback):
            return None

    created = []

    def fake_kv_event_publisher(**kwargs):
        created.append(kwargs)
        return MagicMock(name=f"direct-rank-{kwargs['dp_rank']}")

    monkeypatch.setattr(publisher_mod, "KvEventPublisher", fake_kv_event_publisher)
    monkeypatch.setattr(
        publisher_mod, "WorkerMetricsPublisher", lambda: MagicMock()
    )

    def fake_create_task(coro):
        coro.close()
        return FakeTask()

    monkeypatch.setattr(publisher_mod.asyncio, "create_task", fake_create_task)

    engine = MagicMock()
    engine.get_attention_dp_size.return_value = 2
    pub = publisher_mod.Publisher(
        endpoint=MagicMock(),
        engine=engine,
        worker_id=1234,
        kv_block_size=64,
        metrics_labels=[],
        component_gauges=MagicMock(),
        zmq_endpoint=None,
        enable_local_indexer=True,
    )

    pub.initialize()

    assert pub.publish_kv_cache_events_thread is not None
    assert len(created) == 2
    assert [item["dp_rank"] for item in created] == [0, 1]
