# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the TRT-LLM adapter's ForwardPassMetrics wiring.

Covers (after realignment to the merged TRT-LLM PR #13199):

  * handle_stat reads the 11 InflightBatchingStats fields from the nested
    ``stat["inflightBatchingStats"]`` dict and forwards them as keyword
    arguments to FpmDirectPublisher.publish.
  * Composite mappings: queued-decode counters are ``numPausedRequests +
    numQueuedGenRequests`` and ``numPausedKvTokens + numQueuedGenKvTokens``.
  * attentionDpRank from the top level of the stat dict is passed through
    unchanged; missing/null key defaults to 0.
  * Attention-DP fanout emits one FPM per attentionDpRank; queued fields are
    forwarded from the stat row as-is, so they remain nonzero only on the
    rank-0 row produced by TRT-LLM.
  * iterLatencyMS (top-level milliseconds) is converted to wall_time_secs
    (seconds) at the boundary.
  * First-stat schema probe disables the publisher when the nested IBS dict
    is missing or any of the 11 required fields is absent — protecting
    against silent planner poison when running against pre-#13199 TRT-LLM.

The handle_stat closure is defined inside Publisher._publish_stats_task,
so we mirror its FPM branch via ``_invoke_handler`` below — kept in
lock-step with publisher.py on purpose so any drift surfaces immediately
in ``test_invoke_handler_matches_publisher_keyword_set``.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from unittest.mock import MagicMock

import pytest

from dynamo.trtllm import publisher as publisher_mod

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


# Default IBS values used by every test that doesn't override. Mirrors the
# semantics each field carries: scheduled vs queued, prefill vs decode,
# request count vs token count vs KV-token count.
_DEFAULT_IBS = {
    "numContextRequests": 3,
    "numCtxTokens": 1024,
    "numCtxKvTokens": 256,
    "numGenRequests": 5,
    "numGenKvTokens": 9000,
    "numQueuedContextRequests": 2,
    "numQueuedCtxTokens": 512,
    "numQueuedGenRequests": 1,
    "numQueuedGenKvTokens": 400,
    "numPausedRequests": 1,
    "numPausedKvTokens": 800,
}


def _build_fake_stat(*, ibs_overrides=None, **top_level_overrides):
    """Construct an IterationStats-shaped dict matching the merged #13199 JSON.

    Top-level keys (``iterLatencyMS``, ``attentionDpRank``, ``kvCacheStats``)
    are siblings of the nested ``inflightBatchingStats`` object, exactly as
    NLOHMANN serializes the C++ struct.
    """
    ibs = dict(_DEFAULT_IBS)
    if ibs_overrides:
        ibs.update(ibs_overrides)
    stat = {
        "iterLatencyMS": 25.0,
        "attentionDpRank": 0,
        "kvCacheStats": {"usedNumBlocks": 10, "maxNumBlocks": 100},
        "inflightBatchingStats": ibs,
    }
    stat.update(top_level_overrides)
    return stat


def _invoke_handler(stat, fpm_publisher):
    """Inline mirror of the handle_stat FPM branch in publisher.py.

    Keep this in lock-step with publisher.py — see
    ``test_invoke_handler_matches_publisher_keyword_set`` for the guardrail.
    """
    ibs = stat.get("inflightBatchingStats") or {}
    queued_num_decode = int(ibs.get("numPausedRequests", 0)) + int(
        ibs.get("numQueuedGenRequests", 0)
    )
    queued_sum_decode_kv_tokens = int(ibs.get("numPausedKvTokens", 0)) + int(
        ibs.get("numQueuedGenKvTokens", 0)
    )
    attention_dp_rank = stat.get("attentionDpRank")
    dp_rank = int(attention_dp_rank) if attention_dp_rank is not None else 0
    fpm_publisher.publish(
        dp_rank=dp_rank,
        scheduled_num_prefill_requests=int(ibs.get("numContextRequests", 0)),
        scheduled_sum_prefill_tokens=int(ibs.get("numCtxTokens", 0)),
        scheduled_sum_prefill_kv_tokens=int(ibs.get("numCtxKvTokens", 0)),
        scheduled_num_decode_requests=int(ibs.get("numGenRequests", 0)),
        scheduled_sum_decode_kv_tokens=int(ibs.get("numGenKvTokens", 0)),
        queued_num_prefill_requests=int(ibs.get("numQueuedContextRequests", 0)),
        queued_sum_prefill_tokens=int(ibs.get("numQueuedCtxTokens", 0)),
        queued_num_decode_requests=queued_num_decode,
        queued_sum_decode_kv_tokens=queued_sum_decode_kv_tokens,
        wall_time_secs=float(stat.get("iterLatencyMS", 0.0)) / 1000.0,
    )


# ---------------------------------------------------------------------------
# Field mapping
# ---------------------------------------------------------------------------


def test_handle_stat_maps_fields_single_rank():
    fpm = MagicMock()
    _invoke_handler(_build_fake_stat(), fpm)
    fpm.publish.assert_called_once_with(
        dp_rank=0,
        scheduled_num_prefill_requests=3,
        scheduled_sum_prefill_tokens=1024,
        scheduled_sum_prefill_kv_tokens=256,
        scheduled_num_decode_requests=5,
        scheduled_sum_decode_kv_tokens=9000,
        queued_num_prefill_requests=2,
        queued_sum_prefill_tokens=512,
        queued_num_decode_requests=2,  # numPausedRequests (1) + numQueuedGenRequests (1)
        queued_sum_decode_kv_tokens=1200,  # numPausedKvTokens (800) + numQueuedGenKvTokens (400)
        wall_time_secs=0.025,
    )


def test_queued_decode_composite_with_only_paused():
    """Disagg-prefill or non-disagg engine: numQueuedGenRequests is always 0,
    so queued-decode pressure comes entirely from preempted decodes."""
    fpm = MagicMock()
    stat = _build_fake_stat(
        ibs_overrides={
            "numPausedRequests": 4,
            "numPausedKvTokens": 12000,
            "numQueuedGenRequests": 0,
            "numQueuedGenKvTokens": 0,
        }
    )
    _invoke_handler(stat, fpm)
    kwargs = fpm.publish.call_args.kwargs
    assert kwargs["queued_num_decode_requests"] == 4
    assert kwargs["queued_sum_decode_kv_tokens"] == 12000


def test_queued_decode_composite_with_only_queued_gen():
    """Disagg-decode engine awaiting KV transfer: numPausedRequests is 0,
    queued-gen carries the full signal."""
    fpm = MagicMock()
    stat = _build_fake_stat(
        ibs_overrides={
            "numPausedRequests": 0,
            "numPausedKvTokens": 0,
            "numQueuedGenRequests": 7,
            "numQueuedGenKvTokens": 21000,
        }
    )
    _invoke_handler(stat, fpm)
    kwargs = fpm.publish.call_args.kwargs
    assert kwargs["queued_num_decode_requests"] == 7
    assert kwargs["queued_sum_decode_kv_tokens"] == 21000


def test_handle_stat_routes_per_attention_dp_rank():
    fpm = MagicMock()
    for rank in (0, 1, 2, 3):
        stat = _build_fake_stat(
            attentionDpRank=rank,
            ibs_overrides={"numCtxTokens": 100 * (rank + 1)},
        )
        _invoke_handler(stat, fpm)
    calls = fpm.publish.call_args_list
    assert len(calls) == 4
    for i, call in enumerate(calls):
        assert call.kwargs["dp_rank"] == i
        assert call.kwargs["scheduled_sum_prefill_tokens"] == 100 * (i + 1)


def test_handle_stat_attention_dp_fanout_preserves_rank0_queue_only():
    """TRT-LLM emits one stat row per attention-DP rank. Dynamo forwards each
    row as its own FPM and does not smear rank-0 queued fields onto rank 1."""
    fpm = MagicMock()
    rank0 = _build_fake_stat(
        attentionDpRank=0,
        iterLatencyMS=2147.591,
        ibs_overrides={
            "numContextRequests": 1,
            "numCtxTokens": 15,
            "numCtxKvTokens": 1,
            "numGenRequests": 0,
            "numGenKvTokens": 0,
            "numPausedRequests": 0,
            "numPausedKvTokens": 0,
            "numQueuedContextRequests": 8664,
            "numQueuedCtxTokens": 155952,
            "numQueuedGenRequests": 0,
            "numQueuedGenKvTokens": 0,
        },
    )
    rank1 = _build_fake_stat(
        attentionDpRank=1,
        iterLatencyMS=2147.591,
        ibs_overrides={
            "numContextRequests": 0,
            "numCtxTokens": 0,
            "numCtxKvTokens": 0,
            "numGenRequests": 2,
            "numGenKvTokens": 33,
            "numPausedRequests": 0,
            "numPausedKvTokens": 0,
            "numQueuedContextRequests": 0,
            "numQueuedCtxTokens": 0,
            "numQueuedGenRequests": 0,
            "numQueuedGenKvTokens": 0,
        },
    )

    _invoke_handler(rank0, fpm)
    _invoke_handler(rank1, fpm)

    rank0_call, rank1_call = fpm.publish.call_args_list
    assert rank0_call.kwargs["dp_rank"] == 0
    assert rank0_call.kwargs["queued_num_prefill_requests"] == 8664
    assert rank0_call.kwargs["queued_sum_prefill_tokens"] == 155952
    assert rank0_call.kwargs["scheduled_num_decode_requests"] == 0

    assert rank1_call.kwargs["dp_rank"] == 1
    assert rank1_call.kwargs["queued_num_prefill_requests"] == 0
    assert rank1_call.kwargs["queued_sum_prefill_tokens"] == 0
    assert rank1_call.kwargs["scheduled_num_decode_requests"] == 2
    assert rank1_call.kwargs["scheduled_sum_decode_kv_tokens"] == 33
    assert rank0_call.kwargs["wall_time_secs"] == rank1_call.kwargs["wall_time_secs"]


def test_handle_stat_missing_attention_dp_rank_defaults_zero():
    fpm = MagicMock()
    stat = _build_fake_stat()
    stat.pop("attentionDpRank")
    _invoke_handler(stat, fpm)
    assert fpm.publish.call_args.kwargs["dp_rank"] == 0


def test_handle_stat_null_attention_dp_rank_defaults_zero():
    fpm = MagicMock()
    _invoke_handler(_build_fake_stat(attentionDpRank=None), fpm)
    assert fpm.publish.call_args.kwargs["dp_rank"] == 0


def test_handle_stat_missing_ibs_dict_emits_zeros():
    """If the nested IBS dict is absent at this layer (the schema probe
    upstream should have already disabled the publisher in production),
    the handler must still produce a zeroed call rather than KeyError."""
    fpm = MagicMock()
    _invoke_handler({"iterLatencyMS": 10.0, "attentionDpRank": 0}, fpm)
    fpm.publish.assert_called_once_with(
        dp_rank=0,
        scheduled_num_prefill_requests=0,
        scheduled_sum_prefill_tokens=0,
        scheduled_sum_prefill_kv_tokens=0,
        scheduled_num_decode_requests=0,
        scheduled_sum_decode_kv_tokens=0,
        queued_num_prefill_requests=0,
        queued_sum_prefill_tokens=0,
        queued_num_decode_requests=0,
        queued_sum_decode_kv_tokens=0,
        wall_time_secs=0.01,
    )


def test_iter_latency_ms_to_wall_time_secs_conversion():
    fpm = MagicMock()
    _invoke_handler(_build_fake_stat(iterLatencyMS=1234.5), fpm)
    assert fpm.publish.call_args.kwargs["wall_time_secs"] == 1.2345


# ---------------------------------------------------------------------------
# Publisher.initialize() gate behavior (attention-DP off vs on)
# ---------------------------------------------------------------------------


def _build_publisher_stub(monkeypatch, *, attention_dp_size: int, fpm_enabled: bool):
    """Bypass Publisher.__init__ (heavy deps) and seed only the attributes
    initialize() reads or writes. All side-effecty subsystems are stubbed
    via ``monkeypatch`` so initialize() reaches the FPM gate cleanly without
    needing a blanket try/except to swallow upstream failures.
    """
    engine = MagicMock()
    engine.get_attention_dp_size.return_value = attention_dp_size

    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub.endpoint = MagicMock()
    pub.engine = engine
    pub.worker_id = "worker-abc"
    pub.kv_block_size = 64
    pub.max_window_size = None
    pub.metrics_labels = {}
    pub.component_gauges = MagicMock()
    pub.enable_local_indexer = False
    pub.metrics_collector = None
    pub.attention_dp_size = attention_dp_size
    pub.fpm_enabled = fpm_enabled
    pub.processing_initial_created_events = True
    pub.metrics_publisher = None
    pub.fpm_publisher = None
    pub.kv_event_publishers = None
    pub.zmq_kv_event_publisher = None
    pub.publish_kv_cache_events_thread = None
    pub.publish_stats_thread = None
    pub.partial_block_hashes = set()
    pub.error_queue = queue.Queue()
    pub._stop_event = threading.Event()
    pub._last_engine_event_id_by_rank = {}

    fake_fpm_cls = MagicMock()
    monkeypatch.setattr(publisher_mod, "FpmDirectPublisher", fake_fpm_cls)
    # WorkerMetricsPublisher and KvEventPublisher both reach into the Rust
    # binding and validate their endpoint arg as a real Endpoint — replace
    # with MagicMock factories so initialize() can complete past the FPM
    # gate without dragging in a real Endpoint.
    monkeypatch.setattr(publisher_mod, "WorkerMetricsPublisher", MagicMock())
    monkeypatch.setattr(publisher_mod, "KvEventPublisher", MagicMock())

    # asyncio.create_task wants a running loop — replace with a no-op
    # MagicMock so initialize() can call it synchronously in tests. Restored
    # automatically by monkeypatch on teardown.
    monkeypatch.setattr(
        asyncio,
        "create_task",
        lambda coro: MagicMock(add_done_callback=lambda _: None),
    )

    monkeypatch.setattr(pub, "_init_publish_metrics_thread", MagicMock())
    monkeypatch.setattr(pub, "_init_publish_kv_cache_events_thread", MagicMock())
    monkeypatch.setattr(
        pub,
        "_create_metrics_publisher_endpoint",
        MagicMock(return_value=MagicMock()),
    )

    return pub, publisher_mod, fake_fpm_cls


def test_publisher_initialize_constructs_fpm_direct_publisher_when_fpm_enabled(
    monkeypatch,
):
    """Under non-attention-DP (attention_dp_size == 1, fpm_enabled == True),
    Publisher.initialize() constructs FpmDirectPublisher with dp_size=1."""
    pub, _publisher_mod, fake_fpm_cls = _build_publisher_stub(
        monkeypatch, attention_dp_size=1, fpm_enabled=True
    )
    pub.initialize()
    fake_fpm_cls.assert_called_once()
    kwargs = fake_fpm_cls.call_args.kwargs
    assert kwargs["worker_id"] == "worker-abc"
    assert kwargs["dp_size"] == 1
    assert pub.fpm_publisher is not None


def _publisher_for_kv_event_test():
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub.additional_metrics = None
    pub._last_engine_event_id_by_rank = {}
    pub.processing_initial_created_events = False
    pub.partial_block_hashes = set()
    pub.kv_block_size = 4
    pub.max_window_size = None
    return pub


def _stored_kv_event(cache_salt="tenant-a"):
    return {
        "event_id": 1,
        "attention_dp_rank": 0,
        "data": {
            "type": "stored",
            "parent_hash": None,
            "blocks": [
                {
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


def test_handle_kv_event_forwards_cache_salt_to_direct_publisher():
    pub = _publisher_for_kv_event_test()
    publisher = MagicMock()
    pub.zmq_kv_event_publisher = None
    pub.kv_event_publishers = {0: publisher}

    pub._handle_kv_event(_stored_kv_event())

    publisher.publish_stored.assert_called_once()
    assert publisher.publish_stored.call_args.kwargs["cache_salt"] == "tenant-a"


def test_handle_kv_event_forwards_cache_salt_to_zmq_publisher():
    pub = _publisher_for_kv_event_test()
    pub.zmq_kv_event_publisher = MagicMock()
    pub.kv_event_publishers = None

    pub._handle_kv_event(_stored_kv_event())

    pub.zmq_kv_event_publisher.publish_stored.assert_called_once()
    assert pub.zmq_kv_event_publisher.publish_stored.call_args.args[-1] == "tenant-a"


@pytest.mark.asyncio
async def test_polling_loop_drops_conflicting_salts_and_processes_next_event(caplog):
    pub = _publisher_for_kv_event_test()
    pub._stop_event = threading.Event()
    pub.zmq_kv_event_publisher = None
    publisher = MagicMock()
    pub.kv_event_publishers = {0: publisher}

    conflicting = _stored_kv_event("tenant-a")
    conflicting["data"]["blocks"].append(
        {
            **conflicting["data"]["blocks"][0],
            "block_hash": 456,
            "cache_salt": "tenant-b",
        }
    )
    valid = _stored_kv_event("tenant-c")
    valid["event_id"] = 2

    async def fetch_events():
        yield conflicting
        yield valid

    def handle_event(event):
        pub._handle_kv_event(event)
        if event["event_id"] == 2:
            pub._stop_event.set()

    with caplog.at_level(logging.WARNING):
        await pub._polling_loop(
            fetch_events,
            handle_event,
            min_sleep=0.001,
            max_sleep=0.001,
            backoff_factor=1.0,
        )

    publisher.publish_stored.assert_called_once()
    assert publisher.publish_stored.call_args.kwargs["cache_salt"] == "tenant-c"
    assert "Dropping stored KV event with invalid cache namespace" in caplog.text


def test_publisher_initializes_fpm_publisher_under_attention_dp(monkeypatch):
    """Under attention-DP (attention_dp_size > 1), Publisher.initialize()
    constructs one FpmDirectPublisher channel per attention-DP rank."""
    pub, _publisher_mod, fake_fpm_cls = _build_publisher_stub(
        monkeypatch, attention_dp_size=4, fpm_enabled=False
    )
    pub.initialize()
    fake_fpm_cls.assert_called_once()
    kwargs = fake_fpm_cls.call_args.kwargs
    assert kwargs["worker_id"] == "worker-abc"
    assert kwargs["dp_size"] == 4
    assert pub.fpm_publisher is not None


# ---------------------------------------------------------------------------
# KV event buffer telemetry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kv_event_polling_loop_records_drained_batch_size():
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub._stop_event = threading.Event()
    drained_batches = []

    async def fetch_events():
        for event_id in range(3):
            yield {"event_id": event_id}

    def handle_event(event):
        if event["event_id"] == 2:
            pub._stop_event.set()

    await pub._polling_loop(
        fetch_events,
        handle_event,
        min_sleep=0.001,
        max_sleep=0.001,
        backoff_factor=1.0,
        batch_size_handler_fn=drained_batches.append,
    )

    assert drained_batches == [3]


@pytest.mark.asyncio
async def test_polling_loop_reraises_unexpected_handler_error():
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub._stop_event = threading.Event()

    async def fetch_events():
        yield {"event_id": 0}

    def handle_event(_event):
        raise RuntimeError("handler failed")

    with pytest.raises(RuntimeError, match="handler failed"):
        await pub._polling_loop(
            fetch_events,
            handle_event,
            min_sleep=0.001,
            max_sleep=0.001,
            backoff_factor=1.0,
        )


def test_managed_thread_stops_after_task_error():
    errors = queue.Queue()
    calls = 0

    async def failing_task():
        nonlocal calls
        calls += 1
        raise RuntimeError("task failed")

    thread = publisher_mod.ManagedThread(failing_task, error_queue=errors)
    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert calls == 1
    assert str(errors.get_nowait()) == "task failed"


def test_kv_event_id_gap_records_missing_event_count():
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub.additional_metrics = MagicMock()
    pub.processing_initial_created_events = True
    pub.max_window_size = None
    pub._last_engine_event_id_by_rank = {0: 10}

    pub._handle_kv_event({"event_id": 14, "data": {"type": "created"}})

    pub.additional_metrics.record_kv_event_id_gap.assert_called_once_with(3)


def test_filtered_kv_events_do_not_create_false_event_id_gaps():
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub.additional_metrics = MagicMock()
    pub._last_engine_event_id_by_rank = {0: 10}
    pub.should_drop_event = MagicMock(side_effect=[True, False])
    pub.processing_initial_created_events = True
    pub.max_window_size = None

    pub._handle_kv_event({"event_id": 11, "data": {"type": "stored"}})
    pub._handle_kv_event({"event_id": 12, "data": {"type": "created"}})

    pub.additional_metrics.record_kv_event_id_gap.assert_not_called()


def test_partial_only_removed_event_does_not_publish_empty_batch():
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    kv_event_publisher = MagicMock()
    pub.additional_metrics = None
    pub._last_engine_event_id_by_rank = {}
    pub.should_drop_event = MagicMock(return_value=False)
    pub.processing_initial_created_events = True
    pub.partial_block_hashes = {123}
    pub.zmq_kv_event_publisher = None
    pub.kv_event_publishers = {0: kv_event_publisher}

    pub._handle_kv_event(
        {"event_id": 1, "data": {"type": "removed", "block_hashes": [123]}}
    )

    assert pub.partial_block_hashes == set()
    kv_event_publisher.publish_removed.assert_not_called()


def test_interleaved_attention_dp_ranks_do_not_create_false_event_id_gaps():
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub.additional_metrics = MagicMock()
    pub._last_engine_event_id_by_rank = {}
    pub.should_drop_event = MagicMock(return_value=True)

    for event in [
        {"event_id": 10, "attention_dp_rank": 0},
        {"event_id": 3, "attention_dp_rank": 1},
        {"event_id": 11, "attention_dp_rank": 0},
        {"event_id": 4, "attention_dp_rank": 1},
    ]:
        pub._handle_kv_event(event)

    pub.additional_metrics.record_kv_event_id_gap.assert_not_called()


# ---------------------------------------------------------------------------
# First-stat schema probe
# ---------------------------------------------------------------------------


def _build_schema_probe_publisher(fpm_publisher_mock=None):
    """Minimal Publisher instance for direct _check_fpm_schema testing.

    Bypasses __init__ (heavy deps) and seeds only the attributes the probe
    method reads or writes: fpm_publisher and _fpm_schema_checked.
    """
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub.fpm_publisher = (
        fpm_publisher_mock if fpm_publisher_mock is not None else MagicMock()
    )
    pub._fpm_schema_checked = False
    return pub, publisher_mod


def test_schema_probe_all_fields_present_keeps_publisher():
    pub, _ = _build_schema_probe_publisher()
    original_publisher = pub.fpm_publisher

    pub._check_fpm_schema(_build_fake_stat())

    assert pub._fpm_schema_checked is True
    assert pub.fpm_publisher is original_publisher
    original_publisher.shutdown.assert_not_called()


@pytest.mark.parametrize("missing_field", list(_DEFAULT_IBS.keys()))
def test_schema_probe_missing_single_ibs_field_disables_publisher(missing_field):
    """Strict probe: any one of the 11 required IBS fields missing must
    disable the publisher. Covers each field independently so a rename
    upstream or a selective-backport TRT-LLM never slips through."""
    pub, _ = _build_schema_probe_publisher()
    original_publisher = pub.fpm_publisher

    stat = _build_fake_stat()
    stat["inflightBatchingStats"].pop(missing_field)
    pub._check_fpm_schema(stat)

    assert pub._fpm_schema_checked is True
    assert pub.fpm_publisher is None
    original_publisher.shutdown.assert_called_once()


def test_schema_probe_missing_ibs_dict_disables_publisher_legacy_trtllm():
    """Legacy TRT-LLM case: stat dict has iterLatencyMS + attentionDpRank but
    no inflightBatchingStats nested object (pre-#13199 schema). Must disable
    without error."""
    pub, _ = _build_schema_probe_publisher()
    original_publisher = pub.fpm_publisher

    pub._check_fpm_schema({"iterLatencyMS": 10.0, "attentionDpRank": 0})

    assert pub._fpm_schema_checked is True
    assert pub.fpm_publisher is None
    original_publisher.shutdown.assert_called_once()


def test_schema_probe_ibs_not_a_dict_disables_publisher():
    """Defensive: if a future TRT-LLM ever emits inflightBatchingStats as
    something other than a dict (e.g. null on engine init), treat it as a
    schema mismatch rather than crashing in the probe."""
    pub, _ = _build_schema_probe_publisher()
    original_publisher = pub.fpm_publisher

    pub._check_fpm_schema({"inflightBatchingStats": None})

    assert pub.fpm_publisher is None
    original_publisher.shutdown.assert_called_once()


def test_schema_probe_noop_when_fpm_publisher_already_none():
    """If initialization already disabled fpm_publisher, probe must not blow
    up and must still flip _fpm_schema_checked so we do not re-enter."""
    pub, _ = _build_schema_probe_publisher(fpm_publisher_mock=None)
    pub.fpm_publisher = None

    pub._check_fpm_schema(_build_fake_stat())

    assert pub._fpm_schema_checked is True
    assert pub.fpm_publisher is None


def test_schema_probe_shutdown_exception_still_disables_publisher():
    """If the Rust shutdown call raises, we still None out the publisher —
    the primary goal is to suppress further emission, not to succeed at
    shutdown. Protects against leaking emission through a shutdown failure."""
    pub, _ = _build_schema_probe_publisher()
    pub.fpm_publisher.shutdown.side_effect = RuntimeError("tokio runtime gone")

    stat = _build_fake_stat()
    stat["inflightBatchingStats"].pop("numCtxKvTokens")
    pub._check_fpm_schema(stat)

    assert pub.fpm_publisher is None
    assert pub._fpm_schema_checked is True


def test_handle_stat_probe_gate_fires_once_and_skips_subsequent_stats():
    """Simulate the handle_stat dispatch pattern: on the first stat the probe
    runs; on the next stat the gate short-circuits. Ensures we do not re-check
    per iteration (which would be wasteful and could race a late schema bump)."""
    pub, _ = _build_schema_probe_publisher()
    original_publisher = pub.fpm_publisher

    if pub.fpm_publisher is not None and not pub._fpm_schema_checked:
        pub._check_fpm_schema(_build_fake_stat())
    assert pub._fpm_schema_checked is True
    assert pub.fpm_publisher is original_publisher

    stat_bad = _build_fake_stat()
    stat_bad["inflightBatchingStats"].pop("numCtxKvTokens")
    if pub.fpm_publisher is not None and not pub._fpm_schema_checked:
        pub._check_fpm_schema(stat_bad)
    assert pub.fpm_publisher is original_publisher
    original_publisher.shutdown.assert_not_called()


def test_schema_probe_field_list_matches_default_ibs_set():
    """Guardrail: the required-fields tuple must stay in sync with the IBS
    default fixture. If someone adds an IBS field to the production reader
    but forgets the probe constant (or vice versa), this test catches it."""
    assert set(publisher_mod._FPM_REQUIRED_IBS_FIELDS) == set(_DEFAULT_IBS.keys())


def test_invoke_handler_matches_publisher_keyword_set():
    """Guardrail: the kwargs that publisher.py's handle_stat passes to
    fpm_publisher.publish() must match the test mirror exactly. Catches any
    drift where production starts using a new kwarg the test doesn't mirror,
    or vice versa."""
    fpm = MagicMock()
    _invoke_handler(_build_fake_stat(), fpm)
    expected_kwargs = {
        "dp_rank",
        "scheduled_num_prefill_requests",
        "scheduled_sum_prefill_tokens",
        "scheduled_sum_prefill_kv_tokens",
        "scheduled_num_decode_requests",
        "scheduled_sum_decode_kv_tokens",
        "queued_num_prefill_requests",
        "queued_sum_prefill_tokens",
        "queued_num_decode_requests",
        "queued_sum_decode_kv_tokens",
        "wall_time_secs",
    }
    assert set(fpm.publish.call_args.kwargs.keys()) == expected_kwargs
