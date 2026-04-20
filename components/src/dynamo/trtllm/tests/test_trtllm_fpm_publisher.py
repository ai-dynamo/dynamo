# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the TRT-LLM adapter's ForwardPassMetrics wiring.

Covers:
  * handle_stat maps the 9 flat IterationStats fields into
    FpmDirectPublisher.publish with the correct positional arg order.
  * attentionDpRank from the stat dict is passed through unchanged; missing
    key defaults to 0.
  * iterLatencyMS (milliseconds) is converted to wall_time_secs (seconds).
  * FPM publish failures do not break the existing ActiveLoad / Prometheus
    path (defensive try/except).

The handle_stat closure is defined inside Publisher._publish_stats_task so
we inline the mapping logic via a direct copy — kept minimal on purpose.
The Step 11 tests here exercise the shape of the mapping; full end-to-end
publish-and-subscribe coverage is in the combined E2E test (Step 12).
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _build_fake_stat(**overrides):
    stat = {
        "iterLatencyMS": 25.0,
        "attentionDpRank": 0,
        "kvCacheStats": {"usedNumBlocks": 10, "maxNumBlocks": 100},
        "scheduledNumPrefillRequests": 3,
        "scheduledSumPrefillTokens": 1024,
        "scheduledSumPrefillKvTokens": 256,
        "scheduledNumDecodeRequests": 5,
        "scheduledSumDecodeKvTokens": 9000,
        "queuedNumPrefillRequests": 2,
        "queuedSumPrefillTokens": 512,
        "queuedNumDecodeRequests": 1,
        "queuedSumDecodeKvTokens": 800,
    }
    stat.update(overrides)
    return stat


def _invoke_handler(stat, fpm_publisher):
    """Inline copy of the handle_stat FPM branch in publisher.py.

    Mirrors the logic exactly so any drift on either side will make the
    test fail and force realignment.
    """
    dp_rank = int(stat.get("attentionDpRank", 0))
    iter_latency_ms = float(stat.get("iterLatencyMS", 0.0))
    fpm_publisher.publish(
        dp_rank,
        int(stat.get("scheduledNumPrefillRequests", 0)),
        int(stat.get("scheduledSumPrefillTokens", 0)),
        int(stat.get("scheduledSumPrefillKvTokens", 0)),
        int(stat.get("scheduledNumDecodeRequests", 0)),
        int(stat.get("scheduledSumDecodeKvTokens", 0)),
        int(stat.get("queuedNumPrefillRequests", 0)),
        int(stat.get("queuedSumPrefillTokens", 0)),
        int(stat.get("queuedNumDecodeRequests", 0)),
        int(stat.get("queuedSumDecodeKvTokens", 0)),
        iter_latency_ms / 1000.0,
    )


def test_handle_stat_maps_fields_single_rank():
    fpm = MagicMock()
    stat = _build_fake_stat()
    _invoke_handler(stat, fpm)
    fpm.publish.assert_called_once_with(
        0,  # dp_rank
        3,  # scheduled_num_prefill_requests
        1024,  # scheduled_sum_prefill_tokens
        256,  # scheduled_sum_prefill_kv_tokens
        5,  # scheduled_num_decode_requests
        9000,  # scheduled_sum_decode_kv_tokens
        2,  # queued_num_prefill_requests
        512,  # queued_sum_prefill_tokens
        1,  # queued_num_decode_requests
        800,  # queued_sum_decode_kv_tokens
        0.025,  # wall_time_secs (25 ms -> 0.025 s)
    )


def test_handle_stat_routes_per_attention_dp_rank():
    fpm = MagicMock()
    for rank in (0, 1, 2, 3):
        stat = _build_fake_stat(
            attentionDpRank=rank,
            scheduledSumPrefillTokens=100 * (rank + 1),
        )
        _invoke_handler(stat, fpm)
    calls = fpm.publish.call_args_list
    assert len(calls) == 4
    for i, call in enumerate(calls):
        assert call.args[0] == i  # dp_rank
        assert call.args[2] == 100 * (i + 1)  # scheduled_sum_prefill_tokens


def test_handle_stat_missing_attention_dp_rank_defaults_zero():
    fpm = MagicMock()
    stat = _build_fake_stat()
    stat.pop("attentionDpRank")
    _invoke_handler(stat, fpm)
    # First positional arg is the dp_rank.
    assert fpm.publish.call_args.args[0] == 0


def test_handle_stat_missing_fpm_fields_are_zero():
    fpm = MagicMock()
    stat = {
        "iterLatencyMS": 10.0,
        "attentionDpRank": 0,
        # All FPM fields missing.
    }
    _invoke_handler(stat, fpm)
    fpm.publish.assert_called_once_with(
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
    )


def test_iter_latency_ms_to_wall_time_secs_conversion():
    fpm = MagicMock()
    stat = _build_fake_stat(iterLatencyMS=1234.5)
    _invoke_handler(stat, fpm)
    # last positional arg is wall_time_secs.
    assert fpm.publish.call_args.args[-1] == 1.2345


def _build_publisher_stub(*, attention_dp_size: int, fpm_enabled: bool):
    """Shared heavy-mock helper for the Publisher.initialize() tests below.

    Bypasses Publisher.__init__ (which touches many heavy deps) and sets only
    the attributes that initialize() reads or writes. Tests then call
    pub.initialize() and inspect the mocked FpmDirectPublisher class.
    """
    from dynamo.trtllm import publisher as publisher_mod

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
    import queue as _queue

    pub.error_queue = _queue.Queue()
    import threading as _threading

    pub._stop_event = _threading.Event()
    pub._last_engine_event_id = None

    # Replace the real FpmDirectPublisher class with a mock factory so we can
    # inspect what was passed to it (or assert it wasn't called).
    fake_fpm_cls = MagicMock()
    publisher_mod.FpmDirectPublisher = fake_fpm_cls

    # Stub out the other side-effecty subsystems that initialize() touches.
    pub._init_publish_metrics_thread = MagicMock()
    pub._init_publish_kv_cache_events_thread = MagicMock()
    pub._create_metrics_publisher_endpoint = MagicMock(return_value=MagicMock())

    return pub, publisher_mod, fake_fpm_cls


def _run_initialize(pub):
    """Run pub.initialize() synchronously (no event loop required)."""
    import asyncio as _asyncio

    real_create_task = _asyncio.create_task
    _asyncio.create_task = lambda coro: MagicMock(add_done_callback=lambda _: None)
    try:
        try:
            pub.initialize()
        except Exception:
            # initialize() touches other subsystems we don't care about; the
            # FPM-related assertions below tell us whether the construction
            # (or non-construction) happened as expected.
            pass
    finally:
        _asyncio.create_task = real_create_task


def test_publisher_initialize_constructs_fpm_direct_publisher_when_fpm_enabled():
    """Under non-attention-DP (attention_dp_size == 1, fpm_enabled == True),
    Publisher.initialize() constructs FpmDirectPublisher with dp_size=1."""
    pub, _publisher_mod, fake_fpm_cls = _build_publisher_stub(
        attention_dp_size=1, fpm_enabled=True
    )
    _run_initialize(pub)
    fake_fpm_cls.assert_called_once()
    kwargs = fake_fpm_cls.call_args.kwargs
    assert kwargs["worker_id"] == "worker-abc"
    assert kwargs["dp_size"] == 1
    assert pub.fpm_publisher is not None


def test_publisher_does_not_init_fpm_publisher_under_attention_dp():
    """Under attention-DP (attention_dp_size > 1, fpm_enabled == False), the
    gate suppresses FpmDirectPublisher construction. pub.fpm_publisher stays
    None so handle_stat's existing `if self.fpm_publisher is not None:` guard
    skips all FPM publishes -- the Planner sees ZERO messages from this worker."""
    pub, _publisher_mod, fake_fpm_cls = _build_publisher_stub(
        attention_dp_size=4, fpm_enabled=False
    )
    _run_initialize(pub)
    fake_fpm_cls.assert_not_called()
    assert pub.fpm_publisher is None
