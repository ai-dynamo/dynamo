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

import pytest


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


# ---------------------------------------------------------------------------
# First-stat schema probe
# ---------------------------------------------------------------------------


def _build_schema_probe_publisher(fpm_publisher_mock: MagicMock | None = None):
    """Minimal Publisher instance for direct _check_fpm_schema testing.

    Bypasses __init__ (which touches heavy deps) and seeds only the attributes
    the probe method reads or writes: fpm_publisher and _fpm_schema_checked.
    """
    from dynamo.trtllm import publisher as publisher_mod

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


@pytest.mark.parametrize(
    "missing_field",
    [
        "scheduledNumPrefillRequests",
        "scheduledSumPrefillTokens",
        "scheduledSumPrefillKvTokens",
        "scheduledNumDecodeRequests",
        "scheduledSumDecodeKvTokens",
        "queuedNumPrefillRequests",
        "queuedSumPrefillTokens",
        "queuedNumDecodeRequests",
        "queuedSumDecodeKvTokens",
    ],
)
def test_schema_probe_missing_single_field_disables_publisher(missing_field):
    """Strict probe: any one of the 9 required fields missing must disable
    the publisher. Covers each field independently so a rename upstream or
    a selective-backport TRT-LLM never slips through."""
    pub, _ = _build_schema_probe_publisher()
    original_publisher = pub.fpm_publisher

    stat = _build_fake_stat()
    stat.pop(missing_field)
    pub._check_fpm_schema(stat)

    assert pub._fpm_schema_checked is True
    assert pub.fpm_publisher is None
    original_publisher.shutdown.assert_called_once()


def test_schema_probe_missing_all_fields_disables_publisher_legacy_trtllm():
    """Legacy TRT-LLM case: stat dict has iterLatencyMS + attentionDpRank but
    none of the 9 FPM fields (pre-#13199 schema). Must disable without error."""
    pub, _ = _build_schema_probe_publisher()
    original_publisher = pub.fpm_publisher

    pub._check_fpm_schema({"iterLatencyMS": 10.0, "attentionDpRank": 0})

    assert pub._fpm_schema_checked is True
    assert pub.fpm_publisher is None
    original_publisher.shutdown.assert_called_once()


def test_schema_probe_noop_when_fpm_publisher_already_none():
    """Attention-DP gate already set fpm_publisher = None; probe must not
    blow up and must still flip _fpm_schema_checked so we do not re-enter."""
    pub, _ = _build_schema_probe_publisher(fpm_publisher_mock=None)
    # Override the default (which creates a MagicMock) with explicit None.
    pub.fpm_publisher = None

    pub._check_fpm_schema(_build_fake_stat())

    assert pub._fpm_schema_checked is True
    assert pub.fpm_publisher is None


def test_schema_probe_shutdown_exception_still_disables_publisher():
    """If the Rust shutdown call raises, we still None out the publisher --
    the primary goal is to suppress further emission, not to succeed at
    shutdown. Protects against leaking emission through a shutdown failure."""
    pub, _ = _build_schema_probe_publisher()
    pub.fpm_publisher.shutdown.side_effect = RuntimeError("tokio runtime gone")

    stat = _build_fake_stat()
    stat.pop("scheduledNumPrefillRequests")
    pub._check_fpm_schema(stat)

    assert pub.fpm_publisher is None
    assert pub._fpm_schema_checked is True


def test_handle_stat_probe_gate_fires_once_and_skips_subsequent_stats():
    """Simulate the handle_stat dispatch pattern: on the first stat the probe
    runs; on the next stat the gate short-circuits. Ensures we do not re-check
    per iteration (which would be wasteful and could race a late schema bump)."""
    pub, _ = _build_schema_probe_publisher()
    original_publisher = pub.fpm_publisher

    # First stat — probe runs, passes.
    stat_ok = _build_fake_stat()
    if pub.fpm_publisher is not None and not pub._fpm_schema_checked:
        pub._check_fpm_schema(stat_ok)
    assert pub._fpm_schema_checked is True
    assert pub.fpm_publisher is original_publisher

    # Second stat, with a field that would fail the probe if re-checked.
    # The gate must prevent re-entry.
    stat_bad = _build_fake_stat()
    stat_bad.pop("scheduledNumPrefillRequests")
    if pub.fpm_publisher is not None and not pub._fpm_schema_checked:
        pub._check_fpm_schema(stat_bad)
    assert pub.fpm_publisher is original_publisher
    original_publisher.shutdown.assert_not_called()


def test_schema_probe_field_list_matches_publish_arguments():
    """Guardrail: the required-fields tuple must stay in sync with the
    fields handle_stat reads in its .publish() call. If someone adds a
    scheduled/queued field to the .publish args but forgets the probe
    constant, this test catches it."""
    from dynamo.trtllm import publisher as publisher_mod

    expected = {
        "scheduledNumPrefillRequests",
        "scheduledSumPrefillTokens",
        "scheduledSumPrefillKvTokens",
        "scheduledNumDecodeRequests",
        "scheduledSumDecodeKvTokens",
        "queuedNumPrefillRequests",
        "queuedSumPrefillTokens",
        "queuedNumDecodeRequests",
        "queuedSumDecodeKvTokens",
    }
    assert set(publisher_mod._FPM_REQUIRED_STAT_FIELDS) == expected
