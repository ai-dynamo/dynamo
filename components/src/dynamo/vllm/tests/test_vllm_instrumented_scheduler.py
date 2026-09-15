# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``InstrumentedScheduler._compute_queued`` classification.

Focus: correct handling of ``self.skipped_waiting`` and disaggregated-serving
request states. The production scheduler is heavy to construct (needs a real
``VllmConfig`` + ``KVCacheConfig`` + ``StructuredOutputManager``), so these
tests invoke ``_compute_queued`` as an unbound method against a minimal stub
built with ``object.__new__`` — this exercises the real function body without
spinning up vLLM engine internals.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_synthetic_content_env(monkeypatch):
    """Synthetic-prompt content selection and the giant-KV repeat knobs read
    the process environment; tests that want a non-default path set it
    explicitly."""
    monkeypatch.delenv("DYN_BENCH_PREFILL_CONTENT", raising=False)
    monkeypatch.delenv("DYN_BENCH_POOL_TAG", raising=False)
    monkeypatch.delenv("DYN_BENCH_PREFILL_REAL_SEED", raising=False)
    monkeypatch.delenv("DYN_BENCH_GIANT_KV_THRESHOLD", raising=False)
    monkeypatch.delenv("DYN_BENCH_GIANT_KV_REPEATS", raising=False)


# Module-level import: triggers real site-packages ``vllm`` to load before
# pytest's rootpath insertion adds ``components/src/dynamo`` to ``sys.path``
# (which shadows the real ``vllm`` with the ``dynamo.vllm`` submodule for any
# later bare ``import vllm``). Mirrors the pattern in ``test_vllm_unit.py``,
# which imports ``dynamo.vllm.args`` at module level for the same reason.
# If this import is deferred to inside a test body, the real ``vllm`` will
# not be resolvable and ``instrumented_scheduler`` will fail to load with
# ``ModuleNotFoundError: No module named 'vllm.sampling_params'``.
import dynamo.vllm.instrumented_scheduler as instrumented_scheduler_module  # noqa: E402
from dynamo.vllm.benchmark_points import (  # noqa: E402
    BenchmarkPoints,
    PrefillPointCandidate,
)
from dynamo.vllm.instrumented_scheduler import (  # noqa: E402
    EAGER_WARMUP_REASON,
    BenchmarkConfig,
    BenchmarkPoint,
    InstrumentedScheduler,
    SkippedBenchmarkPoint,
    _BenchPhase,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

STRUCTURED_OUTPUT_WAITING_STATUS = getattr(
    RequestStatus, "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR", None
) or getattr(RequestStatus, "WAITING_FOR_FSM")


def _benchmark_capacity(
    *,
    max_model_len: int = 256,
    max_num_scheduled_tokens: int = 10_000,
    max_num_running_reqs: int = 10_000,
    usable_blocks_without_watermark: int = 1_000,
    usable_blocks_with_watermark: int | None = None,
    grid_invariants_digest: str = "a" * 64,
):
    if usable_blocks_with_watermark is None:
        usable_blocks_with_watermark = usable_blocks_without_watermark
    return instrumented_scheduler_module._BenchmarkCapacityEnvelope(
        max_model_len=max_model_len,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        max_num_running_reqs=max_num_running_reqs,
        usable_blocks_without_watermark=usable_blocks_without_watermark,
        usable_blocks_with_watermark=usable_blocks_with_watermark,
        grid_invariants_digest=grid_invariants_digest,
    )


def _install_test_capacity_preflight(stub, capacity=None):
    capacity = capacity or _benchmark_capacity()
    stub._bench_make_local_capacity = lambda: capacity
    stub._bench_synchronizer = None
    # ``_bench_build_grid`` re-filters the decode capture list against the
    # negotiated request limit before generating the grid; stubs that don't
    # model captures still need the attribute to exist.
    if not hasattr(stub, "_bench_decode_capture_sizes"):
        stub._bench_decode_capture_sizes = []


def _make_request(status, num_tokens: int, num_computed_tokens: int = 0):
    """Build a minimal stand-in for ``vllm.v1.request.Request``.

    Only the three attributes read by ``_compute_queued`` are populated.
    """
    return SimpleNamespace(
        status=status,
        num_tokens=num_tokens,
        num_computed_tokens=num_computed_tokens,
    )


def _run_compute_queued(waiting, skipped_waiting):
    """Invoke the real ``InstrumentedScheduler._compute_queued`` on a stub.

    Bypasses ``__init__`` (which needs full vLLM config) and populates only
    the two attributes the method reads.
    """
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub.waiting = waiting
    stub.skipped_waiting = skipped_waiting
    return InstrumentedScheduler._compute_queued(stub)


class _CachedRequestStub:
    def __init__(self, req_ids=None, num_computed_tokens=None, context_phase_ids=None):
        self.req_ids = req_ids or []
        self.num_computed_tokens = num_computed_tokens or []
        self._context_phase_ids = set(context_phase_ids or [])

    def is_context_phase(self, req_id):
        return req_id in self._context_phase_ids


def _make_new_request(req_id: str, prompt_len: int, num_computed_tokens: int):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=[0] * prompt_len,
        num_computed_tokens=num_computed_tokens,
    )


def _run_extract_scheduled(
    new_reqs,
    num_scheduled_tokens,
    *,
    cached=None,
    bench_decode_ids=None,
):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._prompt_len_per_req = {}
    stub._bench_active = bench_decode_ids is not None
    stub._bench_phase = (
        _BenchPhase.DECODE_SWEEP if bench_decode_ids is not None else _BenchPhase.IDLE
    )
    stub._bench_active_req_ids = set(bench_decode_ids or [])
    output = SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached or _CachedRequestStub(),
        num_scheduled_tokens=num_scheduled_tokens,
    )
    return InstrumentedScheduler._extract_scheduled(stub, output)


# ---------------------------------------------------------------------------
# scheduled_requests classification
# ---------------------------------------------------------------------------


def test_extract_scheduled_counts_normal_new_requests_as_prefill():
    metrics = _run_extract_scheduled(
        [_make_new_request("req-1", prompt_len=128, num_computed_tokens=0)],
        {"req-1": 128},
    )

    assert metrics.num_prefill_requests == 1
    assert metrics.sum_prefill_tokens == 128
    assert metrics.sum_prefill_kv_tokens == 0
    assert metrics.num_decode_requests == 0
    assert metrics.sum_decode_kv_tokens == 0


def test_extract_scheduled_reports_prefill_kv_reads():
    metrics = _run_extract_scheduled(
        [_make_new_request("req-1", prompt_len=128, num_computed_tokens=32)],
        {"req-1": 96},
    )

    assert metrics.num_prefill_requests == 1
    assert metrics.sum_prefill_tokens == 96
    assert metrics.sum_prefill_kv_tokens == 32
    assert metrics.num_decode_requests == 0


def test_extract_scheduled_aggregates_batched_prefill_kv_reads():
    metrics = _run_extract_scheduled(
        [
            _make_new_request("req-1", prompt_len=128, num_computed_tokens=32),
            _make_new_request("req-2", prompt_len=128, num_computed_tokens=32),
            _make_new_request("req-3", prompt_len=128, num_computed_tokens=32),
        ],
        {"req-1": 96, "req-2": 96, "req-3": 96},
    )

    assert metrics.num_prefill_requests == 3
    assert metrics.sum_prefill_tokens == 288
    assert metrics.sum_prefill_kv_tokens == 96
    assert metrics.var_prefill_length == 0.0


def test_extract_scheduled_counts_benchmark_decode_new_requests_as_decode():
    metrics = _run_extract_scheduled(
        [
            _make_new_request("__bench_0", prompt_len=17, num_computed_tokens=16),
            _make_new_request("__bench_1", prompt_len=17, num_computed_tokens=16),
        ],
        {"__bench_0": 1, "__bench_1": 1},
        bench_decode_ids={"__bench_0", "__bench_1"},
    )

    assert metrics.num_prefill_requests == 0
    assert metrics.sum_prefill_tokens == 0
    assert metrics.sum_prefill_kv_tokens == 0
    assert metrics.num_decode_requests == 2
    assert metrics.sum_decode_kv_tokens == 32


@pytest.mark.parametrize(
    ("point_type", "num_prefill", "num_decode", "expected"),
    [
        ("prefill", 1, 0, True),
        ("prefill", 0, 1, False),
        ("decode", 0, 1, True),
        ("decode", 1, 0, False),
    ],
)
def test_benchmark_records_only_current_point_forward_pass_type(
    point_type, num_prefill, num_decode, expected
):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_current_point = BenchmarkPoint(point_type=point_type)
    metrics = SimpleNamespace(
        scheduled_requests=SimpleNamespace(
            num_prefill_requests=num_prefill,
            num_decode_requests=num_decode,
        )
    )

    assert InstrumentedScheduler._bench_should_record_fpm(stub, metrics) is expected


def test_benchmark_wall_time_starts_at_post_go_schedule_timestamp():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._last_update_time = 10.0

    assert (
        InstrumentedScheduler._iteration_wall_time(
            stub,
            now=20.0,
            t_sched=19.0,
            is_benchmark_point=True,
        )
        == 1.0
    )
    assert (
        InstrumentedScheduler._iteration_wall_time(
            stub,
            now=20.0,
            t_sched=19.0,
            is_benchmark_point=False,
        )
        == 10.0
    )


def test_benchmark_timing_stops_before_vllm_state_update(monkeypatch):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._schedule_times = deque([10.0])
    stub._last_update_time = 0.0
    stub._bench_active = True
    stub._bench_current_point = BenchmarkPoint(point_type="decode", benchmark_id=1)
    stub._extract_scheduled = MagicMock(
        return_value=instrumented_scheduler_module.ScheduledRequestMetrics(
            num_decode_requests=1
        )
    )
    stub._compute_queued = MagicMock(return_value=None)
    stub._extract_metrics = MagicMock(return_value="metrics")
    stub._publish_or_record_metrics = MagicMock()
    stub._cleanup_finished = MagicMock()
    clock = {"now": 20.0}

    def parent_update(_self, _scheduler_output, _model_runner_output):
        clock["now"] = 30.0
        return "parent-result"

    monkeypatch.setattr(
        instrumented_scheduler_module.AsyncScheduler,
        "update_from_output",
        parent_update,
    )
    monkeypatch.setattr(
        instrumented_scheduler_module.time,
        "monotonic",
        lambda: clock["now"],
    )
    output = SimpleNamespace(total_num_scheduled_tokens=1)

    result = InstrumentedScheduler.update_from_output(stub, output, object())

    assert result == "parent-result"
    assert stub._extract_metrics.call_args.args[2] == 10.0
    assert stub._last_update_time == 20.0


# ---------------------------------------------------------------------------
# self.waiting classification (existing behaviour — regression coverage)
# ---------------------------------------------------------------------------


def test_waiting_new_requests_count_as_queued_prefill():
    q = _run_compute_queued(
        waiting=[
            _make_request(RequestStatus.WAITING, num_tokens=100),
            _make_request(RequestStatus.WAITING, num_tokens=200),
        ],
        skipped_waiting=[],
    )
    assert q.num_prefill_requests == 2
    assert q.sum_prefill_tokens == 300
    assert q.num_decode_requests == 0
    assert q.sum_decode_kv_tokens == 0


def test_waiting_preempted_requests_count_as_queued_decode():
    q = _run_compute_queued(
        waiting=[
            _make_request(
                RequestStatus.PREEMPTED, num_tokens=512, num_computed_tokens=480
            ),
            _make_request(
                RequestStatus.PREEMPTED, num_tokens=256, num_computed_tokens=240
            ),
        ],
        skipped_waiting=[],
    )
    assert q.num_prefill_requests == 0
    assert q.sum_prefill_tokens == 0
    assert q.num_decode_requests == 2
    # sum_decode_kv_tokens = sum of num_computed_tokens
    assert q.sum_decode_kv_tokens == 720


# ---------------------------------------------------------------------------
# self.skipped_waiting classification (the fix)
# ---------------------------------------------------------------------------


def test_skipped_waiting_for_remote_kvs_counts_as_queued_decode():
    """Disagg decode-engine: request has KV being transferred; should count
    as queued decode, not queued prefill.
    """

    q = _run_compute_queued(
        waiting=[],
        skipped_waiting=[
            _make_request(
                RequestStatus.WAITING_FOR_REMOTE_KVS,
                num_tokens=1000,
                num_computed_tokens=1000,
            ),
            _make_request(
                RequestStatus.WAITING_FOR_REMOTE_KVS,
                num_tokens=500,
                num_computed_tokens=500,
            ),
        ],
    )
    # Must NOT be classified as prefill.
    assert q.num_prefill_requests == 0
    assert q.sum_prefill_tokens == 0
    # Classified as decode with KV = num_computed_tokens.
    assert q.num_decode_requests == 2
    assert q.sum_decode_kv_tokens == 1500


def test_skipped_waiting_for_structured_output_counts_as_queued_prefill():
    """Structured-output grammar compile wait has no KV computed yet; prefill."""

    q = _run_compute_queued(
        waiting=[],
        skipped_waiting=[
            _make_request(STRUCTURED_OUTPUT_WAITING_STATUS, num_tokens=128),
        ],
    )
    assert q.num_prefill_requests == 1
    assert q.sum_prefill_tokens == 128
    assert q.num_decode_requests == 0
    assert q.sum_decode_kv_tokens == 0


def test_skipped_waiting_for_streaming_req_counts_as_queued_prefill():
    q = _run_compute_queued(
        waiting=[],
        skipped_waiting=[
            _make_request(RequestStatus.WAITING_FOR_STREAMING_REQ, num_tokens=64),
        ],
    )
    assert q.num_prefill_requests == 1
    assert q.sum_prefill_tokens == 64
    assert q.num_decode_requests == 0


# ---------------------------------------------------------------------------
# Mixed scenarios -- the realistic disagg decode engine picture
# ---------------------------------------------------------------------------


def test_mixed_disagg_decode_engine_snapshot():
    """Realistic decode-engine snapshot: some local preempts in ``self.waiting``
    plus many ``WAITING_FOR_REMOTE_KVS`` in ``self.skipped_waiting``.
    """

    q = _run_compute_queued(
        waiting=[
            _make_request(
                RequestStatus.PREEMPTED, num_tokens=800, num_computed_tokens=780
            ),
        ],
        skipped_waiting=[
            _make_request(
                RequestStatus.WAITING_FOR_REMOTE_KVS,
                num_tokens=1024,
                num_computed_tokens=1024,
            ),
            _make_request(
                RequestStatus.WAITING_FOR_REMOTE_KVS,
                num_tokens=2048,
                num_computed_tokens=2048,
            ),
            _make_request(
                RequestStatus.WAITING_FOR_REMOTE_KVS,
                num_tokens=512,
                num_computed_tokens=512,
            ),
        ],
    )
    # 0 queued prefill on the decode engine under healthy disagg.
    assert q.num_prefill_requests == 0
    assert q.sum_prefill_tokens == 0
    # 1 preempted (local decode evicted) + 3 remote-KV-waiting.
    assert q.num_decode_requests == 4
    assert q.sum_decode_kv_tokens == 780 + 1024 + 2048 + 512


def test_mixed_prefill_engine_snapshot():
    """Prefill engine side: all queued work is prefill across both queues."""

    q = _run_compute_queued(
        waiting=[
            _make_request(RequestStatus.WAITING, num_tokens=100),
            _make_request(RequestStatus.WAITING, num_tokens=200),
        ],
        skipped_waiting=[
            _make_request(STRUCTURED_OUTPUT_WAITING_STATUS, num_tokens=300),
        ],
    )
    assert q.num_prefill_requests == 3
    assert q.sum_prefill_tokens == 600
    assert q.num_decode_requests == 0


def test_empty_queues():
    q = _run_compute_queued(waiting=[], skipped_waiting=[])
    assert q.num_prefill_requests == 0
    assert q.sum_prefill_tokens == 0
    assert q.num_decode_requests == 0
    assert q.sum_decode_kv_tokens == 0
    assert q.var_prefill_length == 0.0
    assert q.var_decode_kv_tokens == 0.0


# ---------------------------------------------------------------------------
# Variance correctness across both queues
# ---------------------------------------------------------------------------


def test_variance_spans_both_queues():
    """Variance is computed over the union of both queues, not each in
    isolation. Using lengths 100 and 300 → mean 200, var 10000 (population).
    """

    q = _run_compute_queued(
        waiting=[
            _make_request(RequestStatus.WAITING, num_tokens=100),
        ],
        skipped_waiting=[
            _make_request(STRUCTURED_OUTPUT_WAITING_STATUS, num_tokens=300),
        ],
    )
    assert q.num_prefill_requests == 2
    assert q.sum_prefill_tokens == 400
    # Population variance of [100, 300] = 10000.
    assert q.var_prefill_length == pytest.approx(10000.0)


def test_dp_rank_prefers_data_parallel_index():
    """External DP + dense model: vLLM resets ``data_parallel_rank`` to 0 in
    every child but keeps ``data_parallel_index`` as the true global rank.
    The resolver must prefer the index so each DP child gets its own port.
    """
    pc = SimpleNamespace(data_parallel_index=1, data_parallel_rank=0)
    assert InstrumentedScheduler._resolve_dp_rank(pc) == 1


def test_benchmark_synchronizer_aligns_point_and_shares_run_id():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    point = BenchmarkPoint(
        point_type="decode",
        benchmark_id=7,
        total_kv_read_tokens=128,
        batch_size=2,
    )
    follower_result = {}
    rank0_fpms = [{"counter_id": 7, "dp_rank": 0, "wall_time": 0.01}]
    rank1_fpms = [{"counter_id": 7, "dp_rank": 1, "wall_time": 0.02}]

    def synchronize_follower():
        follower_result["run_id"] = rank1.synchronize(point)
        follower_result["group"] = rank1.collect_result(point, rank1_fpms)

    follower = threading.Thread(target=synchronize_follower)
    follower.start()
    try:
        coordinator_run_id = rank0.synchronize(point)
        coordinator_group = rank0.collect_result(point, rank0_fpms)
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert follower_result["run_id"] == coordinator_run_id
        assert rank1.run_id == coordinator_run_id
        expected_rank_results = [
            {"dp_rank": 0, "fpms": rank0_fpms},
            {"dp_rank": 1, "fpms": rank1_fpms},
        ]
        assert coordinator_group.rank_results == expected_rank_results
        assert follower_result["group"].rank_results == expected_rank_results
        assert coordinator_group.stop_requested is False
        assert follower_result["group"].stop_requested is False
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_negotiates_minimum_capacity_and_grid():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    rank0_capacity = _benchmark_capacity(
        max_model_len=383_168,
        usable_blocks_without_watermark=23_944,
        usable_blocks_with_watermark=23_940,
    )
    rank1_capacity = _benchmark_capacity(
        max_model_len=351_104,
        max_num_scheduled_tokens=8_192,
        usable_blocks_without_watermark=21_940,
        usable_blocks_with_watermark=21_936,
    )
    follower_result = {}

    def run_follower():
        follower_result["capacity"] = rank1.negotiate_capacity(rank1_capacity)
        rank1.synchronize_grid(
            grid_digest="b" * 64,
            expected_points=1_368,
            missing_phases=[],
        )
        follower_result["grid_synchronized"] = True

    follower = threading.Thread(target=run_follower)
    follower.start()
    try:
        common = rank0.negotiate_capacity(rank0_capacity)
        rank0.synchronize_grid(
            grid_digest="b" * 64,
            expected_points=1_368,
            missing_phases=[],
        )
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert common == follower_result["capacity"]
        assert common.max_model_len == 351_104
        assert common.max_num_scheduled_tokens == 8_192
        assert common.usable_blocks_without_watermark == 21_940
        assert common.usable_blocks_with_watermark == 21_936
        assert follower_result["grid_synchronized"] is True
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_rejects_capacity_invariant_mismatch():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    follower_error = {}

    def run_follower():
        try:
            rank1.negotiate_capacity(
                _benchmark_capacity(grid_invariants_digest="b" * 64)
            )
        except RuntimeError as error:
            follower_error["error"] = error

    follower = threading.Thread(target=run_follower)
    follower.start()
    try:
        with pytest.raises(RuntimeError, match="grid invariants differ"):
            rank0.negotiate_capacity(_benchmark_capacity())
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert "grid invariants differ" in str(follower_error["error"])
    finally:
        rank1.close()
        rank0.close()


def _synchronizer_pair(timeout: float):
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    ranks = []
    for dp_rank in (0, 1):
        ranks.append(
            instrumented_scheduler_module._BenchmarkSynchronizer(
                dp_rank=dp_rank,
                dp_size=2,
                master_ip="unused",
                port=0,
                timeout=timeout,
                endpoint=endpoint,
            )
        )
    return ranks


@pytest.mark.parametrize("late_rank", [0, 1])
def test_benchmark_synchronizer_capacity_phase_outlasts_the_protocol_timeout(
    monkeypatch, late_rank
):
    """A rank reports capacity only once its host-local warm-up probe is done,
    so the ranks' reports can be far apart; the capacity phase absorbs that
    skew, whichever rank is the late one, while the protocol timeout the
    later phases run on stays short."""
    Synchronizer = instrumented_scheduler_module._BenchmarkSynchronizer
    monkeypatch.setattr(Synchronizer, "CAPACITY_TIMEOUT_SECONDS", 5)
    rank0, rank1 = _synchronizer_pair(timeout=0.2)
    assert rank0.timeout_seconds == 0.2
    assert rank0.capacity_timeout_seconds == 5.0
    delay = 0.6  # past the protocol timeout, inside the capacity budget
    result = {}

    def run(dp_rank, synchronizer):
        if dp_rank == late_rank:
            time.sleep(delay)
        result[dp_rank] = synchronizer.negotiate_capacity(_benchmark_capacity())

    follower = threading.Thread(target=run, args=(1, rank1))
    follower.start()
    try:
        run(0, rank0)
        follower.join(timeout=5)
        assert not follower.is_alive()
        assert result[0] == result[1] == _benchmark_capacity()
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_capacity_phase_is_still_bounded(monkeypatch):
    """The capacity budget never drops below the protocol timeout, and a rank
    whose peer never reports still fails at the budget instead of hanging."""
    Synchronizer = instrumented_scheduler_module._BenchmarkSynchronizer
    monkeypatch.setattr(Synchronizer, "CAPACITY_TIMEOUT_SECONDS", 0)
    rank0, rank1 = _synchronizer_pair(timeout=0.2)
    try:
        assert rank0.capacity_timeout_seconds == 0.2
        with pytest.raises(TimeoutError, match="attention-DP ranks"):
            rank0.negotiate_capacity(_benchmark_capacity())
        with pytest.raises(TimeoutError, match="capacity_result"):
            rank1.negotiate_capacity(_benchmark_capacity())
    finally:
        rank1.close()
        rank0.close()


def _digest_stub(max_num_running_reqs: int):
    """Populate only the attributes ``_bench_grid_invariants_digest`` reads,
    mirroring the activation-time filtering of the decode capture sizes."""
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig()
    stub.block_size = 16
    stub._bench_hash_block_size = 16
    stub.cache_config = SimpleNamespace(block_size=16, enable_prefix_caching=True)
    stub.max_num_running_reqs = max_num_running_reqs
    stub._bench_prefill_cudagraph_mode = "PIECEWISE"
    stub._bench_decode_cudagraph_mode = "FULL"
    stub._bench_cudagraph_capture_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    stub._bench_prefill_capture_sizes = list(stub._bench_cudagraph_capture_sizes)
    stub._bench_decode_capture_sizes = [
        size
        for size in stub._bench_cudagraph_capture_sizes
        if size <= max_num_running_reqs
    ]
    return stub


def test_capacity_digest_ignores_request_limit_filtered_capture_sizes():
    """Ranks that differ only in ``max_num_running_reqs`` filter different
    decode capture lists at activation. The invariants digest must hash the
    unfiltered configuration so ``common()`` negotiates the minimum instead
    of rejecting the ranks as structurally different."""
    small = _digest_stub(max_num_running_reqs=128)
    large = _digest_stub(max_num_running_reqs=256)
    assert small._bench_decode_capture_sizes != large._bench_decode_capture_sizes

    small_digest = InstrumentedScheduler._bench_grid_invariants_digest(small)
    large_digest = InstrumentedScheduler._bench_grid_invariants_digest(large)
    assert small_digest == large_digest

    common = instrumented_scheduler_module._BenchmarkCapacityEnvelope.common(
        [
            _benchmark_capacity(
                max_num_running_reqs=128, grid_invariants_digest=small_digest
            ),
            _benchmark_capacity(
                max_num_running_reqs=256, grid_invariants_digest=large_digest
            ),
        ]
    )
    assert common.max_num_running_reqs == 128


def test_benchmark_synchronizer_rejects_grid_mismatch_before_warmup():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    follower_error = {}

    def run_follower():
        try:
            rank1.synchronize_grid(
                grid_digest="b" * 64,
                expected_points=1_368,
                missing_phases=[],
            )
        except RuntimeError as error:
            follower_error["error"] = error

    follower = threading.Thread(target=run_follower)
    follower.start()
    try:
        with pytest.raises(RuntimeError, match="grid mismatch"):
            rank0.synchronize_grid(
                grid_digest="a" * 64,
                expected_points=1_368,
                missing_phases=[],
            )
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert "grid mismatch" in str(follower_error["error"])
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_shares_timeout_stop_decision():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    point = BenchmarkPoint(point_type="decode", benchmark_id=1)
    follower_result = {}

    def run_follower():
        rank1.synchronize(point)
        follower_result["group"] = rank1.collect_result(
            point,
            [{"counter_id": 1, "dp_rank": 1}],
            stop_deadline_monotonic=(
                instrumented_scheduler_module.time.monotonic() - 1
            ),
        )

    follower = threading.Thread(target=run_follower)
    follower.start()
    try:
        rank0.synchronize(point)
        coordinator_group = rank0.collect_result(
            point,
            [{"counter_id": 1, "dp_rank": 0}],
        )
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert coordinator_group.stop_requested is True
        assert follower_result["group"].stop_requested is True
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_coordinates_boundary_and_cleanup():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    follower_result = {}

    def run_follower():
        follower_result["stop"] = rank1.synchronize_boundary(
            2,
            False,
            stop_deadline_monotonic=(
                instrumented_scheduler_module.time.monotonic() - 1
            ),
        )
        rank1.synchronize_cleanup()
        follower_result["cleaned"] = True

    follower = threading.Thread(target=run_follower)
    follower.start()
    try:
        assert rank0.synchronize_boundary(2, False) is True
        rank0.synchronize_cleanup()
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert follower_result == {"stop": True, "cleaned": True}
        assert rank0._cleanup_complete is True
        assert rank1._cleanup_complete is True
    finally:
        rank1.close()
        rank0.close()


def _stage_pair(timeout=1):
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=timeout,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=timeout,
        endpoint=endpoint,
    )
    return rank0, rank1


def _stage_verdict(synchronizer, budget=2.0):
    """Drive ``stage_poll`` the way the scheduler does: one non-blocking poll
    per idle step until the verdict arrives."""
    end = time.monotonic() + budget
    while time.monotonic() < end:
        verdict = synchronizer.stage_poll()
        if verdict is not None:
            return verdict
        time.sleep(0.005)
    raise AssertionError("no stage verdict within budget")


def test_benchmark_synchronizer_stage_exchange_agrees_when_every_rank_is_ok():
    rank0, rank1 = _stage_pair()
    follower_result = {}

    def follow():
        rank1.stage_report(8, True)
        follower_result["verdict"] = _stage_verdict(rank1)

    follower = threading.Thread(target=follow)
    follower.start()
    try:
        rank0.stage_report(8, True)
        # Non-blocking: rank 0 keeps polling between idle steps.
        assert _stage_verdict(rank0) is True
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert follower_result["verdict"] is True
        # The exchange is closed on both sides once the verdict is out.
        for synchronizer in (rank0, rank1):
            with pytest.raises(RuntimeError, match="without a report"):
                synchronizer.stage_poll()
    finally:
        rank1.close()
        rank0.close()


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_benchmark_synchronizer_stage_exchange_fails_the_group_with_one_rank(
    failing_rank,
):
    rank0, rank1 = _stage_pair()
    follower_result = {}

    def follow():
        rank1.stage_report(16, failing_rank != 1)
        follower_result["verdict"] = _stage_verdict(rank1)

    follower = threading.Thread(target=follow)
    follower.start()
    try:
        rank0.stage_report(16, failing_rank != 0)
        assert _stage_verdict(rank0) is False
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert follower_result["verdict"] is False
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_stage_exchange_times_out_without_follower_report():
    rank0, rank1 = _stage_pair(timeout=0.05)
    try:
        rank0.stage_report(8, True, timeout=0.05)
        with pytest.raises(TimeoutError, match="stage reports.*batch=8"):
            _stage_verdict(rank0)
        # A second report is possible again (the failed exchange is closed),
        # and the late follower learns about the failure instead of waiting.
        rank1.stage_report(8, True)
        with pytest.raises(RuntimeError, match="synchronization failed"):
            _stage_verdict(rank1)
        with pytest.raises(RuntimeError, match="already pending"):
            rank0.stage_report(8, True)
            rank0.stage_report(8, True)
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_stage_exchange_rejects_a_rung_mismatch():
    rank0, rank1 = _stage_pair()
    try:
        rank1.stage_report(4, True)
        rank0.stage_report(8, True)
        with pytest.raises(RuntimeError, match="invalid attention-DP warm-up stage"):
            _stage_verdict(rank0)
        with pytest.raises(RuntimeError, match="synchronization failed"):
            _stage_verdict(rank1)
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_close_flushes_after_cleanup():
    synchronizer = instrumented_scheduler_module._BenchmarkSynchronizer.__new__(
        instrumented_scheduler_module._BenchmarkSynchronizer
    )
    synchronizer._socket = MagicMock()
    synchronizer._timeout_ms = 1_000
    synchronizer._cleanup_complete = False
    synchronizer._flush_on_close = False

    synchronizer.close()
    synchronizer._socket.close.assert_called_once_with(linger=0)

    synchronizer._socket.close.reset_mock()
    synchronizer._cleanup_complete = True
    synchronizer.close()
    synchronizer._socket.close.assert_called_once_with(linger=2_000)


def test_benchmark_synchronizer_commits_before_fast_rank_advances():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    synchronizers = [
        instrumented_scheduler_module._BenchmarkSynchronizer(
            dp_rank=rank,
            dp_size=3,
            master_ip="unused",
            port=0,
            timeout=1,
            endpoint=endpoint,
        )
        for rank in range(3)
    ]
    rank0, rank1, rank2 = synchronizers
    point = BenchmarkPoint(point_type="decode", benchmark_id=1)
    original_rank2_recv = rank2._recv_follower

    def delay_rank2_group_ack(deadline, benchmark_id, expected_type):
        reply = original_rank2_recv(deadline, benchmark_id, expected_type)
        if expected_type == "group":
            instrumented_scheduler_module.time.sleep(0.05)
        return reply

    rank2._recv_follower = delay_rank2_group_ack
    follower_errors = []

    def run_follower(synchronizer, rank):
        try:
            synchronizer.synchronize(point)
            synchronizer.collect_result(
                point,
                [{"counter_id": 1, "dp_rank": rank}],
            )
            assert synchronizer.synchronize_boundary(2, False) is False
            synchronizer.synchronize_cleanup()
        except Exception as error:  # pragma: no cover - asserted below
            follower_errors.append(error)

    followers = [
        threading.Thread(target=run_follower, args=(rank1, 1)),
        threading.Thread(target=run_follower, args=(rank2, 2)),
    ]
    for follower in followers:
        follower.start()
    try:
        rank0.synchronize(point)
        rank0.collect_result(point, [{"counter_id": 1, "dp_rank": 0}])
        assert rank0.synchronize_boundary(2, False) is False
        rank0.synchronize_cleanup()
        for follower in followers:
            follower.join(timeout=2)
            assert not follower.is_alive()
        assert follower_errors == []
    finally:
        for synchronizer in reversed(synchronizers):
            synchronizer.close()


def test_benchmark_synchronizer_rejects_different_points():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    coordinator_point = BenchmarkPoint(point_type="decode", benchmark_id=1)
    follower_point = BenchmarkPoint(
        point_type="decode", benchmark_id=1, total_kv_read_tokens=16
    )
    follower_error = {}

    def synchronize_follower():
        try:
            rank1.synchronize(follower_point)
        except RuntimeError as error:
            follower_error["error"] = error

    follower = threading.Thread(target=synchronize_follower)
    follower.start()
    try:
        with pytest.raises(RuntimeError, match="point mismatch"):
            rank0.synchronize(coordinator_point)
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert "point mismatch" in str(follower_error["error"])
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_propagates_rank_abort():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=1,
        endpoint=endpoint,
    )
    point = BenchmarkPoint(point_type="decode", benchmark_id=1)

    def abort_follower():
        rank1.synchronize(point)
        rank1.abort("decode[0]: output update failed")

    follower = threading.Thread(target=abort_follower)
    follower.start()
    try:
        rank0.synchronize(point)
        with pytest.raises(RuntimeError, match=r"rank 1 aborted.*decode\[0\]"):
            rank0.collect_result(point, [{"counter_id": 1, "dp_rank": 0}])
        follower.join(timeout=2)
        assert not follower.is_alive()
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_does_not_release_stale_ready_rank():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=0.05,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=0.05,
        endpoint=endpoint,
    )
    point = BenchmarkPoint(point_type="decode", benchmark_id=1)
    try:
        with pytest.raises(TimeoutError, match="prepare"):
            rank1.synchronize(point)
        rank1.close()

        with pytest.raises((TimeoutError, instrumented_scheduler_module.zmq.ZMQError)):
            rank0.synchronize(point)
    finally:
        rank1.close()
        rank0.close()


def test_benchmark_synchronizer_gives_armed_rank_time_to_receive_go():
    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=0.05,
        endpoint=endpoint,
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1,
        dp_size=2,
        master_ip="unused",
        port=0,
        timeout=0.05,
        endpoint=endpoint,
    )
    point = BenchmarkPoint(point_type="decode", benchmark_id=1)
    original_coordinate_phase = rank0._coordinate_phase

    def delayed_coordinate_phase(*args, **kwargs):
        original_coordinate_phase(*args, **kwargs)
        if kwargs["expected_type"] == "armed":
            instrumented_scheduler_module.time.sleep(0.075)

    rank0._coordinate_phase = delayed_coordinate_phase
    follower_result = {}

    def synchronize_follower():
        follower_result["run_id"] = rank1.synchronize(point)

    follower = threading.Thread(target=synchronize_follower)
    follower.start()
    try:
        coordinator_run_id = rank0.synchronize(point)
        follower.join(timeout=2)
        assert not follower.is_alive()
        assert follower_result["run_id"] == coordinator_run_id
    finally:
        rank1.close()
        rank0.close()


def test_fpm_publisher_drops_benchmark_metrics_until_resumed():
    publisher = instrumented_scheduler_module._FpmPublisherThread.__new__(
        instrumented_scheduler_module._FpmPublisherThread
    )
    publisher._running = True
    publisher._publishing = threading.Event()
    publisher._queue = instrumented_scheduler_module.queue.Queue()
    metrics = instrumented_scheduler_module.ForwardPassMetrics()

    publisher.publish(metrics)
    assert publisher._queue.empty()

    publisher.resume()
    publisher.publish(metrics)
    assert publisher._queue.get_nowait() is metrics


def test_benchmark_fpm_uses_benchmark_id_and_is_not_published():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_active = True
    stub._bench_current_point = BenchmarkPoint(point_type="decode", benchmark_id=7)
    stub._bench_current_fpms = []
    stub._publisher = MagicMock()
    metrics = instrumented_scheduler_module.ForwardPassMetrics(
        dp_rank=1,
        scheduled_requests=instrumented_scheduler_module.ScheduledRequestMetrics(
            num_decode_requests=2,
            sum_decode_kv_tokens=128,
        ),
    )

    InstrumentedScheduler._publish_or_record_metrics(stub, metrics)

    assert stub._bench_current_fpms[0]["counter_id"] == 7
    assert stub._bench_current_fpms[0]["dp_rank"] == 1
    stub._publisher.publish.assert_not_called()


def test_live_fpm_publishing_resumes_with_publisher_owned_counter():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_active = False
    stub._publisher = MagicMock()
    metrics = instrumented_scheduler_module.ForwardPassMetrics(counter_id=0)

    InstrumentedScheduler._publish_or_record_metrics(stub, metrics)

    stub._publisher.publish.assert_called_once_with(metrics)


def test_benchmark_output_summary_must_match_point_before_go():
    point = BenchmarkPoint(
        point_type="decode",
        benchmark_id=3,
        total_kv_read_tokens=128,
        batch_size=2,
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    # The synchronized output is the admission step, which runs one token
    # short per request (the steady step reads the full 128 afterwards).
    stub._bench_admission_kv_tokens = 126
    matching = {
        "total_num_scheduled_tokens": 2,
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 2,
        "sum_decode_kv_tokens": 126,
    }

    assert (
        InstrumentedScheduler._bench_output_validation_error(stub, point, matching)
        is None
    )

    mismatched = dict(matching, num_decode_requests=1)
    error = InstrumentedScheduler._bench_output_validation_error(
        stub, point, mismatched
    )
    assert "benchmark_id=3 SchedulerOutput does not match" in error

    full_context = dict(matching, sum_decode_kv_tokens=128)
    error = InstrumentedScheduler._bench_output_validation_error(
        stub, point, full_context
    )
    assert error is not None, "the admission step must run one token short"


def test_dp_rank_falls_back_to_rank_when_index_absent():
    pc = SimpleNamespace(data_parallel_rank=2)
    assert InstrumentedScheduler._resolve_dp_rank(pc) == 2


def test_dp_rank_handles_none_rank():
    pc = SimpleNamespace(data_parallel_index=None, data_parallel_rank=None)
    assert InstrumentedScheduler._resolve_dp_rank(pc) == 0


def test_dp_rank_default_zero():
    pc = SimpleNamespace()
    assert InstrumentedScheduler._resolve_dp_rank(pc) == 0


def test_dp_rank_multi_node_start_offset():
    """Multi-node: node 2 runs DP ranks 8..15 with ``--data-parallel-start-rank 8``.
    vLLM spawns each child engine with ``dp_rank = start_rank + local_index``
    (``vllm/v1/engine/utils.py``: ``global_index = start_index + index``) and
    sets ``parallel_config.data_parallel_index = dp_rank`` (``vllm/v1/engine/
    core.py``). The resolver must return the global rank so each child's ZMQ
    port offset matches the parent-side FPM relay subscription, which iterates
    the same global range.
    """
    for global_rank in (8, 9, 15):
        pc = SimpleNamespace(data_parallel_index=global_rank, data_parallel_rank=0)
        assert InstrumentedScheduler._resolve_dp_rank(pc) == global_rank


def test_decode_variance_spans_both_queues():
    """Decode variance mixes local-preempted (``self.waiting``) and
    remote-KV-waiting (``self.skipped_waiting``) into one accumulator.
    KV lengths 500 and 1500 → mean 1000, population variance 250000.
    """

    q = _run_compute_queued(
        waiting=[
            _make_request(
                RequestStatus.PREEMPTED, num_tokens=520, num_computed_tokens=500
            ),
        ],
        skipped_waiting=[
            _make_request(
                RequestStatus.WAITING_FOR_REMOTE_KVS,
                num_tokens=1500,
                num_computed_tokens=1500,
            ),
        ],
    )
    assert q.num_decode_requests == 2
    assert q.sum_decode_kv_tokens == 2000
    assert q.var_decode_kv_tokens == pytest.approx(250000.0)


# ---------------------------------------------------------------------------
# kv_connector_metadata population on benchmark-built SchedulerOutputs
# ---------------------------------------------------------------------------
#
# When a KV connector is configured (e.g. NixlConnector for disagg),
# vLLM's worker-side ``_get_kv_connector_output`` asserts
# ``scheduler_output.kv_connector_metadata is not None`` before calling
# ``bind_connector_metadata``. The parent ``Scheduler.schedule()``
# satisfies that contract by calling ``connector.build_connector_meta(...)``
# on every SchedulerOutput it produces.
#
# ``InstrumentedScheduler`` builds two SchedulerOutputs from scratch
# during ``DYN_BENCHMARK_MODE=decode``:
#
#   1. The synthetic decode batch in ``_bench_inject_fake_decode``.
#   2. The empty drain frame in ``schedule()`` between decode points.
#
# Both must mirror the parent's connector hook or EngineCore dies with
# ``AssertionError`` on the first iteration of the decode sweep.
# (Repro: launching a vLLM disagg decode worker with
# ``--kv-transfer-config '{"kv_connector":"NixlConnector",...}'`` and
# ``DYN_BENCHMARK_MODE=decode`` -- assertion fires before the worker
# can register and the planner never receives ``get_perf_metrics``.)


def _make_decode_sweep_stub(connector, ec_connector=None):
    """Build the minimal stub needed to drive ``schedule()`` into the
    DECODE_SWEEP empty-frame branch without spinning up the parent
    scheduler's vLLM-side state.
    """
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_active = True
    stub._bench_phase = _BenchPhase.DECODE_SWEEP
    stub._bench_active_req_ids = {"__bench_0"}
    stub.kv_cache_manager = MagicMock()
    stub.kv_cache_manager.num_kv_cache_groups = 1
    stub.finished_req_ids = set()
    stub.connector = connector
    stub.ec_connector = ec_connector
    stub._update_after_schedule = MagicMock()
    # Force the empty-frame branch: ``_bench_step`` returns None, drain
    # path is selected because there are active req IDs.
    stub._bench_step = MagicMock(return_value=None)
    # Defensive: if the empty-frame branch isn't taken the test would
    # otherwise fall through to ``_schedule_and_record_time`` which
    # touches real parent state.
    stub._schedule_and_record_time = MagicMock(
        side_effect=AssertionError("empty-frame branch should have returned")
    )
    return stub


def test_decode_sweep_empty_frame_attaches_kv_connector_metadata():
    """Parent's ``build_connector_meta`` must be called on the empty drain
    frame; metadata is then attached to the returned SchedulerOutput.
    """
    sentinel = object()
    connector = MagicMock()
    connector.build_connector_meta = MagicMock(return_value=sentinel)

    stub = _make_decode_sweep_stub(connector=connector)
    out = InstrumentedScheduler.schedule(stub)

    assert out.kv_connector_metadata is sentinel
    connector.build_connector_meta.assert_called_once_with(out)
    # ec_connector is None on the stub; the ec field stays untouched.
    assert out.ec_connector_metadata is None


def test_decode_sweep_empty_frame_attaches_ec_connector_metadata_when_set():
    kv_meta = object()
    ec_meta = object()
    connector = MagicMock()
    connector.build_connector_meta = MagicMock(return_value=kv_meta)
    ec_connector = MagicMock()
    ec_connector.build_connector_meta = MagicMock(return_value=ec_meta)

    stub = _make_decode_sweep_stub(connector=connector, ec_connector=ec_connector)
    out = InstrumentedScheduler.schedule(stub)

    assert out.kv_connector_metadata is kv_meta
    assert out.ec_connector_metadata is ec_meta
    connector.build_connector_meta.assert_called_once_with(out)
    ec_connector.build_connector_meta.assert_called_once_with(out)


def test_decode_sweep_empty_frame_no_connector_leaves_metadata_none():
    """No connector configured (aggregated worker without
    --kv-transfer-config): the empty frame is returned with both
    metadata fields still None -- exercising the ``getattr(..., None)``
    guard in the fix.
    """
    stub = _make_decode_sweep_stub(connector=None)
    out = InstrumentedScheduler.schedule(stub)

    assert out.kv_connector_metadata is None
    assert out.ec_connector_metadata is None


# ---------------------------------------------------------------------------
# Prompt padding in _bench_inject_fake_decode (batch>1 OOB regression)
# ---------------------------------------------------------------------------
#
# vLLM's worker (gpu_model_runner._update_states_after_model_execute) writes
# a ``-1`` placeholder into ``token_ids_cpu[req_idx, num_tokens_no_spec]``
# after every async-scheduling sample, where ``num_tokens_no_spec`` equals
# the request's prompt length. If the synthetic decode prompt is exactly
# ``ctx_len`` long, the placeholder lands at position ``ctx_len`` -- the
# exact slot the next decode iteration's request reads as its input token
# when the InputBatch slot gets reused. The embedding lookup OOBs because
# -1 is out of vocab.
#
# Padding the synthetic prompt by +1 keeps the placeholder write at
# ``ctx_len + 1`` (out of the read range) and leaves position ``ctx_len``
# as a valid token id (0).


def test_bench_inject_fake_decode_pads_prompt_for_async_placeholder():
    """The injected NewRequestData must carry ``ctx_len + 1`` prompt tokens
    (not ``ctx_len``) and ``num_computed_tokens == ctx_len`` so the worker
    reads input at position ``ctx_len`` from a guaranteed-zero prompt slot.

    Bypasses ``Request`` construction by short-circuiting allocate_slots
    on the first iteration -- the function still builds and returns the
    SchedulerOutput when the batch was empty due to KV exhaustion.
    """
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_seq = 0
    stub._bench_active_req_ids = set()
    stub.requests = {}
    stub.running = []
    stub.finished_req_ids = set()
    stub._bench_block_hasher = None
    stub.kv_cache_manager = MagicMock()
    stub.kv_cache_manager.num_kv_cache_groups = 1
    stub.kv_cache_manager.take_new_block_ids = MagicMock(return_value=None)
    stub.connector = None
    stub.ec_connector = None

    captured_num_new_tokens: list[int] = []

    def _allocate_slots(req, num_new_tokens, **kwargs):
        captured_num_new_tokens.append(num_new_tokens)
        return None  # short-circuit the loop body before NewRequestData append

    stub.kv_cache_manager.allocate_slots = _allocate_slots

    InstrumentedScheduler._bench_inject_fake_decode(stub, context_lengths=[16])

    # Critical regression assertion: the +1 padding is applied.
    assert captured_num_new_tokens == [17], (
        f"Expected allocate_slots(req, ctx_len + 1 = 17, ...) to leave room "
        f"for the async-scheduler placeholder write at position ctx_len. "
        f"Got num_new_tokens={captured_num_new_tokens}."
    )


# ---------------------------------------------------------------------------
# Decode-grid sizing must account for the +1-padded allocation
# ---------------------------------------------------------------------------
#
# ``_bench_inject_fake_decode`` allocates ``ctx_len + 1`` tokens per request
# (rounded UP to the next block boundary by the KV cache manager). If
# ``_bench_generate_decode_grid`` keeps sizing ``max_batch`` from a raw
# ``ctx_len`` token count it will under-count blocks per request and the
# allocator will silently truncate the batch on boundary points
# (``KV exhausted at ctx_len=...``). The benchmark would then record the
# point under the wrong (over-stated) batch size.


def _grid_stub_with_kv_capacity(num_gpu_blocks: int, block_size: int):
    """Bypass ``__init__`` and populate only the attributes
    ``_bench_generate_decode_grid`` reads."""
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_grid = []
    stub._bench_config = BenchmarkConfig()
    stub.cache_config = SimpleNamespace(
        num_gpu_blocks=num_gpu_blocks,
        enable_prefix_caching=True,
    )
    stub.block_size = block_size
    stub.max_model_len = 256
    stub.max_num_scheduled_tokens = 10_000
    # Generous so the KV cap (not max_num_running_reqs) drives the boundary.
    stub.max_num_running_reqs = 10_000
    stub._bench_decode_capture_sizes = [8]
    stub._bench_decode_cudagraph_mode = "FULL"
    stub._bench_feasible_max_decode_batch_size = 0
    return stub


def test_decode_grid_sizes_max_batch_from_padded_allocation():
    """Each emitted decode point's ``batch_size`` must be feasible at the
    actual per-request allocation size of
    ``ceil((ctx_len + 1) / block_size)`` blocks. A regression that sized
    the cap from raw ``ctx_len`` would emit batches that the allocator
    truncates -- e.g. ctx_len=block_size yields 2 blocks/req, but the
    old code would advertise ``num_gpu_blocks // 1`` requests.
    """
    block_size = 16
    num_gpu_blocks = 64
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks, block_size)

    InstrumentedScheduler._bench_generate_decode_grid(stub)

    assert len(stub._bench_grid) > 0, "decode grid should produce points"
    for point in stub._bench_grid:
        assert InstrumentedScheduler._bench_decode_point_feasible(
            stub, point.batch_size, point.total_kv_read_tokens
        ), (
            f"point total_kv={point.total_kv_read_tokens} "
            f"batch_size={point.batch_size} exceeds live KV capacity"
        )
    assert stub._bench_feasible_max_decode_batch_size == num_gpu_blocks - 1
    batch_sizes = {point.batch_size for point in stub._bench_grid}
    assert {8, 9, 16, 32, num_gpu_blocks - 1}.issubset(batch_sizes)


def test_decode_grid_first_ctx_yields_block_aligned_capacity():
    """At ``ctx_len == block_size`` the per-request allocation is exactly
    2 blocks (16 prompt + 1 placeholder = 17 tokens, rounded up). The
    grid's largest batch for this ctx must respect that.
    """
    block_size = 16
    num_gpu_blocks = 100
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks, block_size)

    # One of the 100 blocks is the null sentinel, leaving 99 // 2 == 49.
    assert InstrumentedScheduler._bench_decode_point_feasible(stub, 49, 49 * block_size)
    assert not InstrumentedScheduler._bench_decode_point_feasible(
        stub, 50, 50 * block_size
    )


def test_decode_grid_reserves_padding_and_sample_under_model_length():
    """The fake request adds one prompt-padding token and samples one token."""
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=64, block_size=16)
    stub.max_model_len = 8

    assert InstrumentedScheduler._bench_decode_point_feasible(stub, 1, 6)
    assert not InstrumentedScheduler._bench_decode_point_feasible(stub, 1, 7)
    assert InstrumentedScheduler._bench_max_decode_kv_read_tokens(stub, 1) == 6


def test_decode_grid_accounts_for_all_hybrid_kv_cache_groups():
    """Every KV-cache group allocates from the same physical block pool."""
    block_size = 16
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=100, block_size=block_size)
    stub.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=[
                SimpleNamespace(block_size=16),
                SimpleNamespace(block_size=32),
            ]
        )
    )

    # A 17-token allocation uses 2 + 1 blocks across the two groups.
    assert InstrumentedScheduler._bench_decode_point_feasible(stub, 33, 33 * block_size)
    assert not InstrumentedScheduler._bench_decode_point_feasible(
        stub, 34, 34 * block_size
    )


def test_decode_block_footprint_excludes_cross_attention_groups():
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=100, block_size=16)
    cross_attention_manager = object.__new__(
        instrumented_scheduler_module.CrossAttentionManager
    )
    cross_attention_manager.block_size = 16
    stub.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=[
                SimpleNamespace(block_size=16),
                cross_attention_manager,
            ]
        )
    )

    # The 17 decoder tokens consume two self-attention blocks and zero
    # cross-attention blocks because the synthetic request has no encoder input.
    assert InstrumentedScheduler._bench_blocks_per_req(stub, 17) == 2


def test_decode_grid_leaves_kv_cache_watermark_free():
    block_size = 16
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=100, block_size=block_size)
    stub.kv_cache_manager = SimpleNamespace(watermark_blocks=3)

    # 100 total - 1 null - 3 watermark leaves 96 blocks, or 48 requests.
    assert InstrumentedScheduler._bench_decode_point_feasible(stub, 48, 48 * block_size)
    assert not InstrumentedScheduler._bench_decode_point_feasible(
        stub, 49, 49 * block_size
    )


def test_decode_grid_uses_live_free_block_count_after_manager_reservations():
    block_size = 16
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=100, block_size=block_size)
    stub.kv_cache_manager = SimpleNamespace(
        watermark_blocks=3,
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 90),
    )

    # The pool has already removed null/sink reservations: (90 - 3) // 2.
    assert InstrumentedScheduler._bench_decode_point_feasible(stub, 43, 43 * block_size)
    assert not InstrumentedScheduler._bench_decode_point_feasible(
        stub, 44, 44 * block_size
    )


def test_decode_grid_uses_common_attention_dp_capacity():
    # Regression for a DEP4 GLM-5.2 run where vLLM auto-fit two ranks to
    # 5,987 blocks / 383,168 tokens and two ranks to 5,486 blocks / 351,104
    # tokens. Independent grids had the same count but diverged at point 20:
    # 383,103 versus 351,039 total KV-read tokens.
    larger_rank = _grid_stub_with_kv_capacity(num_gpu_blocks=5_987, block_size=64)
    smaller_rank = _grid_stub_with_kv_capacity(num_gpu_blocks=5_486, block_size=64)
    larger_rank.max_model_len = 383_168
    smaller_rank.max_model_len = 351_104
    capture_sizes = [
        1,
        2,
        4,
        *range(8, 257, 8),
        *range(272, 513, 16),
    ]
    for stub in (larger_rank, smaller_rank):
        stub.max_num_scheduled_tokens = 8_192
        stub.max_num_running_reqs = 1_024
        stub._bench_decode_capture_sizes = capture_sizes
    common = _benchmark_capacity(
        max_model_len=351_104,
        max_num_scheduled_tokens=8_192,
        max_num_running_reqs=1_024,
        usable_blocks_without_watermark=5_485,
        usable_blocks_with_watermark=5_485,
    )
    larger_rank._bench_negotiated_capacity = common
    smaller_rank._bench_negotiated_capacity = common

    InstrumentedScheduler._bench_generate_decode_grid(larger_rank)
    InstrumentedScheduler._bench_generate_decode_grid(smaller_rank)

    larger_grid = [point.__dict__ for point in larger_rank._bench_grid]
    smaller_grid = [point.__dict__ for point in smaller_rank._bench_grid]
    assert larger_grid == smaller_grid
    # Steady-coordinate normalization merges each batch's sub-2B presets into
    # one point, so batch=1 keeps 19 ladder entries and its feasibility
    # boundary sits at index 18.
    assert len(larger_grid) == 1_266
    assert larger_rank._bench_grid[18].total_kv_read_tokens == 351_039

    # The common grid must remain feasible under each rank's original local
    # capacity when the conservative shared envelope is removed.
    for stub in (larger_rank, smaller_rank):
        stub._bench_negotiated_capacity = None
        assert all(
            InstrumentedScheduler._bench_decode_point_feasible(
                stub, point.batch_size, point.total_kv_read_tokens
            )
            for point in stub._bench_grid
        )


@pytest.mark.parametrize(
    ("mode", "prefill_points", "decode_points", "expected_missing_phases"),
    [
        ("prefill", 0, 0, ["prefill"]),
        ("decode", 0, 0, ["decode"]),
        ("agg", 0, 0, ["prefill", "decode"]),
        ("agg", 1, 0, ["decode"]),
        ("agg", 0, 1, ["prefill"]),
        ("agg", 1, 1, []),
    ],
)
def test_benchmark_grid_tracks_each_requested_empty_phase(
    mode, prefill_points, decode_points, expected_missing_phases
):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig(mode=mode)
    stub._bench_explicit_points = None
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []
    _install_test_capacity_preflight(stub)

    def generate_prefill_grid():
        stub._bench_grid.extend(
            BenchmarkPoint(point_type="prefill") for _ in range(prefill_points)
        )

    def generate_decode_grid():
        stub._bench_grid.extend(
            BenchmarkPoint(point_type="decode") for _ in range(decode_points)
        )

    stub._bench_generate_prefill_grid = generate_prefill_grid
    stub._bench_generate_decode_grid = generate_decode_grid

    InstrumentedScheduler._bench_build_grid(stub)

    assert stub._bench_expected_points == prefill_points + decode_points
    assert stub._bench_missing_phases == expected_missing_phases


def test_benchmark_grid_has_no_point_cap():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig(mode="prefill")
    stub._bench_explicit_points = None
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []
    stub._bench_grid_error = None
    _install_test_capacity_preflight(stub)

    def generate_prefill_grid():
        stub._bench_grid.extend(
            BenchmarkPoint(point_type="prefill", total_prefill_tokens=index)
            for index in range(1, 4098)
        )

    stub._bench_generate_prefill_grid = generate_prefill_grid

    InstrumentedScheduler._bench_build_grid(stub)

    assert stub._bench_expected_points == 4097
    real_points = [
        point
        for point in stub._bench_grid
        if EAGER_WARMUP_REASON not in point.sample_reasons
    ]
    assert len(real_points) == 4097
    assert [point.benchmark_id for point in real_points] == list(range(1, 4098))
    assert stub._bench_grid_error is None


def test_benchmark_grid_assigns_stable_contiguous_ids_and_digest():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig(mode="prefill")
    stub._bench_explicit_points = None
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []
    stub._bench_grid_error = None
    _install_test_capacity_preflight(stub)

    def generate_prefill_grid():
        stub._bench_grid.extend(
            [
                BenchmarkPoint(point_type="prefill", total_prefill_tokens=8),
                BenchmarkPoint(point_type="prefill", total_prefill_tokens=16),
            ]
        )

    stub._bench_generate_prefill_grid = generate_prefill_grid

    InstrumentedScheduler._bench_build_grid(stub)

    real_points = [
        point
        for point in stub._bench_grid
        if EAGER_WARMUP_REASON not in point.sample_reasons
    ]
    warmup_points = [
        point
        for point in stub._bench_grid
        if EAGER_WARMUP_REASON in point.sample_reasons
    ]
    # Real points own the contiguous 1..N range (native-artifact contract);
    # discarded eager-warmup replicas take IDs after the real range even
    # though they execute first.
    assert [point.benchmark_id for point in real_points] == [1, 2]
    assert [point.benchmark_id for point in warmup_points] == [3, 4]
    assert len(stub._bench_grid_digest) == 64


def _explicit_grid_stub(mode="agg", points=None):
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=64, block_size=8)
    stub._bench_config = BenchmarkConfig(mode=mode)
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []
    stub._bench_hash_block_size = 8
    stub._bench_prefill_capture_sizes = [8, 16]
    stub._bench_prefill_cudagraph_mode = "PIECEWISE"
    stub.num_lookahead_tokens = 0
    stub._bench_skipped_points = []
    stub._bench_explicit_points = BenchmarkPoints.model_validate(
        points
        or {
            "schema_version": 1,
            "prefill": [
                {
                    "total_prefill_tokens": 8,
                    "total_kv_read_tokens": 0,
                    "batch_size": 1,
                }
            ],
            "decode": [{"total_kv_read_tokens": 16, "batch_size": 1}],
        }
    )
    _install_test_capacity_preflight(stub)
    return stub


@pytest.mark.parametrize(
    ("mode", "expected_points"),
    [
        ("prefill", [("prefill", 8, 0, 1)]),
        ("decode", [("decode", 0, 16, 1)]),
        ("agg", [("prefill", 8, 0, 1), ("decode", 0, 16, 1)]),
    ],
)
def test_explicit_points_replace_generated_grid(mode, expected_points):
    stub = _explicit_grid_stub(mode)

    InstrumentedScheduler._bench_build_grid(stub)

    assert [
        (
            point.point_type,
            point.total_prefill_tokens,
            point.total_kv_read_tokens,
            point.batch_size,
        )
        for point in stub._bench_grid
    ] == expected_points
    assert [point.benchmark_id for point in stub._bench_grid] == list(
        range(1, len(expected_points) + 1)
    )
    assert all("explicit" in point.sample_reasons for point in stub._bench_grid)


def test_empty_explicit_points_are_a_noop():
    stub = _explicit_grid_stub(
        "agg",
        {"schema_version": 1, "prefill": [], "decode": []},
    )

    InstrumentedScheduler._bench_build_grid(stub)

    assert list(stub._bench_grid) == []
    assert stub._bench_expected_points == 0
    assert stub._bench_missing_phases == []


def test_explicit_infeasible_point_reports_source_index():
    stub = _explicit_grid_stub(
        "prefill",
        {
            "schema_version": 1,
            "prefill": [
                {
                    "total_prefill_tokens": 10_001,
                    "total_kv_read_tokens": 0,
                    "batch_size": 1,
                }
            ],
            "decode": [],
        },
    )

    with pytest.raises(ValueError, match=r"prefill\[0\].*infeasible"):
        InstrumentedScheduler._bench_build_grid(stub)


def test_explicit_decode_respects_scheduled_token_limit():
    stub = _explicit_grid_stub(
        "decode",
        {
            "schema_version": 1,
            "prefill": [],
            "decode": [{"total_kv_read_tokens": 2, "batch_size": 2}],
        },
    )
    # The limit is read through the negotiated capacity envelope, so the
    # constraint must be installed there rather than on the stub attribute.
    _install_test_capacity_preflight(
        stub, _benchmark_capacity(max_num_scheduled_tokens=1)
    )

    with pytest.raises(ValueError, match=r"decode\[0\].*infeasible"):
        InstrumentedScheduler._bench_build_grid(stub)


def test_explicit_runtime_failure_is_not_silently_skipped():
    stub = _explicit_grid_stub("decode")
    InstrumentedScheduler._bench_build_grid(stub)
    point = stub._bench_grid[0]

    with pytest.raises(RuntimeError, match=r"benchmark_id=1.*injection_failed"):
        InstrumentedScheduler._bench_skip_point(stub, point, "injection_failed")


# ---------------------------------------------------------------------------
# Prefill KV-read grid and seed lifecycle
# ---------------------------------------------------------------------------


def test_benchmark_hasher_uses_vllm_hash_granularity(monkeypatch, tmp_path):
    parent_init_args = {}

    def fake_parent_init(self, **kwargs):
        parent_init_args.update(kwargs)
        self.block_size = kwargs["block_size"]
        self.cache_config = SimpleNamespace(
            enable_prefix_caching=True,
            prefix_caching_hash_algo="builtin",
        )

    monkeypatch.setattr(
        instrumented_scheduler_module.AsyncScheduler,
        "__init__",
        fake_parent_init,
    )
    monkeypatch.setattr(
        instrumented_scheduler_module,
        "_FpmPublisherThread",
        MagicMock(),
    )
    caching_hash_fn = MagicMock()
    monkeypatch.setattr(
        instrumented_scheduler_module,
        "get_hash_fn_by_name",
        MagicMock(return_value=caching_hash_fn),
    )
    monkeypatch.setattr(
        instrumented_scheduler_module,
        "init_none_hash",
        MagicMock(),
    )
    block_hasher = MagicMock()
    block_hasher_factory = MagicMock(return_value=block_hasher)
    monkeypatch.setattr(
        instrumented_scheduler_module,
        "get_request_block_hasher",
        block_hasher_factory,
    )
    monkeypatch.delenv(
        instrumented_scheduler_module.ENV_FPM_BENCHMARK_OUTPUT_PATH,
        raising=False,
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_master_ip="127.0.0.1",
        ),
        additional_config={
            "benchmark": {"output_path": str(tmp_path / "benchmark.json")}
        },
    )
    scheduler = InstrumentedScheduler(
        vllm_config=vllm_config,
        kv_cache_config=object(),
        structured_output_manager=object(),
        block_size=32,
        hash_block_size=16,
    )

    assert parent_init_args["hash_block_size"] == 16
    assert scheduler._bench_hash_block_size == 16
    assert scheduler._bench_block_hasher is block_hasher
    block_hasher_factory.assert_called_once_with(16, caching_hash_fn)


def test_agg_resolves_piecewise_prefill_and_full_decode_capture_views(tmp_path):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._fpm_dp_rank = 0
    stub.max_num_running_reqs = 8
    stub._bench_hash_block_size = 16
    stub.cache_config = SimpleNamespace(enable_prefix_caching=False)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_master_ip="127.0.0.1",
        ),
        additional_config={
            "benchmark": {"mode": "agg", "output_path": str(tmp_path / "out.json")}
        },
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
            max_cudagraph_capture_size=16,
        ),
        speculative_config=None,
    )

    InstrumentedScheduler._bench_init(stub, vllm_config)

    assert stub._bench_prefill_cudagraph_mode == "PIECEWISE"
    assert stub._bench_prefill_capture_sizes == [1, 2, 4, 8, 16]
    assert stub._bench_decode_cudagraph_mode == "FULL"
    assert stub._bench_decode_capture_sizes == [1, 2, 4, 8]
    assert stub._bench_max_cudagraph_capture_size == 16


def test_cudagraph_disabled_uses_geometric_fallback(tmp_path):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._fpm_dp_rank = 0
    stub.max_num_running_reqs = 8
    stub._bench_hash_block_size = 16
    stub.cache_config = SimpleNamespace(enable_prefix_caching=False)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_master_ip="127.0.0.1",
        ),
        additional_config={
            "benchmark": {
                "mode": "prefill",
                "output_path": str(tmp_path / "out.json"),
            }
        },
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
            max_cudagraph_capture_size=0,
        ),
        speculative_config=None,
    )

    InstrumentedScheduler._bench_init(stub, vllm_config)

    assert stub._bench_prefill_cudagraph_mode == "NONE"
    assert stub._bench_decode_cudagraph_mode == "NONE"
    assert instrumented_scheduler_module._cudagraph_axis_points([], 10) == [
        1,
        2,
        4,
        8,
        10,
    ]


def test_decode_benchmark_rejects_speculative_decoding(tmp_path):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._fpm_dp_rank = 0
    stub.max_num_running_reqs = 8
    stub._bench_hash_block_size = 16
    stub.cache_config = SimpleNamespace(enable_prefix_caching=False)
    vllm_config = SimpleNamespace(
        additional_config={
            "benchmark": {
                "mode": "decode",
                "output_path": str(tmp_path / "out.json"),
            }
        },
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
            max_cudagraph_capture_size=0,
        ),
        speculative_config=object(),
    )

    with pytest.raises(ValueError, match="does not yet support speculative"):
        InstrumentedScheduler._bench_init(stub, vllm_config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prefill_max_new_token_samples", 1, "must be at least 2"),
        ("prefill_max_kv_read_token_samples", 1, "must be at least 2"),
        ("decode_max_kv_read_token_samples", 1, "must be at least 2"),
        ("decode_max_batch_size_samples", 1, "must be at least 2"),
        ("prefix_max_batch_size_samples", 0, "must be positive"),
    ],
)
def test_benchmark_rejects_invalid_sampling_limits(tmp_path, field, value, message):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    config = {"output_path": str(tmp_path / "out.json"), field: value}
    vllm_config = SimpleNamespace(additional_config={"benchmark": config})

    with pytest.raises(ValueError, match=message):
        InstrumentedScheduler._bench_init(stub, vllm_config)


def _prefill_grid_stub(
    *,
    block_size: int = 8,
    num_gpu_blocks: int = 64,
):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_grid = []
    stub._bench_config = BenchmarkConfig()
    stub.max_num_scheduled_tokens = 40
    stub.max_num_running_reqs = 8
    stub.max_model_len = 128
    stub.cache_config = SimpleNamespace(num_gpu_blocks=num_gpu_blocks)
    stub.block_size = block_size
    stub._bench_hash_block_size = block_size
    stub._bench_prefill_capture_sizes = [8, 16]
    stub._bench_prefill_cudagraph_mode = "PIECEWISE"
    stub.num_lookahead_tokens = 0
    _install_test_capacity_preflight(
        stub,
        _benchmark_capacity(
            max_model_len=stub.max_model_len,
            max_num_scheduled_tokens=stub.max_num_scheduled_tokens,
            max_num_running_reqs=stub.max_num_running_reqs,
            usable_blocks_without_watermark=num_gpu_blocks - 1,
        ),
    )
    return stub


def test_cudagraph_axis_keeps_all_boundaries_and_geometric_tail():
    assert instrumented_scheduler_module._cudagraph_axis_points([1, 2, 4, 8], 32) == [
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        32,
    ]


def test_uniform_axis_limit_retains_endpoints_and_evenly_removes_middle():
    values = list(range(10))

    assert instrumented_scheduler_module._uniformly_limit_axis(values, 4) == [
        0,
        3,
        6,
        9,
    ]
    assert instrumented_scheduler_module._uniformly_limit_axis(values, 10) == values
    with pytest.raises(ValueError, match="at least 2"):
        instrumented_scheduler_module._uniformly_limit_axis(values, 1)


def test_cudagraph_axis_limit_preserves_eager_tail_below_twenty_percent():
    capture_sizes = list(range(1, 41))
    candidates = instrumented_scheduler_module._cudagraph_axis_points(capture_sizes, 80)

    assert candidates == [*range(1, 42), 80]
    assert instrumented_scheduler_module._limit_cudagraph_axis(
        candidates, capture_sizes, 4
    ) == [1, 40, 41, 80]


def test_cudagraph_axis_limit_preserves_eager_tail_at_twenty_percent():
    capture_sizes = list(range(1, 9))
    candidates = instrumented_scheduler_module._cudagraph_axis_points(capture_sizes, 10)

    assert candidates == list(range(1, 11))
    assert instrumented_scheduler_module._limit_cudagraph_axis(
        candidates, capture_sizes, 5
    ) == [1, 5, 8, 9, 10]


def test_cudagraph_axis_limit_uniformly_samples_tail_above_twenty_percent():
    capture_sizes = [8]
    candidates = instrumented_scheduler_module._cudagraph_axis_points(capture_sizes, 64)

    assert candidates == [8, 9, 16, 32, 64]
    assert instrumented_scheduler_module._limit_cudagraph_axis(
        candidates, capture_sizes, 3
    ) == [8, 16, 64]


def test_cudagraph_axis_appends_exact_non_power_of_two_limit():
    assert instrumented_scheduler_module._cudagraph_axis_points([256], 1000) == [
        256,
        257,
        512,
        1000,
    ]


def test_cudagraph_axis_does_not_add_eager_tail_below_larger_capture():
    assert instrumented_scheduler_module._cudagraph_axis_points([8, 64], 32) == [
        8,
        9,
        32,
    ]


def test_iteration_totals_are_distributed_evenly_and_exactly():
    assert instrumented_scheduler_module._balanced_partition(513, 4) == [
        129,
        128,
        128,
        128,
    ]
    assert instrumented_scheduler_module._balanced_partition(
        40, 3, unit=8, minimum_units=1
    ) == [16, 16, 8]
    assert InstrumentedScheduler._bench_decode_context_lengths(10, 3) == [4, 3, 3]


def test_prefill_grid_uses_total_tokens_and_piecewise_boundaries():
    stub = _prefill_grid_stub()

    InstrumentedScheduler._bench_generate_prefill_grid(stub)

    total_tokens = {point.total_prefill_tokens for point in stub._bench_grid}
    assert total_tokens == {8, 9, 16, 17, 32, 40}
    assert any(point.total_kv_read_tokens == 0 for point in stub._bench_grid)
    assert any(point.total_kv_read_tokens > 0 for point in stub._bench_grid)
    assert all(
        sum(
            InstrumentedScheduler._bench_prefill_new_token_lengths(
                point.total_prefill_tokens, point.batch_size
            )
        )
        == point.total_prefill_tokens
        for point in stub._bench_grid
    )
    post_capture = next(
        point
        for point in stub._bench_grid
        if point.total_prefill_tokens == 9 and point.batch_size == 1
    )
    assert post_capture.expected_capture_size == 16
    assert post_capture.padding_tokens == 7
    assert post_capture.sample_reasons == ["post_capture"]
    eager_tail = next(
        point
        for point in stub._bench_grid
        if point.total_prefill_tokens == 32 and point.batch_size == 1
    )
    assert eager_tail.expected_capture_size is None
    assert eager_tail.expected_cudagraph_mode == "NONE"
    assert eager_tail.sample_reasons == ["eager_tail", "geometric_tail"]
    engine_limit = next(
        point
        for point in stub._bench_grid
        if point.total_prefill_tokens == 40 and point.batch_size == 1
    )
    assert engine_limit.sample_reasons == ["eager_tail", "engine_limit"]


def test_prefill_grid_uniformly_limits_new_tokens_batch_and_kv_axes():
    stub = _prefill_grid_stub(num_gpu_blocks=512)
    stub.max_num_scheduled_tokens = 40
    stub._bench_prefill_capture_sizes = list(range(1, 41))
    stub._bench_config.prefill_max_new_token_samples = 4
    stub._bench_config.prefill_max_kv_read_token_samples = 3
    stub._bench_config.prefix_max_batch_size_samples = 1

    InstrumentedScheduler._bench_generate_prefill_grid(stub)

    assert sorted({point.total_prefill_tokens for point in stub._bench_grid}) == [
        1,
        14,
        27,
        40,
    ]
    assert {point.batch_size for point in stub._bench_grid} == {1}
    for total_tokens in (1, 14, 27, 40):
        points = [
            point.total_kv_read_tokens
            for point in stub._bench_grid
            if point.total_prefill_tokens == total_tokens
        ]
        assert len(points) <= 3
        assert points[0] == InstrumentedScheduler._bench_max_prefill_kv_read_tokens(
            stub, total_tokens, 1
        )
        assert points[-1] == 0


def test_prefill_grid_runs_larger_workload_coordinates_first():
    stub = _prefill_grid_stub(num_gpu_blocks=512)

    InstrumentedScheduler._bench_generate_prefill_grid(stub)

    coordinates = [
        (
            point.total_prefill_tokens,
            point.batch_size,
            point.total_kv_read_tokens,
        )
        for point in stub._bench_grid
    ]
    assert coordinates == sorted(coordinates, reverse=True)


def test_prefill_grid_uses_common_attention_dp_capacity():
    larger_rank = _prefill_grid_stub(num_gpu_blocks=64)
    smaller_rank = _prefill_grid_stub(num_gpu_blocks=48)
    common = _benchmark_capacity(
        max_model_len=96,
        max_num_scheduled_tokens=40,
        max_num_running_reqs=8,
        usable_blocks_without_watermark=47,
        usable_blocks_with_watermark=47,
    )
    larger_rank._bench_negotiated_capacity = common
    smaller_rank._bench_negotiated_capacity = common

    InstrumentedScheduler._bench_generate_prefill_grid(larger_rank)
    InstrumentedScheduler._bench_generate_prefill_grid(smaller_rank)

    larger_grid = [point.__dict__ for point in larger_rank._bench_grid]
    smaller_grid = [point.__dict__ for point in smaller_rank._bench_grid]
    assert larger_grid == smaller_grid

    for stub in (larger_rank, smaller_rank):
        stub._bench_negotiated_capacity = None
        assert all(
            InstrumentedScheduler._bench_prefill_point_feasible(
                stub,
                point.total_prefill_tokens,
                point.batch_size,
                point.total_kv_read_tokens,
            )
            for point in stub._bench_grid
        )


def test_explicit_prefill_point_uses_negotiated_scheduled_token_limit():
    """An explicit point at the negotiated ``max_num_scheduled_tokens`` must
    get identical cudagraph metadata on every rank regardless of the rank's
    local limit — otherwise ``sample_reasons`` (engine_limit vs
    geometric_tail) and therefore the per-point digests diverge."""
    at_limit_rank = _prefill_grid_stub()
    above_limit_rank = _prefill_grid_stub()
    above_limit_rank.max_num_scheduled_tokens = 48
    common = _benchmark_capacity(
        max_model_len=128,
        max_num_scheduled_tokens=40,
        max_num_running_reqs=8,
        usable_blocks_without_watermark=63,
    )
    candidate = PrefillPointCandidate(
        total_prefill_tokens=40, batch_size=1, total_kv_read_tokens=0
    )

    points = []
    for stub in (at_limit_rank, above_limit_rank):
        stub._bench_negotiated_capacity = common
        points.append(
            InstrumentedScheduler._bench_materialize_prefill_candidate(
                stub, candidate, "points[0]"
            )
        )

    assert points[0] == points[1]
    assert "engine_limit" in points[0].sample_reasons
    assert "geometric_tail" not in points[1].sample_reasons


def test_agg_grid_contains_piecewise_prefill_then_full_decode_points():
    stub = _prefill_grid_stub()
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []
    stub._bench_grid_error = None
    stub._bench_feasible_max_decode_batch_size = 0
    stub._bench_config.mode = "agg"
    stub._bench_explicit_points = None
    stub._bench_decode_capture_sizes = [1, 2, 4, 8]
    stub._bench_decode_cudagraph_mode = "FULL"

    InstrumentedScheduler._bench_build_grid(stub)

    points = list(stub._bench_grid)
    point_types = [point.point_type for point in points]
    first_decode = point_types.index("decode")
    assert all(point_type == "prefill" for point_type in point_types[:first_decode])
    assert all(point_type == "decode" for point_type in point_types[first_decode:])
    assert {
        point.expected_cudagraph_mode
        for point in points
        if point.point_type == "prefill" and point.expected_capture_size is not None
    } == {"PIECEWISE"}
    assert {
        point.expected_cudagraph_mode
        for point in points
        if point.point_type == "prefill" and point.expected_capture_size is None
    } == {"NONE"}
    assert {
        point.expected_cudagraph_mode
        for point in points
        if point.point_type == "decode"
    } == {"FULL"}


def test_agg_eager_warmups_stay_contiguous_with_their_phase():
    """``_bench_pop_next()`` treats a type mismatch at the queue front as
    "phase complete", so eager warmups of both types prepended as a single
    run would end PREFILL_SWEEP at the first decode warmup and DECODE_SWEEP
    at the first real prefill point, silently dropping every real point."""
    stub = _prefill_grid_stub()
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []
    stub._bench_grid_error = None
    stub._bench_feasible_max_decode_batch_size = 0
    stub._bench_config.mode = "agg"
    stub._bench_explicit_points = None
    # Captures end below the feasible max batch so decode also has eager
    # shapes; prefill already has them (max tokens 40 > largest capture 16).
    stub._bench_decode_capture_sizes = [1, 2, 4]
    stub._bench_decode_cudagraph_mode = "FULL"

    InstrumentedScheduler._bench_build_grid(stub)

    points = list(stub._bench_grid)
    warmup_types = {
        point.point_type
        for point in points
        if instrumented_scheduler_module.EAGER_WARMUP_REASON in point.sample_reasons
    }
    assert warmup_types == {"prefill", "decode"}, "need warmups on both phases"

    # Drain the grid exactly as the phase machine does: prefill until the
    # front stops matching, then decode.
    drained = 0
    for phase in ("prefill", "decode"):
        while InstrumentedScheduler._bench_pop_next(stub, phase) is not None:
            drained += 1
    assert not stub._bench_grid, "phase transitions must consume every point"
    assert drained == len(points)


def test_prefill_kv_read_ladder_is_total_block_aligned():
    stub = _prefill_grid_stub(block_size=8)

    points = InstrumentedScheduler._bench_prefill_kv_read_points(stub, 16, 3)

    assert points == [0, 24, 32, 64, 128, 256, 360]
    assert all(total % stub._bench_hash_block_size == 0 for total in points)
    for total in points[1:]:
        per_request = InstrumentedScheduler._bench_prefill_kv_read_lengths(
            stub, total, 3
        )
        assert sum(per_request) == total
        assert max(per_request) - min(per_request) <= stub._bench_hash_block_size


def test_prefill_kv_read_ladder_is_uniformly_limited_with_endpoints():
    stub = _prefill_grid_stub(block_size=8)
    stub._bench_config.prefill_max_kv_read_token_samples = 4

    assert InstrumentedScheduler._bench_prefill_kv_read_points(stub, 16, 3) == [
        0,
        32,
        128,
        360,
    ]


def test_decode_kv_read_ladder_keeps_every_power_of_two_and_exact_maximum():
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=64, block_size=16)

    # Presets 9 (all ctx=1) and 16 (mixed ctx 1/2) both measure at 18 after
    # the admission clamp, so they normalize into a single steady coordinate;
    # every preset at or above 2 * batch_size keeps its exact value.
    assert InstrumentedScheduler._bench_decode_kv_read_points(stub, 9) == [
        18,
        32,
        64,
        128,
        256,
        512,
        999,
    ]


def test_decode_kv_read_points_are_normalized_steady_coordinates():
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=64, block_size=16)

    for batch_size in (1, 8, 24):
        points = InstrumentedScheduler._bench_decode_kv_read_points(stub, batch_size)
        assert points, f"batch_size={batch_size} produced an empty ladder"
        assert len(points) == len(set(points))
        assert points[0] == 2 * batch_size
        assert all(
            InstrumentedScheduler._bench_decode_steady_kv_tokens(batch_size, value)
            == value
            for value in points
        ), "normalization must be idempotent: labels equal measured coordinates"


def test_decode_kv_read_points_merge_colliding_sub_2b_presets():
    # Non-power-of-two batch: presets 24 (all ctx=1) and 32 (8 ctx=2 +
    # 16 ctx=1) both measure at 48, a coordinate absent from the raw preset
    # ladder. They must collapse into one point instead of duplicating it.
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=64, block_size=16)

    assert InstrumentedScheduler._bench_decode_steady_kv_tokens(24, 24) == 48
    assert InstrumentedScheduler._bench_decode_steady_kv_tokens(24, 32) == 48

    points = InstrumentedScheduler._bench_decode_kv_read_points(stub, 24)
    assert 48 in points
    assert 24 not in points
    assert 32 not in points
    assert points.count(48) == 1


def test_decode_kv_read_ladder_boundaries_at_model_len_floor():
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=64, block_size=16)

    # max_model_len=4 bounds every request to ctx=2, so max_kv is exactly
    # 2 * batch_size and the ladder collapses to that single steady point.
    stub.max_model_len = 4
    assert InstrumentedScheduler._bench_decode_kv_read_points(stub, 4) == [8]

    # Below the two-step floor (ctx=2 needs max(ctx, 2) + 2 slots) no point
    # is feasible: the ladder must be empty rather than emit a normalized
    # coordinate above the validated maximum.
    stub.max_model_len = 3
    assert InstrumentedScheduler._bench_decode_kv_read_points(stub, 4) == []


def test_decode_grid_uniformly_limits_batch_and_kv_axes():
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks=64, block_size=16)
    stub._bench_decode_capture_sizes = list(range(1, 64))
    stub._bench_config.decode_max_batch_size_samples = 4
    stub._bench_config.decode_max_kv_read_token_samples = 3

    InstrumentedScheduler._bench_generate_decode_grid(stub)

    assert sorted({point.batch_size for point in stub._bench_grid}) == [1, 22, 42, 63]
    for batch_size in (1, 22, 42, 63):
        sampled = [
            point.total_kv_read_tokens
            for point in stub._bench_grid
            if point.batch_size == batch_size
        ]
        full_axis = InstrumentedScheduler._bench_decode_kv_read_points(stub, batch_size)
        assert len(sampled) <= 3
        assert sampled[0] == full_axis[0]
        assert sampled[-1] == full_axis[-1]


def test_prefill_kv_read_ladder_falls_back_to_miss_when_cache_is_disabled():
    stub = _prefill_grid_stub(block_size=8)
    stub.cache_config.enable_prefix_caching = False

    assert InstrumentedScheduler._bench_prefill_kv_read_points(stub, 16, 3) == [0]


def test_prefill_batch_axis_keeps_first_configured_number_of_samples():
    stub = _prefill_grid_stub(
        block_size=8,
        num_gpu_blocks=8,
    )

    assert InstrumentedScheduler._bench_prefill_batch_sizes(stub, 10) == [1, 2, 4]

    stub._bench_config.prefix_max_batch_size_samples = 4
    assert InstrumentedScheduler._bench_prefill_batch_sizes(stub, 10) == [
        1,
        2,
        4,
        7,
    ]


def test_prefill_batch_axis_filters_per_request_model_length():
    stub = _prefill_grid_stub()
    stub.max_model_len = 5

    assert not InstrumentedScheduler._bench_prefill_point_feasible(stub, 8, 1, 0)
    assert InstrumentedScheduler._bench_prefill_point_feasible(stub, 8, 2, 0)
    assert InstrumentedScheduler._bench_prefill_batch_sizes(stub, 8) == [2, 4, 8]


def test_prefill_batch_grid_uses_live_free_block_count_after_manager_reservations():
    stub = _prefill_grid_stub(
        block_size=8,
        num_gpu_blocks=100,
    )
    stub.kv_cache_manager = SimpleNamespace(
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 7)
    )

    # The live pool, rather than configured capacity, determines the legal max.
    assert InstrumentedScheduler._bench_prefill_batch_sizes(stub, 10) == [1, 2, 4]


def test_prefill_kv_read_grid_accounts_for_eagle_cache_block_drop():
    stub = _prefill_grid_stub(block_size=8)
    stub.kv_cache_manager = SimpleNamespace(use_eagle=True)

    assert InstrumentedScheduler._bench_prefill_kv_read_points(stub, 8, 1) == [0]

    points = InstrumentedScheduler._bench_prefill_kv_read_points(stub, 40, 1)
    assert points[0] == 0
    assert all(point % 8 == 0 for point in points)
    assert InstrumentedScheduler._bench_seed_prompt_len(stub, 16) == 24

    # The extra seed block must also fit under max_model_len.
    assert InstrumentedScheduler._bench_prefill_kv_read_points(stub, 9, 1)[-1] == 112


def test_prefill_eagle_kv_read_requires_more_than_one_drop_block_per_request():
    stub = _prefill_grid_stub(block_size=8)
    stub.kv_cache_manager = SimpleNamespace(use_eagle=True)

    assert not InstrumentedScheduler._bench_prefill_point_feasible(stub, 17, 2, 16)
    assert InstrumentedScheduler._bench_prefill_point_feasible(stub, 18, 2, 16)


def test_prefill_eagle_partial_hash_hit_uses_hash_drop_granularity():
    stub = _prefill_grid_stub(block_size=16)
    stub._bench_hash_block_size = 8
    stub.kv_cache_manager = SimpleNamespace(
        use_eagle=True,
        coordinator=SimpleNamespace(enable_partial_hash_hits=True),
    )

    assert InstrumentedScheduler._bench_eagle_cache_drop_tokens(stub) == 8
    assert InstrumentedScheduler._bench_seed_prompt_len(stub, 16) == 24
    assert not InstrumentedScheduler._bench_prefill_point_feasible(stub, 8, 1, 8)
    assert InstrumentedScheduler._bench_prefill_point_feasible(stub, 9, 1, 8)


def test_prefill_fake_seed_feasibility_uses_uncapped_allocation():
    stub = _prefill_grid_stub(block_size=8)
    stub._bench_prefill_blocks_per_req = MagicMock(return_value=1)
    stub._bench_blocks_per_req = MagicMock(return_value=1)
    stub._bench_usable_blocks = MagicMock(return_value=8)

    assert InstrumentedScheduler._bench_prefill_point_feasible(stub, 8, 1, 8)

    stub._bench_blocks_per_req.assert_called_once_with(
        8,
        has_cache_hit=False,
        apply_admission_cap=False,
    )


def test_mamba_connector_uses_scheduler_per_group_cache_lookup():
    stub = _prefill_grid_stub(block_size=8)
    coordinator = SimpleNamespace(
        find_longest_cache_hit_per_group=MagicMock(return_value=(([], []), (16, 8)))
    )
    stub.kv_cache_manager = SimpleNamespace(
        use_eagle=True,
        coordinator=coordinator,
        get_computed_blocks=MagicMock(side_effect=AssertionError("wrong lookup")),
    )
    stub.connector = object()
    stub.has_mamba_layers = True
    request = SimpleNamespace(block_hashes=[b"hash"], num_tokens=40)

    assert InstrumentedScheduler._bench_eagle_cache_drop_tokens(stub) == 0
    assert InstrumentedScheduler._bench_seed_prompt_len(stub, 16) == 16
    assert InstrumentedScheduler._bench_cached_kv_read_tokens(stub, request) == 16
    coordinator.find_longest_cache_hit_per_group.assert_called_once_with(
        request.block_hashes, request.num_tokens - 1
    )


def test_prefill_kv_read_validation_does_not_record_prefix_cache_stats():
    stub = _prefill_grid_stub(block_size=8)
    coordinator = SimpleNamespace(
        find_longest_cache_hit=MagicMock(return_value=(([],), 16, 0))
    )
    get_computed_blocks = MagicMock(
        side_effect=AssertionError("stats-recording lookup should not be used")
    )
    stub.kv_cache_manager = SimpleNamespace(
        coordinator=coordinator,
        get_computed_blocks=get_computed_blocks,
    )
    stub.connector = None
    stub.has_mamba_layers = False
    request = SimpleNamespace(block_hashes=[b"hash"], num_tokens=40)

    assert InstrumentedScheduler._bench_cached_kv_read_tokens(stub, request) == 16
    coordinator.find_longest_cache_hit.assert_called_once_with(
        request.block_hashes, request.num_tokens - 1
    )
    get_computed_blocks.assert_not_called()


def test_prefill_kv_read_uses_fake_cache_and_measures_immediately():
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_grid = deque([point])
    stub._bench_config = SimpleNamespace(mode="prefill")
    stub._bench_active_req_ids = set()
    stub._bench_current_point = None
    stub._bench_current_fpms = []
    stub._bench_drain_pending = False
    stub._bench_seq = 7
    stub._bench_hash_block_size = 8
    stub._schedule_times = deque()
    stub.requests = {}
    stub._bench_sync_pending = False
    stub.kv_cache_manager = SimpleNamespace(new_step_starts=MagicMock())

    calls = []

    def inject(**kwargs):
        calls.append(kwargs)
        stub._bench_active_req_ids.add(f"request-{len(calls)}")
        return len(kwargs["prompt_lens"])

    stub._bench_inject_prefill = inject
    stub._bench_cache_fake_prefixes = MagicMock(return_value=True)

    InstrumentedScheduler._bench_step_prefill(stub)

    # The measured point carries the prefill provenance stamp.
    assert stub._bench_current_point == replace(
        point, sample_reasons=[*point.sample_reasons, "prefill_fake_prefix"]
    )
    assert (
        InstrumentedScheduler._kvwarm_seed_regime(stub, stub._bench_current_point)
        == "fake_prefix"
    )
    seed_salts = stub._bench_cache_fake_prefixes.call_args.kwargs["cache_salts"]
    assert len(seed_salts) == point.batch_size
    assert len(set(seed_salts)) == point.batch_size
    stub._bench_cache_fake_prefixes.assert_called_once_with(
        prefix_lengths=[16, 16, 8],
        cache_salts=seed_salts,
    )
    stub.kv_cache_manager.new_step_starts.assert_called_once_with()
    assert calls == [
        {
            "prompt_lens": [25, 24, 16],
            "max_tokens": 1,
            "cache_salts": seed_salts,
            "expected_kv_read_tokens": [16, 16, 8],
        }
    ]
    assert stub._bench_sync_pending is True


def _realseed_prefill_stub(point, monkeypatch, seq=0, drop=0, points=None):
    monkeypatch.setenv("DYN_BENCH_PREFILL_REAL_SEED", "on")
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_grid = deque(points if points is not None else [point])
    stub._bench_config = SimpleNamespace(mode="prefill")
    stub._bench_active_req_ids = set()
    stub._bench_current_point = None
    stub._bench_current_fpms = []
    stub._bench_drain_pending = False
    stub._bench_seq = seq
    stub._bench_hash_block_size = 8
    stub._schedule_times = deque()
    stub._bench_skipped_points = []
    stub._bench_sync_pending = False
    stub.requests = {}
    stub.kv_cache_manager = SimpleNamespace(new_step_starts=MagicMock())
    stub._bench_eagle_cache_drop_tokens = lambda: drop
    # Injective, prefix-consistent content: a fresh id per salt, repeated.
    ids: dict[str, int] = {}
    stub._salts_requested = []

    def synthetic(salt, n):
        stub._salts_requested.append(salt)
        return [ids.setdefault(salt, 100 + len(ids))] * n

    stub._bench_synthetic_token_ids = synthetic
    stub._bench_cache_fake_prefixes = MagicMock(
        side_effect=AssertionError("fake prefix path must not run under real-seed")
    )
    calls = []

    def inject(**kwargs):
        calls.append(kwargs)
        return len(kwargs["prompt_lens"])

    stub._bench_inject_prefill = inject
    return stub, calls


def _stamped(point, reason="prefill_real_seed"):
    return replace(point, sample_reasons=[*point.sample_reasons, reason])


def test_prefill_real_seed_stages_warms_then_measures(monkeypatch):
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    stub, calls = _realseed_prefill_stub(point, monkeypatch)

    # Shot 1: staging computes the seeded prefixes (unbooked, one fixed salt
    # per slot); the stamped point is parked.
    InstrumentedScheduler._bench_step_prefill(stub)
    chain = stub._bench_rsc[3]
    assert chain["salts"] == [
        "__bench_rsc_bp3_slot0",
        "__bench_rsc_bp3_slot1",
        "__bench_rsc_bp3_slot2",
    ]
    assert calls == [
        {"prompt_lens": [16, 16, 8], "max_tokens": 1, "cache_salts": chain["salts"]}
    ]
    assert stub._bench_current_point is None
    assert stub._bench_realseed_ready[0] == _stamped(point)
    assert stub._bench_sync_pending is False

    # Shot 2: same-shape warm pass, unbooked; chain depth recorded.
    InstrumentedScheduler._bench_step_prefill(stub)
    assert chain["depth"] == [16, 16, 8]
    warm = calls[1]
    assert warm["prompt_lens"] == [25, 24, 16]
    assert warm["cache_salts"] == chain["salts"]
    assert "expected_kv_read_tokens" not in warm
    prefix_ids = [stub._bench_synthetic_token_ids(s, 1)[0] for s in chain["salts"]]
    for slot, prompt in enumerate(warm["prompt_token_ids_list"]):
        kv = [16, 16, 8][slot]
        assert prompt[:kv] == [prefix_ids[slot]] * kv, "seeded prefix must lead"
        assert len(prompt) == warm["prompt_lens"][slot]
    assert stub._bench_current_point is None
    assert stub._bench_realseed_stage == "measure"

    # Shot 3: measured pass validates the hit; the fresh tail comes from a
    # different salt than the warm tail.
    InstrumentedScheduler._bench_step_prefill(stub)
    measured = calls[2]
    assert measured["prompt_lens"] == [25, 24, 16]
    assert measured["expected_kv_read_tokens"] == [16, 16, 8]
    assert measured["cache_salts"] == chain["salts"]
    assert [p[:8] for p in measured["prompt_token_ids_list"]] == [
        p[:8] for p in warm["prompt_token_ids_list"]
    ]
    assert any("rswarm" in s for s in stub._salts_requested)
    assert any("__bench_rsm_" in s for s in stub._salts_requested)
    assert (
        measured["prompt_token_ids_list"][0][16:]
        != warm["prompt_token_ids_list"][0][16:]
    )
    assert stub._bench_current_point == _stamped(point)
    assert (
        InstrumentedScheduler._kvwarm_seed_regime(stub, stub._bench_current_point)
        == "real_prefix"
    )
    assert stub._bench_sync_pending is True
    assert stub._bench_realseed_ready is None
    assert stub._bench_realseed_stage == "warm"
    stub._bench_cache_fake_prefixes.assert_not_called()
    # Seeded blocks have a producer (the staging/warm passes), so the fake
    # path's same-step hit guard must not be touched here.
    stub.kv_cache_manager.new_step_starts.assert_not_called()


def test_prefill_real_seed_measured_prompt_covers_eagle_drop_block(monkeypatch):
    """Under EAGLE/MTP the lookup drops the last matched block, so the chain
    holds kv+drop tokens and the measured prompt must reproduce all of them
    for the hit to come back as exactly kv."""
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=48,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    stub, calls = _realseed_prefill_stub(point, monkeypatch, drop=8)
    InstrumentedScheduler._bench_step_prefill(stub)  # staging
    assert calls[0]["prompt_lens"] == [24, 24, 16]
    InstrumentedScheduler._bench_step_prefill(stub)  # warm
    InstrumentedScheduler._bench_step_prefill(stub)  # measured
    measured = calls[2]
    chain = stub._bench_rsc[3]
    assert chain["depth"] == [24, 24, 16]
    # new tokens 48 -> [16, 16, 16]; kv 40 -> [16, 16, 8]; prompt = new + kv.
    assert measured["prompt_lens"] == [32, 32, 24]
    assert measured["expected_kv_read_tokens"] == [16, 16, 8]
    prefix_ids = [stub._bench_synthetic_token_ids(s, 1)[0] for s in chain["salts"]]
    for slot, (prompt, seed_len, total) in enumerate(
        zip(measured["prompt_token_ids_list"], [24, 24, 16], [32, 32, 24])
    ):
        assert prompt[:seed_len] == [prefix_ids[slot]] * seed_len
        assert len(prompt) == total
        assert prompt[seed_len] != prefix_ids[slot], "tail is fresh content"


def test_prefill_real_seed_staging_skips_zero_kv_slots(monkeypatch):
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=32,
        batch_size=3,
        rows=[[9, 0], [8, 16], [8, 16]],
    )
    stub, calls = _realseed_prefill_stub(point, monkeypatch)
    InstrumentedScheduler._bench_step_prefill(stub)  # staging
    chain = stub._bench_rsc[3]
    assert calls[0]["prompt_lens"] == [16, 16], "no empty prompt is injected"
    assert calls[0]["cache_salts"] == chain["salts"][1:]
    assert stub._bench_realseed_ready is not None
    InstrumentedScheduler._bench_step_prefill(stub)  # warm
    InstrumentedScheduler._bench_step_prefill(stub)  # measured
    measured = calls[2]
    assert measured["expected_kv_read_tokens"] == [0, 16, 16]
    assert measured["prompt_lens"] == [9, 24, 24]
    assert len(measured["prompt_token_ids_list"][0]) == 9
    assert stub._bench_sync_pending is True


def test_prefill_real_seed_skips_staging_when_chain_is_deep_enough(monkeypatch):
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    stub, calls = _realseed_prefill_stub(point, monkeypatch)
    chain = InstrumentedScheduler._bench_realseed_chain(stub, 3)
    chain["depth"] = [64, 64, 64]

    InstrumentedScheduler._bench_step_prefill(stub)

    assert calls == [], "a chain deeper than the point computes nothing"
    assert stub._bench_realseed_ready[0] == _stamped(point)
    assert chain["depth"] == [64, 64, 64]
    assert stub._bench_realseed_staged is False


def test_prefill_real_seed_reuses_chain_across_points_of_same_batch(monkeypatch):
    a = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    b = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=80,
        batch_size=3,
    )
    c = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=24,
        batch_size=3,
    )
    d = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=20,
        total_kv_read_tokens=40,
        batch_size=2,
    )
    stub, calls = _realseed_prefill_stub(a, monkeypatch, points=[a, b, c, d])
    for _ in range(3):  # a: staging, warm, measured
        InstrumentedScheduler._bench_step_prefill(stub)
    stub._bench_current_point = None  # the measured requests have drained
    InstrumentedScheduler._bench_step_prefill(stub)  # b: staging with the same salts
    chain = stub._bench_rsc[3]
    assert calls[3]["prompt_lens"] == [32, 24, 24]
    assert calls[3]["cache_salts"] == chain["salts"]
    for _ in range(2):
        InstrumentedScheduler._bench_step_prefill(stub)
    assert chain["depth"] == [32, 24, 24]
    stub._bench_current_point = None
    n = len(calls)
    InstrumentedScheduler._bench_step_prefill(stub)  # c: shallower, no staging
    assert len(calls) == n and stub._bench_realseed_staged is False
    InstrumentedScheduler._bench_step_prefill(stub)  # c: warm
    InstrumentedScheduler._bench_step_prefill(stub)  # c: measured
    assert calls[-1]["expected_kv_read_tokens"] == [8, 8, 8]
    assert calls[-1]["cache_salts"] == chain["salts"]
    assert chain["depth"] == [32, 24, 24], "depth is never lowered"
    stub._bench_current_point = None
    InstrumentedScheduler._bench_step_prefill(stub)  # d: other batch size, own chain
    assert calls[-1]["cache_salts"] == [
        "__bench_rsc_bp2_slot0",
        "__bench_rsc_bp2_slot1",
    ]
    assert stub._bench_rsc[2]["depth"] == [0, 0]


def test_prefill_real_seed_staging_failure_skips_point(monkeypatch):
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    stub, calls = _realseed_prefill_stub(point, monkeypatch)
    stub._bench_inject_prefill = MagicMock(return_value=0)

    InstrumentedScheduler._bench_step_prefill(stub)

    skipped = stub._bench_skipped_points[0]
    assert skipped.reason == "real_seed_injection_failed"
    assert (
        InstrumentedScheduler._kvwarm_seed_regime(stub, skipped.point) == "real_prefix"
    )
    assert getattr(stub, "_bench_realseed_ready", None) is None
    assert stub._bench_current_point is None


def test_prefill_real_seed_explicit_point_failure_raises(monkeypatch):
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
        sample_reasons=["explicit"],
    )
    stub, calls = _realseed_prefill_stub(point, monkeypatch)
    stub._bench_inject_prefill = MagicMock(return_value=0)
    with pytest.raises(RuntimeError, match="real_seed_injection_failed"):
        InstrumentedScheduler._bench_step_prefill(stub)


def test_prefill_real_seed_validation_miss_restages_once_then_skips(monkeypatch):
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    stub, calls = _realseed_prefill_stub(point, monkeypatch)
    InstrumentedScheduler._bench_step_prefill(stub)  # staging
    InstrumentedScheduler._bench_step_prefill(stub)  # warm
    chain = stub._bench_rsc[3]

    def miss(**kwargs):
        calls.append(kwargs)
        return 0 if "expected_kv_read_tokens" in kwargs else len(kwargs["prompt_lens"])

    stub._bench_inject_prefill = miss
    InstrumentedScheduler._bench_step_prefill(stub)  # measured: hit validation misses
    # Healed: the chain is forgotten and re-staged, the point is parked again.
    assert stub._bench_skipped_points == []
    assert calls[-1] == {
        "prompt_lens": [16, 16, 8],
        "max_tokens": 1,
        "cache_salts": chain["salts"],
    }
    assert stub._bench_realseed_ready[0] == _stamped(point)
    assert stub._bench_realseed_stage == "warm"
    InstrumentedScheduler._bench_step_prefill(stub)  # warm again
    InstrumentedScheduler._bench_step_prefill(stub)  # measured misses again -> skip
    assert stub._bench_skipped_points[0].reason == "real_seed_cache_validation_failed"
    assert (
        InstrumentedScheduler._kvwarm_seed_regime(
            stub, stub._bench_skipped_points[0].point
        )
        == "real_prefix"
    )
    assert stub._bench_current_point is None
    assert stub._bench_realseed_ready is None
    assert stub._bench_sync_pending is False


def test_prefill_real_seed_waits_for_each_shot_to_drain(monkeypatch):
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    stub, calls = _realseed_prefill_stub(point, monkeypatch)

    def inject(**kwargs):
        calls.append(kwargs)
        for i in range(len(kwargs["prompt_lens"])):
            rid = f"req-{len(calls)}-{i}"
            stub._bench_active_req_ids.add(rid)
            stub.requests[rid] = object()
        return len(kwargs["prompt_lens"])

    stub._bench_inject_prefill = inject
    stub._bench_point_deadline = 0.0
    stub._bench_stop_requested = False
    stub._kvwarm_borrowed_ids = set()
    stub.finish_requests = MagicMock()
    stub._bench_transition_to_timeout_done = lambda: False

    InstrumentedScheduler._bench_step_prefill(stub)  # staging injected
    assert len(calls) == 1
    InstrumentedScheduler._bench_step_prefill(stub)  # requests alive: nothing new
    assert len(calls) == 1 and stub._bench_realseed_stage == "warm"
    stub.requests.clear()  # staging requests finished
    InstrumentedScheduler._bench_step_prefill(stub)  # cleanup + drain requested
    assert stub._bench_drain_pending is True and len(calls) == 1
    InstrumentedScheduler._bench_step_prefill(stub)  # drained: warm shot
    assert len(calls) == 2 and stub._bench_realseed_stage == "measure"


def test_prefill_real_seed_parked_point_finishes_before_stop_boundary(monkeypatch):
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    stub, calls = _realseed_prefill_stub(point, monkeypatch)
    InstrumentedScheduler._bench_step_prefill(stub)  # staging, point parked
    stub._bench_stop_at_timeout_boundary = MagicMock(return_value=True)
    InstrumentedScheduler._bench_step_prefill(stub)  # warm still runs
    InstrumentedScheduler._bench_step_prefill(stub)  # measured still runs
    assert len(calls) == 3 and stub._bench_sync_pending is True
    stub._bench_stop_at_timeout_boundary.assert_not_called()


def test_prefill_real_seed_staging_content_is_measured_prefix(monkeypatch):
    """The staging prompt is built inside _bench_inject_prefill from the
    slot salt; the measured prefix is built in the pending step from the
    same salt. Both must agree token for token (per DP rank)."""
    for dp_rank in (0, 1):
        created = []

        class FakeRequest:
            def __init__(self, request_id, prompt_token_ids, cache_salt, **kwargs):
                self.request_id = request_id
                self.prompt_token_ids = list(prompt_token_ids)
                self.cache_salt = cache_salt
                created.append(self)

        monkeypatch.setattr(instrumented_scheduler_module, "Request", FakeRequest)
        monkeypatch.setattr(
            instrumented_scheduler_module, "SamplingParams", lambda **kwargs: object()
        )
        point = BenchmarkPoint(
            point_type="prefill",
            total_prefill_tokens=25,
            total_kv_read_tokens=40,
            batch_size=3,
        )
        stub, _ = _realseed_prefill_stub(point, monkeypatch)
        del stub._bench_inject_prefill  # use the real one
        del stub._bench_synthetic_token_ids  # use the real generator
        stub._bench_vocab_size = 1000
        stub._fpm_dp_rank = dp_rank
        stub._bench_block_hasher = None
        stub.add_request = MagicMock()
        stub._bench_cached_kv_read_tokens = (
            lambda req: 16 if len(req.prompt_token_ids) > 16 else 8
        )

        InstrumentedScheduler._bench_step_prefill(stub)  # staging
        staged = {r.cache_salt: r.prompt_token_ids for r in created}
        created.clear()
        stub._bench_active_req_ids.clear()  # staging requests finished
        InstrumentedScheduler._bench_step_prefill(stub)  # warm
        stub._bench_active_req_ids.clear()
        InstrumentedScheduler._bench_step_prefill(stub)  # measured
        measured = created[-3:]
        for req, kv, total in zip(measured, [16, 16, 8], [25, 24, 16]):
            assert req.prompt_token_ids[:kv] == staged[req.cache_salt][:kv]
            assert len(req.prompt_token_ids) == total
        assert stub._bench_sync_pending is True


def test_prefill_real_seed_switch_parsing(monkeypatch):
    monkeypatch.delenv("DYN_BENCH_PREFILL_REAL_SEED", raising=False)
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    assert InstrumentedScheduler._bench_realseed_on(stub) is False
    for value in ("on", "1", "true", "ON"):
        monkeypatch.setenv("DYN_BENCH_PREFILL_REAL_SEED", value)
        assert InstrumentedScheduler._bench_realseed_on(stub) is True
    monkeypatch.setenv("DYN_BENCH_PREFILL_REAL_SEED", "off")
    assert InstrumentedScheduler._bench_realseed_on(stub) is False


def test_seed_regime_for_prefill_rows():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    base = BenchmarkPoint(
        point_type="prefill", total_prefill_tokens=8, total_kv_read_tokens=16
    )
    assert InstrumentedScheduler._kvwarm_seed_regime(stub, base) == "not_applicable"
    real = replace(base, sample_reasons=["prefill_real_seed"])
    fake = replace(base, sample_reasons=["prefill_fake_prefix"])
    assert InstrumentedScheduler._kvwarm_seed_regime(stub, real) == "real_prefix"
    assert InstrumentedScheduler._kvwarm_seed_regime(stub, fake) == "fake_prefix"
    # The stamp is part of the point digest the DP READY handshake compares,
    # so ranks that disagree on the switch fail loudly instead of mixing.
    assert instrumented_scheduler_module._benchmark_point_digest(
        real
    ) != instrumented_scheduler_module._benchmark_point_digest(base)


def test_bench_inject_prefill_uses_explicit_prompt_token_ids(monkeypatch):
    created = []

    class FakeRequest:
        def __init__(self, request_id, prompt_token_ids, cache_salt, **kwargs):
            self.request_id = request_id
            self.prompt_token_ids = prompt_token_ids
            self.cache_salt = cache_salt
            created.append(self)

    monkeypatch.setattr(instrumented_scheduler_module, "Request", FakeRequest)
    monkeypatch.setattr(
        instrumented_scheduler_module, "SamplingParams", lambda **kwargs: object()
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_seq = 4
    stub._bench_block_hasher = None
    stub._bench_active_req_ids = set()
    stub._bench_synthetic_token_ids = MagicMock(
        side_effect=AssertionError("explicit prompts must bypass salt generation")
    )
    stub.add_request = MagicMock()

    injected = InstrumentedScheduler._bench_inject_prefill(
        stub,
        prompt_lens=[3, 2],
        max_tokens=1,
        cache_salts=["seed-0", "seed-1"],
        prompt_token_ids_list=[(7, 7, 9), [5, 6]],
    )

    assert injected == 2
    assert [r.prompt_token_ids for r in created] == [[7, 7, 9], [5, 6]]
    assert [r.cache_salt for r in created] == ["seed-0", "seed-1"]
    with pytest.raises(ValueError, match="prompt_token_ids_list must match"):
        InstrumentedScheduler._bench_inject_prefill(
            stub, prompt_lens=[3], max_tokens=1, prompt_token_ids_list=[[1], [2]]
        )
    with pytest.raises(ValueError, match="entry length"):
        InstrumentedScheduler._bench_inject_prefill(
            stub, prompt_lens=[3], max_tokens=1, prompt_token_ids_list=[[1, 2]]
        )


def test_prefill_fake_cache_validation_miss_skips_measured_point():
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=25,
        total_kv_read_tokens=40,
        batch_size=3,
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_grid = deque([point])
    stub._bench_config = SimpleNamespace(mode="prefill")
    stub._bench_active_req_ids = set()
    stub._bench_current_point = None
    stub._bench_current_fpms = []
    stub._bench_drain_pending = False
    stub._bench_seq = 0
    stub._bench_hash_block_size = 8
    stub._schedule_times = deque()
    stub._bench_skipped_points = []
    stub.kv_cache_manager = SimpleNamespace(new_step_starts=MagicMock())
    stub._bench_cache_fake_prefixes = MagicMock(return_value=True)
    stub._bench_inject_prefill = MagicMock(return_value=0)

    InstrumentedScheduler._bench_step_prefill(stub)

    assert stub._bench_current_point is None
    assert stub._bench_skipped_points[0].reason == "fake_prefix_cache_validation_failed"
    seed_salts = stub._bench_cache_fake_prefixes.call_args.kwargs["cache_salts"]
    stub._bench_inject_prefill.assert_called_once_with(
        prompt_lens=[25, 24, 16],
        max_tokens=1,
        cache_salts=seed_salts,
        expected_kv_read_tokens=[16, 16, 8],
    )


def test_fake_prefix_cache_allocates_caches_and_releases_blocks(monkeypatch):
    created_requests = []

    class FakeRequest:
        def __init__(self, request_id, prompt_token_ids, cache_salt, **kwargs):
            self.request_id = request_id
            self.prompt_token_ids = prompt_token_ids
            self.cache_salt = cache_salt
            created_requests.append(self)

    monkeypatch.setattr(instrumented_scheduler_module, "Request", FakeRequest)
    monkeypatch.setattr(
        instrumented_scheduler_module, "SamplingParams", lambda **kwargs: object()
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_seq = 4
    stub._bench_block_hasher = None
    stub.kv_cache_manager = SimpleNamespace(
        allocate_slots=MagicMock(side_effect=[object(), object(), object()]),
        free=MagicMock(),
        reset_prefix_cache=MagicMock(return_value=True),
    )

    assert InstrumentedScheduler._bench_cache_fake_prefixes(
        stub,
        prefix_lengths=[16, 16, 8],
        cache_salts=["salt-0", "salt-1", "salt-2"],
    )

    assert [req.request_id for req in created_requests] == [
        "__bench_fake_prefix_4",
        "__bench_fake_prefix_5",
        "__bench_fake_prefix_6",
    ]
    assert [len(req.prompt_token_ids) for req in created_requests] == [16, 16, 8]
    assert stub.kv_cache_manager.allocate_slots.call_args_list == [
        call(
            created_requests[0],
            16,
            full_sequence_must_fit=True,
            has_scheduled_reqs=False,
        ),
        call(
            created_requests[1],
            16,
            full_sequence_must_fit=True,
            has_scheduled_reqs=True,
        ),
        call(
            created_requests[2],
            8,
            full_sequence_must_fit=True,
            has_scheduled_reqs=True,
        ),
    ]
    assert stub.kv_cache_manager.free.call_args_list == [
        call(created_requests[0]),
        call(created_requests[1]),
        call(created_requests[2]),
    ]
    stub.kv_cache_manager.reset_prefix_cache.assert_not_called()
    assert stub._bench_seq == 7


def test_fake_prefix_cache_rolls_back_partial_allocation(monkeypatch):
    class FakeRequest:
        def __init__(self, request_id, **kwargs):
            self.request_id = request_id

    monkeypatch.setattr(instrumented_scheduler_module, "Request", FakeRequest)
    monkeypatch.setattr(
        instrumented_scheduler_module, "SamplingParams", lambda **kwargs: object()
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_seq = 0
    stub._bench_block_hasher = None
    stub.kv_cache_manager = SimpleNamespace(
        allocate_slots=MagicMock(side_effect=[object(), None]),
        free=MagicMock(),
        reset_prefix_cache=MagicMock(return_value=True),
    )

    assert not InstrumentedScheduler._bench_cache_fake_prefixes(
        stub,
        prefix_lengths=[8, 8],
        cache_salts=["salt-0", "salt-1"],
    )

    assert stub.kv_cache_manager.free.call_count == 2
    stub.kv_cache_manager.reset_prefix_cache.assert_called_once_with()
    assert stub._bench_seq == 0


def test_fake_prefix_cache_rolls_back_after_allocation_exception(monkeypatch):
    class FakeRequest:
        def __init__(self, request_id, **kwargs):
            self.request_id = request_id

    monkeypatch.setattr(instrumented_scheduler_module, "Request", FakeRequest)
    monkeypatch.setattr(
        instrumented_scheduler_module, "SamplingParams", lambda **kwargs: object()
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_seq = 0
    stub._bench_block_hasher = None
    stub.kv_cache_manager = SimpleNamespace(
        allocate_slots=MagicMock(side_effect=[object(), RuntimeError("allocate")]),
        free=MagicMock(),
        reset_prefix_cache=MagicMock(return_value=True),
    )

    with pytest.raises(RuntimeError, match="allocate"):
        InstrumentedScheduler._bench_cache_fake_prefixes(
            stub,
            prefix_lengths=[8, 8],
            cache_salts=["salt-0", "salt-1"],
        )

    assert stub.kv_cache_manager.free.call_count == 2
    stub.kv_cache_manager.reset_prefix_cache.assert_called_once_with()
    assert stub._bench_seq == 0


def test_benchmark_clear_prefix_cache_is_required_and_idempotent():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_prefix_cache_cleared = False
    stub.kv_cache_manager = SimpleNamespace(
        reset_prefix_cache=MagicMock(return_value=True)
    )
    stub.deferred_frees = deque()  # nothing fenced

    assert InstrumentedScheduler._bench_clear_prefix_cache(stub) is True
    assert InstrumentedScheduler._bench_clear_prefix_cache(stub) is True

    stub.kv_cache_manager.reset_prefix_cache.assert_called_once_with()
    assert stub._bench_prefix_cache_cleared is True

    failed = InstrumentedScheduler.__new__(InstrumentedScheduler)
    failed._bench_prefix_cache_cleared = False
    failed.deferred_frees = deque()  # nothing fenced
    failed.kv_cache_manager = SimpleNamespace(
        reset_prefix_cache=MagicMock(return_value=False)
    )
    with pytest.raises(RuntimeError, match="failed to clear synthetic prefix cache"):
        InstrumentedScheduler._bench_clear_prefix_cache(failed)


def test_benchmark_abort_clears_synthetic_prefix_cache_before_deactivation():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_synchronizer = None
    stub._bench_cleanup_requests = MagicMock()
    stub._bench_clear_prefix_cache = MagicMock()
    stub._bench_write_results = MagicMock()
    stub._bench_deactivate = MagicMock()

    InstrumentedScheduler._bench_abort(stub, RuntimeError("benchmark failed"))

    stub._bench_cleanup_requests.assert_called_once_with()
    # An abort has no later benchmark step to retry from, so it must attempt
    # the reset even while released blocks are still fenced.
    stub._bench_clear_prefix_cache.assert_called_once_with(allow_pending=True)
    stub._bench_write_results.assert_called_once_with()
    stub._bench_deactivate.assert_called_once_with(resume_publisher=True)
    assert stub._bench_grid_error == "benchmark failed"


def test_benchmark_abort_is_fail_closed_when_prefix_cleanup_fails():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_synchronizer = None
    stub._bench_cleanup_requests = MagicMock()
    stub._bench_clear_prefix_cache = MagicMock(
        side_effect=RuntimeError("cache still referenced")
    )
    stub._bench_write_results = MagicMock()
    stub._bench_deactivate = MagicMock()

    with pytest.raises(RuntimeError, match="synthetic prefix-cache cleanup failed"):
        InstrumentedScheduler._bench_abort(stub, RuntimeError("benchmark failed"))

    stub._bench_write_results.assert_called_once_with()
    stub._bench_deactivate.assert_called_once_with(resume_publisher=False)
    assert "cache still referenced" in stub._bench_grid_error


def test_prefill_batch_validation_is_atomic(monkeypatch):
    created_cache_salts = []

    class FakeRequest:
        def __init__(self, request_id, cache_salt, **kwargs):
            self.request_id = request_id
            self.cache_salt = cache_salt
            created_cache_salts.append(cache_salt)

    monkeypatch.setattr(instrumented_scheduler_module, "Request", FakeRequest)
    monkeypatch.setattr(
        instrumented_scheduler_module, "SamplingParams", lambda **kwargs: object()
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_seq = 4
    stub._bench_block_hasher = None
    stub._bench_active_req_ids = set()
    stub._bench_cached_kv_read_tokens = MagicMock(side_effect=[16, 8, 16])
    stub.add_request = MagicMock()

    injected = InstrumentedScheduler._bench_inject_prefill(
        stub,
        prompt_lens=[40, 41, 42],
        max_tokens=1,
        cache_salts=["seed-0", "seed-1", "seed-2"],
        expected_kv_read_tokens=[16, 16, 16],
    )

    assert injected == 0
    assert created_cache_salts == ["seed-0", "seed-1"]
    assert stub._bench_seq == 4
    assert stub._bench_active_req_ids == set()
    stub.add_request.assert_not_called()


def test_benchmark_output_marks_skipped_kv_point_invalid(tmp_path):
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=24,
        total_kv_read_tokens=16,
    )
    output_path = tmp_path / "benchmark.json"
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig(output_path=str(output_path))
    stub._bench_expected_points = 1
    stub._bench_results = []
    stub._bench_skipped_points = [
        SkippedBenchmarkPoint(point=point, reason="seed_cache_validation_failed")
    ]
    stub._bench_missing_phases = []
    stub.max_num_scheduled_tokens = 40
    stub.max_num_running_reqs = 8
    stub.max_model_len = 128
    stub.block_size = 8
    stub.cache_config = SimpleNamespace(num_gpu_blocks=64)

    InstrumentedScheduler._bench_write_results(stub)

    output = json.loads(output_path.read_text())
    assert output["schema_version"] == 2
    assert output["valid"] is False
    assert output["coverage"] == {
        "expected_points": 1,
        "completed_points": 0,
        "skipped_points": 1,
    }
    assert output["skipped_points"] == [
        {
            "point": point.__dict__,
            "kv_seed_regime": "not_applicable",
            "reason": "seed_cache_validation_failed",
        }
    ]
    assert output["missing_phases"] == []


def test_benchmark_timing_excludes_engine_startup_and_sums_measured_groups(
    monkeypatch, tmp_path
):
    timestamps = iter(["2026-07-10T12:00:00Z", "2026-07-10T12:00:09Z"])
    monotonic_times = iter([100.0, 109.0])
    monkeypatch.setattr(
        instrumented_scheduler_module, "_utc_now_rfc3339", lambda: next(timestamps)
    )
    monkeypatch.setattr(
        instrumented_scheduler_module.time,
        "monotonic",
        lambda: next(monotonic_times),
    )

    output_path = tmp_path / "benchmark.json"
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig(output_path=str(output_path))
    stub._bench_start_monotonic = None
    stub._bench_started_at = None
    stub._bench_completed_at = None
    stub._bench_elapsed_seconds = None
    stub._bench_expected_points = 0
    stub._bench_results = []
    stub._bench_skipped_points = []
    stub._bench_missing_phases = ["prefill"]
    stub._bench_iteration_groups = [{"wall_time": 1.25}, {"wall_time": 2.5}]
    stub.max_num_scheduled_tokens = 40
    stub.max_num_running_reqs = 8
    stub.max_model_len = 128
    stub.block_size = 8
    stub.cache_config = SimpleNamespace(num_gpu_blocks=64)

    InstrumentedScheduler._bench_start_timing(stub)
    InstrumentedScheduler._bench_write_results(stub)

    output = json.loads(output_path.read_text())
    assert output["timing"] == {
        "started_at": "2026-07-10T12:00:00Z",
        "completed_at": "2026-07-10T12:00:09Z",
        "benchmark_elapsed_seconds": 9.0,
        "measured_iteration_seconds": 3.75,
    }


def test_benchmark_soft_timeout_stops_after_saving_current_point(monkeypatch):
    point = BenchmarkPoint(
        point_type="decode",
        benchmark_id=1,
        total_kv_read_tokens=48,
        batch_size=3,
    )
    fpms = [
        {
            "scheduled_requests": {
                "num_decode_requests": 3,
                "sum_decode_kv_tokens": 48,
            }
        }
    ]
    stub = _benchmark_save_stub(point, fpms)
    stub._bench_config = BenchmarkConfig(timeout=1)
    stub._bench_start_monotonic = 0.0
    stub._bench_deadline_monotonic = 1.0
    stub._bench_expected_points = 2
    stub._bench_stop_requested = False
    stub._bench_stop_reason = None
    stub._bench_drain_pending = True
    stub._bench_phase = _BenchPhase.DECODE_SWEEP
    monkeypatch.setattr(instrumented_scheduler_module.time, "monotonic", lambda: 2.0)

    InstrumentedScheduler._bench_save_current_point(stub)

    assert len(stub._bench_results) == 1
    assert stub._bench_stop_requested is True
    assert stub._bench_stop_reason == "timeout"
    assert InstrumentedScheduler._bench_transition_to_timeout_done(stub) is True
    assert stub._bench_phase == _BenchPhase.DONE
    assert stub._bench_drain_pending is False


def test_benchmark_soft_timeout_is_checked_before_next_point(monkeypatch):
    completed_point = BenchmarkPoint(point_type="decode", benchmark_id=1, batch_size=1)
    next_point = BenchmarkPoint(point_type="decode", benchmark_id=2, batch_size=2)
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig(timeout=1)
    stub._bench_start_monotonic = 0.0
    stub._bench_deadline_monotonic = 1.0
    stub._bench_expected_points = 2
    stub._bench_results = [
        instrumented_scheduler_module.BenchmarkPointResult(
            point=completed_point, fpms=[]
        )
    ]
    stub._bench_skipped_points = []
    stub._bench_grid = deque([next_point])
    stub._bench_synchronizer = None
    stub._bench_stop_requested = False
    stub._bench_stop_reason = None
    stub._bench_drain_pending = False
    stub._bench_phase = _BenchPhase.DECODE_SWEEP
    monkeypatch.setattr(instrumented_scheduler_module.time, "monotonic", lambda: 2.0)

    assert InstrumentedScheduler._bench_stop_at_timeout_boundary(stub, "decode")

    assert list(stub._bench_grid) == [next_point]
    assert stub._bench_phase == _BenchPhase.DONE
    assert stub._bench_stop_reason == "timeout"


def test_benchmark_done_coordinates_cleanup_and_deactivates_before_publish():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_phase = _BenchPhase.DONE
    calls = MagicMock()
    stub._bench_start_timing = MagicMock()
    stub._bench_build_grid = MagicMock()
    stub._bench_clear_prefix_cache = MagicMock(return_value=True)  # nothing fenced
    stub._bench_synchronizer = MagicMock()
    stub._bench_finish_timing = MagicMock()
    stub._bench_deactivate = MagicMock()
    stub._bench_write_results = MagicMock()
    calls.attach_mock(stub._bench_clear_prefix_cache, "clear")
    calls.attach_mock(stub._bench_synchronizer.synchronize_cleanup, "sync_cleanup")
    calls.attach_mock(stub._bench_finish_timing, "finish")
    calls.attach_mock(stub._bench_deactivate, "deactivate")
    calls.attach_mock(stub._bench_write_results, "write")

    InstrumentedScheduler._bench_step(stub)

    assert calls.mock_calls == [
        call.clear(),
        call.sync_cleanup(),
        call.finish(),
        call.deactivate(),
        call.write(),
    ]


def test_benchmark_output_marks_timeout_result_partial_and_usable(tmp_path):
    point = BenchmarkPoint(point_type="decode", benchmark_id=1, batch_size=1)
    fpm = {"counter_id": 1, "dp_rank": 0, "wall_time": 0.25}
    output_path = tmp_path / "benchmark.json"
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig(output_path=str(output_path), timeout=1)
    stub._bench_expected_points = 2
    stub._bench_results = [
        instrumented_scheduler_module.BenchmarkPointResult(point=point, fpms=[fpm])
    ]
    stub._bench_iteration_groups = [
        {
            "benchmark_id": 1,
            "point": point.__dict__,
            "expected_dp_ranks": [0],
            "complete": True,
            "wall_time": 0.25,
            "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
        }
    ]
    stub._bench_skipped_points = []
    stub._bench_missing_phases = []
    stub._bench_stop_reason = "timeout"
    stub._bench_started_at = "2026-07-13T12:00:00Z"
    stub._bench_completed_at = "2026-07-13T12:00:01Z"
    stub._bench_start_monotonic = 0.0
    stub._bench_elapsed_seconds = 1.0
    stub.max_num_scheduled_tokens = 40
    stub.max_num_running_reqs = 8
    stub.max_model_len = 128
    stub.block_size = 8
    stub.cache_config = SimpleNamespace(num_gpu_blocks=64)

    InstrumentedScheduler._bench_write_results(stub)

    output = json.loads(output_path.read_text())
    assert output["status"] == "partial"
    assert output["valid"] is False
    assert output["usable"] is True
    assert output["stop_reason"] == "timeout"
    assert output["coverage"] == {
        "expected_points": 2,
        "completed_points": 1,
        "skipped_points": 0,
    }

    stub._bench_expected_points = 3
    stub._bench_skipped_points = [
        SkippedBenchmarkPoint(
            point=BenchmarkPoint(point_type="decode", benchmark_id=2),
            reason="shape mismatch",
        )
    ]
    InstrumentedScheduler._bench_write_results(stub)
    output_with_skip = json.loads(output_path.read_text())
    assert output_with_skip["status"] == "partial"
    assert output_with_skip["usable"] is False


def test_benchmark_output_marks_requested_empty_phase_invalid(tmp_path):
    output_path = tmp_path / "benchmark.json"
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig(mode="decode", output_path=str(output_path))
    stub._bench_expected_points = 0
    stub._bench_results = []
    stub._bench_skipped_points = []
    stub._bench_missing_phases = ["decode"]
    stub.max_num_scheduled_tokens = 40
    stub.max_num_running_reqs = 8
    stub.max_model_len = 8
    stub.block_size = 16
    stub.cache_config = SimpleNamespace(num_gpu_blocks=64)

    InstrumentedScheduler._bench_write_results(stub)

    output = json.loads(output_path.read_text())
    assert output["coverage"] == {
        "expected_points": 0,
        "completed_points": 0,
        "skipped_points": 0,
    }
    assert output["missing_phases"] == ["decode"]
    assert output["valid"] is False


def _benchmark_save_stub(point: BenchmarkPoint, fpms: list[dict]):
    for fpm in fpms:
        fpm.setdefault("counter_id", point.benchmark_id)
        fpm.setdefault("dp_rank", 0)
        fpm.setdefault("wall_time", 0.01)
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_current_point = point
    stub._bench_current_fpms = fpms
    stub._bench_expected_fpms = 1
    stub._bench_results = []
    stub._bench_iteration_groups = []
    stub._bench_skipped_points = []
    stub._bench_synchronizer = None
    stub._bench_dp_size = 1
    stub._fpm_dp_rank = 0
    return stub


def test_prefill_point_with_measured_kv_mismatch_is_skipped():
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=24,
        total_kv_read_tokens=16,
    )
    stub = _benchmark_save_stub(
        point,
        [
            {
                "scheduled_requests": {
                    "num_prefill_requests": 1,
                    "sum_prefill_tokens": 24,
                    "sum_prefill_kv_tokens": 8,
                }
            }
        ],
    )

    InstrumentedScheduler._bench_save_current_point(stub)

    assert stub._bench_results == []
    assert stub._bench_skipped_points == [
        SkippedBenchmarkPoint(point=point, reason="measured_kv_read_mismatch")
    ]


def test_prefill_point_with_exact_batch_shape_is_saved():
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=24,
        total_kv_read_tokens=48,
        batch_size=3,
    )
    fpms = [
        {
            "scheduled_requests": {
                "num_prefill_requests": 3,
                "sum_prefill_tokens": 24,
                "sum_prefill_kv_tokens": 48,
            }
        }
    ]
    stub = _benchmark_save_stub(point, fpms)

    InstrumentedScheduler._bench_save_current_point(stub)

    assert stub._bench_results == [
        instrumented_scheduler_module.BenchmarkPointResult(point=point, fpms=fpms)
    ]
    assert stub._bench_skipped_points == []
    assert stub._bench_iteration_groups == [
        {
            "benchmark_id": 0,
            "point": point.__dict__,
            "expected_dp_ranks": [0],
            "complete": True,
            "wall_time": 0.01,
            "rank_results": [{"dp_rank": 0, "fpms": fpms}],
        }
    ]


@pytest.mark.parametrize("fpm_count", [0, 2])
def test_benchmark_point_rejects_non_single_fpm_count(fpm_count):
    point = BenchmarkPoint(
        point_type="decode",
        benchmark_id=4,
        total_kv_read_tokens=48,
        batch_size=3,
    )
    fpm = {
        "scheduled_requests": {
            "num_decode_requests": 3,
            "sum_decode_kv_tokens": 48,
        }
    }
    stub = _benchmark_save_stub(point, [fpm.copy() for _ in range(fpm_count)])

    with pytest.raises(RuntimeError, match="exactly one FPM"):
        InstrumentedScheduler._bench_save_current_point(stub)


def test_decode_point_with_no_fpm_stops_waiting_at_deadline(monkeypatch):
    point = BenchmarkPoint(
        point_type="decode",
        benchmark_id=4,
        total_kv_read_tokens=48,
        batch_size=3,
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_drain_if_pending = MagicMock(return_value=False)
    stub._bench_active_req_ids = {"request"}
    stub._bench_current_point = point
    stub._bench_current_fpms = []
    stub._bench_point_deadline = 1.0
    stub._bench_save_current_point = MagicMock(
        side_effect=RuntimeError("exactly one FPM")
    )
    monkeypatch.setattr(instrumented_scheduler_module.time, "monotonic", lambda: 2.0)

    with pytest.raises(RuntimeError, match="exactly one FPM"):
        InstrumentedScheduler._bench_step_decode(stub)
    stub._bench_save_current_point.assert_called_once_with()


def test_prefill_point_with_measured_batch_size_mismatch_is_skipped():
    point = BenchmarkPoint(
        point_type="prefill",
        total_prefill_tokens=48,
        total_kv_read_tokens=32,
        batch_size=3,
    )
    stub = _benchmark_save_stub(
        point,
        [
            {
                "scheduled_requests": {
                    "num_prefill_requests": 2,
                    "sum_prefill_tokens": 48,
                    "sum_prefill_kv_tokens": 32,
                }
            }
        ],
    )

    InstrumentedScheduler._bench_save_current_point(stub)

    assert stub._bench_results == []
    assert stub._bench_skipped_points == [
        SkippedBenchmarkPoint(point=point, reason="measured_batch_size_mismatch")
    ]


def test_decode_point_with_exact_shape_is_saved():
    point = BenchmarkPoint(point_type="decode", total_kv_read_tokens=48, batch_size=3)
    fpms = [
        {
            "scheduled_requests": {
                "num_decode_requests": 3,
                "sum_decode_kv_tokens": 48,
            }
        }
    ]
    stub = _benchmark_save_stub(point, fpms)

    InstrumentedScheduler._bench_save_current_point(stub)

    assert stub._bench_results == [
        instrumented_scheduler_module.BenchmarkPointResult(point=point, fpms=fpms)
    ]
    assert stub._bench_skipped_points == []


def test_decode_point_with_measured_batch_size_mismatch_is_skipped():
    point = BenchmarkPoint(point_type="decode", total_kv_read_tokens=48, batch_size=3)
    stub = _benchmark_save_stub(
        point,
        [
            {
                "scheduled_requests": {
                    "num_decode_requests": 2,
                    "sum_decode_kv_tokens": 32,
                }
            }
        ],
    )

    InstrumentedScheduler._bench_save_current_point(stub)

    assert stub._bench_results == []
    assert stub._bench_skipped_points == [
        SkippedBenchmarkPoint(point=point, reason="measured_batch_size_mismatch")
    ]


def test_decode_point_with_measured_context_mismatch_is_skipped():
    point = BenchmarkPoint(point_type="decode", total_kv_read_tokens=48, batch_size=3)
    stub = _benchmark_save_stub(
        point,
        [
            {
                "scheduled_requests": {
                    "num_decode_requests": 3,
                    "sum_decode_kv_tokens": 47,
                }
            }
        ],
    )

    InstrumentedScheduler._bench_save_current_point(stub)

    assert stub._bench_results == []
    assert stub._bench_skipped_points == [
        SkippedBenchmarkPoint(point=point, reason="measured_decode_context_mismatch")
    ]


def test_zero_request_decode_injection_is_skipped_immediately():
    point = BenchmarkPoint(point_type="decode", total_kv_read_tokens=48, batch_size=3)
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_drain_if_pending = MagicMock(return_value=False)
    stub._bench_active_req_ids = set()
    stub._bench_grid = deque([point])
    stub._bench_current_point = None
    stub._bench_current_fpms = []
    stub._bench_skipped_points = []
    stub.deferred_frees = deque()  # nothing fenced
    stub._bench_cleanup_requests = MagicMock()
    stub._bench_inject_fake_decode = MagicMock(
        return_value=SimpleNamespace(total_num_scheduled_tokens=0)
    )

    output = InstrumentedScheduler._bench_step_decode(stub)

    assert output is None
    assert stub._bench_current_point is None
    # The fake-injection path stamps the KV seed regime before skipping.
    expected_point = replace(point, sample_reasons=["kvwarm_fake_fallback"])
    assert stub._bench_skipped_points == [
        SkippedBenchmarkPoint(point=expected_point, reason="decode_injection_failed")
    ]
    # The admission step is injected one token short of the coordinate.
    stub._bench_inject_fake_decode.assert_called_once_with([15, 15, 15])
    stub._bench_cleanup_requests.assert_called_once_with()


# ---------------------------------------------------------------------------
# Steady-state decode measurement (two-step points)
# ---------------------------------------------------------------------------


def _steady_injection_stub(point: BenchmarkPoint):
    """Populate only what the injection branch of ``_bench_step_decode``
    reads; the injection itself is captured."""
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_drain_pending = False
    stub._bench_active_req_ids = set()
    stub._bench_grid = deque([point])
    stub._bench_current_point = None
    stub._bench_current_fpms = []
    stub.deferred_frees = deque()  # nothing fenced
    stub._bench_stop_at_timeout_boundary = MagicMock(return_value=False)
    stub._bench_inject_fake_decode = MagicMock(
        return_value=SimpleNamespace(total_num_scheduled_tokens=point.batch_size)
    )
    return stub


def test_decode_injection_admits_one_token_short():
    point = BenchmarkPoint(
        point_type="decode", benchmark_id=5, total_kv_read_tokens=128, batch_size=2
    )
    stub = _steady_injection_stub(point)

    output = InstrumentedScheduler._bench_step_decode(stub)

    assert output is not None
    stub._bench_inject_fake_decode.assert_called_once_with([63, 63])
    assert stub._bench_admission_kv_tokens == 126
    assert stub._bench_extra_steps_left == 1
    assert stub._bench_expected_fpms == 2
    # No clamping: the point keeps its grid coordinate.
    assert stub._bench_current_point.total_kv_read_tokens == 128
    assert "context_clamped" not in stub._bench_current_point.sample_reasons
    assert stub._bench_sync_pending is True


def test_decode_ctx1_point_is_clamped_and_recorded_at_measured_coordinate():
    # total_kv == batch_size means every request targets a 1-token context,
    # which cannot give up a token for the admission step.
    point = BenchmarkPoint(
        point_type="decode", benchmark_id=6, total_kv_read_tokens=4, batch_size=4
    )
    stub = _steady_injection_stub(point)

    output = InstrumentedScheduler._bench_step_decode(stub)

    assert output is not None
    stub._bench_inject_fake_decode.assert_called_once_with([1, 1, 1, 1])
    assert stub._bench_admission_kv_tokens == 4
    # The steady step reads ctx=2 per request; the point is recorded at the
    # coordinate it actually measures.
    assert stub._bench_current_point.total_kv_read_tokens == 8
    assert "context_clamped" in stub._bench_current_point.sample_reasons
    assert stub._bench_current_point.benchmark_id == 6


def test_steady_step_dispatch_then_wait_then_save():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_drain_pending = False
    stub._bench_active_req_ids = {"__bench_0"}
    stub._bench_current_fpms = []
    stub._bench_extra_steps_left = 1
    stub._bench_expected_fpms = 2
    stub._bench_point_deadline = 0.0
    steady_output = object()
    stub._bench_make_steady_step = MagicMock(return_value=steady_output)

    assert InstrumentedScheduler._bench_step_decode(stub) is steady_output
    assert stub._bench_extra_steps_left == 0

    # Only the admission FPM has arrived: keep waiting, no save.
    stub._bench_save_current_point = MagicMock()
    stub._bench_current_fpms = [{"admission": True}]
    assert InstrumentedScheduler._bench_step_decode(stub) is None
    stub._bench_save_current_point.assert_not_called()

    # Second (steady) FPM arrived: save and enter drain.
    stub._bench_current_fpms = [{"admission": True}, {"steady": True}]
    stub._bench_cleanup_requests = MagicMock()
    stub._bench_transition_to_timeout_done = MagicMock(return_value=False)
    assert InstrumentedScheduler._bench_step_decode(stub) is None
    stub._bench_save_current_point.assert_called_once_with()
    assert stub._bench_drain_pending is True


def test_steady_step_unavailable_waits_out_the_deadline():
    import time

    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_drain_pending = False
    stub._bench_active_req_ids = {"__bench_0"}
    stub._bench_current_fpms = [{"admission": True}]
    stub._bench_extra_steps_left = 1
    stub._bench_expected_fpms = 2
    stub._bench_point_deadline = 0.0  # no deadline yet -> not timed out
    stub._bench_make_steady_step = MagicMock(return_value=None)
    stub._bench_save_current_point = MagicMock()

    # Builder failed: no dispatch, no save, state intact for a retry.
    assert InstrumentedScheduler._bench_step_decode(stub) is None
    stub._bench_save_current_point.assert_not_called()
    assert stub._bench_extra_steps_left == 1

    # Once the deadline passes, the point flows into the normal save path
    # (where the admission-only FPM fails shape validation and the group
    # skips together).
    stub._bench_point_deadline = time.monotonic() - 1.0
    stub._bench_cleanup_requests = MagicMock()
    stub._bench_transition_to_timeout_done = MagicMock(return_value=False)
    assert InstrumentedScheduler._bench_step_decode(stub) is None
    stub._bench_save_current_point.assert_called_once_with()


def test_make_steady_step_builds_production_shaped_output():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    requests = [
        SimpleNamespace(
            request_id=f"__bench_{index}",
            num_computed_tokens=63 + index,
            num_output_tokens=0,
            num_output_placeholders=1,
            is_finished=lambda: False,
        )
        for index in range(2)
    ]
    stub.running = requests
    stub._bench_active_req_ids = {r.request_id for r in requests}
    blocks = MagicMock()
    blocks.get_block_ids.return_value = None
    kv = MagicMock()
    kv.allocate_slots.return_value = blocks
    kv.num_kv_cache_groups = 1
    stub.kv_cache_manager = kv
    stub.num_lookahead_tokens = 0
    stub.needs_kv_cache_zeroing = False
    stub.finished_req_ids = set()
    stub.connector = None
    stub.ec_connector = None

    output = InstrumentedScheduler._bench_make_steady_step(stub)

    assert output.scheduled_new_reqs == []
    cached = output.scheduled_cached_reqs
    assert cached.req_ids == ["__bench_0", "__bench_1"]
    assert cached.resumed_req_ids == set()
    assert cached.new_token_ids == []
    assert cached.num_computed_tokens == [63, 64]
    assert cached.num_output_tokens == [1, 1]
    assert output.total_num_scheduled_tokens == 2
    assert output.num_scheduled_tokens == {"__bench_0": 1, "__bench_1": 1}
    kv.allocate_slots.assert_any_call(
        requests[0], 1, num_lookahead_tokens=0, delay_cache_blocks=True
    )
    blocks.get_block_ids.assert_called_with(allow_none=True)


def test_make_steady_step_returns_none_when_kv_exhausted():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    request = SimpleNamespace(
        request_id="__bench_0",
        num_computed_tokens=63,
        num_output_tokens=0,
        num_output_placeholders=1,
        is_finished=lambda: False,
    )
    stub.running = [request]
    stub._bench_active_req_ids = {"__bench_0"}
    kv = MagicMock()
    kv.allocate_slots.return_value = None
    stub.kv_cache_manager = kv
    stub.num_lookahead_tokens = 0

    assert InstrumentedScheduler._bench_make_steady_step(stub) is None


def _steady_fpm(sum_kv: int, batch: int, bench_id: int = 1, rank: int = 0) -> dict:
    return {
        "counter_id": bench_id,
        "dp_rank": rank,
        "wall_time": 1.0,
        "scheduled_requests": {
            "num_decode_requests": batch,
            "sum_decode_kv_tokens": sum_kv,
        },
    }


def test_save_records_only_the_steady_fpm():
    point = BenchmarkPoint(
        point_type="decode", benchmark_id=1, total_kv_read_tokens=128, batch_size=2
    )
    admission = _steady_fpm(126, 2)
    steady = _steady_fpm(128, 2)
    stub = _benchmark_save_stub(point, [admission, steady])
    stub._bench_expected_fpms = 2

    InstrumentedScheduler._bench_save_current_point(stub)

    assert len(stub._bench_results) == 1
    assert stub._bench_results[0].fpms == [steady]
    assert stub._bench_skipped_points == []


def test_save_admission_only_fpm_becomes_validation_skip():
    point = BenchmarkPoint(
        point_type="decode", benchmark_id=1, total_kv_read_tokens=128, batch_size=2
    )
    stub = _benchmark_save_stub(point, [_steady_fpm(126, 2)])
    stub._bench_expected_fpms = 2  # steady FPM never arrived

    InstrumentedScheduler._bench_save_current_point(stub)

    assert stub._bench_results == []
    assert stub._bench_skipped_points == [
        SkippedBenchmarkPoint(point=point, reason="measured_decode_context_mismatch")
    ]


@pytest.mark.timeout(60)
def test_two_step_group_skip_traverses_barrier_without_deadlock():
    """One rank misses its steady FPM while the other has both: the short
    rank must still enter collect_result so the whole group leaves the
    barrier together and skips the point consistently."""
    import time

    endpoint = f"inproc://benchmark-sync-{uuid.uuid4().hex}"
    rank0 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=0, dp_size=2, master_ip="unused", port=0, timeout=5, endpoint=endpoint
    )
    rank1 = instrumented_scheduler_module._BenchmarkSynchronizer(
        dp_rank=1, dp_size=2, master_ip="unused", port=0, timeout=5, endpoint=endpoint
    )
    point = BenchmarkPoint(
        point_type="decode", benchmark_id=1, total_kv_read_tokens=128, batch_size=2
    )

    def build(rank, synchronizer, fpms):
        stub = _benchmark_save_stub(point, fpms)
        stub._bench_expected_fpms = 2
        stub._bench_synchronizer = synchronizer
        stub._bench_dp_size = 2
        stub._fpm_dp_rank = rank
        stub._bench_deadline_monotonic = time.monotonic() + 30.0
        return stub

    # rank0 timed out with only the admission FPM; rank1 got both.
    stub0 = build(0, rank0, [_steady_fpm(126, 2, rank=0)])
    stub1 = build(1, rank1, [_steady_fpm(126, 2, rank=1), _steady_fpm(128, 2, rank=1)])

    errors: dict[int, Exception] = {}

    def save(rank, stub):
        try:
            InstrumentedScheduler._bench_save_current_point(stub)
        except Exception as error:  # pragma: no cover - failure diagnostics
            errors[rank] = error

    follower = threading.Thread(target=save, args=(1, stub1))
    follower.start()
    try:
        save(0, stub0)
        follower.join(timeout=10)
        assert not follower.is_alive(), "rank1 deadlocked in collect_result"
    finally:
        rank1.close()
        rank0.close()

    assert errors == {}
    for stub in (stub0, stub1):
        assert stub._bench_results == []
        assert [s.reason for s in stub._bench_skipped_points] == [
            "measured_decode_context_mismatch"
        ]


def test_steady_fpm_gate_only_fires_for_two_step_decode_points():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._last_update_time = 100.0
    stub._bench_current_point = BenchmarkPoint(point_type="decode")
    stub._bench_expected_fpms = 2
    stub._bench_current_fpms = [{"admission": True}]
    assert InstrumentedScheduler._bench_steady_fpm_expected(stub) is True

    stub._bench_current_fpms = []  # admission FPM not recorded yet
    assert InstrumentedScheduler._bench_steady_fpm_expected(stub) is False

    stub._bench_current_fpms = [{"admission": True}]
    stub._bench_expected_fpms = 1  # single-step point
    assert InstrumentedScheduler._bench_steady_fpm_expected(stub) is False

    stub._bench_expected_fpms = 2
    stub._bench_current_point = BenchmarkPoint(point_type="prefill")
    assert InstrumentedScheduler._bench_steady_fpm_expected(stub) is False

    stub._bench_current_point = BenchmarkPoint(point_type="decode")
    stub._last_update_time = 0.0  # previous update was an empty step
    assert InstrumentedScheduler._bench_steady_fpm_expected(stub) is False


# ---------------------------------------------------------------------------
# Synthetic-prompt randomization (salt-paired random token ids)
# ---------------------------------------------------------------------------


def _vocab_stub(vocab_size: int, dp_rank: int = 0):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_vocab_size = vocab_size
    stub._fpm_dp_rank = dp_rank
    return stub


def test_synthetic_token_ids_are_salt_deterministic_and_in_vocab():
    stub = _vocab_stub(151_000)

    first = InstrumentedScheduler._bench_synthetic_token_ids(stub, "salt-a", 64)
    second = InstrumentedScheduler._bench_synthetic_token_ids(stub, "salt-a", 64)
    other = InstrumentedScheduler._bench_synthetic_token_ids(stub, "salt-b", 64)

    assert first == second, "same salt must reproduce the same sequence"
    assert first != other
    assert all(1 <= token < 151_000 for token in first)
    assert len(set(first)) > 1, "prompts must not be constant"


def test_synthetic_token_ids_prefix_stability_pairs_seed_with_request():
    """The fake prefix-cache pairing invariant: the seed request draws
    ``prefix_tokens`` ids and the measuring request draws its full prompt
    from the SAME salt, so the seed must be a strict prefix of the prompt --
    otherwise the block hashes cannot match and every kv>0 point dies with
    ``fake_prefix_cache_validation_failed``."""
    stub = _vocab_stub(151_000)

    seed = InstrumentedScheduler._bench_synthetic_token_ids(stub, "pair-salt", 48)
    prompt = InstrumentedScheduler._bench_synthetic_token_ids(stub, "pair-salt", 320)

    assert prompt[:48] == seed


def test_synthetic_token_ids_decorrelate_across_dp_ranks():
    """Lockstep keeps salts identical on every attention-DP rank; the rank
    must be mixed into the seed so DEP expert routing is not fed the same
    token stream N times over."""
    rank0 = _vocab_stub(151_000, dp_rank=0)
    rank1 = _vocab_stub(151_000, dp_rank=1)

    tokens0 = InstrumentedScheduler._bench_synthetic_token_ids(rank0, "same", 64)
    tokens1 = InstrumentedScheduler._bench_synthetic_token_ids(rank1, "same", 64)

    assert tokens0 != tokens1


def test_synthetic_token_ids_fall_back_to_zeros_without_vocab():
    for vocab_size in (0, 1):
        stub = _vocab_stub(vocab_size)
        tokens = InstrumentedScheduler._bench_synthetic_token_ids(stub, "s", 8)
        assert tokens == [0] * 8
    # Stubs without the attribute (legacy construction paths) also fall back.
    bare = InstrumentedScheduler.__new__(InstrumentedScheduler)
    assert InstrumentedScheduler._bench_synthetic_token_ids(bare, "s", 4) == [0] * 4


def test_seed_and_measuring_request_share_block_hashes():
    """End-to-end pairing through vLLM's real block hasher: the seeded
    prefix request and the (longer) measuring request must produce identical
    hashes for every full prefix block, with the same cache salt."""
    block_size = 16
    caching_hash_fn = instrumented_scheduler_module.get_hash_fn_by_name("sha256")
    instrumented_scheduler_module.init_none_hash(caching_hash_fn)
    hasher = instrumented_scheduler_module.get_request_block_hasher(
        block_size, caching_hash_fn
    )
    stub = _vocab_stub(151_000)
    salt = "block-hash-salt"
    prefix_tokens = 64  # 4 full blocks
    prompt_len = 128

    seed_req = instrumented_scheduler_module.Request(
        request_id="__bench_fake_prefix_0",
        prompt_token_ids=InstrumentedScheduler._bench_synthetic_token_ids(
            stub, salt, prefix_tokens
        ),
        sampling_params=instrumented_scheduler_module.SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=hasher,
        cache_salt=salt,
    )
    measuring_req = instrumented_scheduler_module.Request(
        request_id="__bench_0",
        prompt_token_ids=InstrumentedScheduler._bench_synthetic_token_ids(
            stub, salt, prompt_len
        ),
        sampling_params=instrumented_scheduler_module.SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=hasher,
        cache_salt=salt,
    )

    seed_hashes = list(seed_req.block_hashes)
    measuring_hashes = list(measuring_req.block_hashes)
    assert len(seed_hashes) == prefix_tokens // block_size
    assert measuring_hashes[: len(seed_hashes)] == seed_hashes


def test_kvwarm_point_need_is_one_plus_repeats(monkeypatch):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    # Admission writes at the injected length, then one steady write per
    # repeated step: default repeats 3 -> need 4. The figure does not depend
    # on the point, because every real-KV point runs the repeats.
    assert InstrumentedScheduler._kvwarm_point_need(stub) == 4
    monkeypatch.setenv("DYN_BENCH_GIANT_KV_REPEATS", "5")
    assert InstrumentedScheduler._kvwarm_point_need(stub) == 6
    # Repeats floor at one steady step: two positions at minimum.
    monkeypatch.setenv("DYN_BENCH_GIANT_KV_REPEATS", "0")
    assert InstrumentedScheduler._kvwarm_point_need(stub) == 2


def test_kvwarm_covers_requires_ready_chains_deep_enough():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._kvwarm_plan = {2: 100}
    stub._kvwarm_building = False
    stub._kvwarm_stage_batch = 2
    stub._kvwarm_chain_ids = ["c0", "c1"]
    stub._kvwarm_chain_prompts = {"c0": [1] * 50, "c1": [1] * 50}
    point = SimpleNamespace(batch_size=2, total_kv_read_tokens=10_000)
    # injected + need (1 + default repeats 3) must fit inside every chain's
    # prompt depth.
    assert InstrumentedScheduler._kvwarm_covers(stub, point, [46, 46])
    assert not InstrumentedScheduler._kvwarm_covers(stub, point, [47, 46])
    # A fleet still under construction never covers.
    stub._kvwarm_building = True
    assert not InstrumentedScheduler._kvwarm_covers(stub, point, [10, 10])
    stub._kvwarm_building = False
    # Fewer live chains than the point's batch never covers.
    wide = SimpleNamespace(batch_size=3, total_kv_read_tokens=10_000)
    assert not InstrumentedScheduler._kvwarm_covers(stub, wide, [10, 10, 10])


def _kvwarm_text_stub():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_grid_digest = "grid-digest"
    stub._fpm_dp_rank = 0
    stub._kvwarm_load_texts = lambda: ["alpha", "bravo", "charlie", "delta"]
    stub._kvwarm_tokenizer = lambda: SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [ord(c) for c in text]
    )
    return stub


def test_kvwarm_chain_tokens_are_deterministic_and_extend_monotonically():
    first = InstrumentedScheduler._kvwarm_chain_token_ids(_kvwarm_text_stub(), 1, 6)
    again = InstrumentedScheduler._kvwarm_chain_token_ids(_kvwarm_text_stub(), 1, 6)
    assert first == again and len(first) >= 6
    # Deepening the same chain only extends: the shallow draw stays a strict
    # prefix (prefix-cache generational extension depends on this).
    stub = _kvwarm_text_stub()
    shallow = InstrumentedScheduler._kvwarm_chain_token_ids(stub, 2, 4)
    deep = InstrumentedScheduler._kvwarm_chain_token_ids(stub, 2, 12)
    assert deep[: len(shallow)] == shallow
    other = InstrumentedScheduler._kvwarm_chain_token_ids(_kvwarm_text_stub(), 3, 6)
    assert other != first


def test_synthetic_content_pool_windows_share_prefix_across_lengths(monkeypatch):
    monkeypatch.setenv("DYN_BENCH_PREFILL_CONTENT", "sharegpt")
    monkeypatch.delenv("DYN_BENCH_POOL_TAG", raising=False)
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_vocab_size = 1_000
    stub._fpm_dp_rank = 0
    stub._bench_prefill_pool = list(range(1, 1_001))
    short = InstrumentedScheduler._bench_synthetic_token_ids(stub, "s", 48)
    long = InstrumentedScheduler._bench_synthetic_token_ids(stub, "s", 320)
    # Window start depends only on the seed: any two lengths are strict
    # prefixes -- the invariant the fake prefix-cache pairing relies on.
    assert long[:48] == short
    # Wrap-around past the pool end preserves both length and the invariant.
    wrapped = InstrumentedScheduler._bench_synthetic_token_ids(stub, "s", 1_500)
    assert len(wrapped) == 1_500 and wrapped[:320] == long


def test_synthetic_content_chain_mode_routes_through_kvwarm_chains(monkeypatch):
    monkeypatch.setenv("DYN_BENCH_PREFILL_CONTENT", "sharegpt_chain")
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_vocab_size = 1_000
    stub._fpm_dp_rank = 0
    seen = []
    stub._kvwarm_chain_token_ids = (
        lambda idx, depth: seen.append((idx, depth)) or [7] * depth
    )
    out = InstrumentedScheduler._bench_synthetic_token_ids(stub, "salt", 9)
    assert out == [7] * 9
    ((idx, depth),) = seen
    # Derived chain indices stay clear of the grid's own chain range and are
    # deterministic per seed.
    assert depth == 9 and 20_000_000 <= idx < 21_000_000
    InstrumentedScheduler._bench_synthetic_token_ids(stub, "salt", 9)
    assert seen[1] == (idx, 9)


def test_content_pool_is_built_once_and_guards_small_datasets(monkeypatch):
    monkeypatch.setenv("DYN_BENCH_PREFILL_CONTENT", "sharegpt")
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_vocab_size = 1_000
    stub._fpm_dp_rank = 0
    calls = []
    stub._kvwarm_load_texts = lambda: calls.append(1) or ["x" * 5_000]
    stub._kvwarm_tokenizer = lambda: SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [1] * len(text)
    )
    InstrumentedScheduler._bench_synthetic_token_ids(stub, "a", 16)
    InstrumentedScheduler._bench_synthetic_token_ids(stub, "b", 16)
    assert calls == [1]
    tiny = InstrumentedScheduler.__new__(InstrumentedScheduler)
    tiny._bench_vocab_size = 1_000
    tiny._fpm_dp_rank = 0
    tiny._kvwarm_load_texts = lambda: ["too small"]
    tiny._kvwarm_tokenizer = lambda: SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [1] * len(text)
    )
    with pytest.raises(RuntimeError, match="pool too small"):
        InstrumentedScheduler._bench_synthetic_token_ids(tiny, "a", 16)


def test_kvwarm_decode_reorder_pins_warmup_replicas_first():
    real = [
        BenchmarkPoint(point_type="decode", total_kv_read_tokens=kv, batch_size=b)
        for b, kv in ((8, 128), (8, 4096), (16, 256))
    ]
    replica = replace(real[0], sample_reasons=["eager_warmup"])
    ordered = InstrumentedScheduler._kvwarm_order_decode_points([*real, replica])
    assert ordered[0] is replica
    assert [(p.batch_size, p.total_kv_read_tokens) for p in ordered[1:]] == [
        (16, 256),
        (8, 4096),
        (8, 128),
    ]


def test_eager_warmup_points_dedupe_and_flag():
    EAGER_WARMUP_REASON = instrumented_scheduler_module.EAGER_WARMUP_REASON
    grid = [
        BenchmarkPoint(
            point_type="decode",
            batch_size=513,
            total_kv_read_tokens=513,
            expected_capture_size=None,
        ),
        BenchmarkPoint(
            point_type="decode",
            batch_size=513,
            total_kv_read_tokens=2048,
            expected_capture_size=None,
        ),
        BenchmarkPoint(
            point_type="decode",
            batch_size=512,
            total_kv_read_tokens=512,
            expected_capture_size=512,
        ),
        BenchmarkPoint(
            point_type="prefill",
            batch_size=1,
            total_prefill_tokens=1024,
            total_kv_read_tokens=0,
            expected_capture_size=None,
        ),
        BenchmarkPoint(
            point_type="prefill",
            batch_size=2,
            total_prefill_tokens=1024,
            total_kv_read_tokens=4096,
            expected_capture_size=None,
        ),
        BenchmarkPoint(
            point_type="prefill",
            batch_size=1,
            total_prefill_tokens=256,
            total_kv_read_tokens=0,
            expected_capture_size=256,
        ),
    ]
    stub = SimpleNamespace(_bench_grid=grid)
    replicas = InstrumentedScheduler._bench_eager_warmup_points(stub)

    assert [(p.point_type, p.batch_size, p.total_prefill_tokens) for p in replicas] == [
        ("decode", 513, 0),
        ("prefill", 1, 1024),
    ]
    assert all(p.sample_reasons == [EAGER_WARMUP_REASON] for p in replicas)
    # originals untouched
    assert all(EAGER_WARMUP_REASON not in p.sample_reasons for p in grid)


def test_warmup_replica_with_failed_validation_is_discarded_not_skipped():
    """A warmup replica whose FPM fails shape validation must be discarded:
    recording it in skipped_points would mark the whole artifact unusable
    even though every real measurement succeeded."""
    point = BenchmarkPoint(
        point_type="decode",
        benchmark_id=9,
        total_kv_read_tokens=48,
        batch_size=3,
        sample_reasons=[instrumented_scheduler_module.EAGER_WARMUP_REASON],
    )
    stub = _benchmark_save_stub(
        point,
        [
            {
                "scheduled_requests": {
                    "num_decode_requests": 3,
                    "sum_decode_kv_tokens": 47,  # mismatch -> validation failure
                }
            }
        ],
    )

    InstrumentedScheduler._bench_save_current_point(stub)

    assert stub._bench_results == []
    assert stub._bench_skipped_points == []
    assert stub._bench_current_point is None


def test_warmup_replica_injection_failure_is_discarded_not_skipped():
    """A warmup replica that dies BEFORE producing an FPM (decode injection
    shortfall) must also stay out of skipped-point accounting: every
    ``_bench_skip_point`` caller shares the central warmup exemption."""
    point = BenchmarkPoint(
        point_type="decode",
        benchmark_id=9,
        total_kv_read_tokens=48,
        batch_size=3,
        sample_reasons=[instrumented_scheduler_module.EAGER_WARMUP_REASON],
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_drain_if_pending = MagicMock(return_value=False)
    stub._bench_active_req_ids = set()
    stub._bench_grid = deque([point])
    stub._bench_current_point = None
    stub._bench_current_fpms = []
    stub._bench_skipped_points = []
    stub.deferred_frees = deque()  # nothing fenced
    stub._bench_cleanup_requests = MagicMock()
    stub._bench_inject_fake_decode = MagicMock(
        return_value=SimpleNamespace(total_num_scheduled_tokens=0)
    )

    output = InstrumentedScheduler._bench_step_decode(stub)

    assert output is None
    assert stub._bench_current_point is None
    assert stub._bench_skipped_points == []
    stub._bench_cleanup_requests.assert_called_once_with()


def test_skip_point_exempts_warmup_replicas_on_every_path():
    """The exemption lives in ``_bench_skip_point`` itself, so fake-prefix,
    injection, and validation failures are all covered; real points still
    book a skipped entry."""
    warmup = BenchmarkPoint(
        point_type="prefill",
        benchmark_id=1,
        total_prefill_tokens=64,
        batch_size=1,
        sample_reasons=[instrumented_scheduler_module.EAGER_WARMUP_REASON],
    )
    real = BenchmarkPoint(
        point_type="prefill",
        benchmark_id=2,
        total_prefill_tokens=64,
        batch_size=1,
    )
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_skipped_points = []

    for reason in (
        "fake_prefix_cache_allocation_failed",
        "prefill_injection_failed",
        "measured_batch_size_mismatch",
    ):
        InstrumentedScheduler._bench_skip_point(stub, warmup, reason)
    assert stub._bench_skipped_points == []

    InstrumentedScheduler._bench_skip_point(stub, real, "prefill_injection_failed")
    assert stub._bench_skipped_points == [
        SkippedBenchmarkPoint(point=real, reason="prefill_injection_failed")
    ]


# ---------------------------------------------------------------------------
# KV warm-up lifecycle hardening (review follow-ups)
# ---------------------------------------------------------------------------


def _kvwarm_no_dataset_resolution():
    raise AssertionError("the gate test must not resolve the real dataset")


def _kvwarm_gate_stub(*, state_groups=(), experts=8, ep=True, prefix=True):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=ep),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_experts=experts), hf_text_config=None
        ),
    )
    stub.cache_config = SimpleNamespace(enable_prefix_caching=prefix)
    groups = [
        SimpleNamespace(kv_cache_spec=type(name, (), {})()) for name in state_groups
    ]
    stub.kv_cache_manager = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups)
    )
    # Host-local inputs the gate proves: a pool that holds one chain at the
    # depth cap (max_model_len - 4 = 60 tokens) and a tokenizer. The dataset
    # itself is never resolved here (that would download it); tests that
    # exercise the real loader point it at a temporary dump.
    stub.max_model_len = 64
    stub._kvwarm_resolve_dataset = _kvwarm_no_dataset_resolution
    stub._kvwarm_load_texts = lambda: ["alpha " * 20, "bravo " * 20]
    stub._kvwarm_tokenizer = lambda: SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [ord(c) for c in text]
    )
    return stub


def _kvwarm_sharegpt_file(tmp_path, bodies):
    """Write a ShareGPT-shaped dump whose conversations all land in the
    collection half of the hash split (``_kvwarm_load_texts`` keeps only
    bodies whose sha256 first byte is even)."""
    items = []
    for body in bodies:
        assert hashlib.sha256(body.encode()).digest()[0] % 2 == 0, body
        items.append({"conversations": [{"from": "human", "value": body}]})
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return str(path)


def _kvwarm_collection_bodies(count, length=80):
    bodies = []
    seed = 0
    while len(bodies) < count:
        body = (f"conversation {seed} " * length)[:length]
        seed += 1
        if hashlib.sha256(body.encode()).digest()[0] % 2 == 0:
            bodies.append(body)
    return bodies


def test_kvwarm_gate_rejects_recurrent_state_layers(monkeypatch):
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    stub = _kvwarm_gate_stub(state_groups=("FullAttentionSpec", "MambaSpec"))
    assert InstrumentedScheduler._kvwarm_warm_eligible(stub) is False
    assert stub._kvwarm_meta["skip_reason"] == "hybrid_state_layers_unsupported"


def test_kvwarm_gate_admits_pure_attention_moe_ep(monkeypatch):
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    stub = _kvwarm_gate_stub(state_groups=("FullAttentionSpec",))
    assert InstrumentedScheduler._kvwarm_warm_eligible(stub) is True


def test_kvwarm_gate_disables_warmup_when_dataset_unavailable(monkeypatch):
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    stub = _kvwarm_gate_stub()

    def _boom():
        raise RuntimeError("no egress")

    stub._kvwarm_load_texts = _boom
    assert InstrumentedScheduler._kvwarm_warm_eligible(stub) is False
    assert stub._kvwarm_meta["skip_reason"] == "dataset_unavailable: no egress"


def test_kvwarm_gate_proves_every_host_local_input_up_front(monkeypatch):
    """Dataset content, tokenizer construction and pool depth are decided at
    eligibility time, where the verdict still travels in the capacity
    envelope, instead of surfacing as an exception from the first stage
    build on one rank."""
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    empty = _kvwarm_gate_stub()
    empty._kvwarm_load_texts = lambda: []
    assert InstrumentedScheduler._kvwarm_warm_eligible(empty) is False
    assert empty._kvwarm_meta["skip_reason"] == "dataset_empty"

    no_tokenizer = _kvwarm_gate_stub()

    def _missing():
        raise OSError("tokenizer files missing")

    no_tokenizer._kvwarm_tokenizer = _missing
    assert InstrumentedScheduler._kvwarm_warm_eligible(no_tokenizer) is False
    assert no_tokenizer._kvwarm_meta["skip_reason"] == (
        "tokenizer_unavailable: tokenizer files missing"
    )

    # A chain draws every conversation at most once, so the pool must hold
    # one chain at the depth cap: 240 tokens cannot reach 1024 - 4.
    shallow = _kvwarm_gate_stub()
    shallow.max_model_len = 1024
    assert InstrumentedScheduler._kvwarm_warm_eligible(shallow) is False
    assert shallow._kvwarm_meta["skip_reason"] == (
        "content_too_shallow: 240 tokens < 1020"
    )


def test_kvwarm_gate_probe_parses_once_and_stops_at_the_depth_cap(
    monkeypatch, tmp_path
):
    """The gate parses the dataset through the real loader (cached for the
    stage builds, never parsed twice) and tokenizes only until the pool is
    shown to hold one chain at the cap."""
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    bodies = _kvwarm_collection_bodies(6)
    path = _kvwarm_sharegpt_file(tmp_path, bodies)
    stub = _kvwarm_gate_stub()
    stub.max_model_len = 4 + 2 * len(bodies[0]) + 1  # cap needs three bodies
    stub._kvwarm_resolve_dataset = lambda: path
    del stub._kvwarm_load_texts
    encoded = []

    def _encode(text, add_special_tokens=False):
        encoded.append(text)
        return [1] * len(text)

    stub._kvwarm_tokenizer = lambda: SimpleNamespace(encode=_encode)
    assert InstrumentedScheduler._kvwarm_warm_eligible(stub) is True
    assert encoded == bodies[:3]
    meta = stub._kvwarm_meta
    assert meta["skip_reason"] is None
    assert meta["dataset"]["conversations"] == len(bodies)
    assert meta["dataset"]["path"] == path
    # Cached: the stage builds reuse the parsed pool without touching the
    # file again.
    (tmp_path / "sharegpt.json").unlink()
    assert InstrumentedScheduler._kvwarm_load_texts(stub) == bodies
    assert stub._kvwarm_texts == bodies


def test_kvwarm_gate_reads_an_empty_dump_as_dataset_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    stub = _kvwarm_gate_stub()
    stub._kvwarm_resolve_dataset = lambda: str(path)
    del stub._kvwarm_load_texts
    assert InstrumentedScheduler._kvwarm_warm_eligible(stub) is False
    assert stub._kvwarm_meta["skip_reason"] == "dataset_empty"


def test_kvwarm_seed_regime_vocabulary(monkeypatch):
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._kvwarm_meta = {"warm_eligible": True, "skip_reason": None}
    decode = BenchmarkPoint(point_type="decode", total_kv_read_tokens=64, batch_size=2)
    prefill = BenchmarkPoint(
        point_type="prefill", total_prefill_tokens=64, batch_size=1
    )
    regime = InstrumentedScheduler._kvwarm_seed_regime
    assert regime(stub, prefill) == "not_applicable"
    assert regime(stub, decode) == "unstamped"
    assert regime(stub, replace(decode, sample_reasons=["kvwarm_real_kv"])) == "real_kv"
    assert (
        regime(stub, replace(decode, sample_reasons=["kvwarm_fake_fallback"]))
        == "fake_fallback"
    )
    stub._kvwarm_meta = {
        "warm_eligible": False,
        "skip_reason": "prefix_caching_disabled",
    }
    assert regime(stub, decode) == "skip:prefix_caching_disabled"
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "off")
    assert regime(stub, decode) == "legacy"


def test_kvwarm_release_heavy_state_drops_benchmark_only_memory():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._kvwarm_chain_ids = []
    stub._kvwarm_texts = ["x"] * 10
    stub._kvwarm_tok = object()
    stub._kvwarm_token_cache = {1: ([1, 2], 0, [0])}
    stub._kvwarm_chain_prompts = {"c": [1]}
    stub._bench_prefill_pool = [1] * 100
    InstrumentedScheduler._kvwarm_release_heavy_state(stub)
    assert stub._kvwarm_texts is None
    assert stub._kvwarm_tok is None
    assert stub._kvwarm_token_cache is None
    assert stub._bench_prefill_pool is None


def test_kvwarm_partial_chain_loss_fails_the_stage():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._kvwarm_chain_ids = ["chain-a", "chain-b"]
    stub._kvwarm_chain_prompts = {"chain-a": [1] * 8, "chain-b": [1] * 8}
    stub.requests = {
        "chain-b": SimpleNamespace(request_id="chain-b", num_computed_tokens=2)
    }
    stub.running = []
    stub._kvwarm_plan = {4: 64}
    stub._kvwarm_stage_batch = 4
    stub._kvwarm_building = True
    stub._bench_synchronizer = None
    meta = {"stages": []}
    stub._kvwarm_meta_init = lambda: meta
    stub._kvwarm_shed_chains = MagicMock()

    assert InstrumentedScheduler._kvwarm_monitor_build(stub) is True
    assert stub._kvwarm_plan[4] == 0
    stub._kvwarm_shed_chains.assert_called_once_with()
    assert meta["stages"] == [{"batch": 4, "failed": True, "vanished": 1}]


def test_capacity_envelope_folds_kvwarm_eligibility():
    Envelope = instrumented_scheduler_module._BenchmarkCapacityEnvelope
    base = dict(
        max_model_len=1024,
        max_num_scheduled_tokens=512,
        max_num_running_reqs=64,
        usable_blocks_without_watermark=100,
        usable_blocks_with_watermark=90,
        grid_invariants_digest="0" * 64,
    )
    eligible = Envelope(**base, kvwarm_eligible=True)
    ineligible = Envelope(**base, kvwarm_eligible=False)
    assert Envelope.common([eligible, eligible]).kvwarm_eligible is True
    assert Envelope.common([eligible, ineligible]).kvwarm_eligible is False
    # Older peers that do not report the field are treated as eligible.
    assert Envelope.from_dict(base).kvwarm_eligible is True
    with pytest.raises(RuntimeError, match="kvwarm_eligible"):
        Envelope.from_dict({**base, "kvwarm_eligible": "yes"})


def test_kvwarm_prepare_follows_peer_ineligibility(monkeypatch):
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    meta = {"warm_eligible": True, "skip_reason": None}
    stub._kvwarm_meta_init = lambda: meta
    stub._kvwarm_warm_eligible = lambda: True
    stub._bench_negotiated_capacity = SimpleNamespace(kvwarm_eligible=False)
    stub._bench_grid = deque()
    InstrumentedScheduler._kvwarm_prepare(stub, "decode")
    assert meta["skip_reason"] == "peer_ineligible"
    assert stub._kvwarm_eligible_cache is False


def _kvwarm_planner_stub(usable_blocks, groups=1, block_size=16):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    meta = {"warm_eligible": True, "skip_reason": None}
    stub._kvwarm_meta_init = lambda: meta
    stub._kvwarm_warm_eligible = lambda: True
    stub._bench_negotiated_capacity = None
    stub.max_model_len = 8192
    stub.cache_config = SimpleNamespace(block_size=block_size)
    stub.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[object()] * groups)
    )
    stub._bench_blocks_per_req = lambda depth, **_: groups * -(-depth // block_size)
    stub._bench_usable_blocks = lambda batch, reserve_watermark=False: usable_blocks
    return stub


def test_kvwarm_shadow_tail_blocks_worst_case():
    stub = _kvwarm_planner_stub(usable_blocks=0)
    # ctx one slot short of a boundary: 1 block for the admission write plus
    # ceil(headroom / block_size) for the steady steps.
    assert InstrumentedScheduler._kvwarm_shadow_tail_blocks(stub, 3) == 2
    assert InstrumentedScheduler._kvwarm_shadow_tail_blocks(stub, 1) == 2  # floor 2
    assert InstrumentedScheduler._kvwarm_shadow_tail_blocks(stub, 20) == 3
    two = _kvwarm_planner_stub(usable_blocks=0, groups=2)
    assert InstrumentedScheduler._kvwarm_shadow_tail_blocks(two, 3) == 4


def test_kvwarm_prepare_reserves_shadow_tail_blocks(monkeypatch):
    """A rung whose chains exactly fill the pool must leave room for the
    shadows' private tail blocks, or injection dies with
    "Cannot get N free blocks from the pool" (seen at batch=1024 on B200)."""
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    monkeypatch.setenv("DYN_BENCH_GIANT_KV_REPEATS", "3")
    batch, ctx = 4, 1000
    want = ctx + 1 + 3  # max(ctx) + 1 + repeats of steady-write headroom
    chain_blocks = -(-want // 16) * batch  # 63 blocks per chain, 252 total
    stub = _kvwarm_planner_stub(usable_blocks=chain_blocks)
    stub._bench_grid = deque(
        [
            BenchmarkPoint(
                point_type="decode",
                total_kv_read_tokens=batch * ctx,
                batch_size=batch,
            )
        ]
    )
    InstrumentedScheduler._kvwarm_prepare(stub, "decode")
    depth = stub._kvwarm_plan[batch]
    tail = InstrumentedScheduler._kvwarm_shadow_tail_blocks(stub, 3)
    assert tail == 2
    assert depth > 8
    assert depth < want, "chains alone filling the pool must be trimmed"
    assert (stub._bench_blocks_per_req(depth) + tail) * batch <= chain_blocks
    # The trimmed stage can no longer serve the point: it falls back to fake
    # injection instead of crashing the run.
    point = stub._bench_grid[-1]
    assert not InstrumentedScheduler._kvwarm_plan_covers(stub, point)


def test_kvwarm_plan_trims_depth_by_the_negotiated_pool(monkeypatch):
    """Every rank must derive the same plan, so the pool that trims a rung's
    depth is the group's negotiated figure, not this rank's own."""
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    monkeypatch.setenv("DYN_BENCH_GIANT_KV_REPEATS", "3")
    batch, ctx = 4, 1000
    want = ctx + 1 + 3
    chain_blocks = -(-want // 16) * batch
    stub = _kvwarm_planner_stub(usable_blocks=10**6)  # this rank: plenty
    stub._bench_negotiated_capacity = _benchmark_capacity(
        max_model_len=8192,
        usable_blocks_without_watermark=chain_blocks,
        usable_blocks_with_watermark=chain_blocks,
    )
    point = BenchmarkPoint(
        point_type="decode", total_kv_read_tokens=batch * ctx, batch_size=batch
    )
    stub._bench_grid = deque([point])
    InstrumentedScheduler._kvwarm_prepare(stub, "decode")
    depth = stub._kvwarm_plan[batch]
    tail = InstrumentedScheduler._kvwarm_shadow_tail_blocks(stub, 3)
    assert depth < want
    assert (stub._bench_blocks_per_req(depth) + tail) * batch <= chain_blocks
    # Without a negotiated envelope the local pool applies and the rung fits.
    local = _kvwarm_planner_stub(usable_blocks=10**6)
    local._bench_grid = deque([point])
    InstrumentedScheduler._kvwarm_prepare(local, "decode")
    assert local._kvwarm_plan[batch] == want


def test_kvwarm_plan_depth_cap_follows_the_negotiated_model_length(monkeypatch):
    """The plan caps chain depth by the group's model length, not this
    rank's, so every rank plans the same rungs; the gate's content probe
    uses the same cap (``_kvwarm_depth_cap``)."""
    monkeypatch.setenv("DYN_BENCH_KV_WARMUP", "on")
    stub = _kvwarm_planner_stub(usable_blocks=10**6)
    stub.max_model_len = 8192
    stub._bench_negotiated_capacity = _benchmark_capacity(max_model_len=128)
    point = BenchmarkPoint(
        point_type="decode", total_kv_read_tokens=1_000, batch_size=1
    )
    stub._bench_grid = deque([point])
    assert InstrumentedScheduler._kvwarm_depth_cap(stub) == 124
    InstrumentedScheduler._kvwarm_prepare(stub, "decode")
    assert stub._kvwarm_plan[1] == 124
    stub._bench_negotiated_capacity = None
    assert InstrumentedScheduler._kvwarm_depth_cap(stub) == 8188


@pytest.mark.parametrize("ctx", [12, 13, 14, 15])
def test_kvwarm_plan_depth_covers_the_shadow_span_at_block_boundaries(ctx):
    """The plan margin, ``_kvwarm_point_need`` and the block check in
    ``_kvwarm_register_shadow`` must reserve the same ``injected + 1 +
    repeats`` span for every real-KV point (all of them run the repeated
    steady steps). A plan that budgeted a single steady step for non-giant
    points let ``_kvwarm_covers`` accept the rung's deepest point while its
    shadow needed one block more than the chain held whenever the chain
    depth landed on a block boundary -- a fatal 'too shallow' mid-run."""
    block_size = 16
    stub = _kvwarm_planner_stub(usable_blocks=10**6, block_size=block_size)
    # Default (non-giant) settings from the autouse fixture: repeats 3.
    repeats = InstrumentedScheduler._kvwarm_giant_repeats(stub)
    assert repeats == 3
    # batch=1: the point's context is ctx + 1 and it is admitted at ctx.
    point = BenchmarkPoint(
        point_type="decode", total_kv_read_tokens=ctx + 1, batch_size=1
    )
    stub._bench_grid = deque([point])
    InstrumentedScheduler._kvwarm_prepare(stub, "decode")
    depth = stub._kvwarm_plan[1]
    assert depth >= ctx + 1 + repeats
    assert InstrumentedScheduler._kvwarm_plan_covers(stub, point)
    # Live-chain coverage agrees with the plan for a chain of exactly that depth.
    stub._kvwarm_building = False
    stub._kvwarm_stage_batch = 1
    stub._kvwarm_chain_ids = ["chain"]
    stub._kvwarm_chain_prompts = {"chain": [1] * depth}
    assert InstrumentedScheduler._kvwarm_covers(stub, point, [ctx])
    # ...and so does the shadow's block check, with the headroom the point
    # will actually run (repeats steady steps) and a chain holding only the
    # blocks its prompt needs.
    chain = [_FakeBlock(i) for i in range(-(-depth // block_size))]
    mgr = _FakeManager(chain, cow=False)
    stub.kv_cache_manager = SimpleNamespace(
        block_pool=_FakePool(),
        coordinator=SimpleNamespace(single_type_managers=[mgr]),
    )
    table, _ = InstrumentedScheduler._kvwarm_register_shadow(
        stub, "shadow", "chain", ctx, repeats
    )
    assert len(table[0]) == -(-(ctx + 1 + repeats) // block_size)


def test_kvwarm_shadow_block_check_rejects_a_single_steady_step_margin():
    """The former non-giant plan built a 16-token chain (one block) for a rung
    whose deepest point is admitted at 13 tokens; a shadow that runs the
    default three steady steps writes positions 13..16, and position 16 needs
    a second block the chain never held."""
    stub, mgr, pool, chain = _shadow_stub(cow=False)
    mgr.req_to_blocks["chain"] = chain[:1]
    with pytest.raises(RuntimeError, match="too shallow"):
        InstrumentedScheduler._kvwarm_register_shadow(stub, "shadow", "chain", 13, 3)


def _fenced_frees(*, pending: bool) -> deque:
    """``Scheduler.deferred_frees`` shape: ``(fence_seq, blocks)`` entries
    that ``update_from_output`` drains once the fenced step is processed."""
    return deque([(3, [])]) if pending else deque()


def _kvwarm_busy_stub(point: BenchmarkPoint, *, chains=("chain-a",)):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._kvwarm_plan = {4: 64, 2: 64}
    stub._bench_active_req_ids = set()
    stub._bench_current_point = None
    stub._bench_grid = deque([point])
    stub._bench_deadline_monotonic = None
    stub._bench_stop_requested = False
    stub._kvwarm_chain_ids = list(chains)
    stub._kvwarm_building = False
    stub._bench_synchronizer = None
    stub._kvwarm_start_stage = MagicMock()
    stub.deferred_frees = _fenced_frees(pending=False)

    def shed():
        # Like ``_kvwarm_shed_chains``: only a fleet that still holds chains
        # can leave blocks behind the fence; an empty shed is a no-op.
        if stub._kvwarm_chain_ids:
            stub.deferred_frees = _fenced_frees(pending=True)
        stub._kvwarm_chain_ids = []
        stub._kvwarm_stage_batch = None

    stub._kvwarm_shed_chains = MagicMock(side_effect=shed)
    return stub


def test_kvwarm_step_busy_stops_building_after_soft_timeout():
    point = BenchmarkPoint(point_type="decode", total_kv_read_tokens=256, batch_size=4)
    stub = _kvwarm_busy_stub(point)
    stub._bench_deadline_monotonic = 0.0  # already elapsed
    stub._kvwarm_shed_chains = MagicMock()  # parked chains: freed at once
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is False
    stub._kvwarm_shed_chains.assert_called_once_with()
    stub._kvwarm_start_stage.assert_not_called()


def test_kvwarm_step_busy_yields_one_idle_step_while_shed_blocks_are_fenced():
    """A shed whose blocks are still behind the deferred-free fence hands the
    step to the real scheduler (True) so the in-flight output can drain them;
    the fake-fallback point proceeds (False) once the fence is clear."""
    point = BenchmarkPoint(point_type="decode", total_kv_read_tokens=4096, batch_size=4)
    stub = _kvwarm_busy_stub(point)
    stub._kvwarm_plan_covers = lambda pt: False  # fake-fallback point

    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    stub._kvwarm_shed_chains.assert_called_once_with()
    stub._kvwarm_start_stage.assert_not_called()

    stub.deferred_frees.clear()  # update_from_output drained the fence
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is False
    stub._kvwarm_shed_chains.assert_called_once_with()


def test_kvwarm_stage_switch_waits_for_fenced_frees_before_building():
    """Switching rungs sheds the old fleet; the next fleet is launched only
    once the old fleet's blocks have actually returned to the pool."""
    point = BenchmarkPoint(point_type="decode", total_kv_read_tokens=64, batch_size=2)
    stub = _kvwarm_busy_stub(point)
    stub._kvwarm_plan_covers = lambda pt: True
    stub._kvwarm_stage_batch = 4

    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    stub._kvwarm_start_stage.assert_not_called()

    stub.deferred_frees.clear()
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    stub._kvwarm_start_stage.assert_called_once_with(2, 64)
    assert stub._kvwarm_shed_chains.call_count == 2


class _FakeBlock:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 1
        self.is_null = False


class _FakePool:
    """``BlockPool`` reference semantics: ``touch`` is +1, ``get_new_blocks``
    hands out blocks at ref 1, ``free_blocks`` is -1 and a block joins the
    free queue only when it reaches 0 (a free past 0 is a double free)."""

    def __init__(self, next_id=1000):
        self.next_id = next_id
        self.touched = []
        self.freed = []
        self.free_queue = []

    def touch(self, blocks):
        self.touched.extend(blocks)
        for b in blocks:
            b.ref_cnt += 1

    def get_new_blocks(self, n):
        out = []
        for _ in range(n):
            out.append(_FakeBlock(self.next_id))
            self.next_id += 1
        return out

    def free_blocks(self, ordered_blocks):
        for b in ordered_blocks:
            assert b.ref_cnt > 0, f"double free of block {b.block_id}"
            b.ref_cnt -= 1
            self.freed.append(b)
            if b.ref_cnt == 0:
                self.free_queue.append(b)


class _FakeManager:
    """``SingleTypeKVCacheManager`` surface used by shadow registration.
    ``_apply_cow`` mirrors vLLM's: the table slot is redirected to the CoW
    block, which takes the retention ref, and the (source, cow) pair waits
    for ``take_pending_cow_copies``."""

    block_size = 16

    def __init__(self, chain_blocks, cow=True):
        self.req_to_blocks = {"chain": chain_blocks}
        self.num_cached_block = {}
        self.cows = []
        self._pending_cow_copies = []
        if cow:
            self._apply_cow = self._cow

    def _cow(self, req_id, idx, src, dst):
        assert self.req_to_blocks[req_id][idx] is src
        self.req_to_blocks[req_id][idx] = dst
        self._pending_cow_copies.append((src, dst))
        dst.ref_cnt += 1
        self.cows.append((src.block_id, dst.block_id))

    def take_pending_cow_copies(self):
        pending, self._pending_cow_copies = self._pending_cow_copies, []
        return pending

    def pop_blocks_for_free(self, req_id):
        self.num_cached_block.pop(req_id, None)
        return self.req_to_blocks.pop(req_id, [])


def _take_kv_cache_block_copies(manager):
    """``KVCacheManager.take_kv_cache_block_copies``: drain every manager's
    pending (source, cow) pairs into copy descriptors plus the retained
    endpoints (both blocks of every pair)."""
    pending = []
    for mgr in manager.coordinator.single_type_managers:
        pending.extend(mgr.take_pending_cow_copies())
    copies = [(src.block_id, dst.block_id) for src, dst in pending]
    return copies, [block for pair in pending for block in pair]


def _shadow_stub(cow=True):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    chain = [_FakeBlock(i) for i in range(10)]  # 160 tokens
    mgr = _FakeManager(chain, cow=cow)
    pool = _FakePool()
    manager = SimpleNamespace(
        block_pool=pool, coordinator=SimpleNamespace(single_type_managers=[mgr])
    )
    if cow:
        # A vLLM whose managers fork with ``_apply_cow`` also drains the forks
        # at the manager level; older ones offer neither.
        manager.take_kv_cache_block_copies = lambda: _take_kv_cache_block_copies(
            manager
        )
    stub.kv_cache_manager = manager
    stub.cache_config = SimpleNamespace(block_size=16)
    return stub, mgr, pool, chain


def test_kvwarm_shadow_registration_shares_prefix_and_forks_tail_with_cow():
    stub, mgr, pool, chain = _shadow_stub(cow=True)
    # ctx=40 -> 2 shared full blocks, writes at 40..43 land in block 2 only
    table, zero_ids = InstrumentedScheduler._kvwarm_register_shadow(
        stub, "shadow", "chain", 40, 3
    )
    # Shared prefix takes the shadow's ref; the CoW source takes the hit-ref
    # that the retained release after the copy will consume (production
    # semantics: the source is a prefix-cache hit of the request).
    assert [b.block_id for b in pool.touched] == [0, 1, 2]
    assert all(chain[i].ref_cnt == 2 for i in range(3))
    assert mgr.cows == [(2, 1000)]
    assert table == ([0, 1, 1000],)
    assert zero_ids == []
    assert mgr.num_cached_block["shadow"] == 2


def test_kvwarm_shadow_registration_zero_fills_tail_without_cow():
    stub, mgr, pool, chain = _shadow_stub(cow=False)
    # ctx=47 with headroom 3 -> writes 47..50 span blocks 2 and 3
    table, zero_ids = InstrumentedScheduler._kvwarm_register_shadow(
        stub, "shadow", "chain", 47, 3
    )
    assert table == ([0, 1, 1000, 1001],)
    assert zero_ids == [1000, 1001]
    assert [b.block_id for b in mgr.req_to_blocks["chain"]] == list(range(10))


def test_kvwarm_shadow_registration_takes_tail_before_touching_prefix():
    """A pool that cannot supply the private tail must fail before any chain
    block is over-referenced (a leaked ref breaks reset_prefix_cache)."""
    stub, mgr, pool, chain = _shadow_stub(cow=True)

    def exhausted(n):
        raise ValueError(f"Cannot get {n} free blocks from the pool")

    pool.get_new_blocks = exhausted
    with pytest.raises(ValueError, match="free blocks"):
        InstrumentedScheduler._kvwarm_register_shadow(stub, "shadow", "chain", 40, 3)
    assert pool.touched == []
    assert all(b.ref_cnt == 1 for b in chain)
    assert "shadow" not in mgr.req_to_blocks


def test_kvwarm_shadow_pool_shortfall_matches_tail_arithmetic():
    stub, mgr, pool, chain = _shadow_stub(cow=True)
    pool.get_num_free_blocks = lambda: 3
    # ctx=40 -> writes 40..43 need block 2 only (1 fresh); ctx=47 -> writes
    # 47..50 span blocks 2 and 3 (2 fresh): 3 fresh blocks in total.
    assert InstrumentedScheduler._kvwarm_shadow_pool_shortfall(stub, [40, 47], 3) == 0
    pool.get_num_free_blocks = lambda: 2
    assert InstrumentedScheduler._kvwarm_shadow_pool_shortfall(stub, [40, 47], 3) == 1
    # Without the pool API the check is skipped rather than guessed.
    del pool.get_num_free_blocks
    assert InstrumentedScheduler._kvwarm_shadow_pool_shortfall(stub, [40, 47], 3) == 0


def test_kvwarm_shadow_registration_rejects_too_shallow_chain():
    stub, mgr, pool, chain = _shadow_stub()
    with pytest.raises(RuntimeError, match="too shallow"):
        InstrumentedScheduler._kvwarm_register_shadow(stub, "shadow", "chain", 158, 3)


def test_kvwarm_chain_parks_only_after_in_flight_tokens_drain():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    chain = SimpleNamespace(
        request_id="chain-a", num_computed_tokens=8, num_output_placeholders=1
    )
    stub._kvwarm_chain_ids = ["chain-a"]
    stub._kvwarm_chain_prompts = {"chain-a": [1] * 8}
    stub.requests = {"chain-a": chain}
    stub.running = [chain]
    stub._kvwarm_building = True
    stub._kvwarm_stage_batch = 1
    stub._bench_synchronizer = None
    stub._kvwarm_meta_init = lambda: {"stages": []}
    # In flight: parked at once (no new steps get scheduled) but the stage
    # stays pending until the in-flight token lands.
    assert InstrumentedScheduler._kvwarm_monitor_build(stub) is True
    assert stub.running == []
    assert stub._kvwarm_building is True
    # Drained: the stage completes.
    chain.num_output_placeholders = 0
    assert InstrumentedScheduler._kvwarm_monitor_build(stub) is True
    assert stub._kvwarm_building is False


# ---------------------------------------------------------------------------
# Attention-DP: a stage's verdict is the group's, never one rank's
# ---------------------------------------------------------------------------
#
# Under attention-DP every rank builds the same rung (the plan comes from the
# negotiated envelope) but a chain can vanish, or the pool can fall short of
# the shadows' tails, on one rank only. Zeroing the plan there alone would
# send that rank to fake injection while its peers inject real KV: READY
# summaries differ and the sweep aborts. The outcome is therefore reported
# through the synchronizer's non-blocking stage exchange and applied only
# once the group verdict is in; meanwhile every step is handed to the real
# scheduler (the DP forward is collective, so no rank may block).


def _kvwarm_group_stage_stub(*, chains=("chain-a", "chain-b"), synchronizer=None):
    point = BenchmarkPoint(point_type="decode", total_kv_read_tokens=256, batch_size=4)
    stub = _kvwarm_busy_stub(point, chains=chains)
    stub._kvwarm_plan = {4: 128}  # covers the point: max(63) + 1 + 3 <= 128
    stub._bench_synchronizer = MagicMock() if synchronizer is None else synchronizer
    stub._bench_synchronizer.timeout_seconds = 10.0
    stub._kvwarm_building = True
    stub._kvwarm_stage_batch = 4
    stub._kvwarm_stage_t0 = time.monotonic()
    stub._kvwarm_chain_prompts = {chain: [1] * 8 for chain in chains}
    stub.requests = {
        chain: SimpleNamespace(
            request_id=chain, num_computed_tokens=8, num_output_placeholders=0
        )
        for chain in chains
    }
    stub.running = []
    meta = {"stages": []}
    stub._kvwarm_meta_init = lambda: meta
    stub._kvwarm_stage_shadow_shortfall = lambda batch: 0
    return stub, meta


def test_kvwarm_stage_verdict_is_taken_from_the_group():
    stub, meta = _kvwarm_group_stage_stub()
    sync = stub._bench_synchronizer
    sync.stage_poll.side_effect = [None, None, True]

    # Build complete: the local outcome is reported, not applied.
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    sync.stage_report.assert_called_once()
    (batch, ok), kwargs = sync.stage_report.call_args
    # No soft deadline armed in the stub: the budget is the protocol timeout.
    assert (batch, ok) == (4, True) and kwargs["timeout"] == 10.0
    assert stub._kvwarm_building is False
    assert stub._kvwarm_stage_reported[:2] == (4, True)
    assert meta["stages"] == []
    # Pending verdict: every step is an idle step for the real scheduler.
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    assert meta["stages"] == []
    # Verdict in: the stage is ready and the point flow resumes.
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    assert stub._kvwarm_stage_reported is None
    assert [entry["batch"] for entry in meta["stages"]] == [4]
    assert meta["stages"][0]["depth"] == 8 and "failed" not in meta["stages"][0]
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is False
    assert stub._kvwarm_plan[4] == 128
    stub._kvwarm_shed_chains.assert_not_called()
    assert sync.stage_poll.call_count == 3


def test_kvwarm_group_fallback_zeroes_the_plan_and_sheds_a_healthy_fleet():
    stub, meta = _kvwarm_group_stage_stub()
    sync = stub._bench_synchronizer
    sync.stage_poll.side_effect = [None, False]

    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True  # reported ok
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True  # pending
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True  # verdict: no
    assert stub._kvwarm_plan[4] == 0
    stub._kvwarm_shed_chains.assert_called_once_with()
    (entry,) = meta["stages"]
    assert entry["batch"] == 4 and entry["failed"] is True
    assert entry["group_fallback"] is True
    # The rung's points now take fake injection: the shed blocks drain behind
    # the fence first, then the fake path proceeds.
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    stub.deferred_frees.clear()
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is False
    stub._kvwarm_start_stage.assert_not_called()


def test_kvwarm_local_stage_failure_is_reported_before_it_is_applied():
    stub, meta = _kvwarm_group_stage_stub()
    del stub.requests["chain-a"]  # vanished during the build
    sync = stub._bench_synchronizer
    sync.stage_poll.side_effect = [None, False]

    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    # Survivors are released at once, but the plan waits for the group.
    stub._kvwarm_shed_chains.assert_called_once_with()
    (batch, ok), _ = sync.stage_report.call_args
    assert (batch, ok) == (4, False)
    assert stub._kvwarm_plan[4] == 128
    assert meta["stages"] == []
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    assert stub._kvwarm_plan[4] == 0
    assert meta["stages"] == [
        {"batch": 4, "failed": True, "vanished": 1, "group_fallback": True}
    ]


def test_kvwarm_stage_pool_shortfall_fails_the_rung_for_the_group():
    """The per-point pool check of injection would skip a point on one rank
    alone; under attention-DP the rung's worst case is checked once the
    fleet is parked and goes into the shared verdict."""
    stub, meta = _kvwarm_group_stage_stub()
    stub._kvwarm_stage_shadow_shortfall = lambda batch: 3
    sync = stub._bench_synchronizer
    sync.stage_poll.side_effect = [False]

    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    (batch, ok), _ = sync.stage_report.call_args
    assert (batch, ok) == (4, False)
    stub._kvwarm_shed_chains.assert_called_once_with()
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    assert meta["stages"] == [
        {"batch": 4, "failed": True, "pool_shortfall": 3, "group_fallback": True}
    ]


def test_kvwarm_soft_timeout_mid_build_abandons_the_stage_through_the_group():
    """A rank that reaches the soft timeout while its fleet is still
    building must not walk off to the boundary handshake while a peer is
    waiting for its stage report."""
    stub, meta = _kvwarm_group_stage_stub()
    stub.requests["chain-a"].num_computed_tokens = 2  # still building
    stub._bench_deadline_monotonic = 0.0  # soft timeout elapsed
    sync = stub._bench_synchronizer
    sync.stage_poll.side_effect = [False]

    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    (batch, ok), _ = sync.stage_report.call_args
    assert (batch, ok) == (4, False)
    stub._kvwarm_shed_chains.assert_called_once_with()
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    assert meta["stages"] == [
        {"batch": 4, "failed": True, "soft_timeout": True, "group_fallback": True}
    ]
    stub._kvwarm_start_stage.assert_not_called()


def test_kvwarm_stage_settles_locally_without_a_synchronizer():
    stub, meta = _kvwarm_group_stage_stub()
    stub._bench_synchronizer = None
    stub._kvwarm_stage_shadow_shortfall = MagicMock()

    assert InstrumentedScheduler._kvwarm_step_busy(stub) is True
    assert stub._kvwarm_stage_reported is None
    assert [entry["batch"] for entry in meta["stages"]] == [4]
    stub._kvwarm_stage_shadow_shortfall.assert_not_called()
    assert InstrumentedScheduler._kvwarm_step_busy(stub) is False


def test_kvwarm_stage_shadow_shortfall_takes_the_rung_worst_case(monkeypatch):
    monkeypatch.setenv("DYN_BENCH_GIANT_KV_REPEATS", "3")
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._kvwarm_plan = {4: 128, 2: 128}
    stub._bench_grid = deque(
        [
            BenchmarkPoint(point_type="decode", total_kv_read_tokens=256, batch_size=4),
            BenchmarkPoint(point_type="decode", total_kv_read_tokens=188, batch_size=4),
            BenchmarkPoint(point_type="decode", total_kv_read_tokens=64, batch_size=2),
        ]
    )
    seen = []

    def shortfall(injected, headroom):
        seen.append((list(injected), headroom))
        return 5 if injected[0] == 46 else 0

    stub._kvwarm_shadow_pool_shortfall = shortfall
    assert InstrumentedScheduler._kvwarm_stage_shadow_shortfall(stub, 4) == 5
    # Only this rung's covered points, with the full repeat count as headroom.
    assert seen == [([63] * 4, 3), ([46] * 4, 3)]


def test_kvwarm_stage_sync_timeout_reaches_the_soft_deadline():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    synchronizer = SimpleNamespace(timeout_seconds=10.0)
    stub._bench_deadline_monotonic = time.monotonic() + 100.0
    budget = InstrumentedScheduler._kvwarm_stage_sync_timeout(stub, synchronizer)
    assert 109.0 < budget <= 110.0
    stub._bench_deadline_monotonic = 0.0  # already elapsed
    assert InstrumentedScheduler._kvwarm_stage_sync_timeout(stub, synchronizer) == 10.0
    stub._bench_deadline_monotonic = None
    assert InstrumentedScheduler._kvwarm_stage_sync_timeout(stub, synchronizer) == 10.0


# ---------------------------------------------------------------------------
# Benchmark request retirement goes through the scheduler's abort path
# ---------------------------------------------------------------------------
#
# Chains and benchmark requests are retired with ``finish_requests`` +
# ``RequestStatus.FINISHED_ABORTED`` (vllm/v1/core/sched/scheduler.py): it
# drops them from the waiting, skipped and running queues and runs
# ``_free_request`` (KV-connector and encoder-cache callbacks,
# ``finished_req_ids`` for the worker, the deferred-free fence, the
# ``self.requests`` removal). Editing those structures by hand skipped every
# callback and left a chain still queued in ``waiting`` mid-build to be
# re-admitted against a ``self.requests`` entry that no longer existed.


def test_kvwarm_shed_chains_aborts_every_chain_through_finish_requests():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub.finish_requests = MagicMock()
    stub._kvwarm_chain_ids = ["chain-a", "chain-b"]
    stub._kvwarm_chain_prompts = {"chain-a": [1] * 8, "chain-b": [1] * 8}
    stub._kvwarm_stage_batch = 2
    stub._kvwarm_building = True

    InstrumentedScheduler._kvwarm_shed_chains(stub)

    stub.finish_requests.assert_called_once_with(
        ["chain-a", "chain-b"], RequestStatus.FINISHED_ABORTED
    )
    assert stub._kvwarm_chain_ids == []
    assert stub._kvwarm_chain_prompts == {}
    assert stub._kvwarm_stage_batch is None
    assert stub._kvwarm_building is False

    # Nothing left to shed: the abort path is not entered at all.
    InstrumentedScheduler._kvwarm_shed_chains(stub)
    stub.finish_requests.assert_called_once()


def test_bench_cleanup_finishes_live_requests_and_forgets_borrowed_shadows():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub.finish_requests = MagicMock()
    stub._bench_active_req_ids = {"__bench_0", "__bench_1", "__bench_2"}
    # __bench_2 already left through vLLM's own finish path.
    stub.requests = {"__bench_0": object(), "__bench_1": object()}
    stub._kvwarm_borrowed_ids = {"__bench_1", "__bench_9"}
    stub._schedule_times = deque([1.0])
    stub._bench_extra_steps_left = 2

    InstrumentedScheduler._bench_cleanup_requests(stub)

    ids, status = stub.finish_requests.call_args.args
    assert sorted(ids) == ["__bench_0", "__bench_1"]
    assert status is RequestStatus.FINISHED_ABORTED
    assert stub._bench_active_req_ids == set()
    assert stub._kvwarm_borrowed_ids == {"__bench_9"}
    assert len(stub._schedule_times) == 0
    assert stub._bench_extra_steps_left == 0


# ---------------------------------------------------------------------------
# Deferred-free fence: nothing draws from the pool while released blocks
# are still owed to it
# ---------------------------------------------------------------------------
#
# With ``defer_block_free`` (async scheduling on a KV consumer) the parent's
# ``_free_request_blocks`` parks the blocks of a request whose last step is
# still in flight in ``deferred_frees``; ``update_from_output`` returns them
# to the pool once that step's output is processed. A shed or cleanup that
# lands in this window must yield one idle step instead of injecting into a
# pool that is short of those blocks, and the prefix-cache reset must wait
# for them (a block with ref_cnt > 0 makes ``reset_prefix_cache`` fail).


def test_bench_step_decode_waits_for_fenced_frees_before_injecting():
    point = BenchmarkPoint(point_type="decode", total_kv_read_tokens=48, batch_size=3)
    stub = _steady_injection_stub(point)
    stub.deferred_frees = _fenced_frees(pending=True)

    assert InstrumentedScheduler._bench_step_decode(stub) is None
    stub._bench_inject_fake_decode.assert_not_called()
    stub._bench_stop_at_timeout_boundary.assert_not_called()
    assert list(stub._bench_grid) == [point]

    stub.deferred_frees.clear()
    assert InstrumentedScheduler._bench_step_decode(stub) is not None
    stub._bench_inject_fake_decode.assert_called_once_with([15, 15, 15])


def test_bench_done_step_idles_until_fenced_frees_drain():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_phase = _BenchPhase.DONE
    stub._bench_start_timing = MagicMock()
    stub._bench_build_grid = MagicMock()
    stub._bench_clear_prefix_cache = MagicMock(return_value=False)
    stub._bench_synchronizer = MagicMock()
    stub._bench_finish_timing = MagicMock()
    stub._bench_deactivate = MagicMock()
    stub._bench_write_results = MagicMock()

    assert InstrumentedScheduler._bench_step(stub) is None

    stub._bench_clear_prefix_cache.assert_called_once_with()
    stub._bench_synchronizer.synchronize_cleanup.assert_not_called()
    stub._bench_finish_timing.assert_not_called()
    stub._bench_deactivate.assert_not_called()
    stub._bench_write_results.assert_not_called()
    assert stub._bench_phase == _BenchPhase.DONE


def test_clear_prefix_cache_waits_for_fenced_frees_and_tolerates_a_fenced_abort():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_prefix_cache_cleared = False
    stub.kv_cache_manager = SimpleNamespace(
        reset_prefix_cache=MagicMock(return_value=False)
    )
    stub.deferred_frees = _fenced_frees(pending=True)

    # DONE step: no attempt while blocks are fenced; the step idles instead.
    assert InstrumentedScheduler._bench_clear_prefix_cache(stub) is False
    stub.kv_cache_manager.reset_prefix_cache.assert_not_called()

    # Abort: try anyway; a failure with fenced blocks is reported, not fatal
    # (the abort re-raises its own error, which ends the engine core).
    assert (
        InstrumentedScheduler._bench_clear_prefix_cache(stub, allow_pending=True)
        is False
    )
    stub.kv_cache_manager.reset_prefix_cache.assert_called_once_with()
    assert stub._bench_prefix_cache_cleared is False

    # Nothing fenced and still failing: a leak, which raises.
    stub.deferred_frees.clear()
    with pytest.raises(RuntimeError, match="failed to clear"):
        InstrumentedScheduler._bench_clear_prefix_cache(stub, allow_pending=True)

    # Drained and the reset succeeds: cleared, and the flag keeps its meaning.
    stub.kv_cache_manager.reset_prefix_cache.return_value = True
    assert InstrumentedScheduler._bench_clear_prefix_cache(stub) is True
    assert stub._bench_prefix_cache_cleared is True


def test_schedule_advances_the_deferred_free_fence_for_benchmark_steps():
    """Benchmark-built outputs bypass the parent's ``schedule()``, which is
    where ``sched_step_seq`` advances for every non-empty step (matched by
    ``processed_step_seq`` in ``update_from_output``). Without the mirror a
    request retired while its benchmark step is still in flight compares as
    already processed and is freed at once instead of behind the fence."""
    stub = _make_decode_sweep_stub(connector=None)
    stub._bench_synchronize_output = MagicMock()
    stub._schedule_times = deque()
    stub.defer_block_free = True
    stub.sched_step_seq = 5

    stub._bench_step = MagicMock(
        return_value=SimpleNamespace(total_num_scheduled_tokens=3)
    )
    InstrumentedScheduler.schedule(stub)
    assert stub.sched_step_seq == 6
    stub._update_after_schedule.assert_called_once()

    # An empty benchmark output (injection shortfall) advances nothing, like
    # the parent's 0-token steps.
    stub._bench_step = MagicMock(
        return_value=SimpleNamespace(total_num_scheduled_tokens=0)
    )
    InstrumentedScheduler.schedule(stub)
    assert stub.sched_step_seq == 6

    # Without the fence the counter is never touched.
    stub.defer_block_free = False
    stub._bench_step = MagicMock(
        return_value=SimpleNamespace(total_num_scheduled_tokens=3)
    )
    InstrumentedScheduler.schedule(stub)
    assert stub.sched_step_seq == 6


# ---------------------------------------------------------------------------
# Shadow registration: all-or-nothing across KV-cache groups, and the full
# reference lifecycle of one shadow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failure", ["pool_exhausted", "too_shallow"])
def test_kvwarm_shadow_registration_unwinds_earlier_groups_on_failure(failure):
    """Hybrid layouts register one shadow per KV-cache group. A failure in a
    later group must leave no trace of the earlier ones: their tails go back
    to the pool and their chain blocks keep exactly the chain's reference
    (an over-referenced chain block breaks reset_prefix_cache)."""
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    chain_a = [_FakeBlock(i) for i in range(10)]
    chain_b = [_FakeBlock(100 + i) for i in range(10)]
    mgr_a = _FakeManager(chain_a, cow=True)
    mgr_b = _FakeManager(chain_b, cow=True)
    pool = _FakePool()
    stub.kv_cache_manager = SimpleNamespace(
        block_pool=pool,
        coordinator=SimpleNamespace(single_type_managers=[mgr_a, mgr_b]),
    )
    stub.cache_config = SimpleNamespace(block_size=16)
    if failure == "pool_exhausted":
        take = pool.get_new_blocks

        def get_new_blocks(n):
            if pool.next_id > 1000:  # the first group's tail is already out
                raise ValueError(f"Cannot get {n} free blocks from the pool")
            return take(n)

        pool.get_new_blocks = get_new_blocks
        expected = pytest.raises(ValueError, match="free blocks")
    else:
        mgr_b.req_to_blocks["chain"] = chain_b[:2]
        expected = pytest.raises(RuntimeError, match="too shallow")

    with expected:
        InstrumentedScheduler._kvwarm_register_shadow(stub, "shadow", "chain", 40, 3)

    # ctx=40, headroom 3: one private tail block per group. The first group's
    # tail was taken and is back in the pool; the second was never taken.
    assert [b.block_id for b in pool.freed] == [1000]
    assert pool.free_queue == pool.freed
    assert pool.touched == []
    assert all(b.ref_cnt == 1 for b in chain_a + chain_b)
    for mgr in (mgr_a, mgr_b):
        assert "shadow" not in mgr.req_to_blocks
        assert "shadow" not in mgr.num_cached_block
        assert mgr.cows == []
        assert mgr.take_pending_cow_copies() == []


@pytest.mark.parametrize("cow", [True, False])
def test_kvwarm_shadow_lifecycle_returns_every_reference(cow):
    """Reference accounting of one shadow from registration to chain release,
    step by step against vLLM 0.28:

    1. ``_kvwarm_register_shadow``: shared prefix +1 (``BlockPool.touch``);
       fresh tail block at ref 1 (``BlockPool.get_new_blocks``). With CoW the
       source tail block gets +1 (the hit-ref a partial prefix hit carries in
       production) and the fresh block +1 retention
       (``SingleTypeKVCacheManager._apply_cow``).
    2. Retention release (CoW only): ``_kvwarm_inject_borrowed`` drains
       ``KVCacheManager.take_kv_cache_block_copies`` right after registration
       and returns both endpoints through ``BlockPool.free_blocks`` (-1 each);
       the copies themselves ride on the admission step.
    3. Shadow release: ``finish_requests`` -> ``_free_request`` ->
       ``_free_request_blocks`` -> ``KVCacheManager.free`` ->
       ``SingleTypeKVCacheManager.free`` =
       ``free_blocks(reversed(pop_blocks_for_free(req_id)))``.
    4. Chain release: the same path for the chain.

    Afterwards every block sits at ref 0 exactly once: nothing was freed
    past 0 and nothing stays referenced (which would fail
    ``reset_prefix_cache``)."""
    stub, mgr, pool, chain = _shadow_stub(cow=cow)
    # ctx=40 -> blocks 0,1 shared; the shadow writes 40..43 in block 2 only.
    ctx, headroom = 40, 3

    # 1. registration
    table, zero_ids = InstrumentedScheduler._kvwarm_register_shadow(
        stub, "shadow", "chain", ctx, headroom
    )
    shadow_blocks = mgr.req_to_blocks["shadow"]
    src, fresh = chain[2], shadow_blocks[2]
    assert shadow_blocks[:2] == chain[:2] and fresh is not src
    assert table == ([0, 1, fresh.block_id],)
    assert [b.ref_cnt for b in chain[:2]] == [2, 2]
    if cow:
        assert (src.ref_cnt, fresh.ref_cnt) == (2, 2)
        assert zero_ids == []
        # 2. retention release
        copies = InstrumentedScheduler._kvwarm_take_cow_copies(stub)
        assert copies == [(src.block_id, fresh.block_id)]
        assert (src.ref_cnt, fresh.ref_cnt) == (1, 1)
        assert pool.free_queue == []
        assert mgr.take_pending_cow_copies() == []
    else:
        assert (src.ref_cnt, fresh.ref_cnt) == (1, 1)
        assert zero_ids == [fresh.block_id]

    # 3. shadow release
    pool.free_blocks(reversed(mgr.pop_blocks_for_free("shadow")))
    assert "shadow" not in mgr.num_cached_block
    assert fresh.ref_cnt == 0 and pool.free_queue == [fresh]
    assert [b.ref_cnt for b in chain] == [1] * len(chain)

    # 4. chain release
    pool.free_blocks(reversed(mgr.pop_blocks_for_free("chain")))
    assert all(b.ref_cnt == 0 for b in chain)
    assert mgr.req_to_blocks == {}
    assert sorted(b.block_id for b in pool.free_queue) == sorted(
        b.block_id for b in [*chain, fresh]
    )
    assert len(pool.free_queue) == len(chain) + 1


def _kvwarm_injection_stub(chain_ids):
    """Everything ``_kvwarm_inject_borrowed`` reads, over the fake pool and a
    CoW manager; every chain owns ten blocks (160 tokens) at ref 1, parked."""
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    chains = {
        chain_id: [_FakeBlock(100 * index + i) for i in range(10)]
        for index, chain_id in enumerate(chain_ids)
    }
    mgr = _FakeManager(chains[chain_ids[0]], cow=True)
    mgr.req_to_blocks = dict(chains)
    pool = _FakePool()
    pool.get_num_free_blocks = lambda: 10
    manager = SimpleNamespace(
        block_pool=pool,
        coordinator=SimpleNamespace(single_type_managers=[mgr]),
        num_kv_cache_groups=1,
    )
    manager.take_kv_cache_block_copies = lambda: _take_kv_cache_block_copies(manager)
    stub.kv_cache_manager = manager
    stub.cache_config = SimpleNamespace(block_size=16)
    stub._kvwarm_chain_ids = list(chain_ids)
    stub._kvwarm_chain_prompts = {chain_id: list(range(160)) for chain_id in chain_ids}
    stub.requests = {
        chain_id: SimpleNamespace(request_id=chain_id) for chain_id in chain_ids
    }
    stub.running = []
    stub.finished_req_ids = set()
    stub._bench_seq = 0
    stub._bench_active_req_ids = set()
    stub._kvwarm_borrowed_ids = set()
    stub._bench_block_hasher = None
    stub._bench_extra_steps_left = 3
    stub.connector = None
    stub.ec_connector = None
    stub.defer_block_free = True
    stub.deferred_frees = deque()
    stub._free_cow_retained_blocks = MagicMock()
    return stub, mgr, pool, chains


def test_kvwarm_inject_borrowed_releases_cow_retentions_before_the_step_runs():
    """Under ``defer_block_free`` the parent's ``_free_cow_retained_blocks``
    would park the retention release behind the fence and drain it in the
    admission step's ``update_from_output`` -- inside the steady step's
    measured window. The injection releases it at once instead: the chain
    and the shadow's own table keep both endpoints alive until the point's
    untimed cleanup."""
    stub, mgr, pool, chains = _kvwarm_injection_stub(["chain-a"])
    chain = chains["chain-a"]

    output = InstrumentedScheduler._kvwarm_inject_borrowed(stub, [40])

    shadow = mgr.req_to_blocks["__bench_0"]
    src, cow = chain[2], shadow[2]
    assert output.total_num_scheduled_tokens == 1
    assert output.kv_cache_block_copies == [(src.block_id, cow.block_id)]
    assert output.scheduled_new_reqs[0].block_ids == ([0, 1, cow.block_id],)
    assert output.scheduled_new_reqs[0].num_computed_tokens == 40
    assert len(output.scheduled_new_reqs[0].prompt_token_ids) == 41
    # Retentions released here (-1 each); the chain still owns the source and
    # the shadow's table still owns the fork, so nothing reaches the pool.
    assert pool.freed == [src, cow]
    assert (src.ref_cnt, cow.ref_cnt) == (1, 1)
    assert pool.free_queue == []
    assert mgr.take_pending_cow_copies() == []
    assert list(stub.deferred_frees) == []
    stub._free_cow_retained_blocks.assert_not_called()
    assert stub._bench_active_req_ids == {"__bench_0"}
    assert stub._kvwarm_borrowed_ids == {"__bench_0"}
    assert stub.requests["__bench_0"].status == RequestStatus.RUNNING


def test_kvwarm_inject_borrowed_drops_queued_copies_when_a_later_shadow_fails():
    """A failure while registering the second shadow leaves the first one's
    fork queued in the manager. The step it would have ridden on is never
    built, so the injection drops the queue and releases the retentions;
    the abort path's cleanup then finds every block at exactly the chain's
    and the shadow's own references and the prefix-cache reset succeeds."""
    stub, mgr, pool, chains = _kvwarm_injection_stub(["chain-a", "chain-b"])
    chain_a, chain_b = chains["chain-a"], chains["chain-b"]
    take = pool.get_new_blocks

    def get_new_blocks(n):
        if pool.next_id > 1000:  # the first shadow's tail is already out
            raise ValueError(f"Cannot get {n} free blocks from the pool")
        return take(n)

    pool.get_new_blocks = get_new_blocks

    with pytest.raises(ValueError, match="free blocks"):
        InstrumentedScheduler._kvwarm_inject_borrowed(stub, [40, 40])

    cow = mgr.req_to_blocks["__bench_0"][2]
    assert mgr.take_pending_cow_copies() == []
    # First shadow: shared prefix +1, source hit-ref released, fork held by
    # the shadow's table only.
    assert [b.ref_cnt for b in chain_a] == [2, 2, 1] + [1] * 7
    assert cow.ref_cnt == 1
    # Second shadow: nothing registered, its chain untouched.
    assert "__bench_1" not in mgr.req_to_blocks
    assert all(b.ref_cnt == 1 for b in chain_b)
    assert stub._bench_active_req_ids == {"__bench_0"}

    # The abort path: finish the shadows, shed the chains, reset the cache.
    def finish_requests(req_ids, status):
        assert status is RequestStatus.FINISHED_ABORTED
        for req_id in req_ids:
            pool.free_blocks(reversed(mgr.pop_blocks_for_free(req_id)))
            stub.requests.pop(req_id, None)

    stub.finish_requests = finish_requests
    stub._schedule_times = deque()
    stub._kvwarm_stage_batch = 2
    stub._kvwarm_building = False
    stub._bench_prefix_cache_cleared = False
    every_block = [*chain_a, *chain_b, cow]
    stub.kv_cache_manager.reset_prefix_cache = lambda: all(
        b.ref_cnt == 0 for b in every_block
    )

    InstrumentedScheduler._bench_cleanup_requests(stub)
    assert InstrumentedScheduler._bench_clear_prefix_cache(stub, allow_pending=True)
    assert stub._bench_prefix_cache_cleared is True
    assert sorted(b.block_id for b in pool.free_queue) == sorted(
        b.block_id for b in every_block
    )
    assert len(pool.free_queue) == len(every_block)


# ---------------------------------------------------------------------------
# Fake decode injection commits its blocks to the prefix cache in the
# untimed window
# ---------------------------------------------------------------------------


def test_bench_inject_fake_decode_caches_blocks_before_the_request_runs():
    """``allocate_slots(..., delay_cache_blocks=True)`` leaves the prefix-cache
    commit to the caller. The injection must do it right there, after the
    allocation and before the request joins ``running``: otherwise the async
    scheduler commits every block inside the admission step's
    ``update_from_output`` and that CPU loop is booked into the steady step's
    inter-update wall time."""
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_seq = 0
    stub._bench_active_req_ids = set()
    stub.requests = {}
    stub.finished_req_ids = set()
    stub._bench_block_hasher = None
    stub.connector = None
    stub.ec_connector = None
    stub.kv_cache_manager = MagicMock()
    stub.kv_cache_manager.num_kv_cache_groups = 1
    stub.kv_cache_manager.take_new_block_ids = MagicMock(return_value=None)

    order: list[str] = []

    class _Running(list):
        def append(self, req):
            order.append("running")
            super().append(req)

    stub.running = _Running()
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([0, 1],)

    def allocate_slots(req, num_new_tokens, **kwargs):
        order.append("allocate")
        assert kwargs.get("delay_cache_blocks") is True
        return blocks

    def cache_blocks(req, num_computed_tokens):
        order.append("cache")
        # The full context, not the padded prompt, is what the request has
        # computed and what its block hashes may be committed for.
        assert num_computed_tokens == req.num_computed_tokens == 16

    stub.kv_cache_manager.allocate_slots = allocate_slots
    stub.kv_cache_manager.cache_blocks = cache_blocks

    output = InstrumentedScheduler._bench_inject_fake_decode(stub, context_lengths=[16])

    assert order == ["allocate", "cache", "running"]
    assert output.total_num_scheduled_tokens == 1
    assert stub._bench_active_req_ids == {"__bench_0"}
    assert stub.requests["__bench_0"].status == RequestStatus.RUNNING
