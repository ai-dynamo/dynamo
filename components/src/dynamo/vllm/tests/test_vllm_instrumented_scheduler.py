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

import json
import threading
import uuid
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

# Module-level import: triggers real site-packages ``vllm`` to load before
# pytest's rootpath insertion adds ``components/src/dynamo`` to ``sys.path``
# (which shadows the real ``vllm`` with the ``dynamo.vllm`` submodule for any
# later bare ``import vllm``). Mirrors the pattern in ``test_vllm_unit.py``,
# which imports ``dynamo.vllm.args`` at module level for the same reason.
# If this import is deferred to inside a test body, the real ``vllm`` will
# not be resolvable and ``instrumented_scheduler`` will fail to load with
# ``ModuleNotFoundError: No module named 'vllm.sampling_params'``.
import dynamo.vllm.instrumented_scheduler as instrumented_scheduler_module  # noqa: E402
from dynamo.vllm.instrumented_scheduler import (  # noqa: E402
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


def test_benchmark_synchronizer_close_flushes_after_cleanup():
    synchronizer = instrumented_scheduler_module._BenchmarkSynchronizer.__new__(
        instrumented_scheduler_module._BenchmarkSynchronizer
    )
    synchronizer._socket = MagicMock()
    synchronizer._timeout_ms = 1_000
    synchronizer._cleanup_complete = False

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
        except Exception as error:
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
    matching = {
        "total_num_scheduled_tokens": 2,
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 2,
        "sum_decode_kv_tokens": 128,
    }

    assert InstrumentedScheduler._bench_output_validation_error(point, matching) is None

    mismatched = dict(matching, num_decode_requests=1)
    error = InstrumentedScheduler._bench_output_validation_error(point, mismatched)
    assert "benchmark_id=3 SchedulerOutput does not match" in error


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
    stub._bench_config = SimpleNamespace(mode=mode)
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []

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
    stub._bench_config = SimpleNamespace(mode="prefill")
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []
    stub._bench_grid_error = None

    def generate_prefill_grid():
        stub._bench_grid.extend(
            BenchmarkPoint(point_type="prefill", total_prefill_tokens=index)
            for index in range(1, 4098)
        )

    stub._bench_generate_prefill_grid = generate_prefill_grid

    InstrumentedScheduler._bench_build_grid(stub)

    assert stub._bench_expected_points == 4097
    assert len(stub._bench_grid) == 4097
    assert [point.benchmark_id for point in stub._bench_grid] == list(range(1, 4098))
    assert stub._bench_grid_error is None


def test_benchmark_grid_assigns_stable_contiguous_ids_and_digest():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = SimpleNamespace(mode="prefill")
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []
    stub._bench_grid_error = None

    def generate_prefill_grid():
        stub._bench_grid.extend(
            [
                BenchmarkPoint(point_type="prefill", total_prefill_tokens=8),
                BenchmarkPoint(point_type="prefill", total_prefill_tokens=16),
            ]
        )

    stub._bench_generate_prefill_grid = generate_prefill_grid

    InstrumentedScheduler._bench_build_grid(stub)

    assert [point.benchmark_id for point in stub._bench_grid] == [1, 2]
    assert len(stub._bench_grid_digest) == 64


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
        assert points[0] == 0
        assert points[-1] == InstrumentedScheduler._bench_max_prefill_kv_read_tokens(
            stub, total_tokens, 1
        )


def test_agg_grid_contains_piecewise_prefill_then_full_decode_points():
    stub = _prefill_grid_stub()
    stub._bench_grid = deque()
    stub._bench_grid_built = False
    stub._bench_missing_phases = []
    stub._bench_grid_error = None
    stub._bench_feasible_max_decode_batch_size = 0
    stub._bench_config.mode = "agg"
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

    assert InstrumentedScheduler._bench_decode_kv_read_points(stub, 9) == [
        9,
        16,
        32,
        64,
        128,
        256,
        512,
        999,
    ]


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

    points = InstrumentedScheduler._bench_prefill_kv_read_points(stub, 40, 1)
    assert points[0] == 0
    assert all(point % 8 == 0 for point in points)
    assert InstrumentedScheduler._bench_seed_prompt_len(stub, 16) == 24

    # The extra seed block must also fit under max_model_len.
    assert InstrumentedScheduler._bench_prefill_kv_read_points(stub, 1, 1)[-1] == 112


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
        find_longest_cache_hit=MagicMock(return_value=(([],), 16))
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

    assert stub._bench_current_point is point
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

    InstrumentedScheduler._bench_clear_prefix_cache(stub)
    InstrumentedScheduler._bench_clear_prefix_cache(stub)

    stub.kv_cache_manager.reset_prefix_cache.assert_called_once_with()
    assert stub._bench_prefix_cache_cleared is True

    failed = InstrumentedScheduler.__new__(InstrumentedScheduler)
    failed._bench_prefix_cache_cleared = False
    failed.kv_cache_manager = SimpleNamespace(
        reset_prefix_cache=MagicMock(return_value=False)
    )
    with pytest.raises(RuntimeError, match="failed to clear synthetic prefix cache"):
        InstrumentedScheduler._bench_clear_prefix_cache(failed)


def test_benchmark_abort_clears_synthetic_prefix_cache_before_deactivation():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_cleanup_requests = MagicMock()
    stub._bench_clear_prefix_cache = MagicMock()
    stub._bench_write_results = MagicMock()
    stub._bench_deactivate = MagicMock()

    InstrumentedScheduler._bench_abort(stub, RuntimeError("benchmark failed"))

    stub._bench_cleanup_requests.assert_called_once_with()
    stub._bench_clear_prefix_cache.assert_called_once_with()
    stub._bench_write_results.assert_called_once_with()
    stub._bench_deactivate.assert_called_once_with(resume_publisher=True)
    assert stub._bench_grid_error == "benchmark failed"


def test_benchmark_abort_is_fail_closed_when_prefix_cleanup_fails():
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
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
        {"point": point.__dict__, "reason": "seed_cache_validation_failed"}
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
    stub._bench_clear_prefix_cache = MagicMock()
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
    stub._bench_cleanup_requests = MagicMock()
    stub._bench_inject_fake_decode = MagicMock(
        return_value=SimpleNamespace(total_num_scheduled_tokens=0)
    )

    output = InstrumentedScheduler._bench_step_decode(stub)

    assert output is None
    assert stub._bench_current_point is None
    assert stub._bench_skipped_points == [
        SkippedBenchmarkPoint(point=point, reason="decode_injection_failed")
    ]
    stub._bench_inject_fake_decode.assert_called_once_with([16, 16, 16])
    stub._bench_cleanup_requests.assert_called_once_with()
