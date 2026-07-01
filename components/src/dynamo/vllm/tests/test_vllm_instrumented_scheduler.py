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
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.request import RequestStatus  # noqa: E402

# Module-level import: triggers real site-packages ``vllm`` to load before
# pytest's rootpath insertion adds ``components/src/dynamo`` to ``sys.path``
# (which shadows the real ``vllm`` with the ``dynamo.vllm`` submodule for any
# later bare ``import vllm``). Mirrors the pattern in ``test_vllm_unit.py``,
# which imports ``dynamo.vllm.args`` at module level for the same reason.
# If this import is deferred to inside a test body, the real ``vllm`` will
# not be resolvable and ``instrumented_scheduler`` will fail to load with
# ``ModuleNotFoundError: No module named 'vllm.sampling_params'``.
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

    InstrumentedScheduler._bench_inject_fake_decode(stub, ctx_len=16, batch_size=1)

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
    stub._bench_config = SimpleNamespace(
        decode_length_granularity=2,
        decode_batch_size_granularity=2,
    )
    stub.cache_config = SimpleNamespace(num_gpu_blocks=num_gpu_blocks)
    stub.block_size = block_size
    stub.max_model_len = 256
    # Generous so the KV cap (not max_num_running_reqs) drives the boundary.
    stub.max_num_running_reqs = 10_000
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
        ctx_len = point.context_length
        bs = point.batch_size
        blocks_per_req = -(-(ctx_len + 1) // block_size)  # ceil
        max_feasible = num_gpu_blocks // blocks_per_req
        assert bs <= max_feasible, (
            f"point ctx_len={ctx_len} batch_size={bs} would exceed KV "
            f"capacity: needs {bs * blocks_per_req} blocks, only "
            f"{num_gpu_blocks} available "
            f"(blocks_per_req={blocks_per_req}, max_feasible={max_feasible})."
        )


def test_decode_grid_first_ctx_yields_block_aligned_capacity():
    """At ``ctx_len == block_size`` the per-request allocation is exactly
    2 blocks (16 prompt + 1 placeholder = 17 tokens, rounded up). The
    grid's largest batch for this ctx must respect that.
    """
    block_size = 16
    num_gpu_blocks = 100
    stub = _grid_stub_with_kv_capacity(num_gpu_blocks, block_size)

    InstrumentedScheduler._bench_generate_decode_grid(stub)

    boundary_points = [p for p in stub._bench_grid if p.context_length == block_size]
    assert len(boundary_points) > 0, (
        f"grid missing ctx_len={block_size} entries: {stub._bench_grid}"
    )
    # 100 blocks // 2 blocks-per-req == 50 max batch.
    assert max(p.batch_size for p in boundary_points) <= 50, (
        f"boundary ctx_len={block_size}: max batch must not exceed "
        f"num_gpu_blocks // 2 = 50; got "
        f"{[p.batch_size for p in boundary_points]}"
    )


# ---------------------------------------------------------------------------
# Prefill KV-read grid and seed lifecycle
# ---------------------------------------------------------------------------


def _prefill_grid_stub(*, kv_read_granularity: int, block_size: int = 8):
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_grid = []
    stub._bench_config = SimpleNamespace(
        prefill_isl_granularity=2,
        prefill_kv_read_granularity=kv_read_granularity,
    )
    stub.max_num_scheduled_tokens = 40
    stub.block_size = block_size
    return stub


def test_prefill_kv_read_grid_default_preserves_miss_only_sweep():
    stub = _prefill_grid_stub(kv_read_granularity=1)

    InstrumentedScheduler._bench_generate_prefill_grid(stub)

    assert [(p.isl, p.kv_read_tokens) for p in stub._bench_grid] == [
        (10, 0),
        (40, 0),
    ]


def test_prefill_kv_read_grid_crosses_with_isl_and_aligns_to_blocks():
    stub = _prefill_grid_stub(kv_read_granularity=3, block_size=8)

    InstrumentedScheduler._bench_generate_prefill_grid(stub)

    assert [(p.isl, p.kv_read_tokens) for p in stub._bench_grid] == [
        (10, 0),
        (10, 8),
        (40, 0),
        (40, 16),
        (40, 32),
    ]
    assert all(p.kv_read_tokens % stub.block_size == 0 for p in stub._bench_grid)
    assert all(p.kv_read_tokens <= p.isl - 1 for p in stub._bench_grid)


def test_prefill_kv_read_grid_accounts_for_eagle_cache_block_drop():
    stub = _prefill_grid_stub(kv_read_granularity=3, block_size=8)
    stub.kv_cache_manager = SimpleNamespace(use_eagle=True)

    assert InstrumentedScheduler._bench_prefill_kv_read_points(stub, 40) == [
        0,
        8,
        24,
    ]
    assert InstrumentedScheduler._bench_seed_prompt_len(stub, 16) == 24


def test_mamba_connector_uses_scheduler_per_group_cache_lookup():
    stub = _prefill_grid_stub(kv_read_granularity=3, block_size=8)
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


def test_prefill_kv_read_seed_drains_then_measures_with_same_salt():
    point = BenchmarkPoint(point_type="prefill", isl=40, kv_read_tokens=16)
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_grid = deque([point])
    stub._bench_config = SimpleNamespace(mode="prefill")
    stub._bench_active_req_ids = set()
    stub._bench_current_point = None
    stub._bench_current_fpms = []
    stub._bench_pending_seed_point = None
    stub._bench_pending_seed_salt = None
    stub._bench_drain_pending = False
    stub._bench_seq = 7
    stub._schedule_times = deque()
    stub.requests = {}

    calls = []

    def inject(**kwargs):
        calls.append(kwargs)
        stub._bench_active_req_ids.add(f"request-{len(calls)}")
        return 1

    stub._bench_inject_prefill = inject
    stub._bench_cleanup_requests = stub._bench_active_req_ids.clear

    InstrumentedScheduler._bench_step_prefill(stub)

    assert stub._bench_current_point is None
    assert stub._bench_pending_seed_point is point
    assert calls[0]["prompt_len"] == point.kv_read_tokens
    assert calls[0]["max_tokens"] == 1
    seed_salt = calls[0]["cache_salt"]

    # Simulate the seed finishing. The next step requests an async drain.
    stub.requests.clear()
    InstrumentedScheduler._bench_step_prefill(stub)
    assert stub._bench_drain_pending is True
    assert len(calls) == 1

    # After the drain, the full ISL request is measured against the seeded KV.
    InstrumentedScheduler._bench_step_prefill(stub)
    assert stub._bench_current_point is point
    assert calls[1] == {
        "prompt_len": point.isl,
        "max_tokens": 1,
        "cache_salt": seed_salt,
        "expected_kv_read_tokens": point.kv_read_tokens,
    }


def test_prefill_kv_read_validation_miss_skips_measured_point():
    point = BenchmarkPoint(point_type="prefill", isl=40, kv_read_tokens=16)
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_grid = deque()
    stub._bench_config = SimpleNamespace(mode="prefill")
    stub._bench_active_req_ids = set()
    stub._bench_current_point = None
    stub._bench_current_fpms = []
    stub._bench_pending_seed_point = point
    stub._bench_pending_seed_salt = "seed-salt"
    stub._bench_drain_pending = False
    stub._schedule_times = deque()
    stub._bench_skipped_points = []
    stub._bench_inject_prefill = MagicMock(return_value=0)

    InstrumentedScheduler._bench_step_prefill(stub)

    assert stub._bench_current_point is None
    assert stub._bench_pending_seed_point is None
    assert stub._bench_skipped_points[0].reason == "seed_cache_validation_failed"
    stub._bench_inject_prefill.assert_called_once_with(
        prompt_len=point.isl,
        max_tokens=1,
        cache_salt="seed-salt",
        expected_kv_read_tokens=point.kv_read_tokens,
    )


def test_benchmark_output_marks_skipped_kv_point_invalid(tmp_path):
    point = BenchmarkPoint(point_type="prefill", isl=40, kv_read_tokens=16)
    output_path = tmp_path / "benchmark.json"
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_config = BenchmarkConfig(output_path=str(output_path))
    stub._bench_expected_points = 1
    stub._bench_results = []
    stub._bench_skipped_points = [
        SkippedBenchmarkPoint(point=point, reason="seed_cache_validation_failed")
    ]
    stub.max_num_scheduled_tokens = 40
    stub.max_num_running_reqs = 8
    stub.max_model_len = 128
    stub.block_size = 8
    stub.cache_config = SimpleNamespace(num_gpu_blocks=64)

    InstrumentedScheduler._bench_write_results(stub)

    output = json.loads(output_path.read_text())
    assert output["valid"] is False
    assert output["coverage"] == {
        "expected_points": 1,
        "completed_points": 0,
        "skipped_points": 1,
    }
    assert output["skipped_points"] == [
        {"point": point.__dict__, "reason": "seed_cache_validation_failed"}
    ]


def test_prefill_point_with_measured_kv_mismatch_is_skipped():
    point = BenchmarkPoint(point_type="prefill", isl=40, kv_read_tokens=16)
    stub = InstrumentedScheduler.__new__(InstrumentedScheduler)
    stub._bench_current_point = point
    stub._bench_current_fpms = [{"scheduled_requests": {"sum_prefill_kv_tokens": 8}}]
    stub._bench_results = []
    stub._bench_skipped_points = []

    InstrumentedScheduler._bench_save_current_point(stub)

    assert stub._bench_results == []
    assert stub._bench_skipped_points == [
        SkippedBenchmarkPoint(point=point, reason="measured_kv_read_mismatch")
    ]
