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

from types import SimpleNamespace

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
from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler  # noqa: E402

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
    from unittest.mock import MagicMock

    from dynamo.vllm.instrumented_scheduler import _BenchPhase

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
    from unittest.mock import MagicMock

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
    from unittest.mock import MagicMock

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
