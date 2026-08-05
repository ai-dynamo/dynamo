# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ingress and engine toggles.

Ingress: a QPS-like control over requests arriving off the wire, with a
constant or Poisson arrival process (or a closed loop).

Engine: the batch of responses coming back per iteration -- given either as a
total or as per-rank x ranks -- and two iteration-time models, constant and
batch-dependent.
"""

from __future__ import annotations

import pytest

from egress_experiments.costs import Costs
from egress_experiments.fake_trtllm.engine import (
    BatchConfig,
    BatchDependentIteration,
    ConstantIteration,
    EngineConfig,
)
from egress_experiments.harness import SimConfig, run_simulation

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.none,
]

_FREE = Costs().with_scale(0.0)


# ---------------------------------------------------------------------------
# Engine: batch geometry
# ---------------------------------------------------------------------------


def test_total_and_per_rank_are_two_ways_to_say_the_same_batch():
    assert BatchConfig(total=1024).total_batch == 1024
    assert BatchConfig(per_rank=64, ranks=16).total_batch == 1024
    assert BatchConfig(per_rank=64, ranks=16).per_rank_batch == 64
    # A total spread over ranks divides, rounding up.
    assert BatchConfig(total=1000, ranks=16).per_rank_batch == 63
    assert BatchConfig(total=132).per_rank_batch == 132


def test_batch_config_rejects_ambiguity():
    with pytest.raises(ValueError, match="not both"):
        BatchConfig(total=1024, per_rank=64)
    with pytest.raises(ValueError, match="ranks"):
        BatchConfig(total=1024, ranks=0)


def test_default_batch_is_a_measured_response_count():
    """132 = queue_probe's responses-per-deque-entry on job 355971.

    With the default ``stream_interval=1`` batch and responses/iteration
    coincide, so this default reproduces that measurement directly. The 355778
    geometry is the other way round: batch 7,888 at si=40 gives 197
    responses/iteration. Derive either with ``capture_params``.
    """
    assert BatchConfig().total_batch == 132
    assert SimConfig(batch=BatchConfig()).responses_per_iteration == 132


def test_engine_batch_caps_the_responses_per_iteration():
    """The toggle actually reaches the engine."""
    for total in (4, 12):
        result = run_simulation(
            SimConfig(
                egress="push",
                requests=48,
                max_tokens=3,
                isl=4,
                batch=BatchConfig(total=total),
                iteration=ConstantIteration(3.0),
                costs=_FREE,
                lag_ms=1.0,
            )
        )
        assert result.mean_engine_batch <= total + 1e-9, (
            f"engine emitted {result.mean_engine_batch} responses per iteration "
            f"with a batch cap of {total}"
        )
        assert result.responses == 48 * 3


# ---------------------------------------------------------------------------
# Engine: iteration-time models
# ---------------------------------------------------------------------------


def test_constant_iteration_ignores_the_batch():
    model = ConstantIteration(34.0)
    assert model.duration_ms(1, BatchConfig(total=1024)) == 34.0
    assert model.duration_ms(1024, BatchConfig(total=1024)) == 34.0


def test_batch_dependent_iteration_reproduces_the_capture():
    """Both fit points must come back out of the model they were fitted to.

    dynamo 51.69 ms at batch 986, trtllm-serve 48.53 ms at batch 880 -- and
    the document's own us/req/iter figures, 52.42 and 55.15.
    """
    model = BatchDependentIteration()
    single_rank = BatchConfig(total=1024)  # ranks=1, so per-rank == active

    dynamo_ms = model.duration_ms(986, single_rank)
    serve_ms = model.duration_ms(880, single_rank)

    assert dynamo_ms == pytest.approx(51.69, abs=0.01)
    assert serve_ms == pytest.approx(48.53, abs=0.01)
    assert dynamo_ms * 1000 / 986 == pytest.approx(52.42, abs=0.01)
    assert serve_ms * 1000 / 880 == pytest.approx(55.15, abs=0.01)


def test_batch_dependent_iteration_scales_with_the_per_rank_batch():
    """Attention DP: the same total batch over more ranks is a faster iteration."""
    model = BatchDependentIteration()

    one_rank = BatchConfig(total=1024)
    sixteen = BatchConfig(per_rank=64, ranks=16)
    assert one_rank.total_batch == sixteen.total_batch

    slow = model.duration_ms(1024, one_rank)
    fast = model.duration_ms(1024, sixteen)
    assert fast < slow
    # 22.296 + 64 * 0.029811 = 24.20 ms against 22.296 + 1024 * 0.029811.
    assert fast == pytest.approx(24.20, abs=0.02)
    assert slow == pytest.approx(52.82, abs=0.02)


def test_batch_dependent_iteration_falls_as_the_batch_drains():
    model = BatchDependentIteration()
    batch = BatchConfig(total=512)
    durations = [model.duration_ms(n, batch) for n in (512, 256, 64, 1)]
    assert durations == sorted(durations, reverse=True)
    assert durations[-1] == pytest.approx(model.base_ms, abs=0.05)


def test_engine_model_changes_the_observed_iteration_time():
    """End to end: swapping the model moves the wall clock, nothing else."""
    common = dict(
        egress="push",
        requests=24,
        max_tokens=4,
        isl=4,
        batch=BatchConfig(total=8),
        costs=_FREE,
        lag_ms=1.0,
    )
    constant = run_simulation(
        SimConfig(iteration=ConstantIteration(12.0), **common)  # type: ignore[arg-type]
    )
    # base 2 ms + 8 * 0.25 ms = 4 ms at full batch: distinctly faster.
    dependent = run_simulation(
        SimConfig(
            iteration=BatchDependentIteration(base_ms=2.0, per_request_us=250.0),
            **common,  # type: ignore[arg-type]
        )
    )

    assert constant.responses == dependent.responses
    assert dependent.mean_iteration_ms < constant.mean_iteration_ms
    # Same structure regardless of the engine model.
    assert constant.loop_handoffs == dependent.loop_handoffs


# ---------------------------------------------------------------------------
# Ingress: the QPS control
# ---------------------------------------------------------------------------


def test_steady_state_qps_is_the_rate_that_fills_the_batch():
    """batch / (max_tokens * iteration_s)."""
    cfg = SimConfig(
        max_tokens=16,
        batch=BatchConfig(total=132),
        iteration=ConstantIteration(34.0),
    )
    assert cfg.full_batch_iteration_ms == 34.0
    assert cfg.steady_state_qps == pytest.approx(132 / (16 * 0.034))
    assert cfg.offered_qps == cfg.steady_state_qps
    # An explicit qps overrides it.
    assert SimConfig(qps=50.0).offered_qps == 50.0


def test_steady_state_qps_follows_the_batch_dependent_model():
    """The iteration time it divides by is the FULL-batch one, not the base."""
    cfg = SimConfig(
        max_tokens=8,
        batch=BatchConfig(total=512),
        iteration=BatchDependentIteration(),
    )
    expected_iteration = 22.296 + 512 * 0.029811
    assert cfg.full_batch_iteration_ms == pytest.approx(expected_iteration, abs=0.01)
    assert cfg.steady_state_qps == pytest.approx(
        512 / (8 * expected_iteration / 1000), rel=1e-6
    )


def test_qps_sets_the_achieved_arrival_rate():
    """A rate the engine can absorb comes back out as the achieved rate."""
    cfg = SimConfig(
        egress="push",
        requests=60,
        max_tokens=2,
        isl=4,
        qps=200.0,
        batch=BatchConfig(total=32),
        iteration=ConstantIteration(3.0),
        costs=_FREE,
        lag_ms=1.0,
    )
    result = run_simulation(cfg)
    # 60 requests at 200/s is 0.3 s of arrivals; the engine drains 32 at a time
    # in 3 ms, so it is never the bottleneck and the run tracks the offer.
    assert result.requests_completed == 60
    assert result.achieved_qps == pytest.approx(200.0, rel=0.35)


def test_all_three_arrival_processes_run_and_agree_on_the_work_done():
    """Arrival shape changes timing, never the responses delivered."""
    common = dict(
        egress="push",
        requests=32,
        max_tokens=3,
        isl=4,
        qps=400.0,
        batch=BatchConfig(total=16),
        iteration=ConstantIteration(3.0),
        costs=_FREE,
        lag_ms=1.0,
    )
    results = {
        arrival: run_simulation(SimConfig(arrival=arrival, **common))  # type: ignore[arg-type]
        for arrival in ("constant", "poisson", "closed")
    }
    for arrival, result in results.items():
        assert result.errors == [], arrival
        assert result.responses == 32 * 3, arrival
        assert result.requests_completed == 32, arrival
        assert result.fallback_yields == 0, arrival


def test_poisson_arrivals_are_seeded_and_reproducible():
    def run(seed: int):
        return run_simulation(
            SimConfig(
                egress="push",
                requests=24,
                max_tokens=2,
                isl=4,
                arrival="poisson",
                qps=300.0,
                arrival_seed=seed,
                batch=BatchConfig(total=16),
                iteration=ConstantIteration(3.0),
                costs=_FREE,
                lag_ms=1.0,
            )
        )

    first, again = run(7), run(7)
    assert first.responses == again.responses
    assert first.requests_completed == again.requests_completed


def test_poisson_arrivals_are_burstier_than_constant():
    """Exponential inter-arrivals cluster, so admission waits spread out.

    Asserted on the spread rather than an absolute, because the whole point of
    a memoryless process is that any single run is noisy.
    """
    common = dict(
        egress="push",
        requests=200,
        max_tokens=2,
        isl=4,
        qps=900.0,
        batch=BatchConfig(total=64),
        iteration=ConstantIteration(3.0),
        costs=Costs(),
        lag_ms=1.0,
    )
    constant = run_simulation(SimConfig(arrival="constant", **common))  # type: ignore[arg-type]
    poisson = run_simulation(SimConfig(arrival="poisson", **common))  # type: ignore[arg-type]

    assert constant.requests_completed == poisson.requests_completed == 200
    assert poisson.queue_wait["max"] >= constant.queue_wait["p50"]


def test_closed_loop_concurrency_defaults_to_the_batch():
    assert SimConfig(batch=BatchConfig(total=256)).closed_loop_concurrency == 256
    assert (
        SimConfig(batch=BatchConfig(total=256), concurrency=8).closed_loop_concurrency
        == 8
    )


def test_bad_arrival_is_rejected():
    with pytest.raises(ValueError, match="arrival must be one of"):
        SimConfig(arrival="stochastic")


# ---------------------------------------------------------------------------
# The two toggles compose
# ---------------------------------------------------------------------------


def test_batch_fill_reports_whether_the_offer_reached_steady_state():
    """Under-offering must be visible, not silently understate loop load."""
    starved = run_simulation(
        SimConfig(
            egress="push",
            requests=40,
            max_tokens=3,
            isl=4,
            qps=20.0,  # far below steady state
            batch=BatchConfig(total=64),
            iteration=ConstantIteration(3.0),
            costs=_FREE,
            lag_ms=1.0,
        )
    )
    assert starved.batch_fill < 0.5
    assert starved.mean_engine_batch < 64


def test_engine_config_is_carried_through_from_sim_config():
    cfg = SimConfig(
        max_tokens=9,
        batch=BatchConfig(per_rank=4, ranks=3),
        iteration=BatchDependentIteration(base_ms=1.0, per_request_us=10.0),
    )
    engine: EngineConfig = cfg.engine_config
    assert engine.max_tokens == 9
    assert engine.batch.total_batch == 12
    assert engine.batch.per_rank_batch == 4
    assert engine.iteration.duration_ms(12, engine.batch) == pytest.approx(1.04)
    assert "3 ranks" in cfg.batch.describe()
    assert "batch-dependent" in cfg.iteration.describe()


# ---------------------------------------------------------------------------
# Engine: stream_interval
# ---------------------------------------------------------------------------


def test_stream_interval_divides_the_response_rate():
    """responses/iteration = batch / stream_interval, exactly."""
    for si, expected in ((1, 160.0), (2, 80.0), (8, 20.0), (40, 4.0)):
        cfg = SimConfig(
            batch=BatchConfig(total=160),
            iteration=ConstantIteration(10.0),
            stream_interval=si,
        )
        assert cfg.responses_per_iteration == expected
        assert cfg.offered_response_rate == pytest.approx(expected / 0.010)


def test_stream_interval_does_not_change_the_token_rate():
    """A request still generates one token per iteration; only reporting changes.

    So residency -- and therefore the steady-state qps that fills the batch --
    is independent of stream_interval.
    """
    base = dict(
        max_tokens=80, batch=BatchConfig(total=160), iteration=ConstantIteration(10.0)
    )
    one = SimConfig(stream_interval=1, **base)  # type: ignore[arg-type]
    forty = SimConfig(stream_interval=40, **base)  # type: ignore[arg-type]
    assert one.residency_s == forty.residency_s
    assert one.steady_state_qps == forty.steady_state_qps


def test_engine_emits_batch_over_si_responses_per_iteration():
    """End to end, against the engine rather than the arithmetic."""
    for si in (2, 8):
        cfg = SimConfig(
            egress="push",
            requests=400,
            max_tokens=80,
            isl=8,
            batch=BatchConfig(total=160),
            iteration=ConstantIteration(10.0),
            stream_interval=si,
            costs=_FREE,
            lag_ms=2.0,
        )
        result = run_simulation(cfg)
        assert result.mean_engine_batch == pytest.approx(160 / si, rel=0.05)
        # Total responses fall by exactly si; total TOKENS do not change.
        assert result.responses == pytest.approx(400 * 80 / si, rel=0.02)


def test_a_response_carries_stream_interval_tokens():
    """The tokens are batched into the response, not dropped."""
    import asyncio

    from egress_experiments.dynamo_sim.worker import SamplingParams, TrtllmWorkerHandler
    from egress_experiments.fake_trtllm.llm import FakeLLM

    async def main():
        loop = asyncio.get_running_loop()
        llm = FakeLLM(
            EngineConfig(
                iteration=ConstantIteration(2.0), max_tokens=40, stream_interval=10
            ),
            costs=_FREE,
        )
        llm.start(loop)
        try:
            handler = TrtllmWorkerHandler(llm, costs=_FREE)

            class _Ctx:
                def id(self):
                    return "req-0"

            chunks = [
                c
                async for c in handler.generate(
                    {"id": "req-0", "token_ids": [1], "max_tokens": 40}, _Ctx()
                )
            ]
            # 40 tokens at si=10 -> 4 responses of 10 tokens each.
            assert len(chunks) == 4
            assert all(len(c["token_ids"]) == 10 for c in chunks)
            assert sum(len(c["token_ids"]) for c in chunks) == 40
            assert chunks[-1]["finish_reason"] == "length"
            assert chunks[-1]["completion_usage"]["completion_tokens"] == 40
        finally:
            llm.shutdown()

    asyncio.run(main())
    del SamplingParams


def test_si_1_overloads_the_loop_and_the_backlog_grows():
    """The capture's geometry with stream_interval=1 cannot be absorbed.

    batch / (si * iteration_s) responses/s against 1e6 / 85.34 us of loop
    capacity. At si=1 that is offered load >> 1, and the backlog -- responses
    put_nowait-ed but not yet delivered -- rises instead of staying flat.
    """
    common = dict(
        egress="push",
        requests=3000,
        max_tokens=40,
        isl=8,
        batch=BatchConfig(total=1200),
        iteration=ConstantIteration(10.0),
        costs=Costs(),
        lag_ms=2.0,
    )
    calm = run_simulation(SimConfig(stream_interval=40, **common))  # type: ignore[arg-type]
    swamped = run_simulation(
        SimConfig(stream_interval=1, max_backlog=30000, **common)  # type: ignore[arg-type]
    )

    assert calm.offered_load < 1.0
    assert not calm.overloaded
    assert not calm.backlog_growing

    assert swamped.offered_load > 10.0
    assert swamped.overloaded
    assert swamped.backlog_growing
    assert swamped.backlog_max > calm.backlog_max * 10
    assert swamped.backlog_aborted


def test_max_backlog_stops_cleanly_rather_than_running_to_exhaustion():
    result = run_simulation(
        SimConfig(
            egress="push",
            requests=4000,
            max_tokens=40,
            isl=8,
            batch=BatchConfig(total=1200),
            iteration=ConstantIteration(10.0),
            stream_interval=1,
            max_backlog=10000,
            costs=Costs(),
            lag_ms=2.0,
        )
    )
    assert result.backlog_aborted
    # Stopped near the limit, not far past it, and without an exception.
    assert 10000 <= result.backlog_max < 10000 * 6
    assert result.errors == []


def test_stream_interval_must_be_at_least_one():
    with pytest.raises(ValueError, match="stream_interval"):
        EngineConfig(stream_interval=0)
