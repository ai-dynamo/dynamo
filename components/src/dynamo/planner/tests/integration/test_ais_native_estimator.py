# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner integration coverage against the real AISimulate native estimator."""

from __future__ import annotations

import math

import pytest
from aisimulate_core.sdk.engine import EngineHandle, compile_engine

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import AISPerfModelSpec, PlannerConfig
from dynamo.planner.core.perf_model.ais_adapter import PlannerEnginePerfModel
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    TickInput,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter

pytestmark = [
    pytest.mark.aiconfigurator,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _offline_ais(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _config() -> PlannerConfig:
    return PlannerConfig.model_construct(
        ais_perf_model=AISPerfModelSpec(
            roles={
                role: {
                    "model": "Qwen/Qwen3-32B",
                    "system": "h200_sxm",
                    "backend": "vllm",
                    "backend_version": "current",
                    "worker_type": role,
                }
                for role in ("prefill", "decode")
            }
        ),
        max_num_fpm_samples=16,
        load_min_observations=5,
        fpm_sample_bucket_size=16,
        ttft_ms=500.0,
        itl_ms=50.0,
        speculative_nextn=0,
    )


def _capabilities() -> EngineCapabilities:
    return EngineCapabilities(
        max_num_batched_tokens=4096,
        max_num_seqs=128,
        max_kv_tokens=1_000_000,
        kv_cache_block_size=16,
    )


def test_planner_uses_native_ais_for_estimates_and_capacity() -> None:
    assert compile_engine
    prefill_model = PlannerEnginePerfModel(
        worker_type="prefill", config=_config(), capabilities=_capabilities()
    )
    decode_model = PlannerEnginePerfModel(
        worker_type="decode", config=_config(), capabilities=_capabilities()
    )

    for model in (prefill_model, decode_model):
        diagnostics = model._engine_diagnostics()
        assert diagnostics["source"] == "aic"
        assert diagnostics["readiness"] == "ready"
        assert diagnostics["last_warning"] is None
        assert model.has_sufficient_data()

    prefill_fpm = ForwardPassMetrics(
        worker_id="prefill-0",
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=1,
            sum_prefill_tokens=1024,
        ),
    )
    decode_fpm = ForwardPassMetrics(
        worker_id="decode-0",
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=8,
            sum_decode_kv_tokens=8192,
        ),
    )

    prefill_s = prefill_model.estimate_queued_prefill_time(
        [prefill_fpm], max_num_batched_tokens=4096, add_next_request=False
    )
    decode_s = decode_model.estimate_scheduled_decode_itl(
        [decode_fpm], add_next_request=False
    )
    assert prefill_s is not None and prefill_s > 0.0
    assert decode_s is not None and decode_s > 0.0

    prefill_capacity = prefill_model.find_engine_capacity_rps(
        isl=1024, osl=128, ttft_sla_ms=10_000.0
    )
    decode_capacity = decode_model.find_engine_capacity_rps(
        isl=1024, osl=128, itl_sla_ms=1000.0
    )
    assert prefill_capacity is not None and prefill_capacity.rps > 0.0
    assert decode_capacity is not None and decode_capacity.rps > 0.0


def test_planner_uses_real_ais_regression_fallback_after_tuning() -> None:
    config = PlannerConfig.model_construct(
        ais_perf_model=None,
        max_num_fpm_samples=16,
        load_min_observations=2,
        fpm_sample_bucket_size=16,
        speculative_nextn=0,
    )
    model = PlannerEnginePerfModel(
        worker_type="decode",
        config=config,
        capabilities=_capabilities(),
    )

    diagnostics = model._engine_diagnostics()
    assert diagnostics["source"] == "fallback_regression"
    assert diagnostics["readiness"] == "insufficient_data"

    for counter_id, (requests, kv_tokens, wall_time) in enumerate(
        ((1, 100, 0.01), (2, 200, 0.02)),
        start=1,
    ):
        model.add_observations(
            {
                ("decode-0", 0): ForwardPassMetrics(
                    worker_id="decode-0",
                    counter_id=counter_id,
                    wall_time=wall_time,
                    scheduled_requests=ScheduledRequestMetrics(
                        num_decode_requests=requests,
                        sum_decode_kv_tokens=kv_tokens,
                    ),
                )
            }
        )

    diagnostics = model._engine_diagnostics()
    assert diagnostics["source"] == "fallback_regression"
    assert diagnostics["readiness"] == "ready"
    assert diagnostics["retained_observations"] == 2

    itl_s = model.estimate_scheduled_decode_itl(
        [
            ForwardPassMetrics(
                worker_id="decode-0",
                scheduled_requests=ScheduledRequestMetrics(
                    num_decode_requests=1,
                    sum_decode_kv_tokens=100,
                ),
            )
        ],
        add_next_request=False,
    )
    assert itl_s == pytest.approx(0.01, rel=1e-6)


def test_ngram_worker_depth_preserves_native_speculation_identity() -> None:
    config = _config()
    config.ais_perf_model.roles["decode"]["speculation"] = {
        "kind": "ngram",
        "params": {"num_speculative_tokens": 2},
    }
    capabilities = _capabilities()
    model = PlannerEnginePerfModel(
        worker_type="decode", config=config, capabilities=capabilities
    )
    assert model.has_sufficient_data()

    capabilities.speculative_nextn = 2
    model.update_capabilities(capabilities)
    assert model.has_sufficient_data()
    identity = model._engine_diagnostics()["provenance"]["config"]
    assert identity["nextn"] == 0
    assert identity["speculation"]["params"]["num_speculative_tokens"] == 2

    capabilities.speculative_nextn = 3
    with pytest.raises(ValueError, match="speculation depth conflicts"):
        model.update_capabilities(capabilities)


def _speculative_model(role="decode", *, ngram=True):
    config = _config()
    identity = {
        **config.ais_perf_model.roles["decode"],
        "backend_version": "0.24.0",
        "worker_type": role,
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
    }
    if ngram:
        identity["speculation"] = {
            "kind": "ngram",
            "params": {"num_speculative_tokens": 3},
        }
    config.ais_perf_model = AISPerfModelSpec(roles={role: identity})
    config.load_min_observations = 2
    caps = EngineCapabilities(
        max_num_batched_tokens=16,
        max_num_seqs=16,
        max_kv_tokens=32768,
        kv_cache_block_size=16,
        speculative_nextn=3,
    )
    return (
        PlannerEnginePerfModel(worker_type=role, config=config, capabilities=caps),
        config,
        caps,
    )


def _decode_fpm(count=1):
    return ForwardPassMetrics(
        worker_id="decode-0",
        counter_id=1,
        wall_time=0.01,
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=count, sum_decode_kv_tokens=count * 128
        ),
    )


@pytest.mark.parametrize("role", ["decode", "aggregated"])
def test_speculative_op_prediction_survives_unqualified_decode_observations(
    role,
) -> None:
    model, _, _ = _speculative_model(role)
    fpm = _decode_fpm()
    before = model.estimate_scheduled_decode_itl([fpm], add_next_request=False)
    assert before is not None and before > 0
    model.add_observations({("decode-0", 0): fpm})
    assert model._engine_diagnostics()["retained_observations"] == 0
    assert model.has_sufficient_data()
    assert model.estimate_scheduled_decode_itl(
        [fpm], add_next_request=False
    ) == pytest.approx(before)
    capacity = model.find_engine_capacity_rps(isl=64, osl=64, accept_length=4)
    assert capacity is not None and capacity.rps > 0


def test_speculative_op_divisible_batch_matches_complete_native_round() -> None:
    model, _, _ = _speculative_model()
    actual = model.estimate_scheduled_decode_itl(
        [_decode_fpm(4)], add_next_request=False
    )
    oracle = EngineHandle.compile(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "vllm",
        backend_version="0.24.0",
        kv_block_size=16,
        worker_type="decode",
        speculation={"kind": "ngram", "params": {"num_speculative_tokens": 3}},
    )
    # Native context is ISL + OSL//2 + 1: 126 + 2//2 + 1 = 128 KV/request.
    # Both queries represent four logical requests, each with four verify positions.
    expected_ms = oracle.decode_step_latency(4, 126, 2)
    assert actual == pytest.approx(expected_ms / 1000, rel=1e-9)


def test_speculative_op_pure_prefill_correction_remains_available() -> None:
    model, _, _ = _speculative_model("prefill")
    scheduled = ScheduledRequestMetrics(num_prefill_requests=1, sum_prefill_tokens=8)
    sample = ForwardPassMetrics(
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=1, sum_prefill_tokens=8
        )
    )
    before = model.estimate_queued_prefill_time(
        [sample], max_num_batched_tokens=16, add_next_request=False
    )
    assert before is not None and before > 0
    for counter in (1, 2):
        model.add_observations(
            {
                ("prefill-0", 0): ForwardPassMetrics(
                    worker_id="prefill-0",
                    counter_id=counter,
                    scheduled_requests=scheduled,
                    wall_time=before * 1.25,
                )
            }
        )
    after = model.estimate_queued_prefill_time(
        [sample], max_num_batched_tokens=16, add_next_request=False
    )
    assert after == pytest.approx(before * 1.25, rel=1e-6)


def test_withheld_speculative_history_replays_raw_after_same_width_mode_change() -> (
    None
):
    model, config, caps = _speculative_model(ngram=False)
    fpm = _decode_fpm()
    model.add_observations({("decode-0", 0): fpm})
    assert model._engine_diagnostics()["retained_observations"] == 0
    assert not model._pending_iterations
    config.ais_perf_model.roles["decode"]["estimation_mode"] = "fpm_regression"
    model.update_capabilities(caps)
    assert model._engine_diagnostics()["source"] == "fallback_regression"
    assert model._engine_diagnostics()["retained_observations"] == 1
    assert model._retained_iterations == [[fpm]]
    assert not model._pending_iterations


def test_regression_width_change_discards_durations_but_same_width_replays() -> None:
    model, config, caps = _speculative_model(ngram=False)
    config.ais_perf_model.roles["decode"]["estimation_mode"] = "fpm_regression"
    model.update_capabilities(caps)

    def learn(target, duration):
        for count in (1, 2):
            sample = ForwardPassMetrics(
                worker_id="decode-0",
                counter_id=count,
                wall_time=count * duration,
                scheduled_requests=_decode_fpm(count).scheduled_requests,
            )
            target.add_observations({("decode-0", 0): sample})

    learn(model, 0.01)
    query = [_decode_fpm()]
    old_prediction = model.estimate_scheduled_decode_itl(query, add_next_request=False)
    assert old_prediction == pytest.approx(0.01, rel=1e-6)
    caps.max_num_seqs += 1
    model.update_capabilities(caps)
    assert model.has_sufficient_data()
    assert model.num_observations == 2
    assert model.estimate_scheduled_decode_itl(
        query, add_next_request=False
    ) == pytest.approx(old_prediction)

    caps.speculative_nextn = 1
    model.update_capabilities(caps)
    assert not model.has_sufficient_data()
    assert model.num_observations == 0
    assert model.estimate_scheduled_decode_itl(query, add_next_request=False) is None
    learn(model, 0.02)
    clean = PlannerEnginePerfModel(
        worker_type="decode", config=config, capabilities=caps
    )
    learn(clean, 0.02)
    assert model.has_sufficient_data()
    assert model.num_observations == 2
    assert model.estimate_scheduled_decode_itl(
        query, add_next_request=False
    ) == pytest.approx(
        clean.estimate_scheduled_decode_itl(query, add_next_request=False), rel=1e-6
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment,loads,queued_prefill,mode,expected,reason",
    [
        ("agg", [(3, 2), (3, 2)], 0, "op_level", 3, "scale_up"),
        ("agg", [(3, 2), (1, 0)], 0, "op_level", 2, "scale_down_refused_consolidation"),
        ("agg", [(2, 0), (2, 0)], 1, "op_level", 2, "scale_down_refused_consolidation"),
        ("agg", [(3, 2), (3, 2)], 0, "fpm_regression", 2, "insufficient_data"),
        ("agg", [(4, 0), (4, 0)], 1, "op_level", 3, "scale_up"),
        ("decode", [(3, 2), (3, 2)], 0, "op_level", 3, "scale_up"),
    ],
    ids=[
        "all-saturated",
        "mixed-healthy-saturated",
        "post-only-saturated",
        "cold-model",
        "prefill-no-headroom",
        "decode-only",
    ],
)
async def test_native_budget_outcome_reaches_load_scaling(
    deployment, loads, queued_prefill, mode, expected, reason
):
    # The live schedule (3 * W4 <= B16) is legal. Including two queued decodes
    # exceeds one verification round; it must retain a load vote, not become None.
    role = "aggregated" if deployment == "agg" else "decode"
    _, role_config, caps = _speculative_model(role, ngram=False)
    role_config.ais_perf_model.roles[role]["estimation_mode"] = mode
    config = PlannerConfig(
        mode=deployment,
        served_model_name="test",
        optimization_target="sla",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        ttft_ms=1000,
        itl_ms=1000,
        load_min_observations=5,
        ais_perf_model=role_config.ais_perf_model,
    )
    caps.num_gpu = 1
    adapter = OrchestratorEngineAdapter(
        config, WorkerCapabilities(decode=caps), clock=VirtualClock()
    )
    tick = adapter.initial_tick(start_s=0)
    assert tick.need_worker_fpm
    fpms = {}
    for index, (scheduled, queued) in enumerate(loads):
        worker = f"decode-{index}"
        fpms[(worker, 0)] = ForwardPassMetrics(
            worker_id=worker,
            counter_id=1,
            wall_time=0.01,
            scheduled_requests=ScheduledRequestMetrics(
                num_decode_requests=scheduled,
                sum_decode_kv_tokens=scheduled * 128,
            ),
            queued_requests=QueuedRequestMetrics(
                num_decode_requests=queued,
                sum_decode_kv_tokens=queued * 128,
                num_prefill_requests=int(queued_prefill > 0),
                sum_prefill_tokens=queued_prefill,
            ),
        )
    effects = await adapter.tick(
        tick,
        TickInput(
            now_s=tick.at_s,
            worker_counts=WorkerCounts(ready_num_decode=2, expected_num_decode=2),
            fpm_observations=FpmObservations(decode=fpms),
        ),
    )
    desired = effects.scale_to.num_decode if effects.scale_to is not None else 2
    assert desired == expected
    assert effects.diagnostics.load_decision_reason == reason
    for latency in (
        effects.diagnostics.estimated_ttft_ms,
        effects.diagnostics.estimated_itl_ms,
    ):
        assert latency is None or (math.isfinite(latency) and latency >= 0)
