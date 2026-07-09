# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner integration coverage against the real AIC native estimator."""

from __future__ import annotations

import pytest
from aiconfigurator.sdk.engine import compile_engine

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.parallelization import PickedParallelConfig
from dynamo.planner.config.planner_config import AICPerfModelSpec, PlannerConfig
from dynamo.planner.core.perf_model.rust_adapter import PlannerEnginePerfModel
from dynamo.planner.core.types import EngineCapabilities

pytestmark = [
    pytest.mark.aic_full,
    pytest.mark.aiconfigurator,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _offline_aic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _config() -> PlannerConfig:
    pick = PickedParallelConfig()
    return PlannerConfig.model_construct(
        aic_perf_model=AICPerfModelSpec.model_construct(
            hf_id="Qwen/Qwen3-32B",
            system="h200_sxm",
            backend="vllm",
            backend_version="0.14.0",
            prefill_pick=pick,
            decode_pick=pick,
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


def test_planner_uses_native_aic_for_estimates_and_capacity() -> None:
    assert compile_engine
    prefill_model = PlannerEnginePerfModel(
        worker_type="prefill", config=_config(), capabilities=_capabilities()
    )
    decode_model = PlannerEnginePerfModel(
        worker_type="decode", config=_config(), capabilities=_capabilities()
    )

    for model in (prefill_model, decode_model):
        diagnostics = model._rust_diagnostics()
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

    prefill_ms = prefill_model.estimate_queued_prefill_time(
        [prefill_fpm], max_num_batched_tokens=4096, add_next_request=False
    )
    decode_ms = decode_model.estimate_scheduled_decode_itl(
        [decode_fpm], add_next_request=False
    )
    assert prefill_ms is not None and prefill_ms > 0.0
    assert decode_ms is not None and decode_ms > 0.0

    prefill_capacity = prefill_model.find_engine_capacity_rps(
        isl=1024, osl=128, ttft_sla_ms=10_000.0
    )
    decode_capacity = decode_model.find_engine_capacity_rps(
        isl=1024, osl=128, itl_sla_ms=1000.0
    )
    assert prefill_capacity is not None and prefill_capacity.rps > 0.0
    assert decode_capacity is not None and decode_capacity.rps > 0.0
