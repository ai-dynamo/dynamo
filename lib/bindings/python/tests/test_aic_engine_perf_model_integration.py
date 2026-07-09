# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict native AIC coverage for the wheel's Mocker performance shim."""

from __future__ import annotations

import json

import pytest

pytestmark = [
    pytest.mark.aic_full,
    pytest.mark.aiconfigurator,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.mocker,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
]

AIC_MODEL = "Qwen/Qwen3-32B"
AIC_SYSTEM = "h200_sxm"
AIC_BACKEND_VERSION = "0.14.0"


@pytest.fixture(autouse=True)
def _offline_aic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _native_model(worker_type: str):
    from aiconfigurator.sdk.engine import compile_engine

    from dynamo.mocker import AicEngineConfig, EnginePerfLimits, RustEnginePerfModel

    assert callable(compile_engine)
    return RustEnginePerfModel.from_native(
        aic_config=AicEngineConfig(
            model_name=AIC_MODEL,
            backend="vllm",
            system_name=AIC_SYSTEM,
            backend_version=AIC_BACKEND_VERSION,
            tp_size=1,
            attention_dp_size=1,
        ),
        worker_type=worker_type,
        limits=EnginePerfLimits(
            max_num_batched_tokens=4096,
            max_num_seqs=128,
            max_kv_tokens=1_000_000,
        ),
    )


def test_native_aic_model_estimates_prefill_and_decode() -> None:
    from dynamo.common.forward_pass_metrics import (
        ForwardPassMetrics,
        ScheduledRequestMetrics,
    )
    from dynamo.mocker import EngineCapacityRequest

    prefill_model = _native_model("prefill")
    decode_model = _native_model("decode")

    prefill = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=1,
            sum_prefill_tokens=1024,
        )
    )
    decode = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=8,
            sum_decode_kv_tokens=8192,
        )
    )

    prefill_time = prefill_model.estimate_forward_pass_time([prefill])
    decode_time = decode_model.estimate_forward_pass_time([decode])
    assert prefill_time is not None and prefill_time > 0.0
    assert decode_time is not None and decode_time > 0.0

    for model in (prefill_model, decode_model):
        diagnostics = json.loads(model.diagnostics())
        assert diagnostics["source"] == "aic"
        assert diagnostics["readiness"] == "ready"
        assert diagnostics["last_warning"] is None

    capacity = decode_model.find_engine_capacity_rps(
        EngineCapacityRequest(isl=1024, osl=128, itl_sla_ms=1000.0)
    )
    assert capacity is not None
    assert capacity.rps > 0.0
    assert capacity.itl_ms is not None and capacity.itl_ms > 0.0
