# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, network-free coverage of Dynamo replay against AIC 0.10."""

from __future__ import annotations

import pytest
from aiconfigurator.sdk.engine import compile_engine
from aiconfigurator.sdk.memory import estimate_num_gpu_blocks

from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay

pytestmark = [
    pytest.mark.aic_full,
    pytest.mark.aiconfigurator,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.mocker,
    pytest.mark.pre_merge,
]

AIC_MODEL = "Qwen/Qwen3-32B"
AIC_SYSTEM = "h200_sxm"
AIC_BACKEND_VERSION = "0.14.0"


@pytest.fixture(autouse=True)
def _offline_aic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _engine_args(worker_type: str | None = None) -> MockEngineArgs:
    worker_options = {"worker_type": worker_type} if worker_type is not None else {}
    return MockEngineArgs(
        aic_backend="vllm",
        aic_system=AIC_SYSTEM,
        aic_backend_version=AIC_BACKEND_VERSION,
        aic_model_path=AIC_MODEL,
        aic_tp_size=1,
        block_size=64,
        max_num_batched_tokens=4096,
        max_num_seqs=128,
        **worker_options,
    )


def test_real_aic_memory_estimates_gpu_blocks() -> None:
    blocks = estimate_num_gpu_blocks(
        model_path=AIC_MODEL,
        system=AIC_SYSTEM,
        backend="vllm",
        backend_version=AIC_BACKEND_VERSION,
        tp_size=1,
        scheduler_block_size=64,
        max_num_tokens=4096,
        max_batch_size=128,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
    )
    assert blocks > 0


def test_aggregated_replay_uses_native_aic_engine() -> None:
    assert compile_engine
    report = run_synthetic_trace_replay(
        128,
        8,
        2,
        extra_engine_args=_engine_args(),
        replay_concurrency=1,
        replay_mode="offline",
        arrival_interval_ms=0.0,
    )
    assert report["num_requests"] == 2
    assert report["mean_ttft_ms"] > 0.0
    assert report["mean_tpot_ms"] > 0.0


def test_disaggregated_replay_uses_native_aic_engine() -> None:
    report = run_synthetic_trace_replay(
        128,
        8,
        2,
        prefill_engine_args=_engine_args("prefill"),
        decode_engine_args=_engine_args("decode"),
        num_prefill_workers=1,
        num_decode_workers=1,
        replay_concurrency=1,
        replay_mode="offline",
        arrival_interval_ms=0.0,
    )
    assert report["num_requests"] == 2
    assert report["mean_ttft_ms"] > 0.0
    assert report["mean_tpot_ms"] > 0.0
