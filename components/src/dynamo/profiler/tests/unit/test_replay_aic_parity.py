# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.engine import compile_engine
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database

from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay

AIC_PARITY_MODEL = "Qwen/Qwen3-32B"
AIC_PARITY_SYSTEM = "h200_sxm"
AIC_PARITY_BACKEND = "vllm"
AIC_PARITY_VERSION = "0.14.0"

pytestmark = [
    pytest.mark.aic_full,
    pytest.mark.aiconfigurator,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture(autouse=True)
def _offline_aic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _aic_replay_args():
    payload = {
        "block_size": 512,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 65536,
        "num_gpu_blocks": 100000,
        "speedup_ratio": 1.0,
        "aic_backend": AIC_PARITY_BACKEND,
        "aic_system": AIC_PARITY_SYSTEM,
        "aic_backend_version": AIC_PARITY_VERSION,
        "aic_tp_size": 1,
        "aic_model_path": AIC_PARITY_MODEL,
    }
    return MockEngineArgs.from_json(json.dumps(payload))


def _run_aic_static_point(isl: int, osl: int, batch_size: int):
    assert compile_engine

    database = get_database(
        system=AIC_PARITY_SYSTEM,
        backend=AIC_PARITY_BACKEND,
        version=AIC_PARITY_VERSION,
    )
    backend = get_backend(AIC_PARITY_BACKEND)
    model = get_model(
        model_path=AIC_PARITY_MODEL,
        model_config=ModelConfig(tp_size=1),
        backend_name=AIC_PARITY_BACKEND,
    )
    session = InferenceSession(model, database, backend)
    summary = session.run_static(
        runtime_config=RuntimeConfig(
            batch_size=batch_size,
            beam_width=1,
            isl=isl,
            osl=osl,
            prefix=0,
        ),
        mode="static",
        stride=32,
    )
    return summary.get_summary_df().to_dict(orient="records")[0]


def test_run_synthetic_concurrency_replay_matches_aic_static_point_no_prefix():
    isl = 1024
    report = run_synthetic_trace_replay(
        isl,
        128,
        8,
        extra_engine_args=_aic_replay_args(),
        num_workers=1,
        replay_mode="offline",
        replay_concurrency=8,
        arrival_interval_ms=0.0,
    )
    aic = _run_aic_static_point(
        isl=isl,
        osl=128,
        batch_size=8,
    )
    expected_ttft_ms = aic["context_latency"] + aic["tpot"]

    assert report["mean_ttft_ms"] == pytest.approx(expected_ttft_ms, rel=0.05)
    assert report["mean_tpot_ms"] == pytest.approx(aic["tpot"], rel=0.05)
    assert report["output_throughput_tok_s"] == pytest.approx(
        aic["tokens/s/gpu"], rel=0.05
    )
