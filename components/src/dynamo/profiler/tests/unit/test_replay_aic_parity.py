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
AIC_PARITY_VERSIONS = {
    "vllm": "0.14.0",
    "sglang": "0.5.6.post2",
}
AIC_PARITY_BACKENDS = [
    pytest.param("vllm", id="vllm"),
    pytest.param("sglang", id="sglang"),
]

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


def _aic_replay_args(backend_name: str):
    payload = {
        "block_size": 512,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 65536,
        "num_gpu_blocks": 100000,
        "speedup_ratio": 1.0,
        "aic_backend": backend_name,
        "aic_system": AIC_PARITY_SYSTEM,
        "aic_backend_version": AIC_PARITY_VERSIONS[backend_name],
        "aic_tp_size": 1,
        "aic_model_path": AIC_PARITY_MODEL,
    }
    if backend_name == "sglang":
        payload["engine_type"] = "sglang"
        payload["sglang"] = {
            "page_size": 512,
            "max_prefill_tokens": 65536,
            "chunked_prefill_size": 65536,
        }
    return MockEngineArgs.from_json(json.dumps(payload))


def _aic_disagg_replay_args(
    backend_name: str,
    *,
    tp_size: int,
    worker_type: str,
    max_num_seqs: int,
    max_num_batched_tokens: int,
):
    payload = {
        "block_size": 512,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "num_gpu_blocks": 50000,
        "speedup_ratio": 1.0,
        "aic_backend": backend_name,
        "aic_system": AIC_PARITY_SYSTEM,
        "aic_backend_version": AIC_PARITY_VERSIONS[backend_name],
        "aic_tp_size": tp_size,
        "aic_model_path": AIC_PARITY_MODEL,
        "worker_type": worker_type,
    }
    if backend_name == "sglang":
        payload["engine_type"] = "sglang"
        payload["sglang"] = {
            "page_size": 512,
            "max_prefill_tokens": 65536,
            "chunked_prefill_size": 65536,
        }
    return MockEngineArgs.from_json(json.dumps(payload))


def _run_aic_static_point(backend_name: str, isl: int, osl: int, batch_size: int):
    assert compile_engine

    database = get_database(
        system=AIC_PARITY_SYSTEM,
        backend=backend_name,
        version=AIC_PARITY_VERSIONS[backend_name],
    )
    backend = get_backend(backend_name)
    model = get_model(
        model_path=AIC_PARITY_MODEL,
        model_config=ModelConfig(tp_size=1),
        backend_name=backend_name,
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


@pytest.mark.parametrize("backend_name", AIC_PARITY_BACKENDS)
@pytest.mark.parametrize("isl", [256, 512, 1024, 2048, 4096])
def test_run_synthetic_concurrency_replay_matches_aic_static_point_no_prefix(
    backend_name, isl
):
    report = run_synthetic_trace_replay(
        isl,
        128,
        8,
        extra_engine_args=_aic_replay_args(backend_name),
        num_workers=1,
        replay_mode="offline",
        replay_concurrency=8,
        arrival_interval_ms=0.0,
    )
    aic = _run_aic_static_point(
        backend_name=backend_name,
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


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    (
        "backend_name",
        "isl",
        "osl",
        "request_count",
        "replay_concurrency",
        "total_gpu_budget",
        "prefill_tp",
        "decode_tp",
        "prefill_bs",
        "decode_bs",
        "prefill_workers",
        "decode_workers",
    ),
    [
        pytest.param(
            "vllm",
            1024,
            512,
            1440,
            720,
            20,
            1,
            2,
            1,
            120,
            6,
            5,
            id="vllm",
        ),
        pytest.param(
            "sglang",
            1024,
            512,
            2944,
            1472,
            24,
            2,
            2,
            1,
            184,
            6,
            6,
            id="sglang",
        ),
    ],
)
def test_run_synthetic_disagg_replay_preserves_aic_local_optimum(
    backend_name,
    isl,
    osl,
    request_count,
    replay_concurrency,
    total_gpu_budget,
    prefill_tp,
    decode_tp,
    prefill_bs,
    decode_bs,
    prefill_workers,
    decode_workers,
):
    prefill_args = _aic_disagg_replay_args(
        backend_name,
        tp_size=prefill_tp,
        worker_type="prefill",
        max_num_seqs=prefill_bs,
        max_num_batched_tokens=isl,
    )
    decode_args = _aic_disagg_replay_args(
        backend_name,
        tp_size=decode_tp,
        worker_type="decode",
        max_num_seqs=decode_bs,
        max_num_batched_tokens=200000,
    )

    variants = [
        ("picked", prefill_workers, decode_workers),
        ("p_minus_2_d_plus_2", prefill_workers - 2, decode_workers + 2),
        ("p_plus_2_d_minus_2", prefill_workers + 2, decode_workers - 2),
    ]
    reports = {}
    for variant_name, p_workers, d_workers in variants:
        report = run_synthetic_trace_replay(
            isl,
            osl,
            request_count,
            prefill_engine_args=prefill_args,
            decode_engine_args=decode_args,
            num_prefill_workers=p_workers,
            num_decode_workers=d_workers,
            replay_concurrency=replay_concurrency,
            replay_mode="offline",
            router_mode="round_robin",
            arrival_interval_ms=0.0,
        )
        reports[variant_name] = report["output_throughput_tok_s"] / total_gpu_budget

    assert reports["picked"] > reports["p_minus_2_d_plus_2"]
    assert reports["picked"] > reports["p_plus_2_d_minus_2"]
