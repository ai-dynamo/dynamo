# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.replay import run_synthetic_trace_replay

from .replay_utils import (
    AIC_PARITY_BACKENDS,
    _aic_disagg_replay_args,
    _aic_replay_args,
    _run_aic_static_point,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


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


@pytest.mark.timeout(30)
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
            marks=pytest.mark.vllm,
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
            marks=pytest.mark.sglang,
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
        is_prefill=True,
        max_num_seqs=prefill_bs,
        max_num_batched_tokens=isl,
    )
    decode_args = _aic_disagg_replay_args(
        backend_name,
        tp_size=decode_tp,
        is_prefill=False,
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
