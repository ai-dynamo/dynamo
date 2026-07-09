# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

import dynamo.replay.main as replay_main
from dynamo.mocker import MockEngineArgs

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_load_engine_args_estimates_aic_blocks(monkeypatch):
    calls = []

    def fake_estimate_num_gpu_blocks(**kwargs):
        calls.append(kwargs)
        return 46000

    monkeypatch.setattr(
        replay_main, "estimate_num_gpu_blocks", fake_estimate_num_gpu_blocks
    )

    engine_args = replay_main._load_engine_args(
        json.dumps(
            {
                "aic_backend": "vllm",
                "aic_system": "h200_sxm",
                "aic_model_path": "/models/mock",
                "aic_tp_size": 4,
                "block_size": 64,
                "max_num_batched_tokens": 4096,
                "gpu_memory_utilization": 0.8,
            }
        )
    )

    assert engine_args.num_gpu_blocks == 46000
    assert calls == [
        {
            "backend_name": "vllm",
            "system": "h200_sxm",
            "model_path": "/models/mock",
            "tp_size": 4,
            "block_size": 64,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.8,
            "mem_fraction_static": 0.88,
            "free_gpu_memory_fraction": None,
            "backend_version": None,
            "moe_tp_size": None,
            "moe_ep_size": None,
            "attention_dp_size": None,
            "gemm_dtype": None,
            "moe_dtype": None,
            "fmha_dtype": None,
            "kv_cache_dtype": None,
            "comm_dtype": None,
        }
    ]


def test_resolve_aic_blocks_preserves_explicit_zero_inputs(monkeypatch):
    calls = []

    def fake_estimate_num_gpu_blocks(**kwargs):
        calls.append(kwargs)
        return 46000

    monkeypatch.setattr(
        replay_main, "estimate_num_gpu_blocks", fake_estimate_num_gpu_blocks
    )

    raw = {
        "engine_type": "sglang",
        "aic_backend": "sglang",
        "aic_model_path": "/models/mock",
        "aic_tp_size": 0,
        "block_size": 0,
        "max_num_batched_tokens": 0,
        "gpu_memory_utilization": 0.0,
        "mem_fraction_static": 0.0,
        "free_gpu_memory_fraction": 0.0,
        "sglang": {"page_size": 0},
    }

    replay_main._resolve_aic_num_gpu_blocks(raw)

    assert raw["num_gpu_blocks"] == 46000
    assert calls[0]["tp_size"] == 0
    assert calls[0]["block_size"] == 0
    assert calls[0]["max_num_batched_tokens"] == 0
    assert calls[0]["gpu_memory_utilization"] == 0.0
    assert calls[0]["mem_fraction_static"] == 0.0
    assert calls[0]["free_gpu_memory_fraction"] == 0.0


def test_resolve_aic_blocks_scales_engine_pool_by_attention_dp(monkeypatch):
    # estimate_num_gpu_blocks returns a PER-RANK count; offline replay models a single KV
    # pool per engine, so _resolve_aic_num_gpu_blocks scales it by attention_dp_size to the
    # engine-wide pool (under DP-attention each rank holds a full KV replica). dp=1/unset is
    # unchanged. (The live mocker replicates one scheduler per dp rank, so it keeps per-rank
    # -- this scaling is offline-only.) Regression for DP-attention KV under-provisioning.
    monkeypatch.setattr(replay_main, "estimate_num_gpu_blocks", lambda **kw: 1000)

    def _resolve(dp):
        raw = {
            "aic_backend": "vllm",
            "aic_model_path": "/models/mock",
            "aic_tp_size": 1,
            "block_size": 64,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.8,
        }
        if dp is not None:
            raw["aic_attention_dp_size"] = dp
        replay_main._resolve_aic_num_gpu_blocks(raw)
        return raw["num_gpu_blocks"]

    assert _resolve(8) == 8000  # dp=8 -> engine pool is 8x the per-rank estimate
    assert _resolve(1) == 1000  # no DP-attention -> per-rank unchanged
    assert _resolve(None) == 1000  # unset -> per-rank unchanged


def test_invalid_json_num_gpu_blocks_type_is_rejected():
    with pytest.raises(Exception, match="num_gpu_blocks"):
        MockEngineArgs.from_json(
            json.dumps(
                {
                    "aic_backend": "vllm",
                    "aic_system": "h200_sxm",
                    "aic_model_path": "/models/mock",
                    "num_gpu_blocks": "bad",
                }
            )
        )


def test_memory_fraction_setters_validate_range():
    engine_args = MockEngineArgs(gpu_memory_utilization=0.8, mem_fraction_static=0.7)

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        engine_args.gpu_memory_utilization = 1.1
    assert engine_args.gpu_memory_utilization == 0.8

    with pytest.raises(ValueError, match="mem_fraction_static"):
        engine_args.mem_fraction_static = -0.1
    assert engine_args.mem_fraction_static == 0.7

    engine_args.gpu_memory_utilization = None
    engine_args.mem_fraction_static = None

    assert engine_args.gpu_memory_utilization is None
    assert engine_args.mem_fraction_static is None


def test_json_rejects_invalid_memory_fraction_types():
    with pytest.raises(Exception, match="gpu_memory_utilization"):
        MockEngineArgs.from_json(json.dumps({"gpu_memory_utilization": "bad"}))

    with pytest.raises(Exception, match="mem_fraction_static"):
        MockEngineArgs.from_json(json.dumps({"mem_fraction_static": {}}))
