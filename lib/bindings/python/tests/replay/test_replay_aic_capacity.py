# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

import dynamo._internal.aic as aic
import dynamo.replay.main as replay_main
from dynamo.mocker import MockEngineArgs, PlannerReplayBridge
from dynamo.replay import run_synthetic_trace_replay

from .replay_utils import _write_trace_and_args

# Tests in this file drive the Rust AIC callback, which imports
# aiconfigurator.sdk.engine (Phase 1.5 compile_engine API). Skip if absent —
# PyPI aiconfigurator releases predating PR #1200 don't ship it.
pytest.importorskip("aiconfigurator.sdk.engine")

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
                "aic_systems_path": "/tmp/aic-systems",
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
            "systems_path": "/tmp/aic-systems",
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


def test_programmatic_replay_estimates_unset_aic_blocks(monkeypatch):
    calls = []

    class FakeAicSession:
        def predict_prefill(self, batch_size, effective_isl, prefix):
            return float(batch_size + effective_isl + prefix)

        def predict_decode(self, batch_size, isl, osl):
            return float(batch_size + isl + osl)

    def fake_estimate_num_gpu_blocks(**kwargs):
        calls.append(kwargs)
        return 100

    def fake_create_session(*_args):
        return FakeAicSession()

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", fake_estimate_num_gpu_blocks)
    monkeypatch.setattr(aic, "create_session", fake_create_session)

    engine_args = MockEngineArgs(
        aic_backend="vllm",
        aic_system="h200_sxm",
        aic_model_path="/models/mock",
        aic_tp_size=2,
        block_size=2,
        max_num_batched_tokens=16,
        max_num_seqs=2,
    )

    report = run_synthetic_trace_replay(
        4,
        2,
        1,
        extra_engine_args=engine_args,
        replay_concurrency=1,
        replay_mode="offline",
        arrival_interval_ms=0.0,
    )

    assert report["num_requests"] == 1
    assert calls == [
        {
            "backend_name": "vllm",
            "system": "h200_sxm",
            "model_path": "/models/mock",
            "tp_size": 2,
            "block_size": 2,
            "max_num_batched_tokens": 16,
            "gpu_memory_utilization": 0.9,
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


def test_planner_bridge_materializes_unset_aic_blocks(tmp_path, monkeypatch):
    calls = []

    class FakeAicSession:
        def predict_prefill(self, batch_size, effective_isl, prefix):
            return float(batch_size + effective_isl + prefix)

        def predict_decode(self, batch_size, isl, osl):
            return float(batch_size + isl + osl)

    def fake_estimate_num_gpu_blocks(**kwargs):
        calls.append(kwargs)
        return 100

    def fake_create_session(*_args):
        return FakeAicSession()

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", fake_estimate_num_gpu_blocks)
    monkeypatch.setattr(aic, "create_session", fake_create_session)

    trace_path = _write_trace_and_args(tmp_path)
    agg_args = MockEngineArgs(
        aic_backend="vllm",
        aic_system="h200_sxm",
        aic_model_path="/models/agg",
        aic_tp_size=2,
        block_size=2,
        max_num_batched_tokens=16,
        max_num_seqs=2,
    )

    PlannerReplayBridge(
        trace_file=trace_path,
        extra_engine_args=agg_args,
        num_workers=1,
    )

    prefill_args = MockEngineArgs(
        aic_backend="vllm",
        aic_system="h200_sxm",
        aic_model_path="/models/prefill",
        aic_tp_size=2,
        block_size=2,
        max_num_batched_tokens=16,
        max_num_seqs=2,
        worker_type="prefill",
    )
    decode_args = MockEngineArgs(
        aic_backend="vllm",
        aic_system="h200_sxm",
        aic_model_path="/models/decode",
        aic_tp_size=2,
        block_size=2,
        max_num_batched_tokens=16,
        max_num_seqs=2,
        worker_type="decode",
    )

    PlannerReplayBridge.create_disagg(
        trace_file=trace_path,
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        num_prefill_workers=1,
        num_decode_workers=1,
    )

    def expected(model_path):
        return {
            "backend_name": "vllm",
            "system": "h200_sxm",
            "model_path": model_path,
            "tp_size": 2,
            "block_size": 2,
            "max_num_batched_tokens": 16,
            "gpu_memory_utilization": 0.9,
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

    assert calls == [
        expected("/models/agg"),
        expected("/models/prefill"),
        expected("/models/decode"),
    ]


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
