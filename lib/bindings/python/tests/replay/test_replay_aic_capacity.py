# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

import dynamo._internal.aic as aic
import dynamo.replay.main as replay_main
from dynamo.llm import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay

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
            "backend_version": None,
            "moe_tp_size": None,
            "moe_ep_size": None,
            "attention_dp_size": None,
        }
    ]


def test_programmatic_replay_estimates_unset_aic_blocks(monkeypatch):
    calls = []

    class FakeAicSession:
        def predict_prefill(self, batch_size, effective_isl, prefix):
            return float(batch_size + effective_isl + prefix)

        def predict_decode(self, batch_size, isl, osl):
            return float(batch_size + isl + osl)

    def fake_estimate_num_gpu_blocks(*args):
        calls.append(args)
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
        (
            "vllm",
            "h200_sxm",
            "/models/mock",
            2,
            2,
            16,
            0.9,
            0.88,
            None,
            None,
            None,
            None,
        )
    ]


def test_programmatic_aic_dump_preserves_unset_blocks():
    engine_args = MockEngineArgs(
        aic_backend="vllm",
        aic_system="h200_sxm",
        aic_model_path="/models/mock",
        gpu_memory_utilization=0.8,
    )

    payload = json.loads(engine_args.dump_json())

    assert payload["num_gpu_blocks"] is None
    assert payload["gpu_memory_utilization"] == 0.8
    assert payload["mem_fraction_static"] is None

    restored = MockEngineArgs.from_json(engine_args.dump_json())
    restored_payload = json.loads(restored.dump_json())

    assert restored.num_gpu_blocks == 16384
    assert restored_payload["num_gpu_blocks"] is None
