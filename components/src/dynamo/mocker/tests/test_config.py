#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from dynamo.mocker import config


def test_build_mocker_engine_args_preserves_cli_mapped_fields(monkeypatch):
    args = argparse.Namespace(
        engine_type="sglang",
        num_gpu_blocks=2048,
        block_size=128,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        preemption_mode="fifo",
        speedup_ratio=2.0,
        decode_speedup_ratio=3.0,
        dp_size=4,
        startup_time=1.5,
        planner_profile_data=Path("/tmp/perf_data.npz"),
        is_prefill_worker=True,
        is_decode_worker=False,
        durable_kv_events=False,
        kv_transfer_bandwidth=123.0,
        reasoning=json.dumps(
            {
                "start_thinking_token_id": 11,
                "end_thinking_token_id": 12,
                "thinking_ratio": 0.25,
            }
        ),
        sglang_schedule_policy="lpm",
        sglang_page_size=16,
        sglang_max_prefill_tokens=8192,
        sglang_chunked_prefill_size=2048,
        sglang_clip_max_new_tokens=1024,
        sglang_schedule_conservativeness=0.8,
        aic_perf_model=True,
        aic_system="h200_sxm",
        aic_backend_version="0.5.6.post2",
        aic_tp_size=8,
        model_path="/models/mock",
    )

    captured = {}

    def fake_mock_engine_args(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(config, "MockEngineArgs", fake_mock_engine_args)

    engine_args = config.build_mocker_engine_args(args)

    assert captured["kwargs"] == {
        "engine_type": "sglang",
        "num_gpu_blocks": 2048,
        "block_size": 128,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 4096,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "preemption_mode": "fifo",
        "speedup_ratio": 2.0,
        "decode_speedup_ratio": 3.0,
        "dp_size": 4,
        "startup_time": 1.5,
        "worker_type": "prefill",
        "planner_profile_data": Path("/tmp/perf_data.npz"),
        "enable_local_indexer": True,
        "kv_transfer_bandwidth": 123.0,
        "reasoning": captured["kwargs"]["reasoning"],
        "sglang": captured["kwargs"]["sglang"],
        "aic_backend": "sglang",
        "aic_system": "h200_sxm",
        "aic_backend_version": "0.5.6.post2",
        "aic_tp_size": 8,
        "aic_model_path": "/models/mock",
    }
    assert engine_args.planner_profile_data == Path("/tmp/perf_data.npz")
    assert engine_args.reasoning is not None
    assert engine_args.sglang is not None
