#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

from dynamo.mocker import config


def _planner_profile_data_npz_path() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "benchmarks/results/H200_TP1P_TP1D_perf_data.npz"
    )


def test_build_mocker_engine_args_preserves_cli_mapped_fields():
    planner_profile_data = _planner_profile_data_npz_path()
    assert planner_profile_data.exists()

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
        planner_profile_data=planner_profile_data,
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
        sglang_page_size=128,
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

    engine_args = config.build_mocker_engine_args(args)
    payload = json.loads(engine_args.dump_json())

    assert payload == {
        "engine_type": "sglang",
        "num_gpu_blocks": 2048,
        "block_size": 128,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 4096,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "speedup_ratio": 2.0,
        "decode_speedup_ratio": 3.0,
        "dp_size": 4,
        "startup_time": 1.5,
        "worker_type": "prefill",
        "planner_profile_data": str(planner_profile_data),
        "aic_backend": "sglang",
        "aic_system": "h200_sxm",
        "aic_backend_version": "0.5.6.post2",
        "aic_tp_size": 8,
        "aic_model_path": "/models/mock",
        "enable_local_indexer": True,
        "bootstrap_port": None,
        "kv_bytes_per_token": None,
        "kv_transfer_bandwidth": 123.0,
        "reasoning": {
            "start_thinking_token_id": 11,
            "end_thinking_token_id": 12,
            "thinking_ratio": 0.25,
        },
        "zmq_kv_events_port": None,
        "zmq_replay_port": None,
        "preemption_mode": "fifo",
        "router_queue_policy": None,
        "sglang": {
            "schedule_policy": "lpm",
            "page_size": 128,
            "max_prefill_tokens": 8192,
            "chunked_prefill_size": 2048,
            "clip_max_new_tokens": 1024,
            "schedule_conservativeness": 0.8,
        },
        "has_perf_model": True,
    }
