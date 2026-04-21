# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dynamo.llm import EngineType, EntrypointArgs, MockEngineArgs

MODULE_PATH = Path(__file__).resolve().parents[2] / "config.py"
SPEC = importlib.util.spec_from_file_location("dynamo_mocker_config", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.unit,
]


def make_args(**overrides):
    defaults = {
        "extra_engine_args": None,
        "engine_type": "vllm",
        "num_gpu_blocks": 16384,
        "block_size": None,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "preemption_mode": "lifo",
        "speedup_ratio": 1.0,
        "decode_speedup_ratio": 1.0,
        "dp_size": 1,
        "startup_time": None,
        "durable_kv_events": False,
        "kv_transfer_bandwidth": 64.0,
        "reasoning": None,
        "sglang_schedule_policy": None,
        "sglang_page_size": None,
        "sglang_max_prefill_tokens": None,
        "sglang_chunked_prefill_size": None,
        "sglang_clip_max_new_tokens": None,
        "sglang_schedule_conservativeness": None,
        "aic_perf_model": False,
        "aic_system": None,
        "aic_backend_version": None,
        "aic_tp_size": None,
        "aic_moe_tp_size": None,
        "aic_moe_ep_size": None,
        "aic_attention_dp_size": None,
        "model_path": None,
        "is_prefill_worker": False,
        "is_decode_worker": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_runtime_config_uses_normalized_sglang_page_size_alias():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(engine_type="sglang", block_size=None, sglang_page_size=16)
    )

    block_size, runtime_config = CONFIG.build_runtime_config(engine_args)

    assert block_size == 16
    assert runtime_config.total_kv_blocks == 16384
    assert runtime_config.max_num_seqs == 256
    assert runtime_config.max_num_batched_tokens == 8192


def test_build_mocker_engine_args_rejects_mismatched_sglang_sizes():
    with pytest.raises(Exception, match="block_size and sglang.page_size to match"):
        CONFIG.build_mocker_engine_args(
            make_args(engine_type="sglang", block_size=8, sglang_page_size=4)
        )


def test_load_mocker_engine_args_from_json_file_normalizes_page_size(tmp_path):
    config_path = tmp_path / "engine_args.json"
    config_path.write_text(
        '{"engine_type":"sglang","sglang":{"page_size":32},"num_gpu_blocks":1024}'
    )

    engine_args = CONFIG.load_mocker_engine_args(
        make_args(extra_engine_args=config_path)
    )

    assert engine_args.block_size == 32
    assert engine_args.num_gpu_blocks == 1024


def test_worker_overrides_drive_runtime_config_for_prefill_worker():
    engine_args = CONFIG.build_mocker_engine_args(make_args(is_prefill_worker=True))
    worker_args = CONFIG.apply_worker_engine_args_overrides(
        engine_args,
        bootstrap_port=9001,
        kv_bytes_per_token=128,
    )

    block_size, runtime_config = CONFIG.build_runtime_config(worker_args)

    assert block_size == 64
    assert worker_args.bootstrap_port == 9001
    assert runtime_config.bootstrap_port == 9001
    assert runtime_config.bootstrap_host is not None


def test_runtime_config_disables_local_indexer_for_decode_worker():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(is_decode_worker=True, durable_kv_events=False)
    )

    _, runtime_config = CONFIG.build_runtime_config(engine_args)

    assert engine_args.enable_local_indexer is True
    assert runtime_config.enable_local_indexer is False


def test_entrypoint_args_accept_typed_mocker_engine_args():
    engine_args = CONFIG.build_mocker_engine_args(make_args())

    entrypoint_args = EntrypointArgs(
        engine_type=EngineType.Mocker,
        mocker_engine_args=engine_args,
        kv_cache_block_size=engine_args.block_size,
    )

    assert entrypoint_args is not None


def test_build_mocker_engine_args_preserves_cli_mapped_fields(tmp_path):
    planner_profile_data = tmp_path / "planner_profile_data.npz"
    np.savez(
        planner_profile_data,
        prefill_isl=np.array([128.0, 256.0]),
        prefill_ttft_ms=np.array([4.0, 8.0]),
        decode_active_kv_tokens=np.array([1024.0, 2048.0]),
        decode_context_length=np.array([128.0, 256.0]),
        decode_itl=np.array([[1.0, 1.5], [2.0, 2.5]]),
    )

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

    engine_args = CONFIG.build_mocker_engine_args(args)
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
        "aic_moe_tp_size": None,
        "aic_moe_ep_size": None,
        "aic_attention_dp_size": None,
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
    }

    assert "has_perf_model" not in payload


def test_mock_engine_args_from_json_ignores_legacy_has_perf_model_field():
    payload = {
        "engine_type": "vllm",
        "num_gpu_blocks": 2048,
        "block_size": 128,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "worker_type": "decode",
        "has_perf_model": True,
    }

    engine_args = MockEngineArgs.from_json(json.dumps(payload))

    assert engine_args.num_gpu_blocks == 2048
    assert engine_args.block_size == 128
    assert engine_args.max_num_seqs is None
    assert engine_args.max_num_batched_tokens is None
    assert engine_args.worker_type == "decode"
