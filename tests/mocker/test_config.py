# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamo.llm import EngineType, EntrypointArgs

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "components/src/dynamo/mocker/config.py"
)
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
