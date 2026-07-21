# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dynamo.llm import EngineType, EntrypointArgs
from dynamo.mocker import MockEngineArgs
from dynamo.mocker.args import parse_args

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
        "num_gpu_blocks": None,
        "block_size": None,
        "max_model_len": None,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "enable_prefix_caching": True,
        "g1_backend": "kvbm",
        "enable_chunked_prefill": True,
        "preemption_mode": "lifo",
        "speedup_ratio": 1.0,
        "decode_speedup_ratio": 1.0,
        "dp_size": 1,
        "startup_time": None,
        "kv_transfer_bandwidth": 64.0,
        "kv_transfer_timing_mode": "full_prompt",
        "reasoning": None,
        "response_replay_trace_path": None,
        "sglang_schedule_policy": None,
        "sglang_page_size": None,
        "sglang_max_prefill_tokens": None,
        "sglang_chunked_prefill_size": None,
        "sglang_clip_max_new_tokens": None,
        "sglang_schedule_conservativeness": None,
        "trtllm_capacity_scheduler_policy": None,
        "aic_perf_model": False,
        "aic_system": None,
        "aic_backend": None,
        "aic_backend_version": None,
        "aic_tp_size": None,
        "aic_moe_tp_size": None,
        "aic_moe_ep_size": None,
        "aic_attention_dp_size": None,
        "aic_nextn": None,
        "aic_nextn_accept_rates": None,
        "aic_mtp_seed": 42,
        "gpu_memory_utilization": None,
        "mem_fraction_static": None,
        "free_gpu_memory_fraction": None,
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
    assert runtime_config.context_length == 0
    assert runtime_config.total_kv_blocks == 16384
    assert runtime_config.max_num_seqs == 256
    assert runtime_config.max_num_batched_tokens == 8192
    assert runtime_config.runtime_data["output_replay_consumer"] == "true"


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


def test_build_mocker_engine_args_trtllm_defaults_block_size():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(engine_type="trtllm", block_size=None)
    )

    # TRT-LLM PyTorch backend default tokens_per_block.
    assert engine_args.block_size == 32


def test_build_mocker_engine_args_trtllm_accepts_guaranteed_no_evict():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(
            engine_type="trtllm",
            trtllm_capacity_scheduler_policy="guaranteed_no_evict",
        )
    )

    assert engine_args.block_size == 32


def test_build_mocker_engine_args_trtllm_accepts_native_g1():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(engine_type="trtllm", g1_backend="native")
    )

    assert engine_args.g1_backend == "native"
    assert engine_args.block_size == 32


@pytest.mark.parametrize("engine_type", ["vllm", "trtllm"])
def test_build_mocker_engine_args_accepts_native_g1_with_mtp(engine_type):
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(
            engine_type=engine_type,
            g1_backend="native",
            aic_nextn=1,
        )
    )

    assert engine_args.g1_backend == "native"
    assert engine_args.aic_nextn == 1


def test_build_mocker_engine_args_trtllm_rejects_unsupported_policy():
    with pytest.raises(Exception, match="guaranteed_no_evict"):
        CONFIG.build_mocker_engine_args(
            make_args(
                engine_type="trtllm",
                trtllm_capacity_scheduler_policy="max_utilization",
            )
        )


def test_load_mocker_engine_args_from_json_file_accepts_trtllm(tmp_path):
    config_path = tmp_path / "engine_args.json"
    config_path.write_text(
        '{"engine_type":"trtllm",'
        '"trtllm":{"capacity_scheduler_policy":"guaranteed_no_evict"},'
        '"num_gpu_blocks":1024}'
    )

    engine_args = CONFIG.load_mocker_engine_args(
        make_args(extra_engine_args=config_path)
    )

    assert engine_args.num_gpu_blocks == 1024
    # block_size omitted from JSON -> normalized to the TRT-LLM default.
    assert engine_args.block_size == 32


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


def test_g3_args_allow_kv_bytes_per_token_worker_override():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(
            model_path="/models/mock",
            kv_bytes_per_token=None,
            num_g2_blocks=8192,
            num_g3_blocks=16384,
        )
    )
    assert engine_args.kv_bytes_per_token is None
    assert engine_args.num_g2_blocks == 8192
    assert engine_args.num_g3_blocks == 16384

    worker_args = CONFIG.apply_worker_engine_args_overrides(
        engine_args,
        kv_bytes_per_token=131072,
    )
    assert worker_args.kv_bytes_per_token == 131072
    assert worker_args.num_g3_blocks == 16384


def test_g4_args_allow_kv_bytes_per_token_worker_override():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(
            model_path="/models/mock",
            kv_bytes_per_token=None,
            num_g2_blocks=8192,
            enable_g4_storage=True,
            bandwidth_g2_to_g4_gbps=4.0,
            bandwidth_g4_to_g2_gbps=4.0,
        )
    )
    assert engine_args.kv_bytes_per_token is None
    assert engine_args.num_g2_blocks == 8192
    assert engine_args.enable_g4_storage is True
    assert engine_args.bandwidth_g2_to_g4_gbps == 4.0
    assert engine_args.bandwidth_g4_to_g2_gbps == 4.0

    worker_args = CONFIG.apply_worker_engine_args_overrides(
        engine_args,
        kv_bytes_per_token=131072,
    )
    assert worker_args.kv_bytes_per_token == 131072
    assert worker_args.enable_g4_storage is True


def test_runtime_config_disables_local_indexer_for_decode_worker():
    engine_args = CONFIG.build_mocker_engine_args(make_args(is_decode_worker=True))

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
        kv_bytes_per_token=131072,
        kv_transfer_bandwidth=123.0,
        kv_transfer_timing_mode="destination_missing",
        response_replay_trace_path=None,
        num_g2_blocks=8192,
        num_g3_blocks=16384,
        offload_batch_size=32,
        bandwidth_g1_to_g2_gbps=14.0,
        bandwidth_g2_to_g1_gbps=14.0,
        bandwidth_g2_to_g3_gbps=7.0,
        bandwidth_g3_to_g2_gbps=7.0,
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
        gpu_memory_utilization=None,
        mem_fraction_static=None,
    )

    engine_args = CONFIG.build_mocker_engine_args(args)
    assert engine_args.num_gpu_blocks == 2048
    assert engine_args.block_size == 128
    assert engine_args.max_num_seqs == 64
    assert engine_args.max_num_batched_tokens == 4096
    assert engine_args.enable_prefix_caching is False
    assert engine_args.enable_local_indexer is True
    assert engine_args.dp_size == 4
    assert engine_args.worker_type == "prefill"
    assert engine_args.gpu_memory_utilization is None
    assert engine_args.mem_fraction_static is None
    assert engine_args.aic_backend == "sglang"
    assert engine_args.aic_system == "h200_sxm"
    assert engine_args.aic_backend_version == "0.5.6.post2"
    assert engine_args.aic_tp_size == 8
    assert engine_args.aic_model_path == "/models/mock"
    assert engine_args.aic_moe_tp_size is None
    assert engine_args.aic_moe_ep_size is None
    assert engine_args.aic_attention_dp_size is None
    assert engine_args.bootstrap_port is None
    assert engine_args.kv_transfer_timing_mode == "destination_missing"
    assert engine_args.num_g2_blocks == 8192
    assert engine_args.num_g3_blocks == 16384
    assert engine_args.offload_batch_size == 32
    assert engine_args.bandwidth_g1_to_g2_gbps == 14.0
    assert engine_args.bandwidth_g2_to_g1_gbps == 14.0
    assert engine_args.bandwidth_g2_to_g3_gbps == 7.0
    assert engine_args.bandwidth_g3_to_g2_gbps == 7.0


def test_aic_backend_override_decouples_from_engine_type():
    args = make_args(
        engine_type="vllm",
        aic_perf_model=True,
        aic_system="h200_sxm",
        aic_backend="trtllm",
        aic_tp_size=4,
        num_gpu_blocks=16384,
    )

    engine_args = CONFIG.build_mocker_engine_args(args)

    assert engine_args.aic_backend == "trtllm"


def test_build_mocker_engine_args_propagates_mtp_configuration():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(
            aic_perf_model=True,
            aic_system="h200_sxm",
            model_path="/models/mock",
            num_gpu_blocks=128,
            aic_nextn=3,
            aic_nextn_accept_rates="1,0.5",
            aic_mtp_seed=99,
        )
    )

    assert engine_args.aic_nextn == 3
    assert engine_args.aic_nextn_accept_rates == "1,0.5,0"
    assert engine_args.aic_mtp_seed == 99


def test_worker_override_offsets_mtp_seed():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(aic_nextn=1, aic_mtp_seed=2**64 - 1)
    )

    worker_args = CONFIG.apply_worker_engine_args_overrides(engine_args, aic_mtp_seed=0)

    assert worker_args.aic_mtp_seed == 0


def test_mocker_cli_accepts_mtp_configuration():
    args = parse_args(
        [
            "--aic-nextn",
            "3",
            "--aic-nextn-accept-rates",
            "1,0.5",
            "--aic-mtp-seed",
            "99",
        ]
    )

    assert args.aic_nextn == 3
    assert args.aic_nextn_accept_rates == "1,0.5"
    assert args.aic_mtp_seed == 99


def test_mocker_cli_accepts_native_g1_for_trtllm():
    args = parse_args(["--engine-type", "trtllm", "--g1-backend", "native"])

    assert args.engine_type == "trtllm"
    assert args.g1_backend == "native"


def test_mocker_cli_accepts_max_model_len():
    args = parse_args(["--max-model-len", "32768"])

    engine_args = CONFIG.build_mocker_engine_args(args)
    _, runtime_config = CONFIG.build_runtime_config(engine_args)

    assert engine_args.max_model_len == 32768
    assert runtime_config.context_length == 32768


@pytest.mark.parametrize("value", ["0", "-1"])
def test_mocker_cli_rejects_non_positive_max_model_len(value):
    with pytest.raises(SystemExit):
        parse_args(["--max-model-len", value])


def test_build_mocker_engine_args_keeps_max_model_len_explicit_only():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(model_path="/models/mock", num_gpu_blocks=4096)
    )

    assert engine_args.max_model_len is None


def test_build_mocker_engine_args_preserves_explicit_max_model_len():
    engine_args = CONFIG.build_mocker_engine_args(
        make_args(
            model_path="/models/mock",
            max_model_len=32768,
            num_gpu_blocks=4096,
        )
    )

    assert engine_args.max_model_len == 32768


def test_replay_engine_args_keeps_max_model_len_explicit_only():
    import dynamo.replay.main as replay_main

    engine_args = replay_main._load_engine_args(
        json.dumps(
            {
                "num_gpu_blocks": 4096,
                "aic_model_path": "/models/mock",
            }
        )
    )

    assert engine_args.max_model_len is None


def test_replay_engine_args_preserves_explicit_max_model_len():
    import dynamo.replay.main as replay_main

    engine_args = replay_main._load_engine_args(
        json.dumps(
            {
                "num_gpu_blocks": 4096,
                "max_model_len": 32768,
                "aic_model_path": "/models/mock",
            }
        )
    )

    assert engine_args.max_model_len == 32768


def test_replay_attention_dp_sets_rank_topology_with_explicit_kv_capacity():
    import dynamo.replay.main as replay_main

    engine_args = replay_main._load_engine_args(
        json.dumps(
            {
                "num_gpu_blocks": 4096,
                "aic_attention_dp_size": 4,
            }
        )
    )

    assert engine_args.num_gpu_blocks == 4096
    assert engine_args.dp_size == 4


def test_replay_rejects_mismatched_dp_topology():
    import dynamo.replay.main as replay_main

    with pytest.raises(ValueError, match="dp_size must match"):
        replay_main._load_engine_args(
            json.dumps(
                {
                    "num_gpu_blocks": 4096,
                    "dp_size": 2,
                    "aic_attention_dp_size": 4,
                }
            )
        )


def test_replay_rejects_dp_topology_without_aic_attention_dp():
    import dynamo.replay.main as replay_main

    with pytest.raises(ValueError, match="dp_size must match"):
        replay_main._load_engine_args(
            json.dumps(
                {
                    "num_gpu_blocks": 4096,
                    "dp_size": 2,
                    "aic_backend": "vllm",
                }
            )
        )


def test_replay_engine_args_compute_kv_bytes_for_g3_before_validation(monkeypatch):
    import dynamo.replay.main as replay_main

    calls = []

    def fake_compute_kv_bytes_per_token(model_path, kv_cache_dtype="auto"):
        calls.append((model_path, kv_cache_dtype))
        return 131072

    monkeypatch.setattr(
        replay_main, "compute_kv_bytes_per_token", fake_compute_kv_bytes_per_token
    )

    engine_args = replay_main._load_engine_args(
        json.dumps(
            {
                "num_gpu_blocks": 4096,
                "num_g2_blocks": 8192,
                "num_g3_blocks": 16384,
                "aic_model_path": "/models/mock",
            }
        )
    )

    assert engine_args.num_g2_blocks == 8192
    assert engine_args.num_g3_blocks == 16384
    assert calls == [("/models/mock", "auto")]


def test_replay_engine_args_compute_kv_bytes_for_g4_before_validation(monkeypatch):
    import dynamo.replay.main as replay_main

    calls = []

    def fake_compute_kv_bytes_per_token(model_path, kv_cache_dtype="auto"):
        calls.append((model_path, kv_cache_dtype))
        return 131072

    monkeypatch.setattr(
        replay_main, "compute_kv_bytes_per_token", fake_compute_kv_bytes_per_token
    )

    engine_args = replay_main._load_engine_args(
        json.dumps(
            {
                "num_gpu_blocks": 4096,
                "num_g2_blocks": 8192,
                "enable_g4_storage": True,
                "aic_model_path": "/models/mock",
            }
        )
    )

    assert engine_args.num_g2_blocks == 8192
    assert engine_args.enable_g4_storage is True
    assert calls == [("/models/mock", "auto")]


def test_get_kv_cache_dtype_bytes_supports_int8():
    # AIC KVCacheQuantMode allows int8; the byte map must size it at 1 byte
    # instead of silently falling back to 2, or offload KV-byte estimates and
    # transfer latency are overstated.
    from types import SimpleNamespace

    from dynamo.mocker.utils.kv_cache import get_kv_cache_dtype_bytes

    cfg = SimpleNamespace(dtype="bfloat16")
    assert get_kv_cache_dtype_bytes(cfg, "int8") == 1
    assert get_kv_cache_dtype_bytes(cfg, "fp8") == 1
    assert get_kv_cache_dtype_bytes(cfg, "auto") == 2  # model default dtype


def test_replay_engine_args_forwards_aic_kv_cache_dtype(monkeypatch):
    # Offload KV-byte estimation must use the configured (normalized) KV dtype,
    # not always "auto".
    import dynamo.replay.main as replay_main

    calls = []

    def fake_compute_kv_bytes_per_token(model_path, kv_cache_dtype="auto"):
        calls.append((model_path, kv_cache_dtype))
        return 131072

    monkeypatch.setattr(
        replay_main, "compute_kv_bytes_per_token", fake_compute_kv_bytes_per_token
    )

    replay_main._load_engine_args(
        json.dumps(
            {
                "num_gpu_blocks": 4096,
                "num_g2_blocks": 8192,
                "num_g3_blocks": 16384,
                "aic_model_path": "/models/mock",
                "aic_kv_cache_dtype": "fp8",
            }
        )
    )

    assert calls == [("/models/mock", "fp8")]


def test_build_mocker_engine_args_estimates_aic_blocks(monkeypatch):
    calls = []

    def fake_estimate_num_gpu_blocks(**kwargs):
        calls.append(kwargs)
        return 46000

    monkeypatch.setattr(CONFIG, "estimate_num_gpu_blocks", fake_estimate_num_gpu_blocks)

    engine_args = CONFIG.build_mocker_engine_args(
        make_args(
            aic_perf_model=True,
            model_path="/models/mock",
            aic_system="h200_sxm",
            aic_tp_size=4,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.8,
            mem_fraction_static=0.7,
        )
    )

    assert engine_args.num_gpu_blocks == 46000
    assert engine_args.gpu_memory_utilization == 0.8
    assert engine_args.mem_fraction_static == 0.7
    assert calls == [
        {
            "backend_name": "vllm",
            "system": "h200_sxm",
            "model_path": "/models/mock",
            "tp_size": 4,
            "block_size": 64,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.8,
            "mem_fraction_static": 0.7,
            "free_gpu_memory_fraction": None,
            "backend_version": None,
            "moe_tp_size": None,
            "moe_ep_size": None,
            "attention_dp_size": None,
        }
    ]


def test_build_mocker_engine_args_falls_back_when_aic_estimator_missing(
    monkeypatch, caplog
):
    def missing_memory(module_name):
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr("dynamo._internal.aic.importlib.import_module", missing_memory)

    engine_args = CONFIG.build_mocker_engine_args(
        make_args(aic_perf_model=True, model_path="/models/mock")
    )

    assert engine_args.num_gpu_blocks == 16384
    assert "Falling back to default num_gpu_blocks=16384" in caplog.text
    assert "--num-gpu-blocks-override" in caplog.text


def test_aic_capacity_estimation_preserves_explicit_zero_inputs(monkeypatch):
    calls = []

    def fake_estimate_num_gpu_blocks(**kwargs):
        calls.append(kwargs)
        return 46000

    monkeypatch.setattr(CONFIG, "estimate_num_gpu_blocks", fake_estimate_num_gpu_blocks)

    blocks = CONFIG._estimate_aic_num_gpu_blocks(
        engine_type="sglang",
        block_size=0,
        max_num_batched_tokens=0,
        aic_backend="sglang",
        aic_system=None,
        aic_backend_version=None,
        aic_tp_size=0,
        aic_model_path="/models/mock",
        aic_moe_tp_size=None,
        aic_moe_ep_size=None,
        aic_attention_dp_size=None,
        gpu_memory_utilization=0.0,
        mem_fraction_static=0.0,
        free_gpu_memory_fraction=0.0,
        sglang_page_size=0,
    )

    assert blocks == 46000
    assert calls[0]["tp_size"] == 0
    assert calls[0]["block_size"] == 0
    assert calls[0]["max_num_batched_tokens"] == 0
    assert calls[0]["gpu_memory_utilization"] == 0.0
    assert calls[0]["mem_fraction_static"] == 0.0
    assert calls[0]["free_gpu_memory_fraction"] == 0.0


def test_build_mocker_engine_args_estimates_sglang_blocks_with_static_fraction(
    monkeypatch,
):
    calls = []

    def fake_estimate_num_gpu_blocks(**kwargs):
        calls.append(kwargs)
        return 32000

    monkeypatch.setattr(CONFIG, "estimate_num_gpu_blocks", fake_estimate_num_gpu_blocks)

    engine_args = CONFIG.build_mocker_engine_args(
        make_args(
            engine_type="sglang",
            aic_perf_model=True,
            model_path="/models/mock",
            sglang_page_size=128,
            mem_fraction_static=0.77,
        )
    )

    assert engine_args.num_gpu_blocks == 32000
    assert calls[0]["backend_name"] == "sglang"
    assert calls[0]["block_size"] == 128
    assert calls[0]["mem_fraction_static"] == 0.77


def test_build_mocker_engine_args_explicit_blocks_skip_aic_estimate(monkeypatch):
    def fail_estimate_num_gpu_blocks(**_kwargs):
        raise AssertionError("estimator should not be called")

    monkeypatch.setattr(CONFIG, "estimate_num_gpu_blocks", fail_estimate_num_gpu_blocks)

    engine_args = CONFIG.build_mocker_engine_args(
        make_args(
            aic_perf_model=True,
            num_gpu_blocks=12345,
            model_path="/models/mock",
        )
    )

    assert engine_args.num_gpu_blocks == 12345


def test_load_mocker_engine_args_estimates_json_aic_blocks(tmp_path, monkeypatch):
    def fake_estimate_num_gpu_blocks(**kwargs):
        assert kwargs["model_path"] == "/models/from-cli"
        assert kwargs["block_size"] == 64
        return 47000

    monkeypatch.setattr(CONFIG, "estimate_num_gpu_blocks", fake_estimate_num_gpu_blocks)

    config_path = tmp_path / "engine_args.json"
    config_path.write_text(json.dumps({"aic_backend": "vllm"}))

    engine_args = CONFIG.load_mocker_engine_args(
        make_args(extra_engine_args=config_path, model_path="/models/from-cli")
    )

    assert engine_args.num_gpu_blocks == 47000


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
