# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dynamo.common.kv_state_agent import worker_raw_topic
from dynamo.mocker import config as CONFIG
from dynamo.mocker.args import parse_args
from dynamo.mocker.config import normalize_mocker_config
from dynamo.mocker.utils import kv_cache

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.unit,
]


def test_removed_engine_classes_are_not_exported():
    import dynamo._core as core
    import dynamo.mocker as mocker

    for name in ("MockEngineArgs", "SglangArgs", "TrtllmArgs"):
        assert not hasattr(core, name)
        assert not hasattr(mocker, name)


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_canonical_cli_config_reaches_runtime_and_replay(backend):
    from dynamo._core import run_mocker_synthetic_trace_replay

    config = CONFIG.build_mocker_engine_args(
        parse_args(
            ["--engine-type", backend, "--max-model-len", "8", "--num-gpu-blocks", "64"]
        )
    )
    config["engine"]["timing_model"] = {
        "type": "fixed",
        "prefill_ms": 1.0,
        "decode_ms": 1.0,
    }
    restored = normalize_mocker_config(json.dumps(config))
    assert restored == config
    _, runtime = CONFIG.build_runtime_config(restored)
    assert runtime.context_length == 8
    assert runtime.total_kv_blocks == 64
    for isl, output, status in [(3, 5, "completed"), (8, 0, "rejected")]:
        result = run_mocker_synthetic_trace_replay(
            isl,
            10,
            1,
            extra_engine_args=restored,
            replay_concurrency=1,
            capture_per_request=True,
        )
        assert result.per_request[0]["output_length"] == output
        assert result.per_request[0]["terminal_status"] == status


def test_runtime_overrides_do_not_change_shared_engine_input(monkeypatch):
    monkeypatch.setenv("DYN_HTTP_RPC_HOST", "127.0.0.1")
    base = CONFIG.build_mocker_engine_args(
        parse_args(["--disaggregation-mode", "prefill"])
    )
    worker = CONFIG.apply_worker_engine_args_overrides(
        base, bootstrap_port=9001, kv_bytes_per_token=128, ais_mtp_seed=9
    )
    assert base["dynamo"]["bootstrap_port"] is None
    assert worker["engine"]["aic_mtp_seed"] == 9
    _, runtime = CONFIG.build_runtime_config(worker)
    assert runtime.bootstrap_port == 9001
    assert runtime.bootstrap_host == "127.0.0.1"


@pytest.mark.parametrize(
    "legacy_flags",
    [
        ["--sglang-page-size", "32", "--sglang-schedule-policy", "fcfs"],
        ["--sglang-page-size=32", "--sglang-schedule-policy=fcfs", "--block-size=32"],
    ],
)
def test_legacy_sglang_cli_aliases_warn_and_match_canonical_config(legacy_flags):
    with pytest.warns(FutureWarning) as warnings:
        legacy = CONFIG.build_mocker_engine_args(
            parse_args(["--engine-type", "sglang", *legacy_flags])
        )
    canonical = CONFIG.build_mocker_engine_args(
        parse_args(
            [
                "--engine-type",
                "sglang",
                "--block-size",
                "32",
                "--sglang-schedule-policy",
                "fifo",
            ]
        )
    )
    assert legacy == canonical
    assert len(warnings) == 2
    for warning in warnings:
        message = str(warning.message)
        assert "deprecated" in message
        assert "Dynamo 1.8.0 (two releases after 1.6.0)" in message
    assert "use --sglang-schedule-policy fifo" in str(warnings[0].message)
    assert "use --block-size" in str(warnings[1].message)


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_legacy_sglang_page_size_does_not_change_other_backends(backend):
    flags = ["--engine-type", backend, "--block-size", "64"]
    with pytest.warns(FutureWarning, match="--sglang-page-size is deprecated"):
        legacy = CONFIG.build_mocker_engine_args(
            parse_args([*flags, "--sglang-page-size", "16"])
        )
    assert legacy == CONFIG.build_mocker_engine_args(parse_args(flags))


@pytest.mark.parametrize(
    "flags",
    [
        ["--sglang-page-size", "16", "--block-size", "32"],
        ["--block-size", "32", "--sglang-page-size", "16"],
    ],
)
def test_legacy_sglang_page_size_rejects_conflicting_block_size(flags, capsys):
    with pytest.warns(FutureWarning, match="--sglang-page-size is deprecated"):
        with pytest.raises(SystemExit, match="2"):
            parse_args(["--engine-type", "sglang", *flags])
    assert "--sglang-page-size and --block-size must match" in capsys.readouterr().err


def test_profile_is_an_explicit_timing_provider(tmp_path):
    from dynamo._core import run_mocker_synthetic_trace_replay

    path = tmp_path / "profile.npz"
    np.savez(
        path,
        prefill_isl=np.array([1.0, 100.0]),
        prefill_ttft_ms=np.array([100.0, 100.0]),
        decode_active_kv_tokens=np.array([1.0, 100.0]),
        decode_context_length=np.array([1.0, 100.0]),
        decode_itl=np.array([[200.0, 200.0], [200.0, 200.0]]),
    )
    profile = {
        "type": "external",
        "provider": "dynamo_profile",
        "config": {"path": str(path)},
    }
    config = normalize_mocker_config({"engine": {"timing_model": profile}})
    result = run_mocker_synthetic_trace_replay(
        4, 2, 1, extra_engine_args=config, arrival_interval_ms=10
    )
    assert result.summary["mean_ttft_ms"] == 100
    assert result.summary["mean_tpot_ms"] == 200
    config["engine"]["timing_model"] = {
        "type": "fixed",
        "prefill_ms": 2.0,
        "decode_ms": 3.0,
    }
    result = run_mocker_synthetic_trace_replay(
        4, 2, 1, extra_engine_args=config, arrival_interval_ms=10
    )
    assert result.summary["mean_ttft_ms"] == 2
    assert result.summary["mean_tpot_ms"] == 3
    with pytest.raises(ValueError, match="planner_profile_data"):
        normalize_mocker_config({"dynamo": {"planner_profile_data": str(path)}})


@pytest.mark.parametrize(
    "bad",
    [
        {"engine_type": "sglang"},
        {"engine": {"num_gpu_blocks": "bad"}},
        {"engine": {"max_model_len": 0}},
        {"gpu_memory_utilization": 1.1},
        {"gpu_memory_utilization": float("nan")},
        {"dynamo": {"kv_state_agent": True}},
    ],
)
def test_canonical_validation_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        normalize_mocker_config(bad)


def test_sglang_generate_capability_is_opt_in():
    assert parse_args([]).sglang_generate is False
    assert parse_args(["--sglang-generate"]).sglang_generate is True


def test_get_kv_cache_dtype_bytes_supports_int8():
    # AIC KVCacheQuantMode allows int8; the byte map must size it at 1 byte
    # instead of silently falling back to 2, or KV-transfer latency is
    # overstated.
    from types import SimpleNamespace

    from dynamo.mocker.utils.kv_cache import get_kv_cache_dtype_bytes

    cfg = SimpleNamespace(dtype="bfloat16")
    assert get_kv_cache_dtype_bytes(cfg, "int8") == 1
    assert get_kv_cache_dtype_bytes(cfg, "fp8") == 1
    assert get_kv_cache_dtype_bytes(cfg, "auto") == 2


def test_compute_kv_bytes_reads_local_config_json_without_transformers(
    monkeypatch, tmp_path
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "num_attention_heads": 8,
                "hidden_size": 64,
                "torch_dtype": "bfloat16",
            }
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", None)

    assert kv_cache.compute_kv_bytes_per_token(str(tmp_path)) == 256


def test_compute_kv_bytes_unwraps_multimodal_text_config_json(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "some_vlm",
                "vision_config": {"hidden_size": 1},
                "text_config": {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 8,
                    "hidden_size": 64,
                    "dtype": "bfloat16",
                },
            }
        )
    )

    assert kv_cache.compute_kv_bytes_per_token(str(tmp_path)) == 256


def test_compute_kv_bytes_uses_transformers_text_config_for_hub_ids(monkeypatch):
    """A bare hub ID still resolves through transformers, unwrapping wrappers."""
    text_config = SimpleNamespace(
        num_hidden_layers=2,
        num_key_value_heads=4,
        num_attention_heads=8,
        hidden_size=64,
        dtype="bfloat16",
    )
    config = SimpleNamespace(get_text_config=lambda: text_config)

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            assert model_path == "org/model"
            return config

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoConfig=FakeAutoConfig)
    )

    assert kv_cache.compute_kv_bytes_per_token("org/model") == 256


def test_compute_kv_bytes_unwraps_nested_thinker_text_config(monkeypatch, tmp_path):
    # Qwen2.5-Omni keeps the serving LLM under thinker_config.text_config.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_5_omni",
                "thinker_config": {
                    "model_type": "qwen2_5_omni_thinker",
                    "audio_config": {"d_model": 1},
                    "vision_config": {"hidden_size": 1},
                    "text_config": _KV_TEXT_CONFIG,
                },
                "talker_config": {"hidden_size": 1, "num_hidden_layers": 1},
            }
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", None)

    assert kv_cache.compute_kv_bytes_per_token(str(tmp_path)) == 256


def test_compute_kv_bytes_falls_back_to_transformers_for_gpt2_style_config(
    monkeypatch, tmp_path
):
    # GPT-2 stores n_layer/n_head/n_embd; only transformers' attribute_map maps
    # them, so the raw config.json must not be trusted for this layout.
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "n_layer": 2, "n_head": 8, "n_embd": 64})
    )
    seen = []

    def from_pretrained(model_path, **kwargs):
        seen.append(model_path)
        return SimpleNamespace(
            num_hidden_layers=2, num_attention_heads=8, hidden_size=64
        )

    monkeypatch.setitem(
        sys.modules, "transformers", _fake_transformers(from_pretrained)
    )

    # No num_key_value_heads: defaults to num_attention_heads; no dtype: float16.
    assert kv_cache.compute_kv_bytes_per_token(str(tmp_path)) == 2 * 2 * 8 * 8 * 2
    assert seen == [str(tmp_path)]


def test_compute_kv_bytes_returns_none_for_unreadable_or_incomplete_config(
    monkeypatch, tmp_path
):
    def from_pretrained(model_path, **kwargs):
        return SimpleNamespace(model_type="unknown")  # no size fields

    monkeypatch.setitem(
        sys.modules, "transformers", _fake_transformers(from_pretrained)
    )

    assert kv_cache.compute_kv_bytes_per_token(str(tmp_path)) is None  # no config

    (tmp_path / "config.json").write_text("{not json")
    assert kv_cache.compute_kv_bytes_per_token(str(tmp_path)) is None

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown"}))
    assert kv_cache.compute_kv_bytes_per_token(str(tmp_path)) is None


def test_compute_kv_bytes_propagates_unexpected_errors(monkeypatch):
    def from_pretrained(model_path, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules, "transformers", _fake_transformers(from_pretrained)
    )

    with pytest.raises(RuntimeError, match="boom"):
        kv_cache.compute_kv_bytes_per_token("org/model")


def test_response_plane_defaults_to_tcp_and_accepts_quic(monkeypatch):
    monkeypatch.delenv("DYN_RESPONSE_PLANE", raising=False)

    assert parse_args([]).response_plane == "tcp"
    assert parse_args(["--response-plane", "quic"]).response_plane == "quic"
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "quic")
    assert parse_args([]).response_plane == "quic"

    with pytest.raises(SystemExit):
        parse_args(["--response-plane", "invalid"])


_KV_TEXT_CONFIG = {
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "num_attention_heads": 8,
    "hidden_size": 64,
    "torch_dtype": "bfloat16",
}


def _fake_transformers(from_pretrained):
    return SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=from_pretrained))


def test_mocker_cli_maps_native_host_offload_flags():
    args = parse_args(
        [
            "--kv-bytes-per-token",
            "1024",
            "--num-host-blocks",
            "128",
            "--host-offload-d2h-bandwidth-gbps",
            "12.5",
            "--host-offload-h2d-bandwidth-gbps",
            "0",
        ]
    )

    engine_args = CONFIG.build_mocker_engine_args(args)["engine"]

    # The CLI resolves shared transfer/cache geometry before upstream validation.
    assert engine_args["kv_transfer_bytes_per_token"] == 1024
    assert engine_args["kv_cache_bytes_per_token"] == 1024
    assert engine_args["native_host_offload"]["num_host_blocks"] == 128
    assert engine_args["native_host_offload"]["d2h_bandwidth_gbps"] == 12.5
    assert engine_args["native_host_offload"]["h2d_bandwidth_gbps"] == 0.0


@pytest.mark.asyncio
async def test_native_host_offload_resolves_model_geometry_before_validation(
    monkeypatch,
):
    # The CLI module sets this process default on import; keep it test-scoped.
    monkeypatch.setenv("DYN_COMPUTE_THREADS", "0")
    mocker_main = importlib.import_module("dynamo.mocker.main")
    args = parse_args(["--model-path", "test-model", "--num-host-blocks", "16"])
    captured = []

    async def prefetch_model(_model):
        return "/test-model"

    async def launch_workers(_args, config):
        captured.append(config)

    monkeypatch.setattr(mocker_main, "parse_args", lambda: args)
    monkeypatch.setattr(mocker_main, "prefetch_model", prefetch_model)
    monkeypatch.setattr(mocker_main, "compute_kv_bytes_per_token", lambda *_: 2048)
    monkeypatch.setattr(mocker_main, "launch_workers", launch_workers)
    await mocker_main.worker()

    assert len(captured) == 1
    engine = captured[0]["engine"]
    assert engine["kv_cache_bytes_per_token"] == 2048
    assert engine["kv_transfer_bytes_per_token"] == 2048
    assert engine["native_host_offload"]["num_host_blocks"] == 16


@pytest.mark.parametrize(
    "flags, error",
    [
        (["--num-host-blocks", "8"], "requires --zmq-kv-events-ports"),
        (["--zmq-kv-events-ports", "5557"], "requires --num-host-blocks"),
    ],
)
def test_kv_state_agent_requires_zmq_ports_and_host_blocks(flags, error):
    with pytest.raises(ValueError, match=error):
        parse_args(["--kv-state-agent", *flags])


@pytest.mark.parametrize(
    "flags",
    [
        ["--kv-state-agent-router-hints"],
        ["--kv-state-agent-raw-host", "10.0.0.5"],
        ["--kv-state-agent-cache-slot", "0"],
    ],
)
def test_kv_state_agent_flags_require_kv_state_agent(flags):
    with pytest.raises(ValueError, match="requires --kv-state-agent"):
        parse_args(flags)


_KV_STATE_AGENT_FLAGS = ["--kv-state-agent", "--zmq-kv-events-ports", "5557"]
_KV_STATE_AGENT_FLAGS += ["--num-host-blocks", "8"]


@pytest.mark.parametrize(
    "flags, error",
    [
        (["--kv-state-agent-cache-slot", "0"], "requires --kv-state-agent-raw-host"),
        (["--kv-state-agent-raw-host", "127.0.0.1"], "non-loopback"),
        (["--kv-state-agent-raw-host", "10.0.0.5"], "cache-slot"),
    ],
)
def test_kv_state_agent_needs_a_connectable_host_and_a_cache_slot(flags, error):
    # As vLLM's dynamo_state_agent raw_advertise_host and cache_owner_ids.
    with pytest.raises(ValueError, match=error):
        parse_args([*_KV_STATE_AGENT_FLAGS, *flags])


def test_kv_state_agent_descriptors_cover_each_dp_rank():
    args = parse_args(
        [
            *_KV_STATE_AGENT_FLAGS,
            "--kv-state-agent-router-hints",
            "--kv-state-agent-raw-host",
            "10.0.0.5",
            "--kv-state-agent-cache-slot",
            "40",
            "--kv-bytes-per-token",
            "1024",
            "--data-parallel-size",
            "2",
        ]
    )
    owners = CONFIG.kv_state_agent_cache_owner_ids(2, args.kv_state_agent_cache_slot)
    topic = worker_raw_topic(owners)
    config = CONFIG.apply_worker_engine_args_overrides(
        CONFIG.build_mocker_engine_args(args),
        zmq_kv_events_port=5557,
        zmq_kv_events_topic=topic,
        kv_state_agent=True,
    )
    assert config["dynamo"]["kv_state_agent"] is True
    assert config["dynamo"]["zmq_kv_events_topic"] == topic

    descriptors = CONFIG.build_kv_state_agent_descriptors(
        config,
        cache_owner_ids=owners,
        kv_state_endpoint=args.endpoint,
        raw_advertise_host=args.kv_state_agent_raw_host,
        zmq_kv_events_port=5557,
        router_hint_worker_type="aggregated",
    )
    assert [d["global_dp_rank"] for d in descriptors] == [0, 1]
    assert [d["raw_zmq_endpoint"] for d in descriptors] == [
        "tcp://10.0.0.5:5557",
        "tcp://10.0.0.5:5558",
    ]
    # Operator-assigned slots, as KVCR_CACHE_SLOT in the vLLM KVCR manifests.
    assert [d["cache_owner_id"].rsplit("/", 1)[1] for d in descriptors] == [
        f"{40:032x}",
        f"{41:032x}",
    ]
    # One topic per worker, published by every rank, as a vLLM worker's.
    assert [d["raw_topic"] for d in descriptors] == [topic, topic]
    for descriptor in descriptors:
        assert descriptor["cache_owner_id"].startswith(
            descriptor["indexer_domain_id"] + "/"
        )
        assert descriptor["router_hint_source"]["worker_type"] == "aggregated"


def test_worker_topology_config_splits_the_transfer_domain_into_islands():
    from dynamo.common.utils.topology import TopologyConfig

    topology = TopologyConfig(
        topology_domains={"rdma-island": "node-3", "zone": "us-east"},
        kv_transfer_domain="rdma-island",
        kv_transfer_enforcement="required",
    )
    for worker_id, island in ((0, "node-3-0"), (7, "node-3-0"), (8, "node-3-1")):
        config = CONFIG.worker_topology_config(topology, worker_id, 8)
        assert config.topology_domains == {"rdma-island": island, "zone": "us-east"}
        assert config.kv_transfer_enforcement == "required"
    assert CONFIG.worker_topology_config(topology, 9, None) is topology
    disabled = TopologyConfig()
    assert CONFIG.worker_topology_config(disabled, 9, 8) is disabled
