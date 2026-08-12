# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model/hardware resolution + KV-cache parallel-config validity via AIConfigurator.

Uses models whose configs resolve without HF auth (DeepSeek-V3, Qwen3-32B)."""

import pytest

import aisimulate.sweeper.model_hw as mh_mod
from aisimulate.sweeper.model_hw import (
    NoViableParallelConfig,
    parallel_configs_for,
    resolve_model_hardware,
)

DEEPSEEK = "deepseek-ai/DeepSeek-V3"
QWEN = "Qwen/Qwen3-32B"


@pytest.mark.model(DEEPSEEK)
def test_resolve_deepseek_is_moe_mla_wideep():
    mh = resolve_model_hardware(DEEPSEEK, "h200_sxm", backend="trtllm")
    assert mh.is_moe and mh.mla and mh.enable_wideep
    assert mh.weight_bytes > 0
    assert mh.max_context == 163840  # DeepSeek-V3 max context


@pytest.mark.model(QWEN)
def test_resolve_dense_qwen():
    mh = resolve_model_hardware(QWEN, "h200_sxm", backend="trtllm")
    assert not mh.is_moe
    assert not mh.mla
    assert not mh.enable_wideep  # dense models never enable wideEP
    assert mh.max_context == 40960  # Qwen3-32B max context


@pytest.mark.model(QWEN)
def test_unknown_hardware_sku_raises():
    # A typo/unknown SKU must fail loudly rather than silently using default VRAM/GPUs.
    with pytest.raises(ValueError, match="unknown hardware_sku"):
        resolve_model_hardware(QWEN, "h200_typo", backend="trtllm")


def test_aic_core_system_spec_contract(monkeypatch):
    monkeypatch.setattr(
        mh_mod,
        "get_model_config_from_model_path",
        lambda model: {
            "architecture": "Qwen3ForCausalLM",
            "context": 40960,
        },
    )
    monkeypatch.setattr(mh_mod, "check_is_moe", lambda model_config: False)
    monkeypatch.setattr(
        mh_mod.perf_database,
        "load_system_spec",
        lambda hardware: {
            "gpu": {"mem_capacity": 80},
            "node": {"num_gpus_per_node": 8},
        },
    )
    monkeypatch.setattr(
        mh_mod,
        "_estimate_model_weight_bytes",
        lambda model: 123,
    )

    mh = resolve_model_hardware("model", "hardware", backend="trtllm")

    assert mh.vram_per_gpu == 80
    assert mh.gpus_per_node == 8
    assert mh.weight_bytes == 123


@pytest.mark.model(DEEPSEEK)
def test_max_seq_len_defaults_to_model_context(monkeypatch):
    # Omitting max_seq_len uses the model's max context length.
    seen = {}

    def fake_feasible(shapes, *, max_seq_len, **kwargs):
        seen["max_seq_len"] = max_seq_len
        return dict.fromkeys(shapes, 10_000_000)

    monkeypatch.setattr(mh_mod, "feasible_shape_tokens", fake_feasible)
    parallel_configs_for(
        DEEPSEEK, "gb200", gpu_budget=16, deployment_mode="agg", backend="trtllm"
    )
    assert seen["max_seq_len"] == 163840  # DeepSeek-V3 max context


# --- KV-cache validity (the sole feasibility filter; no weight floor) ---


@pytest.mark.model(DEEPSEEK)
def test_kv_filter_keeps_only_feasible_shapes(monkeypatch):
    # Pretend only workers with >= 4 GPUs hold a sequence (KV estimate stubbed).
    def fake_feasible(shapes, **kwargs):
        return {s: 100_000 for s in dict.fromkeys(shapes) if s.gpus_per_worker >= 4}

    monkeypatch.setattr(mh_mod, "feasible_shape_tokens", fake_feasible)
    cfgs = parallel_configs_for(
        DEEPSEEK,
        "gb200",
        gpu_budget=16,
        deployment_mode="agg",
        backend="trtllm",
        max_seq_len=8192,
    )
    assert cfgs
    assert all(
        c.shape.gpus_per_worker >= 4 for c in cfgs
    )  # KV decides; no weight floor
    assert all(c.total_gpus <= 16 for c in cfgs)


@pytest.mark.model(DEEPSEEK)
def test_kv_filter_disagg_requires_both_roles_feasible(monkeypatch):
    def fake_feasible(shapes, **kwargs):
        return {s: 100_000 for s in dict.fromkeys(shapes) if s.gpus_per_worker >= 4}

    monkeypatch.setattr(mh_mod, "feasible_shape_tokens", fake_feasible)
    cfgs = parallel_configs_for(
        DEEPSEEK,
        "gb200",
        gpu_budget=16,
        deployment_mode="disagg",
        backend="trtllm",
        max_seq_len=8192,
    )
    assert cfgs
    for c in cfgs:
        assert c.prefill.shape.gpus_per_worker >= 4
        assert c.decode.shape.gpus_per_worker >= 4


@pytest.mark.model(DEEPSEEK)
def test_kv_filter_no_feasible_shape_raises(monkeypatch):
    monkeypatch.setattr(mh_mod, "feasible_shape_tokens", lambda shapes, **kwargs: {})
    with pytest.raises(NoViableParallelConfig, match="KV-cache estimate"):
        parallel_configs_for(
            DEEPSEEK,
            "gb200",
            gpu_budget=16,
            deployment_mode="agg",
            backend="trtllm",
            max_seq_len=8192,
        )


@pytest.mark.model(DEEPSEEK)
def test_kv_path_end_to_end_deepseek_gb200():
    cfgs = parallel_configs_for(
        DEEPSEEK,
        "gb200",
        gpu_budget=16,
        deployment_mode="agg",
        backend="trtllm",
        max_seq_len=8192,
    )
    assert cfgs
    # DeepSeek-V3 OOMs at 2 GPUs/worker; smallest feasible worker is >= 4 GPUs.
    assert all(c.shape.gpus_per_worker >= 4 for c in cfgs)
    assert all(c.total_gpus <= 16 for c in cfgs)


@pytest.mark.model(DEEPSEEK)
def test_kv_path_tiny_budget_raises():
    # 2 GPUs cannot hold DeepSeek-V3 at any shape -> no feasible config.
    with pytest.raises(NoViableParallelConfig):
        parallel_configs_for(
            DEEPSEEK,
            "gb200",
            gpu_budget=2,
            deployment_mode="agg",
            backend="trtllm",
            max_seq_len=8192,
        )
