# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache feasibility wrapper around AIC Core's memory estimator."""

import pytest

import aisimulate.spica.kv_estimate as kv_estimate_mod
from aisimulate.spica.kv_estimate import (
    NoPerfDatabase,
    estimate_kv_tokens,
    feasible_shape_tokens,
    memory_fraction_kind,
    resolve_backend_version,
)
from aisimulate.spica.parallel_enum import ParallelShape

_COMMON = dict(model_name="m", hardware_sku="hw", backend="trtllm", backend_version="v")


def test_memory_fraction_kind():
    assert memory_fraction_kind("trtllm") == "of_free"
    assert memory_fraction_kind("vllm") == "of_total"
    assert memory_fraction_kind("sglang") == "of_total"


def test_estimate_kv_tokens_returns_capacity(monkeypatch):
    monkeypatch.setattr(
        kv_estimate_mod,
        "estimate_kv_cache",
        lambda *args, **kwargs: {"total_kv_size_tokens": 123456},
    )
    sh = ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4)
    assert estimate_kv_tokens(sh, **_COMMON) == 123456


def test_estimate_kv_tokens_oom_returns_none(monkeypatch):
    def boom(*a, **k):
        raise ValueError(
            "no KV budget: non-KV memory (361903882240 bytes) meets/exceeds the KV-cacheable budget"
        )

    monkeypatch.setattr(kv_estimate_mod, "estimate_kv_cache", boom)
    sh = ParallelShape(tp=1, dp=2, moe_tp=1, moe_ep=2)
    assert estimate_kv_tokens(sh, **_COMMON) is None


def test_estimate_kv_tokens_propagates_other_errors(monkeypatch):
    def boom(*a, **k):
        raise ValueError("incompatible memory fraction")

    monkeypatch.setattr(kv_estimate_mod, "estimate_kv_cache", boom)
    sh = ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1)
    with pytest.raises(ValueError, match="incompatible"):
        estimate_kv_tokens(sh, **_COMMON)


def test_feasible_shape_tokens_filters_short_and_oom_and_dedups(monkeypatch):
    # token capacity == tp_size * 10000; tp_size==1 OOMs (no KV budget).
    calls = []

    def fake(model_path, system, backend, *, tp_size, **k):
        calls.append(tp_size)
        if tp_size == 1:
            raise ValueError("no KV budget: ...")
        return {"total_kv_size_tokens": tp_size * 10000}

    monkeypatch.setattr(kv_estimate_mod, "estimate_kv_cache", fake)
    big = ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4)  # 40000 -> feasible
    small = ParallelShape(
        tp=2, dp=1, moe_tp=1, moe_ep=2
    )  # 20000 -> < max_seq_len -> dropped
    oom = ParallelShape(tp=1, dp=2, moe_tp=1, moe_ep=2)  # None -> dropped

    feasible = feasible_shape_tokens(
        [big, big, small, oom],  # big repeated -> deduped
        model_name="m",
        hardware_sku="hw",
        backend="trtllm",
        backend_version="v",
        max_seq_len=25000,
    )
    assert set(feasible) == {big}
    assert feasible[big] == 40000
    assert calls == [4, 2, 1]  # one estimate per distinct shape


def test_feasible_shape_tokens_resolves_version_when_missing(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.spica.kv_estimate.get_latest_database_version",
        lambda *a, **k: None,
    )
    with pytest.raises(NoPerfDatabase):
        feasible_shape_tokens(
            [ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1)],
            model_name="m",
            hardware_sku="nosuch_sku",
            backend="trtllm",
            max_seq_len=8192,
        )


def test_resolve_backend_version_missing(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.spica.kv_estimate.get_latest_database_version",
        lambda *a, **k: None,
    )
    with pytest.raises(NoPerfDatabase):
        resolve_backend_version("nosuch_sku", "trtllm")


# --- integration: real AIC Core estimate ---


@pytest.mark.model("Qwen/Qwen3-32B")
def test_real_estimate_dense_qwen_feasible():
    ver = resolve_backend_version("h200_sxm", "trtllm")
    sh = ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1)  # dense Qwen3-32B
    tokens = estimate_kv_tokens(
        sh,
        model_name="Qwen/Qwen3-32B",
        hardware_sku="h200_sxm",
        backend="trtllm",
        backend_version=ver,
    )
    assert tokens is not None and tokens > 0


@pytest.mark.model("deepseek-ai/DeepSeek-V3")
def test_real_estimate_deepseek_two_gpus_oom():
    ver = resolve_backend_version("gb200", "trtllm")
    sh = ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=2)  # MoE TEP at 2 GPUs
    tokens = estimate_kv_tokens(
        sh,
        model_name="deepseek-ai/DeepSeek-V3",
        hardware_sku="gb200",
        backend="trtllm",
        backend_version=ver,
    )
    assert tokens is None  # 2 GPUs cannot hold DeepSeek-V3 -> no KV budget
