# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aisimulate import aic

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


def test_materializer_sets_rank_local_capacity_without_forwarding_nextn(
    monkeypatch,
) -> None:
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 46000

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    lowered = aic.materialize_aic_num_gpu_blocks(
        {
            "engine_type": "vllm",
            "aic_backend": "vllm",
            "aic_system": "h200_sxm",
            "aic_model_path": "test-model",
            "aic_attention_dp_size": 2,
            "aic_pp_size": 3,
            "aic_nextn": 3,
            "systems_path": "/tmp/custom-systems.yaml",
            "block_size": 64,
        }
    )

    assert lowered["num_gpu_blocks"] == 46000
    assert lowered["dp_size"] == 2
    assert calls[0]["attention_dp_size"] == 2
    assert calls[0]["pp_size"] == 3
    assert calls[0]["systems_path"] == "/tmp/custom-systems.yaml"
    assert "nextn" not in calls[0]


def test_capacity_wrapper_owns_backend_defaults_and_quant_normalization(
    monkeypatch,
) -> None:
    calls = []

    def estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return 123

    from aiconfigurator_core.sdk import memory

    monkeypatch.setattr(memory, "estimate_num_gpu_blocks", estimate)
    blocks = aic.estimate_num_gpu_blocks(
        backend_name="vllm",
        system="h200_sxm",
        model_path="test-model",
        tp_size=1,
        block_size=64,
        max_num_batched_tokens=4096,
        pp_size=3,
        gemm_dtype="int4",
        fmha_dtype="auto",
        systems_path="/tmp/custom-systems.yaml",
    )

    assert blocks == 123
    args, kwargs = calls[0]
    assert args == ("test-model", "h200_sxm", "vllm")
    assert kwargs["backend_version"] == "0.19.0"
    assert kwargs["memory_fraction_kind"] == "of_total"
    assert kwargs["memory_fraction_value"] == 0.9
    assert kwargs["pp_size"] == 3
    assert kwargs["systems_path"] == "/tmp/custom-systems.yaml"
    assert kwargs["gemm_quant_mode"] == "int4_wo"
    assert kwargs["fmha_quant_mode"] is None
    assert "nextn" not in kwargs


def test_explicit_capacity_is_preserved_without_estimation(monkeypatch) -> None:
    monkeypatch.setattr(
        aic,
        "estimate_num_gpu_blocks",
        lambda **_kwargs: pytest.fail("explicit capacity must not be estimated"),
    )

    raw = {
        "aic_backend": "vllm",
        "aic_model_path": "test-model",
        "num_gpu_blocks": 17,
    }
    assert aic.materialize_aic_num_gpu_blocks(raw) == raw


def test_materializer_preserves_explicit_zero_values(monkeypatch) -> None:
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    aic.materialize_aic_num_gpu_blocks(
        {
            "aic_backend": "sglang",
            "aic_model_path": "test-model",
            "aic_tp_size": 0,
            "aic_attention_dp_size": 0,
            "block_size": 0,
            "max_num_batched_tokens": 0,
            "gpu_memory_utilization": 0.0,
            "mem_fraction_static": 0.0,
            "free_gpu_memory_fraction": 0.0,
        }
    )

    assert calls[0]["tp_size"] == 0
    assert calls[0]["attention_dp_size"] == 0
    assert calls[0]["block_size"] == 0
    assert calls[0]["max_num_batched_tokens"] == 0
    assert calls[0]["gpu_memory_utilization"] == 0.0
    assert calls[0]["mem_fraction_static"] == 0.0
    assert calls[0]["free_gpu_memory_fraction"] == 0.0
