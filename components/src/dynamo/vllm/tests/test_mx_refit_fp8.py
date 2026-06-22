# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm import main as vllm_main
from dynamo.vllm.mx_refit import extension, fp8


pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class Qwen3ScaleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.attn = torch.nn.Module()
        layer.self_attn.attn.q_scale = torch.nn.Parameter(torch.tensor(-1.0))
        layer.self_attn.attn.k_scale = torch.nn.Parameter(torch.tensor(-1.0))
        layer.self_attn.attn.v_scale = torch.nn.Parameter(torch.tensor(-1.0))
        layer.self_attn.attn.prob_scale = torch.nn.Parameter(torch.tensor(-1.0))


def test_mx_refit_fp8_modules_do_not_import_nemo_rl():
    source = inspect.getsource(extension) + inspect.getsource(fp8)
    assert "import nemo_rl" not in source
    assert "from nemo_rl" not in source


def test_maybe_init_mx_refit_fp8_requires_mx_env(monkeypatch):
    monkeypatch.delenv("DYN_MX_REFIT_ENABLED", raising=False)
    args = SimpleNamespace(
        model="Qwen/Qwen3-8B-Base",
        quantization="fp8",
        kv_cache_dtype="fp8",
        tensor_parallel_size=4,
    )

    assert vllm_main._maybe_init_mx_refit_fp8(args) is False


def test_maybe_init_mx_refit_fp8_skips_bf16_mx(monkeypatch):
    monkeypatch.setenv("DYN_MX_REFIT_ENABLED", "1")
    args = SimpleNamespace(
        model="Qwen/Qwen3-8B-Base",
        quantization=None,
        kv_cache_dtype="auto",
        tensor_parallel_size=4,
    )

    assert vllm_main._maybe_init_mx_refit_fp8(args) is False


def test_maybe_init_mx_refit_fp8_passes_engine_args(monkeypatch):
    calls = []

    def fake_init_fp8(vllm_cfg, model_name, *, model_parallel_size):
        calls.append(
            {
                "vllm_cfg": vllm_cfg,
                "model_name": model_name,
                "model_parallel_size": model_parallel_size,
            }
        )

    monkeypatch.setenv("DYN_MX_REFIT_ENABLED", "1")
    monkeypatch.setattr(fp8, "init_fp8", fake_init_fp8)
    args = SimpleNamespace(
        model="Qwen/Qwen3-8B-Base",
        quantization="fp8",
        kv_cache_dtype="fp8",
        tensor_parallel_size=4,
    )

    assert vllm_main._maybe_init_mx_refit_fp8(args) is True
    assert calls == [
        {
            "vllm_cfg": {
                "precision": "fp8",
                "kv_cache_dtype": "fp8",
                "async_engine": True,
                "use_deep_gemm": False,
            },
            "model_name": "Qwen/Qwen3-8B-Base",
            "model_parallel_size": 4,
        }
    ]


def test_load_fp8_qkv_scale_weights_matches_qwen3_layout():
    model = Qwen3ScaleModule()
    weights = [
        ("model.layers.0.self_attn.attn.q_scale", torch.tensor([1.25])),
        ("model.layers.0.self_attn.k_scale", torch.tensor([1.5])),
        ("model.layers.0.self_attn.v_scale", torch.tensor([1.75])),
        ("model.layers.0.mlp.down_proj.weight", torch.ones(2, 2)),
    ]

    remaining = extension._load_fp8_qkv_scale_weights(weights, model)

    assert [name for name, _ in remaining] == ["model.layers.0.mlp.down_proj.weight"]
    attn = model.model.layers[0].self_attn.attn
    assert attn.q_scale.item() == 1.25
    assert attn.k_scale.item() == 1.5
    assert attn.v_scale.item() == 1.75


def test_load_fp8_qkv_scale_weights_recreates_deleted_scale_parameters():
    model = Qwen3ScaleModule()
    attn = model.model.layers[0].self_attn.attn
    del attn.q_scale
    del attn.prob_scale
    weights = [("model.layers.0.self_attn.q_scale", torch.tensor([3.25]))]

    remaining = extension._load_fp8_qkv_scale_weights(weights, model)

    assert remaining == []
    assert isinstance(attn.q_scale, torch.nn.Parameter)
    assert isinstance(attn.prob_scale, torch.nn.Parameter)
    assert attn.q_scale.item() == 3.25
    assert attn.prob_scale.item() == -1.0


def test_load_fp8_qkv_scale_weights_keeps_unresolved_scales_for_vllm_loader():
    weights = [("model.layers.0.self_attn.k_scale", torch.tensor([1.5]))]

    remaining = extension._load_fp8_qkv_scale_weights(weights, torch.nn.Linear(1, 1))

    assert remaining == weights
