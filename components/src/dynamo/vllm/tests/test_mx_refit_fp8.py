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


def test_mx_refit_fp8_helper_keeps_v6_scope():
    source = inspect.getsource(fp8)
    assert "Fp8MoEMethod" not in source


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


def test_mx_maybe_process_fp8_kv_cache_uses_kv_only_helper(monkeypatch):
    model = torch.nn.Linear(1, 1)
    calls = []

    def fake_process_fp8_kv_cache_modules(processed_model, target_device):
        calls.append((processed_model, target_device))
        return 1

    monkeypatch.setattr(
        extension,
        "_process_fp8_kv_cache_modules",
        fake_process_fp8_kv_cache_modules,
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(cache_dtype="fp8")
            ),
            model=model,
        )
    )

    extension.MxRefitWorkerExtension._mx_maybe_process_fp8_kv_cache(worker)

    assert calls == [(model, next(model.parameters()).device)]


def test_fp8_load_weights_quantizes_refit_weight(monkeypatch):
    loaded_weights = []

    class Model(torch.nn.Module):
        def load_weights(self, weights):
            loaded_weights.extend(weights)

    fp8_name = "model.layers.0.mlp.gate_proj.weight"

    def fake_is_fp8_weight(name, _model):
        return name == fp8_name

    monkeypatch.setattr(fp8, "_is_fp8_weight", fake_is_fp8_weight)
    dense_weight = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    norm_weight = torch.ones(2, dtype=torch.bfloat16)

    fp8.load_weights(
        [(fp8_name, dense_weight), ("model.norm.weight", norm_weight)],
        SimpleNamespace(model=Model()),
    )

    assert [name for name, _ in loaded_weights] == [
        fp8_name,
        f"{fp8_name}_scale_inv",
        "model.norm.weight",
    ]
    assert loaded_weights[0][1].dtype == torch.float8_e4m3fn
    assert loaded_weights[1][1].dtype == torch.float32
    assert loaded_weights[2][1] is norm_weight


def test_fp8_load_weights_direct_copies_exact_refit_weight(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Module()
            self.layer.weight = torch.nn.Parameter(
                torch.empty(2, 2, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.layer.weight_scale = torch.nn.Parameter(
                torch.empty(1, 1, dtype=torch.float32),
                requires_grad=False,
            )
            self.loaded_weights = []

        def load_weights(self, weights):
            self.loaded_weights.extend(weights)

    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda name, _model: True)
    monkeypatch.setattr(fp8, "maybe_post_process_fp8_weight_block", lambda _layer: None)
    model = Model()

    fp8.load_weights(
        [("layer.weight", torch.arange(4, dtype=torch.bfloat16).reshape(2, 2))],
        SimpleNamespace(model=model),
    )

    assert model.loaded_weights == []
    assert model.layer.weight.dtype == torch.float8_e4m3fn
    assert model.layer.weight_scale.dtype == torch.float32


def test_fp8_load_weights_direct_copies_fused_refit_slice(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            layer = self.model.layers[0]
            layer.self_attn = torch.nn.Module()
            layer.self_attn.qkv_proj = torch.nn.Module()
            qkv = layer.self_attn.qkv_proj
            qkv.output_partition_sizes = [2, 1, 1]
            qkv.weight_block_size = [1, 1]
            qkv.weight = torch.nn.Parameter(
                torch.zeros(4, 2, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            qkv.weight_scale_inv = torch.nn.Parameter(
                torch.zeros(4, 1, dtype=torch.float32),
                requires_grad=False,
            )
            self.loaded_weights = []

        def load_weights(self, weights):
            self.loaded_weights.extend(weights)

    def fake_cast(tensor, *, weight_block_size):
        del weight_block_size
        scale = torch.full(
            (tensor.shape[0], 1, 1),
            7.0,
            dtype=torch.float32,
        )
        return tensor.to(torch.float8_e4m3fn), scale

    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda name, _model: True)
    monkeypatch.setattr(fp8, "cast_tensor_to_fp8_blockwise", fake_cast)
    monkeypatch.setattr(fp8, "_fp8_refit_variants", lambda _module, w, s: [(w, s)])
    model = Model()
    source = torch.tensor([[4.0, 5.0]], dtype=torch.bfloat16)

    fp8.load_weights(
        [("model.layers.0.self_attn.k_proj.weight", source)],
        SimpleNamespace(model=model),
    )

    qkv = model.model.layers[0].self_attn.qkv_proj
    assert model.loaded_weights == []
    assert torch.equal(qkv.weight[2].float(), source.to(torch.float8_e4m3fn).float()[0])
    assert qkv.weight_scale_inv[2, 0].item() == 7.0
    assert qkv.weight_scale_inv[[0, 1, 3]].sum().item() == 0.0


def test_fp8_load_weights_uses_matching_processed_scale_variant(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Module()
            self.layer.weight = torch.nn.Parameter(
                torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.layer.weight_scale_inv = torch.nn.Parameter(
                torch.zeros(2, 1, dtype=torch.float32),
                requires_grad=False,
            )
            self.loaded_weights = []

        def load_weights(self, weights):
            self.loaded_weights.extend(weights)

    def fake_cast(tensor, *, weight_block_size):
        del weight_block_size
        return tensor.to(torch.float8_e4m3fn), torch.ones(1, 1, 1)

    def fake_variants(_module, weight, scale):
        raw_scale = scale
        processed_scale = torch.full((2, 1), 3.0, dtype=torch.float32)
        return [(weight, raw_scale), (weight, processed_scale)]

    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda name, _model: True)
    monkeypatch.setattr(fp8, "cast_tensor_to_fp8_blockwise", fake_cast)
    monkeypatch.setattr(fp8, "_fp8_refit_variants", fake_variants)
    model = Model()

    fp8.load_weights(
        [("layer.weight", torch.arange(4, dtype=torch.bfloat16).reshape(2, 2))],
        SimpleNamespace(model=model),
    )

    assert model.loaded_weights == []
    assert torch.equal(model.layer.weight_scale_inv, torch.full((2, 1), 3.0))
