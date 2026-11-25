# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for vLLM GMS read-only meta post-processing and rebuild."""

from __future__ import annotations

import contextvars
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

import torch
from gpu_memory_service.integrations.vllm import model_loader

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def _install_vllm_stub(
    monkeypatch: pytest.MonkeyPatch, *, api_style: str
) -> SimpleNamespace:
    """Install a minimal ``vllm.model_executor`` module tree for CPU tests.

    ``api_style`` selects between vLLM 0.33 (``is_deferred_attention_layer``
    under ``model_loader.utils`` and 1-arg ``process_weights_after_loading(dtype)``)
    and vLLM 0.34+ (``layers.attention`` export and 0-arg hook).
    """
    weights_pre_processed: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "test_weights_pre_processed", default=False
    )

    @contextmanager
    def weights_already_processed():
        token = weights_pre_processed.set(True)
        try:
            yield
        finally:
            weights_pre_processed.reset(token)

    def is_weights_pre_processed() -> bool:
        return weights_pre_processed.get()

    class QuantizeMethodBase:
        supports_pre_processed_weights = False

    class BaseKVCacheMethod(QuantizeMethodBase):
        supports_pre_processed_weights = True

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            if is_weights_pre_processed():
                if api_style == "0.33":
                    for name in ("k_scale", "v_scale", "q_scale", "prob_scale"):
                        if hasattr(layer, name):
                            delattr(layer, name)
                if hasattr(layer, "_k_scale"):
                    layer._k_scale_float = layer._k_scale.item()
                    layer._v_scale_float = layer._v_scale.item()
                    layer._q_scale_float = layer._q_scale.item()
                    layer._k_scale_cpu = layer._k_scale.detach().to("cpu")
                    layer._v_scale_cpu = layer._v_scale.detach().to("cpu")
                return
            # Unprocessed path reads tensor values and fails on meta tensors.
            if layer.k_scale > 0.0:
                layer._k_scale.fill_(layer.k_scale.item())

    class CompressedTensorsKVCacheMethod(BaseKVCacheMethod):
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # Mirrors vLLM 0.33 / 0.34 overriding BaseKVCacheMethod without
            # checking is_weights_pre_processed().
            layer._k_scale = layer.k_scale
            layer._v_scale = layer.v_scale
            layer._q_scale = layer.q_scale
            del layer.k_scale, layer.v_scale, layer.q_scale

    class _DeferredAttnBase(torch.nn.Module):
        pass

    def is_deferred_attention_layer(layer: torch.nn.Module) -> bool:
        return isinstance(layer, _DeferredAttnBase)

    retie_calls: list[bool | None] = []

    def maybe_retie_word_embeddings(model: torch.nn.Module, model_config) -> None:
        untied = getattr(model_config, "word_embeddings_untied_by_checkpoint", False)
        retie_calls.append(untied)
        if untied:
            raise NotImplementedError("Cannot compare meta tensors in retie")

    loader_utils_mod = ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils_mod.initialize_model = lambda **kwargs: kwargs["model_config"]._model
    loader_utils_mod.maybe_retie_word_embeddings = maybe_retie_word_embeddings
    loader_utils_mod.process_weights_after_loading = lambda *args, **kwargs: None

    layers_attention_mod = ModuleType("vllm.model_executor.layers.attention")
    if api_style == "0.33":
        loader_utils_mod.is_deferred_attention_layer = is_deferred_attention_layer
    else:
        layers_attention_mod.is_deferred_attention_layer = is_deferred_attention_layer

    base_config_mod = ModuleType("vllm.model_executor.layers.quantization.base_config")
    base_config_mod.QuantizeMethodBase = QuantizeMethodBase

    kv_cache_mod = ModuleType("vllm.model_executor.layers.quantization.kv_cache")
    kv_cache_mod.BaseKVCacheMethod = BaseKVCacheMethod

    ct_mod = ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
    )
    ct_mod.CompressedTensorsKVCacheMethod = CompressedTensorsKVCacheMethod

    model_exec_utils_mod = ModuleType("vllm.model_executor.utils")
    model_exec_utils_mod.weights_already_processed = weights_already_processed
    model_exec_utils_mod.is_weights_pre_processed = is_weights_pre_processed

    @contextmanager
    def set_default_torch_dtype(_dtype):
        yield

    torch_utils_mod = ModuleType("vllm.utils.torch_utils")
    torch_utils_mod.set_default_torch_dtype = set_default_torch_dtype

    vllm_pkg = ModuleType("vllm")
    vllm_pkg.__path__ = []
    model_exec_pkg = ModuleType("vllm.model_executor")
    model_exec_pkg.__path__ = []
    layers_pkg = ModuleType("vllm.model_executor.layers")
    layers_pkg.__path__ = []
    quant_pkg = ModuleType("vllm.model_executor.layers.quantization")
    quant_pkg.__path__ = []
    ct_pkg = ModuleType("vllm.model_executor.layers.quantization.compressed_tensors")
    ct_pkg.__path__ = []
    model_loader_pkg = ModuleType("vllm.model_executor.model_loader")
    model_loader_pkg.__path__ = []
    vllm_utils_pkg = ModuleType("vllm.utils")
    vllm_utils_pkg.__path__ = []

    vllm_pkg.model_executor = model_exec_pkg
    vllm_pkg.utils = vllm_utils_pkg
    model_exec_pkg.layers = layers_pkg
    model_exec_pkg.model_loader = model_loader_pkg
    model_exec_pkg.utils = model_exec_utils_mod
    layers_pkg.attention = layers_attention_mod
    layers_pkg.quantization = quant_pkg
    quant_pkg.base_config = base_config_mod
    quant_pkg.kv_cache = kv_cache_mod
    quant_pkg.compressed_tensors = ct_pkg
    ct_pkg.compressed_tensors = ct_mod
    model_loader_pkg.utils = loader_utils_mod
    vllm_utils_pkg.torch_utils = torch_utils_mod

    modules = {
        "vllm": vllm_pkg,
        "vllm.model_executor": model_exec_pkg,
        "vllm.model_executor.utils": model_exec_utils_mod,
        "vllm.model_executor.layers": layers_pkg,
        "vllm.model_executor.layers.attention": layers_attention_mod,
        "vllm.model_executor.layers.quantization": quant_pkg,
        "vllm.model_executor.layers.quantization.base_config": base_config_mod,
        "vllm.model_executor.layers.quantization.kv_cache": kv_cache_mod,
        "vllm.model_executor.layers.quantization.compressed_tensors": ct_pkg,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors": ct_mod,
        "vllm.model_executor.model_loader": model_loader_pkg,
        "vllm.model_executor.model_loader.utils": loader_utils_mod,
        "vllm.utils": vllm_utils_pkg,
        "vllm.utils.torch_utils": torch_utils_mod,
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return SimpleNamespace(
        QuantizeMethodBase=QuantizeMethodBase,
        BaseKVCacheMethod=BaseKVCacheMethod,
        CompressedTensorsKVCacheMethod=CompressedTensorsKVCacheMethod,
        DeferredAttnBase=_DeferredAttnBase,
        is_weights_pre_processed=is_weights_pre_processed,
        retie_calls=retie_calls,
    )


def _make_kv_layer(kv_method) -> torch.nn.Module:
    layer = torch.nn.Module()
    for name in (
        "k_scale",
        "v_scale",
        "q_scale",
        "prob_scale",
        "k_zero_point",
        "v_zero_point",
        "q_zero_point",
    ):
        layer.register_parameter(
            name, torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        )
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        layer.register_buffer(name, torch.tensor(1.0, dtype=torch.float32))
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer._q_scale_float = 1.0
    layer._prob_scale_float = 1.0
    layer.quant_method = kv_method
    return layer


@pytest.mark.parametrize("api_style", ["0.33", "0.34"])
def test_create_meta_model_isolates_meta_failures_across_layers(
    monkeypatch: pytest.MonkeyPatch, api_style: str
) -> None:
    """One failing meta quant method must not abort later layers or attn hooks."""
    vllm_stub = _install_vllm_stub(monkeypatch, api_style=api_style)
    monkeypatch.setattr(model_loader, "setup_meta_tensor_workaround", lambda: None)

    class _PreProcessedMoeMethod(vllm_stub.QuantizeMethodBase):
        supports_pre_processed_weights = True

        def __init__(self) -> None:
            self.moe_kernel = None

        def process_weights_after_loading(self, layer) -> None:
            raise NotImplementedError("torch.allclose not implemented for meta")

    class _NonPreProcessedFailing(vllm_stub.QuantizeMethodBase):
        supports_pre_processed_weights = False

        def process_weights_after_loading(self, layer) -> None:
            raise RuntimeError("legacy quant requires weight values")

    class _SubsequentLinear(vllm_stub.QuantizeMethodBase):
        supports_pre_processed_weights = False

        def __init__(self) -> None:
            self.calls = 0

        def process_weights_after_loading(self, layer) -> None:
            self.calls += 1

    class _DeferredAttn(vllm_stub.DeferredAttnBase):
        def __init__(self) -> None:
            super().__init__()
            self.hook_args: list[tuple[object, ...]] = []

        if api_style == "0.33":

            def process_weights_after_loading(self, act_dtype) -> None:
                self.hook_args.append((act_dtype,))

        else:

            def process_weights_after_loading(self) -> None:
                self.hook_args.append(())

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = torch.nn.Module()
            self.moe.quant_method = _PreProcessedMoeMethod()
            self.legacy = torch.nn.Module()
            self.legacy.quant_method = _NonPreProcessedFailing()
            self.kv = _make_kv_layer(vllm_stub.BaseKVCacheMethod())
            self.after = torch.nn.Module()
            self.after.quant_method = _SubsequentLinear()
            self.after_tp_updates = 0
            self.after.update_param_tp_status = lambda: setattr(
                self, "after_tp_updates", self.after_tp_updates + 1
            )
            self.attn = _DeferredAttn()
            self.model_hook_calls = 0

        def process_weights_after_loading(self) -> None:
            self.model_hook_calls += 1

    with torch.device("meta"):
        expected_model = _Model()

    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        word_embeddings_untied_by_checkpoint=True,
        _model=expected_model,
    )

    model = model_loader._create_meta_model(SimpleNamespace(), model_config)

    assert vllm_stub.retie_calls == [False]
    assert model_config.word_embeddings_untied_by_checkpoint is True
    assert model.after.quant_method.calls == 1
    assert model.after_tp_updates == 1
    assert model.model_hook_calls == 1
    assert model.attn.hook_args == (
        [(torch.bfloat16,)] if api_style == "0.33" else [()]
    )
    deferred = getattr(model, model_loader._DEFERRED_QUANT_MODULES_ATTR)
    assert deferred == [model.moe]


@pytest.mark.parametrize("api_style", ["0.33", "0.34"])
def test_load_read_mode_finalizes_moe_kernels_and_kv_scales(
    monkeypatch: pytest.MonkeyPatch, api_style: str
) -> None:
    """RO attach rebuilds MoE kernels and FP8 KV scales including _prob_scale_float."""
    vllm_stub = _install_vllm_stub(monkeypatch, api_style=api_style)
    monkeypatch.setattr(model_loader, "setup_meta_tensor_workaround", lambda: None)
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *_args: None)

    class _NvFp4MoeMethod(vllm_stub.QuantizeMethodBase):
        supports_pre_processed_weights = True

        def __init__(self) -> None:
            self.moe_kernel = None
            self.rebuilt_pre_processed = False

        def process_weights_after_loading(self, layer) -> None:
            if not vllm_stub.is_weights_pre_processed():
                raise NotImplementedError("Cannot run torch.allclose on meta tensor")
            self.rebuilt_pre_processed = True
            # Mirrors TrtLlmNvFp4ExpertsBase registering g1_scale_c post-load.
            layer.register_parameter(
                "g1_scale_c",
                torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False),
            )
            self.moe_kernel = "compiled_nvfp4_kernel"

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = torch.nn.Module()
            self.moe.quant_method = _NvFp4MoeMethod()
            self.moe_tp_updates = 0
            self.moe.update_param_tp_status = lambda: setattr(
                self, "moe_tp_updates", self.moe_tp_updates + 1
            )
            self.attn_base = _make_kv_layer(vllm_stub.BaseKVCacheMethod())
            self.attn_ct = _make_kv_layer(vllm_stub.CompressedTensorsKVCacheMethod())

    with torch.device("meta"):
        meta_model = _Model()

    def _fake_materialize(
        _client, target: torch.nn.Module, *, device_index: int
    ) -> None:
        assert device_index == 0
        for layer in (target.attn_base, target.attn_ct):
            layer._buffers["_k_scale"] = torch.tensor(0.5, dtype=torch.float32)
            layer._buffers["_v_scale"] = torch.tensor(0.25, dtype=torch.float32)
            layer._buffers["_q_scale"] = torch.tensor(0.125, dtype=torch.float32)
            layer._buffers["_prob_scale"] = torch.tensor(0.0625, dtype=torch.float32)

    monkeypatch.setattr(model_loader, "materialize_module_from_gms", _fake_materialize)

    closed: list[bool] = []
    gms_client = SimpleNamespace(total_bytes=1 << 20, close=lambda: closed.append(True))
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        word_embeddings_untied_by_checkpoint=False,
        _model=meta_model,
    )

    loaded = model_loader._load_read_mode(
        gms_client, SimpleNamespace(), model_config, 0
    )

    assert not closed
    assert not hasattr(loaded, model_loader._DEFERRED_QUANT_MODULES_ATTR)
    assert loaded.moe.quant_method.rebuilt_pre_processed is True
    assert loaded.moe.quant_method.moe_kernel == "compiled_nvfp4_kernel"
    assert loaded.moe_tp_updates == 1
    assert "g1_scale_c" in loaded.moe._parameters

    for layer in (loaded.attn_base, loaded.attn_ct):
        assert layer._k_scale_float == 0.5
        assert layer._v_scale_float == 0.25
        assert layer._q_scale_float == 0.125
        assert layer._prob_scale_float == 0.0625
        assert float(layer._k_scale_cpu) == 0.5
        assert float(layer._v_scale_cpu) == 0.25
        assert not [n for n, p in layer.named_parameters() if p.is_meta]


def test_load_read_mode_closes_client_on_per_head_kv_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-head KV scales fail closed and release the GMS client."""
    vllm_stub = _install_vllm_stub(monkeypatch, api_style="0.34")
    monkeypatch.setattr(model_loader, "setup_meta_tensor_workaround", lambda: None)
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *_args: None)

    with torch.device("meta"):
        meta_model = torch.nn.Sequential(
            _make_kv_layer(vllm_stub.CompressedTensorsKVCacheMethod())
        )

    def _fake_materialize(
        _client, target: torch.nn.Module, *, device_index: int
    ) -> None:
        target[0]._buffers["_k_scale"] = torch.tensor([0.5, 0.75])
        target[0]._buffers["_v_scale"] = torch.tensor([0.5, 0.75])
        target[0]._buffers["_q_scale"] = torch.tensor(0.25)

    monkeypatch.setattr(model_loader, "materialize_module_from_gms", _fake_materialize)

    closed: list[bool] = []
    gms_client = SimpleNamespace(total_bytes=0, close=lambda: closed.append(True))
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        word_embeddings_untied_by_checkpoint=False,
        _model=meta_model,
    )

    with pytest.raises(RuntimeError, match="per-head KV-cache scales"):
        model_loader._load_read_mode(gms_client, SimpleNamespace(), model_config, 0)

    assert closed == [True]


def test_load_read_mode_skips_vllm_finalize_when_model_has_no_quant_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models without quant_method or deferred hooks skip importing vLLM in finalize."""
    monkeypatch.setitem(sys.modules, "vllm.model_executor", None)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", None)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.quantization.kv_cache", None
    )
    monkeypatch.setattr(
        model_loader, "_create_meta_model", lambda *_args: torch.nn.Linear(4, 4)
    )
    monkeypatch.setattr(
        model_loader, "materialize_module_from_gms", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *_args: None)

    gms_client = SimpleNamespace(total_bytes=128, close=lambda: None)
    model = model_loader._load_read_mode(
        gms_client, SimpleNamespace(), SimpleNamespace(), 0
    )

    assert isinstance(model, torch.nn.Linear)


def test_load_read_mode_retries_failed_hooks_after_kv_scale_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferred attention and model hooks retry post-materialization after KV scales."""
    vllm_stub = _install_vllm_stub(monkeypatch, api_style="0.34")
    monkeypatch.setattr(model_loader, "setup_meta_tensor_workaround", lambda: None)
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *_args: None)

    observed_scales: list[tuple[float, float]] = []

    class _FailingMetaAttention(vllm_stub.DeferredAttnBase):
        def __init__(self) -> None:
            super().__init__()
            self.kv = _make_kv_layer(vllm_stub.BaseKVCacheMethod())

        def process_weights_after_loading(self, dtype=None, **kwargs) -> None:
            del dtype, kwargs
            if self.kv._k_scale.is_meta:
                raise NotImplementedError("Cannot inspect meta kv scales")
            observed_scales.append((self.kv._k_scale_float, self.kv._prob_scale_float))

    class _RootModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = _FailingMetaAttention()
            self.root_runs = 0

        def process_weights_after_loading(self, **kwargs) -> None:
            del kwargs
            if self.attn.kv._prob_scale.is_meta:
                raise RuntimeError("root hook needs materialized scales")
            self.root_runs += 1

    with torch.device("meta"):
        meta_model = _RootModel()

    def _fake_materialize(
        _client, target: torch.nn.Module, *, device_index: int
    ) -> None:
        assert device_index == 0
        target.attn.kv._buffers["_k_scale"] = torch.tensor(0.5, dtype=torch.float32)
        target.attn.kv._buffers["_v_scale"] = torch.tensor(0.25, dtype=torch.float32)
        target.attn.kv._buffers["_q_scale"] = torch.tensor(0.125, dtype=torch.float32)
        target.attn.kv._buffers["_prob_scale"] = torch.tensor(
            0.0625, dtype=torch.float32
        )

    monkeypatch.setattr(model_loader, "materialize_module_from_gms", _fake_materialize)
    gms_client = SimpleNamespace(total_bytes=128, close=lambda: None)
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        word_embeddings_untied_by_checkpoint=False,
        _model=meta_model,
    )

    loaded = model_loader._load_read_mode(
        gms_client, SimpleNamespace(), model_config, 0
    )

    assert observed_scales == [(0.5, 0.0625)]
    assert loaded.root_runs == 1


def test_load_read_mode_raises_if_deferred_hook_still_fails_post_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferred hooks must fail loudly after materialization rather than swallowing errors."""
    vllm_stub = _install_vllm_stub(monkeypatch, api_style="0.34")
    monkeypatch.setattr(model_loader, "setup_meta_tensor_workaround", lambda: None)
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *_args: None)

    class _AlwaysFailingAttn(vllm_stub.DeferredAttnBase):
        def process_weights_after_loading(self) -> None:
            raise RuntimeError("hook broken even on live weights")

    with torch.device("meta"):
        meta_model = _AlwaysFailingAttn()

    monkeypatch.setattr(
        model_loader, "materialize_module_from_gms", lambda *_args, **_kwargs: None
    )
    closed: list[bool] = []
    gms_client = SimpleNamespace(total_bytes=0, close=lambda: closed.append(True))
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        word_embeddings_untied_by_checkpoint=False,
        _model=meta_model,
    )

    with pytest.raises(RuntimeError, match="hook broken even on live weights"):
        model_loader._load_read_mode(gms_client, SimpleNamespace(), model_config, 0)

    assert closed == [True]


def test_finalize_kv_cache_rejects_non_scalar_prob_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-scalar _q_scale or _prob_scale fails closed during KV rebuild."""
    vllm_stub = _install_vllm_stub(monkeypatch, api_style="0.34")
    layer = _make_kv_layer(vllm_stub.BaseKVCacheMethod())
    layer._buffers["_prob_scale"] = torch.tensor([0.5, 0.25], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="per-head KV-cache scales"):
        model_loader._finalize_kv_cache_layer(
            layer, layer.quant_method, vllm_stub.BaseKVCacheMethod
        )


def test_real_vllm_symbol_contracts() -> None:
    """Verify real vLLM imports and symbols when vllm.model_executor is installed."""
    pytest.importorskip("vllm.model_executor")

    from vllm.model_executor.layers.attention import is_deferred_attention_layer
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.model_loader import utils as vllm_loader_utils
    from vllm.model_executor.utils import weights_already_processed

    assert callable(is_deferred_attention_layer)
    assert callable(vllm_loader_utils.initialize_model)
    assert callable(vllm_loader_utils.maybe_retie_word_embeddings)
    assert callable(weights_already_processed)
    assert issubclass(BaseKVCacheMethod, QuantizeMethodBase)
    assert BaseKVCacheMethod.supports_pre_processed_weights is True
