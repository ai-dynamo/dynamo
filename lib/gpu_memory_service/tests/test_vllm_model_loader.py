# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for vLLM GMS model-loader helpers."""

from __future__ import annotations

import contextlib
import types

import pytest

torch = pytest.importorskip("torch", reason="torch is required")

try:
    from gpu_memory_service.integrations.vllm import model_loader
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _FakeGMSClient:
    total_bytes = 0

    def __init__(self) -> None:
        self.close_best_effort: bool | None = None

    def close(self, *, best_effort: bool = False) -> None:
        self.close_best_effort = best_effort


def test_create_meta_model_skips_vllm_post_load_hooks(monkeypatch):
    """Meta construction should not run CUDA-capable vLLM post-load hooks."""
    model = torch.nn.Module()
    model_config = types.SimpleNamespace(dtype=torch.float32)
    calls: list[str] = []

    def initialize_model(*, vllm_config, model_config):
        calls.append("initialize")
        return model

    def process_weights_after_loading(*_args, **_kwargs):
        calls.append("post_load")
        raise AssertionError("post-load hooks must not run on the meta model")

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.model_executor.model_loader.utils":
            return types.SimpleNamespace(
                initialize_model=initialize_model,
                process_weights_after_loading=process_weights_after_loading,
            )
        if name == "vllm.utils.torch_utils":
            return types.SimpleNamespace(
                set_default_torch_dtype=lambda _dtype: contextlib.nullcontext()
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(model_loader, "setup_meta_tensor_workaround", lambda: None)
    monkeypatch.setattr(
        model_loader,
        "vllm_meta_init_workarounds",
        lambda: contextlib.nullcontext(),
    )

    loaded = model_loader._create_meta_model(object(), model_config)

    assert loaded is model
    assert calls == ["initialize"]


def test_load_read_mode_uses_best_effort_cleanup(monkeypatch):
    """Read-side import errors should not be masked by CUDA cleanup sync."""
    gms_client = _FakeGMSClient()

    def fail_create_meta_model(*_args, **_kwargs):
        raise RuntimeError("root cause")

    monkeypatch.setattr(model_loader, "_create_meta_model", fail_create_meta_model)

    with pytest.raises(RuntimeError, match="root cause"):
        model_loader._load_read_mode(gms_client, object(), object(), device_index=0)

    assert gms_client.close_best_effort is True


def test_load_read_mode_rebuilds_mla_after_materialization(monkeypatch):
    """Read-side imports rebuild MLA state after GMS materialization."""
    gms_client = _FakeGMSClient()
    model_config = type("ModelConfig", (), {"dtype": torch.float32})()
    calls: list[str] = []

    class _MLA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kv_b_proj = object()
            self.kv_lora_rank = 1
            self.num_heads = 1

        def process_weights_after_loading(self, dtype):
            calls.append(f"mla:{dtype}")

    model = torch.nn.Module()
    model.mla = _MLA()

    monkeypatch.setattr(
        model_loader,
        "_create_meta_model",
        lambda *_args, **_kwargs: calls.append("create") or model,
    )
    monkeypatch.setattr(
        model_loader,
        "materialize_module_from_gms",
        lambda *_args, **_kwargs: calls.append("materialize"),
    )
    monkeypatch.setattr(
        model_loader,
        "_process_fused_moe_kernels_after_gms_materialization",
        lambda *_args, **_kwargs: calls.append("moe"),
    )
    monkeypatch.setattr(
        model_loader,
        "get_mx_load_context",
        lambda *_args, **_kwargs: None,
    )

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.utils.torch_utils":
            return types.SimpleNamespace(
                set_default_torch_dtype=lambda _dtype: contextlib.nullcontext()
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(torch.cuda, "_exchange_device", lambda _device: -1)
    monkeypatch.setattr(torch.cuda, "_maybe_exchange_device", lambda _device: -1)

    loaded = model_loader._load_read_mode(
        gms_client,
        object(),
        model_config,
        device_index=0,
    )

    assert loaded is model
    assert calls == [
        "create",
        "materialize",
        "moe",
        "mla:torch.float32",
    ]


def test_load_read_mode_rebuilds_fused_moe_before_mla(monkeypatch):
    """Read-side imports rebuild MoE runtime kernels before MLA state."""
    gms_client = _FakeGMSClient()
    model_config = type("ModelConfig", (), {"dtype": torch.float32})()
    calls: list[str] = []
    model = torch.nn.Module()

    monkeypatch.setattr(
        model_loader,
        "_create_meta_model",
        lambda *_args, **_kwargs: calls.append("create") or model,
    )
    monkeypatch.setattr(
        model_loader,
        "materialize_module_from_gms",
        lambda *_args, **_kwargs: calls.append("materialize"),
    )
    monkeypatch.setattr(
        model_loader,
        "_process_fused_moe_kernels_after_gms_materialization",
        lambda *_args, **_kwargs: calls.append("moe"),
    )
    monkeypatch.setattr(
        model_loader,
        "_process_mla_weights_after_gms_materialization",
        lambda *_args, **_kwargs: calls.append("mla"),
    )
    monkeypatch.setattr(
        model_loader,
        "get_mx_load_context",
        lambda *_args, **_kwargs: None,
    )

    loaded = model_loader._load_read_mode(
        gms_client,
        object(),
        model_config,
        device_index=0,
    )

    assert loaded is model
    assert calls == ["create", "materialize", "moe", "mla"]
