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
        "mla:torch.float32",
    ]
