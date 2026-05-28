# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for vLLM GMS model-loader helpers."""

from __future__ import annotations

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
