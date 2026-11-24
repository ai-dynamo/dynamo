# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GMS torch module materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from gpu_memory_service.client.torch import module as gms_module
from gpu_memory_service.client.torch.module import materialize_module_from_gms
from gpu_memory_service.client.torch.tensor import TensorMetadata

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@dataclass
class _FakeSpec:
    tensor: torch.Tensor
    meta: TensorMetadata

    def materialize(self, _manager: Any, _device_index: int) -> torch.Tensor:
        return self.tensor


def _spec_for(tensor: torch.Tensor, tensor_type: str = "parameter") -> _FakeSpec:
    return _FakeSpec(
        tensor=tensor,
        meta=TensorMetadata(
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            stride=list(tensor.stride()),
            tensor_type=tensor_type,
        ),
    )


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 4, bias=False)
        self.packed = torch.nn.Parameter(
            torch.empty(4, 8, dtype=torch.float32), requires_grad=True
        )


def test_materialize_replaces_meta_parameter_with_reshaped_and_repacked_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meta placeholders predate post-load weight transforms (reshape/repack)."""
    reshaped = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8)
    repacked = torch.arange(16, dtype=torch.int32).reshape(4, 4)

    monkeypatch.setattr(
        gms_module.GMSTensorSpec,
        "load_all",
        classmethod(
            lambda cls, _mgr: {
                "proj.weight": _spec_for(reshaped),
                "packed": _spec_for(repacked),
            }
        ),
    )

    with torch.device("meta"):
        model = _ToyModel()

    materialize_module_from_gms(object(), model, device_index=0)  # type: ignore[arg-type]

    assert isinstance(model.proj.weight, torch.nn.Parameter)
    assert model.proj.weight.shape == (2, 2, 8)
    assert model.proj.weight.dtype == torch.float32
    assert model.proj.weight.requires_grad is True
    torch.testing.assert_close(model.proj.weight.data, reshaped, rtol=0, atol=0)

    assert isinstance(model.packed, torch.nn.Parameter)
    assert model.packed.shape == (4, 4)
    assert model.packed.dtype == torch.int32
    assert model.packed.requires_grad is False
    torch.testing.assert_close(model.packed.data, repacked, rtol=0, atol=0)


def test_materialize_rejects_shape_or_dtype_mismatch_on_concrete_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-meta parameters still fail fast on shape or dtype mismatch."""
    wrong_shape = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8)

    monkeypatch.setattr(
        gms_module.GMSTensorSpec,
        "load_all",
        classmethod(lambda cls, _mgr: {"proj.weight": _spec_for(wrong_shape)}),
    )

    model = _ToyModel()
    with pytest.raises(RuntimeError, match="Shape/dtype mismatch for proj.weight"):
        materialize_module_from_gms(object(), model, device_index=0)  # type: ignore[arg-type]
