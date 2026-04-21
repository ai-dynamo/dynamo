# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only coverage for `_iter_module_tensors` property-descriptor handling.

MoE routing classes like TRT-LLM's DeepSeekV3MoeRoutingMethod expose a
read-only `@property` (e.g. `e_score_correction_bias`) that returns a tensor
fetched from another module. Iterating these via `getattr` surfaces a tensor
that cannot be `setattr`-assigned later during materialize, crashing the
shadow-engine RO path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is required")

from gpu_memory_service.client.torch.module import _iter_module_tensors


class _PropertyOnlyModule(torch.nn.Module):
    """Module whose tensor attribute is a read-only @property."""

    def __init__(self) -> None:
        super().__init__()
        self._backing = torch.arange(4, dtype=torch.float32)

    @property
    def routing_bias(self) -> torch.Tensor:
        return self._backing


def test_iter_skips_property_descriptor_even_when_raises():
    class _Raises(torch.nn.Module):
        @property
        def boom(self) -> torch.Tensor:
            raise AttributeError("cannot materialize outside runtime")

    # Must not raise; the property is skipped before getattr runs.
    assert list(_iter_module_tensors(_Raises())) == []


def test_iter_skips_property_returning_cpu_tensor():
    # CPU tensors are already filtered by is_cuda, but we must not yield
    # a (name, tensor, "tensor_attr") entry for the property regardless.
    names = [name for name, _, _ in _iter_module_tensors(_PropertyOnlyModule())]
    assert "routing_bias" not in names


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_iter_yields_cuda_params_but_not_cuda_property_tensor():
    class _Mixed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 2, bias=False, device="cuda")
            self._backing = torch.ones(2, device="cuda", dtype=torch.float32)

        @property
        def computed_bias(self) -> torch.Tensor:
            return self._backing

    names = [name for name, _, _ in _iter_module_tensors(_Mixed())]
    # Real CUDA parameter under a submodule is still discovered via recursion.
    assert "linear.weight" in names
    # The @property is skipped even though it resolves to a CUDA tensor.
    assert "computed_bias" not in names
