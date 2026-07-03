# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publisher-side rebind of non-parameter tensors to private clones."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gpu_memory_service.client.torch.module import (  # noqa: E402
    rebind_nonparameter_tensors,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4, device="cuda"))
        self.register_buffer(
            "expert_map", torch.arange(4, device="cuda", dtype=torch.int32)
        )
        self._k_scale = torch.ones(1, device="cuda")
        self.scales = [
            torch.full((1,), 2.0, device="cuda"),
            torch.full((1,), 3.0, device="cuda"),
        ]
        self.private_scale = torch.ones(1, device="cuda")


def _fake_manager(*tensors: torch.Tensor) -> SimpleNamespace:
    """Manager double whose mappings cover exactly the given tensors."""
    mappings = {
        tensor.data_ptr(): SimpleNamespace(
            aligned_size=tensor.numel() * tensor.element_size()
        )
        for tensor in tensors
    }
    return SimpleNamespace(mappings=mappings)


@requires_cuda
def test_rebind_moves_gms_resident_nonparameters_only():
    model = _Layer()
    manager = _fake_manager(
        model.weight, model.expert_map, model._k_scale, model.scales[0]
    )

    weight_ptr = model.weight.data_ptr()
    buffer_ptr = model.expert_map.data_ptr()
    scale_ptr = model._k_scale.data_ptr()
    list_ptr = model.scales[0].data_ptr()
    untracked_list_ptr = model.scales[1].data_ptr()
    private_ptr = model.private_scale.data_ptr()

    rebound_bytes = rebind_nonparameter_tensors(manager, model)

    expected_bytes = sum(
        t.numel() * t.element_size()
        for t in (model.expert_map, model._k_scale, model.scales[0])
    )
    assert rebound_bytes == expected_bytes

    # Parameters keep their (shared, read-only) binding.
    assert model.weight.data_ptr() == weight_ptr

    # GMS-resident buffer, tensor attr, and list element are rebound...
    assert model.expert_map.data_ptr() != buffer_ptr
    assert model._k_scale.data_ptr() != scale_ptr
    assert model.scales[0].data_ptr() != list_ptr

    # ...preserving values, and the clones are writable.
    assert torch.equal(
        model.expert_map.cpu(), torch.arange(4, dtype=torch.int32)
    )
    assert model.scales[0].item() == 2.0
    model._k_scale.fill_(4.0)
    assert model._k_scale.item() == 4.0

    # Tensors outside the GMS mappings are already private and untouched.
    assert model.scales[1].data_ptr() == untracked_list_ptr
    assert model.private_scale.data_ptr() == private_ptr


@requires_cuda
def test_rebind_is_idempotent_after_clones_leave_gms():
    model = _Layer()
    manager = _fake_manager(model.expert_map, model._k_scale)

    first = rebind_nonparameter_tensors(manager, model)
    assert first > 0

    # The clones no longer point into the tracked mappings.
    assert rebind_nonparameter_tensors(manager, model) == 0
