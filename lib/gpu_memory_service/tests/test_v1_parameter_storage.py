# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from _deps import HAS_TORCH

if not HAS_TORCH:
    pytest.skip("PyTorch is required", allow_module_level=True)

import torch
from gpu_memory_service.v1.mapping import LocalMapping
from gpu_memory_service.v1.parameter_storage import (
    copy_non_parameter_tensors_to_default_allocator,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_copy_out_preserves_tensorimpls_and_nonparameter_aliases() -> None:
    model = torch.nn.Module()
    source = torch.arange(64, dtype=torch.float32)
    source_storage = source.untyped_storage()
    model.weight = torch.nn.Parameter(source[:32])
    model.overlapping_weight = torch.nn.Parameter(source[16:40])
    model.strided_weight = torch.nn.Parameter(source[32:52:2])
    model.register_buffer("overlap", model.weight[8:24])
    model.overlap_alias = model.overlap[4:12]
    model.register_buffer(
        "empty_view",
        torch.empty(0, dtype=torch.float32).set_(
            source_storage,
            44,
            (0, 3),
            (5, 1),
        ),
    )
    model.register_buffer(
        "disjoint",
        torch.empty(0, dtype=torch.float32).set_(
            source_storage,
            48,
            (4,),
            (1,),
        ),
    )
    workspace = torch.empty(0, dtype=torch.float32).set_(
        source_storage,
        56,
        (4,),
        (1,),
    )
    del source

    mapping = LocalMapping(
        "mixed",
        source_storage.nbytes(),
        source_storage.nbytes(),
        source_storage.data_ptr(),
        source_storage.nbytes(),
    )
    tensor_impls = {
        name: int(tensor._cdata)
        for name, tensor in (
            ("weight", model.weight),
            ("overlapping_weight", model.overlapping_weight),
            ("strided_weight", model.strided_weight),
            ("overlap", model.overlap),
            ("overlap_alias", model.overlap_alias),
            ("empty_view", model.empty_view),
            ("disjoint", model.disjoint),
            ("workspace", workspace),
        )
    }
    original_storage = int(source_storage._cdata)
    empty_layout = (
        model.empty_view.storage_offset(),
        model.empty_view.shape,
        model.empty_view.stride(),
    )

    accounting = copy_non_parameter_tensors_to_default_allocator(model, (mapping,))

    assert {
        name: int(tensor._cdata)
        for name, tensor in (
            ("weight", model.weight),
            ("overlapping_weight", model.overlapping_weight),
            ("strided_weight", model.strided_weight),
            ("overlap", model.overlap),
            ("overlap_alias", model.overlap_alias),
            ("empty_view", model.empty_view),
            ("disjoint", model.disjoint),
            ("workspace", workspace),
        )
    } == tensor_impls
    assert int(model.weight.untyped_storage()._cdata) == original_storage
    assert int(model.overlapping_weight.untyped_storage()._cdata) == original_storage
    assert int(model.strided_weight.untyped_storage()._cdata) == original_storage
    assert int(model.empty_view.untyped_storage()._cdata) != original_storage
    assert (
        model.empty_view.storage_offset(),
        model.empty_view.shape,
        model.empty_view.stride(),
    ) == empty_layout
    assert int(model.overlap.untyped_storage()._cdata) != original_storage
    assert (
        model.overlap.untyped_storage()._cdata
        == model.overlap_alias.untyped_storage()._cdata
    )
    assert (
        model.disjoint.untyped_storage()._cdata
        != model.overlap.untyped_storage()._cdata
    )
    assert int(workspace.untyped_storage()._cdata) != original_storage
    assert workspace.untyped_storage()._cdata != model.disjoint.untyped_storage()._cdata
    assert model.weight.tolist() == list(map(float, range(32)))
    assert model.overlap.tolist() == list(map(float, range(8, 24)))
    assert model.disjoint.tolist() == [48.0, 49.0, 50.0, 51.0]
    assert workspace.tolist() == [56.0, 57.0, 58.0, 59.0]

    with torch.inference_mode():
        model.overlap_alias.fill_(23)
    assert model.overlap.tolist()[4:12] == [23.0] * 8
    assert model.weight.tolist()[12:20] == list(map(float, range(12, 20)))
    assert accounting == (51 * 4, (16 + 4 + 4) * 4)
