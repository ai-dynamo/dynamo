# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load diff: writer-side selection and where the descriptor is stored."""

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)

import torch
from gpu_memory_service.integrations.vllm.load_diff import writer

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


def _config(extra):
    return SimpleNamespace(load_config=SimpleNamespace(model_loader_extra_config=extra))


def test_option_selects_load_diff_and_rejects_unknown_modes():
    assert writer.selected(_config({"gms_reconstruction_mode": "load_diff"}))
    assert not writer.selected(_config({}))
    assert not writer.selected(_config(None))
    with pytest.raises(ValueError, match="gms_reconstruction_mode"):
        writer.selected(_config({"gms_reconstruction_mode": "glm"}))


def test_descriptor_goes_on_an_allocation_a_parameter_lives_in():
    model = torch.nn.Linear(2, 2)
    workspace = torch.zeros(64)
    mappings = {
        workspace.data_ptr(): SimpleNamespace(
            aligned_size=workspace.nbytes, allocation_id="workspace"
        ),
        model.weight.data_ptr(): SimpleNamespace(
            aligned_size=model.weight.nbytes, allocation_id="weight"
        ),
    }
    manager = SimpleNamespace(mappings=mappings)
    assert writer.descriptor_allocation(manager, model) == "weight"
    with pytest.raises(RuntimeError, match="parameter in GMS memory"):
        writer.descriptor_allocation(SimpleNamespace(mappings={}), model)
