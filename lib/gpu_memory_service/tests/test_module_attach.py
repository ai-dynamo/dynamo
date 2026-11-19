# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attaching published tensors to a module that exposes them through properties."""

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)

import torch
from gpu_memory_service.client.torch import module

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


def _attach(monkeypatch, model, published):
    """Attach ``published``, a list of (name, tensor type, tensor), to ``model``."""
    specs = {
        name: SimpleNamespace(
            meta=SimpleNamespace(tensor_type=kind),
            materialize=lambda manager, device, tensor=tensor: tensor,
        )
        for name, kind, tensor in published
    }
    monkeypatch.setattr(module.GMSTensorSpec, "load_all", lambda manager: specs)
    module.materialize_module_from_gms(object(), model, device_index=0)


class _Experts(torch.nn.Module):
    """Like vLLM's MoE layers: the expert map is a property without a setter
    over a buffer, and the writer's traversal publishes both names."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_expert_map", torch.empty(4, device="meta"))

    @property
    def expert_map(self):
        return self._expert_map


def test_attach_leaves_read_only_properties_to_their_source(monkeypatch):
    model = _Experts()

    _attach(
        monkeypatch,
        model,
        [
            ("expert_map", "tensor_attr", torch.arange(4)),
            ("_expert_map", "buffer", torch.arange(4)),
        ],
    )

    assert model.expert_map.tolist() == [0, 1, 2, 3]


def test_attach_refuses_a_read_only_property_whose_source_is_not_attached(
    monkeypatch,
):
    """The property's source is not a published name, so it stays on meta and
    the property cannot return what the writer published."""
    with pytest.raises(RuntimeError, match="'expert_map' does not match"):
        _attach(
            monkeypatch, _Experts(), [("expert_map", "tensor_attr", torch.arange(4))]
        )


_NAN = float("nan")


def test_attach_accepts_a_nan_behind_a_read_only_property(monkeypatch):
    """A NaN is not equal to itself, but the property still returns what the
    writer published."""
    model = _Experts()

    _attach(
        monkeypatch,
        model,
        [
            ("expert_map", "tensor_attr", torch.tensor([0.0, _NAN, 2.0, 3.0])),
            ("_expert_map", "buffer", torch.tensor([0.0, _NAN, 2.0, 3.0])),
        ],
    )

    assert torch.isnan(model.expert_map[1])
    assert model.expert_map[[0, 2, 3]].tolist() == [0.0, 2.0, 3.0]


@pytest.mark.parametrize(
    ("published", "source"),
    [
        pytest.param([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 4.0], id="another-number"),
        pytest.param(
            [0.0, _NAN, 2.0, 3.0], [0.0, _NAN, 2.0, 4.0], id="another-number-by-a-nan"
        ),
        pytest.param(
            [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, _NAN, 3.0], id="nan-for-a-number"
        ),
    ],
)
def test_attach_refuses_a_read_only_property_with_other_contents(
    monkeypatch, published, source
):
    with pytest.raises(RuntimeError, match="'expert_map' does not match"):
        _attach(
            monkeypatch,
            _Experts(),
            [
                ("expert_map", "tensor_attr", torch.tensor(published)),
                ("_expert_map", "buffer", torch.tensor(source)),
            ],
        )


class _Settable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._holder = SimpleNamespace(value=None)

    @property
    def table(self):
        return self._holder.value

    @table.setter
    def table(self, value):
        self._holder.value = value


def test_attach_still_assigns_a_property_with_a_setter(monkeypatch):
    model = _Settable()

    _attach(monkeypatch, model, [("table", "tensor_attr", torch.arange(4))])

    assert model.table.tolist() == [0, 1, 2, 3]
