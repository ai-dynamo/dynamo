# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.integrations.common import utils as integration_utils


pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


class _FakeAllocator:
    def __init__(self) -> None:
        self.total_bytes = 4096
        self.mappings = {"first": object(), "second": object()}
        self.calls: list[object] = []

    def commit(self) -> bool:
        self.calls.append("commit")
        return True

    def connect(self, lock_type) -> None:
        self.calls.append(("connect", lock_type))

    def remap_all_vas(self) -> None:
        self.calls.append("remap_all_vas")


def test_finalize_gms_write_commits_and_remaps(monkeypatch):
    allocator = _FakeAllocator()
    monkeypatch.setattr(integration_utils.torch.cuda, "synchronize", lambda: None)

    total_bytes = integration_utils.finalize_gms_write(allocator)

    assert total_bytes == 4096
    assert allocator.calls == [
        "commit",
        ("connect", RequestedLockType.RO),
        "remap_all_vas",
    ]


def test_staged_gms_write_registers_all_models_before_commit(monkeypatch):
    allocator = _FakeAllocator()
    target_model = torch.nn.Linear(2, 2)
    draft_model = torch.nn.Linear(2, 2)
    calls: list[object] = []

    monkeypatch.setattr(integration_utils.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        integration_utils,
        "register_module_tensors",
        lambda allocator_arg, model, *, namespace=None, tensor_names=None: calls.append(
            ("register", allocator_arg, model, namespace, tensor_names)
        ),
    )
    monkeypatch.setattr(
        integration_utils,
        "collect_module_tensor_names",
        lambda model: {
            id(target_model): frozenset({"target.weight"}),
            id(draft_model): frozenset({"draft.weight"}),
        }[id(model)],
    )

    integration_utils.stage_gms_write_model(
        allocator,
        target_model,
        namespace="target",
    )
    integration_utils.stage_gms_write_model(
        allocator,
        draft_model,
        namespace="draft",
    )

    total_bytes = integration_utils.finalize_staged_gms_write(allocator)

    assert total_bytes == 4096
    assert calls == [
        (
            "register",
            allocator,
            target_model,
            "target",
            frozenset({"target.weight"}),
        ),
        (
            "register",
            allocator,
            draft_model,
            "draft",
            frozenset({"draft.weight"}),
        ),
    ]
    assert allocator.calls == [
        "commit",
        ("connect", RequestedLockType.RO),
        "remap_all_vas",
    ]


def test_staged_gms_write_rejects_duplicate_namespace():
    allocator = _FakeAllocator()
    model = torch.nn.Linear(2, 2)

    try:
        integration_utils.stage_gms_write_model(allocator, model, namespace="target")
        with pytest.raises(RuntimeError, match="already staged"):
            integration_utils.stage_gms_write_model(
                allocator,
                model,
                namespace="target",
            )
    finally:
        integration_utils._staged_gms_writes.pop(id(allocator), None)


def test_staged_gms_write_filters_to_stage_time_tensor_names(monkeypatch):
    allocator = _FakeAllocator()
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(2, 2))
    monkeypatch.setattr(
        integration_utils,
        "collect_module_tensor_names",
        lambda model: frozenset({"weight"}),
    )
    integration_utils.stage_gms_write_model(allocator, model, namespace="target")
    model.lora_weight = torch.nn.Parameter(torch.empty(2, 2))

    calls: list[object] = []
    monkeypatch.setattr(integration_utils.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        integration_utils,
        "register_module_tensors",
        lambda allocator_arg,
        model_arg,
        *,
        namespace=None,
        tensor_names=None: calls.append((namespace, tensor_names)),
    )

    integration_utils.finalize_staged_gms_write(allocator)

    assert calls == [("target", frozenset({"weight"}))]
