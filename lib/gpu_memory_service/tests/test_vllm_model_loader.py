# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight-pool ownership in the vLLM first writer."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)

import torch
from gpu_memory_service.client.torch import allocator
from gpu_memory_service.integrations.vllm import model_loader

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


class _Pool:
    def __init__(self, events: list):
        self._events = events

    def __del__(self):
        self._events.append("released")


def test_publication_keeps_initial_value_storage_alive(monkeypatch):
    """Readers materialize non-Parameters from their initial GMS storage."""
    storage = torch.ones(8).untyped_storage()
    alive = weakref.ref(storage)
    monkeypatch.setattr(model_loader, "_pending_gms_client", SimpleNamespace())
    monkeypatch.setattr(model_loader, "_pending_retained_gms_tensors", [storage])
    monkeypatch.setattr(model_loader, "_published_gms_tensor_owners", [])
    monkeypatch.setattr(model_loader, "publish_gms_write", lambda manager: None)
    del storage

    assert model_loader.publish_pending_gms_write()
    gc.collect()

    assert alive() is not None


def test_pool_held_by_a_reference_cycle_is_released(monkeypatch):
    events = []
    pool = _Pool(events)
    pool.cycle = pool
    manager = SimpleNamespace(tag="weights")
    state = SimpleNamespace(manager=manager, mem_pool=pool, callback_failure=None)
    del pool
    monkeypatch.setattr(allocator, "_tag_states", {"weights": state})

    allocator.release_gms_mem_pool(manager)

    assert events == ["released"]
    assert state.mem_pool is None


def test_pool_that_is_still_referenced_is_reported(monkeypatch):
    events = []
    pool = _Pool(events)
    manager = SimpleNamespace(tag="weights")
    state = SimpleNamespace(manager=manager, mem_pool=pool, callback_failure=None)
    monkeypatch.setattr(allocator, "_tag_states", {"weights": state})
    warning = Mock()
    monkeypatch.setattr(allocator.logger, "warning", warning)

    allocator.release_gms_mem_pool(manager)

    assert events == []
    warning.assert_called_once()
